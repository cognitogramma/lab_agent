"""The agent loop: the model decides what happens next.

This replaces the fixed graph the pipeline used to run. That graph encoded the
route in Python -- intake, then process, then judge the window, then emit -- and
consulted a model at exactly one node. It was reliable and it was a pipeline
with a model bolted into a socket. Everything it could ever do was decided
before a researcher opened their mouth.

Here the route is the model's to choose. It sees the tool surface, it sees where
the run stands, and it decides: read the state, search what the same sample did
last time, sweep the windows, ask a question, stop. The loop below does exactly
three things -- pass the conversation to the model, run the tools it asked for,
hand back the results -- and repeats until the model stops asking.

Giving up the fixed route does not mean giving up the guarantees, because the
guarantees were never in the route. They are in the tool surface, which
validates and rejects in code, and in tier 0, which no model can reach at all.
The three that matter here:

**A rejection is a result, not a crash.** ``Rejected`` carries what failed, by
how much, and what would be valid instead. It goes back to the model as the
outcome of its call. A loop that swallowed refusals would leave the model
guessing at why nothing happened, and it would guess wrong in a way that looks
like progress.

**Commits need a person.** Every tool is tiered, and every ``commit`` tool goes
through :class:`ApprovalPolicy` before it runs. Reads and computes are free --
they cannot change a run, and making a researcher approve each one trains them
to approve without reading, which is worse than not asking.

**The loop is capped.** A model that keeps calling tools stops at
``max_steps``, and the cap is reported rather than hidden. An agent that ran
forty tools to answer a question the researcher never sees is not an agent that
worked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

from polyrmc.agent.budget import Budget, BudgetExceeded, Usage
from polyrmc.agent.lab import load_lab_context, system_prompt_with_lab
from polyrmc.agent.lanes import load_prompt
from polyrmc.agent.toolcall import (
    AssistantTurn,
    Message,
    ToolCall,
    ToolCallingProvider,
    ToolResult,
    as_openai_tools,
)
from polyrmc.agent.toolspec import Tool, ToolContext, build_registry, schemas_for
from polyrmc.tools import Rejected

DEFAULT_MAX_STEPS = 12
"""Tool-calling rounds allowed per researcher turn.

Generous enough for a full run -- load, check, sweep, pick, emit, verify,
extract, with reads in between -- and short enough that a model looping on a
refusal it will not read gets stopped while the researcher is still watching.
"""


@dataclass
class Approval:
    """The answer to "may this run?"."""

    allowed: bool
    reason: str = ""
    remember: bool = False


AskFn = Callable[[Tool, ToolCall], Approval]


def deny_all(tool: Tool, call: ToolCall) -> Approval:
    """Refuse every commit. The default when nobody is at the terminal.

    A non-interactive session must not commit by default. The alternative --
    proceeding because no one was there to object -- is indistinguishable from
    a researcher having approved it, which is the one thing the audit trail
    exists to prevent.
    """
    return Approval(
        False,
        "no one is at the terminal to approve a commit; run interactively to allow it",
    )


def allow_all(tool: Tool, call: ToolCall) -> Approval:
    """Approve everything. For tests, and for an explicitly unattended run."""
    return Approval(True, "unattended mode")


@dataclass
class ApprovalPolicy:
    """Decides which calls run, and asks a person about the rest.

    ``always`` holds tool names the researcher has approved for the rest of the
    session. It is per-tool and never per-argument: "always allow pick_window"
    is a coherent standing decision, whereas "always allow emit_csv to that one
    path" invites the case where the path changed and nobody noticed.
    """

    ask: AskFn = deny_all
    always: set[str] = field(default_factory=set)

    def check(self, tool: Tool, call: ToolCall) -> Approval:
        if not tool.mutating:
            return Approval(True)
        if tool.name in self.always:
            return Approval(True, "approved earlier this session")
        answer = self.ask(tool, call)
        if answer.allowed and answer.remember:
            self.always.add(tool.name)
        return answer


EventKind = Literal[
    "text", "call", "result", "refused", "denied", "error", "capped", "spend"
]


@dataclass
class Event:
    """One thing that happened, for the caller to display.

    The loop returns events rather than printing, so the same orchestrator
    drives an interactive terminal, a test, and anything else later, without
    any of them inheriting the others' idea of formatting.
    """

    kind: EventKind
    text: str = ""
    tool: str = ""
    call: ToolCall | None = None

    def render(self) -> str:
        # ASCII only. A Windows console is cp1252 by default, and a session that
        # dies encoding an arrow after the model has already committed its work
        # loses the transcript of what it did.
        if self.kind == "call":
            return f"  -> {self.call.render() if self.call else self.tool}"
        if self.kind == "result":
            return _indent(self.text)
        if self.kind == "refused":
            return _indent(f"REJECTED {self.text}")
        if self.kind == "denied":
            return _indent(f"declined: {self.text}")
        if self.kind == "error":
            return _indent(f"tool error: {self.text}")
        if self.kind in {"capped", "spend"}:
            return f"\n[{self.text}]"
        return self.text


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


class Orchestrator:
    """One conversation, with tools, over one run.

    Holds the transcript so a researcher can ask a follow-up without restating
    the run, and so the model can see what it already tried rather than
    rediscovering it.
    """

    def __init__(
        self,
        context: ToolContext,
        provider: ToolCallingProvider,
        approvals: ApprovalPolicy | None = None,
        registry: dict[str, Tool] | None = None,
        system_prompt: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        budget: Budget | None = None,
    ) -> None:
        self.context = context
        self.provider = provider
        self.approvals = approvals or ApprovalPolicy()
        self.registry = registry or build_registry()
        self.max_steps = max_steps
        # The session runs on the researcher's own key. What it spent is part of
        # what it did, so it is tracked alongside the tool trace rather than
        # left to whatever the provider's dashboard says tomorrow.
        self.budget = budget or Budget()

        self.lab_context, self.lab_path = load_lab_context()
        self.system_prompt = system_prompt or default_system_prompt()
        self.messages: list[Message] = []
        self.schemas = as_openai_tools(schemas_for(self.registry))
        # Every call attempted this session, with its outcome. The narration a
        # researcher reads is generated from this rather than authored, so what
        # the agent says it did cannot drift from what it ran.
        self.trace: list[tuple[ToolCall, ToolResult]] = []

    # --- driving --------------------------------------------------------

    def ask(self, user_text: str) -> list[Event]:
        """Run one researcher turn to completion, and return what happened."""
        self.messages.append(Message(role="user", text=user_text))
        events: list[Event] = []
        turn_usage = Usage()

        for step in range(self.max_steps):
            try:
                self.budget.check()
            except BudgetExceeded as error:
                events.append(Event("capped", str(error)))
                events.append(Event("spend", _spend_line(turn_usage, self.budget)))
                return events

            turn = self._converse()
            self.budget.record(turn.usage)
            turn_usage.add(turn.usage)
            self.messages.append(
                Message(role="assistant", text=turn.text, tool_calls=turn.tool_calls)
            )
            if turn.text.strip():
                events.append(Event("text", turn.text))

            if not turn.wants_tools:
                events.append(Event("spend", _spend_line(turn_usage, self.budget)))
                return events

            results: list[ToolResult] = []
            for call in turn.tool_calls:
                events.append(Event("call", tool=call.name, call=call))
                result, event = self._invoke(call)
                results.append(result)
                events.append(event)
                self.trace.append((call, result))

            self.messages.append(Message(role="tool", results=results))

        events.append(
            Event(
                "capped",
                f"stopped after {self.max_steps} tool rounds without a final answer. "
                "The run is wherever the last committed tool left it -- ask for "
                "get_state to see where.",
            )
        )
        events.append(Event("spend", _spend_line(turn_usage, self.budget)))
        return events

    def _converse(self) -> AssistantTurn:
        return self.provider.converse(self.system_prompt, self.messages, self.schemas)

    def _invoke(self, call: ToolCall) -> tuple[ToolResult, Event]:
        """Run one tool call, converting every outcome into a result.

        Nothing raises out of here. A model that asked for something impossible
        needs to be told what happened so it can change course; an exception
        that escaped would end the session instead, losing the transcript along
        with it.
        """
        tool = self.registry.get(call.name)
        if tool is None:
            message = (
                f"no tool named {call.name!r}. Available: "
                f"{', '.join(sorted(self.registry))}."
            )
            return (
                ToolResult(call.id, call.name, message, ok=False),
                Event("error", message, tool=call.name, call=call),
            )

        verdict = self.approvals.check(tool, call)
        if not verdict.allowed:
            message = (
                f"the researcher declined this call: {verdict.reason or 'no reason given'}. "
                "Do not attempt the same change another way. Ask them what they would prefer."
            )
            return (
                ToolResult(call.id, call.name, message, ok=False),
                Event("denied", verdict.reason or "declined", tool=call.name, call=call),
            )

        try:
            output = tool.handler(self.context, dict(call.arguments))
        except Rejected as error:
            return (
                ToolResult(call.id, call.name, str(error), ok=False),
                Event("refused", str(error), tool=call.name, call=call),
            )
        except Exception as error:  # noqa: BLE001 - a tool fault must not end the session
            message = f"{type(error).__name__}: {error}"
            return (
                ToolResult(call.id, call.name, message, ok=False),
                Event("error", message, tool=call.name, call=call),
            )

        return (
            ToolResult(call.id, call.name, output, ok=True),
            Event("result", output, tool=call.name, call=call),
        )

    # --- reporting ------------------------------------------------------

    def narrate(self) -> str:
        """What was actually run this session, generated from the trace."""
        if not self.trace:
            return "(no tools have been called this session)"
        lines = []
        for call, result in self.trace:
            mark = "ok " if result.ok else "REJ"
            lines.append(f"  {mark} {call.render()}")
        return "\n".join(lines)

    @property
    def committed(self) -> list[str]:
        """Names of the mutating calls that actually ran."""
        return [
            call.name
            for call, result in self.trace
            if result.ok and (t := self.registry.get(call.name)) is not None and t.mutating
        ]


def _spend_line(turn: Usage, budget: Budget) -> str:
    """This turn's spend, and the session's, on one line.

    Both, because they answer different questions. The turn figure says whether
    the question just asked was an expensive one; the session figure is what the
    key is actually being billed for.
    """
    line = (
        f"this turn: {turn.requests} request(s), "
        f"{turn.input_tokens:,} in / {turn.output_tokens:,} out"
    )
    if budget.rates is not None:
        line += f", ~${budget.rates.cost(turn):.4f}"
    return f"{line}  |  {budget.render()}"


def default_system_prompt(start: str | Path | None = None) -> str:
    """The base rules, the orchestrator's own instructions, and LAB.md.

    ``system.md`` is shared with the rest of the agent layer and states what the
    model may never do. ``orchestrator.md`` adds how to drive. LAB.md is
    appended whole by :func:`system_prompt_with_lab`, so the agent carries this
    lab's conventions rather than general knowledge of light scattering.
    """
    base = f"{load_prompt('system.md')}\n\n{load_prompt('orchestrator.md')}"
    return system_prompt_with_lab(base, start)


GREETING = (
    "PolyRMC research agent. Point me at a run, ask about one we processed "
    "before, or ask a question about the measurement."
)


__all__ = [
    "DEFAULT_MAX_STEPS",
    "GREETING",
    "Approval",
    "ApprovalPolicy",
    "AskFn",
    "Budget",
    "BudgetExceeded",
    "Event",
    "EventKind",
    "Orchestrator",
    "Usage",
    "allow_all",
    "default_system_prompt",
    "deny_all",
]
