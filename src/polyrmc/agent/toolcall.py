"""Tool-calling transport: the model asks for actions, and gets results back.

This is what separates an orchestrator from a pipeline. The older
:mod:`polyrmc.agent.provider` speaks text in, text out -- adequate for a judge
that picks an index from a shortlist, and useless for a model that has to decide
*what to do next*. A judge is handed the question; an orchestrator has to find
it. That needs a transport where the model can name a tool, receive what it
returned, and decide again with the answer in hand.

The neutral shapes here -- :class:`Message`, :class:`ToolCall`,
:class:`ToolResult` -- exist so the loop in :mod:`polyrmc.agent.orchestrator`
never learns which provider is behind it. That is not portability for its own
sake. The whole premise of this system is that a model outage costs judgment and
not correctness, and that promise is only checkable if the thing being swapped
out has a boundary you can point at.

Two live adapters, both over LangChain's chat interface, which normalises tool
calls into the same ``AIMessage.tool_calls`` shape whichever provider answered:

* Gemini (default) -- the pinned model the rest of the pipeline already uses.
* Claude -- available by setting ``ANTHROPIC_API_KEY``.

And one scripted adapter, :class:`ScriptedProvider`, which is what the test
suite runs against. The loop is exercised end to end with no network, no key,
and no nondeterminism, because the interesting failures in an agentic loop --
a tool that rejects, a model that repeats itself, a cap that has to hold -- are
all reachable offline and none of them should need a live provider to test.

No provider SDK is imported at module scope.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, Sequence

from polyrmc.agent.budget import Usage

# Kept in one place because two things have to agree on it: the adapter that
# builds the client, and the CLI that tells a researcher which key is missing.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"

GEMINI_KEYS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
CLAUDE_KEYS = ("ANTHROPIC_API_KEY",)


class ProviderUnavailable(RuntimeError):
    """No usable model client. Says which key or package is missing.

    Raised at construction rather than at first call, so a session that cannot
    reach a model says so before a researcher has typed a question into it.
    """


@dataclass(frozen=True)
class ToolCall:
    """One action the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """One line, for the transcript a person reads."""
        if not self.arguments:
            return f"{self.name}()"
        inner = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({inner})"


@dataclass(frozen=True)
class ToolResult:
    """What a tool returned, on its way back to the model.

    ``ok`` is false for a refusal. A refusal is still a result and is still fed
    back: the commit tier writes its rejections as recovery instructions --
    what failed, by how much, what would be valid instead -- and a model that
    never sees them will retry the same invalid action instead of correcting.
    """

    call_id: str
    name: str
    content: str
    ok: bool = True


@dataclass
class Message:
    """One entry in the conversation, in provider-neutral form."""

    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)


@dataclass
class AssistantTurn:
    """What the model said, what it wants run, and what it cost."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = "unknown"
    usage: Usage = field(default_factory=lambda: Usage(requests=1))

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ToolCallingProvider(Protocol):
    """Anything that can continue a conversation, with tools available."""

    name: str
    model: str

    def converse(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantTurn: ...


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def _first_key(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


class LangChainToolProvider:
    """Adapter over any LangChain chat model that supports ``bind_tools``.

    One implementation serves both providers. LangChain already normalises the
    wire differences -- Gemini's ``functionCall`` parts and Claude's
    ``tool_use`` blocks both arrive as ``AIMessage.tool_calls`` -- so writing
    two adapters would mean maintaining two copies of the same translation for
    no gain.
    """

    def __init__(self, client: Any, name: str, model: str) -> None:
        self.client = client
        self.name = name
        self.model = model

    # --- translation ---------------------------------------------------------

    @staticmethod
    def _to_langchain(messages: Sequence[Message]) -> list[Any]:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        converted: list[Any] = []
        for message in messages:
            if message.role == "user":
                converted.append(HumanMessage(content=message.text))
            elif message.role == "assistant":
                converted.append(
                    AIMessage(
                        content=message.text,
                        tool_calls=[
                            {
                                "name": call.name,
                                "args": dict(call.arguments),
                                "id": call.id,
                                "type": "tool_call",
                            }
                            for call in message.tool_calls
                        ],
                    )
                )
            else:
                # Every result becomes its own ToolMessage. Both providers pair
                # results to calls by id and will error on a call left
                # unanswered, so the loop must return one per call even when a
                # tool refused.
                for result in message.results:
                    converted.append(
                        ToolMessage(
                            content=result.content,
                            tool_call_id=result.call_id,
                            name=result.name,
                            status="success" if result.ok else "error",
                        )
                    )
        return converted

    def converse(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantTurn:
        from langchain_core.messages import SystemMessage

        bound = self.client.bind_tools(list(tools)) if tools else self.client
        payload = [SystemMessage(content=system), *self._to_langchain(messages)]
        reply = bound.invoke(payload)

        calls = [
            ToolCall(
                id=str(raw.get("id") or _new_call_id()),
                name=str(raw.get("name", "")),
                arguments=dict(raw.get("args") or {}),
            )
            for raw in getattr(reply, "tool_calls", []) or []
        ]
        return AssistantTurn(
            text=_text_of(reply),
            tool_calls=calls,
            model=self.model,
            usage=_usage_of(reply),
        )


def _usage_of(reply: Any) -> Usage:
    """Read the token counts off a reply.

    Absence is reported as zero tokens against one request rather than raising.
    A provider that stops populating ``usage_metadata`` should cost the session
    its accounting, not its answer -- and the request count, which is what the
    free tier actually rations, is known either way.
    """
    metadata = getattr(reply, "usage_metadata", None) or {}
    try:
        return Usage(
            requests=1,
            input_tokens=int(metadata.get("input_tokens") or 0),
            output_tokens=int(metadata.get("output_tokens") or 0),
        )
    except (AttributeError, TypeError, ValueError):
        return Usage(requests=1)


def _text_of(reply: Any) -> str:
    """Flatten a reply's content to text.

    Content is a plain string from Gemini and a list of typed blocks from
    Claude, and a turn that requests a tool often carries no text at all.
    """
    content = getattr(reply, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def gemini_provider(model: str | None = None, temperature: float = 0.2) -> LangChainToolProvider:
    """Build the Gemini orchestrator client, or say what is missing."""
    if not _first_key(GEMINI_KEYS):
        raise ProviderUnavailable(
            "Gemini needs GOOGLE_API_KEY or GEMINI_API_KEY. Put one in .env, or "
            "run with --provider none for the offline surface."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ProviderUnavailable(
            "langchain-google-genai is not installed. "
            "pip install langchain-google-genai"
        ) from error

    name = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    client = ChatGoogleGenerativeAI(
        model=name,
        temperature=temperature,
        # The key may be under either name; langchain-google-genai reads only
        # the first. Naming the wrong variable here is what made --judge model
        # unusable once already.
        google_api_key=_first_key(GEMINI_KEYS),
    )
    return LangChainToolProvider(client, name="gemini", model=name)


def claude_provider(model: str | None = None, temperature: float = 0.2) -> LangChainToolProvider:
    """Build the Claude orchestrator client, or say what is missing."""
    if not _first_key(CLAUDE_KEYS):
        raise ProviderUnavailable(
            "Claude needs ANTHROPIC_API_KEY. Put one in .env, or use "
            "--provider gemini, which is the default and already keyed here."
        )
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ProviderUnavailable(
            "langchain-anthropic is not installed. pip install langchain-anthropic"
        ) from error

    name = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL
    client = ChatAnthropic(model=name, temperature=temperature)
    return LangChainToolProvider(client, name="claude", model=name)


PROVIDERS = {"gemini": gemini_provider, "claude": claude_provider}


def build_provider(kind: str, model: str | None = None) -> ToolCallingProvider:
    """Resolve a provider by name."""
    factory = PROVIDERS.get(kind)
    if factory is None:
        raise ProviderUnavailable(
            f"{kind!r} is not a known provider. Available: {sorted(PROVIDERS)}."
        )
    return factory(model)


class ScriptedProvider:
    """A provider that replays a fixed list of turns.

    The whole orchestrator is testable through this: hand it a turn that calls
    a tool, then a turn that reads the result and answers, and the loop runs
    with no network and no nondeterminism. Running out of script raises rather
    than returning something plausible, because a loop that made one more call
    than the test expected is exactly the bug worth failing on.
    """

    name = "scripted"

    def __init__(self, turns: Sequence[AssistantTurn | str], model: str = "scripted") -> None:
        self.turns = [
            AssistantTurn(text=t) if isinstance(t, str) else t for t in turns
        ]
        self.model = model
        self.calls: list[tuple[str, list[Message], list[dict[str, Any]]]] = []

    def converse(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantTurn:
        self.calls.append((system, list(messages), list(tools)))
        if not self.turns:
            raise ProviderUnavailable(
                "scripted provider is exhausted: the loop asked for another turn "
                "than the script provides"
            )
        turn = self.turns.pop(0)
        # Filled in here so tests can write ToolCall(id="", ...) and still get
        # a well-formed pairing back.
        turn.tool_calls = [
            call if call.id else ToolCall(_new_call_id(), call.name, call.arguments)
            for call in turn.tool_calls
        ]
        return turn


def as_openai_tools(schemas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise plain ``{name, description, parameters}`` dicts for binding."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return [convert_to_openai_tool(schema) for schema in schemas]


def summarise_arguments(arguments: dict[str, Any], limit: int = 120) -> str:
    """Render call arguments compactly for a terminal line."""
    if not arguments:
        return ""
    text = json.dumps(arguments, default=str, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = [
    "CLAUDE_KEYS",
    "Usage",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "GEMINI_KEYS",
    "PROVIDERS",
    "AssistantTurn",
    "LangChainToolProvider",
    "Message",
    "ProviderUnavailable",
    "ScriptedProvider",
    "ToolCall",
    "ToolCallingProvider",
    "ToolResult",
    "as_openai_tools",
    "build_provider",
    "claude_provider",
    "gemini_provider",
    "summarise_arguments",
]
