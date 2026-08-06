"""The agent loop: nodes, conditional routing, and interrupts.

Design section 10. The graph carries the run from a raw file to an emitted CSV,
pausing wherever a judgment belongs to a person.

Three properties are load-bearing.

**Interrupts, not busy-waiting.** A judgment checkpoint returns control with a
question attached. Nothing spins, nothing times out into a default, and the run
state at the moment of the pause is exactly the state that resumes. A checkpoint
that quietly proceeded after a timeout would be indistinguishable from a
researcher having approved it.

**Narration is generated from what happened.** Every node appends to
``execution_history``, and the report is rendered from that plus the state. The
agent cannot describe a step it did not run, or omit one it did, because it is
not authoring the account -- it is reading it back.

**Deterministic signals stay deterministic.** A file arriving means process a
run. That is a fact about the input, not a question for a classifier, and
spending a model call on it would add latency, cost, and a failure mode to a
decision that has none.

The graph is written natively rather than on an orchestration framework. The
node set is small and fixed, the routing is a handful of conditionals, and
keeping it in plain Python means the control flow is inspectable in the same way
the measurement path is -- which is the whole premise of the system. The provider
abstraction leaves any framework free to sit behind it.

No arithmetic over the series happens here, and no node in this module touches
the measurement core except through the tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from polyrmc.agent.lanes import (
    Reconciliation,
    Shortlist,
    load_prompt,
    reconcile,
    run_llm_lane,
    run_vlm_lane,
)
from polyrmc.agent.lab import load_lab_context, system_prompt_with_lab
from polyrmc.agent.provider import Resilient
from polyrmc.agent.render import default_render_plot
from polyrmc.config import RunConfig
from polyrmc.state import RunState
from polyrmc.tools import Rejected, ToolSurface


class Node(str, Enum):
    """Every stage the run can be at."""

    INTAKE = "intake"
    CHECK_DECLARATIONS = "check_declarations"
    RESOLVE_CONFLICTS = "resolve_conflicts"
    PROCESS = "process"
    REVIEW_ANOMALIES = "review_anomalies"
    JUDGE_WINDOW = "judge_window"
    REVIEW_ANCHORS = "review_anchors"
    EMIT = "emit"
    VERIFY = "verify"
    DONE = "done"


@dataclass
class Interrupt:
    """The graph has paused and needs a person.

    Carries everything needed to answer without going back through the agent:
    the question, the options where the choice is bounded, and why it is being
    asked at all.
    """

    node: Node
    question: str
    options: list[Any] = field(default_factory=list)
    because: str = ""

    def __str__(self) -> str:
        text = f"[{self.node.value}] {self.question}"
        if self.options:
            text += f"\n  options: {self.options}"
        if self.because:
            text += f"\n  {self.because}"
        return text


@dataclass
class Step:
    """The result of advancing the graph by one node."""

    node: Node
    next_node: Node
    narration: str
    interrupt: Interrupt | None = None

    @property
    def paused(self) -> bool:
        return self.interrupt is not None


class AgentGraph:
    """Drives one run, pausing at every judgment that belongs to a person.

    Holds no model client of its own. Model calls happen only inside the judge
    lanes, each wrapped so that an outage costs judgment and not correctness.
    """

    def __init__(
        self,
        tools: ToolSurface,
        guard: Resilient | None = None,
        vlm_guard: Resilient | None = None,
        render_plot: Callable[[RunState, list[int]], bytes | None] | None = None,
        system_prompt: str | None = None,
        output_csv: str | Path | None = None,
    ) -> None:
        self.tools = tools
        self.guard = guard or Resilient()
        self.vlm_guard = vlm_guard or Resilient()
        # Default to a real renderer. Left as None, the visual lane has nothing
        # to look at and the two-lane judge quietly becomes a one-lane judge
        # while still reporting a two-lane status.
        self.render_plot = render_plot or default_render_plot
        # LAB.md is loaded whole and appended, so the agent carries this lab's
        # conventions rather than general knowledge of light scattering.
        self.lab_context, self.lab_path = load_lab_context()
        self.system_prompt = system_prompt or system_prompt_with_lab(
            load_prompt("system.md")
        )
        self.output_csv = Path(output_csv) if output_csv else None
        self.node = Node.INTAKE
        self.pending: Interrupt | None = None
        self.emitted: Path | None = None
        self.checks: list[Any] = []

    # --- state ---------------------------------------------------------------

    @property
    def state(self) -> RunState:
        return self.tools.state

    @property
    def config(self) -> RunConfig:
        return self.tools.config

    def _record(self, message: str) -> str:
        """Append to the history the narration is later generated from."""
        self.state.execution_history.append(message)
        return message

    # --- driving -------------------------------------------------------------

    def run(self, max_steps: int = 20) -> list[Step]:
        """Advance until the graph finishes or asks for a person.

        Returns every step taken, so the caller can narrate the whole stretch
        rather than only its outcome.
        """
        steps: list[Step] = []
        for _ in range(max_steps):
            if self.node is Node.DONE:
                break
            step = self.advance()
            steps.append(step)
            if step.paused:
                break
        return steps

    def advance(self) -> Step:
        """Run exactly one node."""
        handler = {
            Node.INTAKE: self._intake,
            Node.CHECK_DECLARATIONS: self._check_declarations,
            Node.RESOLVE_CONFLICTS: self._resolve_conflicts,
            Node.PROCESS: self._process,
            Node.REVIEW_ANOMALIES: self._review_anomalies,
            Node.JUDGE_WINDOW: self._judge_window,
            Node.REVIEW_ANCHORS: self._review_anchors,
            Node.EMIT: self._emit,
            Node.VERIFY: self._verify,
        }[self.node]

        step = handler()
        self.pending = step.interrupt
        # A paused node stays current, so resuming re-enters it with whatever
        # the researcher supplied rather than skipping past the question.
        self.node = self.node if step.paused else step.next_node
        return step

    def resume(self) -> Step:
        """Continue after a researcher has answered the pending interrupt."""
        if self.pending is None:
            raise RuntimeError("nothing is pending; call advance()")
        self.pending = None
        return self.advance()

    # --- nodes ---------------------------------------------------------------

    def _intake(self) -> Step:
        """Section 6.0. Nothing here is inferred from the data."""
        outstanding = [q.key for q in self.tools.declarations.unanswered()]
        if outstanding:
            return Step(
                Node.INTAKE,
                Node.INTAKE,
                self._record(
                    f"intake: {len(outstanding)} question(s) outstanding "
                    f"({', '.join(outstanding[:4])})"
                ),
                Interrupt(
                    Node.INTAKE,
                    "What is this run? These have to be declared, not inferred.",
                    options=outstanding,
                    because=(
                        "Processing a run under the wrong physical model produces "
                        "numbers that look entirely normal and are entirely wrong."
                    ),
                ),
            )
        # Ingest first. The consistency check tests declarations against the
        # trace, so it has nothing to test until the trace is loaded -- running
        # it here would skip every check and report the run as consistent.
        return Step(
            Node.INTAKE,
            Node.PROCESS,
            self._record(
                f"intake: {len(self.state.declarations)} declaration(s) recorded"
            ),
        )

    def _check_declarations(self) -> Step:
        """Section 6.0.1. Code tests each declaration against the trace."""
        try:
            conflicts = self.tools.check_declarations()
        except Rejected as error:
            return Step(
                Node.CHECK_DECLARATIONS,
                Node.REVIEW_ANOMALIES,
                self._record(f"declaration check skipped: {error}"),
            )
        if conflicts:
            return Step(
                Node.CHECK_DECLARATIONS,
                Node.RESOLVE_CONFLICTS,
                self._record(f"declaration check: {len(conflicts)} conflict(s) raised"),
            )
        return Step(
            Node.CHECK_DECLARATIONS,
            Node.REVIEW_ANOMALIES,
            self._record("declaration check: the data supports every declaration"),
        )

    def _resolve_conflicts(self) -> Step:
        """A conflict is never resolved by code, and never silently."""
        unresolved = self.state.unresolved_conflicts
        if not unresolved:
            return Step(
                Node.RESOLVE_CONFLICTS,
                Node.REVIEW_ANOMALIES,
                self._record("conflicts: all resolved with recorded reasons"),
            )
        detail = "\n".join(
            f"  {c.check}: declared {c.declared}, observed {c.observed}"
            for c in unresolved
        )
        return Step(
            Node.RESOLVE_CONFLICTS,
            Node.RESOLVE_CONFLICTS,
            self._record(f"conflicts: {len(unresolved)} awaiting a person"),
            Interrupt(
                Node.RESOLVE_CONFLICTS,
                f"The data disagrees with {len(unresolved)} declaration(s):\n{detail}",
                options=[c.check for c in unresolved],
                because=(
                    "The data can disagree with a correct declaration for real "
                    "physical reasons, and only you know which. Resolving one needs "
                    "a reason, which is recorded."
                ),
            ),
        )

    def _process(self) -> Step:
        """The deterministic measurement path. No model call reaches this."""
        try:
            self.tools.load_run()
            band = self.tools.sweep_windows()
        except (Rejected, ValueError) as error:
            return Step(
                Node.PROCESS,
                Node.PROCESS,
                self._record(f"processing failed: {error}"),
                Interrupt(
                    Node.PROCESS,
                    str(error),
                    because="The measurement path refused this run.",
                ),
            )
        return Step(
            Node.PROCESS,
            Node.CHECK_DECLARATIONS,
            self._record(
                f"processed: {len(self.state.anomalies)} anomaly region(s), "
                f"{len(self.state.censored_indices)} censored sample(s), "
                f"{len(self.state.anchor_candidates)} anchor candidate(s), "
                f"{len(band)} window(s) in the safe band"
            ),
        )

    def _review_anomalies(self) -> Step:
        """Section 6.4. A region the rules could not name needs a person.

        Deterministic classification handles the unambiguous cases and leaves
        the rest undecided rather than guessing, because excising something the
        rules could not name deletes data on a hunch. Those regions surface
        here instead of silently blocking the emit later.
        """
        pending = [a for a in self.state.anomalies if a.decision == "pending"]
        if not pending:
            return Step(
                Node.REVIEW_ANOMALIES,
                Node.JUDGE_WINDOW,
                self._record(
                    f"anomalies: all {len(self.state.anomalies)} region(s) decided"
                ),
            )
        listing = "\n".join(
            f"  ({a.start_index}, {a.end_index}) {a.n_points} sample(s), "
            f"flagged by {', '.join(a.detected_by)}"
            for a in pending[:10]
        )
        return Step(
            Node.REVIEW_ANOMALIES,
            Node.REVIEW_ANOMALIES,
            self._record(f"anomalies: {len(pending)} region(s) awaiting a decision"),
            Interrupt(
                Node.REVIEW_ANOMALIES,
                f"{len(pending)} region(s) could not be classified "
                f"deterministically:\n{listing}",
                options=[(a.start_index, a.end_index) for a in pending],
                because=(
                    "The detectors fired but the rules could not name what these "
                    "are. Excising a region nobody can name deletes data on a "
                    "guess, so they stay in the trace until you decide."
                ),
            ),
        )

    def _judge_window(self) -> Step:
        """Section 6.7 stages 2 and 3. Both lanes, then reconciliation."""
        candidates = self.tools.get_shortlist()
        if not candidates:
            return Step(
                Node.JUDGE_WINDOW,
                Node.REVIEW_ANCHORS,
                self._record("window: no safe band was computed; nothing to judge"),
            )

        shortlist = Shortlist(
            candidates=candidates,
            noise_floor=self.state.noise_floor or 0.0,
            context={
                "run_id": self.state.run_id,
                "experiment_type": self.config.experiment_type,
                "anomalies": len(self.state.anomalies),
                "censored": len(self.state.censored_indices),
            },
        )

        llm, llm_errors = run_llm_lane(shortlist, self.guard)
        image = self.render_plot(self.state, shortlist.windows) if self.render_plot else None
        vlm, vlm_errors = run_vlm_lane(shortlist, self.vlm_guard, image)

        self.state.llm_lane = llm
        self.state.vlm_lane = vlm
        outcome: Reconciliation = reconcile(llm, vlm, shortlist)
        self.state.judge_status = outcome.status
        self.state.lanes_disagreed = outcome.disagreed
        self.state.fallback_taken = outcome.fallback_taken

        for message in llm_errors + vlm_errors:
            self._record(f"judge: {message}")

        if outcome.needs_researcher:
            return Step(
                Node.JUDGE_WINDOW,
                Node.JUDGE_WINDOW,
                self._record("window: the two lanes disagree; escalating"),
                Interrupt(
                    Node.JUDGE_WINDOW,
                    outcome.message,
                    options=shortlist.windows,
                    because=(
                        "Neither lane is preferred by default and the two are not "
                        "averaged: a compromise window is one nobody argued for."
                    ),
                ),
            )

        self.tools.pick_window(outcome.window_length)
        return Step(
            Node.JUDGE_WINDOW,
            Node.REVIEW_ANCHORS,
            self._record(f"window {outcome.window_length} ({outcome.status})"),
        )

    def _review_anchors(self) -> Step:
        """Baseline correction is proposed here and committed only by a person."""
        candidates = self.state.anchor_candidates
        if len(candidates) < self.config.baseline.min_anchors:
            return Step(
                Node.REVIEW_ANCHORS,
                Node.EMIT,
                self._record(
                    f"baseline: {len(candidates)} candidate(s), fewer than the "
                    f"minimum of {self.config.baseline.min_anchors}. No correction "
                    "is offerable, which is also the default."
                ),
            )
        if self.state.baseline_applied or self.state.anchors_selected:
            return Step(
                Node.REVIEW_ANCHORS,
                Node.EMIT,
                self._record(
                    f"baseline: {len(self.state.anchors_selected)} anchor(s) "
                    f"selected by {self.state.anchor_selected_by}, "
                    f"applied={self.state.baseline_applied}"
                ),
            )
        return Step(
            Node.REVIEW_ANCHORS,
            Node.EMIT,
            self._record(
                f"baseline: {len(candidates)} anchor candidate(s) proposed, none "
                "selected. Default is no correction."
            ),
            Interrupt(
                Node.REVIEW_ANCHORS,
                f"{len(candidates)} baseline anchor candidate(s) are available. "
                "Correct the drift, or leave the trace as measured?",
                options=[c.id for c in candidates] + ["no correction"],
                because=(
                    "Anchor selection decides what counts as signal. A curve fitted "
                    "to anchors that were really part of the kinetics gives a smooth, "
                    "plausible, wrong trace that nothing downstream will catch -- so "
                    "this one is yours, and no correction is a valid answer."
                ),
            ),
        )

    def _emit(self) -> Step:
        path = self.output_csv or Path(self.config.source_file).with_suffix(
            ".processed.csv"
        )
        try:
            result = self.tools.emit_csv(path)
            self.emitted = Path(result.changed["path"])
        except Rejected as error:
            return Step(
                Node.EMIT,
                Node.EMIT,
                self._record(f"emit refused: {error}"),
                Interrupt(
                    Node.EMIT,
                    str(error),
                    because="The emit gate refuses a run that is not fit to leave.",
                ),
            )
        return Step(Node.EMIT, Node.VERIFY, self._record(f"emitted {self.emitted}"))

    def _verify(self) -> Step:
        """The agent checks its own work, and a failure is not a footnote."""
        if self.emitted is None:
            return Step(
                Node.VERIFY, Node.DONE, self._record("verification skipped: nothing emitted")
            )
        self.checks = self.tools.verify_output(self.emitted)
        failures = [c for c in self.checks if c.failed]
        narration = self._record(
            f"verification: {len(self.checks) - len(failures)} of {len(self.checks)} "
            f"checks passed or skipped, {len(failures)} failed"
        )
        if failures:
            detail = "\n".join(f"  {c.number}. {c.name}: {c.detail}" for c in failures)
            return Step(
                Node.VERIFY,
                Node.VERIFY,
                narration,
                Interrupt(
                    Node.VERIFY,
                    f"The emitted file fails {len(failures)} check(s):\n{detail}",
                    because=(
                        "A failed check names an invariant that broke. It is not a "
                        "warning to read past."
                    ),
                ),
            )
        return Step(Node.VERIFY, Node.DONE, narration)

    # --- narration -----------------------------------------------------------

    def narrate(self) -> str:
        """Describe the run from what actually happened.

        Generated from ``execution_history`` and state rather than authored, so
        it cannot drift from the run it claims to describe.
        """
        lines = [f"Run {self.state.run_id}:"]
        lines.extend(f"  - {entry}" for entry in self.state.execution_history)

        if self.state.lanes_disagreed:
            lines.append(
                "  ! the numeric and visual readings disagreed, which usually means "
                "the run is unusual rather than that a lane is wrong"
            )
        if self.state.fallback_taken:
            lines.append(
                "  ! no judge was reachable; the conservative option in the safe "
                "band was taken and the result remains valid"
            )
        for reason in self.state.blocked:
            lines.append(f"  blocked: {reason}")
        if self.pending:
            lines.append(f"  waiting on you: {self.pending.question.splitlines()[0]}")
        return "\n".join(lines)


GREETING = (
    "Point me at an ARGEN run, or ask me about the one we just finished."
)


__all__ = ["GREETING", "AgentGraph", "Interrupt", "Node", "Step"]
