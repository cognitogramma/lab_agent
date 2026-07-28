"""Run orchestration with a human confirmation gate.

Automatic experiment-type classification is on the roadmap but is not wired in.
The reason for keeping a human in this loop stands on its own: a misclassified
run produces numbers that look completely normal and are completely wrong.

:class:`Orchestrator` therefore never infers experiment type. It stages runs and
requires an explicit confirmation before any of them executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from polyrmc.config import RunConfig
from polyrmc.pipeline import new_run_id, run_part1, run_part2
from polyrmc.provenance import summarize_loop
from polyrmc.state import RunState
from polyrmc.tier1.loop import Judge

ExperimentType = Literal["dilution_trajectory", "fixed_concentration"]
Confirmer = Callable[["PendingRun"], ExperimentType | None]
"""Returns the confirmed experiment type, or None to skip the run."""


@dataclass
class PendingRun:
    """A file staged for processing, awaiting human confirmation."""

    path: Path
    proposed_type: ExperimentType | None = None
    reason: str = "experiment type is supplied, never inferred"
    confirmed_type: ExperimentType | None = None

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed_type is not None


@dataclass
class Orchestrator:
    """Stages runs, requires confirmation, then executes the pipeline."""

    template: RunConfig
    judge: Judge | None = None
    pending: list[PendingRun] = field(default_factory=list)
    completed: list[RunState] = field(default_factory=list)

    def discover(self, directory: str | Path, pattern: str = "*.txt") -> list[PendingRun]:
        """Stage every matching file in a directory. Nothing runs yet."""
        found = [PendingRun(path=path) for path in sorted(Path(directory).glob(pattern))]
        self.pending.extend(found)
        return found

    def confirm_all(self, confirmer: Confirmer) -> None:
        """Ask the confirmer for each staged run's experiment type."""
        for run in self.pending:
            if run.is_confirmed:
                continue
            run.confirmed_type = confirmer(run)

    def execute(self, run: PendingRun, channel: str | None = None) -> RunState:
        """Process one confirmed run through both halves of the pipeline.

        Raises if the run has not been confirmed. That guard is the whole point
        of this class and is not optional.
        """
        if not run.is_confirmed:
            raise RuntimeError(
                f"{run.path.name} has no confirmed experiment type. The orchestrator "
                "does not infer it: a misclassified run yields numbers that look "
                "normal and are wrong."
            )

        config = self.template.model_copy(
            update={
                "run_id": new_run_id(),
                "source_file": run.path,
                "experiment_type": run.confirmed_type,
            }
        )

        state, csv_path = run_part1(config, channel=channel, judge=self.judge)
        if run.confirmed_type == "dilution_trajectory":
            part2 = run_part2(csv_path, config, judge=self.judge)
            state.fit = part2.fit
            state.parameters = part2.parameters
            state.loops.update(part2.loops)
            state.warnings.extend(part2.warnings)

        self.completed.append(state)
        return state

    def report(self, state: RunState) -> str:
        """Human-readable summary, including judge-versus-fallback status."""
        lines = [f"run {state.run_id} <- {Path(state.source_file).name}"]
        lines.extend(f"  {summarize_loop(loop)}" for loop in state.loops.values())
        if state.parameters:
            parameters = state.parameters
            lines.append(f"  A2*Mw = {parameters.a2_mw:.6g} cm^3/g (calibration-free)")
            if parameters.k_d is not None:
                lines.append(f"  k_D   = {parameters.k_d:.6g}")
            if parameters.calibrated:
                lines.append(f"  Mw    = {parameters.mw:.6g} g/mol")
                lines.append(f"  A2    = {parameters.a2:.6g} cm^3 mol/g^2")
            else:
                lines.append("  Mw, A2: not reported (alpha not calibrated)")
        lines.extend(f"  warning: {warning}" for warning in state.warnings)
        return "\n".join(lines)
