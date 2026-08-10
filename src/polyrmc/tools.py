"""The tool surface: read, compute, commit.

Design section 10.2. The conversation above this layer is unrestricted -- a
model may attempt anything. It can only *act* through these functions, and the
commit tier refuses invalid actions. That is the whole point: **validation lives
in tools, not in prompts.** A prompt that asks a model not to do something is a
request; a tool that rejects it is a guarantee, and it keeps working when the
model is having a bad day, is a different model, or is replaced by a script.

Three tiers, with different powers:

*Read* returns what is already known. It cannot change anything.

*Compute* runs deterministic measurement code. The caller invokes; it never
performs the arithmetic itself.

*Commit* changes the run, and validates first. A rejection is the caller's
error-recovery signal, so it says what failed, by how much, and what would be
acceptable instead -- a bare "invalid" leaves a model to guess, and it will
guess wrong in a way that looks like progress.

No model client appears anywhere in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polyrmc.config import RunConfig
from polyrmc.intake import DeclarationSet, build_intake_form
from polyrmc.state import (
    AnchorCandidate,
    AnomalyRecord,
    AnomalyType,
    Conflict,
    RunState,
)
from polyrmc.tier0.anchors import (
    apply_correction,
    evaluate_drift,
    fit_drift,
    revert_correction,
    validate_selection,
)
from polyrmc.tier0.consistency import check_declarations
from polyrmc.verify import CheckResult, verify_output


class Rejected(Exception):
    """A commit refused, with an actionable reason.

    The message format is deliberate. It names what failed, the measured value
    against the limit, and what would be valid instead -- everything the caller
    needs to correct itself without another round trip.
    """


@dataclass
class Result:
    """Outcome of a commit call."""

    ok: bool
    message: str
    changed: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSurface:
    """Every action available against one run.

    Holds the state and config rather than taking them per call, so a caller
    cannot accidentally validate against one run and commit against another.
    """

    state: RunState
    config: RunConfig
    declarations: DeclarationSet = field(default_factory=DeclarationSet)

    # Samples no judgment may treat as clean signal. Computed at load time and
    # kept, so anchor validation and the window sweep agree on what is suspect.
    _suspect: np.ndarray | None = None

    # --- read tier: unrestricted, local, cannot change anything --------------

    def get_state(self) -> RunState:
        return self.state

    def list_anomalies(self, type: str | None = None) -> list[AnomalyRecord]:
        if type is None:
            return list(self.state.anomalies)
        return [a for a in self.state.anomalies if a.anomaly_type == type]

    def get_shortlist(self) -> list[Any]:
        loop = self.state.loops.get("smoothing_window")
        if loop is None or not loop.candidates_by_iteration:
            return []
        return list(loop.candidates_by_iteration[-1])

    def get_intake_form(self, metadata: dict[str, str] | None = None):
        return build_intake_form(
            metadata=metadata or {}, declarations=self.state.declarations
        )

    def list_conflicts(self) -> list[Conflict]:
        return list(self.state.declaration_conflicts)

    def list_anchor_candidates(self) -> list[AnchorCandidate]:
        return list(self.state.anchor_candidates)

    def list_blocked(self) -> list[str]:
        return list(self.state.blocked)

    def list_paths(self) -> list[Any]:
        return list(self.state.paths)

    def search_past_runs(
        self,
        query: str | None = None,
        sample_name: str | None = None,
        runs_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Search historical .provenance.json sidecar files from previous research."""
        from polyrmc.memory.registry import RunRegistry

        dir_to_scan = runs_dir or Path("data/runs")
        registry = RunRegistry(dir_to_scan)
        matches = registry.search(query=query, sample_name=sample_name)
        return [m.to_dict() for m in matches]

    def declared(self, key: str, default: str = "unknown") -> str:
        """The value a person declared for ``key``, as text.

        ``state.declarations`` holds :class:`~polyrmc.state.Declaration` models,
        not bare values. Reaching for ``.get(key).get("value")`` treats one as a
        dict and raises the moment a declaration actually exists -- which is to
        say, on every real run and on none of the empty ones.
        """
        declaration = self.state.declarations.get(key)
        if declaration is None:
            return default
        value = getattr(declaration, "value", declaration)
        return default if value is None else str(value)

    def compare_current_run(
        self, runs_dir: str | Path | None = None
    ) -> dict[str, Any]:
        """Compare current run results against historical benchmark runs."""
        from polyrmc.memory.registry import RunRegistry

        dir_to_scan = runs_dir or Path("data/runs")
        registry = RunRegistry(dir_to_scan)
        target_summary = {
            "run_id": self.state.run_id,
            "sample_name": self.declared("sample_name"),
            "a2_mw": self.state.parameters.a2_mw if self.state.parameters else None,
            "k_d": self.state.parameters.k_d if self.state.parameters else None,
        }
        return registry.compare(target_summary)

    def search_historical_figures(
        self,
        query: str | None = None,
        method: str | None = None,
        sample_type: str | None = None,
        goal: str | None = None,
        ledger_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Search labelled figures from past reports, via the figure-labeller ledger.

        The path is supplied by the caller or read from ``POLYRMC_FIGURE_LEDGER``.
        It used to be a hardcoded absolute path into one person's home
        directory, and a second call site carried a *different* hardcoded path,
        so the chat session and the tool surface searched different ledgers and
        neither said which.
        """
        import os

        from polyrmc.memory.registry import FigureRegistry

        target = ledger_path or os.environ.get("POLYRMC_FIGURE_LEDGER")
        if not target:
            return []
        registry = FigureRegistry(target)
        matches = registry.search(query=query, method=method, sample_type=sample_type, goal=goal)
        return [asdict(m) for m in matches]

    # --- compute tier: deterministic; the caller invokes, never performs -----

    def load_run(self, path: str | Path | None = None) -> RunState:
        """Ingest, clean, detect, splice, censor, propose anchors.

        Everything deterministic up to the point a smoothing window has to be
        chosen. Stops there on purpose: the caller makes that judgment.
        """
        from polyrmc.pipeline import prepare_run

        if path is not None:
            self.config.source_file = Path(path)
        state, suspect = prepare_run(self.config)
        # Declarations belong to the run, not to the loading of it, so they
        # survive a reload.
        state.declarations = dict(self.state.declarations)
        state.declaration_conflicts = list(self.state.declaration_conflicts)
        self.state = state
        self._suspect = suspect
        return state

    def sweep_windows(self) -> list[Any]:
        """Compute the safe band. Does not resolve it."""
        from polyrmc.pipeline import sweep_windows

        if self.state.corrected_signal is None:
            raise Rejected(
                "REJECTED. sweep_windows needs a loaded run. Call load_run first."
            )
        return sweep_windows(self.state, self.config, self._suspect)


    def check_declarations(self, metadata: dict[str, str] | None = None) -> list[Conflict]:
        """Test every declaration against the trace, and record what conflicts."""
        if self.state.time_s is None or self.state.raw_signal is None:
            raise Rejected(
                "REJECTED. check_declarations needs a loaded run. Call load_run first."
            )
        declared = {k: d.value for k, d in self.state.declarations.items()}
        report = check_declarations(
            self.state.time_s,
            self.state.raw_signal,
            declared,
            metadata=metadata or {},
        )
        self.state.declaration_conflicts = report.conflicts
        return report.conflicts

    def fit_drift(self, anchor_ids: list[int], kind: str = "poly"):
        """Fit a drift model to the named candidates. Does not apply it.

        Fitting is separate from applying because the two ceilings that decide
        whether a correction may be offered at all are measured at fit time --
        before anything is subtracted, not after.
        """
        selected = self._resolve_anchors(anchor_ids)
        return fit_drift(
            self.state.time_s,
            self.state.spliced_signal,
            selected,
            kind=kind,
            config=self.config.baseline,
        )

    def verify_output(self, processed_csv: str | Path) -> list[CheckResult]:
        return verify_output(
            processed_csv,
            self.state,
            m0_sensitivity_ceiling=self.config.m0_sensitivity_ceiling,
            drift_slope_ceiling=self.config.baseline.drift_slope_ceiling,
        )

    # --- commit tier: validates, and rejects ---------------------------------

    def set_declaration(self, key: str, value: Any, author: str) -> Result:
        """Record what a person says this run is."""
        try:
            declaration = self.declarations.record(key, value, author=author)
        except (KeyError, ValueError) as error:
            raise Rejected(f"REJECTED. {error}") from error
        self.state.declarations[key] = declaration
        for derived_key, derived in self.declarations.declarations.items():
            if derived.derived_from == key:
                self.state.declarations[derived_key] = derived
        self.declarations.refresh_form()
        return Result(True, f"{key} = {value!r}, declared by {author}", {key: value})

    def resolve_conflict(self, check: str, reason: str, author: str) -> Result:
        """Mark a conflict as understood, with a recorded reason.

        Resolving is an assertion by a named person that the disagreement is
        explained. It never happens by default, and never without a reason:
        an unexplained dismissal is indistinguishable from not having looked.
        """
        if not str(reason).strip():
            raise Rejected(
                f"REJECTED. Resolving {check!r} requires a reason. A conflict "
                "dismissed without one cannot be reviewed later."
            )
        if not str(author).strip():
            raise Rejected(f"REJECTED. Resolving {check!r} requires a named author.")

        for conflict in self.state.declaration_conflicts:
            if conflict.check == check:
                conflict.resolved = True
                conflict.resolution_reason = str(reason).strip()
                conflict.resolved_by = str(author).strip()
                return Result(True, f"{check} resolved by {author}")

        known = [c.check for c in self.state.declaration_conflicts]
        raise Rejected(
            f"REJECTED. There is no conflict named {check!r}. "
            f"Open conflicts: {known or 'none'}."
        )

    def pick_window(self, window_length: int) -> Result:
        """Choose the smoothing window, from inside the safe band only."""
        band = [int(c.value) for c in self.get_shortlist()]
        if not band:
            raise Rejected(
                "REJECTED. No safe band has been computed, so no window can be "
                "picked. Run the smoothing sweep first."
            )
        polyorder = self.config.smoothing.polyorder
        if window_length % 2 == 0:
            raise Rejected(
                f"REJECTED. Window {window_length} is even. Savitzky-Golay requires "
                f"an odd window. Safe band is {sorted(band)}."
            )
        if window_length <= polyorder:
            raise Rejected(
                f"REJECTED. Window {window_length} does not exceed polyorder "
                f"{polyorder}. Safe band is {sorted(band)}."
            )
        if window_length not in band:
            raise Rejected(
                f"REJECTED. Window {window_length} is not in the safe band. "
                f"Safe band is {sorted(band)}. Choose from the band, or state a case "
                "for relaxing the landmark tolerance and ask the researcher."
            )
        self.state.window_length = window_length
        return Result(True, f"window {window_length} selected", {"window": window_length})

    def set_decision(self, index_range: tuple[int, int], decision: str) -> Result:
        """Record what to do with a flagged region."""
        allowed = {"excise", "keep", "correct"}
        if decision not in allowed:
            raise Rejected(
                f"REJECTED. {decision!r} is not a decision. Valid: {sorted(allowed)}."
            )
        for record in self.state.anomalies:
            if (record.start_index, record.end_index) == tuple(index_range):
                if decision == "correct" and record.anomaly_type is not (
                    AnomalyType.BASELINE_DRIFT
                ):
                    raise Rejected(
                        f"REJECTED. 'correct' applies only to baseline_drift; this "
                        f"region is {record.anomaly_type}. Drift is subtracted; "
                        "everything else is excised or kept."
                    )
                record.decision = decision
                return Result(True, f"{index_range} -> {decision}")
        known = [(r.start_index, r.end_index) for r in self.state.anomalies]
        raise Rejected(
            f"REJECTED. No detected region matches {tuple(index_range)}. "
            f"Detected regions: {known or 'none'}."
        )

    def select_anchors(self, anchor_ids: list[int], selected_by: str) -> Result:
        """Choose baseline anchors, over the code-generated candidates only."""
        selected = self._resolve_anchors(anchor_ids)
        check = validate_selection(
            selected,
            self.state.time_s,
            config=self.config.baseline,
            excluded=self._excluded_mask(),
            transition=self.state.transition_region,
        )
        if not check.valid:
            raise Rejected("REJECTED. " + " ".join(check.rejections))

        self.state.anchors_selected = list(anchor_ids)
        self.state.anchor_selected_by = selected_by
        message = f"{len(anchor_ids)} anchor(s) selected by {selected_by}"
        if check.confirmations:
            message += ". Needs confirmation: " + " ".join(check.confirmations)
        return Result(True, message, {"confirmations": check.confirmations})

    def apply_baseline_correction(self, kind: str = "poly", override: bool = False) -> Result:
        """Subtract the fitted drift. Never automatic.

        ``override`` is the researcher's explicit decision to proceed past a
        ceiling. It is recorded, and it is the only way past one.
        """
        if not self.state.anchors_selected:
            raise Rejected(
                "REJECTED. No validated anchor selection exists. Call select_anchors "
                "first; a correction without one has nothing to fit to."
            )
        if self.state.baseline_applied:
            raise Rejected(
                "REJECTED. A correction is already applied. Revert it before "
                "applying another, so the uncorrected values stay recoverable."
            )

        selected = self._resolve_anchors(self.state.anchors_selected)
        drift = fit_drift(
            self.state.time_s,
            self.state.spliced_signal,
            selected,
            kind=kind,
            config=self.config.baseline,
        )
        check = validate_selection(
            selected,
            self.state.time_s,
            config=self.config.baseline,
            excluded=self._excluded_mask(),
            transition=self.state.transition_region,
            drift=drift,
        )
        if not check.valid and not override:
            raise Rejected(
                "REJECTED. "
                + " ".join(check.rejections)
                + " Pass override=True only with explicit researcher approval, which "
                "will be recorded."
            )

        curve = evaluate_drift(drift, self.state.time_s)
        self.state.baseline = curve
        self.state.corrected_signal = apply_correction(self.state.spliced_signal, curve)
        self.state.drift_model = drift
        self.state.baseline_applied = True
        if override:
            self.state.warnings.append(
                "baseline correction applied over a failed ceiling by explicit "
                "override: " + " ".join(check.rejections)
            )
        return Result(
            True,
            f"{kind} correction applied over {len(selected)} anchor(s); "
            f"slope ratio {drift.max_slope_ratio:.3f}, fold-change impact "
            f"{drift.fold_change_impact:+.3f}",
        )

    def revert_baseline_correction(self) -> Result:
        """Undo the correction, recovering the uncorrected values exactly."""
        if not self.state.baseline_applied or self.state.drift_model is None:
            raise Rejected("REJECTED. No correction is applied, so there is nothing to revert.")
        curve = evaluate_drift(self.state.drift_model, self.state.time_s)
        self.state.spliced_signal = revert_correction(self.state.corrected_signal, curve)
        self.state.corrected_signal = self.state.spliced_signal
        self.state.baseline = np.zeros_like(self.state.spliced_signal)
        self.state.baseline_applied = False
        self.state.drift_model = None
        return Result(True, "correction reverted; uncorrected values restored")

    def set_config(self, key: str, value: Any) -> Result:
        """Change one configuration value, validated against its own schema."""
        target, _, leaf = key.rpartition(".")
        holder = self.config
        for part in filter(None, target.split(".")):
            holder = getattr(holder, part, None)
            if holder is None:
                raise Rejected(f"REJECTED. {key!r} is not a configuration path.")
        if leaf not in type(holder).model_fields:
            valid = sorted(type(holder).model_fields)
            raise Rejected(f"REJECTED. {key!r} is not a known config key. Valid: {valid}.")
        try:
            setattr(holder, leaf, value)
        except Exception as error:  # pydantic validation
            raise Rejected(f"REJECTED. {key} = {value!r} is invalid: {error}") from error
        return Result(True, f"{key} = {value!r}", {key: value})

    def emit_csv(self, path: str | Path | None = None) -> Result:
        """Compute the columns and write the boundary CSV, or refuse.

        The refusal comes first. A run carrying an unresolved conflict, no
        chosen window, or an undecided anomaly region is not fit to leave, and
        the gate is here rather than in a prompt so it holds regardless of who
        or what is driving.
        """
        reasons: list[str] = []
        if self.state.unresolved_conflicts:
            names = ", ".join(c.check for c in self.state.unresolved_conflicts)
            reasons.append(
                f"{len(self.state.unresolved_conflicts)} unresolved declaration "
                f"conflict(s): {names}. A run processed under a declaration the data "
                "contradicts must not be emitted."
            )
        if self.state.window_length is None and not self.state.loops.get(
            "smoothing_window"
        ):
            reasons.append("no smoothing window has been chosen")
        pending = [
            (r.start_index, r.end_index)
            for r in self.state.anomalies
            if r.decision == "pending"
        ]
        if pending:
            reasons.append(
                f"{len(pending)} anomaly region(s) still pending a decision: "
                f"{pending[:5]}"
            )
        if reasons:
            raise Rejected("REJECTED. " + " ".join(reasons))

        from polyrmc.pipeline import finish_run

        if self.state.corrected_signal is None:
            raise Rejected(
                "REJECTED. emit_csv needs a loaded run. Call load_run first."
            )
        _state, written = finish_run(
            self.state,
            self.config,
            self.config.channel,
            int(self.state.window_length),
            output_csv=path,
        )
        return Result(True, f"emitted {written}", {"path": str(written)})

    # --- internals -----------------------------------------------------------

    def _resolve_anchors(self, anchor_ids: list[int]) -> list[AnchorCandidate]:
        by_id = {c.id: c for c in self.state.anchor_candidates}
        unknown = [i for i in anchor_ids if i not in by_id]
        if unknown:
            raise Rejected(
                f"REJECTED. Unknown anchor id(s) {unknown}. "
                f"Available: {sorted(by_id) or 'none -- no candidates were proposed'}. "
                "Anchors are selected from the generated candidate set; free-form "
                "index ranges are not accepted."
            )
        return [by_id[i] for i in anchor_ids]

    def _excluded_mask(self) -> np.ndarray | None:
        if self.state.raw_signal is None:
            return None
        mask = np.zeros(self.state.raw_signal.size, dtype=bool)
        for record in self.state.anomalies:
            mask[record.start_index : record.end_index + 1] = True
        if self.state.censored_indices:
            mask[self.state.censored_indices] = True
        return mask


__all__ = ["Rejected", "Result", "ToolSurface"]
