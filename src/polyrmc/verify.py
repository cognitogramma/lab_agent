"""Checks the pipeline runs against its own output.

Design section 13. This is what makes the system self-correcting rather than
confidently wrong: the agent can call it on work it has just produced, and a
failure names the invariant that broke rather than reporting a number that
looks fine.

Every check is deterministic and reads only the emitted CSV and the run state.
None of them needs a model, and none of them may pass by default -- a check that
cannot run reports ``skipped`` with the reason, because a silently-passed check
leaves a run carrying the appearance of verification it never had.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from polyrmc.io_csv import COLUMNS, read_processed_csv
from polyrmc.state import RunState
from polyrmc.tier0.anchors import evaluate_drift

Status = Literal["pass", "fail", "skipped"]


@dataclass
class CheckResult:
    """One verification check and what it found."""

    number: int
    name: str
    status: Status
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == "fail"


def verify_output(
    processed_csv: str | Path,
    state: RunState,
    m0_sensitivity_ceiling: float = 0.05,
    drift_slope_ceiling: float = 0.10,
    max_residual_acf: float = 0.2,
) -> list[CheckResult]:
    """Run every section 13 check against an emitted run.

    Check 1 -- identical inputs give a byte-identical CSV -- is absent by
    design: it compares two runs, and this function sees one. It lives in the
    regression suite instead.
    """
    frame, metadata = read_processed_csv(processed_csv)
    results: list[CheckResult] = [
        _check_declarations_consistent(state),
        _check_safe_band_non_empty(state),
        _check_residual_whitens(state, max_residual_acf),
        _check_mw_over_m0_starts_at_one(frame),
        _check_layout(frame, metadata),
        _check_part2_can_build_its_axis(frame),
        _check_monotonic_seconds(frame, state),
        _check_no_reindexing(frame, state),
        _check_reversibility(state),
        _check_drift_slope(state, drift_slope_ceiling),
        _check_anchor_guard(state),
        _check_m0_sensitivity(state, m0_sensitivity_ceiling),
        _check_censored_columns_are_blank(frame, state),
    ]
    return results


def _check_safe_band_non_empty(state: RunState) -> CheckResult:
    """2 -- an empty safe band means the sweep or the tolerances are wrong.

    An empty band is a signal, not a crash. It says the sweep range is too
    narrow, the tolerances too tight, or the trace genuinely pathological -- and
    the fix is to look, never to widen the tolerance until something passes.
    """
    loop = state.loops.get("smoothing_window")
    if loop is None:
        return CheckResult(2, "safe band non-empty", "skipped", "no sweep was run")
    band = [c.value for c in (loop.candidates_by_iteration or [[]])[-1]]
    if not band:
        return CheckResult(
            2,
            "safe band non-empty",
            "fail",
            "no window passed both the whitening and landmark gates. Widen the "
            "sweep range or examine the trace; do not relax the tolerances to "
            "make something pass.",
        )
    return CheckResult(
        2, "safe band non-empty", "pass", f"{len(band)} window(s) certified: {band}"
    )


def _check_residual_whitens(state: RunState, ceiling: float) -> CheckResult:
    """3 -- at the chosen window the residual is structureless.

    A clearly positive lag-1 autocorrelation means the filter left real
    structure behind, so it is under-fitting. Negative values are expected from
    a Savitzky-Golay filter and are not penalised.
    """
    if state.smoothed_signal is None or state.spliced_signal is None:
        return CheckResult(
            3, "residual whitens", "skipped", "the run carries no smoothed trace"
        )
    from polyrmc.tier0.smoothing import residual_autocorrelation

    acf = residual_autocorrelation(state.spliced_signal, state.smoothed_signal)
    if acf > ceiling:
        return CheckResult(
            3,
            "residual whitens",
            "fail",
            f"residual lag-1 autocorrelation is {acf:.3f}, above the {ceiling:.2f} "
            f"ceiling at window {state.window_length}: the filter is leaving real "
            "structure in the residual",
        )
    return CheckResult(
        3,
        "residual whitens",
        "pass",
        f"residual lag-1 autocorrelation {acf:.3f} at window {state.window_length}",
    )


def summarize(results: list[CheckResult]) -> str:
    """One line per check, for a terminal or a narration."""
    lines = []
    for result in results:
        marker = {"pass": "ok     ", "fail": "FAILED ", "skipped": "skipped"}[
            result.status
        ]
        lines.append(f"  {marker} {result.number:2d}. {result.name}: {result.detail}")
    return "\n".join(lines)


def _check_layout(frame, metadata) -> CheckResult:
    """5 -- the file round-trips under its own declared header length."""
    absent = [name for name in COLUMNS if name not in frame.columns]
    if absent:
        return CheckResult(5, "layout A round-trip", "fail", f"missing {absent}")
    if not metadata:
        return CheckResult(
            5, "layout A round-trip", "fail", "no header metadata was recovered"
        )
    return CheckResult(
        5,
        "layout A round-trip",
        "pass",
        f"re-read {len(frame)} rows and {len(metadata)} header field(s) using the "
        "declared length",
    )


def _check_monotonic_seconds(frame, state: RunState) -> CheckResult:
    """6a -- seconds is strictly monotonic."""
    seconds = frame["seconds"].to_numpy(dtype=float)
    if seconds.size < 2:
        return CheckResult(6, "seconds monotonic", "skipped", "fewer than two rows")
    steps = np.diff(seconds)
    if np.any(steps <= 0):
        first = int(np.flatnonzero(steps <= 0)[0])
        return CheckResult(
            6,
            "seconds monotonic",
            "fail",
            f"time does not advance at row {first} ({seconds[first]} -> "
            f"{seconds[first + 1]}); the de-wrap failed or rows were reordered",
        )
    return CheckResult(
        6, "seconds monotonic", "pass", f"strictly increasing across {seconds.size} rows"
    )


def _check_no_reindexing(frame, state: RunState) -> CheckResult:
    """6b -- surviving timestamps match the raw ones at their own indices."""
    if state.time_s is None:
        return CheckResult(6, "no re-indexing", "skipped", "run state carries no time axis")
    seconds = frame["seconds"].to_numpy(dtype=float)
    if seconds.size != state.time_s.size:
        return CheckResult(
            6,
            "no re-indexing",
            "fail",
            f"the emitted file has {seconds.size} rows but the run carried "
            f"{state.time_s.size}. Survivors were re-indexed, which shortens the "
            "time axis and distorts every derivative taken from it.",
        )
    if not np.allclose(seconds, state.time_s, rtol=0, atol=1e-9):
        return CheckResult(
            6, "no re-indexing", "fail", "emitted timestamps differ from the raw ones"
        )
    return CheckResult(
        6, "no re-indexing", "pass", "every timestamp sits at its original index"
    )


def _check_censored_columns_are_blank(frame, state: RunState) -> CheckResult:
    """11 -- no saturated index carries a value in any computed column."""
    if not state.censored_indices:
        return CheckResult(
            11,
            "censored values excluded",
            "skipped",
            "nothing was censored: either no sample reached the ceiling, or no "
            "ceiling has been declared for this instrument",
        )
    index = np.asarray(state.censored_indices, dtype=int)
    leaked: list[str] = []
    for name in ("ls8", "ls8_smoothed", "I_rayleigh", "Mw_over_M0"):
        values = frame[name].to_numpy(dtype=float)
        inside = index[index < values.size]
        if inside.size and np.isfinite(values[inside]).any():
            leaked.append(name)
    if leaked:
        return CheckResult(
            11,
            "censored values excluded",
            "fail",
            f"{leaked} carry values at censored indices. Saturation is censoring: "
            "capping or retention leaked through.",
        )
    return CheckResult(
        11,
        "censored values excluded",
        "pass",
        f"{index.size} censored sample(s) blank in every computed column",
    )


def _check_anchor_guard(state: RunState) -> CheckResult:
    """9 -- no selected anchor intersects the transition region."""
    if not state.anchors_selected:
        return CheckResult(
            9, "anchor guard", "skipped", "no anchors were selected on this run"
        )
    if state.transition_region is None:
        return CheckResult(
            9,
            "anchor guard",
            "fail",
            "anchors were selected but no transition region was established, so the "
            "guard that makes lassoing the kinetics impossible never ran",
        )
    low, high = state.transition_region
    by_id = {c.id: c for c in state.anchor_candidates}
    for anchor_id in state.anchors_selected:
        anchor = by_id.get(anchor_id)
        if anchor is None:
            return CheckResult(
                9, "anchor guard", "fail", f"selected anchor {anchor_id} is not a candidate"
            )
        start, end = anchor.index_range
        if start <= high and end >= low:
            return CheckResult(
                9,
                "anchor guard",
                "fail",
                f"anchor {anchor_id} ({start}-{end}) intersects the transition "
                f"({low}-{high})",
            )
    return CheckResult(
        9,
        "anchor guard",
        "pass",
        f"{len(state.anchors_selected)} anchor(s), none touching the transition",
    )


def _check_drift_slope(state: RunState, ceiling: float) -> CheckResult:
    """8 -- the drift model is slower than the kinetics."""
    if state.drift_model is None:
        return CheckResult(8, "drift slope ceiling", "skipped", "no drift model was fitted")
    ratio = state.drift_model.max_slope_ratio
    if ratio > ceiling:
        return CheckResult(
            8,
            "drift slope ceiling",
            "fail",
            f"drift moves at {100 * ratio:.1f}% of the signal's own maximum slope, "
            f"above the {100 * ceiling:.0f}% ceiling: it is absorbing the kinetics",
        )
    return CheckResult(
        8,
        "drift slope ceiling",
        "pass",
        f"drift moves at {100 * ratio:.1f}% of the signal's maximum slope",
    )


def _check_reversibility(state: RunState) -> CheckResult:
    """7 -- reverting an applied correction reproduces the uncorrected values."""
    if not state.baseline_applied or state.drift_model is None:
        return CheckResult(
            7, "correction reversible", "skipped", "no correction was applied"
        )
    if state.corrected_signal is None or state.spliced_signal is None:
        return CheckResult(
            7,
            "correction reversible",
            "fail",
            "a correction was applied but the uncorrected values were not retained",
        )
    curve = evaluate_drift(state.drift_model, state.time_s)
    recovered = state.corrected_signal + curve
    finite = np.isfinite(state.spliced_signal) & np.isfinite(recovered)
    if not np.allclose(
        recovered[finite], state.spliced_signal[finite], rtol=0, atol=1e-9
    ):
        return CheckResult(
            7,
            "correction reversible",
            "fail",
            "adding the fitted drift back does not reproduce the uncorrected values",
        )
    return CheckResult(
        7, "correction reversible", "pass", "revert reproduces the uncorrected values"
    )


def _check_m0_sensitivity(state: RunState, ceiling: float) -> CheckResult:
    """10 -- the two M0 definitions agree, or the run is flagged."""
    m0 = state.m0
    if m0 is None:
        return CheckResult(10, "M0 sensitivity", "skipped", "no M0 was selected")
    divergence = abs(m0.sensitivity)
    if not np.isfinite(divergence):
        return CheckResult(
            10, "M0 sensitivity", "skipped", "the divergence could not be computed"
        )
    if divergence > ceiling:
        flagged = bool(m0.escalations or m0.blocked)
        if flagged:
            return CheckResult(
                10,
                "M0 sensitivity",
                "pass",
                f"the definitions diverge by {100 * divergence:.1f}%, and the run is "
                "flagged for it",
            )
        return CheckResult(
            10,
            "M0 sensitivity",
            "fail",
            f"the definitions diverge by {100 * divergence:.1f}%, above the "
            f"{100 * ceiling:.0f}% ceiling, without the run being flagged",
        )
    return CheckResult(
        10,
        "M0 sensitivity",
        "pass",
        f"the two definitions agree to {100 * divergence:.1f}%, so the choice does "
        "not move the result",
    )


def _check_mw_over_m0_starts_at_one(frame) -> CheckResult:
    """4 -- Mw/M0 is 1.0 at the anchor and plausible thereafter."""
    values = frame["Mw_over_M0"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return CheckResult(
            4, "Mw/M0 anchored", "skipped", "the column is blocked on this run"
        )
    opening = float(np.median(finite[: min(50, finite.size)]))
    if abs(opening - 1.0) > 0.10:
        return CheckResult(
            4,
            "Mw/M0 anchored",
            "fail",
            f"the column opens at {opening:.3f} rather than 1.0, which points at a "
            "baseline or calibration error",
        )
    if np.any(finite < 0):
        return CheckResult(
            4,
            "Mw/M0 anchored",
            "fail",
            "the column goes negative, which is not a physical aggregation index",
        )
    return CheckResult(
        4,
        "Mw/M0 anchored",
        "pass",
        f"opens at {opening:.3f} and stays positive, ending at {finite[-1]:.3f}",
    )


def _check_part2_can_build_its_axis(frame) -> CheckResult:
    """5b -- the emitted file supports building Kc/I_R against c."""
    concentration = frame["concentration"].to_numpy(dtype=float)
    intensity = frame["I_rayleigh"].to_numpy(dtype=float)
    if not np.isfinite(intensity).any():
        intensity = frame["ls8_smoothed"].to_numpy(dtype=float)
        basis = "ls8_smoothed (I_rayleigh blocked)"
    else:
        basis = "I_rayleigh"

    usable = np.isfinite(concentration) & np.isfinite(intensity) & (intensity != 0)
    if usable.sum() < 2:
        return CheckResult(
            5,
            "part 2 can build Kc/I vs c",
            "skipped",
            "fewer than two rows carry both a concentration and an intensity",
        )
    varies = not np.allclose(
        concentration[usable], concentration[usable][0], rtol=1e-12
    )
    if not varies:
        return CheckResult(
            5,
            "part 2 can build Kc/I vs c",
            "skipped",
            "concentration is constant: a fixed-concentration run has one point on "
            "the axis and no virial slope to fit",
        )
    return CheckResult(
        5,
        "part 2 can build Kc/I vs c",
        "pass",
        f"{int(usable.sum())} usable points against {basis}",
    )


def _check_declarations_consistent(state: RunState) -> CheckResult:
    """0 -- beyond section 13: a contradicted declaration must not reach a file.

    Numbered zero because it is not one of the specification's eleven. It sits
    before them: the other checks all ask whether the processing was done
    correctly, and this one asks whether it was done on the right physical model
    at all. A run that is flawless under the wrong model passes everything else.
    """
    unresolved = state.unresolved_conflicts
    if unresolved:
        names = ", ".join(c.check for c in unresolved)
        return CheckResult(
            0,
            "declarations consistent",
            "fail",
            f"{len(unresolved)} unresolved conflict(s) ({names}). A run processed "
            "under a declaration the data contradicts should not have been emitted.",
        )
    if not state.declarations:
        return CheckResult(
            0,
            "declarations consistent",
            "skipped",
            "no declarations were recorded on this run",
        )
    return CheckResult(
        0,
        "declarations consistent",
        "pass",
        f"{len(state.declarations)} declaration(s), no unresolved conflicts",
    )


__all__ = ["CheckResult", "summarize", "verify_output"]
