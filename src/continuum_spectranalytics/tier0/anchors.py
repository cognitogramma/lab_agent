"""Baseline correction over a code-generated anchor set.

Design section 6.6. Drift is an amplitude-domain problem, so the drift is
modelled and subtracted with **every timestamp left in place**. Excising a
drifting stretch would delete the real kinetic signal riding on top of it.

    y_obs(t) = y_signal(t) + y_drift(t)

**Why this step carries more risk than smoothing.** Anchor selection determines
what counts as signal. Smoothing errors are bounded: the safe band caps the
damage and the residual test exposes it. Baseline errors are not. Fit a curve to
anchors that were really part of the kinetics, subtract it, and the result is
smooth, plausible, and physically wrong -- and no downstream check will flag it,
because nothing about the corrected trace looks unusual. That asymmetry is why
the default is no correction and why every check below rejects rather than warns.

The researcher's mental model is lassoing baseline regions on a plot. That is
implemented as **selection over a code-generated candidate set**, never as
free-form geometry: code enumerates which regions are eligible, and a judge --
model or human -- picks among them by id. A freehand drag snaps to candidates.
Code owns every index computation.

The load-bearing guard is the transition region. Anchors may not intersect it,
which makes it structurally impossible to lasso the kinetics as baseline. Not
discouraged: impossible.

Nothing here applies anything. :func:`fit_drift` fits and measures, and the two
ceilings that decide whether a correction is even offerable are computed at fit
time -- before the correction is presented, not after it has been applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_spectranalytics.config import BaselineConfig
from continuum_spectranalytics.state import AnchorCandidate, DriftModel
from continuum_spectranalytics.tier0.landmarks import (
    lag_for,
    lag_slope,
    lag_slope_profile,
    noise_equivalent_slope,
    transition_region,
)

# Samples averaged at each end when measuring total fold change. A single
# endpoint carries the full noise of one measurement.
FOLD_CHANGE_POINTS = 50


@dataclass
class SelectionCheck:
    """Outcome of validating an anchor selection."""

    rejections: list[str] = field(default_factory=list)
    """Conditions that make the selection invalid. Non-empty means refuse."""

    confirmations: list[str] = field(default_factory=list)
    """Conditions permitted only with explicit researcher confirmation."""

    @property
    def valid(self) -> bool:
        return not self.rejections


def total_fold_change(signal: np.ndarray, n_points: int = FOLD_CHANGE_POINTS) -> float:
    """Ratio of the trace's final level to its initial level."""
    values = np.asarray(signal, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    take = min(n_points, finite.size // 2) or 1
    start = float(np.median(finite[:take]))
    end = float(np.median(finite[-take:]))
    if start == 0:
        return float("nan")
    return end / start


def rolling_median(profile: np.ndarray, window: int) -> np.ndarray:
    """Median of a profile over a rolling window, ignoring NaN.

    The lag-slope at a single sample is one noisy realisation: on a genuinely
    flat stretch it scatters around the noise-equivalent slope, so testing each
    sample against that value individually rejects roughly half of a perfectly
    good anchor region and shatters what remains into fragments too short to
    qualify. Taking the median over the window first estimates the region's
    slope instead of one sample's, which is the quantity the flatness test is
    actually about.
    """
    values = np.asarray(profile, dtype=float)
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    half = max(1, int(window) // 2)
    for i in range(n):
        chunk = values[max(0, i - half) : min(n, i + half + 1)]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            out[i] = float(np.median(chunk))
    return out


def local_scatter(signal: np.ndarray, window: int) -> np.ndarray:
    """Rolling robust scatter, measured on successive differences.

    Differencing first removes the local trend. Raw rolling variance is
    dominated by the trend wherever the signal is moving, so on any sloped
    stretch it reports excess scatter that is not there -- and would disqualify
    exactly the gently sloping regions that make the best anchors.
    """
    values = np.asarray(signal, dtype=float)
    n = values.size
    profile = np.full(n, np.nan, dtype=float)
    if n < 3:
        return profile

    differences = np.diff(values)
    half = max(1, int(window) // 2)
    for i in range(n):
        low = max(0, i - half)
        high = min(differences.size, i + half)
        chunk = differences[low:high]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size >= 2:
            # Successive differences of independent samples inflate the spread
            # by sqrt(2); dividing it back out puts this on the same scale as
            # the noise floor it gets compared against.
            profile[i] = float(np.std(chunk) / np.sqrt(2.0))
    return profile


def propose_anchors(
    time_s: np.ndarray,
    signal: np.ndarray,
    noise_floor: float,
    config: BaselineConfig | None = None,
    excluded: np.ndarray | None = None,
    transition: tuple[int, int] | None = None,
) -> tuple[list[AnchorCandidate], tuple[int, int] | None]:
    """Enumerate the regions eligible to be used as baseline.

    A candidate is a contiguous index range that is near-flat against the noise,
    not unusually scattered, long enough to define a level, free of excised or
    flagged samples, and outside the transition region.

    Returns the candidates and the transition region that constrained them, so a
    caller can render the guard alongside what it excluded.

    Parameters
    ----------
    excluded:
        Boolean mask of samples that may not appear in any anchor -- excised
        indices and every flagged anomaly region.
    transition:
        Precomputed transition region. Computed here when omitted.
    """
    config = config or BaselineConfig()
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(signal, dtype=float)
    n = values.size

    if transition is None:
        transition = transition_region(
            time_s, values, edge_fraction=config.transition_edge_fraction
        )

    if n < 3:
        return [], transition

    lag = lag_for(n)
    # The region's slope, not one sample's -- see rolling_median.
    slope_profile = rolling_median(lag_slope_profile(time_s, values, lag), lag)
    scatter_profile = local_scatter(values, lag)
    slope_ceiling = config.anchor_flatness_tolerance * noise_equivalent_slope(
        time_s, noise_floor, lag
    )
    scatter_ceiling = config.anchor_variance_factor * noise_floor

    # Every condition is required, and an unmeasurable sample fails all of them:
    # "we could not tell" is not "flat".
    eligible = (
        np.isfinite(values)
        & np.isfinite(slope_profile)
        & (slope_profile <= slope_ceiling)
        & np.isfinite(scatter_profile)
        & (scatter_profile <= scatter_ceiling)
    )
    if excluded is not None:
        eligible &= ~np.asarray(excluded, dtype=bool)
    if transition is not None:
        low, high = transition
        eligible[low : high + 1] = False

    total_duration = float(time_s[-1] - time_s[0])
    if total_duration <= 0:
        return [], transition
    min_duration = config.min_anchor_duration * total_duration

    candidates: list[AnchorCandidate] = []
    for start, end in _contiguous(eligible):
        duration = float(time_s[end] - time_s[start])
        if duration < min_duration:
            continue
        window = values[start : end + 1]
        finite = window[np.isfinite(window)]
        if finite.size < 2:
            continue
        candidates.append(
            AnchorCandidate(
                id=len(candidates),
                index_range=(start, end),
                duration_fraction=duration / total_duration,
                mean_level=float(np.mean(finite)),
                local_slope=float(np.nanmedian(slope_profile[start : end + 1])),
                local_variance=float(np.var(finite)),
                position_fraction=(
                    float(time_s[(start + end) // 2] - time_s[0]) / total_duration
                ),
            )
        )
    return candidates, transition


def validate_selection(
    selected: list[AnchorCandidate],
    time_s: np.ndarray,
    config: BaselineConfig | None = None,
    excluded: np.ndarray | None = None,
    transition: tuple[int, int] | None = None,
    drift: DriftModel | None = None,
) -> SelectionCheck:
    """Refuse a selection that cannot support a correction.

    The two ceilings involving ``drift`` are the checks that make this step
    recoverable, and they only exist once a model has been fitted -- which is
    why fitting is separate from applying.
    """
    config = config or BaselineConfig()
    time_s = np.asarray(time_s, dtype=float)
    check = SelectionCheck()

    if len(selected) < config.min_anchors:
        check.rejections.append(
            f"{len(selected)} anchor region(s) selected, fewer than the minimum of "
            f"{config.min_anchors}. A single anchor fixes a level but cannot define "
            "a trend, so there is nothing to extrapolate between."
        )

    total_duration = float(time_s[-1] - time_s[0]) if time_s.size > 1 else 0.0
    if total_duration > 0:
        coverage = sum(
            float(time_s[a.index_range[1]] - time_s[a.index_range[0]]) for a in selected
        ) / total_duration
        if coverage < config.min_anchor_coverage:
            check.rejections.append(
                f"anchors cover {100 * coverage:.1f} percent of the run, below the "
                f"{100 * config.min_anchor_coverage:.0f} percent minimum. There is not "
                "enough support to constrain a drift model over the whole trace."
            )

    for anchor in selected:
        low, high = anchor.index_range
        if transition is not None and low <= transition[1] and high >= transition[0]:
            check.rejections.append(
                f"anchor {anchor.id} ({low}-{high}) intersects the transition region "
                f"({transition[0]}-{transition[1]}). Fitting baseline through the "
                "kinetics is the failure this guard exists to prevent."
            )
        if excluded is not None and np.any(np.asarray(excluded, dtype=bool)[low : high + 1]):
            check.rejections.append(
                f"anchor {anchor.id} ({low}-{high}) contains excised or flagged "
                "samples, which are not baseline by definition."
            )

    # One-sided fits are permitted, but they extrapolate rather than interpolate
    # across the transition, so they are a decision rather than a default.
    if transition is not None and selected:
        before = [a for a in selected if a.index_range[1] < transition[0]]
        after = [a for a in selected if a.index_range[0] > transition[1]]
        if not before or not after:
            side = "before" if before else "after"
            check.confirmations.append(
                f"every anchor lies {side} the transition. There is no lever arm "
                "across it, so the drift is extrapolated rather than interpolated. "
                "Permitted with explicit confirmation, logged as a one-sided fit "
                "with reduced confidence."
            )

    if drift is not None:
        if drift.max_slope_ratio > config.drift_slope_ceiling:
            check.rejections.append(
                f"the fitted drift moves at {100 * drift.max_slope_ratio:.1f} percent "
                f"of the signal's own maximum slope, above the "
                f"{100 * config.drift_slope_ceiling:.0f} percent ceiling. A drift "
                "model that can move as fast as the kinetics will absorb them."
            )
        if abs(drift.fold_change_impact) > config.drift_impact_ceiling:
            check.rejections.append(
                f"applying this correction changes the total fold change by "
                f"{100 * abs(drift.fold_change_impact):.1f} percent, above the "
                f"{100 * config.drift_impact_ceiling:.0f} percent ceiling. A "
                "correction that materially moves the scientific conclusion needs a "
                "human, not a default."
            )

    return check


def fit_drift(
    time_s: np.ndarray,
    signal: np.ndarray,
    selected: list[AnchorCandidate],
    kind: str = "poly",
    config: BaselineConfig | None = None,
) -> DriftModel:
    """Fit a drift curve to the selected anchors. Does not apply it.

    The polynomial order is capped at ``min(2, n_anchors - 1)``: two anchors fix
    a line and no more, and an order that exceeds what the anchors constrain is
    fitting noise between them.

    The two ceiling measurements come back on the model, so the caller can
    decide whether to offer the correction at all before anything is subtracted.
    """
    config = config or BaselineConfig()
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(signal, dtype=float)

    if not selected:
        raise ValueError("cannot fit a drift model with no anchors selected")

    indices = np.concatenate(
        [np.arange(a.index_range[0], a.index_range[1] + 1) for a in selected]
    )
    usable = indices[np.isfinite(values[indices])]
    if usable.size < 2:
        raise ValueError("the selected anchors contain fewer than two finite samples")

    # Centred and scaled, so a second-order fit over a 22-hour axis stays
    # conditioned. Both factors are stored on the model: without them the fit
    # cannot be re-evaluated, and the correction would not be reversible.
    origin = float(time_s[0])
    span = float(time_s[-1] - time_s[0]) or 1.0
    scaled = (time_s - origin) / span

    if kind == "poly":
        order = min(2, len(selected) - 1)
        if order < 1:
            raise ValueError(
                "a polynomial drift model needs at least two anchor regions"
            )
        coefficients = np.polyfit(scaled[usable], values[usable], order)
        baseline = np.polyval(coefficients, scaled)
        model_order: int | None = int(order)
        stored = [float(c) for c in coefficients]
    elif kind == "piecewise_linear":
        midpoints = np.array(
            [(a.index_range[0] + a.index_range[1]) // 2 for a in selected], dtype=int
        )
        levels = np.array([a.mean_level for a in selected], dtype=float)
        order_by_time = np.argsort(scaled[midpoints])
        # np.interp holds the end values flat outside the anchor span, which is
        # the conservative choice: it never extrapolates a slope past the last
        # place the drift was actually observed.
        baseline = np.interp(
            scaled, scaled[midpoints][order_by_time], levels[order_by_time]
        )
        model_order = None
        stored = [float(v) for v in levels[order_by_time]]
    else:
        raise ValueError(
            f"unknown drift model {kind!r}. Only 'poly' and 'piecewise_linear' are "
            "permitted -- splines and higher orders have enough freedom to follow "
            "the transition and would absorb the kinetics."
        )

    # What gets subtracted is the trend, referenced to its value at the start.
    # The fit's constant term is the sample's own scattering level, not drift;
    # taking it out would zero the trace at t=0 and make Mw/M0 meaningless.
    reference_level = float(baseline[0])
    correction = baseline - reference_level

    signal_slope = lag_slope(time_s, values)
    drift_slope = lag_slope(time_s, correction)
    slope_ratio = drift_slope / signal_slope if signal_slope > 0 else float("inf")

    before = total_fold_change(values)
    after = total_fold_change(values - correction)
    if np.isfinite(before) and before != 0 and np.isfinite(after):
        impact = (after - before) / before
    else:
        impact = float("inf")

    return DriftModel(
        kind=kind,
        order=model_order,
        coefficients=stored,
        anchor_ranges=[a.index_range for a in selected],
        time_origin=origin,
        time_scale=span,
        reference_level=reference_level,
        max_slope_ratio=float(slope_ratio),
        fold_change_impact=float(impact),
    )


def evaluate_drift(drift: DriftModel, time_s: np.ndarray) -> np.ndarray:
    """Reconstruct what would be subtracted, from the recorded model alone.

    Returns the trend referenced to its value at the first sample, so it starts
    at zero -- this is the correction, not the fitted curve. Reconstructing from
    the record rather than from a cached array is what makes the correction
    auditable: anyone holding the sidecar can regenerate exactly what was taken
    out.
    """
    time_s = np.asarray(time_s, dtype=float)
    scaled = (time_s - drift.time_origin) / (drift.time_scale or 1.0)

    if drift.kind == "poly":
        curve = np.polyval(np.asarray(drift.coefficients, dtype=float), scaled)
    else:
        midpoints = np.array(
            [(low + high) // 2 for low, high in drift.anchor_ranges], dtype=int
        )
        levels = np.asarray(drift.coefficients, dtype=float)
        order_by_time = np.argsort(scaled[midpoints])
        curve = np.interp(
            scaled, scaled[midpoints][order_by_time], levels[order_by_time]
        )
    return curve - drift.reference_level


def apply_correction(
    signal: np.ndarray, drift: np.ndarray
) -> np.ndarray:
    """Subtract the drift, preserving length, NaNs, and index alignment."""
    values = np.asarray(signal, dtype=float)
    if values.shape != np.shape(drift):
        raise ValueError("signal and drift must have the same shape")
    return values - np.asarray(drift, dtype=float)


def revert_correction(corrected: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """Add the drift back, recovering the uncorrected values exactly."""
    return np.asarray(corrected, dtype=float) + np.asarray(drift, dtype=float)


def _contiguous(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of each contiguous True run."""
    flagged = np.flatnonzero(np.asarray(mask, dtype=bool))
    if flagged.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(flagged) > 1)
    starts = np.concatenate(([flagged[0]], flagged[breaks + 1]))
    ends = np.concatenate((flagged[breaks], [flagged[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


__all__ = [
    "FOLD_CHANGE_POINTS",
    "SelectionCheck",
    "apply_correction",
    "evaluate_drift",
    "fit_drift",
    "local_scatter",
    "propose_anchors",
    "revert_correction",
    "rolling_median",
    "total_fold_change",
    "validate_selection",
]
