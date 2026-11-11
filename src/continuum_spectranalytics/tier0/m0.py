"""The M0 anchor: what every value in Mw/M0 is divided by.

Design section 6.8. M0 divides the entire aggregation-index column, so an error
here is a multiplicative scale error across every value rather than random
noise. That is why it cannot be left as a taste preference.

**Rejected as the default -- the first point.** A single sample carries the full
noise of one measurement; at the reference dataset's noise floor that is about 2
percent relative, propagated as a systematic scale bias into every value in the
column. It is also maximally exposed to startup transients, which is exactly
what the opening seconds of an acquisition contain.

**Adopted -- an adaptive initial window, bounded by the lag phase.** The largest
initial span that lies inside a genuinely flat region, does not reach into the
transition, and is no longer than a capped fraction of the run. Averaging N
samples reduces the noise on M0 as sqrt(N): a 60-sample window is roughly eight
times better than one point.

**Where no flat initial region exists, that is a finding, not a parameter.** It
means the trace was already changing at t=0 -- acquisition began after the
process started -- so M0 is not a baseline at all but the state of the sample
when recording happened to begin. The run falls back to the first point, says so
in plain language, and the reason is recorded. It is never a silent fallback.

**Sensitivity is reported every run.** The fold change is computed under both
definitions and the divergence reported. Where they agree the choice did not
matter; where they diverge the run is telling you something. That converts an
unresolvable preference into a measured quantity.

.. note::
   On the default tolerances these two mechanisms overlap. The flatness gate
   that admits a window is tighter than the sensitivity ceiling that would flag
   one, so a disturbance large enough to move the fold change by 5 percent has
   already disqualified the window and is reported through the fallback
   instead. The sensitivity report is therefore a second line of defence rather
   than the first. Both tolerances want calibrating against the golden runs
   before this step is relied on in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_spectranalytics.config import BaselineConfig
from continuum_spectranalytics.tier0.anchors import local_scatter, rolling_median
from continuum_spectranalytics.tier0.landmarks import (
    lag_for,
    lag_slope_profile,
    noise_equivalent_slope,
    transition_region,
)


@dataclass
class M0Result:
    """The chosen reference level and how it was arrived at."""

    value: float
    method: str
    """``adaptive_window`` or ``first_point``."""

    index_range: tuple[int, int]
    """Inclusive span averaged. A single point has start == end."""

    n_points: int
    first_point_value: float
    """The same reference under the rejected definition, for comparison."""

    sensitivity: float
    """Relative divergence in total fold change between the two definitions."""

    blocked: list[str] = field(default_factory=list)
    """Reasons recorded when the adaptive window could not be used."""

    escalations: list[str] = field(default_factory=list)

    @property
    def adaptive(self) -> bool:
        return self.method == "adaptive_window"


def select_m0(
    time_s: np.ndarray,
    signal: np.ndarray,
    noise_floor: float,
    max_window_fraction: float = 0.05,
    sensitivity_ceiling: float = 0.05,
    config: BaselineConfig | None = None,
    excluded: np.ndarray | None = None,
    transition: tuple[int, int] | None = None,
) -> M0Result:
    """Choose the M0 reference window and measure how much the choice matters.

    Parameters
    ----------
    max_window_fraction:
        Cap on the window as a fraction of the run. The window is a baseline
        estimate, not a summary of the early kinetics.
    sensitivity_ceiling:
        Divergence in total fold change between the adaptive and first-point
        definitions above which the run escalates.
    excluded:
        Samples that may not enter the window -- excised, censored, or flagged.
    """
    config = config or BaselineConfig()
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(signal, dtype=float)

    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        raise ValueError("signal has no finite samples, so no M0 can be chosen")
    first_index = int(finite[0])
    first_point = float(values[first_index])

    if transition is None:
        transition = transition_region(
            time_s, values, edge_fraction=config.transition_edge_fraction
        )

    window = _initial_flat_window(
        time_s,
        values,
        noise_floor=noise_floor,
        max_window_fraction=max_window_fraction,
        config=config,
        excluded=excluded,
        transition=transition,
    )

    blocked: list[str] = []
    if window is None:
        blocked.append(
            "M0 fell back to the first point: no flat initial region exists. The "
            "trace is already changing at t=0, which means acquisition began after "
            "the process had started -- M0 is not a baseline here but the state of "
            "the sample when recording began. This is a finding about the "
            "acquisition, not a parameter choice."
        )
        start, end = first_index, first_index
        method = "first_point"
        value = first_point
    else:
        start, end = window
        chunk = values[start : end + 1]
        value = float(np.mean(chunk[np.isfinite(chunk)]))
        method = "adaptive_window"

    sensitivity = _fold_change_divergence(values, value, first_point)
    escalations: list[str] = []
    if np.isfinite(sensitivity) and abs(sensitivity) > sensitivity_ceiling:
        escalations.append(
            f"the two M0 definitions disagree by {100 * abs(sensitivity):.1f} percent "
            f"in total fold change, above the {100 * sensitivity_ceiling:.0f} percent "
            "ceiling. Where they agree the choice does not matter; that they diverge "
            "here means the opening of this trace is not a stable baseline."
        )

    return M0Result(
        value=value,
        method=method,
        index_range=(start, end),
        n_points=end - start + 1,
        first_point_value=first_point,
        sensitivity=float(sensitivity),
        blocked=blocked,
        escalations=escalations,
    )


def _initial_flat_window(
    time_s: np.ndarray,
    values: np.ndarray,
    noise_floor: float,
    max_window_fraction: float,
    config: BaselineConfig,
    excluded: np.ndarray | None,
    transition: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """The largest initial span that is flat at the noise floor.

    Built from the same eligibility test that generates baseline anchors, but
    restricted to the start of the run -- a flat stretch in the middle is not a
    starting level, however flat it is.
    """
    n = values.size
    lag = lag_for(n)
    if n <= lag + 1 or noise_floor <= 0:
        return None

    slope_profile = rolling_median(lag_slope_profile(time_s, values, lag), lag)
    scatter_profile = local_scatter(values, lag)
    slope_ceiling = config.anchor_flatness_tolerance * noise_equivalent_slope(
        time_s, noise_floor, lag
    )

    eligible = (
        np.isfinite(values)
        & np.isfinite(slope_profile)
        & (slope_profile <= slope_ceiling)
        & np.isfinite(scatter_profile)
        & (scatter_profile <= config.anchor_variance_factor * noise_floor)
    )
    if excluded is not None:
        eligible &= ~np.asarray(excluded, dtype=bool)

    cap = max(2, int(round(max_window_fraction * n)))
    limit = cap
    if transition is not None:
        # Stop short of the transition. The lag phase bounds the window; a
        # window that reaches into the kinetics is measuring the rise, not the
        # level it started from.
        limit = min(limit, max(0, transition[0]))
    if limit < 2:
        return None

    # Walk forward from the first sample. The window has to *start* at the
    # beginning of the run, so the first ineligible sample ends it.
    end = -1
    for i in range(min(limit, n)):
        if not eligible[i]:
            break
        end = i
    if end < 1:
        return None

    # The design's explicit gate: the slope across the chosen window must be
    # indistinguishable from zero at the noise floor.
    if not _slope_is_zero_at_noise(time_s[: end + 1], values[: end + 1], noise_floor):
        return None
    return (0, end)


def _slope_is_zero_at_noise(
    time_s: np.ndarray, window: np.ndarray, noise_floor: float
) -> bool:
    """Is the trend across this window smaller than its own uncertainty?

    Compares the fitted slope against the standard error of that slope under
    pure noise. Testing against zero would reject every real window, since
    nothing measured is exactly flat.
    """
    usable = np.isfinite(window)
    if usable.sum() < 3:
        return False
    t = np.asarray(time_s, dtype=float)[usable]
    y = np.asarray(window, dtype=float)[usable]

    centred = t - t.mean()
    denominator = float(np.sum(centred**2))
    if denominator <= 0:
        return False
    slope = float(np.sum(centred * (y - y.mean())) / denominator)
    standard_error = noise_floor / np.sqrt(denominator)
    if standard_error <= 0:
        return False
    return abs(slope) <= 2.0 * standard_error


def _fold_change_divergence(
    values: np.ndarray, adaptive: float, first_point: float
) -> float:
    """Relative difference in total fold change between the two definitions."""
    finite = values[np.isfinite(values)]
    if finite.size < 2 or adaptive == 0 or first_point == 0:
        return float("nan")
    take = min(50, finite.size // 2) or 1
    end = float(np.median(finite[-take:]))
    by_adaptive = end / adaptive
    by_first = end / first_point
    if by_first == 0:
        return float("nan")
    return (by_adaptive - by_first) / by_first


__all__ = ["M0Result", "select_m0"]
