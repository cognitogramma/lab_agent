"""Kinetic landmarks measured over a lag, not pointwise.

Design section 6.7.1. This is a required implementation note rather than a
preference, and getting it wrong is not subtle: **pointwise gradients at
realistic noise levels are dominated by noise**, whatever percentile is taken.
A trace whose real maximum slope is 1e-5 per second, carrying noise of 0.002 at
1 Hz sampling, has a pointwise gradient of order 0.003 per second almost
everywhere -- two orders of magnitude above the kinetics. Every candidate
smoothing window then looks like it destroys the landmark, every anchor region
looks too steep to be flat, and the safe band comes back empty for reasons that
have nothing to do with the data.

Measuring the same slope across a lag comparable to the kinetic timescale
divides the noise contribution by the lag while leaving the real slope intact,
which is what makes both the smoothing safe band and the transition region
computable at all.

    lag       = max(21, n // 40)
    lag_slope = percentile(|y[i+lag] - y[i]| / (t[i+lag] - t[i]), 99.5)

The 99.5th percentile rather than the maximum, so a single surviving spike does
not define the kinetics.
"""

from __future__ import annotations

import numpy as np

LANDMARK_PERCENTILE = 99.5

# Below this the lag is comparable to the noise correlation length and the
# measurement stops being a lag measurement at all.
MIN_LAG = 21


def lag_for(n_points: int) -> int:
    """Lag in samples for a series of this length."""
    return max(MIN_LAG, int(n_points) // 40)


def lag_slope_profile(
    time_s: np.ndarray, signal: np.ndarray, lag: int | None = None
) -> np.ndarray:
    """Absolute slope across the lag, one value per sample.

    Each value is assigned to the **midpoint** of the span it was measured over,
    so a landmark lands where it actually happened rather than a half-lag early.
    Samples with no complete span, and spans touching a NaN, are NaN here: a
    censored region has no measurable slope and must not be filled with one.
    """
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(signal, dtype=float)
    n = values.size
    lag = lag_for(n) if lag is None else int(lag)

    profile = np.full(n, np.nan, dtype=float)
    if lag < 1 or n <= lag:
        return profile

    rise = np.abs(values[lag:] - values[:-lag])
    run = time_s[lag:] - time_s[:-lag]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(run > 0, rise / run, np.nan)

    midpoints = np.arange(slope.size) + lag // 2
    profile[midpoints] = slope
    return profile


def lag_slope(
    time_s: np.ndarray,
    signal: np.ndarray,
    lag: int | None = None,
    percentile: float = LANDMARK_PERCENTILE,
) -> float:
    """The trace's characteristic maximum slope, in signal units per second."""
    profile = lag_slope_profile(time_s, signal, lag)
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, percentile))


def noise_equivalent_slope(
    time_s: np.ndarray, noise_floor: float, lag: int | None = None
) -> float:
    """The lag-slope pure noise alone would produce.

    A difference of two independent samples each carrying noise sigma has
    standard deviation ``sigma*sqrt(2)``, spread over the lag's time span. This
    is the yardstick a region has to be flat against -- comparing a local slope
    to zero would reject every real region, since nothing measured is exactly
    flat.
    """
    time_s = np.asarray(time_s, dtype=float)
    n = time_s.size
    lag = lag_for(n) if lag is None else int(lag)
    if n <= lag or noise_floor <= 0:
        return 0.0

    span = float(np.median(time_s[lag:] - time_s[:-lag]))
    if span <= 0:
        return 0.0
    return float(np.sqrt(2.0) * noise_floor / span)


def transition_region(
    time_s: np.ndarray,
    signal: np.ndarray,
    edge_fraction: float = 0.20,
    lag: int | None = None,
) -> tuple[int, int] | None:
    """The span containing the kinetics, as an inclusive index range.

    Located at the maximum lag-slope and extended outward until the lag-slope
    falls below ``edge_fraction`` of that maximum.

    This is the single most important guard in baseline correction. Anchors are
    forbidden from intersecting it, which makes it *structurally* impossible to
    lasso the kinetics and subtract them as baseline -- the failure mode that
    produces a smooth, plausible, physically wrong trace that no downstream
    check can detect.

    Returns None when no slope can be measured, which the caller must treat as
    "the guard could not be established" rather than "there is no transition".
    """
    values = np.asarray(signal, dtype=float)
    n = values.size
    lag = lag_for(n) if lag is None else int(lag)

    profile = lag_slope_profile(time_s, values, lag)
    finite = np.isfinite(profile)
    if not finite.any():
        return None

    peak = int(np.nanargmax(profile))
    threshold = float(profile[peak]) * float(edge_fraction)
    if not np.isfinite(threshold) or threshold <= 0:
        return None

    # NaN is not "below threshold" -- it is unknown, and walking through it
    # would let the region swallow a censored span whose slope nobody measured.
    start = peak
    while start > 0 and finite[start - 1] and profile[start - 1] >= threshold:
        start -= 1
    end = peak
    while end < n - 1 and finite[end + 1] and profile[end + 1] >= threshold:
        end += 1

    # Each profile value summarises a span of `lag` samples around its midpoint,
    # so the region it implies is wider than the profile indices alone.
    half = lag // 2
    return (max(0, start - half), min(n - 1, end + half))


__all__ = [
    "LANDMARK_PERCENTILE",
    "MIN_LAG",
    "lag_for",
    "lag_slope",
    "lag_slope_profile",
    "noise_equivalent_slope",
    "transition_region",
]
