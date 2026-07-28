"""Deterministic anomaly detectors and the features a classifier reasons about.

Five independent detectors run over the trace. Each returns a boolean mask;
:func:`detect_anomalies` merges them into regions and computes the numeric
features that a tier-1 classifier is later asked to reason about. The detectors
compute, the classifier judges -- no detector here calls a model.

Derivatives are taken with respect to reconstructed **time**, not row index,
because interior row drops make sampling non-uniform.

.. note::
   The Hampel detector uses a rolling median as a *detector statistic*. That is
   distinct from -- and not in conflict with -- the prohibition on rolling
   median as a *smoother*: nothing here modifies the signal.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from polyrmc.config import DetectorConfig
from polyrmc.state import AnomalyRecord

_MAD_TO_SIGMA = 1.4826


def robust_sigma(values: np.ndarray) -> float:
    """Noise scale from the median absolute deviation, ignoring NaNs."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    mad = np.nanmedian(np.abs(finite - np.nanmedian(finite)))
    return float(_MAD_TO_SIGMA * mad)


def noise_floor(signal: np.ndarray) -> float:
    """Estimate the noise floor from successive differences.

    Differencing removes smooth structure, so the MAD of the difference series
    reflects point-to-point noise rather than real signal amplitude. The
    ``sqrt(2)`` divides out the variance inflation from differencing.
    """
    diffs = np.diff(signal[np.isfinite(signal)])
    if diffs.size == 0:
        return 0.0
    return float(robust_sigma(diffs) / np.sqrt(2.0))


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median with edge clamping."""
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.nanmedian(strides, axis=-1)


def derivative_edges(
    time_s: np.ndarray, signal: np.ndarray, n_sigmas: float = 6.0, window: int = 101
) -> np.ndarray:
    """Flag samples that depart from the straight line through their neighbours.

    This is a second-difference (curvature) test, not a first-derivative one,
    and the distinction matters on this data. A first-derivative threshold
    flags the whole fast-decay portion of an ACD trace, because the real signal
    genuinely changes quickly there -- rapid change is the measurement, not an
    artifact. The curvature statistic is exactly zero for any straight line and
    is unchanged by a smooth trend, so it responds to abrupt departures only.

    Neighbour interpolation is weighted by elapsed time, so non-uniform
    sampling does not register as curvature.

    The threshold uses a **rolling** noise scale, not a global one. Scattering
    noise grows with intensity, and an ACD trace spans more than an order of
    magnitude in signal: on a representative trace the local noise runs from
    about 11 counts at the start to 0.4 at the end. A single global scale sits
    in the middle and flags hundreds of perfectly ordinary samples at the
    high-intensity end.
    """
    if signal.size < 3:
        return np.zeros(signal.size, dtype=bool)

    span = time_s[2:] - time_s[:-2]
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(span > 0, (time_s[1:-1] - time_s[:-2]) / span, 0.5)
    predicted = signal[:-2] + weight * (signal[2:] - signal[:-2])
    residual = signal[1:-1] - predicted

    if residual.size < 3:
        return np.zeros(signal.size, dtype=bool)
    window = min(window if window % 2 else window + 1, (residual.size // 2) * 2 + 1)
    window = max(window, 3)

    deviation = np.abs(residual - _rolling_median(residual, window))
    scale = _MAD_TO_SIGMA * _rolling_median(deviation, window)
    global_scale = robust_sigma(residual)
    if global_scale == 0 and not np.any(scale > 0):
        return np.zeros(signal.size, dtype=bool)
    scale = np.maximum(scale, global_scale * 1e-6)

    flagged = np.zeros(signal.size, dtype=bool)
    with np.errstate(invalid="ignore"):
        flagged[1:-1] = deviation > n_sigmas * scale
    return flagged


def hampel_outliers(
    signal: np.ndarray, window: int = 31, n_sigmas: float = 5.0
) -> np.ndarray:
    """Flag samples lying outside a robust envelope around the local trend.

    The classical Hampel filter measures deviation from a rolling *median*.
    That is trend-biased: on the steep early portion of a decay trace the
    median lags the data, and ordinary points are flagged as outliers. The
    local trend is therefore modelled with a low-order polynomial fit
    (Savitzky-Golay, the same family used elsewhere in this pipeline) and the
    robust MAD envelope is applied to the residual, which is flat by
    construction.
    """
    if signal.size < window:
        window = max(3, (signal.size // 2) * 2 + 1)
    if window % 2 == 0:
        window += 1
    if window <= 2 or signal.size < 5:
        return np.zeros(signal.size, dtype=bool)

    filled = np.array(signal, dtype=float, copy=True)
    missing = ~np.isfinite(filled)
    if missing.all():
        return np.zeros(signal.size, dtype=bool)
    if missing.any():
        indices = np.arange(filled.size)
        filled[missing] = np.interp(indices[missing], indices[~missing], filled[~missing])

    trend = savgol_filter(filled, window_length=window, polyorder=2)
    deviation = np.abs(filled - trend)
    scale = _MAD_TO_SIGMA * _rolling_median(deviation, window)
    scale = np.maximum(scale, robust_sigma(deviation) * 1e-3)

    with np.errstate(invalid="ignore"):
        flagged = deviation > n_sigmas * scale
    flagged[missing] = False
    return flagged


def saturation_runs(signal: np.ndarray, min_run: int = 5) -> np.ndarray:
    """Flag runs of identical values -- a detector pinned at its rail.

    A genuinely noisy detector does not report the same value many times in a
    row, so a long constant run indicates saturation or a stuck reading.
    """
    mask = np.zeros(signal.size, dtype=bool)
    if signal.size < min_run:
        return mask
    same_as_previous = np.isclose(signal[1:], signal[:-1], rtol=0, atol=0)
    start = 0
    while start < same_as_previous.size:
        if not same_as_previous[start]:
            start += 1
            continue
        end = start
        while end < same_as_previous.size and same_as_previous[end]:
            end += 1
        run_length = end - start + 1
        if run_length >= min_run:
            mask[start : end + 1] = True
        start = end + 1
    return mask


def local_variance_anomalies(
    signal: np.ndarray, window: int = 31, ratio_threshold: float = 6.0
) -> np.ndarray:
    """Flag windows whose local *detrended* variance exceeds the noise level.

    The variance is taken over successive differences rather than raw values.
    Raw local variance is dominated by the trend wherever the signal is moving,
    so on a decaying trace it exceeds the noise floor almost everywhere and the
    detector flags the entire fast-decay region. Differencing removes the
    trend, leaving a statistic that responds to genuine excess scatter.

    The reference level is a rolling median of the variance profile itself,
    rather than a single global noise figure. Scattering noise scales with
    intensity, so a global reference is far too low at the bright end of an ACD
    trace and far too high at the dim end.
    """
    if signal.size < window:
        return np.zeros(signal.size, dtype=bool)
    if window % 2 == 0:
        window += 1

    differences = np.diff(signal, prepend=signal[0])
    half = window // 2
    padded = np.pad(differences, half, mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(padded, window)
    local = np.nanvar(strides, axis=-1)

    # Compare against the local variance level, measured over a much longer
    # span so a genuine burst of scatter cannot set its own reference.
    reference_window = min(window * 10 + 1, (signal.size // 2) * 2 + 1)
    reference = _rolling_median(local, max(reference_window, window))
    if not np.any(reference > 0):
        return np.zeros(signal.size, dtype=bool)

    with np.errstate(invalid="ignore"):
        return local > ratio_threshold * reference


def pelt_changepoints(
    signal: np.ndarray,
    penalty: float | None = None,
    min_segment: int = 10,
    max_points: int = 8000,
) -> list[int]:
    """Exact change-point detection in mean by PELT (L2 cost).

    Pruned Exact Linear Time: the same optimum as dynamic programming, with
    candidates pruned once they can no longer be optimal. Returns interior
    change-point indices.

    Parameters
    ----------
    penalty:
        Cost of adding a change point. Defaults to ``2 * sigma^2 * log(n)``,
        the BIC-style choice for L2 cost with an estimated noise scale.
    max_points:
        Series longer than this are decimated before the search and the
        resulting indices are mapped back. A 22-hour acquisition at 1 Hz does
        not carry change points that need single-sample resolution, and the
        search is superlinear in practice, so the full-resolution run on an
        80,000-point trace is not worth its cost.

    Notes
    -----
    This models a **piecewise-constant mean**. On a smoothly varying trace it
    will therefore return many points, approximating the curve with steps --
    correct for the model, but not a list of physical transitions. Callers that
    want abrupt transitions should filter by step magnitude; see
    :func:`polyrmc.tier0.smoothing.find_landmarks`.
    """
    values = np.asarray(signal, dtype=float)
    if values.size == 0:
        return []
    values = np.nan_to_num(values, nan=float(np.nanmedian(values)))

    stride = 1
    if values.size > max_points:
        stride = int(np.ceil(values.size / max_points))
        values = values[::stride]
        min_segment = max(3, min_segment // stride)

    n = values.size
    if n < 2 * min_segment:
        return []

    if penalty is None:
        # The penalty must scale with the data, never with an absolute
        # constant. Kc/I_R is of order 1e-15 in instrument units and 1e-6 after
        # calibration; a fixed floor such as 1e-12 would swamp the segment
        # costs in the first case and be ignored in the second, so the same
        # trace would yield different change points before and after
        # calibration -- silently, and with no error to flag it.
        sigma = noise_floor(values)
        if sigma <= 0:
            spread = float(np.std(values))
            sigma = spread * 1e-6 if spread > 0 else 1.0
        penalty = 2.0 * sigma**2 * np.log(n)

    prefix = np.concatenate([[0.0], np.cumsum(values)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(values**2)])

    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    previous = np.zeros(n + 1, dtype=int)
    active = np.array([0], dtype=np.int64)

    for end in range(min_segment, n + 1):
        eligible = active[end - active >= min_segment]
        if eligible.size == 0:
            continue

        # L2 cost of every candidate segment [start, end), computed at once.
        length = (end - eligible).astype(float)
        total = prefix[end] - prefix[eligible]
        segment = (prefix_sq[end] - prefix_sq[eligible]) - total * total / length

        objective = best[eligible] + segment + penalty
        winner = int(np.argmin(objective))
        best[end] = objective[winner]
        previous[end] = eligible[winner]

        keep = eligible[best[eligible] + segment <= best[end]]
        recent = active[end - active < min_segment]
        active = np.concatenate([keep, recent, np.array([end], dtype=np.int64)])

    points: list[int] = []
    cursor = n
    while cursor > 0:
        start = int(previous[cursor])
        if start > 0:
            points.append(start * stride)
        cursor = start
    return sorted(points)


def step_profile(values: np.ndarray, context: int = 20) -> np.ndarray:
    """Magnitude of the level change across every index, over ``context`` samples.

    Used to tell an abrupt transition from the ordinary sample-to-sample change
    of a smooth trend: on a smooth curve every index has a similar step, so a
    real transition stands out as an outlier in this profile.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    profile = np.zeros(n)
    if n < 2 * context + 1:
        return profile

    prefix = np.concatenate([[0.0], np.cumsum(values)])
    index = np.arange(context, n - context)
    before = (prefix[index] - prefix[index - context]) / context
    after = (prefix[index + context] - prefix[index]) / context
    profile[index] = np.abs(after - before)
    return profile


def mask_to_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Group a boolean mask into inclusive ``(start, end)`` regions."""
    if not mask.any():
        return []
    indices = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate([[indices[0]], indices[breaks + 1]])
    ends = np.concatenate([indices[breaks], [indices[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _longest_constant_run(values: np.ndarray) -> float:
    """Length of the longest run of exactly repeated values."""
    if values.size == 0:
        return 0.0
    same = np.isclose(values[1:], values[:-1], rtol=0, atol=0)
    longest = current = 1
    for repeated in same:
        current = current + 1 if repeated else 1
        longest = max(longest, current)
    return float(longest)


def _local_trend(
    time_s: np.ndarray,
    signal: np.ndarray,
    start: int,
    end: int,
    left: slice,
    right: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straight line fitted to the context either side of a region.

    Fitted excluding the flagged region itself, then extrapolated across it, so
    that "how far does this depart from what the trace was doing" is answered
    without the anomaly influencing the answer.
    """
    context_times = np.concatenate([time_s[left], time_s[right]])
    context_values = np.concatenate([signal[left], signal[right]])
    finite = np.isfinite(context_values)

    region_times = time_s[start : end + 1]
    origin = float(time_s[start])

    if finite.sum() >= 3 and np.ptp(context_times[finite]) > 0:
        coefficients = np.polyfit(
            context_times[finite] - origin, context_values[finite], 1
        )
        evaluate = lambda times: np.polyval(coefficients, times - origin)  # noqa: E731
    else:
        level = float(np.nanmedian(context_values)) if finite.any() else 0.0
        evaluate = lambda times: np.full(np.shape(times), level)  # noqa: E731

    return evaluate(time_s[left]), evaluate(region_times), evaluate(time_s[right])


def region_features(
    time_s: np.ndarray,
    signal: np.ndarray,
    start: int,
    end: int,
    context: int = 25,
) -> dict[str, float]:
    """Compute the numeric features a classifier reasons about.

    The model never sees the 80,000-point series; it sees these numbers. That
    is what keeps "models judge, they never compute" enforceable rather than
    aspirational.
    """
    n = signal.size
    left = slice(max(0, start - context), start)
    right = slice(end + 1, min(n, end + 1 + context))
    region = signal[start : end + 1]

    before = signal[left]
    after = signal[right]
    sigma = noise_floor(signal)

    # Everything below is measured against the local *trend*, fitted to the
    # surrounding context and extrapolated across the region. On an ACD trace
    # the signal is always moving, so comparing raw levels before and after
    # would report a large "shift" for every region and make
    # recovers_to_baseline meaningless.
    trend_before, trend_region, trend_after = _local_trend(
        time_s, signal, start, end, left, right
    )
    before_residual = float(np.nanmedian(before - trend_before)) if before.size else 0.0
    after_residual = float(np.nanmedian(after - trend_after)) if after.size else 0.0
    region_residual = region - trend_region

    duration = float(time_s[end] - time_s[start]) if end > start else 0.0
    amplitude = float(np.nanmax(np.abs(region_residual))) if region.size else 0.0
    unique_fraction = float(np.unique(region).size / region.size) if region.size else 1.0
    level_shift = after_residual - before_residual

    # How many samples actually depart from the trend, as opposed to how many
    # the detectors flagged: a 31-wide rolling window reports a 1-sample spike
    # as a 31-sample region, which would otherwise disqualify it from being
    # recognised as transient.
    core_points = (
        float(np.sum(np.abs(region_residual) > 5.0 * sigma)) if sigma > 0 else 0.0
    )

    return {
        "core_points": core_points,
        # Longest stretch of identical values. Reported in absolute samples as
        # well as a fraction: when detectors merge a stuck run with its noisy
        # edges the fraction is diluted, but the run itself is still the
        # physical signature of a detector pinned at its rail.
        "longest_run_points": _longest_constant_run(region),
        "longest_run_fraction": _longest_constant_run(region) / max(region.size, 1),
        "n_points": float(end - start + 1),
        "duration_s": duration,
        "amplitude": amplitude,
        "amplitude_in_sigma": float(amplitude / sigma) if sigma > 0 else 0.0,
        "level_shift": level_shift,
        "level_shift_in_sigma": float(level_shift / sigma) if sigma > 0 else 0.0,
        "recovers_to_baseline": float(
            abs(level_shift) < 3 * sigma if sigma > 0 else 0.0
        ),
        "unique_value_fraction": unique_fraction,
        "region_variance_ratio": (
            float(np.nanvar(region_residual) / sigma**2) if sigma > 0 else 0.0
        ),
    }


def detect_anomalies(
    time_s: np.ndarray,
    signal: np.ndarray,
    channel: str,
    config: DetectorConfig | None = None,
) -> tuple[list[AnomalyRecord], list[int]]:
    """Run every detector and return unclassified regions plus change points.

    The returned records carry ``detected_by`` and ``features`` but leave
    ``anomaly_type`` unset -- assigning a type is a judgment call, made either
    by :func:`classify_deterministic` or by a tier-1 classifier.
    """
    config = config or DetectorConfig()

    masks = {
        "derivative_edge": derivative_edges(
            time_s, signal, n_sigmas=config.derivative_n_sigmas
        ),
        "hampel": hampel_outliers(
            signal, window=config.hampel_window, n_sigmas=config.hampel_n_sigmas
        ),
        "saturation_run": saturation_runs(signal, min_run=config.saturation_min_run),
        "local_variance": local_variance_anomalies(
            signal,
            window=config.hampel_window,
            ratio_threshold=config.variance_ratio_threshold,
        ),
    }
    change_points = pelt_changepoints(signal, penalty=config.changepoint_penalty)

    combined = np.zeros(signal.size, dtype=bool)
    for mask in masks.values():
        combined |= mask

    records: list[AnomalyRecord] = []
    for start, end in mask_to_regions(combined):
        fired = [name for name, mask in masks.items() if mask[start : end + 1].any()]
        records.append(
            AnomalyRecord(
                start_index=int(start),
                end_index=int(end),
                channel=channel,
                detected_by=fired,
                features=region_features(time_s, signal, start, end),
            )
        )
    return records, change_points
