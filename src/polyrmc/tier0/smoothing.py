"""Adaptive Savitzky-Golay smoothing and safe-by-construction window candidates.

Savitzky-Golay only. Rolling-median and moving-average smoothers are prohibited
and deliberately absent from this module: they attenuate the real transitions
the experiment exists to measure. Polynomial order is pinned low (2, ceiling 3)
because higher orders reproduce noise as false structure.

Window length is selected per file, never fixed. :func:`propose_windows`
generates the candidate set -- every candidate both whitens the residual and
preserves the kinetic landmarks, so any of them is defensible. A tier-1 judge
picks among them; it can never construct one.

The smoothed trace is a persisted data product and the substrate for every
part-2 calculation, not a display nicety. That is why landmark preservation is
a hard gate rather than a preference.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_prominences, savgol_filter

from polyrmc.config import SmoothingConfig
from polyrmc.state import Candidate
from polyrmc.tier0.detect import noise_floor, pelt_changepoints, step_profile

MAX_POLYORDER = 3


def savgol_smooth(signal: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay filter that tolerates NaN gaps.

    NaNs are interpolated for the filter pass and restored afterward, so
    excised regions never acquire fabricated values.
    """
    if polyorder > MAX_POLYORDER:
        raise ValueError(
            f"polyorder {polyorder} exceeds the ceiling of {MAX_POLYORDER}: higher "
            "orders reproduce noise as false structure"
        )
    if window % 2 == 0:
        raise ValueError("Savitzky-Golay window length must be odd")
    if window <= polyorder:
        raise ValueError("window length must exceed the polynomial order")
    if window > signal.size:
        raise ValueError("window length exceeds the series length")

    missing = ~np.isfinite(signal)
    if missing.all():
        raise ValueError("signal has no finite samples")

    working = np.array(signal, dtype=float, copy=True)
    if missing.any():
        indices = np.arange(signal.size)
        working[missing] = np.interp(indices[missing], indices[~missing], signal[~missing])

    smoothed = savgol_filter(working, window_length=window, polyorder=polyorder)
    smoothed[missing] = np.nan
    return smoothed


def residual_autocorrelation(signal: np.ndarray, smoothed: np.ndarray, lag: int = 1) -> float:
    """Lag-1 autocorrelation of the smoothing residual.

    Interpretation is **one-sided**, which matters for how it is used as a gate.
    Subtracting a local fit from white noise leaves a residual that is
    negatively autocorrelated -- around -0.8 at a 5-point window, approaching
    zero as the window widens. Negative values are therefore the signature of a
    filter that removed noise and nothing else, not a defect.

    Positive autocorrelation is the diagnostic that matters: it means real
    structure was smoothed away and is now sitting in the residual.
    """
    residual = signal - smoothed
    residual = residual[np.isfinite(residual)]
    if residual.size <= lag + 1:
        return 0.0
    centered = residual - residual.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return 0.0
    numerator = float(np.dot(centered[:-lag], centered[lag:]))
    return numerator / denominator


def find_landmarks(
    signal: np.ndarray,
    prominence_sigmas: float = 5.0,
    step_outlier_sigmas: float = 6.0,
    context: int = 20,
    min_landmark_width: float = 5.0,
    reference_window: int = 11,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Locate the kinetic features smoothing must not erase.

    Returns prominent extrema and *abrupt* transitions.

    Detection runs on a lightly pre-smoothed copy of the trace, not on the raw
    samples. A noise excursion large enough to clear the prominence and width
    thresholds would otherwise be treated as a landmark that every real window
    destroys, and the gate would reject all of them -- which is what happens on
    a baseline-corrected trace, where what remains is close to pure noise. A
    feature that does not survive the mildest smoothing the pipeline would ever
    apply is not a feature the pipeline is committed to preserving.

    The change-point filter is the important part. PELT models a
    piecewise-constant mean, so on a smooth exponential decay it returns many
    points -- it is approximating a curve with steps. Those are not physical
    transitions, and treating them as landmarks makes every window look like it
    destroys the signal. A genuine transition is instead an **outlier** in the
    step profile: a level change much larger than the same span produces
    elsewhere on the trace. Smooth trends produce a uniform profile and so
    contribute no landmarks, which is the desired behavior.
    """
    if not np.isfinite(signal).any():
        return np.array([], dtype=int), []
    finite = reference_trace(signal, reference_window)

    # The noise scale must come from the RAW trace. Estimating it from the
    # pre-smoothed reference understates it badly -- smoothing is precisely
    # what removes the point-to-point variation that noise_floor measures --
    # which collapses the prominence threshold and turns thousands of noise
    # wiggles into "landmarks". On a real 79k-point trace that produced 5475
    # extrema and made every candidate window look destructive.
    sigma = noise_floor(_fill(signal))
    prominence = prominence_sigmas * sigma if sigma > 0 else None

    # A minimum width is what separates a kinetic feature from a noise spike.
    # Without it, a 1-sample excursion counts as a landmark, and demanding that
    # smoothing preserve it is incoherent -- removing it is the filter's job.
    peaks, _ = find_peaks(finite, prominence=prominence, width=min_landmark_width)
    troughs, _ = find_peaks(-finite, prominence=prominence, width=min_landmark_width)
    extrema = np.sort(np.concatenate([peaks, troughs])).astype(int)

    profile = step_profile(finite, context=context)
    interior = profile[profile > 0]
    if interior.size == 0:
        return extrema, []
    median = float(np.median(interior))
    scale = float(1.4826 * np.median(np.abs(interior - median)))
    threshold = median + step_outlier_sigmas * scale if scale > 0 else np.inf

    abrupt = [
        point
        for point in pelt_changepoints(finite)
        if 0 <= point < profile.size and profile[point] > threshold
    ]

    if exclude is not None and exclude.any():
        # A region the detectors flagged is not a kinetic feature the filter is
        # obliged to preserve -- we already suspect it. Without this, an
        # unclassified region that splicing conservatively left in place goes
        # on to veto every smoothing window, so being careful at one stage
        # makes the next stage impossible.
        extrema = np.array(
            [index for index in extrema if not exclude[index]], dtype=int
        )
        abrupt = [point for point in abrupt if not exclude[point]]

    return extrema, abrupt


def reference_trace(
    signal: np.ndarray, reference_window: int = 11, polyorder: int = 2
) -> np.ndarray:
    """The trace against which landmark preservation is measured.

    This is the signal after the mildest smoothing the pipeline would ever
    apply. Comparing candidate windows against it -- rather than against the
    raw samples -- means the metric reports how much *additional* structure a
    window destroys, and stops charging every window for the noise that
    smoothing is supposed to remove.
    """
    filled = _fill(signal)
    window = min(reference_window, filled.size if filled.size % 2 else filled.size - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return filled
    return savgol_filter(filled, window_length=window, polyorder=polyorder)


def _fill(values: np.ndarray) -> np.ndarray:
    """Interpolate NaNs so scipy's peak routines can run."""
    filled = np.array(values, dtype=float, copy=True)
    missing = ~np.isfinite(filled)
    if missing.any() and not missing.all():
        indices = np.arange(filled.size)
        filled[missing] = np.interp(indices[missing], indices[~missing], filled[~missing])
    return filled


def _classify_extrema(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """True where an index is a local maximum of ``values``."""
    is_maximum = np.zeros(indices.size, dtype=bool)
    interior = (indices > 0) & (indices < values.size - 1)
    inner = indices[interior]
    is_maximum[interior] = (values[inner] >= values[inner - 1]) & (
        values[inner] >= values[inner + 1]
    )
    return is_maximum


def _prominence_at(
    values: np.ndarray,
    indices: np.ndarray,
    is_maximum: np.ndarray,
    search_radius: int = 0,
) -> np.ndarray:
    """Topographic prominence of each landmark, matched within a search radius.

    Prominence is the right measure of "how much does this feature stand out",
    because it is evaluated over the feature's own span. Measuring height
    against a fixed-width neighborhood instead understates a broad peak badly:
    a +-20 sample window sitting inside a 70-sample-wide peak reports a small
    fraction of the real amplitude.

    Two details make this comparable between the raw and smoothed traces:

    * the maximum/minimum call is supplied rather than re-derived, so a
      flattened shoulder is not silently measured with the opposite sign;
    * the landmark is matched to the nearest extremum of the same type within
      ``search_radius``, because smoothing shifts features slightly. A feature
      that genuinely vanished has no match and correctly scores zero.
    """
    if indices.size == 0:
        return np.array([])

    result = np.zeros(indices.size)
    for sign, mask in ((1.0, is_maximum), (-1.0, ~is_maximum)):
        if not mask.any():
            continue
        oriented = sign * values
        extrema, _ = find_peaks(oriented)
        if extrema.size == 0:
            continue
        prominences = peak_prominences(oriented, extrema)[0]
        for position in np.flatnonzero(mask):
            distance = np.abs(extrema - indices[position])
            nearby = distance <= max(search_radius, 1)
            if nearby.any():
                result[position] = float(prominences[nearby].max())
    return result


def landmark_attenuation(
    signal: np.ndarray,
    smoothed: np.ndarray,
    extrema: np.ndarray,
    change_points: list[int],
    context: int = 20,
    search_radius: int = 0,
) -> float:
    """Largest fractional amplitude loss at any landmark.

    For extrema this is the loss of topographic prominence; for abrupt
    transitions it is the shrinkage of the level step. Zero means every
    landmark survived at full amplitude.

    ``search_radius`` should be about half the smoothing window, so a feature
    that merely shifted is not scored as a feature that was destroyed.
    """
    raw = _fill(signal)
    filtered = _fill(smoothed)
    losses: list[float] = []
    n = raw.size

    extrema = np.asarray(extrema, dtype=int)
    if extrema.size:
        is_maximum = _classify_extrema(raw, extrema)
        raw_prominence = _prominence_at(raw, extrema, is_maximum, search_radius=1)
        smoothed_prominence = _prominence_at(
            filtered, extrema, is_maximum, search_radius=max(search_radius, context)
        )
        for before, after in zip(raw_prominence, smoothed_prominence):
            if before > 0:
                losses.append(1.0 - after / before)

    for index in change_points:
        before_slice = slice(max(0, index - context), index)
        after_slice = slice(index, min(n, index + context))
        if raw[before_slice].size == 0 or raw[after_slice].size == 0:
            continue
        raw_step = abs(np.median(raw[after_slice]) - np.median(raw[before_slice]))
        smoothed_step = abs(
            np.median(filtered[after_slice]) - np.median(filtered[before_slice])
        )
        if raw_step > 0:
            losses.append(1.0 - smoothed_step / raw_step)

    if not losses:
        return 0.0
    return float(max(0.0, max(losses)))


def candidate_windows(n_points: int, config: SmoothingConfig) -> list[int]:
    """Odd window lengths spanning the configured range, log-spaced."""
    upper = min(config.max_window, n_points if n_points % 2 else n_points - 1)
    if upper <= config.min_window:
        return [config.min_window] if config.min_window < n_points else []
    raw = np.unique(
        np.geomspace(config.min_window, upper, num=config.n_candidates).astype(int)
    )
    windows = sorted({int(w) if w % 2 else int(w) + 1 for w in raw})
    return [w for w in windows if config.polyorder < w <= upper]


def conservative_window(n_points: int, config: SmoothingConfig) -> int:
    """The safest window: the shortest one, i.e. the least smoothing applied.

    Used when a bounded loop exhausts its iterations. Under-smoothing costs
    precision; over-smoothing destroys the transitions being measured, so the
    conservative direction is unambiguous.
    """
    windows = candidate_windows(n_points, config)
    if not windows:
        raise ValueError("no valid smoothing window fits this series")
    return windows[0]


def propose_windows(
    signal: np.ndarray,
    config: SmoothingConfig | None = None,
    exclude: np.ndarray | None = None,
) -> list[Candidate]:
    """Generate smoothing windows that are safe by construction.

    A window is offered only if it both whitens the residual and preserves
    every landmark within tolerance. The judge therefore chooses among options
    that are all defensible, which is what keeps a bad model response bounded:
    the worst case is a suboptimal window inside an already-validated band.
    """
    config = config or SmoothingConfig()
    extrema, change_points = find_landmarks(
        signal,
        reference_window=config.landmark_reference_window,
        exclude=exclude,
    )
    # Landmark amplitudes are compared against a lightly smoothed reference,
    # not the raw trace. Measuring against raw charges every window for the
    # noise it was supposed to remove, and on real traces nothing then passes.
    # The cost is a known bias: a window equal to the reference scores zero
    # attenuation by construction, so it is favoured, and on noisy traces the
    # candidate set can collapse to that single window -- which leaves a judge
    # nothing to decide. Widening this is the main open issue in tier 1.
    reference = reference_trace(
        signal, config.landmark_reference_window, config.polyorder
    )

    candidates: list[Candidate] = []
    for window in candidate_windows(signal.size, config):
        try:
            smoothed = savgol_smooth(signal, window, config.polyorder)
        except ValueError:
            continue

        autocorr = residual_autocorrelation(signal, smoothed)
        attenuation = landmark_attenuation(
            reference, smoothed, extrema, change_points, search_radius=window // 2
        )

        # One-sided by design: negative autocorrelation means noise was removed
        # cleanly, positive means real structure ended up in the residual.
        if autocorr > config.max_residual_autocorr:
            continue
        if attenuation > config.max_landmark_attenuation:
            continue

        candidates.append(
            Candidate(
                index=len(candidates),
                value=window,
                label=f"window={window}, polyorder={config.polyorder}",
                features={
                    "window": float(window),
                    "residual_autocorr": autocorr,
                    "landmark_attenuation": attenuation,
                    "residual_std": float(np.nanstd(signal - smoothed)),
                    "n_landmarks": float(len(extrema) + len(change_points)),
                },
            )
        )
    return candidates
