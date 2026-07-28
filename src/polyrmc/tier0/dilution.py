"""The automatic continuous dilution trajectory.

Solvent enters at rate Q while liquid is withdrawn at the same rate, so the
cuvette volume V stays fixed and concentration decays exponentially:

.. math:: c(t) = c_0 \\, e^{-Qt/V}

This converts the time axis into a concentration axis, which is what turns a
single ~1 hour experiment into the equivalent of a dilution series.
"""

from __future__ import annotations

import numpy as np

from polyrmc.config import DilutionConfig


def concentration_at(time_s: np.ndarray, config: DilutionConfig) -> np.ndarray:
    """Concentration in g/cm^3 at each elapsed time.

    Time must be measured from the start of dilution, not from the start of
    acquisition, if the two differ.
    """
    time_s = np.asarray(time_s, dtype=float)
    decay = config.flow_rate_cm3_per_s / config.cell_volume_cm3
    return config.c0_g_per_cm3 * np.exp(-decay * time_s)


def time_for_concentration(concentration: float, config: DilutionConfig) -> float:
    """Elapsed time at which the trajectory reaches a given concentration."""
    if concentration <= 0:
        raise ValueError("concentration must be positive")
    if concentration > config.c0_g_per_cm3:
        raise ValueError("concentration exceeds the initial value c0")
    decay = config.flow_rate_cm3_per_s / config.cell_volume_cm3
    return float(-np.log(concentration / config.c0_g_per_cm3) / decay)


def dilution_time_constant(config: DilutionConfig) -> float:
    """The V/Q time constant of the exponential decay, in seconds."""
    return config.cell_volume_cm3 / config.flow_rate_cm3_per_s


def plateau_swing(signal: np.ndarray, fraction: float = 0.5, n_bins: int = 10) -> float:
    """How much the trace still swings over its final ``fraction`` of the run.

    Returned as the peak-to-peak range of binned medians divided by their
    median, so 0 means a flat plateau and 1 means the trace still moves by its
    own magnitude.

    This exists because an endpoint summary is only meaningful if the trace
    settled. A run dominated by sporadic large aggregates drifting through the
    scattering volume never plateaus -- its final value depends on where the
    acquisition happened to stop, and quoting it as Mw/M0 states far more
    confidence than the data supports.
    """
    finite = signal[np.isfinite(signal)]
    if finite.size < n_bins * 2:
        return 0.0

    tail = finite[int(finite.size * (1.0 - fraction)) :]
    if tail.size < n_bins * 2:
        return 0.0

    bins = np.array_split(tail, n_bins)
    levels = np.array([np.median(b) for b in bins if b.size])
    centre = float(np.median(levels))
    if centre == 0:
        return float("inf")
    return float((levels.max() - levels.min()) / abs(centre))


def aggregation_index(signal: np.ndarray, reference_points: int = 50) -> np.ndarray:
    """Mw/M0: scattering normalized to its own initial value.

    Taken within a single file, so the instrument scale factor alpha cancels
    and no calibration is required. Only meaningful for fixed-concentration
    aggregation runs, where a rise means mass is accumulating rather than
    concentration changing.
    """
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        raise ValueError("signal has no finite samples")
    reference = float(np.nanmedian(finite[: min(reference_points, finite.size)]))
    if reference <= 0:
        # A non-positive reference means the blank equals or exceeds the sample
        # at this channel, so there is no solute signal to normalise against.
        # Dividing anyway yields a large negative ratio that looks like a
        # number and means nothing -- on a real run the dim channels produced
        # Mw/M0 of -4 and -24 this way.
        raise ValueError(
            f"reference excess scattering is {reference:.6g}, not positive: the "
            "blank meets or exceeds the sample on this channel, so Mw/M0 is "
            "undefined"
        )
    return signal / reference
