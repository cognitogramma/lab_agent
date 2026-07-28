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
    if reference == 0:
        raise ValueError("initial scattering level is zero; cannot normalize")
    return signal / reference
