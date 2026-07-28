"""Optical constant, Rayleigh ratio, and the instrument scale factor alpha.

The instrument reports scattering in its own internal units, so a conversion
``I_meas = alpha * I_R`` is needed to reach the absolute Rayleigh ratio.

What alpha does and does not affect is narrower than it first appears:

    Kc/I_meas = (1/alpha) * (1/Mw + 2*A2*c + 3*A3*c^2 + ...)

    intercept        = 1 / (alpha * Mw)
    slope            = 2*A2 / alpha
    slope/intercept  = 2 * A2 * Mw          <- alpha cancels

So **A2*Mw is calibration-free**, as is Mw/M0 (a ratio within one file). Only
separating A2 and Mw into absolute units requires alpha. The pipeline can
therefore be built and validated before alpha is measured.

Working assumption: alpha is a fixed scale factor for a given instrument
configuration, not a per-point or per-run normalization. This is assumed, not
verified. It would be falsified by curve shape changing across a gain or
configuration change that the metadata does not record.
"""

from __future__ import annotations

import numpy as np

from polyrmc.config import AVOGADRO, TOLUENE_RAYLEIGH_RATIO, OpticalConfig


def optical_constant(config: OpticalConfig) -> float:
    """The constant K for vertically polarized light.

    .. math:: K = \\frac{4 \\pi^2 n_0^2 (dn/dc)^2}{N_A \\lambda_0^4}

    Units are mol*cm^2/g^2, so that Kc/I_R comes out in mol/g and its
    intercept is 1/Mw.
    """
    wavelength_cm = config.wavelength_nm * 1e-7
    return float(
        4.0
        * np.pi**2
        * config.solvent_refractive_index**2
        * config.dn_dc**2
        / (AVOGADRO * wavelength_cm**4)
    )


def scattering_vector(config: OpticalConfig) -> float:
    """Magnitude of the scattering vector q, in cm^-1.

    .. math:: q = \\frac{4 \\pi n_0}{\\lambda_0} \\sin(\\theta/2)
    """
    wavelength_cm = config.wavelength_nm * 1e-7
    return float(
        4.0
        * np.pi
        * config.solvent_refractive_index
        * np.sin(np.radians(config.angle_deg) / 2.0)
        / wavelength_cm
    )


def alpha_from_toluene(
    toluene_counts: float,
    solvent_counts: float = 0.0,
    wavelength_nm: float = 660.0,
) -> float:
    """Instrument scale factor from a toluene reference measurement.

    Toluene's absolute Rayleigh ratio is known independently, so measuring it
    on this instrument fixes the counts-to-cm^-1 conversion.
    """
    key = int(round(wavelength_nm))
    if key not in TOLUENE_RAYLEIGH_RATIO:
        raise ValueError(
            f"no tabulated toluene Rayleigh ratio at {wavelength_nm} nm; "
            f"available: {sorted(TOLUENE_RAYLEIGH_RATIO)}"
        )
    excess = toluene_counts - solvent_counts
    if excess <= 0:
        raise ValueError("toluene counts must exceed the solvent blank")
    return float(excess / TOLUENE_RAYLEIGH_RATIO[key])


def alpha_from_known_mw(intercept: float, known_mw: float) -> float:
    """Back-solve alpha from a sample of independently known molar mass.

    Since ``intercept = 1/(alpha * Mw)``, a standard with known Mw fixes alpha.
    BSA (66,430 g/mol) and lysozyme (14,300 g/mol) both appear in the
    validation set, so this is an independent cross-check on the toluene route
    -- the two agreeing is itself evidence for the constant-alpha assumption.
    """
    if intercept <= 0:
        raise ValueError("intercept must be positive to back-solve alpha")
    return float(1.0 / (intercept * known_mw))


def excess_scattering(
    measured_counts: np.ndarray, solvent_blank: float
) -> np.ndarray:
    """Subtract the buffer blank to isolate scattering from the solute."""
    excess = np.asarray(measured_counts, dtype=float) - solvent_blank
    return excess


def rayleigh_ratio(
    excess_counts: np.ndarray, alpha: float | None
) -> tuple[np.ndarray, bool]:
    """Convert instrument counts to an absolute Rayleigh ratio if possible.

    Returns the series and whether it is on an absolute scale. When alpha is
    unknown the raw excess counts are returned unchanged: every quantity the
    pipeline validates first -- A2*Mw, Mw/M0, and the fit range itself -- is
    invariant to a constant y-scale.
    """
    excess = np.asarray(excess_counts, dtype=float)
    if alpha is None:
        return excess, False
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return excess / alpha, True
