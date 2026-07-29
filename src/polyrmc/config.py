"""Run configuration and physical constants.

Everything that varies per experiment lives in :class:`RunConfig`. Nothing in
:mod:`polyrmc.tier0` reads global state -- config is passed in explicitly so a
run is reproducible from its recorded configuration alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"

AVOGADRO = 6.02214076e23  # mol^-1, exact by SI definition

# Rayleigh ratio of toluene, vertically polarized, 90 deg. Wavelength-dependent;
# the 660 nm value is the one relevant to the ARGEN. Units: cm^-1.
TOLUENE_RAYLEIGH_RATIO = {
    633: 1.087e-5,
    660: 9.44e-6,
}


class OpticalConfig(BaseModel):
    """Optical constants for the scattering calculation."""

    wavelength_nm: float = Field(660.0, gt=0, description="Vacuum wavelength lambda_0.")
    angle_deg: float = Field(90.0, gt=0, lt=180, description="Detection angle theta.")
    solvent_refractive_index: float = Field(1.333, gt=0, description="n_0.")
    dn_dc: float = Field(..., description="Refractive index increment, cm^3/g.")

    # Absolute scale factor: I_meas = alpha * I_R. None until the toluene
    # reference is measured on this instrument. A2*Mw and Mw/M0 do not need it.
    alpha: float | None = Field(
        None, gt=0, description="Instrument scale factor; None if uncalibrated."
    )


class DilutionConfig(BaseModel):
    """Automatic continuous dilution trajectory: c(t) = c0 * exp(-Q t / V)."""

    c0_g_per_cm3: float = Field(..., gt=0, description="Initial concentration.")
    flow_rate_cm3_per_s: float = Field(..., gt=0, description="Solvent inflow rate Q.")
    cell_volume_cm3: float = Field(1.0, gt=0, description="Fixed cuvette volume V.")


class SmoothingConfig(BaseModel):
    """Adaptive Savitzky-Golay settings.

    Polynomial order is pinned low deliberately: orders above 3 reproduce noise
    as false structure. Rolling-median and moving-average smoothers are
    prohibited outright -- they attenuate the real transitions the experiment
    exists to measure -- and are therefore not implemented anywhere.
    """

    polyorder: int = Field(2, ge=1, le=3)
    min_window: int = Field(5, ge=3, description="Shortest candidate window (odd).")
    max_window: int = Field(301, ge=5, description="Longest candidate window (odd).")
    n_candidates: int = Field(12, ge=2, description="Candidate windows to propose.")
    landmark_reference_window: int = Field(
        11,
        ge=3,
        description="Mild smoothing applied before landmark detection, so that "
        "noise excursions are not mistaken for kinetic features.",
    )
    max_landmark_attenuation: float = Field(
        0.30,
        gt=0,
        lt=1,
        description="Largest tolerated prominence loss at a landmark. Not zero: "
        "removing the noise riding on a peak lowers its measured prominence by "
        "roughly 0.1 even at the shortest window, so a tighter bound would admit "
        "no window at all.",
    )
    max_residual_autocorr: float = Field(
        0.2,
        gt=0,
        lt=1,
        description="Largest *positive* lag-1 autocorrelation of the residual. "
        "Negative values are expected and are not penalized -- see "
        "polyrmc.tier0.smoothing.residual_autocorrelation.",
    )

    @model_validator(mode="after")
    def _check_windows(self) -> SmoothingConfig:
        if self.max_window <= self.min_window:
            raise ValueError("max_window must exceed min_window")
        if self.polyorder >= self.min_window:
            raise ValueError("min_window must exceed polyorder")
        return self


class DetectorConfig(BaseModel):
    """Thresholds for the deterministic anomaly detectors."""

    hampel_window: int = Field(31, ge=3, description="Rolling MAD window (odd).")
    hampel_n_sigmas: float = Field(5.0, gt=0)
    derivative_n_sigmas: float = Field(6.0, gt=0)
    saturation_min_run: int = Field(5, ge=2, description="Repeats that count as stuck.")
    variance_ratio_threshold: float = Field(6.0, gt=1)
    changepoint_penalty: float | None = Field(
        None, description="PELT penalty; None selects a BIC-style default."
    )


class FitRangeConfig(BaseModel):
    """Candidate generation for the linear region of Kc/I_R vs c."""

    min_points: int = Field(20, ge=5, description="Fewest points in a usable range.")
    n_candidates: int = Field(10, ge=2)
    max_residual_autocorr: float = Field(0.25, gt=0, lt=1)
    min_r_squared: float = Field(0.95, gt=0, lt=1)


PLACEHOLDER_MODEL = "gemini-SET-A-CURRENT-MODEL-ID"
"""Deliberately invalid default, so an unpinned judge fails before it spends a run.

Google rotates Gemini model identifiers, and a stale one would surface as an API
error only after the file was ingested and its candidates generated. Read the
current id from Google's model documentation and set it in the run config.
"""


class LoopConfig(BaseModel):
    """Bounds on a tier-1 propose -> judge -> re-propose loop."""

    max_iterations: int = Field(3, ge=1, description="Hard cap; then fall back.")
    model: str = Field(PLACEHOLDER_MODEL, description="Pinned judge model id.")
    max_tokens: int = Field(4000, ge=256)
    replay: bool = Field(
        False, description="Reuse recorded decisions instead of calling the model."
    )


class RunConfig(BaseModel):
    """Everything needed to reproduce one processing run."""

    run_id: str
    source_file: Path

    # Supplied, never inferred: misclassifying a run makes every downstream
    # number wrong in a way that looks entirely normal.
    experiment_type: Literal["dilution_trajectory", "fixed_concentration"]

    # The instrument writes ls1..ls16; only ls8 is the measurement. The others
    # are dropped at load and never reach any analysis stage.
    channel: str = Field("ls8", description="Scattering channel to analyse.")

    optical: OpticalConfig
    dilution: DilutionConfig | None = None
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    detectors: DetectorConfig = Field(default_factory=DetectorConfig)
    fit_range: FitRangeConfig = Field(default_factory=FitRangeConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)

    solvent_blank_counts: float = Field(
        0.0, ge=0, description="Buffer blank subtracted from the raw channel."
    )
    k_h: float = Field(
        0.0, description="Hydrodynamic term in k_D = 2*A2Mw + k_H."
    )

    @model_validator(mode="after")
    def _dilution_required(self) -> RunConfig:
        if self.experiment_type == "dilution_trajectory" and self.dilution is None:
            raise ValueError("dilution_trajectory runs require a DilutionConfig")
        return self
