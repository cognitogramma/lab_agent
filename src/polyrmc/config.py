"""Run configuration and physical constants.

Everything that varies per experiment lives in :class:`RunConfig`. Nothing in
:mod:`polyrmc.tier0` reads global state -- config is passed in explicitly so a
run is reproducible from its recorded configuration alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Validated(BaseModel):
    """Config models validate on assignment, not only on construction.

    ``set_config`` exists to refuse an out-of-range value. Without this, pydantic
    checks a field when the object is built and never again, so a later
    assignment of polyorder = 9 would be accepted silently -- and the tool whose
    whole job is to reject it would report success.
    """

    model_config = ConfigDict(validate_assignment=True)


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


class OpticalConfig(_Validated):
    """Optical constants for the scattering calculation."""

    wavelength_nm: float = Field(660.0, gt=0, description="Vacuum wavelength lambda_0.")
    angle_deg: float = Field(90.0, gt=0, lt=180, description="Detection angle theta.")
    solvent_refractive_index: float = Field(1.333, gt=0, description="n_0.")
    # None means the question is still open, exactly as it does for ls_units and
    # toluene_ref below. It blocks K -- and so Mw, A2 and A3 -- rather than
    # stalling the run, because loading a trace, finding its anomalies and
    # choosing a smoothing window need no optics at all. The alternative was
    # requiring an answer before the file could be opened, which is how a
    # researcher ends up typing a plausible number to get past a dialog.
    dn_dc: float | None = Field(
        None, description="Refractive index increment, cm^3/g. None blocks K."
    )

    # Absolute scale factor: I_meas = alpha * I_R. None until the toluene
    # reference is measured on this instrument. A2*Mw and Mw/M0 do not need it.
    alpha: float | None = Field(
        None, gt=0, description="Instrument scale factor; None if uncalibrated."
    )


class DilutionConfig(_Validated):
    """Automatic continuous dilution trajectory: c(t) = c0 * exp(-Q t / V)."""

    c0_g_per_cm3: float = Field(..., gt=0, description="Initial concentration.")
    flow_rate_cm3_per_s: float = Field(..., gt=0, description="Solvent inflow rate Q.")
    cell_volume_cm3: float = Field(1.0, gt=0, description="Fixed cuvette volume V.")


class SmoothingConfig(_Validated):
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


class DetectorConfig(_Validated):
    """Thresholds for the deterministic anomaly detectors."""

    hampel_window: int = Field(31, ge=3, description="Rolling MAD window (odd).")
    hampel_n_sigmas: float = Field(5.0, gt=0)
    derivative_n_sigmas: float = Field(6.0, gt=0)
    saturation_min_run: int = Field(5, ge=2, description="Repeats that count as stuck.")
    saturation_ceiling: float | None = Field(
        None,
        description="Declared full-scale reading of this detector. None means it "
        "has not been declared, in which case saturation masking is blocked. It "
        "is never read off the trace: the maximum of a railed trace is the rail, "
        "so the data would confirm any ceiling it was asked about.",
    )
    saturation_run_ceiling: int = Field(
        2, ge=1, description="Longest saturated run still treated as a transient."
    )
    saturation_fraction_ceiling: float = Field(
        0.01, gt=0, lt=1, description="Saturated fraction above which the run escalates."
    )
    variance_ratio_threshold: float = Field(6.0, gt=1)
    changepoint_penalty: float | None = Field(
        None, description="PELT penalty; None selects a BIC-style default."
    )


class FitRangeConfig(_Validated):
    """Candidate generation for the linear region of Kc/I_R vs c."""

    min_points: int = Field(20, ge=5, description="Fewest points in a usable range.")
    n_candidates: int = Field(10, ge=2)
    max_residual_autocorr: float = Field(0.25, gt=0, lt=1)
    min_r_squared: float = Field(0.95, gt=0, lt=1)


class BaselineConfig(_Validated):
    """Tolerances for anchor-based baseline correction.

    Every one of these is a guard rather than a knob. Smoothing errors are
    bounded -- the safe band caps the damage and the residual test exposes it.
    Baseline errors are not: subtracting a curve fitted to anchors that were
    really part of the kinetics yields a smooth, plausible, physically wrong
    trace, and no downstream check will flag it. Hence a default of no
    correction, and a validator that rejects rather than warns.
    """

    anchor_flatness_tolerance: float = Field(
        1.0,
        gt=0,
        description="Local slope ceiling, as a multiple of the noise-equivalent "
        "slope. Compared against noise rather than zero, since nothing real is "
        "exactly flat.",
    )
    anchor_variance_factor: float = Field(
        2.0,
        gt=0,
        description="Local scatter ceiling, as a multiple of the global noise "
        "floor. The design says 'a small factor' without pinning one; this is "
        "that factor, and it is a calibration target rather than a settled value.",
    )
    min_anchors: int = Field(2, ge=1, description="A single anchor cannot define a trend.")
    min_anchor_duration: float = Field(
        0.02, gt=0, lt=1, description="Shortest anchor, as a fraction of the run."
    )
    min_anchor_coverage: float = Field(
        0.10, gt=0, le=1, description="Total anchor duration, as a fraction of the run."
    )
    transition_edge_fraction: float = Field(
        0.20,
        gt=0,
        lt=1,
        description="Lag-slope threshold defining the edge of the transition region.",
    )
    drift_slope_ceiling: float = Field(
        0.10,
        gt=0,
        lt=1,
        description="Max drift slope as a fraction of the signal's own maximum. A "
        "drift model that can move as fast as the kinetics will absorb them.",
    )
    drift_impact_ceiling: float = Field(
        0.20,
        gt=0,
        description="Max permitted change in total fold change. A correction that "
        "materially moves the scientific conclusion requires a human, not a default.",
    )


class LoopConfig(_Validated):
    """Bounds on a tier-1 propose -> judge -> re-propose loop."""

    max_iterations: int = Field(3, ge=1, description="Hard cap; then fall back.")
    # A stable (non-preview) id, so a recorded run stays replayable. Pro tiers
    # return a zero free-tier quota; this one runs on the free plan.
    model: str = Field("gemini-3.6-flash", description="Pinned judge model id.")
    max_tokens: int = Field(4000, ge=256)
    replay: bool = Field(
        False, description="Reuse recorded decisions instead of calling the model."
    )


class RunConfig(_Validated):
    """Everything needed to reproduce one processing run."""

    run_id: str
    source_file: Path

    # Supplied, never inferred: misclassifying a run makes every downstream
    # number wrong in a way that looks entirely normal. The two spellings are
    # the design's, so a config, a declaration, and a fingerprint hard filter
    # all key on the same strings.
    #
    # None is "not yet declared", not "either". concentration_column already has
    # an explicit branch for it that costs the column and records why, so an
    # undeclared run loads and is inspectable and simply cannot produce a
    # concentration axis until somebody says which experiment this was.
    experiment_type: Literal["acd_dilution", "fixed_conc_aggregation"] | None = None

    # The instrument writes ls1..ls16; only ls8 is the measurement. The others
    # are dropped at load and never reach any analysis stage.
    channel: str = Field("ls8", description="Scattering channel to analyse.")

    optical: OpticalConfig = Field(default_factory=OpticalConfig)
    dilution: DilutionConfig | None = None

    # A fixed-concentration run has a c0 but no dilution geometry, so it needs
    # somewhere to put it. On a dilution run the DilutionConfig copy is the one
    # in force, since Q, V and c0 are one coherent set there.
    c0_g_per_cm3: float | None = Field(
        None, gt=0, description="Initial concentration, for runs with no dilution."
    )

    @property
    def initial_concentration(self) -> float | None:
        """c0 from wherever this run type keeps it."""
        if self.dilution is not None:
            return self.dilution.c0_g_per_cm3
        return self.c0_g_per_cm3

    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    detectors: DetectorConfig = Field(default_factory=DetectorConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    fit_range: FitRangeConfig = Field(default_factory=FitRangeConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)

    # Declared at intake, never inferred. None means the question is still open,
    # which blocks I_rayleigh rather than stalling the run.
    ls_units: Literal["count_rate", "normalized"] | None = Field(
        None,
        description="Whether the scattering channel is a detector count rate or "
        "already normalised. None blocks the absolute Rayleigh ratio.",
    )
    toluene_ref: float | None = Field(
        None, gt=0, description="Toluene reading on this instrument, same units as the channel."
    )
    r_toluene_cm_inv: float | None = Field(
        None,
        gt=0,
        description="Rayleigh ratio of toluene at this wavelength, cm^-1. Declared "
        "per instrument rather than defaulted: the sources in this project "
        "disagree by about 26 percent at 660 nm, and the choice scales Mw and A2 "
        "without disturbing A2*Mw or Mw/M0, so no internal check would catch it.",
    )

    m0_max_window: float = Field(
        0.05,
        gt=0,
        lt=1,
        description="Cap on the M0 window as a fraction of the run. It is a "
        "baseline estimate, not a summary of the early kinetics.",
    )
    m0_sensitivity_ceiling: float = Field(
        0.05,
        gt=0,
        description="Max tolerated divergence in total fold change between the "
        "adaptive-window and first-point definitions of M0.",
    )

    solvent_blank_counts: float = Field(
        0.0, ge=0, description="Buffer blank subtracted from the raw channel."
    )
    k_h: float = Field(
        0.0, description="Hydrodynamic term in k_D = 2*A2Mw + k_H."
    )

    @model_validator(mode="after")
    def _dilution_required(self) -> RunConfig:
        if self.experiment_type == "acd_dilution" and self.dilution is None:
            raise ValueError("acd_dilution runs require a DilutionConfig")
        return self
