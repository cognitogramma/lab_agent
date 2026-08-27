"""Typed state schema for a processing run.

Covers raw and derived arrays, cleaning and splice bookkeeping, anomaly records
with split authorship between code and model, loop control state, and pinned
model provenance.

Anomaly authorship is split deliberately. ``detected_by`` names the
deterministic detector that fired; ``classified_by`` names whoever assigned the
type. A reviewer must be able to tell which flags are arithmetic and which are
judgment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class AnomalyType(str, Enum):
    """The artifact classes the pipeline distinguishes.

    The four wire values are the design's classification taxonomy verbatim.
    ``splice_gap`` is bookkeeping rather than a classification: it marks a break
    in acquisition rather than a defect in the signal.
    """

    TRANSIENT_SPIKE = "transient_spike"
    BASELINE_DRIFT = "baseline_drift"
    SATURATION_ARTIFACT = "saturation_artifact"
    STRUCTURAL_NOISE = "structural_noise"
    SPLICE_GAP = "splice_gap"


class TimeFormat(str, Enum):
    """How the elapsed-time field was encoded in the source file."""

    WRAPPING_MMSS = "wrapping_mmss"  # MM:SS(.s), resets to zero every hour
    ABSOLUTE_HHMMSS = "absolute_hhmmss"
    SECONDS = "seconds"


class Declaration(BaseModel):
    """One statement by a person about their own experiment.

    Author and timestamp are mandatory. A declaration is the pipeline's record
    of who asserted a physical fact, and an unattributed assertion cannot be
    questioned later by whoever finds it wrong.
    """

    value: Any
    author: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    recalled_from: str | None = Field(
        None, description="Run id, if this was pre-filled from a previous run."
    )
    derived_from: str | None = Field(
        None,
        description="Set when another declaration fixed this value outright, e.g. "
        "the instrument fixing the wavelength. Records that no separate "
        "judgment was involved.",
    )


class Conflict(BaseModel):
    """A declaration that the data does not appear to support.

    A conflict never overrides the declaration and is never resolved by code.
    The data can be surprising for good reasons; only the researcher knows
    which. Resolution is an explicit act with a recorded reason.
    """

    check: str
    declared: str
    observed: str
    detail: str = ""
    resolved: bool = False
    resolution_reason: str | None = None
    resolved_by: str | None = None


class PathChoice(BaseModel):
    """Which branch a run took where the design leaves the question open.

    Several decisions in the specification are unresolved and cannot be closed
    by design work -- they need someone with direct knowledge of how the run was
    acquired. Every one of them has a defined behaviour for *each* possible
    answer, including the answer not being available yet, and this record says
    which of those branches a given run actually went down.

    That is what keeps an unanswered question from becoming an invisible
    assumption: the run completes, the affected column is blocked or computed by
    a stated substitute, and the reason travels with the output.
    """

    question: str = Field(..., description="The open question, e.g. 'OPEN-1'.")
    decision: str = Field(..., description="The branch taken.")
    because: str = Field(..., description="Why this branch and not another.")
    alternatives: list[str] = Field(
        default_factory=list, description="Branches not taken on this run."
    )
    blocked: list[str] = Field(
        default_factory=list, description="Outputs this branch cannot produce."
    )


class AnomalyRecord(BaseModel):
    """One flagged region of the trace."""

    start_index: int = Field(..., ge=0)
    end_index: int = Field(..., ge=0, description="Inclusive.")
    channel: str

    detected_by: list[str] = Field(
        ..., description="Deterministic detectors that fired. Never a model."
    )
    features: dict[str, float] = Field(
        default_factory=dict, description="Computed features the classifier saw."
    )

    anomaly_type: AnomalyType | None = None
    classified_by: Literal["deterministic", "model", "fallback"] | None = None
    rationale: str | None = Field(
        None, description="Free text if a model assigned the type."
    )

    decision: Literal["pending", "excise", "keep", "correct"] = Field(
        "pending",
        description="What to do with this region. A region still pending when the "
        "run tries to emit is an unfinished judgment, and blocks the emit.",
    )

    @property
    def n_points(self) -> int:
        return self.end_index - self.start_index + 1


class SpliceRecord(BaseModel):
    """Bookkeeping for excised points.

    Survivors are never re-indexed. Excised samples are set to NaN in place so
    the time axis keeps its original length and spacing -- re-indexing would
    shorten the axis and distort every derivative computed from it.
    """

    excised_indices: list[int] = Field(default_factory=list)
    n_original: int = 0
    render_breaks: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Gaps to break the plotted line across (multi-point excisions).",
    )
    bridged: list[int] = Field(
        default_factory=list,
        description="Single-point dropouts safe to bridge invisibly when plotting.",
    )

    @property
    def n_excised(self) -> int:
        return len(self.excised_indices)


class Candidate(BaseModel):
    """One option offered to a judge.

    Candidates are generated deterministically and are safe by construction: a
    judge selects among them, and can never build one.
    """

    index: int = Field(..., ge=0)
    value: Any = Field(..., description="The proposed setting (window, range, ...).")
    features: dict[str, float] = Field(default_factory=dict)
    label: str = ""



class AnchorCandidate(BaseModel):
    """One region eligible to be used as baseline.

    Candidates are enumerated by code. Nothing draws freehand on the measurement
    path: a judge -- model or human -- selects among these by id, and a freehand
    drag in a UI snaps to them. That is what keeps every index computation in
    code while still letting the selection be a matter of taste.
    """

    id: int
    index_range: tuple[int, int]
    """Inclusive."""

    duration_fraction: float = Field(..., description="Span as a fraction of the run.")
    mean_level: float
    local_slope: float = Field(..., description="Lag-slope, signal units per second.")
    local_variance: float
    position_fraction: float = Field(..., description="Midpoint as a fraction of the run.")

    @property
    def n_points(self) -> int:
        return self.index_range[1] - self.index_range[0] + 1


class DriftModel(BaseModel):
    """A fitted drift curve, recorded so the correction can be undone.

    Only two kinds are permitted. Anything with enough freedom to follow the
    transition will absorb the kinetics, so splines and higher-order polynomials
    are not options -- the drift model must be slower than the kinetics by
    construction, and the slope ceiling in the validator enforces that
    numerically.
    """

    kind: Literal["poly", "piecewise_linear"]
    order: int | None = None
    coefficients: list[float] = Field(default_factory=list)
    anchor_ranges: list[tuple[int, int]] = Field(default_factory=list)

    # Not in the published schema, but reversibility requires exact
    # reconstruction, and a polynomial fitted on scaled time cannot be
    # re-evaluated without the scaling that produced it.
    time_origin: float = 0.0
    time_scale: float = 1.0

    reference_level: float = Field(
        0.0,
        description="The fitted curve's value at the first sample, subtracted so "
        "the correction removes the trend and not the level. The constant term "
        "of a fit to baseline regions is the sample's own scattering, not drift; "
        "removing it would zero the trace at t=0 and make every ratio built on "
        "it -- Mw/M0 above all -- meaningless.",
    )

    max_slope_ratio: float = Field(
        ..., description="Drift lag-slope as a fraction of the signal's own."
    )
    fold_change_impact: float = Field(
        ..., description="Relative change in total fold change if applied."
    )
    reversible: bool = True




class LoopState(BaseModel):
    """Every candidate a selection was offered, and which one it took.

    Named for the bounded propose-judge-re-propose loop this used to record.
    The loop is gone -- the agent selects through the tool surface now -- but
    the record is not, because the record was never about the iterations. Which
    candidates passed the gates, and which one was taken, is the scientific
    claim the sidecar exists to carry.
    """

    name: str
    max_iterations: int
    iterations_used: int = 0
    candidates_by_iteration: list[list[Candidate]] = Field(default_factory=list)

    resolved_value: Any = None
    used_fallback: bool = Field(
        False,
        description="True if no candidate passed the gates and nothing was chosen.",
    )
    fallback_reason: str | None = None


class FitResult(BaseModel):
    """Output of a virial fit over a chosen concentration range."""

    slope: float
    intercept: float
    slope_stderr: float
    intercept_stderr: float
    r_squared: float
    n_points: int
    c_min: float
    c_max: float
    order: Literal[1, 2] = 1
    quadratic_coeff: float | None = None

    # alpha cancels in this ratio, so it is available before calibration.
    @property
    def a2_mw(self) -> float:
        """A2 * Mw = slope / (2 * intercept). Calibration-free."""
        return self.slope / (2.0 * self.intercept)


class DerivedParameters(BaseModel):
    """Physical parameters extracted from a run."""

    a2_mw: float = Field(..., description="Calibration-free product, cm^3/g.")
    k_d: float | None = Field(None, description="k_D = 2*A2Mw + k_H.")
    mw: float | None = Field(None, description="Needs alpha.")
    a2: float | None = Field(None, description="Needs alpha, cm^3 mol / g^2.")
    a3: float | None = None
    mw_over_m0_final: float | None = Field(
        None, description="Aggregation index; alpha cancels within a file."
    )
    calibrated: bool = False


class RunState(BaseModel):
    """The complete state of one processing run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    source_file: str

    # Intake -- declared by a human, never inferred from the trace. A non-empty
    # unresolved conflict list blocks emission of the boundary CSV.
    declarations: dict[str, Declaration] = Field(default_factory=dict)
    declaration_conflicts: list[Conflict] = Field(default_factory=list)
    blocked: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons a column could not be computed.",
    )

    # Raw + derived arrays. Kept as numpy arrays, excluded from serialization --
    # the sidecar log records decisions and indices, not bulk data.
    time_s: np.ndarray | None = Field(None, exclude=True)
    raw_signal: np.ndarray | None = Field(None, exclude=True)
    spliced_signal: np.ndarray | None = Field(None, exclude=True)
    baseline: np.ndarray | None = Field(None, exclude=True)
    corrected_signal: np.ndarray | None = Field(None, exclude=True)
    smoothed_signal: np.ndarray | None = Field(None, exclude=True)
    concentration: np.ndarray | None = Field(None, exclude=True)
    kc_over_i: np.ndarray | None = Field(None, exclude=True)

    time_format: TimeFormat | None = None
    n_header_rows: int | None = None
    metadata_lines: list[str] = Field(
        default_factory=list,
        description="The raw file's header lines, verbatim. Carried so the "
        "emitted CSV can put them back on top unchanged.",
    )
    dropped_rows: list[int] = Field(
        default_factory=list, description="Rows dropped for nulls in scattering channels."
    )
    nd_filter_changes: list[int] = Field(
        default_factory=list,
        description="Samples where the ND filter changed. Readings either side of "
        "one of these are not on the same scale.",
    )

    anomalies: list[AnomalyRecord] = Field(default_factory=list)
    splice: SpliceRecord = Field(default_factory=SpliceRecord)
    loops: dict[str, LoopState] = Field(default_factory=dict)

    noise_floor: float | None = None
    window_length: int | None = None
    polyorder: int | None = None


    execution_history: list[str] = Field(
        default_factory=list,
        description="Append-only record of what actually happened, in order. "
        "Narration is generated from this and from state, never authored "
        "independently, so what the agent says cannot drift from what it did.",
    )
    paths: list[PathChoice] = Field(
        default_factory=list,
        description="The branch taken at each unresolved design question.",
    )
    censored_indices: list[int] = Field(
        default_factory=list,
        description="Samples masked as saturated. Censored, not wrong: the true "
        "value is at least the recorded one, so no computed column may carry a "
        "value here.",
    )

    # Baseline correction. The default is no correction: anchor selection
    # decides what counts as signal, and unlike smoothing there is no downstream
    # check that catches getting it wrong.
    transition_region: tuple[int, int] | None = None
    anchor_candidates: list[AnchorCandidate] = Field(default_factory=list)
    anchors_selected: list[int] = Field(
        default_factory=list, description="Candidate ids, not indices."
    )
    anchor_selected_by: Literal["none", "llm", "vlm", "both", "researcher"] = "none"
    drift_model: DriftModel | None = None
    baseline_applied: bool = False

    m0: Any = Field(
        None,
        exclude=True,
        description="The chosen M0 reference (an m0.M0Result). Summarised into "
        "the sidecar rather than serialised here, to keep tier 0 free of a "
        "dependency on this module.",
    )

    fit: FitResult | None = None
    parameters: DerivedParameters | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def unresolved_conflicts(self) -> list[Conflict]:
        return [c for c in self.declaration_conflicts if not c.resolved]
