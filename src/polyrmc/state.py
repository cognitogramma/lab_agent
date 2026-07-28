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
    """The four artifact classes the pipeline distinguishes."""

    TRANSIENT_SPIKE = "transient_spike"
    BASELINE_DRIFT = "baseline_drift"
    SATURATION = "saturation"
    STRUCTURAL_NOISE = "structural_noise"


class TimeFormat(str, Enum):
    """How the elapsed-time field was encoded in the source file."""

    WRAPPING_MMSS = "wrapping_mmss"  # MM:SS(.s), resets to zero every hour
    ABSOLUTE_HHMMSS = "absolute_hhmmss"
    SECONDS = "seconds"


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


class JudgeDecision(BaseModel):
    """A judge's response for one loop iteration."""

    iteration: int = Field(..., ge=0)
    action: Literal["select", "repropose"]
    selected_index: int | None = None
    direction_hint: str | None = Field(
        None, description="Only meaningful when action == 'repropose'."
    )
    rationale: str = ""
    rejected_indices: list[int] = Field(
        default_factory=list,
        description="Logged alongside the accepted one. For fit ranges this is "
        "the scientific claim: which points were excluded from the linear region.",
    )


class ModelProvenance(BaseModel):
    """Pinned identity of the judge that produced a decision."""

    model: str
    prompt_hash: str = Field(..., description="SHA-256 of the exact rendered prompt.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    replayed: bool = Field(
        False, description="True if served from the log rather than the provider."
    )


class LoopState(BaseModel):
    """Full trajectory of one bounded evaluator loop."""

    name: str
    max_iterations: int
    iterations_used: int = 0
    candidates_by_iteration: list[list[Candidate]] = Field(default_factory=list)
    decisions: list[JudgeDecision] = Field(default_factory=list)
    provenance: list[ModelProvenance] = Field(default_factory=list)

    resolved_value: Any = None
    used_fallback: bool = Field(
        False, description="True if the cap was hit and the safe default was taken."
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
    dropped_rows: list[int] = Field(
        default_factory=list, description="Rows dropped for nulls in scattering channels."
    )

    anomalies: list[AnomalyRecord] = Field(default_factory=list)
    splice: SpliceRecord = Field(default_factory=SpliceRecord)
    loops: dict[str, LoopState] = Field(default_factory=dict)

    fit: FitResult | None = None
    parameters: DerivedParameters | None = None
    warnings: list[str] = Field(default_factory=list)
