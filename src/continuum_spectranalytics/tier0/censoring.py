"""Saturation handling: a saturated sample is censored, not wrong.

Design section 6.5.1. When the detector rails, the recorded value is not the
sample's scattering -- it is the largest number the instrument could report. The
true value is **at least** that, and nothing in the file says by how much. That
framing settles what to do with it.

*Cap-and-annotate* was rejected: writing the ceiling asserts a specific value
known to be false, biases every downstream fit low, and is indistinguishable
from real data to anything that does not check the flag. *Keep-and-flag* fails
the same way the moment one consumer forgets the flag, and correctness must not
depend on downstream discipline. What is implemented here is **mask-and-exclude**:
the value becomes NaN, the timestamp stays exactly where it was, and no computed
column carries a number at that index.

Two refinements matter more than the choice itself.

**Run length changes the meaning.** An isolated point is a transient and the
plotted line bridges it. A sustained run means the detector was clipping
throughout, so there is *no information* about the signal across that span; the
line must break, and any fit spanning it is marked as crossing a censored
region.

**Bridging is a render, never an imputation.** This module returns instructions
for drawing and never writes a value into the data. The true value across a
censored region is not interpolated, because it is not known.

The ceiling itself is **declared, never guessed** -- it comes from the
instrument's own documentation by way of intake. Reading it off the data is
circular: the maximum of a railed trace *is* the rail, so any trace would
confirm any ceiling. Without a declared ceiling this module does nothing and
says so, and saturation masking is reported as blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Samples at or above the ceiling. A run no longer than this is a transient
# excursion; anything longer means the detector was clipping across the span.
DEFAULT_RUN_CEILING = 2

# Total saturated fraction above which the run stops being a trace with a few
# bad points and becomes a trace acquired at the wrong gain.
DEFAULT_FRACTION_CEILING = 0.01


@dataclass
class CensoredSpan:
    """One contiguous stretch of samples at the detector ceiling."""

    start_index: int
    end_index: int
    """Inclusive."""

    render: Literal["bridge", "break"]
    """How to draw across it. Never how to fill it."""

    censored: bool
    """True for a sustained run, where no information about the signal survives."""

    @property
    def n_points(self) -> int:
        return self.end_index - self.start_index + 1


@dataclass
class CensorReport:
    """What was masked, how to draw it, and what needs a human."""

    ceiling: float | None
    applied: bool
    spans: list[CensoredSpan] = field(default_factory=list)
    censored_indices: list[int] = field(default_factory=list)
    saturated_fraction: float = 0.0
    escalations: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    @property
    def n_censored(self) -> int:
        return len(self.censored_indices)

    @property
    def sustained_spans(self) -> list[CensoredSpan]:
        return [span for span in self.spans if span.censored]


def saturated_mask(signal: np.ndarray, ceiling: float) -> np.ndarray:
    """Samples at or above the declared full-scale reading.

    At *or above*: a detector that rails can report its ceiling, and a
    downstream rescaling can put a sample fractionally past it. Testing for
    equality would let those through as if they were measurements.
    """
    values = np.asarray(signal, dtype=float)
    return np.isfinite(values) & (values >= ceiling)


def contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of each contiguous True run."""
    flagged = np.flatnonzero(np.asarray(mask, dtype=bool))
    if flagged.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(flagged) > 1)
    starts = np.concatenate(([flagged[0]], flagged[breaks + 1]))
    ends = np.concatenate((flagged[breaks], [flagged[-1]]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def censor_saturation(
    signal: np.ndarray,
    ceiling: float | None,
    run_ceiling: int = DEFAULT_RUN_CEILING,
    fraction_ceiling: float = DEFAULT_FRACTION_CEILING,
    transition_region: tuple[int, int] | None = None,
) -> tuple[np.ndarray, CensorReport]:
    """Mask saturated samples and report what needs escalating.

    Returns a copy of the signal with saturated samples set to NaN, and the
    report describing every span. The input array is never modified, and the
    output has the same length as the input -- survivors are not re-indexed,
    because compacting the array would shorten the time axis and distort every
    derivative taken from it.

    Parameters
    ----------
    ceiling:
        The declared full-scale reading. ``None`` means it has not been declared
        yet, in which case nothing is masked and the omission is reported as
        blocked rather than filled in from the data.
    transition_region:
        Inclusive index range of the kinetic transition, when it is known.
        Saturation overlapping it destroys the maximum-slope landmark, which
        contaminates smoothing-window selection and the run fingerprint, so that
        case escalates on its own.
    """
    values = np.asarray(signal, dtype=float)

    if ceiling is None:
        return np.array(values, copy=True), CensorReport(
            ceiling=None,
            applied=False,
            blocked=[
                "saturation masking: blocked because no detector ceiling has been "
                "declared for this instrument. It is not read off the trace -- the "
                "maximum of a railed trace is the rail, so the data would confirm "
                "any ceiling it was asked about."
            ],
        )

    mask = saturated_mask(values, ceiling)
    masked = np.array(values, copy=True)
    masked[mask] = np.nan

    report = CensorReport(
        ceiling=float(ceiling),
        applied=True,
        censored_indices=np.flatnonzero(mask).tolist(),
        saturated_fraction=float(mask.sum() / mask.size) if mask.size else 0.0,
    )

    for start, end in contiguous_spans(mask):
        n_points = end - start + 1
        sustained = n_points > run_ceiling
        report.spans.append(
            CensoredSpan(
                start_index=start,
                end_index=end,
                render="break" if sustained else "bridge",
                censored=sustained,
            )
        )
        if sustained:
            report.escalations.append(
                f"samples {start}-{end} ({n_points}) sit at the detector ceiling "
                f"{ceiling:g}. The detector was clipping throughout, so there is no "
                "information about the signal across that span. Any fit crossing it "
                "spans a censored region."
            )
        if transition_region is not None:
            low, high = transition_region
            if start <= high and end >= low:
                report.escalations.append(
                    f"samples {start}-{end} saturate inside the kinetic transition "
                    f"({low}-{high}). The maximum-slope landmark is lost, which "
                    "contaminates both the smoothing window and the run fingerprint."
                )

    if report.saturated_fraction > fraction_ceiling:
        report.escalations.append(
            f"{100 * report.saturated_fraction:.2f} percent of samples are saturated, "
            f"above the {100 * fraction_ceiling:.2f} percent ceiling. That is a trace "
            "acquired at the wrong gain rather than a trace with a few bad points."
        )

    return masked, report


def apply_censoring(columns: dict[str, np.ndarray], censored_indices: list[int]) -> None:
    """Blank the censored indices in every computed column, in place.

    The invariant this enforces is that no computed column carries a value where
    the detector was railed. Applying it once at the end, over every column at
    once, is what makes it checkable -- a per-column rule would be re-derived,
    and eventually forgotten, at each new column.
    """
    if not censored_indices:
        return
    index = np.asarray(censored_indices, dtype=int)
    for values in columns.values():
        if values is None:
            continue
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            continue
        inside = index[index < array.size]
        array[inside] = np.nan


__all__ = [
    "DEFAULT_FRACTION_CEILING",
    "DEFAULT_RUN_CEILING",
    "CensorReport",
    "CensoredSpan",
    "apply_censoring",
    "censor_saturation",
    "contiguous_spans",
    "saturated_mask",
]
