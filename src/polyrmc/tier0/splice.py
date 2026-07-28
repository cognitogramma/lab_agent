"""Excise flagged points without disturbing the time axis.

The rule that matters: **survivors are never re-indexed.** Excised samples
become NaN in place, so the array keeps its original length and every timestamp
keeps its original position. Compacting the array instead would shorten the
time axis and distort every derivative taken from it.

Rendering branches on anomaly type. A single-point dropout is bridged
invisibly, because the underlying signal plainly continued through it. A
multi-point excision -- saturation especially -- breaks the plotted line, so
deleted data is never implied to be real signal.
"""

from __future__ import annotations

import numpy as np

from polyrmc.state import AnomalyRecord, AnomalyType, SpliceRecord

# Drift is corrected by subtraction, not excision: removing a drifting stretch
# would delete real kinetic signal along with the drift.
_EXCISABLE = {
    AnomalyType.TRANSIENT_SPIKE,
    AnomalyType.SATURATION,
    AnomalyType.STRUCTURAL_NOISE,
}


def apply_splice(
    signal: np.ndarray, records: list[AnomalyRecord]
) -> tuple[np.ndarray, SpliceRecord]:
    """Blank out excisable regions, preserving length and index alignment.

    Returns the spliced copy and the bookkeeping needed to reproduce or plot it.
    """
    spliced = np.array(signal, dtype=float, copy=True)
    splice = SpliceRecord(n_original=int(signal.size))

    for record in records:
        if record.anomaly_type not in _EXCISABLE:
            continue
        start, end = record.start_index, record.end_index
        spliced[start : end + 1] = np.nan
        splice.excised_indices.extend(range(start, end + 1))

        single_point = record.n_points == 1
        if single_point and record.anomaly_type is not AnomalyType.SATURATION:
            splice.bridged.append(start)
        else:
            splice.render_breaks.append((start, end))

    splice.excised_indices = sorted(set(splice.excised_indices))
    return spliced, splice


def render_segments(
    time_s: np.ndarray, signal: np.ndarray, splice: SpliceRecord
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split the trace into contiguous segments for plotting.

    Bridged single-point dropouts stay inside a segment; multi-point excisions
    end one segment and start the next, leaving a visible gap.
    """
    breaking = np.zeros(signal.size, dtype=bool)
    for start, end in splice.render_breaks:
        breaking[start : end + 1] = True

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    current: list[int] = []
    for index in range(signal.size):
        if breaking[index]:
            if current:
                segments.append((time_s[current], signal[current]))
                current = []
            continue
        current.append(index)
    if current:
        segments.append((time_s[current], signal[current]))
    return segments


def interpolate_bridged(
    time_s: np.ndarray, signal: np.ndarray, splice: SpliceRecord
) -> np.ndarray:
    """Fill only the bridged single-point gaps, by linear interpolation in time.

    Multi-point excisions are deliberately left as NaN. Filling them would
    manufacture data across a stretch where the instrument reported nothing
    usable.
    """
    filled = np.array(signal, dtype=float, copy=True)
    if not splice.bridged:
        return filled

    known = np.isfinite(filled)
    if known.sum() < 2:
        return filled

    for index in splice.bridged:
        if 0 < index < filled.size - 1:
            filled[index] = np.interp(time_s[index], time_s[known], filled[known])
    return filled
