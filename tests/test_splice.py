"""Splicing must not re-index, and must not imply deleted data is real signal."""

from __future__ import annotations

import numpy as np

from polyrmc.state import AnomalyRecord, AnomalyType
from polyrmc.tier0.splice import apply_splice, interpolate_bridged, render_segments


def _record(start, end, anomaly_type):
    return AnomalyRecord(
        start_index=start,
        end_index=end,
        channel="Cell_1",
        detected_by=["test"],
        anomaly_type=anomaly_type,
        classified_by="deterministic",
    )


def test_excision_preserves_length_and_index_alignment():
    signal = np.arange(100, dtype=float)
    records = [_record(40, 44, AnomalyType.SATURATION)]

    spliced, splice = apply_splice(signal, records)

    assert spliced.size == signal.size, "survivors must never be re-indexed"
    assert np.isnan(spliced[40:45]).all()
    assert spliced[39] == 39.0 and spliced[45] == 45.0
    assert splice.n_excised == 5
    assert splice.n_original == 100


def test_baseline_drift_is_never_excised():
    """Drift is corrected by subtraction; excising it would delete kinetics."""
    signal = np.arange(100, dtype=float)
    records = [_record(10, 60, AnomalyType.BASELINE_DRIFT)]

    spliced, splice = apply_splice(signal, records)

    assert splice.n_excised == 0
    assert np.isfinite(spliced).all()


def test_single_point_dropout_is_bridged_but_multipoint_breaks_the_line():
    signal = np.ones(50, dtype=float)
    records = [
        _record(10, 10, AnomalyType.TRANSIENT_SPIKE),
        _record(30, 35, AnomalyType.SATURATION),
    ]

    _spliced, splice = apply_splice(signal, records)

    assert splice.bridged == [10]
    assert splice.render_breaks == [(30, 35)]


def test_single_point_saturation_still_breaks_the_line():
    """Saturation is never bridged, even when only one sample long."""
    signal = np.ones(20, dtype=float)
    _spliced, splice = apply_splice(signal, [_record(5, 5, AnomalyType.SATURATION)])

    assert splice.bridged == []
    assert splice.render_breaks == [(5, 5)]


def test_render_segments_leaves_a_visible_gap_at_a_break():
    time_s = np.arange(50, dtype=float)
    signal = np.ones(50)
    _spliced, splice = apply_splice(signal, [_record(20, 25, AnomalyType.SATURATION)])

    segments = render_segments(time_s, signal, splice)

    assert len(segments) == 2
    assert segments[0][0][-1] == 19.0
    assert segments[1][0][0] == 26.0


def test_interpolation_fills_bridged_points_but_not_excised_stretches():
    time_s = np.arange(50, dtype=float)
    signal = np.arange(50, dtype=float)
    records = [
        _record(10, 10, AnomalyType.TRANSIENT_SPIKE),
        _record(30, 35, AnomalyType.SATURATION),
    ]
    spliced, splice = apply_splice(signal, records)

    filled = interpolate_bridged(time_s, spliced, splice)

    assert np.isclose(filled[10], 10.0), "a one-sample dropout is safe to bridge"
    assert np.isnan(filled[30:36]).all(), "a multi-point gap must stay empty"
