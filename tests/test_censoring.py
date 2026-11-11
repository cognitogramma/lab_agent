"""Saturation is censored data: masked, never capped, never imputed."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_spectranalytics.tier0.censoring import (
    apply_censoring,
    censor_saturation,
    contiguous_spans,
    saturated_mask,
)


@pytest.fixture
def trace() -> np.ndarray:
    return np.linspace(0.1, 0.9, 200)


def test_no_declared_ceiling_masks_nothing_and_says_so(trace):
    """The ceiling is declared. Reading it off the trace would be circular."""
    masked, report = censor_saturation(trace, ceiling=None)

    assert not report.applied
    assert report.n_censored == 0
    assert np.array_equal(masked, trace)
    assert any("no detector ceiling has been declared" in r for r in report.blocked)


def test_saturated_samples_become_nan_not_the_ceiling(trace):
    signal = trace.copy()
    signal[50] = 1.0
    masked, report = censor_saturation(signal, ceiling=1.0)

    assert np.isnan(masked[50])
    # Capping would assert a value known to be false and bias every fit low.
    assert not np.any(masked == 1.0)
    assert report.censored_indices == [50]


def test_values_past_the_ceiling_are_caught_too(trace):
    signal = trace.copy()
    signal[10] = 1.5
    _masked, report = censor_saturation(signal, ceiling=1.0)
    assert report.censored_indices == [10]


def test_the_input_array_is_not_modified(trace):
    signal = trace.copy()
    signal[50] = 1.0
    censor_saturation(signal, ceiling=1.0)
    assert signal[50] == 1.0


def test_length_is_preserved_so_the_time_axis_is_not_shortened(trace):
    signal = trace.copy()
    signal[20:30] = 1.0
    masked, _report = censor_saturation(signal, ceiling=1.0)
    assert masked.size == trace.size


def test_an_isolated_point_bridges(trace):
    signal = trace.copy()
    signal[50] = 1.0
    _masked, report = censor_saturation(signal, ceiling=1.0, run_ceiling=2)

    span = report.spans[0]
    assert span.render == "bridge"
    assert not span.censored
    assert not report.sustained_spans


def test_a_sustained_run_breaks_and_is_marked_censored(trace):
    signal = trace.copy()
    signal[40:60] = 1.0
    _masked, report = censor_saturation(signal, ceiling=1.0, run_ceiling=2)

    span = report.spans[0]
    assert span.render == "break"
    assert span.censored
    assert span.n_points == 20
    assert any("no information about the signal" in e for e in report.escalations)


def test_saturation_inside_the_transition_escalates(trace):
    signal = trace.copy()
    signal[95:99] = 1.0
    _masked, report = censor_saturation(
        signal, ceiling=1.0, transition_region=(90, 120)
    )
    assert any("maximum-slope landmark is lost" in e for e in report.escalations)


def test_too_much_saturation_escalates(trace):
    signal = trace.copy()
    signal[:20] = 1.0
    _masked, report = censor_saturation(signal, ceiling=1.0, fraction_ceiling=0.01)

    assert report.saturated_fraction == pytest.approx(0.10)
    assert any("wrong gain" in e for e in report.escalations)


def test_a_clean_trace_produces_no_spans_and_no_escalations(trace):
    masked, report = censor_saturation(trace, ceiling=1.0)
    assert report.applied
    assert report.spans == []
    assert report.escalations == []
    assert np.array_equal(masked, trace)


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ([False, False, False], []),
        ([True, True, False, True], [(0, 1), (3, 3)]),
        ([True], [(0, 0)]),
    ],
)
def test_contiguous_spans(mask, expected):
    assert contiguous_spans(np.array(mask)) == expected


def test_saturated_mask_ignores_nan():
    values = np.array([0.5, np.nan, 1.0])
    assert saturated_mask(values, 1.0).tolist() == [False, False, True]


def test_censoring_reaches_every_computed_column():
    """Correctness must not depend on each new column remembering the rule."""
    columns = {
        "concentration": np.ones(10),
        "rayleigh": np.ones(10),
        "mw_over_m0": np.ones(10),
    }
    apply_censoring(columns, [3, 4])

    for name, values in columns.items():
        assert np.isnan(values[3]), name
        assert np.isnan(values[4]), name
        assert np.isfinite(values[0]), name


def test_censoring_tolerates_an_index_past_a_shorter_column():
    columns = {"short": np.ones(3)}
    apply_censoring(columns, [1, 99])
    assert np.isnan(columns["short"][1])
