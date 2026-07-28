"""Baseline correction subtracts drift; it never removes samples."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.tier0.baseline import (
    asls_baseline,
    correct_drift,
    polynomial_baseline,
    subtract_baseline,
)


def test_linear_drift_is_removed_without_losing_samples(rng):
    time_s = np.arange(2000, dtype=float)
    clean = 500.0 + rng.normal(0, 2.0, time_s.size)
    drift = 0.25 * time_s
    signal = clean + drift

    corrected, baseline = correct_drift(time_s, signal, method="polynomial", order=1)

    assert corrected.size == signal.size, "no timestamp may be dropped"
    assert baseline.size == signal.size
    trend = np.polyfit(time_s, corrected, 1)[0]
    assert abs(trend) < 0.01, "the linear trend should be gone"


def test_asls_tracks_a_curved_baseline_under_a_peak(rng):
    time_s = np.arange(3000, dtype=float)
    drift = 200.0 + 1e-5 * (time_s - 1500) ** 2
    peak = 150.0 * np.exp(-((time_s - 1500) ** 2) / (2 * 60.0**2))
    signal = drift + peak + rng.normal(0, 1.0, time_s.size)

    corrected, baseline = correct_drift(time_s, signal, method="asls", smoothness=1e7)

    # The baseline should pass under the peak, leaving its height intact.
    assert corrected.max() > 100.0
    assert abs(np.median(corrected[:500])) < 15.0


def test_nan_gaps_survive_baseline_correction():
    time_s = np.arange(500, dtype=float)
    signal = 100.0 + 0.1 * time_s
    signal[200:210] = np.nan

    corrected, _baseline = correct_drift(time_s, signal, method="polynomial", order=1)

    assert np.isnan(corrected[200:210]).all()
    assert np.isfinite(corrected[:200]).all()


def test_subtract_baseline_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        subtract_baseline(np.zeros(10), np.zeros(11))


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown baseline method"):
        correct_drift(np.arange(10, dtype=float), np.zeros(10), method="rolling_median")


def test_baseline_needs_at_least_two_finite_samples():
    signal = np.full(10, np.nan)
    signal[3] = 1.0
    with pytest.raises(ValueError, match="at least two finite samples"):
        asls_baseline(signal)


def test_polynomial_order_must_be_at_least_one():
    with pytest.raises(ValueError, match="order must be at least 1"):
        polynomial_baseline(np.arange(10, dtype=float), np.zeros(10), order=0)
