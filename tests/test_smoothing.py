"""Smoothing: Savitzky-Golay only, low order, landmark-preserving candidates."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.config import SmoothingConfig
from polyrmc.tier0 import smoothing as smoothing_module
from polyrmc.tier0.smoothing import (
    candidate_windows,
    conservative_window,
    find_landmarks,
    landmark_attenuation,
    propose_windows,
    reference_trace,
    residual_autocorrelation,
    savgol_smooth,
)


def test_rolling_median_and_moving_average_are_not_implemented():
    """These are prohibited: they attenuate the transitions being measured."""
    exported = dir(smoothing_module)
    for banned in ("rolling_median", "moving_average", "boxcar", "median_filter"):
        assert not any(banned in name for name in exported), (
            f"{banned} must not exist in the smoothing module"
        )


def test_polyorder_above_the_ceiling_is_refused(rng):
    signal = rng.normal(0, 1.0, 500)
    with pytest.raises(ValueError, match="exceeds the ceiling"):
        savgol_smooth(signal, window=21, polyorder=4)


def test_even_window_is_refused(rng):
    signal = rng.normal(0, 1.0, 500)
    with pytest.raises(ValueError, match="must be odd"):
        savgol_smooth(signal, window=20, polyorder=2)


def test_smoothing_reduces_noise_without_moving_the_mean(rng):
    time_s = np.arange(1000, dtype=float)
    clean = 100.0 + 10.0 * np.sin(time_s / 80.0)
    signal = clean + rng.normal(0, 2.0, time_s.size)

    smoothed = savgol_smooth(signal, window=31, polyorder=2)

    assert np.std(smoothed - clean) < np.std(signal - clean)
    assert abs(np.mean(smoothed) - np.mean(clean)) < 0.5


def test_nan_gaps_survive_smoothing_without_being_invented(rng):
    signal = rng.normal(100, 1.0, 500)
    signal[200:210] = np.nan

    smoothed = savgol_smooth(signal, window=21, polyorder=2)

    assert np.isnan(smoothed[200:210]).all(), "excised points must not gain values"
    assert np.isfinite(smoothed[:200]).all()


def test_candidate_windows_are_odd_and_within_bounds():
    config = SmoothingConfig(min_window=5, max_window=101, n_candidates=8)
    windows = candidate_windows(2000, config)

    assert windows == sorted(set(windows))
    assert all(window % 2 == 1 for window in windows)
    assert all(5 <= window <= 101 for window in windows)


def test_conservative_window_is_the_shortest_offered():
    config = SmoothingConfig(min_window=5, max_window=101, n_candidates=8)
    assert conservative_window(2000, config) == min(candidate_windows(2000, config))


def test_residual_autocorrelation_rises_when_the_window_is_too_wide(rng):
    time_s = np.arange(2000, dtype=float)
    clean = 100.0 + 30.0 * np.exp(-((time_s - 1000) ** 2) / (2 * 40.0**2))
    signal = clean + rng.normal(0, 1.0, time_s.size)

    narrow = residual_autocorrelation(signal, savgol_smooth(signal, 11, 2))
    wide = residual_autocorrelation(signal, savgol_smooth(signal, 301, 2))

    assert abs(wide) > abs(narrow), "over-smoothing leaves structure in the residual"


def test_landmark_attenuation_grows_with_window_width(rng):
    time_s = np.arange(2000, dtype=float)
    signal = (
        100.0
        + 60.0 * np.exp(-((time_s - 1000) ** 2) / (2 * 25.0**2))
        + rng.normal(0, 1.0, time_s.size)
    )
    extrema, change_points = find_landmarks(signal)
    # Preservation is measured against the lightly smoothed reference, which is
    # how propose_windows applies it.
    reference = reference_trace(signal)

    narrow = landmark_attenuation(
        reference, savgol_smooth(signal, 11, 2), extrema, change_points, search_radius=5
    )
    wide = landmark_attenuation(
        reference, savgol_smooth(signal, 201, 2), extrema, change_points, search_radius=100
    )

    assert narrow < 0.2, "a window narrower than the feature must preserve it"
    assert wide > narrow


def test_every_proposed_window_passes_both_gates(rng):
    time_s = np.arange(3000, dtype=float)
    signal = (
        1000.0 * np.exp(-time_s / 1000.0)
        + 40.0 * np.exp(-((time_s - 1500) ** 2) / (2 * 30.0**2))
        + rng.normal(0, 2.0, time_s.size)
    )
    config = SmoothingConfig(min_window=5, max_window=201, n_candidates=10)

    candidates = propose_windows(signal, config)

    assert candidates, "at least one window should be defensible on this trace"
    for candidate in candidates:
        # One-sided: negative autocorrelation is the signature of clean noise
        # removal, so only the positive side is a failure.
        assert candidate.features["residual_autocorr"] <= config.max_residual_autocorr
        assert candidate.features["landmark_attenuation"] <= config.max_landmark_attenuation
        assert candidate.value % 2 == 1


def test_negative_residual_autocorrelation_is_not_penalized(rng):
    """Subtracting a local fit from white noise leaves a negatively correlated
    residual. That is correct behavior, and a two-sided gate would reject every
    short window on any real trace.
    """
    noise = rng.normal(0, 1.0, 3000)
    assert residual_autocorrelation(noise, savgol_smooth(noise, 5, 2)) < -0.5

    candidates = propose_windows(noise, SmoothingConfig(min_window=5, max_window=151))
    assert any(c.value <= 11 for c in candidates), "short windows must remain available"


def test_landmarks_ignore_the_smooth_trend_of_a_decaying_trace(rng):
    """PELT models a piecewise-constant mean, so it segments a smooth curve into
    steps. Those are not physical transitions and must not count as landmarks.
    """
    time_s = np.arange(3000, dtype=float)
    decay = 1000.0 * np.exp(-time_s / 1000.0) + rng.normal(0, 2.0, time_s.size)

    _extrema, change_points = find_landmarks(decay)
    assert change_points == []


def test_landmarks_do_find_an_abrupt_transition(rng):
    time_s = np.arange(3000, dtype=float)
    signal = 1000.0 * np.exp(-time_s / 1000.0) + rng.normal(0, 2.0, time_s.size)
    signal[1500:] += 200.0

    _extrema, change_points = find_landmarks(signal)
    assert change_points, "a 200-unit step must register as a landmark"
    assert min(abs(point - 1500) for point in change_points) < 60


def test_changepoint_search_stays_fast_on_a_long_trace(rng):
    """The real target file is ~79,000 rows; the search decimates to stay usable."""
    import time as time_module

    from polyrmc.tier0.detect import pelt_changepoints

    long_signal = np.concatenate([rng.normal(0, 1.0, 40000), rng.normal(8, 1.0, 40000)])
    started = time_module.perf_counter()
    points = pelt_changepoints(long_signal)
    elapsed = time_module.perf_counter() - started

    assert elapsed < 20.0, f"change-point search took {elapsed:.1f}s"
    assert points and min(abs(point - 40000) for point in points) < 500


def test_candidate_indices_are_contiguous_from_zero(rng):
    signal = rng.normal(100, 1.0, 2000)
    candidates = propose_windows(signal, SmoothingConfig(min_window=5, max_window=151))
    assert [c.index for c in candidates] == list(range(len(candidates)))
