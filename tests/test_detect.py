"""Deterministic detectors and the features handed to a classifier."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.state import AnomalyType
from polyrmc.tier0.classify import classify_all, classify_deterministic
from polyrmc.tier0.detect import (
    derivative_edges,
    detect_anomalies,
    hampel_outliers,
    local_variance_anomalies,
    mask_to_regions,
    noise_floor,
    pelt_changepoints,
    saturation_runs,
)


def test_noise_floor_recovers_the_injected_scale(rng):
    signal = rng.normal(0, 3.0, 5000)
    assert abs(noise_floor(signal) - 3.0) < 0.3


def test_derivative_edge_flags_a_single_point_spike(rng):
    time_s = np.arange(500, dtype=float)
    signal = rng.normal(100, 1.0, 500)
    signal[250] += 60.0

    flagged = derivative_edges(time_s, signal)
    assert flagged[249] or flagged[250] or flagged[251]
    assert flagged.sum() < 15, "a clean series should not flag broadly"


def test_hampel_flags_outliers_but_leaves_clean_data_alone(rng):
    clean = rng.normal(0, 1.0, 2000)
    assert hampel_outliers(clean).sum() < 20

    spiked = clean.copy()
    spiked[[300, 900, 1500]] += 40.0
    flagged = hampel_outliers(spiked)
    assert flagged[300] and flagged[900] and flagged[1500]


def test_saturation_run_detects_a_pinned_detector(rng):
    signal = rng.normal(100, 1.0, 300)
    signal[100:120] = 65535.0

    flagged = saturation_runs(signal, min_run=5)
    assert flagged[100:120].all()
    assert not flagged[:100].any()


def test_saturation_ignores_runs_shorter_than_the_threshold(rng):
    signal = rng.normal(100, 1.0, 300)
    signal[50:52] = 500.0
    assert not saturation_runs(signal, min_run=5).any()


def test_pelt_finds_a_known_level_change(rng):
    signal = np.concatenate([rng.normal(0, 1.0, 400), rng.normal(12, 1.0, 400)])
    points = pelt_changepoints(signal)

    assert points, "a 12-sigma level shift must be detected"
    assert min(abs(point - 400) for point in points) < 15


def test_pelt_returns_nothing_on_stationary_noise(rng):
    assert pelt_changepoints(rng.normal(0, 1.0, 800)) == []


@pytest.mark.parametrize("scale", [1e-15, 1.0, 1e9])
def test_pelt_is_invariant_to_the_units_of_the_signal(rng, scale):
    """Kc/I_R is ~1e-15 in instrument counts and ~1e-6 after calibration.

    An absolute penalty floor would make the same trace yield different change
    points before and after calibration, with nothing to flag the difference.
    """
    base = np.concatenate([rng.normal(0, 1.0, 400), rng.normal(10, 1.0, 400)])

    assert pelt_changepoints(base * scale) == pelt_changepoints(base)


def _heteroscedastic_trace(rng, n=4000):
    """Decaying trace whose noise scales with intensity, as scattering data does."""
    time_s = np.arange(n, dtype=float)
    level = 40000.0 * np.exp(-time_s / 1200.0) + 1000.0
    return time_s, level * (1.0 + rng.normal(0, 3e-4, n))


def test_detectors_tolerate_noise_that_scales_with_intensity(rng):
    """A global noise scale sits mid-range and flags the bright end wholesale.

    On this trace the local noise runs ~30x from start to finish, so the
    thresholds have to be local.
    """
    time_s, signal = _heteroscedastic_trace(rng)

    flagged = derivative_edges(time_s, signal).sum()
    flagged += local_variance_anomalies(signal).sum()

    assert flagged < 0.02 * signal.size, (
        f"{flagged} samples flagged on an artifact-free trace"
    )


def test_a_real_spike_is_still_caught_on_a_heteroscedastic_trace(rng):
    """The local threshold must not be so permissive that it misses artifacts."""
    time_s, signal = _heteroscedastic_trace(rng)
    signal[3000] += 40.0 * noise_floor(signal[2900:3100])

    assert derivative_edges(time_s, signal)[2999:3002].any()


def test_mask_to_regions_groups_consecutive_indices():
    mask = np.zeros(20, dtype=bool)
    mask[[3, 4, 5, 12, 17, 18]] = True
    assert mask_to_regions(mask) == [(3, 5), (12, 12), (17, 18)]


def test_detect_anomalies_records_which_detectors_fired(noisy_trace):
    time_s, signal = noisy_trace
    records, _ = detect_anomalies(time_s, signal, channel="Cell_1")

    assert records
    for record in records:
        assert record.detected_by, "every record names the detectors that fired"
        assert record.features, "every record carries computed features"
        assert record.anomaly_type is None, "detection must not assign a type"


def test_saturation_region_is_classified_deterministically(noisy_trace):
    time_s, signal = noisy_trace
    records, _ = detect_anomalies(time_s, signal, channel="Cell_1")
    settled, _ambiguous = classify_all(records)

    types = {record.anomaly_type for record in settled}
    assert AnomalyType.SATURATION in types
    assert all(record.classified_by == "deterministic" for record in settled)


def test_ambiguous_regions_return_none_rather_than_a_guess():
    from polyrmc.state import AnomalyRecord

    record = AnomalyRecord(
        start_index=10,
        end_index=20,
        channel="Cell_1",
        detected_by=["hampel"],
        features={
            "n_points": 11.0,
            "amplitude_in_sigma": 3.0,
            "level_shift_in_sigma": 1.5,
            "recovers_to_baseline": 0.0,
            "unique_value_fraction": 1.0,
            "region_variance_ratio": 4.0,
        },
    )
    assert classify_deterministic(record) is None
