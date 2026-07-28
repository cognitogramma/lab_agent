"""Fit-range selection: the decision that has no audit trail today."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.config import FitRangeConfig, LoopConfig
from polyrmc.tier0.fit_range import (
    apply_fit_range,
    conservative_fit_range,
    evaluate_range,
    lag1_autocorrelation,
    propose_fit_ranges,
    residual_runs_z,
    sorted_finite,
)
from polyrmc.tier0.optics import optical_constant
from polyrmc.tier0.virial import build_kc_over_i, fit_virial
from polyrmc.tier1.fit_range_selector import select_fit_range
from polyrmc.tier1.judge import StaticJudge

from conftest import OVERLAP_C, TRUE_A2, TRUE_MW, ideal_intensity

TRUE_A2_MW = TRUE_A2 * TRUE_MW


@pytest.fixture
def curved_dataset(optical, dilution, rng):
    """A trajectory that is linear at low c and bends once chains overlap."""
    k = optical_constant(optical)
    time_s = np.arange(0, 4000, 1.0)
    decay = dilution.flow_rate_cm3_per_s / dilution.cell_volume_cm3
    concentration = dilution.c0_g_per_cm3 * np.exp(-decay * time_s)
    intensity = ideal_intensity(concentration, k)
    intensity = intensity * (1.0 + rng.normal(0, 2e-4, time_s.size))
    return concentration, build_kc_over_i(concentration, intensity, k)


def test_sorted_finite_orders_by_ascending_concentration(curved_dataset):
    concentration, ordinate = curved_dataset
    c, _y = sorted_finite(concentration, ordinate)
    assert np.all(np.diff(c) >= 0)


def test_runs_statistic_detects_systematic_curvature():
    curved = np.concatenate([np.ones(50), -np.ones(50), np.ones(50)])
    assert residual_runs_z(curved) < -5

    alternating = np.array([1.0, -1.0] * 75)
    assert residual_runs_z(alternating) > 5


def test_lag1_autocorrelation_separates_white_from_structured(rng):
    white = rng.normal(0, 1.0, 1000)
    structured = np.cumsum(rng.normal(0, 1.0, 1000))

    assert abs(lag1_autocorrelation(white)) < 0.1
    assert lag1_autocorrelation(structured) > 0.9


def test_evaluate_range_flags_curvature_when_the_range_runs_too_far(curved_dataset):
    concentration, ordinate = curved_dataset
    c, y = sorted_finite(concentration, ordinate)

    inside = evaluate_range(c, y, upper=int((c <= OVERLAP_C).sum()))
    beyond = evaluate_range(c, y, upper=c.size)

    assert beyond["curvature_t"] > inside["curvature_t"]


def test_every_proposed_range_stays_within_the_linear_region(curved_dataset):
    concentration, ordinate = curved_dataset
    candidates = propose_fit_ranges(concentration, ordinate, FitRangeConfig(min_points=30))

    assert candidates, "the low-concentration region is linear and must be offered"
    for candidate in candidates:
        assert candidate.features["c_max"] < OVERLAP_C * 1.6, (
            "a candidate reaching well past the overlap concentration should have "
            "failed the curvature or residual gate"
        )


def test_proposed_ranges_recover_the_known_a2_mw(curved_dataset):
    concentration, ordinate = curved_dataset
    candidates = propose_fit_ranges(concentration, ordinate, FitRangeConfig(min_points=30))

    for candidate in candidates:
        assert candidate.features["a2_mw"] == pytest.approx(TRUE_A2_MW, rel=0.05)


def test_candidates_record_what_was_excluded(curved_dataset):
    concentration, ordinate = curved_dataset
    candidates = propose_fit_ranges(concentration, ordinate, FitRangeConfig(min_points=30))

    for candidate in candidates:
        assert "n_excluded" in candidate.features
        assert candidate.features["n_excluded"] >= 0
        assert "excluded" in candidate.label


def test_conservative_choice_is_the_narrowest_range(curved_dataset):
    concentration, ordinate = curved_dataset
    candidates = propose_fit_ranges(concentration, ordinate, FitRangeConfig(min_points=30))

    chosen = conservative_fit_range(candidates)
    assert chosen.features["c_max"] == min(c.features["c_max"] for c in candidates)


def test_apply_fit_range_restricts_to_the_selected_region(curved_dataset):
    concentration, ordinate = curved_dataset
    c_fit, y_fit = apply_fit_range(concentration, ordinate, c_max=1e-3)

    assert c_fit.max() <= 1e-3
    assert c_fit.size == y_fit.size
    assert c_fit.size > 0


def test_selected_range_reproduces_a2_mw_end_to_end(curved_dataset):
    concentration, ordinate = curved_dataset

    loop = select_fit_range(
        concentration,
        ordinate,
        fit_config=FitRangeConfig(min_points=30),
        loop_config=LoopConfig(max_iterations=2),
        judge=StaticJudge(),
    )
    c_fit, y_fit = apply_fit_range(concentration, ordinate, float(loop.resolved_value))
    fit = fit_virial(c_fit, y_fit, order=1)

    assert fit.a2_mw == pytest.approx(TRUE_A2_MW, rel=0.05)
    assert loop.resolved_value is not None


def test_loop_records_rejected_candidates_alongside_the_winner(curved_dataset):
    concentration, ordinate = curved_dataset

    loop = select_fit_range(
        concentration,
        ordinate,
        fit_config=FitRangeConfig(min_points=30),
        judge=StaticJudge(),
    )

    assert loop.decisions, "the trajectory must be recorded"
    decision = loop.decisions[0]
    offered = len(loop.candidates_by_iteration[0])
    assert len(decision.rejected_indices) == offered - 1, (
        "which points were excluded from the linear region is the scientific claim"
    )


def test_too_few_points_yields_no_candidates_rather_than_a_bad_fit():
    concentration = np.linspace(1e-4, 1e-3, 10)
    ordinate = 1.0 / TRUE_MW + 2.0 * TRUE_A2 * concentration

    assert propose_fit_ranges(concentration, ordinate, FitRangeConfig(min_points=30)) == []


def test_conservative_fit_range_refuses_an_empty_candidate_set():
    with pytest.raises(ValueError, match="no fit-range candidates"):
        conservative_fit_range([])
