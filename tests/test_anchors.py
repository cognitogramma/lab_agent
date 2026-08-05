"""Baseline correction: the guards, not just the arithmetic.

The failure this step exists to prevent is subtracting a curve fitted to anchors
that were really part of the kinetics. The result is smooth, plausible, and
physically wrong, and nothing downstream detects it -- so the tests that matter
here are the ones that check a bad selection is *refused*.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.config import BaselineConfig
from polyrmc.state import AnchorCandidate
from polyrmc.tier0.anchors import (
    apply_correction,
    evaluate_drift,
    fit_drift,
    propose_anchors,
    revert_correction,
    total_fold_change,
    validate_selection,
)
from polyrmc.tier0.landmarks import lag_slope, transition_region

RNG = np.random.default_rng(20260805)
NOISE = 0.002


@pytest.fixture
def time_s() -> np.ndarray:
    return np.arange(0.0, 4000.0, 1.0)


@pytest.fixture
def sigmoid(time_s) -> np.ndarray:
    """Flat, a transition in the middle, then flat again."""
    return 0.20 + 0.60 / (1.0 + np.exp(-(time_s - 2000.0) / 120.0))


@pytest.fixture
def drifting(time_s, sigmoid) -> np.ndarray:
    """The same kinetics riding on a slow linear drift, plus noise."""
    return sigmoid + 2.0e-5 * time_s + RNG.normal(0, NOISE, time_s.size)


# --- the transition region, which is the load-bearing guard ------------------


def test_the_transition_region_lands_on_the_kinetics(time_s, sigmoid):
    region = transition_region(time_s, sigmoid)
    assert region is not None
    low, high = region
    assert low < 2000 < high
    # It is the transition, not the whole run.
    assert (high - low) < time_s.size // 2


def test_a_flat_trace_has_no_meaningful_transition(time_s):
    flat = np.full(time_s.size, 0.5) + RNG.normal(0, NOISE, time_s.size)
    region = transition_region(time_s, flat)
    # Noise alone has no landmark, so whatever comes back must not swallow the run.
    assert region is None or (region[1] - region[0]) < time_s.size // 2


# --- candidate generation ----------------------------------------------------


def test_candidates_avoid_the_transition_entirely(time_s, drifting):
    candidates, region = propose_anchors(time_s, drifting, noise_floor=NOISE)

    assert candidates
    assert region is not None
    for anchor in candidates:
        low, high = anchor.index_range
        assert high < region[0] or low > region[1]


def test_candidates_appear_on_both_sides_of_the_transition(time_s, drifting):
    candidates, region = propose_anchors(time_s, drifting, noise_floor=NOISE)
    assert any(a.index_range[1] < region[0] for a in candidates)
    assert any(a.index_range[0] > region[1] for a in candidates)


def test_excluded_samples_never_enter_a_candidate(time_s, drifting):
    excluded = np.zeros(time_s.size, dtype=bool)
    excluded[100:400] = True
    candidates, _region = propose_anchors(
        time_s, drifting, noise_floor=NOISE, excluded=excluded
    )
    for anchor in candidates:
        low, high = anchor.index_range
        assert not excluded[low : high + 1].any()


def test_short_regions_are_not_offered(time_s, drifting):
    config = BaselineConfig(min_anchor_duration=0.30)
    candidates, _region = propose_anchors(
        time_s, drifting, noise_floor=NOISE, config=config
    )
    total = float(time_s[-1] - time_s[0])
    for anchor in candidates:
        assert anchor.duration_fraction >= 0.30 - 1e-9
        assert anchor.duration_fraction <= 1.0
        assert 0.0 <= anchor.position_fraction <= 1.0
        assert total > 0


def test_a_noisy_stretch_is_not_a_candidate(time_s, sigmoid):
    signal = sigmoid + RNG.normal(0, NOISE, time_s.size)
    signal[300:600] += RNG.normal(0, NOISE * 20, 300)
    candidates, _region = propose_anchors(time_s, signal, noise_floor=NOISE)

    for anchor in candidates:
        low, high = anchor.index_range
        assert not (low < 600 and high > 300), "a high-variance stretch was offered"


# --- validation --------------------------------------------------------------


def _anchor(id_: int, low: int, high: int, level: float = 0.2) -> AnchorCandidate:
    return AnchorCandidate(
        id=id_,
        index_range=(low, high),
        duration_fraction=(high - low) / 4000.0,
        mean_level=level,
        local_slope=0.0,
        local_variance=NOISE**2,
        position_fraction=(low + high) / 2 / 4000.0,
    )


def test_a_single_anchor_is_refused(time_s):
    check = validate_selection([_anchor(0, 0, 800)], time_s)
    assert not check.valid
    assert any("cannot define a trend" in r for r in check.rejections)


def test_thin_coverage_is_refused(time_s):
    selected = [_anchor(0, 0, 100), _anchor(1, 3800, 3900)]
    check = validate_selection(selected, time_s)
    assert not check.valid
    assert any("percent of the run" in r for r in check.rejections)


def test_an_anchor_inside_the_transition_is_refused(time_s):
    selected = [_anchor(0, 0, 800), _anchor(1, 1900, 2600)]
    check = validate_selection(selected, time_s, transition=(1800, 2200))
    assert not check.valid
    assert any("intersects the transition region" in r for r in check.rejections)


def test_an_anchor_over_excised_samples_is_refused(time_s):
    excluded = np.zeros(time_s.size, dtype=bool)
    excluded[500:520] = True
    selected = [_anchor(0, 0, 800), _anchor(1, 3000, 3900)]
    check = validate_selection(selected, time_s, excluded=excluded)
    assert not check.valid
    assert any("excised or flagged" in r for r in check.rejections)


def test_a_one_sided_selection_needs_confirmation_but_is_not_refused(time_s):
    selected = [_anchor(0, 0, 700), _anchor(1, 800, 1500)]
    check = validate_selection(selected, time_s, transition=(1800, 2200))

    assert check.valid
    assert any("no lever arm" in c for c in check.confirmations)


def test_a_two_sided_selection_needs_no_confirmation(time_s):
    selected = [_anchor(0, 0, 700), _anchor(1, 3000, 3900)]
    check = validate_selection(selected, time_s, transition=(1800, 2200))

    assert check.valid
    assert check.confirmations == []


# --- fitting, and the two ceilings that decide offerability ------------------


def test_a_drift_model_that_follows_the_kinetics_is_refused(time_s, sigmoid):
    """The check that makes this step recoverable.

    Anchors placed either side of the transition at very different levels let a
    polynomial track the transition itself. The slope ratio must catch it
    *before* the correction is offered.
    """
    selected = [
        _anchor(0, 0, 900, level=0.20),
        _anchor(1, 1500, 1900, level=0.50),
        _anchor(2, 3000, 3900, level=0.80),
    ]
    drift = fit_drift(time_s, sigmoid, selected)
    check = validate_selection(selected, time_s, drift=drift)

    assert drift.max_slope_ratio > 0.10
    assert not check.valid
    assert any("absorb them" in r for r in check.rejections)


def test_spanning_a_step_with_anchors_either_side_is_refused(time_s, drifting):
    """A step trace does not return to baseline, so 'both sides' is a trap.

    The post-transition plateau is signal, not baseline. A line through one
    anchor either side of it takes its slope from the step rather than the
    drift, and the ceiling is what catches that.
    """
    candidates, region = propose_anchors(time_s, drifting, noise_floor=NOISE)
    selected = [
        next(a for a in candidates if a.index_range[1] < region[0]),
        next(a for a in candidates if a.index_range[0] > region[1]),
    ]
    drift = fit_drift(time_s, drifting, selected)

    assert drift.max_slope_ratio > 0.10
    assert not validate_selection(selected, time_s, drift=drift).valid


def test_a_genuine_slow_drift_is_recovered_from_one_side(time_s, drifting):
    """On a step trace the drift is measurable only where the signal is flat."""
    candidates, region = propose_anchors(time_s, drifting, noise_floor=NOISE)
    before = [a for a in candidates if a.index_range[1] < region[0]]
    assert len(before) >= 2, "need two pre-transition anchors to define a trend"
    selected = before[:2]

    drift = fit_drift(time_s, drifting, selected)
    check = validate_selection(selected, time_s, transition=region, drift=drift)

    # The fitted line should recover the 2e-5 per second drift that was injected.
    fitted_rate = drift.coefficients[0] / drift.time_scale
    assert fitted_rate == pytest.approx(2.0e-5, rel=0.5)

    assert drift.max_slope_ratio < 0.10, drift.max_slope_ratio
    assert abs(drift.fold_change_impact) < 0.20
    assert check.valid, check.rejections
    # One-sided, so it extrapolates across the transition and says so.
    assert any("no lever arm" in c for c in check.confirmations)


def test_two_sided_anchors_work_where_the_signal_returns_to_baseline(time_s):
    """A trace that comes back down has real baseline either side of the peak."""
    peak = 0.20 + 0.60 * np.exp(-(((time_s - 2000.0) / 250.0) ** 2))
    signal = peak + 5.0e-6 * time_s + RNG.normal(0, NOISE, time_s.size)

    candidates, region = propose_anchors(time_s, signal, noise_floor=NOISE)
    selected = [
        next(a for a in candidates if a.index_range[1] < region[0]),
        next(a for a in candidates if a.index_range[0] > region[1]),
    ]
    drift = fit_drift(time_s, signal, selected)
    check = validate_selection(selected, time_s, transition=region, drift=drift)

    assert drift.max_slope_ratio < 0.10, drift.max_slope_ratio
    assert check.valid, check.rejections
    assert check.confirmations == []


def test_a_correction_that_moves_the_conclusion_is_refused(time_s):
    """A drift big enough to change the answer is a human's call, not a default."""
    signal = 1.0 + 0.5 * time_s / time_s.max()
    selected = [_anchor(0, 0, 900, level=1.0), _anchor(1, 3000, 3900, level=1.4)]
    drift = fit_drift(time_s, signal, selected)
    check = validate_selection(selected, time_s, drift=drift)

    assert abs(drift.fold_change_impact) > 0.20
    assert any("moves the scientific conclusion" in r for r in check.rejections)


def test_polynomial_order_never_exceeds_what_the_anchors_constrain(time_s, drifting):
    two = [_anchor(0, 0, 900), _anchor(1, 3000, 3900)]
    assert fit_drift(time_s, drifting, two).order == 1

    three = two + [_anchor(2, 1000, 1400)]
    assert fit_drift(time_s, drifting, three).order == 2


def test_higher_order_models_are_not_reachable(time_s, drifting):
    selected = [_anchor(i, i * 500, i * 500 + 300) for i in range(6)]
    assert fit_drift(time_s, drifting, selected).order == 2


def test_splines_are_refused_by_name(time_s, drifting):
    selected = [_anchor(0, 0, 900), _anchor(1, 3000, 3900)]
    with pytest.raises(ValueError, match="absorb the kinetics"):
        fit_drift(time_s, drifting, selected, kind="spline")


def test_piecewise_linear_passes_through_the_anchor_levels(time_s, drifting):
    selected = [
        _anchor(0, 0, 900, level=0.21),
        _anchor(1, 3000, 3900, level=0.87),
    ]
    drift = fit_drift(time_s, drifting, selected, kind="piecewise_linear")
    curve = evaluate_drift(drift, time_s)

    assert drift.order is None
    assert drift.reference_level == pytest.approx(0.21, abs=1e-9)
    # The correction is referenced to the start, so it removes the 0.66 rise
    # between the anchors without removing the 0.21 level the sample sits at.
    assert curve[450] == pytest.approx(0.0, abs=1e-9)
    assert curve[3450] == pytest.approx(0.66, abs=1e-9)
    # Held flat outside the anchor span rather than extrapolated.
    assert curve[0] == pytest.approx(0.0, abs=1e-9)
    assert curve[-1] == pytest.approx(0.66, abs=1e-9)


def test_fitting_alone_changes_nothing(time_s, drifting):
    before = drifting.copy()
    fit_drift(time_s, drifting, [_anchor(0, 0, 900), _anchor(1, 3000, 3900)])
    assert np.array_equal(drifting, before)


# --- reversibility -----------------------------------------------------------


@pytest.mark.parametrize("kind", ["poly", "piecewise_linear"])
def test_a_correction_reverts_exactly(time_s, drifting, kind):
    selected = [_anchor(0, 0, 900, level=0.21), _anchor(1, 3000, 3900, level=0.87)]
    drift = fit_drift(time_s, drifting, selected, kind=kind)

    curve = evaluate_drift(drift, time_s)
    corrected = apply_correction(drifting, curve)
    recovered = revert_correction(corrected, curve)

    assert np.allclose(recovered, drifting, rtol=0, atol=1e-12)


def test_the_drift_curve_regenerates_from_the_record_alone(time_s, drifting):
    selected = [_anchor(0, 0, 900), _anchor(1, 3000, 3900)]
    drift = fit_drift(time_s, drifting, selected)

    # Round-trip through serialization, as the sidecar does.
    from polyrmc.state import DriftModel

    restored = DriftModel.model_validate_json(drift.model_dump_json())
    assert np.allclose(evaluate_drift(restored, time_s), evaluate_drift(drift, time_s))


def test_correction_preserves_length_and_nans(time_s, drifting):
    signal = drifting.copy()
    signal[100:110] = np.nan
    selected = [_anchor(0, 200, 900), _anchor(1, 3000, 3900)]
    curve = evaluate_drift(fit_drift(time_s, signal, selected), time_s)
    corrected = apply_correction(signal, curve)

    assert corrected.size == signal.size
    assert np.isnan(corrected[100:110]).all()


def test_fitting_with_no_anchors_is_refused(time_s, drifting):
    with pytest.raises(ValueError, match="no anchors selected"):
        fit_drift(time_s, drifting, [])


# --- supporting measurements -------------------------------------------------


def test_total_fold_change(time_s, sigmoid):
    assert total_fold_change(sigmoid) == pytest.approx(0.80 / 0.20, rel=0.05)


def test_lag_slope_beats_noise_where_a_pointwise_gradient_would_not(time_s):
    """The reason section 6.7.1 exists at all.

    Uses a transition on the timescale real runs actually show -- the reference
    dataset moves over hours, not minutes. The slower the real kinetics, the
    worse a pointwise gradient does, which is exactly the regime this pipeline
    operates in.
    """
    slow = 0.20 + 0.60 / (1.0 + np.exp(-(time_s - 2000.0) / 600.0))
    noisy = slow + RNG.normal(0, NOISE, time_s.size)

    pointwise = np.percentile(np.abs(np.gradient(noisy, time_s)), 99.5)
    lagged = lag_slope(time_s, noisy)
    truth = float(np.max(np.abs(np.diff(slow) / np.diff(time_s))))

    assert pointwise > 10 * truth, "pointwise should be swamped by noise"
    assert lagged == pytest.approx(truth, rel=0.5)
