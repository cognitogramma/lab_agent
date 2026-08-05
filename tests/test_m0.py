"""M0: an averaged window where one exists, a reported finding where it does not."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.pipeline import run_part1
from polyrmc.tier0.m0 import select_m0
from polyrmc.tier1.judge import StaticJudge

RNG = np.random.default_rng(11)
NOISE = 0.002


@pytest.fixture
def time_s() -> np.ndarray:
    return np.arange(0.0, 4000.0, 1.0)


@pytest.fixture
def lagged(time_s) -> np.ndarray:
    """Flat for a while, then a rise -- a trace with a real lag phase."""
    return 0.20 + 0.60 / (1.0 + np.exp(-(time_s - 2000.0) / 200.0)) + RNG.normal(
        0, NOISE, time_s.size
    )


def test_a_flat_opening_yields_an_averaged_window(time_s, lagged):
    result = select_m0(time_s, lagged, noise_floor=NOISE)

    assert result.adaptive
    assert result.index_range[0] == 0
    assert result.n_points > 1
    assert result.value == pytest.approx(0.20, abs=0.01)


def test_the_window_is_capped_as_a_fraction_of_the_run(time_s, lagged):
    result = select_m0(time_s, lagged, noise_floor=NOISE, max_window_fraction=0.02)
    assert result.n_points <= 0.02 * time_s.size + 1


def test_averaging_beats_the_first_point(time_s):
    """The whole reason for the adaptive window: noise on M0 falls as sqrt(N).

    Compared over the runs where a window was actually found -- where it was
    not, the two definitions are the same value by construction and comparing
    them measures nothing.
    """
    errors_window = []
    errors_point = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        trace = 0.20 + 0.60 / (1.0 + np.exp(-(time_s - 2000.0) / 200.0)) + rng.normal(
            0, NOISE, time_s.size
        )
        result = select_m0(time_s, trace, noise_floor=NOISE)
        if not result.adaptive:
            continue
        errors_window.append(abs(result.value - 0.20))
        errors_point.append(abs(result.first_point_value - 0.20))

    assert len(errors_window) >= 10, "too few adaptive windows to compare"
    assert np.mean(errors_window) < np.mean(errors_point) / 2


def test_a_trace_already_moving_at_t0_falls_back_and_says_so(time_s):
    """Acquisition began after the process started. That is a finding."""
    rising = 0.20 + 0.60 * time_s / time_s.max() + RNG.normal(0, NOISE, time_s.size)
    result = select_m0(time_s, rising, noise_floor=NOISE)

    assert not result.adaptive
    assert result.method == "first_point"
    assert result.n_points == 1
    assert any("no flat initial region exists" in b for b in result.blocked)
    assert any("finding about the acquisition" in b for b in result.blocked)


def test_the_window_never_reaches_into_the_transition(time_s):
    """A window that reaches the rise measures the rise, not the level."""
    early = 0.20 + 0.60 / (1.0 + np.exp(-(time_s - 60.0) / 20.0)) + RNG.normal(
        0, NOISE, time_s.size
    )
    result = select_m0(time_s, early, noise_floor=NOISE, transition=(40, 200))
    if result.adaptive:
        assert result.index_range[1] < 40


def test_excluded_samples_stay_out_of_the_window(time_s, lagged):
    excluded = np.zeros(time_s.size, dtype=bool)
    excluded[20:] = True
    result = select_m0(time_s, lagged, noise_floor=NOISE, excluded=excluded)
    if result.adaptive:
        assert result.index_range[1] < 20


def test_sensitivity_is_reported_whether_or_not_it_escalates(time_s, lagged):
    result = select_m0(time_s, lagged, noise_floor=NOISE)
    assert np.isfinite(result.sensitivity)
    assert abs(result.sensitivity) < 0.05
    assert result.escalations == []


def test_a_disturbed_opening_falls_back_rather_than_diverging(time_s):
    """A first sample 10% low disqualifies the window outright.

    Documents a real interaction between two tolerances: the flatness gate that
    admits a window is tighter than the sensitivity ceiling that would flag one.
    Anything large enough to move the fold change by 5 percent has already failed
    the gate, so on these defaults the divergence is caught by the fallback and
    the sensitivity report is a second line of defence rather than the first.
    Both tolerances want calibrating against the golden runs.
    """
    trace = 0.20 + 0.60 / (1.0 + np.exp(-(time_s - 2000.0) / 200.0)) + RNG.normal(
        0, NOISE, time_s.size
    )
    trace[0] = 0.18

    result = select_m0(time_s, trace, noise_floor=NOISE, sensitivity_ceiling=0.05)

    assert not result.adaptive
    assert any("no flat initial region exists" in b for b in result.blocked)


def test_the_escalation_fires_once_the_ceiling_is_reachable(time_s, lagged):
    """The reporting logic itself, exercised at a ceiling the gate can reach."""
    result = select_m0(time_s, lagged, noise_floor=NOISE, sensitivity_ceiling=1e-9)

    assert result.adaptive
    assert result.escalations
    assert any("diverge" in e for e in result.escalations)


def test_an_all_nan_trace_is_refused(time_s):
    with pytest.raises(ValueError, match="no finite samples"):
        select_m0(time_s, np.full(time_s.size, np.nan), noise_floor=NOISE)


def test_the_pipeline_records_which_definition_it_used(run_config, tmp_path):
    state, csv_path = run_part1(
        run_config, judge=StaticJudge(), output_csv=tmp_path / "m.csv"
    )
    assert state.m0 is not None
    assert state.m0.method in {"adaptive_window", "first_point"}

    header = csv_path.read_text(encoding="utf-8")
    assert "m0_method:" in header
    assert "m0_sensitivity:" in header
