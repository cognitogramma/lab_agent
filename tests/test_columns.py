"""Every branch of the open questions, including the unanswered one."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.tier0.columns import (
    concentration_column,
    mw_over_m0_column,
    rayleigh_column,
)


@pytest.fixture
def time_s() -> np.ndarray:
    return np.arange(0.0, 1000.0, 1.0)


@pytest.fixture
def smoothed(time_s) -> np.ndarray:
    return 1000.0 + 500.0 * time_s / time_s.max()


# --- OPEN-1: experiment type -------------------------------------------------


def test_undeclared_experiment_type_blocks_rather_than_guesses(time_s):
    result = concentration_column(time_s, experiment_type=None)

    assert not result.available
    assert np.isnan(result.values).all()
    assert result.path.decision == "undeclared"
    assert "concentration" in result.path.blocked
    # The trace rises, which is consistent with both branches. Saying so is the
    # point: this is why it cannot be inferred.
    assert "aggregates faster than it dilutes" in result.path.because


def test_dilution_builds_the_exponential_decay(time_s):
    result = concentration_column(
        time_s,
        experiment_type="acd_dilution",
        c0_g_per_cm3=2e-3,
        flow_rate_cm3_per_s=0.001,
        cell_volume_cm3=2.0,
    )

    assert result.available
    assert result.values[0] == pytest.approx(2e-3)
    assert result.values[-1] == pytest.approx(2e-3 * np.exp(-5e-4 * 999.0))
    assert "Q/V = 0.0005 /s" in result.path.because


def test_fixed_concentration_is_constant_c0_not_missing(time_s):
    """A constant column, not a NaN column: it is the bridge to part 2."""
    result = concentration_column(
        time_s, experiment_type="fixed_conc_aggregation", c0_g_per_cm3=1.5e-3
    )

    assert result.available
    assert np.allclose(result.values, 1.5e-3)
    # One point on the concentration axis has no slope, so there is no A2.
    assert "A2" in result.path.blocked


def test_dilution_without_its_geometry_is_blocked_with_the_missing_names(time_s):
    result = concentration_column(
        time_s, experiment_type="acd_dilution", c0_g_per_cm3=2e-3
    )

    assert not result.available
    assert "Q" in result.path.because and "V" in result.path.because


def test_an_unknown_experiment_type_is_refused(time_s):
    with pytest.raises(ValueError, match="not a known experiment type"):
        concentration_column(time_s, experiment_type="aggregation")


# --- OPEN-2: channel units ---------------------------------------------------


def test_undeclared_units_block_the_rayleigh_ratio(smoothed):
    result = rayleigh_column(smoothed, ls_units=None)

    assert not result.available
    assert result.path.decision == "undeclared"
    assert "I_rayleigh" in result.path.blocked


def test_a_normalized_channel_can_never_be_calibrated(smoothed):
    result = rayleigh_column(smoothed, ls_units="normalized")

    assert not result.available
    assert "discarded by the instrument" in result.path.because


def test_a_count_rate_converts_through_the_toluene_reference(smoothed):
    result = rayleigh_column(
        smoothed,
        ls_units="count_rate",
        buffer_blank=100.0,
        toluene_ref=5000.0,
        r_toluene_cm_inv=1.19e-5,
    )

    assert result.available
    assert result.values[0] == pytest.approx((1000.0 - 100.0) / 5000.0 * 1.19e-5)


def test_alpha_is_accepted_as_the_same_quantity(smoothed):
    """alpha = T/R_toluene, so both spellings must give the same column."""
    by_toluene = rayleigh_column(
        smoothed,
        ls_units="count_rate",
        buffer_blank=0.0,
        toluene_ref=5000.0,
        r_toluene_cm_inv=1.19e-5,
    )
    by_alpha = rayleigh_column(
        smoothed, ls_units="count_rate", buffer_blank=0.0, alpha=5000.0 / 1.19e-5
    )

    assert np.allclose(by_toluene.values, by_alpha.values)
    assert by_alpha.path.decision == "count_rate (alpha)"


def test_a_count_rate_without_a_reference_is_blocked(smoothed):
    result = rayleigh_column(smoothed, ls_units="count_rate", buffer_blank=0.0)

    assert not result.available
    assert "toluene_ref" in result.path.because


def test_an_unknown_unit_declaration_is_refused(smoothed):
    with pytest.raises(ValueError, match="not a known unit declaration"):
        rayleigh_column(smoothed, ls_units="counts")


# --- Mw/M0 and its recorded substitution -------------------------------------


def test_mw_over_m0_starts_at_one(smoothed):
    result = mw_over_m0_column(smoothed)
    assert result.values[0] == pytest.approx(1.0, rel=0.05)


def test_falling_back_to_the_smoothed_channel_is_logged(smoothed):
    result = mw_over_m0_column(smoothed, i_rayleigh=np.full(smoothed.size, np.nan))

    assert result.available
    assert any("valid only because the ratio cancels units" in n
               for n in result.path.alternatives)


def test_no_substitution_is_logged_when_i_rayleigh_is_real(smoothed):
    result = mw_over_m0_column(smoothed, i_rayleigh=smoothed * 1e-8)
    assert result.path.alternatives == []


def test_a_varying_concentration_is_divided_out(time_s, smoothed):
    """Otherwise dilution itself reads as disaggregation."""
    concentration = 2e-3 * np.exp(-5e-4 * time_s)
    result = mw_over_m0_column(smoothed, concentration=concentration)

    assert "concentration" in result.path.decision
    # The signal rises 1.5x while c falls; dividing out c must raise the ratio
    # above the raw intensity ratio.
    raw = mw_over_m0_column(smoothed).values
    assert result.values[-1] > raw[-1]


def test_a_constant_concentration_is_not_divided_out(smoothed):
    constant = np.full(smoothed.size, 1.5e-3)
    result = mw_over_m0_column(smoothed, concentration=constant)

    assert "concentration" not in result.path.decision
    assert result.values == pytest.approx(mw_over_m0_column(smoothed).values)


def test_a_non_positive_reference_is_blocked_not_returned_as_a_number(smoothed):
    """Dividing anyway yields a large negative number that looks like a result."""
    result = mw_over_m0_column(smoothed, buffer_blank=1e6)

    assert not result.available
    assert "blank meets or exceeds the sample" in result.path.because


def test_an_all_nan_channel_is_blocked():
    result = mw_over_m0_column(np.full(50, np.nan))
    assert not result.available
    assert "no finite samples" in result.path.because
