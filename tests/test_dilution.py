"""The ACD concentration trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.config import DilutionConfig
from polyrmc.tier0.dilution import (
    aggregation_index,
    concentration_at,
    dilution_time_constant,
    time_for_concentration,
)


def test_concentration_decays_exponentially_from_c0(dilution):
    time_s = np.array([0.0, 1000.0, 2000.0])
    concentration = concentration_at(time_s, dilution)

    assert concentration[0] == pytest.approx(dilution.c0_g_per_cm3)
    assert concentration[1] == pytest.approx(dilution.c0_g_per_cm3 / np.e)
    assert concentration[2] == pytest.approx(dilution.c0_g_per_cm3 / np.e**2)


def test_time_constant_is_volume_over_flow_rate(dilution):
    assert dilution_time_constant(dilution) == pytest.approx(1000.0)


def test_time_for_concentration_inverts_the_trajectory(dilution):
    target = dilution.c0_g_per_cm3 * 0.37
    elapsed = time_for_concentration(target, dilution)

    assert concentration_at(np.array([elapsed]), dilution)[0] == pytest.approx(target)


def test_concentration_above_c0_is_refused(dilution):
    with pytest.raises(ValueError, match="exceeds the initial value"):
        time_for_concentration(dilution.c0_g_per_cm3 * 2, dilution)


def test_aggregation_index_is_normalized_to_the_run_start():
    signal = np.concatenate([np.full(100, 500.0), np.full(100, 1500.0)])
    index = aggregation_index(signal, reference_points=50)

    assert index[0] == pytest.approx(1.0)
    assert index[-1] == pytest.approx(3.0)


def test_aggregation_index_is_invariant_to_the_instrument_scale():
    """Mw/M0 is a ratio within one file, so alpha cancels."""
    signal = np.concatenate([np.full(100, 500.0), np.full(100, 1500.0)])

    np.testing.assert_allclose(
        aggregation_index(signal), aggregation_index(signal * 7.3e4)
    )


def test_dilution_config_rejects_a_non_positive_flow_rate():
    with pytest.raises(ValueError):
        DilutionConfig(c0_g_per_cm3=1e-3, flow_rate_cm3_per_s=0.0)
