"""Declaration-versus-data checks: raise the conflict, never resolve it."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_spectranalytics.tier0.consistency import (
    check_declarations,
    dominant_period_s,
    fit_exponential_decay,
)


@pytest.fixture
def time_s() -> np.ndarray:
    return np.arange(0.0, 4000.0, 1.0)


def decaying(time_s: np.ndarray, rate: float = 5e-4, noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    signal = 0.5 * np.exp(-rate * time_s)
    return signal + noise * rng.standard_normal(time_s.size)


def rising(time_s: np.ndarray) -> np.ndarray:
    return 0.18 + 0.42 / (1.0 + np.exp(-(time_s - 2000.0) / 300.0))


def test_fit_recovers_a_known_decay_rate(time_s):
    fit = fit_exponential_decay(time_s, decaying(time_s, rate=5e-4))
    assert fit is not None
    assert fit.rate_per_s == pytest.approx(5e-4, rel=1e-6)
    assert fit.r_squared > 0.99


def test_fit_refuses_a_trace_with_no_positive_samples(time_s):
    assert fit_exponential_decay(time_s, np.full(time_s.size, -1.0)) is None


def test_rising_trace_conflicts_with_a_declared_dilution(time_s):
    report = check_declarations(
        time_s,
        rising(time_s),
        {"experiment_type": "acd_dilution", "cell_volume_cm3": 2.0,
         "flow_rate_cm3_per_s": 0.001},
    )
    conflict = next(c for c in report.conflicts if c.check == "experiment_type_vs_trace")
    assert conflict.declared == "acd_dilution"
    assert "rises" in conflict.observed
    # The declaration stands: code reports, it does not overrule.
    assert not conflict.resolved
    assert "aggregates faster than it dilutes" in conflict.detail


def test_decay_at_the_declared_rate_passes(time_s):
    report = check_declarations(
        time_s,
        decaying(time_s, rate=5e-4),
        {"experiment_type": "acd_dilution", "cell_volume_cm3": 2.0,
         "flow_rate_cm3_per_s": 0.001},
    )
    assert not report.conflicts
    assert any("experiment_type" in line for line in report.passed)


def test_decay_at_the_wrong_rate_conflicts(time_s):
    """V = 1 instead of 2 doubles Q/V, and the trace no longer matches."""
    report = check_declarations(
        time_s,
        decaying(time_s, rate=5e-4),
        {"experiment_type": "acd_dilution", "cell_volume_cm3": 1.0,
         "flow_rate_cm3_per_s": 0.001},
    )
    conflict = next(c for c in report.conflicts if c.check == "experiment_type_vs_trace")
    assert "Q or V is wrong" in conflict.detail


def test_dilution_signature_conflicts_with_a_declared_fixed_concentration(time_s):
    report = check_declarations(
        time_s,
        decaying(time_s, rate=5e-4),
        {"experiment_type": "fixed_conc_aggregation"},
    )
    conflict = next(c for c in report.conflicts if c.check == "experiment_type_vs_trace")
    assert conflict.declared == "fixed_conc_aggregation"


def test_rising_trace_agrees_with_a_declared_aggregation_run(time_s):
    report = check_declarations(
        time_s, rising(time_s), {"experiment_type": "fixed_conc_aggregation"}
    )
    assert not report.conflicts


def test_an_undeclared_experiment_type_is_skipped_not_passed(time_s):
    report = check_declarations(time_s, rising(time_s), {})
    assert any("experiment_type" in line for line in report.skipped)
    assert not any("experiment_type" in line for line in report.passed)


def test_missing_q_and_v_is_reported_as_a_skip(time_s):
    report = check_declarations(
        time_s, decaying(time_s), {"experiment_type": "acd_dilution"}
    )
    assert any("could not be compared against Q/V" in line for line in report.skipped)


def test_wavelength_mismatch_conflicts(time_s):
    report = check_declarations(
        time_s,
        rising(time_s),
        {"wavelength_nm": 660.0},
        metadata={"Wavelength": "640 nm"},
    )
    conflict = next(c for c in report.conflicts if c.check == "instrument_vs_header")
    assert "640" in conflict.observed


def test_c0_mismatch_conflicts_beyond_tolerance(time_s):
    report = check_declarations(
        time_s,
        rising(time_s),
        {"c0_g_per_cm3": 2.0e-3},
        metadata={"Sample Concentration": "1.0 mg/mL"},
    )
    conflict = next(c for c in report.conflicts if c.check == "c0_vs_header")
    assert "100.0 percent" in conflict.detail


def test_c0_within_tolerance_passes(time_s):
    report = check_declarations(
        time_s,
        rising(time_s),
        {"c0_g_per_cm3": 1.02e-3},
        metadata={"Sample Concentration": "1.0 mg/mL"},
    )
    assert not any(c.check == "c0_vs_header" for c in report.conflicts)


def test_c0_with_unreadable_header_units_is_skipped(time_s):
    report = check_declarations(
        time_s,
        rising(time_s),
        {"c0_g_per_cm3": 1.0e-3},
        metadata={"Sample Concentration": "1.0"},
    )
    assert any("units could not be identified" in line for line in report.skipped)
    assert not any(c.check == "c0_vs_header" for c in report.conflicts)


def test_stop_flow_cycling_is_found_in_the_stir_column(time_s):
    # 20 s stirring, then 100 s quiescent: a 120 s cycle.
    cycle = (time_s % 120.0) < 20.0
    period = dominant_period_s(time_s, cycle.astype(float))
    assert period == pytest.approx(120.0, abs=2.0)


def test_a_continuous_trace_has_no_cycle(time_s):
    assert dominant_period_s(time_s, decaying(time_s, noise=0.001)) is None


def test_cycling_on_a_run_declared_sls_conflicts(time_s):
    stir = ((time_s % 120.0) < 20.0).astype(float)
    report = check_declarations(
        time_s, rising(time_s), {"technique": "SLS"}, stir_speed=stir
    )
    conflict = next(c for c in report.conflicts if c.check == "technique_vs_cycling")
    assert "120 s" in conflict.observed


def test_no_cycling_on_a_run_declared_dls_conflicts(time_s):
    report = check_declarations(
        time_s, decaying(time_s, noise=0.001), {"technique": "DLS"}
    )
    conflict = next(c for c in report.conflicts if c.check == "technique_vs_cycling")
    assert conflict.declared == "DLS"


def test_conflicts_start_unresolved_and_stay_that_way(time_s):
    report = check_declarations(
        time_s, rising(time_s), {"experiment_type": "acd_dilution"}
    )
    assert report.conflicts
    assert report.unresolved == report.conflicts
