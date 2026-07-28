"""The virial fit, and the calibration-free invariant the pipeline rests on."""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.state import FitResult
from polyrmc.tier0.optics import (
    alpha_from_known_mw,
    alpha_from_toluene,
    optical_constant,
    rayleigh_ratio,
)
from polyrmc.tier0.virial import build_kc_over_i, derive_parameters, fit_virial

from conftest import TRUE_A2, TRUE_MW


def _linear_curve(k: float, alpha: float = 1.0, n: int = 400):
    """Kc/I = 1/Mw + 2*A2*c exactly, with intensity scaled by alpha."""
    concentration = np.linspace(1e-4, 1.5e-3, n)
    ordinate = 1.0 / TRUE_MW + 2.0 * TRUE_A2 * concentration
    intensity = alpha * k * concentration / ordinate
    return concentration, intensity


def test_optical_constant_has_the_expected_magnitude(optical):
    k = optical_constant(optical)
    # For water, dn/dc ~ 0.185, 660 nm: K lands around 1e-7 mol cm^2 / g^2.
    assert 1e-8 < k < 1e-6


def test_fit_recovers_mw_and_a2_from_a_clean_curve(optical):
    k = optical_constant(optical)
    concentration, intensity = _linear_curve(k)
    ordinate = build_kc_over_i(concentration, intensity, k)

    fit = fit_virial(concentration, ordinate, order=1)

    assert fit.r_squared > 0.999999
    assert abs(1.0 / fit.intercept - TRUE_MW) / TRUE_MW < 1e-6
    assert abs(fit.slope / 2.0 - TRUE_A2) / TRUE_A2 < 1e-6


@pytest.mark.parametrize("alpha", [1.0, 3.7e4, 1e-3])
def test_a2_mw_is_invariant_to_the_instrument_scale_factor(optical, alpha):
    """slope/intercept = 2*A2*Mw, so alpha cancels.

    This is what lets the pipeline be built and validated before the toluene
    reference is measured.
    """
    k = optical_constant(optical)
    concentration, intensity = _linear_curve(k, alpha=alpha)
    ordinate = build_kc_over_i(concentration, intensity, k)

    fit = fit_virial(concentration, ordinate, order=1)

    assert abs(fit.a2_mw - TRUE_A2 * TRUE_MW) / (TRUE_A2 * TRUE_MW) < 1e-6


def test_uncalibrated_runs_report_a2_mw_but_withhold_mw_and_a2(optical):
    k = optical_constant(optical)
    concentration, intensity = _linear_curve(k, alpha=5000.0)
    ordinate = build_kc_over_i(concentration, intensity, k)
    fit = fit_virial(concentration, ordinate, order=1)

    parameters = derive_parameters(fit, calibrated=False)

    assert parameters.a2_mw == pytest.approx(TRUE_A2 * TRUE_MW, rel=1e-6)
    assert parameters.mw is None, "Mw would silently carry a factor of alpha"
    assert parameters.a2 is None
    assert parameters.k_d == pytest.approx(2.0 * parameters.a2_mw)


def test_calibrated_runs_report_absolute_mw(optical):
    k = optical_constant(optical)
    concentration, intensity = _linear_curve(k, alpha=1.0)
    ordinate = build_kc_over_i(concentration, intensity, k)
    fit = fit_virial(concentration, ordinate, order=1)

    parameters = derive_parameters(fit, calibrated=True)

    assert parameters.mw == pytest.approx(TRUE_MW, rel=1e-6)
    assert parameters.a2 == pytest.approx(TRUE_A2, rel=1e-6)


def test_k_d_includes_the_hydrodynamic_term():
    fit = FitResult(
        slope=2.0,
        intercept=0.5,
        slope_stderr=0.0,
        intercept_stderr=0.0,
        r_squared=1.0,
        n_points=10,
        c_min=0.0,
        c_max=1.0,
    )
    parameters = derive_parameters(fit, calibrated=False, k_h=1.25)

    assert fit.a2_mw == pytest.approx(2.0)
    assert parameters.k_d == pytest.approx(2.0 * 2.0 + 1.25)


def test_alpha_back_solved_from_a_known_standard_matches_the_toluene_route(optical):
    """The two routes agreeing is evidence for the constant-alpha assumption."""
    k = optical_constant(optical)
    alpha = 4.2e4
    concentration, intensity = _linear_curve(k, alpha=alpha)
    ordinate = build_kc_over_i(concentration, intensity, k)
    fit = fit_virial(concentration, ordinate, order=1)

    recovered = alpha_from_known_mw(fit.intercept, TRUE_MW)
    assert recovered == pytest.approx(alpha, rel=1e-5)


def test_toluene_calibration_requires_a_tabulated_wavelength():
    assert alpha_from_toluene(1.0e4, wavelength_nm=660) > 0
    with pytest.raises(ValueError, match="no tabulated toluene"):
        alpha_from_toluene(1.0e4, wavelength_nm=488)


def test_rayleigh_ratio_passes_counts_through_when_uncalibrated():
    counts = np.array([1.0, 2.0, 3.0])
    values, calibrated = rayleigh_ratio(counts, alpha=None)

    assert not calibrated
    np.testing.assert_allclose(values, counts)


def test_non_positive_intensity_becomes_nan_rather_than_a_negative_ordinate():
    concentration = np.array([1e-4, 2e-4, 3e-4])
    intensity = np.array([100.0, 0.0, -5.0])

    ordinate = build_kc_over_i(concentration, intensity, 1e-7)

    assert np.isfinite(ordinate[0])
    assert np.isnan(ordinate[1]) and np.isnan(ordinate[2])
