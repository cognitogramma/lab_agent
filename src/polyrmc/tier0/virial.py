"""Build Kc/I_R vs c and fit the virial expansion.

.. math:: \\frac{Kc}{I_R} = \\frac{1}{M_w} + 2 A_2 c + 3 A_3 c^2 + \\dots

Fitting is ordinary or weighted least squares over an explicitly chosen
concentration range. *Which* range is a judgment call and is made in
:mod:`polyrmc.tier1.fit_range_selector`; this module only computes.
"""

from __future__ import annotations

import numpy as np

from polyrmc.state import DerivedParameters, FitResult


def build_kc_over_i(
    concentration: np.ndarray, intensity: np.ndarray, optical_constant: float
) -> np.ndarray:
    """Form the Kc/I ordinate, with non-positive intensities masked to NaN."""
    concentration = np.asarray(concentration, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if concentration.shape != intensity.shape:
        raise ValueError("concentration and intensity must have the same shape")

    with np.errstate(divide="ignore", invalid="ignore"):
        ordinate = optical_constant * concentration / intensity
    ordinate[~np.isfinite(ordinate)] = np.nan
    ordinate[intensity <= 0] = np.nan
    return ordinate


def weighted_least_squares(
    x: np.ndarray, y: np.ndarray, order: int = 1, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares polynomial fit returning coefficients, covariance, and R^2.

    Coefficients are in ascending order: ``[intercept, slope, quadratic]``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        finite &= np.isfinite(weights)
    x, y = x[finite], y[finite]
    w = np.ones_like(x) if weights is None else weights[finite]

    n_params = order + 1
    if x.size <= n_params:
        raise ValueError(f"need more than {n_params} points to fit order {order}")

    design = np.vander(x, n_params, increasing=True)
    sqrt_w = np.sqrt(w)
    normal = design * sqrt_w[:, None]
    coefficients, *_ = np.linalg.lstsq(normal, y * sqrt_w, rcond=None)

    residual = y - design @ coefficients
    dof = x.size - n_params
    variance = float(np.sum(w * residual**2) / dof) if dof > 0 else 0.0
    covariance = variance * np.linalg.inv(normal.T @ normal)

    total = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r_squared = 1.0 - float(np.sum(w * residual**2)) / total if total > 0 else 0.0

    return coefficients, covariance, r_squared


def fit_virial(
    concentration: np.ndarray,
    kc_over_i: np.ndarray,
    order: int = 1,
    weights: np.ndarray | None = None,
) -> FitResult:
    """Fit the virial expansion over the supplied points.

    The caller is responsible for having already restricted the arrays to the
    chosen linear region -- that choice is the scientific claim, and it is
    recorded upstream rather than made here.
    """
    if order not in (1, 2):
        raise ValueError("virial fit order must be 1 (A2) or 2 (A2 and A3)")

    coefficients, covariance, r_squared = weighted_least_squares(
        concentration, kc_over_i, order=order, weights=weights
    )
    errors = np.sqrt(np.diag(covariance))
    finite = np.isfinite(concentration) & np.isfinite(kc_over_i)

    return FitResult(
        slope=float(coefficients[1]),
        intercept=float(coefficients[0]),
        slope_stderr=float(errors[1]),
        intercept_stderr=float(errors[0]),
        r_squared=float(r_squared),
        n_points=int(finite.sum()),
        c_min=float(np.nanmin(concentration[finite])),
        c_max=float(np.nanmax(concentration[finite])),
        order=order,
        quadratic_coeff=float(coefficients[2]) if order == 2 else None,
    )


def derive_parameters(
    fit: FitResult,
    calibrated: bool,
    k_h: float = 0.0,
    mw_over_m0_final: float | None = None,
) -> DerivedParameters:
    """Turn fit coefficients into physical parameters.

    ``a2_mw`` is always available because alpha cancels in slope/intercept.
    ``mw``, ``a2``, and ``a3`` are populated only when the ordinate was built
    from an absolute Rayleigh ratio -- otherwise they would silently carry a
    factor of alpha and look entirely plausible.
    """
    a2_mw = fit.a2_mw
    parameters = DerivedParameters(
        a2_mw=a2_mw,
        k_d=2.0 * a2_mw + k_h,
        mw_over_m0_final=mw_over_m0_final,
        calibrated=calibrated,
    )

    if calibrated:
        parameters.mw = 1.0 / fit.intercept
        parameters.a2 = fit.slope / 2.0
        if fit.order == 2 and fit.quadratic_coeff is not None:
            parameters.a3 = fit.quadratic_coeff / 3.0

    return parameters
