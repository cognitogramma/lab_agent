"""Whole-trace baseline estimators. NOT on the measurement path.

.. warning::
   Nothing in this module may be used to correct a run. Both estimators here fit
   the *whole trace* and have enough freedom to follow a transition, which means
   they will quietly absorb the kinetics along with the drift -- the exact
   failure that produces a smooth, plausible, physically wrong result. The
   permitted models are a polynomial of order at most two, or piecewise-linear
   interpolation, fitted **only to selected anchor regions** that are barred from
   intersecting the transition. That is :mod:`continuum_spectranalytics.tier0.anchors`.

   These are retained for exploratory plotting and for comparison against the
   anchor-based fit, not for production. An earlier version of the pipeline
   applied :func:`asls_baseline` automatically whenever drift was detected;
   that was a defect.

Drift is modeled and subtracted, leaving **every timestamp in place**. Excising
a drifting stretch would delete the real kinetic signal riding on top of it.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _fill_gaps(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate across NaNs so a solver can run; return the filled copy and mask."""
    finite = np.isfinite(values)
    if finite.all():
        return np.array(values, dtype=float, copy=True), finite
    if finite.sum() < 2:
        raise ValueError("need at least two finite samples to model a baseline")
    filled = np.array(values, dtype=float, copy=True)
    indices = np.arange(values.size)
    filled[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return filled, finite


def asls_baseline(
    signal: np.ndarray,
    smoothness: float = 1e6,
    asymmetry: float = 0.01,
    n_iterations: int = 10,
) -> np.ndarray:
    """Asymmetric least-squares baseline (Eilers & Boelens).

    Balances a second-difference smoothness penalty against an asymmetric
    residual weight, so the fitted curve tracks the underlying drift while
    peaks -- which sit above it -- are down-weighted rather than absorbed.

    Parameters
    ----------
    smoothness:
        Stiffness of the baseline. Larger means flatter.
    asymmetry:
        Weight given to samples above the current baseline. Small values keep
        the baseline underneath positive-going signal.
    """
    filled, _ = _fill_gaps(signal)
    n = filled.size
    if n < 3:
        return np.zeros_like(filled)

    differences = sparse.diags(
        [1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc", dtype=float
    )
    penalty = smoothness * (differences.T @ differences)

    weights = np.ones(n)
    baseline = np.zeros(n)
    for _ in range(n_iterations):
        weighted = sparse.spdiags(weights, 0, n, n, format="csc")
        baseline = spsolve((weighted + penalty).tocsc(), weights * filled)
        weights = np.where(filled > baseline, asymmetry, 1.0 - asymmetry)
    return baseline


def polynomial_baseline(
    time_s: np.ndarray, signal: np.ndarray, order: int = 2, n_iterations: int = 5
) -> np.ndarray:
    """Iteratively reweighted polynomial baseline.

    Cheaper and stiffer than :func:`asls_baseline`; appropriate when drift is
    known to be a slow monotonic trend rather than a wandering baseline.
    """
    if order < 1:
        raise ValueError("baseline polynomial order must be at least 1")
    filled, _ = _fill_gaps(signal)

    # Center and scale time so the Vandermonde system stays well conditioned
    # over a 22-hour axis.
    span = time_s[-1] - time_s[0]
    scaled = (time_s - time_s[0]) / span if span > 0 else time_s - time_s[0]

    weights = np.ones_like(filled)
    baseline = np.zeros_like(filled)
    for _ in range(n_iterations):
        coefficients = np.polyfit(scaled, filled, order, w=weights)
        baseline = np.polyval(coefficients, scaled)
        residual = filled - baseline
        scale = np.median(np.abs(residual - np.median(residual))) * 1.4826
        if scale <= 0:
            break
        # Down-weight points above the baseline so peaks do not drag it upward.
        weights = np.where(residual > 0, 1.0 / (1.0 + (residual / scale) ** 2), 1.0)
    return baseline


def subtract_baseline(
    signal: np.ndarray, baseline: np.ndarray
) -> np.ndarray:
    """Subtract the modeled drift, preserving NaNs and array length."""
    if signal.shape != baseline.shape:
        raise ValueError("signal and baseline must have the same shape")
    return signal - baseline


def correct_drift(
    time_s: np.ndarray,
    signal: np.ndarray,
    method: str = "asls",
    **kwargs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Model and subtract drift. Returns ``(corrected, baseline)``.

    Length and index alignment are preserved -- no sample is removed.
    """
    if method == "asls":
        baseline = asls_baseline(signal, **kwargs)
    elif method == "polynomial":
        baseline = polynomial_baseline(time_s, signal, **kwargs)
    else:
        raise ValueError(f"unknown baseline method: {method!r}")
    return subtract_baseline(signal, baseline), baseline
