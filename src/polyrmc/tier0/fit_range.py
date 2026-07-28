"""Candidate linear regions for the Kc/I_R vs c fit.

This replaces the decision with no audit trail today. The virial expansion is a
low-concentration expansion, and the measured curve flattens once chain overlap
sets in -- for the published alginate data, above roughly 0.0033 g/cm^3.
Someone chooses where to cut, and that choice sets A2 directly.

Here the choice is made explicit. A candidate range is proposed only if it
passes every deterministic linearity gate, so each option is defensible on its
own; a tier-1 judge then selects among them and the rejects are logged
alongside the winner. Which points were *excluded* from the linear region is
the scientific claim a reviewer needs to see.

Every quantity used here is invariant to a constant y-scale, so fit-range
selection does not wait on the toluene calibration.
"""

from __future__ import annotations

import numpy as np

from polyrmc.config import FitRangeConfig
from polyrmc.state import Candidate
from polyrmc.tier0.detect import pelt_changepoints
from polyrmc.tier0.virial import weighted_least_squares


def sorted_finite(
    concentration: np.ndarray, kc_over_i: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite pairs and sort by ascending concentration."""
    concentration = np.asarray(concentration, dtype=float)
    kc_over_i = np.asarray(kc_over_i, dtype=float)
    finite = np.isfinite(concentration) & np.isfinite(kc_over_i)
    c, y = concentration[finite], kc_over_i[finite]
    order = np.argsort(c)
    return c[order], y[order]


def residual_runs_z(residual: np.ndarray) -> float:
    """Wald-Wolfowitz runs statistic on residual signs.

    Systematic curvature makes residual signs cluster into long runs, giving a
    large negative z. Near zero means the residuals alternate as randomly as
    noise should.
    """
    signs = np.sign(residual)
    signs = signs[signs != 0]
    n = signs.size
    if n < 3:
        return 0.0
    n_pos = int((signs > 0).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    runs = 1 + int((signs[1:] != signs[:-1]).sum())
    mean = 2.0 * n_pos * n_neg / n + 1.0
    variance = (
        2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n) / (n**2 * (n - 1.0))
    )
    if variance <= 0:
        return 0.0
    return float((runs - mean) / np.sqrt(variance))


def lag1_autocorrelation(residual: np.ndarray) -> float:
    """Lag-1 autocorrelation of a residual series."""
    if residual.size < 3:
        return 0.0
    centered = residual - residual.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denominator)


def evaluate_range(
    concentration: np.ndarray, kc_over_i: np.ndarray, upper: int, stride: int = 1
) -> dict[str, float] | None:
    """Score the range ``[0, upper)`` of the sorted arrays.

    Returns ``None`` if the range is degenerate. All returned quantities are
    scale-invariant in y except the fitted coefficients themselves.

    Parameters
    ----------
    stride:
        Keep every ``stride``-th point. This is not decoration: Part 1 smooths
        the trace, so neighbouring samples are no longer independent. Left
        unthinned, the residuals of a perfectly good linear fit are strongly
        autocorrelated and the standard errors are badly underestimated, which
        makes the linearity gates reject every candidate range. Thinning to
        roughly the smoothing correlation length restores an honest effective
        sample size.
    """
    stride = max(1, int(stride))
    c = concentration[:upper][::stride]
    y = kc_over_i[:upper][::stride]
    if c.size < 4 or np.ptp(c) == 0:
        return None

    try:
        linear, covariance, r_squared = weighted_least_squares(c, y, order=1)
    except (ValueError, np.linalg.LinAlgError):
        return None

    residual = y - (linear[0] + linear[1] * c)

    # A quadratic term that is significant relative to its own uncertainty is
    # direct evidence the range has run past the linear region.
    curvature_t = 0.0
    if c.size > 6:
        try:
            quadratic, quad_cov, _ = weighted_least_squares(c, y, order=2)
            quad_err = float(np.sqrt(quad_cov[2, 2]))
            if quad_err > 0:
                curvature_t = float(abs(quadratic[2]) / quad_err)
        except (ValueError, np.linalg.LinAlgError):
            pass

    intercept_err = float(np.sqrt(covariance[0, 0]))
    slope_err = float(np.sqrt(covariance[1, 1]))

    return {
        "n_points": float(c.size),
        "c_max": float(c[-1]),
        "c_min": float(c[0]),
        "r_squared": float(r_squared),
        "residual_autocorr": lag1_autocorrelation(residual),
        "residual_runs_z": residual_runs_z(residual),
        "curvature_t": curvature_t,
        "slope": float(linear[1]),
        "intercept": float(linear[0]),
        "slope_rel_err": float(slope_err / abs(linear[1])) if linear[1] != 0 else np.inf,
        "intercept_rel_err": (
            float(intercept_err / abs(linear[0])) if linear[0] != 0 else np.inf
        ),
        # alpha cancels here, so this is directly comparable to published values.
        "a2_mw": float(linear[1] / (2.0 * linear[0])) if linear[0] != 0 else np.nan,
    }


def _candidate_cuts(
    concentration: np.ndarray, kc_over_i: np.ndarray, config: FitRangeConfig
) -> list[int]:
    """Propose upper-bound indices from change points plus a uniform scan."""
    n = concentration.size
    cuts: set[int] = set()

    # Where the data departs from a single global line, the residual changes
    # level -- that is a change point in the residual series.
    try:
        global_fit, _, _ = weighted_least_squares(concentration, kc_over_i, order=1)
        residual = kc_over_i - (global_fit[0] + global_fit[1] * concentration)
        for point in pelt_changepoints(residual, min_segment=max(5, config.min_points // 2)):
            cuts.add(int(point))
    except (ValueError, np.linalg.LinAlgError):
        pass

    scan = np.linspace(config.min_points, n, num=config.n_candidates + 1)
    cuts.update(int(value) for value in scan)

    return sorted(cut for cut in cuts if config.min_points <= cut <= n)


def propose_fit_ranges(
    concentration: np.ndarray,
    kc_over_i: np.ndarray,
    config: FitRangeConfig | None = None,
    stride: int = 1,
) -> list[Candidate]:
    """Generate linear-region candidates that are safe by construction.

    A candidate survives only if it holds enough points, fits well, leaves no
    structure in its residuals, and shows no significant curvature. The value
    carried by each candidate is the upper concentration bound -- the direct
    statement "the linear region ends here".
    """
    config = config or FitRangeConfig()
    c, y = sorted_finite(concentration, kc_over_i)
    if c.size < config.min_points:
        return []

    stride = max(1, int(stride))
    candidates: list[Candidate] = []
    for cut in _candidate_cuts(c, y, config):
        features = evaluate_range(c, y, cut, stride=stride)
        if features is None:
            continue
        if features["n_points"] < config.min_points:
            continue
        if features["r_squared"] < config.min_r_squared:
            continue
        if abs(features["residual_autocorr"]) > config.max_residual_autocorr:
            continue
        if features["curvature_t"] > 3.0:
            continue

        features["n_excluded"] = float(c.size - cut)
        features["fraction_excluded"] = float((c.size - cut) / c.size)
        features["residual_stride"] = float(stride)
        candidates.append(
            Candidate(
                index=len(candidates),
                value=features["c_max"],
                label=(
                    f"c <= {features['c_max']:.4g} g/cm^3 "
                    f"({int(features['n_points'])} points, "
                    f"{int(features['n_excluded'])} excluded)"
                ),
                features=features,
            )
        )
    return candidates


def conservative_fit_range(candidates: list[Candidate]) -> Candidate:
    """The safest candidate: the narrowest, lowest-concentration range.

    Used when a bounded loop exhausts its iterations. The virial expansion is
    most trustworthy at low concentration, so restricting the range is the
    direction that risks precision rather than correctness.
    """
    if not candidates:
        raise ValueError("no fit-range candidates passed the linearity gates")
    return min(candidates, key=lambda candidate: float(candidate.features["c_max"]))


def apply_fit_range(
    concentration: np.ndarray, kc_over_i: np.ndarray, c_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the data to the selected linear region."""
    c, y = sorted_finite(concentration, kc_over_i)
    keep = c <= c_max
    return c[keep], y[keep]
