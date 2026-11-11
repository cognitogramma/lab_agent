"""Declaration-versus-data consistency checks.

Design section 6.0.1. Intake records what the researcher says the run is. This
module asks, in arithmetic only, whether the trace behaves the way that claim
implies -- and raises a conflict when it does not.

The check is cheap and it catches the most expensive error class available here:
a run processed flawlessly under the wrong physical model. Every number that
comes out of such a run looks entirely normal, so nothing downstream will catch
it.

Two rules govern everything below.

**A conflict never overrides the declaration.** The data can disagree with a
correct declaration for real reasons -- a sample that aggregates faster than it
dilutes rises on a dilution run, and it is still a dilution run. Code has no way
to tell that case from a mislabelled file, so it reports and stops rather than
deciding.

**A check that cannot run says so.** Silently passing an unrunnable check is
worse than not having it, because the run then carries the appearance of having
been verified. Skips are returned alongside conflicts with the reason attached.

No model client reaches this module, by the tier-0 invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from continuum_spectranalytics.state import Conflict
from continuum_spectranalytics.tier0.argen_io import lookup_metadata, parse_concentration

# Fraction of the declared Q/V an observed decay rate must reach before it
# counts as matching. Deliberately tighter than a factor of two: the two cuvette
# volumes in use in this project differ by exactly a factor of two, so a
# factor-of-two tolerance would decline to notice the one disagreement this
# check exists to catch. Q and V are known to a few percent; the slack here is
# for the sample's own kinetics contributing to the decay.
DECAY_MATCH_TOLERANCE = 0.7

# How well an exponential must fit before a decay is called clean rather than
# incidental. Used when no Q/V was declared and the rate cannot be compared.
EXPONENTIAL_FIT_R2 = 0.90

# Stop-flow cycling runs roughly two minutes per cycle. A periodicity anywhere
# in this band counts as cycling.
CYCLE_PERIOD_BOUNDS_S = (45.0, 300.0)


@dataclass
class ExponentialFit:
    """Least-squares fit of log(y) against t."""

    rate_per_s: float
    """Decay rate. Positive means the trace decays, negative means it rises."""

    r_squared: float
    n_points: int


@dataclass
class ConsistencyReport:
    """Outcome of the whole check set."""

    conflicts: list[Conflict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    """Checks that could not run, each with the reason."""

    passed: list[str] = field(default_factory=list)

    @property
    def unresolved(self) -> list[Conflict]:
        return [conflict for conflict in self.conflicts if not conflict.resolved]


def fit_exponential_decay(time_s: np.ndarray, signal: np.ndarray) -> ExponentialFit | None:
    """Fit ``y = A*exp(-rate*t)`` by least squares on log(y).

    Returns None when too few strictly positive samples survive -- a trace that
    crosses or sits at zero has no logarithm, and forcing one would invent a
    rate out of the clipping.
    """
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    usable = np.isfinite(time_s) & np.isfinite(signal) & (signal > 0)
    if usable.sum() < 10:
        return None

    t = time_s[usable]
    y = np.log(signal[usable])
    slope, intercept = np.polyfit(t, y, 1)

    predicted = slope * t + intercept
    residual_ss = float(np.sum((y - predicted) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0

    return ExponentialFit(
        rate_per_s=float(-slope), r_squared=float(r_squared), n_points=int(usable.sum())
    )


def dominant_period_s(
    time_s: np.ndarray,
    values: np.ndarray,
    bounds: tuple[float, float] = CYCLE_PERIOD_BOUNDS_S,
) -> float | None:
    """Strongest repeat period within ``bounds``, by autocorrelation.

    Used to detect stop-flow cycling: acquisition that alternates stir, settle,
    and measure produces a periodic structure the continuous model does not
    describe. Returns None when nothing in the band actually repeats.

    The distinction that matters here is **periodic** versus merely **slow**. A
    smooth trace has an autocorrelation that stays near 1 across short lags and
    decays from there, so taking the largest value in a lag band finds the edge
    of the band on any smoothly varying signal and reports it as a cycle. A real
    cycle instead makes the autocorrelation fall away, cross zero, and come back
    up at the period. So the search starts only after the first zero crossing,
    and the peak has to be an interior local maximum. Without that, this
    function raises a cycling conflict on every clean continuous run.
    """
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    usable = np.isfinite(time_s) & np.isfinite(values)
    if usable.sum() < 100:
        return None

    t = time_s[usable]
    y = values[usable]

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None

    # Detrend, so a slow rise or decay does not swamp the periodic component.
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    y = y - y.mean()
    if not np.any(np.abs(y) > 0):
        return None

    # Zero-padded, so this is the linear autocorrelation. The unpadded transform
    # gives the circular one, which wraps the end of the run onto its start and
    # invents correlations that are not in the data.
    padded = 1 << int(np.ceil(np.log2(2 * y.size)))
    spectrum = np.fft.rfft(y, n=padded)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=padded)[: y.size]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]

    low = max(1, int(round(bounds[0] / dt)))
    high = min(correlation.size - 2, int(round(bounds[1] / dt)))
    if high <= low:
        return None

    # Nothing repeats until the correlation has first fallen away.
    negative = np.flatnonzero(correlation[: high + 1] <= 0)
    if negative.size == 0:
        return None
    start = max(low, int(negative[0]) + 1)
    if start >= high:
        return None

    best_lag: int | None = None
    best_value = 0.2  # below this a "peak" is just the top of a noise band
    for lag in range(start, high + 1):
        value = correlation[lag]
        if value > best_value and value >= correlation[lag - 1] and value >= correlation[lag + 1]:
            best_lag, best_value = lag, float(value)

    if best_lag is None:
        return None
    return float(best_lag * dt)


def check_declarations(
    time_s: np.ndarray,
    signal: np.ndarray,
    declared: dict[str, object],
    metadata: dict[str, str] | None = None,
    stir_speed: np.ndarray | None = None,
    concentration_tolerance: float = 0.05,
) -> ConsistencyReport:
    """Test each declaration against what the trace actually does.

    Parameters
    ----------
    declared:
        Declared values keyed as in :mod:`continuum_spectranalytics.intake` -- ``experiment_type``,
        ``technique``, ``instrument``, ``wavelength_nm``, ``c0_g_per_cm3``,
        ``cell_volume_cm3``, ``flow_rate_cm3_per_s``.
    stir_speed:
        The stirring-speed column, when the file has one. Cycling shows up there
        far more clearly than in the scattering channel.
    """
    metadata = metadata or {}
    report = ConsistencyReport()

    _check_experiment_type(time_s, signal, declared, report)
    _check_wavelength(declared, metadata, report)
    _check_c0(declared, metadata, concentration_tolerance, report)
    _check_technique(time_s, signal, declared, stir_speed, report)
    return report


def _check_experiment_type(
    time_s: np.ndarray,
    signal: np.ndarray,
    declared: dict[str, object],
    report: ConsistencyReport,
) -> None:
    """Does the trace carry the dilution signature the declaration implies?"""
    experiment_type = declared.get("experiment_type")
    if experiment_type is None:
        report.skipped.append(
            "experiment_type: not declared, so the dilution signature cannot be tested"
        )
        return

    fit = fit_exponential_decay(time_s, signal)
    if fit is None:
        report.skipped.append(
            "experiment_type: fewer than 10 strictly positive samples, so no "
            "exponential could be fitted"
        )
        return

    volume = declared.get("cell_volume_cm3")
    flow = declared.get("flow_rate_cm3_per_s")
    expected_rate: float | None = None
    if isinstance(volume, (int, float)) and isinstance(flow, (int, float)) and volume > 0:
        expected_rate = float(flow) / float(volume)

    if experiment_type == "acd_dilution":
        if fit.rate_per_s <= 0:
            report.conflicts.append(
                Conflict(
                    check="experiment_type_vs_trace",
                    declared="acd_dilution",
                    observed=f"trace rises (fitted rate {fit.rate_per_s:.3g} /s)",
                    detail=(
                        "A dilution run should decay as exp(-Qt/V). This one rises. "
                        "That happens on a real dilution run when the sample "
                        "aggregates faster than it dilutes, so it is not proof of a "
                        "mislabelled file -- but it needs confirming before the "
                        "concentration axis is trusted."
                    ),
                )
            )
            return

        if expected_rate is None:
            report.skipped.append(
                "experiment_type: no Q and V declared, so the observed decay rate "
                f"({fit.rate_per_s:.3g} /s) could not be compared against Q/V"
            )
            report.passed.append("experiment_type: trace decays, as a dilution run should")
            return

        ratio = fit.rate_per_s / expected_rate
        if ratio < DECAY_MATCH_TOLERANCE or ratio > 1.0 / DECAY_MATCH_TOLERANCE:
            report.conflicts.append(
                Conflict(
                    check="experiment_type_vs_trace",
                    declared=f"acd_dilution with Q/V = {expected_rate:.3g} /s",
                    observed=f"decays at {fit.rate_per_s:.3g} /s ({ratio:.2f} x Q/V)",
                    detail=(
                        "The trace decays, but not at the rate the declared flow rate "
                        "and cuvette volume imply. Either Q or V is wrong, or the "
                        "sample's own kinetics are contributing to the decay."
                    ),
                )
            )
            return
        report.passed.append(
            f"experiment_type: decays at {fit.rate_per_s:.3g} /s, "
            f"{ratio:.2f} x the declared Q/V"
        )
        return

    if experiment_type == "fixed_conc_aggregation":
        decays_cleanly = fit.rate_per_s > 0 and fit.r_squared >= EXPONENTIAL_FIT_R2
        matches_rate = (
            expected_rate is not None
            and fit.rate_per_s > 0
            and DECAY_MATCH_TOLERANCE
            <= fit.rate_per_s / expected_rate
            <= 1.0 / DECAY_MATCH_TOLERANCE
        )
        if decays_cleanly or matches_rate:
            observed = f"clean exponential decay at {fit.rate_per_s:.3g} /s"
            if matches_rate and expected_rate:
                observed += f", within a factor of two of Q/V = {expected_rate:.3g} /s"
            report.conflicts.append(
                Conflict(
                    check="experiment_type_vs_trace",
                    declared="fixed_conc_aggregation",
                    observed=f"{observed} (R^2 = {fit.r_squared:.3f})",
                    detail=(
                        "A fixed-concentration run should show no dilution signature. "
                        "This trace decays exponentially, which is what a dilution "
                        "trajectory looks like. If the declaration is right, the "
                        "decay is physical and worth explaining."
                    ),
                )
            )
            return
        report.passed.append("experiment_type: no dilution signature, as declared")


def _check_wavelength(
    declared: dict[str, object],
    metadata: dict[str, str],
    report: ConsistencyReport,
) -> None:
    """Does the declared instrument agree with the file header?"""
    declared_nm = declared.get("wavelength_nm")
    if declared_nm is None:
        report.skipped.append("wavelength: no instrument declared")
        return

    hit = lookup_metadata(metadata, "wavelength_nm")
    if hit is None:
        report.skipped.append("wavelength: the file header does not record one")
        return

    header_key, raw = hit
    digits = "".join(c for c in str(raw) if c.isdigit() or c == ".")
    try:
        header_nm = float(digits)
    except ValueError:
        report.skipped.append(
            f"wavelength: header {header_key!r} reads {raw!r}, which is not a number"
        )
        return

    if abs(header_nm - float(declared_nm)) > 1.0:
        report.conflicts.append(
            Conflict(
                check="instrument_vs_header",
                declared=f"{float(declared_nm):g} nm",
                observed=f"header {header_key!r} says {header_nm:g} nm",
                detail=(
                    "Wavelength enters the optical constant as lambda^-4, so a "
                    "mismatch here scales every molar mass by the fourth power of "
                    "the ratio."
                ),
            )
        )
        return
    report.passed.append(f"wavelength: header agrees at {header_nm:g} nm")


def _check_c0(
    declared: dict[str, object],
    metadata: dict[str, str],
    tolerance: float,
    report: ConsistencyReport,
) -> None:
    """Does the declared c0 agree with the header's sample concentration?"""
    declared_c0 = declared.get("c0_g_per_cm3")
    if not isinstance(declared_c0, (int, float)):
        report.skipped.append("c_0: not declared")
        return

    hit = lookup_metadata(metadata, "c0_g_per_cm3")
    if hit is None:
        report.skipped.append("c_0: the file header does not record a sample concentration")
        return

    header_key, raw = hit
    header_c0 = parse_concentration(raw)
    if header_c0 is None:
        report.skipped.append(
            f"c_0: header {header_key!r} reads {raw!r}, whose units could not be "
            "identified, so no comparison was made"
        )
        return

    if header_c0 <= 0:
        report.skipped.append(f"c_0: header {header_key!r} is not positive")
        return

    relative = abs(float(declared_c0) - header_c0) / header_c0
    if relative > tolerance:
        report.conflicts.append(
            Conflict(
                check="c0_vs_header",
                declared=f"{float(declared_c0):.6g} g/cm^3",
                observed=f"header {header_key!r} = {raw!r} ({header_c0:.6g} g/cm^3)",
                detail=(
                    f"They differ by {100 * relative:.1f} percent, beyond the "
                    f"{100 * tolerance:.0f} percent tolerance. c0 scales the whole "
                    "concentration axis, and with it Mw."
                ),
            )
        )
        return
    report.passed.append(f"c_0: agrees with the header to within {100 * relative:.1f} percent")


def _check_technique(
    time_s: np.ndarray,
    signal: np.ndarray,
    declared: dict[str, object],
    stir_speed: np.ndarray | None,
    report: ConsistencyReport,
) -> None:
    """Does the acquisition show the stop-flow cycling the technique implies?

    DLS-linked acquisition cycles stir, settle, then measure -- roughly two
    minutes per cycle. Cycling on a run declared SLS means the trace is not the
    clean continuous curve the continuous model assumes; its absence on a run
    declared DLS means the DLS leg was probably not running.
    """
    technique = declared.get("technique")
    if technique is None:
        report.skipped.append("technique: not declared")
        return

    source = "stirring speed" if stir_speed is not None else "the scattering channel"
    series = stir_speed if stir_speed is not None else signal
    period = dominant_period_s(time_s, series)

    if technique == "DLS" and period is None:
        report.conflicts.append(
            Conflict(
                check="technique_vs_cycling",
                declared="DLS",
                observed=f"no stop-flow cycling found in {source}",
                detail=(
                    "DLS-linked acquisition cycles stir, settle, and measure at "
                    "roughly two minutes per cycle. Nothing periodic was detected in "
                    "that band, which suggests the DLS leg was not cycling."
                ),
            )
        )
        return

    if technique == "SLS" and period is not None:
        report.conflicts.append(
            Conflict(
                check="technique_vs_cycling",
                declared="SLS",
                observed=f"{source} repeats every {period:.0f} s",
                detail=(
                    "That is the stop-flow cycling signature of a DLS-linked "
                    "acquisition. The trace then carries stir and settle structure "
                    "rather than a clean continuous curve, and smoothing it as one "
                    "continuous series will average across the cycle."
                ),
            )
        )
        return

    detail = f"repeats every {period:.0f} s" if period else "no cycling detected"
    report.passed.append(f"technique: {detail} in {source}, consistent with {technique}")


__all__ = [
    "CYCLE_PERIOD_BOUNDS_S",
    "DECAY_MATCH_TOLERANCE",
    "EXPONENTIAL_FIT_R2",
    "ConsistencyReport",
    "ExponentialFit",
    "check_declarations",
    "dominant_period_s",
    "fit_exponential_decay",
]
