"""End-to-end processing: Part 1 (clean) and Part 2 (extract).

Part 1  ingest -> clean -> detect artifacts -> splice -> baseline-correct ->
        smooth, emitting the validated CSV.
Part 2  time -> concentration -> Kc/I_R vs c -> A2*Mw, Mw, A3, k_D, Mw/M0.

The CSV is the only thing that passes between them.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from polyrmc.config import RunConfig
from polyrmc.io_csv import read_processed_csv, write_processed_csv
from polyrmc.provenance import write_sidecar
from polyrmc.state import AnomalyType, DerivedParameters, RunState
from polyrmc.tier0 import baseline as baseline_module
from polyrmc.tier0 import optics
from polyrmc.tier0.argen_io import load_argen_file
from polyrmc.tier0.classify import classify_all
from polyrmc.tier0.detect import detect_anomalies
from polyrmc.tier0.dilution import aggregation_index, concentration_at
from polyrmc.tier0.fit_range import apply_fit_range, sorted_finite
from polyrmc.tier0.smoothing import savgol_smooth
from polyrmc.tier0.splice import apply_splice
from polyrmc.tier0.virial import build_kc_over_i, derive_parameters, fit_virial
from polyrmc.tier1.fit_range_selector import select_fit_range
from polyrmc.tier1.loop import Judge
from polyrmc.tier1.smoothing_selector import select_smoothing_window


def run_part1(
    config: RunConfig,
    channel: str | None = None,
    judge: Judge | None = None,
    output_csv: str | Path | None = None,
) -> tuple[RunState, Path]:
    """Clean one channel and emit the validated CSV plus its provenance sidecar.

    Parameters
    ----------
    channel:
        Scattering channel to process. Defaults to the first one found; the
        instrument runs 16 independent cells, so a full session calls this once
        per channel.
    """
    argen = load_argen_file(config.source_file)
    channel = channel or argen.scattering_columns[0]
    if channel not in argen.scattering_columns:
        raise ValueError(
            f"{channel!r} is not a scattering channel; available: "
            f"{argen.scattering_columns}"
        )

    state = RunState(run_id=config.run_id, source_file=str(config.source_file))
    state.time_s = argen.time_s
    state.raw_signal = argen.data[channel].to_numpy(dtype=float)
    state.time_format = argen.time_format
    state.n_header_rows = argen.n_header_rows
    state.dropped_rows = argen.dropped_rows

    if not argen.is_uniformly_sampled:
        state.warnings.append(
            "sampling is non-uniform; time, not row index, is the x-axis"
        )

    records, _change_points = detect_anomalies(
        state.time_s, state.raw_signal, channel, config.detectors
    )
    settled, ambiguous = classify_all(records)
    state.anomalies = settled + ambiguous
    if ambiguous:
        # Unclassified regions are left in the trace. Excising something the
        # rules could not name would delete data on a guess.
        state.warnings.append(
            f"{len(ambiguous)} region(s) could not be classified deterministically "
            "and were left in place pending review"
        )

    state.spliced_signal, state.splice = apply_splice(state.raw_signal, state.anomalies)

    # Only correct drift that was actually detected. On a dilution trajectory
    # the exponential decay *is* the measurement, and a flexible baseline
    # fitted to it would subtract the signal along with any drift.
    has_drift = any(
        record.anomaly_type is AnomalyType.BASELINE_DRIFT for record in state.anomalies
    )
    if has_drift:
        state.corrected_signal, state.baseline = baseline_module.correct_drift(
            state.time_s, state.spliced_signal, method="asls"
        )
    else:
        state.baseline = np.zeros_like(state.spliced_signal)
        state.corrected_signal = state.spliced_signal

    window_loop = select_smoothing_window(
        state.corrected_signal,
        smoothing_config=config.smoothing,
        loop_config=config.loop,
        judge=judge,
        context={"channel": channel, "run_id": config.run_id},
    )
    state.loops["smoothing_window"] = window_loop
    if window_loop.resolved_value is None:
        raise RuntimeError(
            "smoothing window selection produced no usable option; the trace may be "
            "too short or too noisy for any window to pass the safety gates"
        )
    state.smoothed_signal = savgol_smooth(
        state.corrected_signal, int(window_loop.resolved_value), config.smoothing.polyorder
    )

    if config.experiment_type == "dilution_trajectory" and config.dilution is not None:
        state.concentration = concentration_at(state.time_s, config.dilution)
    else:
        state.concentration = np.full_like(state.time_s, np.nan)

    excess = optics.excess_scattering(state.smoothed_signal, config.solvent_blank_counts)
    rayleigh, calibrated = optics.rayleigh_ratio(excess, config.optical.alpha)
    if not calibrated:
        state.warnings.append(
            "alpha is not set: the rayleigh_ratio column holds excess counts, not "
            "cm^-1. A2*Mw and Mw/M0 remain valid; Mw and A2 alone do not."
        )

    try:
        mw_over_m0 = aggregation_index(state.smoothed_signal)
    except ValueError:
        mw_over_m0 = np.full_like(state.time_s, np.nan)

    output_csv = Path(output_csv) if output_csv else _default_output(config, channel)
    write_processed_csv(
        output_csv,
        {
            "time_s": state.time_s,
            "concentration_g_per_cm3": state.concentration,
            "signal_raw": state.raw_signal,
            "signal_smoothed": state.smoothed_signal,
            "rayleigh_ratio_cm_inv": rayleigh,
            "mw_over_m0": mw_over_m0,
        },
        metadata={
            "run_id": config.run_id,
            "channel": channel,
            "source_file": Path(config.source_file).name,
            "experiment_type": config.experiment_type,
            "smoothing_window": window_loop.resolved_value,
            "smoothing_polyorder": config.smoothing.polyorder,
            "alpha_calibrated": calibrated,
            "n_excised": state.splice.n_excised,
        },
    )
    write_sidecar(state, config, output_csv)
    return state, output_csv


def run_part2(
    processed_csv: str | Path,
    config: RunConfig,
    judge: Judge | None = None,
    fit_order: int = 1,
) -> RunState:
    """Extract A2*Mw, Mw, A3, and k_D from a validated CSV."""
    frame, metadata = read_processed_csv(processed_csv)
    state = RunState(
        run_id=metadata.get("run_id", config.run_id), source_file=str(processed_csv)
    )

    concentration = frame["concentration_g_per_cm3"].to_numpy(dtype=float)
    if not np.isfinite(concentration).any():
        raise ValueError(
            "no concentration axis in this file: virial analysis requires a "
            "dilution_trajectory run"
        )

    intensity = frame["rayleigh_ratio_cm_inv"].to_numpy(dtype=float)
    calibrated = str(metadata.get("alpha_calibrated", "False")).lower() == "true"

    k = optics.optical_constant(config.optical)
    state.concentration = concentration
    state.kc_over_i = build_kc_over_i(concentration, intensity, k)

    # Part 1 smoothed this trace, so adjacent samples are correlated. Thinning
    # to the smoothing correlation length is what keeps the linearity gates and
    # the reported uncertainties honest.
    stride = _residual_stride(metadata)

    range_loop = select_fit_range(
        state.concentration,
        state.kc_over_i,
        fit_config=config.fit_range,
        loop_config=config.loop,
        judge=judge,
        context={"run_id": state.run_id, "calibrated": calibrated},
        stride=stride,
    )
    state.loops["fit_range"] = range_loop
    if range_loop.resolved_value is None:
        raise RuntimeError(
            "no concentration range passed the linearity gates; the curve may have "
            "no usable linear region at these concentrations"
        )

    c_fit, y_fit = apply_fit_range(
        state.concentration, state.kc_over_i, float(range_loop.resolved_value)
    )
    state.fit = fit_virial(c_fit[::stride], y_fit[::stride], order=fit_order)
    if stride > 1:
        state.warnings.append(
            f"fit uses every {stride}th point (smoothing window "
            f"{metadata.get('smoothing_window')}): correlated samples carry no "
            "independent information, and counting them would understate the "
            "uncertainty on A2*Mw"
        )

    mw_over_m0 = frame["mw_over_m0"].to_numpy(dtype=float)
    final = float(np.nanmedian(mw_over_m0[-50:])) if np.isfinite(mw_over_m0).any() else None

    state.parameters = derive_parameters(
        state.fit, calibrated=calibrated, k_h=config.k_h, mw_over_m0_final=final
    )
    if not calibrated:
        state.warnings.append(
            "uncalibrated run: A2*Mw and k_D are reported; Mw, A2, and A3 are not, "
            "since they would silently carry a factor of alpha"
        )
    return state


def _residual_stride(metadata: dict[str, str]) -> int:
    """Point spacing at which smoothed samples are effectively independent.

    Read from the smoothing window Part 1 recorded in the CSV header, which is
    why that value is part of the boundary format rather than the sidecar. Half
    the window is the conventional decorrelation length for a Savitzky-Golay
    filter. Falls back to 1 when the field is absent, which keeps files not
    written by this pipeline readable.
    """
    raw = metadata.get("smoothing_window")
    try:
        window = int(float(raw))
    except (TypeError, ValueError):
        return 1
    return max(1, window // 2)


def _default_output(config: RunConfig, channel: str) -> Path:
    from polyrmc.config import RUNS_DIR

    safe_channel = "".join(c if c.isalnum() else "_" for c in channel)
    return RUNS_DIR / f"{config.run_id}_{safe_channel}.csv"


def new_run_id(prefix: str = "run") -> str:
    """Short unique identifier for a processing run."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


__all__ = [
    "run_part1",
    "run_part2",
    "new_run_id",
    "DerivedParameters",
    "sorted_finite",
]
