"""End-to-end processing: Part 1 (clean) and Part 2 (extract).

Part 1  ingest -> clean -> detect artifacts -> splice -> baseline-correct ->
        smooth, emitting the validated CSV.
Part 2  time -> concentration -> Kc/I_R vs c -> A2*Mw, Mw, A3, k_D, Mw/M0.

The CSV is the only thing that passes between them.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import numpy as np

from polyrmc.config import RunConfig
from polyrmc.io_csv import read_processed_csv, write_processed_csv
from polyrmc.provenance import write_sidecar
from polyrmc.state import (
    AnomalyType,
    Candidate,
    DerivedParameters,
    LoopState,
    PathChoice,
    RunState,
)
from polyrmc.tier0 import optics
from polyrmc.tier0.argen_io import gain_change_indices, load_argen_file
from polyrmc.tier0.anchors import propose_anchors
from polyrmc.tier0.censoring import apply_censoring, censor_saturation
from polyrmc.tier0.classify import classify_all
from polyrmc.tier0.columns import (
    concentration_column,
    mw_over_m0_column,
    rayleigh_column,
)
from polyrmc.tier0.detect import detect_anomalies, noise_floor
from polyrmc.tier0.m0 import select_m0
from polyrmc.tier0.dilution import plateau_swing
from polyrmc.tier0.fit_range import apply_fit_range, sorted_finite
from polyrmc.tier0.smoothing import propose_windows, savgol_smooth
from polyrmc.tier0.splice import apply_splice
from polyrmc.tier0.virial import build_kc_over_i, derive_parameters, fit_virial
from polyrmc.tier1.fit_range_selector import select_fit_range
from polyrmc.tier1.loop import Judge
from polyrmc.tier1.smoothing_selector import select_smoothing_window


def prepare_run(
    config: RunConfig, channel: str | None = None
) -> tuple[RunState, np.ndarray]:
    """Everything deterministic up to the point a window must be chosen.

    Load, clean, detect, splice, censor, and propose baseline anchors -- then
    stop. Stopping here is what lets a caller other than :func:`run_part1` make
    the smoothing judgment: the agent graph needs the safe band *before* it is
    resolved, so its two lanes can read the same shortlist and disagree about
    it. A single monolithic function could only ever be driven by the judge
    wired inside it.

    Returns the state and the mask of samples no judgment may treat as clean
    signal -- flagged regions and censored samples both.

    Parameters
    ----------
    channel:
        Scattering channel to process. Defaults to ``config.channel``, which is
        ``ls8`` -- the only channel this lab analyses. The other 15 are dropped
        at load and never reach any stage of the pipeline.
    """
    channel = channel or config.channel
    argen = load_argen_file(config.source_file, channel=channel)

    state = RunState(run_id=config.run_id, source_file=str(config.source_file))
    state.time_s = argen.time_s
    state.raw_signal = argen.data[channel].to_numpy(dtype=float)
    state.time_format = argen.time_format
    state.n_header_rows = argen.n_header_rows
    state.metadata_lines = argen.metadata_lines
    state.dropped_rows = argen.dropped_rows

    if not argen.is_uniformly_sampled:
        state.warnings.append(
            "sampling is non-uniform; time, not row index, is the x-axis"
        )

    # The ND filter rescales the beam. If it moves mid-run, readings either
    # side of the change are on different scales and cannot be compared.
    nd_column = next(
        (c for c in argen.data.columns if re.fullmatch(r"nd\s*_?filter", c, re.I)), None
    )
    state.nd_filter_changes = (
        gain_change_indices(argen.data[nd_column].to_numpy()) if nd_column else []
    )
    if state.nd_filter_changes:
        state.warnings.append(
            f"ND filter changes {len(state.nd_filter_changes)} time(s) during this "
            f"run (first at sample {state.nd_filter_changes[0]}). The scattering "
            "channel is rescaled by an amount the file does not record, so readings "
            "either side are NOT on the same scale. Mw/M0 spanning the change is "
            "not a physical quantity; restrict the analysis to one filter segment "
            "or supply the filter transmission."
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

    # Record what was done with each region. A region left 'pending' is an
    # unfinished judgment and blocks emission through the tool surface, which is
    # exactly what an unclassifiable region should do.
    for record in state.anomalies:
        if record.anomaly_type is None:
            continue
        if record.anomaly_type is AnomalyType.BASELINE_DRIFT:
            record.decision = "keep"
        elif record.start_index in state.splice.excised_indices:
            record.decision = "excise"
        else:
            record.decision = "keep"

    # Saturation is censoring, keyed on the declared detector ceiling. Masked
    # to NaN with the timestamps left in place; no computed column may carry a
    # value at these indices. Undeclared ceiling is its own path: nothing is
    # masked and the omission is reported, because reading the ceiling off the
    # trace is circular.
    state.spliced_signal, censoring = censor_saturation(
        state.spliced_signal,
        ceiling=config.detectors.saturation_ceiling,
        run_ceiling=config.detectors.saturation_run_ceiling,
        fraction_ceiling=config.detectors.saturation_fraction_ceiling,
    )
    # Two detectors can find saturation: the ceiling test above, and the
    # repeated-value test in the anomaly pass. They must agree on what counts as
    # censored, or a run whose detector railed without a declared ceiling gets
    # its points excised by one path while the censoring record stays empty --
    # and the check that no computed column carries a saturated value then
    # reports "skipped" on a run that plainly saturated.
    railed = sorted(
        index
        for record in state.anomalies
        if record.anomaly_type is AnomalyType.SATURATION_ARTIFACT
        for index in range(record.start_index, record.end_index + 1)
    )
    state.censored_indices = sorted(set(censoring.censored_indices) | set(railed))
    state.blocked.extend(censoring.blocked)
    state.warnings.extend(censoring.escalations)
    if railed and not censoring.applied:
        state.warnings.append(
            f"{len(railed)} sample(s) were flagged as saturated by the "
            "repeated-value detector and are treated as censored, even though no "
            "detector ceiling has been declared. Declaring the ceiling would let "
            "the ceiling test find railed samples this detector cannot see -- a "
            "detector held at its rail by a noisy input does not repeat exactly."
        )

    surviving = int(np.isfinite(state.spliced_signal).sum())
    if censoring.applied and surviving < config.smoothing.min_window:
        raise ValueError(
            f"censoring left {surviving} usable sample(s) of "
            f"{state.spliced_signal.size}: a declared ceiling of "
            f"{censoring.ceiling:g} sits at or below almost the entire trace. Either "
            "the ceiling is wrong for this instrument, or the acquisition railed "
            "throughout and there is nothing here to analyse."
        )
    state.paths.append(
        PathChoice(
            question="OPEN-6",
            decision="mask-and-exclude" if censoring.applied else "no ceiling declared",
            because=(
                f"{censoring.n_censored} sample(s) at or above {censoring.ceiling:g} "
                f"masked; {len(censoring.sustained_spans)} sustained span(s) marked "
                "censored"
            )
            if censoring.applied
            else "saturation_ceiling is not declared, so nothing was masked",
            alternatives=["cap-and-annotate (rejected)", "keep-and-flag (rejected)"],
            blocked=censoring.blocked,
        )
    )
    if censoring.sustained_spans:
        state.warnings.append(
            "the smoother interpolates across gaps for its filter pass and restores "
            "NaN afterwards, so no censored sample carries a value -- but samples "
            "within half a window of a sustained censored span were computed partly "
            "from that interpolation, and should be treated as such"
        )

    # Drift is detected and recorded, not corrected. Anchor selection decides
    # what counts as signal, and a curve fitted to anchors that were really part
    # of the kinetics yields a smooth, plausible, physically wrong trace that no
    # downstream check will flag. Until the anchor machinery and its tolerances
    # are calibrated against the golden runs, the specified fallback is to flag
    # and leave alone.
    drift_regions = [
        record
        for record in state.anomalies
        if record.anomaly_type is AnomalyType.BASELINE_DRIFT
    ]

    # Regions the detectors flagged -- including ones classification could not
    # name -- may not appear in an anchor. They are already suspect, so they are
    # not baseline by definition.
    suspect = np.zeros(state.raw_signal.size, dtype=bool)
    for record in state.anomalies:
        suspect[record.start_index : record.end_index + 1] = True
    suspect[state.censored_indices] = True

    state.anchor_candidates, state.transition_region = propose_anchors(
        state.time_s,
        state.spliced_signal,
        noise_floor=noise_floor(state.spliced_signal),
        config=config.baseline,
        excluded=suspect,
    )

    # Proposed, rendered, and left alone. Applying a correction is a commit call
    # requiring researcher approval, so the measurement path here always carries
    # the uncorrected trace.
    state.baseline = np.zeros_like(state.spliced_signal)
    state.corrected_signal = state.spliced_signal
    state.baseline_applied = False
    state.paths.append(
        PathChoice(
            question="OPEN-5",
            decision="propose anchors, do not correct",
            because=(
                f"{len(drift_regions)} drift region(s) detected; "
                f"{len(state.anchor_candidates)} anchor candidate(s) enumerated "
                f"outside the transition region {state.transition_region}. The "
                "default is no correction: applying one is a commit call, because "
                "anchor selection decides what counts as signal and no downstream "
                "check catches getting it wrong."
            ),
            alternatives=["poly over selected anchors", "piecewise_linear"],
        )
    )
    if state.transition_region is not None:
        low, high = state.transition_region
        covered = (high - low + 1) / state.raw_signal.size
        if covered > 0.5:
            state.warnings.append(
                f"the transition region spans {100 * covered:.0f}% of the run "
                f"(samples {low}-{high}), so almost nothing is left that could serve "
                "as baseline. On a dilution run this is expected -- the decay is the "
                "measurement, not drift -- and it is why no anchor is offered."
            )

    if drift_regions and len(state.anchor_candidates) < config.baseline.min_anchors:
        state.warnings.append(
            f"{len(drift_regions)} baseline-drift region(s) detected, but only "
            f"{len(state.anchor_candidates)} anchor candidate(s) survive the flatness, "
            "coverage and transition guards -- fewer than the minimum of "
            f"{config.baseline.min_anchors}. No correction is offerable on this run."
        )
    elif drift_regions:
        state.warnings.append(
            f"{len(drift_regions)} baseline-drift region(s) detected and "
            f"{len(state.anchor_candidates)} anchor candidate(s) proposed. Nothing was "
            "subtracted: select anchors and apply the correction explicitly."
        )

    state.noise_floor = float(noise_floor(state.spliced_signal))
    return state, suspect


def suspect_mask(state: RunState) -> np.ndarray:
    """Samples no judgment may treat as clean signal.

    Derived from state rather than passed around, so the window sweep, the
    anchor proposal and the M0 window cannot drift apart about what counts as
    suspect -- they all ask the same question of the same record.
    """
    mask = np.zeros(state.raw_signal.size, dtype=bool)
    for record in state.anomalies:
        mask[record.start_index : record.end_index + 1] = True
    if state.censored_indices:
        mask[state.censored_indices] = True
    return mask


def sweep_windows(
    state: RunState, config: RunConfig, suspect: np.ndarray | None = None
) -> list[Candidate]:
    """Compute the safe band and record it, without resolving it.

    Every candidate returned has already passed the whitening and landmark
    gates, so nothing downstream -- model or human -- can reach a pathological
    window. Recording the band on the state as an unresolved loop is what lets
    ``pick_window`` validate a later choice against it.
    """
    candidates = propose_windows(
        state.corrected_signal,
        config.smoothing,
        exclude=suspect,
    )
    state.loops["smoothing_window"] = LoopState(
        name="smoothing_window",
        max_iterations=config.loop.max_iterations,
        candidates_by_iteration=[candidates],
    )
    return candidates


def finish_run(
    state: RunState,
    config: RunConfig,
    channel: str,
    window_length: int,
    output_csv: str | Path | None = None,
) -> tuple[RunState, Path]:
    """Smooth at the chosen window, compute the columns, and emit.

    Split from :func:`prepare_run` so that whoever made the smoothing judgment
    -- the tier-1 loop, the two-lane judge, or a researcher -- hands the same
    answer to the same code.
    """
    state.window_length = int(window_length)
    state.polyorder = config.smoothing.polyorder
    state.smoothed_signal = savgol_smooth(
        state.corrected_signal, state.window_length, state.polyorder
    )

    # Every column below routes through an explicit branch, including the branch
    # where the question behind it has not been answered yet. An unanswered
    # question costs that column and a recorded reason; it never silently
    # becomes an assumption.
    concentration_result = concentration_column(
        state.time_s,
        experiment_type=config.experiment_type,
        c0_g_per_cm3=config.initial_concentration,
        flow_rate_cm3_per_s=(
            config.dilution.flow_rate_cm3_per_s if config.dilution else None
        ),
        cell_volume_cm3=config.dilution.cell_volume_cm3 if config.dilution else None,
    )
    state.concentration = concentration_result.values
    state.paths.append(concentration_result.path)
    state.blocked.extend(concentration_result.path.blocked)

    rayleigh_result = rayleigh_column(
        state.smoothed_signal,
        ls_units=config.ls_units,
        buffer_blank=config.solvent_blank_counts,
        toluene_ref=config.toluene_ref,
        r_toluene_cm_inv=config.r_toluene_cm_inv,
        alpha=config.optical.alpha,
    )
    rayleigh = rayleigh_result.values
    state.paths.append(rayleigh_result.path)
    state.blocked.extend(rayleigh_result.path.blocked)
    calibrated = rayleigh_result.available

    # M0 divides every value in the aggregation index, so its window is chosen
    # and its sensitivity measured rather than defaulted to the first sample.
    m0 = select_m0(
        state.time_s,
        state.smoothed_signal - config.solvent_blank_counts,
        noise_floor=noise_floor(state.smoothed_signal),
        max_window_fraction=config.m0_max_window,
        sensitivity_ceiling=config.m0_sensitivity_ceiling,
        config=config.baseline,
        excluded=suspect_mask(state),
        transition=state.transition_region,
    )
    state.m0 = m0
    state.blocked.extend(m0.blocked)
    state.warnings.extend(m0.blocked)
    state.warnings.extend(m0.escalations)

    mw_result = mw_over_m0_column(
        state.smoothed_signal,
        concentration=state.concentration,
        i_rayleigh=rayleigh if calibrated else None,
        buffer_blank=config.solvent_blank_counts,
        reference=m0.value if not calibrated else None,
    )
    mw_over_m0 = mw_result.values
    state.paths.append(mw_result.path)
    state.blocked.extend(mw_result.path.blocked)
    for note in mw_result.path.alternatives:
        state.warnings.append(note)

    excess = optics.excess_scattering(state.smoothed_signal, config.solvent_blank_counts)
    try:
        reference = float(np.nanmedian(excess[np.isfinite(excess)][:50]))
        excess_noise = noise_floor(excess)
        if excess_noise > 0 and reference < 10.0 * excess_noise:
            state.warnings.append(
                f"excess scattering at run start ({reference:.6g}) is only "
                f"{reference / excess_noise:.1f}x the noise level: Mw/M0 on this "
                "channel is dominated by the blank subtraction and should not be "
                "trusted"
            )

        blank_fraction = (
            config.solvent_blank_counts / (reference + config.solvent_blank_counts)
            if reference + config.solvent_blank_counts > 0
            else 1.0
        )
        if blank_fraction > 0.5:
            state.warnings.append(
                f"the blank is {100 * blank_fraction:.0f}% of the raw signal at run "
                "start, so Mw/M0 rests on a small difference between two similar "
                "numbers and is highly sensitive to the blank value"
            )

        swing = plateau_swing(excess)
        if swing > 0.5:
            state.warnings.append(
                f"the trace does not plateau: it still swings by {100 * swing:.0f}% "
                "of its own level over the second half of the run. An endpoint "
                "Mw/M0 is not a meaningful summary of a trace like this -- the "
                "value depends on where acquisition happened to stop"
            )
    except ValueError as error:
        state.warnings.append(f"Mw/M0 diagnostics not computed: {error}")

    # The last word on saturation, applied once over every computed column at
    # the end. A per-column rule would be re-derived at each new column and
    # eventually forgotten; doing it here makes "no censored index carries a
    # value" a single checkable statement.
    apply_censoring(
        {
            "concentration": state.concentration,
            "smoothed": state.smoothed_signal,
            "rayleigh": rayleigh,
            "mw_over_m0": mw_over_m0,
        },
        state.censored_indices,
    )

    output_csv = Path(output_csv) if output_csv else _default_output(config, channel)
    write_processed_csv(
        output_csv,
        {
            "seconds": state.time_s,
            "concentration": state.concentration,
            # The cleaned and spliced channel, not the raw one: the excised
            # points are excised in this file too, or the consumer would see
            # values the pipeline decided were not measurements.
            "ls8": state.spliced_signal,
            "ls8_smoothed": state.smoothed_signal,
            "I_rayleigh": rayleigh,
            "Mw_over_M0": mw_over_m0,
        },
        metadata={
            "run_id": config.run_id,
            "channel": channel,
            "source_file": Path(config.source_file).name,
            "experiment_type": config.experiment_type,
            "smoothing_window": state.window_length,
            "smoothing_polyorder": config.smoothing.polyorder,
            "i_rayleigh_absolute": calibrated,
            "n_excised": state.splice.n_excised,
            "n_censored": len(state.censored_indices),
            "baseline_applied": state.baseline_applied,
            "m0_method": m0.method,
            "m0_window": f"{m0.index_range[0]}-{m0.index_range[1]}",
            "m0_sensitivity": f"{m0.sensitivity:.6g}",
        },
        source_header=state.metadata_lines,
        channel=channel,
    )
    write_sidecar(state, config, output_csv)

    # Automatically generate PNG figures alongside the output CSV and sidecar
    try:
        import pandas as pd
        from polyrmc import figures
    except Exception as error:  # matplotlib or pandas missing: figures are optional
        state.warnings.append(f"PNG figures skipped entirely: {error}")
    else:
        fig_dir = output_csv.parent / "figures"
        df_plot = pd.DataFrame(
            {
                "seconds": state.time_s,
                "ls8": state.spliced_signal,
                "ls8_smoothed": state.smoothed_signal,
                "concentration": state.concentration,
                "I_rayleigh": rayleigh,
                "Mw_over_M0": mw_over_m0,
            }
        )
        for name, draw in (
            ("raw", figures.plot_raw_trace),
            ("smoothing", figures.plot_smoothing_detail),
            ("derived", figures.plot_derived_quantities),
        ):
            _figure(state, name, draw, state, config, df_plot, fig_dir / f"{state.run_id}_{name}.png")

    return state, output_csv


def _figure(state: RunState, name: str, draw, *args, **kwargs) -> None:
    """Draw one figure, and let a failure cost only that figure.

    These were once drawn inside a single ``try``, which meant one broken plot
    took the whole set down with it and said so in a single warning naming only
    the first thing that failed. That is how this module spent a while importing
    a ``plot_virial_fit`` that did not exist while silently producing no figures
    at all. A figure is a presentation artifact: failing to draw one must cost
    that one figure, and must name it.
    """
    try:
        draw(*args, **kwargs)  # the drawing functions create their own directory
    except Exception as error:  # noqa: BLE001 - no figure is worth losing a run over
        state.warnings.append(
            f"figure '{name}' not written: {type(error).__name__}: {error}"
        )


def run_part1(
    config: RunConfig,
    channel: str | None = None,
    judge: Judge | None = None,
    output_csv: str | Path | None = None,
) -> tuple[RunState, Path]:
    """Clean one channel and emit the validated CSV plus its provenance sidecar.

    The batch path: prepare, let the tier-1 loop resolve the window, finish.
    The agent drives the same three stages itself, so both routes run identical
    measurement code and differ only in who makes the judgment.
    """
    channel = channel or config.channel
    state, suspect = prepare_run(config, channel=channel)

    window_loop = select_smoothing_window(
        state.corrected_signal,
        smoothing_config=config.smoothing,
        loop_config=config.loop,
        judge=judge,
        context={"channel": channel, "run_id": config.run_id},
        exclude=suspect,
    )
    state.loops["smoothing_window"] = window_loop
    if window_loop.resolved_value is None:
        raise RuntimeError(
            "smoothing window selection produced no usable option; the trace may be "
            "too short or too noisy for any window to pass the safety gates"
        )
    return finish_run(
        state, config, channel, int(window_loop.resolved_value), output_csv
    )


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

    concentration = frame["concentration"].to_numpy(dtype=float)
    finite_c = concentration[np.isfinite(concentration)]
    if finite_c.size == 0:
        raise ValueError(
            "no concentration axis in this file. The virial expansion is fitted "
            "against concentration, so it needs a run whose concentration is known "
            "-- see the OPEN-1 path recorded in the sidecar for why this column is "
            "blocked."
        )
    if finite_c.size > 1 and np.allclose(finite_c, finite_c[0]):
        raise ValueError(
            f"every sample sits at one concentration ({finite_c[0]:.6g} g/cm^3). "
            "That is a fixed-concentration aggregation run: it yields one point on "
            "the Kc/I_R versus c axis, so there is no slope to fit and no A2. Mw/M0 "
            "in the emitted CSV is what this run type measures."
        )

    # The intensity axis has two branches, because A2*Mw is calibration-free.
    # Kc/I = (1/alpha)*(1/Mw + 2*A2*c + ...), so the slope-to-intercept ratio --
    # and therefore A2*Mw -- is untouched by any constant y-scale. Absolute Mw
    # and A2 are not, which is why they are withheld on the uncalibrated branch
    # rather than reported with a silent factor in them.
    intensity = frame["I_rayleigh"].to_numpy(dtype=float)
    calibrated = str(metadata.get("i_rayleigh_absolute", "False")).lower() == "true"
    if not (calibrated and np.isfinite(intensity).any()):
        calibrated = False
        intensity = frame["ls8_smoothed"].to_numpy(dtype=float) - (
            config.solvent_blank_counts
        )
        state.paths.append(
            PathChoice(
                question="OPEN-2",
                decision="fit on the smoothed channel",
                because=(
                    "I_rayleigh is not on an absolute scale in this file, so the fit "
                    "runs on the blank-subtracted smoothed channel. A constant "
                    "y-scale cancels in slope/intercept, so A2*Mw and k_D are "
                    "unaffected."
                ),
                alternatives=["fit on I_rayleigh in cm^-1"],
                blocked=["Mw", "A2", "A3"],
            )
        )
        state.blocked.extend(["Mw", "A2", "A3"])
    if not np.isfinite(intensity).any():
        raise ValueError(
            "no usable intensity axis: both rayleigh_ratio_cm_inv and "
            "signal_smoothed are empty"
        )

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

    mw_over_m0 = frame["Mw_over_M0"].to_numpy(dtype=float)
    final = float(np.nanmedian(mw_over_m0[-50:])) if np.isfinite(mw_over_m0).any() else None

    state.parameters = derive_parameters(
        state.fit, calibrated=calibrated, k_h=config.k_h, mw_over_m0_final=final
    )
    if not calibrated:
        state.warnings.append(
            "uncalibrated run: A2*Mw and k_D are reported; Mw, A2, and A3 are not, "
            "since they would silently carry a factor of alpha"
        )

    # The virial figure is drawn here rather than in finish_run, because this is
    # where the fit exists. Part 1 has no fit -- it emits the CSV this function
    # reads -- so the call that used to sit there could never have fired.
    try:
        from polyrmc import figures
    except Exception as error:  # matplotlib missing: the figure is optional
        state.warnings.append(f"virial figure skipped: {error}")
    else:
        fig_dir = Path(processed_csv).parent / "figures"
        _figure(
            state,
            "virial",
            figures.plot_virial_fit,
            state,
            config,
            fig_dir / f"{state.run_id}_virial.png",
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
