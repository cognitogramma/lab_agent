"""The per-run figure set, and the cross-run comparison figures.

:mod:`polyrmc.plotting` draws the two-panel overview. This module breaks the
same run into separate figures, because at ~80,000 points one combined figure
hides exactly what a reviewer needs to check: whether the smoothing window
tracked the trace, and whether the blank sat where it should.

Presentation only. Nothing here computes a reported quantity, and no tier-0
module imports it. Statistics computed below (residual spread, blank medians)
exist to label an axis, never to feed a result.

The excision rule from :mod:`polyrmc.plotting` carries over: a multi-sample gap
breaks the plotted line rather than being bridged, so removed data is never
drawn as though it were signal.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written, never displayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polyrmc.config import RunConfig
from polyrmc.plotting import (
    ACCENT,
    EXCISE,
    GAIN,
    INK,
    MUTED,
    RAW,
    _TYPE_COLOUR,
    _style,
)
from polyrmc.state import RunState
from polyrmc.tier0.splice import render_segments

BLANK = "#2E6A52"


def _mark_events(axis, hours: np.ndarray, state: RunState) -> None:
    """ND-filter changes and anomaly regions, drawn identically on every panel."""
    for position, sample in enumerate(state.nd_filter_changes):
        if sample >= hours.size:
            continue
        axis.axvline(
            hours[sample],
            color=GAIN,
            linewidth=1.6,
            zorder=5,
            label="ND filter change" if position == 0 else None,
        )
    seen: set[str] = set()
    for record in state.anomalies:
        colour = _TYPE_COLOUR.get(record.anomaly_type, MUTED)
        name = record.anomaly_type.value if record.anomaly_type else "unclassified"
        start = hours[record.start_index]
        end = hours[min(record.end_index, hours.size - 1)]
        axis.axvspan(
            start,
            max(end, start + 1e-4),
            color=colour,
            alpha=0.20,
            zorder=0,
            label=name.replace("_", " ") if name not in seen else None,
        )
        seen.add(name)


def _footer(figure, axis) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            frameon=False,
            fontsize=7,
            labelcolor=MUTED,
            ncol=min(len(handles), 6),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.045),
            columnspacing=1.3,
            handlelength=1.5,
        )


def _save(figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out_path


def plot_raw_trace(
    state: RunState,
    config: RunConfig,
    frame: pd.DataFrame,
    out_path: str | Path,
    dpi: int = 130,
) -> Path:
    """The raw ls8 channel exactly as the instrument recorded it.

    Nothing is smoothed, excised, or bridged. This is the figure to read when
    asking what the detector actually saw.
    """
    hours = frame["time_s"].to_numpy(dtype=float) / 3600.0
    raw = frame["signal_raw"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(11, 4.6), dpi=dpi)
    figure.patch.set_facecolor("white")
    _style(axis)

    axis.plot(hours, raw, color=INK, linewidth=0.35, alpha=0.85, zorder=3, label="raw ls8")
    _mark_events(axis, hours, state)

    railed = int(np.sum(np.abs(raw - 1.0) <= 1e-9))
    if railed:
        axis.axhline(
            1.0,
            color=EXCISE,
            linewidth=1.0,
            linestyle="--",
            zorder=4,
            label=f"detector full scale ({railed:,} samples)",
        )

    axis.set_xlabel("elapsed time (h)", fontsize=9, color=INK)
    axis.set_ylabel(f"{config.channel} scattering (counts)", fontsize=9, color=INK)
    axis.set_title(
        f"{state.run_id}   .   raw {config.channel}, unprocessed   .   "
        f"{len(frame):,} points over {hours[-1]:.1f} h",
        fontsize=9.5,
        color=INK,
        loc="left",
        pad=10,
    )
    _footer(figure, axis)
    return _save(figure, Path(out_path))


def plot_smoothing_detail(
    state: RunState,
    config: RunConfig,
    frame: pd.DataFrame,
    out_path: str | Path,
    dpi: int = 130,
) -> Path:
    """Raw against smoothed, plus a zoom at native resolution.

    At full extent an 80,000-point trace and its smooth overlay are visually
    identical, which tells a reviewer nothing. The lower panel shows a short
    stretch unaggregated, where the window's effect is actually legible. The
    stretch chosen is the widest local swing, since that is where an
    over-aggressive window would do visible damage.
    """
    hours = frame["time_s"].to_numpy(dtype=float) / 3600.0
    raw = frame["signal_raw"].to_numpy(dtype=float)
    smoothed = frame["signal_smoothed"].to_numpy(dtype=float)
    window = state.loops["smoothing_window"].resolved_value

    figure, (full, zoom) = plt.subplots(
        2,
        1,
        figsize=(11, 6.6),
        dpi=dpi,
        gridspec_kw={"height_ratios": [1.7, 1.0], "hspace": 0.34},
    )
    figure.patch.set_facecolor("white")

    _style(full)
    full.plot(hours, raw, color=RAW, linewidth=0.35, alpha=0.75, zorder=1, label="raw")
    for index, (seg_x, seg_y) in enumerate(render_segments(hours, smoothed, state.splice)):
        full.plot(
            seg_x,
            seg_y,
            color=INK,
            linewidth=0.7,
            zorder=3,
            label="smoothed" if index == 0 else None,
        )
    _mark_events(full, hours, state)
    full.set_ylabel(f"{config.channel} (counts)", fontsize=9, color=INK)
    full.set_title(
        f"{state.run_id}   .   Savitzky-Golay window {window}, order "
        f"{config.smoothing.polyorder}",
        fontsize=9.5,
        color=INK,
        loc="left",
        pad=10,
    )

    _style(zoom)
    span = max(int(hours.size * 0.01), 400)
    start = 0
    if hours.size > span * 2:
        # Excised stretches leave all-NaN slices; score those as unusable rather
        # than letting nanmax warn and return NaN.
        starts = np.arange(0, hours.size - span, max(span // 4, 1))
        spreads = []
        for i in starts:
            chunk = smoothed[i : i + span]
            chunk = chunk[np.isfinite(chunk)]
            spreads.append(float(chunk.max() - chunk.min()) if chunk.size else -np.inf)
        if spreads and np.isfinite(spreads).any():
            start = int(starts[int(np.argmax(spreads))])
    stop = min(start + span, hours.size)

    zoom.plot(
        hours[start:stop],
        raw[start:stop],
        color=RAW,
        linewidth=0.8,
        marker="o",
        markersize=1.4,
        alpha=0.85,
        label="raw",
    )
    zoom.plot(
        hours[start:stop], smoothed[start:stop], color=ACCENT, linewidth=1.6, label="smoothed"
    )
    zoom.set_xlabel("elapsed time (h)", fontsize=9, color=INK)
    zoom.set_ylabel(f"{config.channel} (counts)", fontsize=9, color=INK)
    zoom.set_title(
        f"zoom: {stop - start:,} points at native resolution "
        f"({hours[start]:.2f}-{hours[stop - 1]:.2f} h), the widest local swing",
        fontsize=8.5,
        color=MUTED,
        loc="left",
        pad=6,
    )
    zoom.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc="best")

    _footer(figure, full)
    return _save(figure, Path(out_path))


def plot_derived_quantities(
    state: RunState,
    config: RunConfig,
    frame: pd.DataFrame,
    out_path: str | Path,
    dpi: int = 130,
) -> Path:
    """Excess Rayleigh ratio and the aggregation index against time."""
    hours = frame["time_s"].to_numpy(dtype=float) / 3600.0
    excess = frame["rayleigh_ratio_cm_inv"].to_numpy(dtype=float)
    mw = frame["mw_over_m0"].to_numpy(dtype=float)

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True, dpi=dpi, gridspec_kw={"hspace": 0.18}
    )
    figure.patch.set_facecolor("white")

    _style(top)
    top.plot(hours, excess, color=ACCENT, linewidth=0.7)
    _mark_events(top, hours, state)
    top.set_ylabel("excess Rayleigh ratio (cm$^{-1}$)", fontsize=9, color=INK)
    calibration = "" if config.optical.alpha else "   .   alpha not calibrated, scale is relative"
    top.set_title(
        f"{state.run_id}   .   derived quantities{calibration}",
        fontsize=9.5,
        color=INK,
        loc="left",
        pad=10,
    )

    _style(bottom)
    bottom.plot(hours, mw, color=INK, linewidth=0.9)
    bottom.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    if np.isfinite(mw).any():
        final = float(np.nanmedian(mw[-200:]))
        bottom.axvspan(hours[len(hours) // 2], hours[-1], color=ACCENT, alpha=0.06, zorder=0)
        bottom.annotate(
            f"M$_w$/M$_0$ = {final:.2f}",
            xy=(hours[-1], final),
            xytext=(-8, 6),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=ACCENT,
            fontweight="bold",
        )
    bottom.set_xlabel("elapsed time (h)", fontsize=9, color=INK)
    bottom.set_ylabel("M$_w$/M$_0$", fontsize=9, color=INK)

    _footer(figure, top)
    return _save(figure, Path(out_path))


def plot_blank_comparison(
    state: RunState,
    config: RunConfig,
    frame: pd.DataFrame,
    blank_hours: np.ndarray,
    blank_signal: np.ndarray,
    blank_name: str,
    out_path: str | Path,
    dpi: int = 130,
) -> Path:
    """The sample against the blank that was subtracted from it.

    The blank sets the solvent baseline, so a blank recorded on a different day
    than its sample can shift every excess value in the run. Drawing them on one
    axis makes that visible instead of leaving it implicit in a single number.
    """
    hours = frame["time_s"].to_numpy(dtype=float) / 3600.0
    raw = frame["signal_raw"].to_numpy(dtype=float)
    blank_level = float(np.nanmedian(blank_signal)) if blank_signal.size else float("nan")

    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 6.2),
        dpi=dpi,
        gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.34},
    )
    figure.patch.set_facecolor("white")

    _style(top)
    top.plot(hours, raw, color=INK, linewidth=0.35, alpha=0.8, label="sample (raw ls8)")
    if np.isfinite(blank_level):
        top.axhline(
            blank_level,
            color=BLANK,
            linewidth=1.4,
            linestyle="--",
            zorder=4,
            label=f"blank median = {blank_level:.5g}",
        )
    top.set_ylabel(f"{config.channel} (counts)", fontsize=9, color=INK)
    top.set_xlabel("elapsed time (h)", fontsize=9, color=INK)
    top.set_title(
        f"{state.run_id}   .   sample against its paired blank   .   {blank_name}",
        fontsize=9.5,
        color=INK,
        loc="left",
        pad=10,
    )
    top.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc="best")

    _style(bottom)
    if blank_signal.size:
        bottom.plot(blank_hours, blank_signal, color=BLANK, linewidth=0.5)
        spread = float(np.nanmax(blank_signal) - np.nanmin(blank_signal))
        relative = spread / blank_level if blank_level else float("nan")
        bottom.set_title(
            f"blank trace: {blank_signal.size:,} points over {blank_hours[-1]:.2f} h, "
            f"peak-to-peak {spread:.3g} ({relative:.1%} of its median)",
            fontsize=8.5,
            color=MUTED,
            loc="left",
            pad=6,
        )
    bottom.set_xlabel("blank elapsed time (h)", fontsize=9, color=INK)
    bottom.set_ylabel(f"blank {config.channel}", fontsize=9, color=INK)

    return _save(figure, Path(out_path))


def plot_residual_diagnostics(
    state: RunState,
    config: RunConfig,
    frame: pd.DataFrame,
    out_path: str | Path,
    dpi: int = 130,
) -> Path:
    """Why this smoothing window was accepted.

    The residual is what the filter removed. If it is structureless the filter
    took noise and nothing else; visible structure means real signal was
    absorbed. The autocorrelation panel is the same quantity the selector gated
    on, drawn so a reader can check the gate rather than trust it.
    """
    raw = frame["signal_raw"].to_numpy(dtype=float)
    smoothed = frame["signal_smoothed"].to_numpy(dtype=float)
    hours = frame["time_s"].to_numpy(dtype=float) / 3600.0
    residual = raw - smoothed
    finite = residual[np.isfinite(residual)]
    window = state.loops["smoothing_window"].resolved_value

    figure = plt.figure(figsize=(11, 6.4), dpi=dpi)
    figure.patch.set_facecolor("white")
    grid = figure.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.36, wspace=0.22)

    trace = figure.add_subplot(grid[0, :])
    _style(trace)
    trace.plot(hours, residual, color=MUTED, linewidth=0.3, alpha=0.9)
    trace.axhline(0.0, color=INK, linewidth=0.8, linestyle=":")
    trace.set_xlabel("elapsed time (h)", fontsize=9, color=INK)
    trace.set_ylabel("residual (raw - smoothed)", fontsize=9, color=INK)
    trace.set_title(
        f"{state.run_id}   .   smoothing residual at window {window}   .   "
        "structureless residual means the filter took noise only",
        fontsize=9.5,
        color=INK,
        loc="left",
        pad=10,
    )

    histogram = figure.add_subplot(grid[1, 0])
    _style(histogram)
    if finite.size:
        histogram.hist(finite, bins=80, color=RAW, edgecolor="white", linewidth=0.3)
        histogram.axvline(0.0, color=ACCENT, linewidth=1.0, linestyle="--")
        histogram.set_title(
            f"sigma = {float(np.std(finite)):.4g}", fontsize=8.5, color=MUTED, loc="left"
        )
    histogram.set_xlabel("residual", fontsize=9, color=INK)
    histogram.set_ylabel("count", fontsize=9, color=INK)

    correlogram = figure.add_subplot(grid[1, 1])
    _style(correlogram)
    if finite.size > 50:
        centred = finite - float(np.mean(finite))
        denominator = float(np.dot(centred, centred))
        lags = np.arange(1, 21)
        values = [
            float(np.dot(centred[:-lag], centred[lag:]) / denominator) if denominator else 0.0
            for lag in lags
        ]
        colours = [ACCENT if abs(v) > config.smoothing.max_residual_autocorr else MUTED
                   for v in values]
        correlogram.bar(lags, values, color=colours, width=0.7)
        correlogram.axhline(
            config.smoothing.max_residual_autocorr,
            color=ACCENT,
            linewidth=0.9,
            linestyle="--",
        )
        correlogram.set_title(
            f"lag-1 = {values[0]:+.3f}   .   positive gate at "
            f"{config.smoothing.max_residual_autocorr:+.2f}",
            fontsize=8.5,
            color=MUTED,
            loc="left",
        )
    correlogram.axhline(0.0, color=INK, linewidth=0.8)
    correlogram.set_xlabel("lag (samples)", fontsize=9, color=INK)
    correlogram.set_ylabel("autocorrelation", fontsize=9, color=INK)

    return _save(figure, Path(out_path))


# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------


def _series_of(run_id: str) -> str:
    """Which salt ladder a run belongs to.

    Gd and NaCl are separate experiments. Plotting them on one concentration
    axis would invite a comparison between points that share nothing but a
    number.
    """
    lowered = run_id.lower()
    if "gd" in lowered:
        return "guanidine"
    if "nacl" in lowered:
        return "NaCl"
    return "buffer only"


def _millimolar(run_id: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*mM", run_id)
    return float(match.group(1)) if match else 0.0


def _is_settled(row: dict) -> bool:
    """Whether the endpoint is a meaningful summary of the run."""
    if float(row.get("plateau_swing") or 0) > 0.5:
        return False
    if int(row.get("nd_filter_changes") or 0) > 0:
        return False
    if int(row.get("at_full_scale") or 0) > 100:
        return False
    return True


def plot_series_comparison(rows: list[dict], out_path: str | Path, dpi: int = 130) -> Path:
    """Aggregation index against salt concentration, one panel per ladder.

    Hollow markers are runs whose endpoint is not a meaningful summary -- the
    trace never plateaued, the ND filter moved, or the detector railed. They are
    drawn so the gap is visible, not so the value can be read off.
    """
    usable = [r for r in rows if r.get("mw_over_m0_final") is not None]
    if not usable:
        raise ValueError("no completed runs to plot")

    grouped: dict[str, list[dict]] = {}
    for row in usable:
        grouped.setdefault(_series_of(row["run_id"]), []).append(row)

    ladders = [name for name in ("guanidine", "NaCl") if name in grouped]
    if not ladders:
        ladders = sorted(grouped)

    figure, axes = plt.subplots(
        1, len(ladders), figsize=(5.6 * len(ladders), 4.6), dpi=dpi, squeeze=False
    )
    figure.patch.set_facecolor("white")

    control = grouped.get("buffer only", [])
    control_value = (
        float(np.median([float(r["mw_over_m0_final"]) for r in control])) if control else None
    )

    for axis, name in zip(axes[0], ladders):
        _style(axis)
        points = sorted(grouped[name], key=lambda r: _millimolar(r["run_id"]))
        x = [_millimolar(r["run_id"]) for r in points]
        y = [float(r["mw_over_m0_final"]) for r in points]
        settled = [_is_settled(r) for r in points]

        axis.plot(x, y, color=ACCENT, linewidth=1.0, alpha=0.45, zorder=2)
        for xi, yi, ok in zip(x, y, settled):
            axis.scatter(
                xi,
                yi,
                s=72,
                zorder=3,
                color=ACCENT if ok else "white",
                edgecolor=ACCENT,
                linewidth=1.4,
            )

        axis.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
        if control_value is not None:
            axis.axhline(
                control_value,
                color=BLANK,
                linewidth=1.1,
                linestyle="--",
                label=f"buffer-only control = {control_value:.2f}",
            )
            axis.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc="best")

        axis.set_xlabel(f"{name} concentration (mM)", fontsize=9, color=INK)
        axis.set_ylabel("M$_w$/M$_0$ at end of run", fontsize=9, color=INK)
        axis.set_title(f"{name} ladder   .   {len(points)} runs", fontsize=9.5,
                       color=INK, loc="left", pad=10)

    figure.suptitle(
        "Aggregation index by salt concentration   .   hollow = endpoint not a "
        "meaningful summary (no plateau, ND filter moved, or detector railed)",
        fontsize=9.5,
        color=INK,
        x=0.02,
        ha="left",
        y=1.02,
    )
    return _save(figure, Path(out_path))


def plot_blank_audit(rows: list[dict], out_path: str | Path, dpi: int = 130) -> Path:
    """Blank level per cell, with the acquisition-date split marked.

    Cells whose blank was recorded on a different day than the sample are the
    ones where instrument drift can move every excess value in that run.
    """
    usable = [r for r in rows if r.get("blank_level") is not None]
    if not usable:
        raise ValueError("no runs carry a blank level")

    usable.sort(key=lambda r: int(r.get("cell") or 0))
    cells = [int(r.get("cell") or 0) for r in usable]
    levels = [float(r["blank_level"]) for r in usable]
    same_day = [bool(r.get("blank_same_day", True)) for r in usable]

    figure, axis = plt.subplots(figsize=(10, 4.4), dpi=dpi)
    figure.patch.set_facecolor("white")
    _style(axis)

    for cell, level, ok in zip(cells, levels, same_day):
        axis.scatter(
            cell,
            level,
            s=76,
            zorder=3,
            color=BLANK if ok else "white",
            edgecolor=BLANK if ok else EXCISE,
            linewidth=1.5,
        )

    median = float(np.median(levels))
    axis.axhline(median, color=MUTED, linewidth=0.9, linestyle=":",
                 label=f"median across cells = {median:.5g}")
    axis.set_xticks(cells)
    axis.set_xlabel("cell number", fontsize=9, color=INK)
    axis.set_ylabel("blank median (counts)", fontsize=9, color=INK)
    axis.set_title(
        "Blank level per cell   .   hollow with red edge = blank acquired on a "
        "different day than its sample",
        fontsize=9.5,
        color=INK,
        loc="left",
        pad=10,
    )
    axis.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc="best")
    return _save(figure, Path(out_path))
