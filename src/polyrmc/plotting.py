"""Figures for a processed run.

Presentation only -- nothing here computes a reported quantity, and no tier-0
module imports it.

The one rule that carries over from the pipeline proper is how excisions are
drawn: a single-sample dropout is bridged invisibly, but a multi-sample gap
**breaks the plotted line**, so removed data is never implied to be real
signal. That is the same distinction :mod:`polyrmc.tier0.splice` records, and
it is why the trace is drawn from its segments rather than as one array.
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
from polyrmc.state import AnomalyType, RunState
from polyrmc.tier0.splice import render_segments

INK = "#141A21"
MUTED = "#7C8894"
RULE = "#DCE2E8"
ACCENT = "#8C1D2E"
RAW = "#9FB0C0"
FLAG = "#C8842F"
EXCISE = "#93312B"
GAIN = "#1B4D8F"  # ND filter changes: deliberately outside the crimson family

_TYPE_COLOUR = {
    AnomalyType.TRANSIENT_SPIKE: "#C8842F",
    AnomalyType.SATURATION: "#93312B",
    AnomalyType.BASELINE_DRIFT: "#2E6A52",
    AnomalyType.STRUCTURAL_NOISE: "#6A5AA0",
}


def _style(axis) -> None:
    axis.set_facecolor("white")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(RULE)
    axis.tick_params(colors=MUTED, labelsize=8, length=3)
    axis.grid(True, color=RULE, linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def plot_run(
    state: RunState,
    config: RunConfig,
    frame: pd.DataFrame,
    out_path: str | Path,
    dpi: int = 130,
) -> Path:
    """Two-panel figure: the cleaned trace, and the derived quantity.

    Anomaly regions are shaded in the colour of their classification, so a
    reviewer can see what the pipeline decided and where.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hours = frame["time_s"].to_numpy(dtype=float) / 3600.0
    raw = frame["signal_raw"].to_numpy(dtype=float)
    smoothed = frame["signal_smoothed"].to_numpy(dtype=float)
    mw = frame["mw_over_m0"].to_numpy(dtype=float)
    concentration = frame["concentration_g_per_cm3"].to_numpy(dtype=float)
    is_dilution = bool(np.isfinite(concentration).any())

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=not is_dilution, dpi=dpi,
        gridspec_kw={"height_ratios": [2.0, 1.4], "hspace": 0.32},
    )
    figure.patch.set_facecolor("white")

    # -- top: raw and smoothed trace -------------------------------------
    _style(top)
    top.plot(hours, raw, color=RAW, linewidth=0.35, alpha=0.7, label="raw", zorder=1)

    # Full-scale samples are limits, not measurements. Mark the ceiling when
    # the trace reaches it so a reader is not invited to read the plateau there
    # as real signal.
    if np.nanmax(raw) >= 1.0 - 1e-9:
        top.axhline(1.0, color=EXCISE, linewidth=0.9, linestyle="--", alpha=0.8, zorder=2)
        top.annotate(
            "detector full scale", xy=(hours[0], 1.0), xytext=(2, 3),
            textcoords="offset points", fontsize=7.5, color=EXCISE,
        )

    # Draw the smoothed trace segment by segment so excised stretches leave
    # a visible gap rather than a straight line across missing data.
    segments = render_segments(hours, smoothed, state.splice)
    for index, (segment_x, segment_y) in enumerate(segments):
        top.plot(
            segment_x, segment_y, color=INK, linewidth=0.6, zorder=3,
            label="smoothed" if index == 0 else None,
        )

    # An ND filter change rescales the channel by an unrecorded factor, so the
    # trace either side is not on one scale. This is the single most important
    # thing a reader of this figure needs to know.
    for position, sample in enumerate(state.nd_filter_changes):
        if sample >= hours.size:
            continue
        top.axvline(
            hours[sample], color=GAIN, linewidth=1.8, zorder=5,
            label="ND filter change - scale breaks here" if position == 0 else None,
        )
        bottom.axvline(hours[sample], color=GAIN, linewidth=1.8, zorder=5)

    seen: set[str] = set()
    for record in state.anomalies:
        colour = _TYPE_COLOUR.get(record.anomaly_type, MUTED)
        name = record.anomaly_type.value if record.anomaly_type else "unclassified"
        start = hours[record.start_index]
        end = hours[min(record.end_index, hours.size - 1)]
        top.axvspan(
            start, max(end, start + 1e-4), color=colour, alpha=0.20, zorder=0,
            label=name.replace("_", " ") if name not in seen else None,
        )
        seen.add(name)

    top.set_ylabel(f"{config.channel} scattering (counts)", fontsize=9, color=INK)
    # Legend goes at the foot of the figure, not over the axes: with six
    # possible entries it otherwise collides with the title.
    handles, labels = top.get_legend_handles_labels()

    window = state.loops["smoothing_window"].resolved_value
    title = (
        f"{state.run_id}   ·   channel {config.channel}   ·   "
        f"{len(frame):,} points over {hours[-1]:.1f} h   ·   "
        f"Savitzky-Golay window {window}, order {config.smoothing.polyorder}"
    )
    top.set_title(title, fontsize=9.5, color=INK, loc="left", pad=10)

    # -- bottom: the derived quantity -------------------------------------
    _style(bottom)
    if is_dilution:
        order = np.argsort(concentration)
        bottom.plot(
            concentration[order], frame["rayleigh_ratio_cm_inv"].to_numpy(float)[order],
            color=ACCENT, linewidth=0.9,
        )
        bottom.set_xlabel("concentration (g/cm$^3$)", fontsize=9, color=INK)
        bottom.set_ylabel("excess scattering", fontsize=9, color=INK)
    else:
        bottom.plot(hours, mw, color=ACCENT, linewidth=1.0)
        bottom.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")

        # The plateau test looks at the second half; show the reader the
        # same window it judged, and how much the trace moves inside it.
        if np.isfinite(mw).any():
            half = hours[len(hours) // 2]
            bottom.axvspan(half, hours[-1], color=ACCENT, alpha=0.06, zorder=0)
            final = float(np.nanmedian(mw[-200:]))
            bottom.annotate(
                f"M$_w$/M$_0$ = {final:.2f}",
                xy=(hours[-1], final), xytext=(-8, 6), textcoords="offset points",
                ha="right", fontsize=9, color=ACCENT, fontweight="bold",
            )
        bottom.set_xlabel("elapsed time (h)", fontsize=9, color=INK)
        bottom.set_ylabel("M$_w$/M$_0$", fontsize=9, color=INK)

    notes = []
    if state.splice.n_excised:
        notes.append(f"{state.splice.n_excised:,} samples excised")
    unclassified = sum(1 for r in state.anomalies if r.anomaly_type is None)
    if unclassified:
        notes.append(f"{unclassified} regions unclassified")
    if not config.optical.alpha:
        notes.append("alpha not calibrated")
    if handles:
        figure.legend(
            handles, labels, frameon=False, fontsize=7, labelcolor=MUTED,
            ncol=len(handles), loc="upper center",
            bbox_to_anchor=(0.5, 0.06), columnspacing=1.3, handlelength=1.5,
        )
    if notes:
        figure.text(
            0.5, -0.005, "   ·   ".join(notes), ha="center", fontsize=8, color=MUTED,
        )

    figure.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out_path


def plot_session(rows: list[dict], out_path: str | Path, dpi: int = 130) -> Path:
    """One figure for a whole session: final value per condition, flagged runs marked.

    Runs that never plateau are drawn hollow. Their endpoint is not a summary
    of anything, and the figure should not present it as comparable to the rest.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    usable = [r for r in rows if r.get("mw_over_m0_final") is not None]
    if not usable:
        raise ValueError("no completed runs to plot")

    def short(row: dict) -> str:
        return (row["run_id"].replace("liraglutide_sample_2_(unstable)", "")
                .strip("_").replace("in_", "") or row["run_id"])

    def sort_key(row: dict) -> tuple:
        """Order by salt series, then by concentration numerically.

        Sorting the labels as text puts 500 mM between 5000 and 5500, which
        makes the concentration trend unreadable.
        """
        name = short(row)
        match = re.search(r"(\d+(?:\.\d+)?)\s*mM", name)
        millimolar = float(match.group(1)) if match else 0.0
        return (0 if name.endswith("Gd") else 1, millimolar)

    usable.sort(key=sort_key)
    labels = [short(r) for r in usable]
    values = [float(r["mw_over_m0_final"]) for r in usable]

    def is_settled(row: dict) -> bool:
        """Only mark a run reliable if nothing disqualifies its endpoint."""
        if float(row.get("plateau_swing") or 0) > 0.5:
            return False
        if int(row.get("nd_filter_changes") or 0) > 0:
            return False  # scale breaks; endpoint not comparable to the start
        if int(row.get("at_full_scale") or 0) > 100:
            return False  # detector ran out of range for a meaningful stretch
        return True

    settled_flags = [is_settled(r) for r in usable]

    figure, axis = plt.subplots(figsize=(10, 4.6), dpi=dpi)
    figure.patch.set_facecolor("white")
    _style(axis)

    positions = np.arange(len(usable))
    for position, value, settled in zip(positions, values, settled_flags):
        axis.scatter(
            position, value, s=70, zorder=3,
            color=ACCENT if settled else "white",
            edgecolor=ACCENT, linewidth=1.4,
        )
        axis.vlines(position, 1.0, value, color=ACCENT, alpha=0.35, linewidth=1)

    axis.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=38, ha="right", fontsize=8, color=INK)
    axis.set_ylabel("M$_w$/M$_0$ at end of run", fontsize=9, color=INK)
    axis.set_title(
        "Aggregation index by condition   ·   hollow = endpoint not a meaningful "
        "summary (trace never plateaued, ND filter changed, or detector railed)",
        fontsize=9.5, color=INK, loc="left", pad=12,
    )

    figure.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out_path
