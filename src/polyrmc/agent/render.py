"""Render what the visual judge looks at.

Design section 6.7 stage 2. The VLM lane is supposed to be an *independent*
reading of the same run -- it catches over-smoothing that the residual
statistics tolerate but the eye rejects. Without a picture it has nothing to
read, returns nothing, and the two-lane design silently collapses to one lane.
That is worse than having a single lane on purpose, because the machinery still
reports a judge status as though two readings had happened.

So this module exists to make the second lane real. Three properties matter:

**The overlays are labelled by window, not by position.** The lane answers with
a window length, and it must be able to name one from what it sees. Legend
entries carry the number.

**The raw scatter stays visible underneath.** The whole point of the visual
check is comparing a candidate against the data, not against the other
candidates. Faint points, thin overlays.

**Nothing is measured off the axes.** Ticks are present for orientation, but
the rubric forbids reading values from them, and the lane answers with a choice
rather than a number it derived.

Rendering is deliberately cheap: a few thousand plotted points at modest DPI is
enough to see a rounded onset, and a full-resolution render of eighty thousand
samples costs seconds and buys nothing.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # no display on a lab machine or in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from polyrmc.state import RunState  # noqa: E402
from polyrmc.tier0.smoothing import savgol_smooth  # noqa: E402

# Plotted points. Enough to show the shape of a transition; far fewer than the
# series, because the lane is judging shape rather than counting samples.
MAX_POINTS = 3000

DPI = 110


def _thin(values: np.ndarray, limit: int = MAX_POINTS) -> np.ndarray:
    """Evenly spaced indices, so the shape survives but the cost does not."""
    n = values.size
    if n <= limit:
        return np.arange(n)
    return np.linspace(0, n - 1, limit).astype(int)


def render_window_overlays(
    state: RunState,
    windows: list[int],
    polyorder: int = 2,
    dpi: int = DPI,
) -> bytes | None:
    """The raw trace with each shortlisted window drawn over it, as PNG bytes.

    Returns None when there is nothing to draw, which the lane reports as a
    skipped reading rather than treating as a verdict.
    """
    if state.time_s is None or state.spliced_signal is None or not windows:
        return None

    time_s = np.asarray(state.time_s, dtype=float)
    signal = np.asarray(state.spliced_signal, dtype=float)
    if not np.isfinite(signal).any():
        return None

    keep = _thin(time_s)
    hours = time_s / 3600.0

    figure, axis = plt.subplots(figsize=(9.0, 5.0), dpi=dpi)
    axis.plot(
        hours[keep],
        signal[keep],
        ".",
        markersize=1.6,
        color="#9AA0A6",
        alpha=0.55,
        label="raw",
    )

    colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(windows)))
    for window, colour in zip(sorted(windows), colours):
        try:
            smoothed = savgol_smooth(signal, int(window), polyorder)
        except ValueError:
            # A window the sweep offered but this trace cannot take. Skipping it
            # is honest; drawing a substitute under its label would not be.
            continue
        axis.plot(
            hours[keep],
            smoothed[keep],
            linewidth=1.5,
            color=colour,
            label=f"window {int(window)}",
        )

    axis.set_xlabel("time (hours)")
    axis.set_ylabel("scattering (channel units)")
    axis.set_title(
        f"{state.run_id}: shortlisted smoothing windows over the raw trace"
    )
    axis.legend(loc="best", fontsize=8, framealpha=0.85)
    axis.grid(alpha=0.15)
    figure.tight_layout()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return buffer.getvalue()


def default_render_plot(state: RunState, windows: list[int]) -> bytes | None:
    """Signature the agent graph expects for its ``render_plot`` hook."""
    return render_window_overlays(state, windows, polyorder=state.polyorder or 2)


__all__ = ["DPI", "MAX_POINTS", "default_render_plot", "render_window_overlays"]
