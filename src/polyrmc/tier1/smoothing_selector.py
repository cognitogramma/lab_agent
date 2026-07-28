"""Select the Savitzky-Golay window length for a file.

Second of the two judgment loops, and deliberately the less consequential one:
smoothing affects the precision of a fit, whereas the fit range determines what
is being fitted at all.

With the acquisition-cycle question closed -- this pipeline processes static
light scattering, which is insensitive to the convective motion that forces DLS
into a stop-flow cycle -- the trace is a continuous curve and window selection
is a straightforward sweep over the full series.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from polyrmc.config import LoopConfig, SmoothingConfig
from polyrmc.state import Candidate, LoopState
from polyrmc.tier0.smoothing import propose_windows
from polyrmc.tier1.judge import StaticJudge
from polyrmc.tier1.loop import Judge, run_bounded_loop

_NARROWER = ("narrow", "smaller", "shorter", "less smoothing", "tighter")
_WIDER = ("wide", "larger", "longer", "more smoothing", "smoother")


def conservative_window_candidate(candidates: list[Candidate]) -> Candidate:
    """The shortest offered window -- the least smoothing, hence the safest.

    Under-smoothing costs precision. Over-smoothing destroys the transitions the
    experiment exists to measure, so the conservative direction is unambiguous.
    """
    if not candidates:
        raise ValueError("no smoothing window passed the safety gates")
    return min(candidates, key=lambda candidate: float(candidate.features["window"]))


class _WindowProposer:
    """Deterministic window proposer that honors a direction hint."""

    def __init__(self, signal: np.ndarray, config: SmoothingConfig) -> None:
        self.signal = signal
        self.config = config
        self.last: list[Candidate] = []

    def __call__(self, hint: str | None) -> list[Candidate]:
        candidates = propose_windows(self.signal, self.config)
        if hint and self.last:
            lowered = hint.lower()
            pivot = float(np.median([c.features["window"] for c in self.last]))
            if any(word in lowered for word in _NARROWER):
                candidates = [c for c in candidates if c.features["window"] < pivot]
            elif any(word in lowered for word in _WIDER):
                candidates = [c for c in candidates if c.features["window"] > pivot]
            for position, candidate in enumerate(candidates):
                candidate.index = position

        self.last = candidates
        return candidates


def select_smoothing_window(
    signal: np.ndarray,
    smoothing_config: SmoothingConfig | None = None,
    loop_config: LoopConfig | None = None,
    judge: Judge | None = None,
    context: dict[str, Any] | None = None,
) -> LoopState:
    """Run the bounded loop that chooses the smoothing window.

    Returns the full :class:`LoopState`; ``resolved_value`` is the window length
    in samples.
    """
    smoothing_config = smoothing_config or SmoothingConfig()

    run_context = {
        "n_points": int(signal.size),
        "polyorder": smoothing_config.polyorder,
        "gates": (
            f"|residual lag-1 autocorr| <= {smoothing_config.max_residual_autocorr}, "
            f"landmark attenuation <= {smoothing_config.max_landmark_attenuation}"
        ),
    }
    run_context.update(context or {})

    return run_bounded_loop(
        name="smoothing_window",
        propose=_WindowProposer(signal, smoothing_config),
        judge=judge or StaticJudge(rationale="no judge supplied; conservative default"),
        fallback=conservative_window_candidate,
        config=loop_config,
        context=run_context,
    )
