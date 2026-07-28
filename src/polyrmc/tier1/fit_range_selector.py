"""Select the linear region of Kc/I_R vs c.

Of the two judgment loops this is the consequential one: it replaces the
decision that has no audit trail today. Where the linear region ends sets A2
directly, so the candidate set, the selection, and the rejects are all recorded.

Scale-invariant throughout, so this runs before the toluene calibration is
measured.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from polyrmc.config import FitRangeConfig, LoopConfig
from polyrmc.state import Candidate, LoopState
from polyrmc.tier0.fit_range import (
    conservative_fit_range,
    propose_fit_ranges,
    sorted_finite,
)
from polyrmc.tier1.judge import StaticJudge
from polyrmc.tier1.loop import Judge, run_bounded_loop

_NARROWER = ("narrow", "tight", "lower", "smaller", "restrict", "fewer", "exclude more")
_WIDER = ("wide", "broad", "higher", "larger", "extend", "more points", "include more")


class _FitRangeProposer:
    """Deterministic proposer that responds to a judge's direction hint.

    A hint only reweights which safe candidates are offered next. It can never
    introduce a range that failed the linearity gates.
    """

    def __init__(
        self,
        concentration: np.ndarray,
        kc_over_i: np.ndarray,
        config: FitRangeConfig,
        stride: int = 1,
    ) -> None:
        self.concentration = concentration
        self.kc_over_i = kc_over_i
        self.config = config
        self.stride = stride
        self.last: list[Candidate] = []

    def __call__(self, hint: str | None) -> list[Candidate]:
        candidates = propose_fit_ranges(
            self.concentration, self.kc_over_i, self.config, stride=self.stride
        )
        if hint and self.last:
            lowered = hint.lower()
            previous = [float(c.features["c_max"]) for c in self.last]
            pivot = float(np.median(previous))
            if any(word in lowered for word in _NARROWER):
                candidates = [c for c in candidates if c.features["c_max"] < pivot]
            elif any(word in lowered for word in _WIDER):
                candidates = [c for c in candidates if c.features["c_max"] > pivot]
            # Re-index so the judge always sees a contiguous 0..n-1 set.
            for position, candidate in enumerate(candidates):
                candidate.index = position

        self.last = candidates
        return candidates


def select_fit_range(
    concentration: np.ndarray,
    kc_over_i: np.ndarray,
    fit_config: FitRangeConfig | None = None,
    loop_config: LoopConfig | None = None,
    judge: Judge | None = None,
    context: dict[str, Any] | None = None,
    stride: int = 1,
) -> LoopState:
    """Run the bounded loop that chooses where the linear region ends.

    Returns the full :class:`LoopState`; ``resolved_value`` is the upper
    concentration bound in g/cm^3.

    ``stride`` should reflect the smoothing correlation length introduced by
    Part 1; see :func:`polyrmc.tier0.fit_range.evaluate_range`.
    """
    fit_config = fit_config or FitRangeConfig()
    c, y = sorted_finite(concentration, kc_over_i)

    run_context = {
        "n_points_total": int(c.size),
        "c_range": f"{c.min():.4g} to {c.max():.4g} g/cm^3",
        "quantity": "Kc/I vs c; slope/intercept = 2*A2*Mw (scale-invariant)",
        "residual_stride": stride,
    }
    run_context.update(context or {})

    return run_bounded_loop(
        name="fit_range",
        propose=_FitRangeProposer(c, y, fit_config, stride=stride),
        judge=judge or StaticJudge(rationale="no judge supplied; conservative default"),
        fallback=conservative_fit_range,
        config=loop_config,
        context=run_context,
    )
