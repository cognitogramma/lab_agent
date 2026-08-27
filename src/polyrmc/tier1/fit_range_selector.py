"""Select the linear region of Kc/I_R vs c.

Of the two selections this is the consequential one: it replaces the decision
that has no audit trail today. Where the linear region ends sets A2 directly, so
the candidate set, the selection, and the rejects are all recorded.

Scale-invariant throughout, so this runs before the toluene calibration is
measured.

**No model is consulted here**; see
:mod:`polyrmc.tier1.smoothing_selector` for why. The candidates are proposed by
code that has already applied the linearity gates, and the conservative one is
taken. An agent that wants a different range asks for one through the tool
surface, where the choice is validated against this same offered set.
"""

from __future__ import annotations

import numpy as np

from polyrmc.config import FitRangeConfig
from polyrmc.state import Candidate, LoopState
from polyrmc.tier0.fit_range import (
    conservative_fit_range,
    propose_fit_ranges,
    sorted_finite,
)


def select_fit_range(
    concentration: np.ndarray,
    kc_over_i: np.ndarray,
    fit_config: FitRangeConfig | None = None,
    stride: int = 1,
) -> LoopState:
    """Propose the safe ranges and take the conservative one.

    Returns the full :class:`LoopState`; ``resolved_value`` is the upper
    concentration bound in g/cm^3. Every candidate offered is recorded, because
    which points were excluded from the linear region *is* the scientific claim
    and a reviewer has to be able to see it.

    ``stride`` should reflect the smoothing correlation length introduced by
    Part 1; see :func:`polyrmc.tier0.fit_range.evaluate_range`.
    """
    fit_config = fit_config or FitRangeConfig()
    c, y = sorted_finite(concentration, kc_over_i)

    state = LoopState(name="fit_range", max_iterations=1)
    state.iterations_used = 1
    candidates: list[Candidate] = propose_fit_ranges(c, y, fit_config, stride=stride)
    state.candidates_by_iteration.append(candidates)

    if not candidates:
        state.used_fallback = True
        state.fallback_reason = (
            "no concentration range passed the linearity gates"
        )
        return state

    state.resolved_value = conservative_fit_range(candidates).value
    return state


__all__ = ["select_fit_range"]
