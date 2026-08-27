"""Select the Savitzky-Golay window length for a file.

Second of the two selections, and deliberately the less consequential one:
smoothing affects the precision of a fit, whereas the fit range determines what
is being fitted at all.

With the acquisition-cycle question closed -- this pipeline processes static
light scattering, which is insensitive to the convective motion that forces DLS
into a stop-flow cycle -- the trace is a continuous curve and window selection
is a straightforward sweep over the full series.

**No model is consulted here.** This used to run through a bounded
propose-judge-re-propose harness so that a judge could pick among the certified
windows. The agent picks now, and it picks through the tool surface --
``sweep_windows`` offers the band, ``pick_window`` validates the choice against
it -- which is a better boundary than the harness was: the model sees the
candidates as data, and an invalid choice comes back as a refusal it can read.
So what is left here is the deterministic half, which is the half that carried
the guarantee. It proposes, and it takes the conservative option.

A :class:`LoopState` is still returned, and still records the candidates that
were offered. That was never about the loop -- it is the audit trail. Which
windows passed the gates is a claim about the data, and it stays in the
provenance sidecar whether or not anything iterated to get there.
"""

from __future__ import annotations

import numpy as np

from polyrmc.config import SmoothingConfig
from polyrmc.state import Candidate, LoopState
from polyrmc.tier0.smoothing import propose_windows


def conservative_window_candidate(candidates: list[Candidate]) -> Candidate:
    """The shortest offered window -- the least smoothing, hence the safest.

    Under-smoothing costs precision. Over-smoothing destroys the transitions the
    experiment exists to measure, so the conservative direction is unambiguous.
    """
    if not candidates:
        raise ValueError("no smoothing window passed the safety gates")
    return min(candidates, key=lambda candidate: float(candidate.features["window"]))


def select_smoothing_window(
    signal: np.ndarray,
    smoothing_config: SmoothingConfig | None = None,
    exclude: np.ndarray | None = None,
) -> LoopState:
    """Propose the safe band and take the conservative window from it.

    Returns the full :class:`LoopState`; ``resolved_value`` is the window length
    in samples.
    """
    smoothing_config = smoothing_config or SmoothingConfig()

    state = LoopState(name="smoothing_window", max_iterations=1)
    state.iterations_used = 1
    candidates = propose_windows(signal, smoothing_config, exclude=exclude)
    state.candidates_by_iteration.append(candidates)

    if not candidates:
        # No window cleared the gates. Nothing is chosen and the reason is
        # recorded: a run that smoothed anyway would be reporting a number the
        # gates had already refused.
        state.used_fallback = True
        state.fallback_reason = (
            "the proposer returned no candidate passing its safety gates"
        )
        return state

    state.resolved_value = conservative_window_candidate(candidates).value
    return state


__all__ = ["conservative_window_candidate", "select_smoothing_window"]
