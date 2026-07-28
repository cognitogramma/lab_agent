"""The bounded propose-judge-re-propose harness.

Every subjective decision in the pipeline runs through this function, so the
guarantees hold in one place rather than being re-implemented per decision:

* the candidate set is always produced by deterministic code;
* a judge may only select an index that exists, or ask to re-propose;
* iterations are capped;
* on exhaustion, or on any invalid judge response, the conservative fallback is
  taken and the reason is recorded;
* the full trajectory -- candidates offered, rejected, and accepted -- is kept.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from polyrmc.config import LoopConfig
from polyrmc.state import Candidate, JudgeDecision, LoopState, ModelProvenance


class Judge(Protocol):
    """A judge selects among candidates. It never builds one."""

    def __call__(
        self,
        candidates: list[Candidate],
        context: dict[str, Any],
        iteration: int,
    ) -> tuple[JudgeDecision, ModelProvenance]: ...


Proposer = Callable[[str | None], list[Candidate]]
"""Takes an optional direction hint, returns a safe-by-construction candidate set."""

Fallback = Callable[[list[Candidate]], Candidate]
"""Picks the most conservative option from a candidate set."""


def run_bounded_loop(
    name: str,
    propose: Proposer,
    judge: Judge,
    fallback: Fallback,
    config: LoopConfig | None = None,
    context: dict[str, Any] | None = None,
) -> LoopState:
    """Run one capped evaluator loop and return its full trajectory.

    Parameters
    ----------
    propose:
        Deterministic candidate generator. Called with ``None`` on the first
        iteration and with the judge's direction hint thereafter.
    judge:
        Selects among the offered candidates.
    fallback:
        Conservative choice, used when the cap is hit or a response is invalid.
    """
    config = config or LoopConfig()
    context = context or {}
    state = LoopState(name=name, max_iterations=config.max_iterations)

    hint: str | None = None
    last_candidates: list[Candidate] = []

    for iteration in range(config.max_iterations):
        candidates = propose(hint)
        state.candidates_by_iteration.append(candidates)
        state.iterations_used = iteration + 1

        if not candidates:
            state.used_fallback = True
            state.fallback_reason = (
                "the proposer returned no candidate passing its safety gates"
                if iteration == 0
                else f"re-proposal with hint {hint!r} produced no candidates"
            )
            if last_candidates:
                chosen = fallback(last_candidates)
                state.resolved_value = chosen.value
            return state

        last_candidates = candidates
        decision, provenance = judge(candidates, context, iteration)
        state.decisions.append(decision)
        state.provenance.append(provenance)

        if decision.action == "select":
            valid = {candidate.index for candidate in candidates}
            if decision.selected_index not in valid:
                # A judge that names a candidate outside the offered set has
                # tried to construct one. Refuse it and take the safe option.
                state.used_fallback = True
                state.fallback_reason = (
                    f"judge selected index {decision.selected_index}, which was not "
                    f"among the offered candidates {sorted(valid)}"
                )
                state.resolved_value = fallback(candidates).value
                return state

            selected = next(c for c in candidates if c.index == decision.selected_index)
            state.resolved_value = selected.value
            return state

        hint = decision.direction_hint

    state.used_fallback = True
    state.fallback_reason = (
        f"loop hit its cap of {config.max_iterations} iteration(s) without a selection"
    )
    state.resolved_value = fallback(last_candidates).value
    return state
