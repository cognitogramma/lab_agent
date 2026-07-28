"""Judges -- the only place in the pipeline where a model client lives.

A judge receives the *computed features* of each candidate and nothing else. It
never sees the 80,000-point series, and it is never asked to do arithmetic over
one. That restriction is what makes "models judge, they never compute"
enforceable rather than aspirational.

Three implementations:

* :class:`ModelJudge` -- calls the pinned Claude model.
* :class:`RecordedJudge` -- replays decisions from a provenance record.
* :class:`StaticJudge` -- a deterministic rule, used by the offline test suite
  and as an escape hatch when no API key is configured.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from polyrmc.config import LoopConfig
from polyrmc.state import Candidate, JudgeDecision, ModelProvenance

SYSTEM_PROMPT = """\
You are an analysis judge for a light-scattering measurement pipeline.

You are shown a set of candidate settings that deterministic code has already
generated and verified. Every candidate is safe: each one passed the numerical
gates for its decision. Your job is to choose the best one on scientific
grounds, or to ask for a re-proposal in a stated direction.

Hard constraints:
- You may only select a candidate by its listed index. You cannot invent,
  modify, interpolate, or combine candidates.
- You must not perform arithmetic on the underlying data. The numbers you are
  shown were computed for you; reason about them, do not recompute them.
- If no candidate is defensible, choose "repropose" and say which direction to
  move in. You have a limited number of re-proposals before the pipeline falls
  back to the most conservative option on its own.

State your reasoning in one or two sentences. Be specific about which feature
drove the choice.
"""


class JudgeResponse(BaseModel):
    """Structured response schema the model is constrained to."""

    action: Literal["select", "repropose"] = Field(
        ..., description="Select a candidate, or request a new candidate set."
    )
    selected_index: int | None = Field(
        None, description="Index of the chosen candidate; required when selecting."
    )
    direction_hint: str | None = Field(
        None,
        description="When reproposing, which direction to move (e.g. 'narrower range').",
    )
    rationale: str = Field(..., description="One or two sentences of justification.")


def render_prompt(
    decision_name: str,
    candidates: list[Candidate],
    context: dict[str, Any],
    iteration: int,
) -> str:
    """Render the judge prompt from candidate features only."""
    lines = [
        f"Decision: {decision_name}",
        f"Iteration: {iteration}",
    ]
    if context:
        lines.append("")
        lines.append("Run context:")
        for key, value in sorted(context.items()):
            lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append(f"Candidates ({len(candidates)}), all of which passed the safety gates:")
    for candidate in candidates:
        features = ", ".join(
            f"{key}={value:.6g}" for key, value in sorted(candidate.features.items())
        )
        lines.append(f"  [{candidate.index}] {candidate.label or candidate.value}")
        lines.append(f"        {features}")

    lines.append("")
    lines.append("Select the best candidate by index, or request a re-proposal.")
    return "\n".join(lines)


def prompt_hash(system: str, prompt: str, model: str) -> str:
    """SHA-256 over the exact rendered prompt and pinned model id."""
    payload = json.dumps(
        {"system": system, "prompt": prompt, "model": model}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_decision(
    response: JudgeResponse, candidates: list[Candidate], iteration: int
) -> JudgeDecision:
    """Convert a raw response into a decision, recording what was rejected."""
    rejected = [
        candidate.index
        for candidate in candidates
        if candidate.index != response.selected_index
    ]
    return JudgeDecision(
        iteration=iteration,
        action=response.action,
        selected_index=response.selected_index,
        direction_hint=response.direction_hint,
        rationale=response.rationale,
        rejected_indices=rejected if response.action == "select" else [],
    )


class ModelJudge:
    """Judge backed by the pinned Claude model."""

    def __init__(
        self,
        decision_name: str,
        config: LoopConfig | None = None,
        client: Any | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.decision_name = decision_name
        self.config = config or LoopConfig()
        self.system_prompt = system_prompt
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            # Imported lazily so tier-0 tests and offline runs never need the
            # provider package installed or an API key present.
            from langchain_anthropic import ChatAnthropic

            self._client = ChatAnthropic(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
            ).with_structured_output(JudgeResponse)
        return self._client

    def __call__(
        self, candidates: list[Candidate], context: dict[str, Any], iteration: int
    ) -> tuple[JudgeDecision, ModelProvenance]:
        prompt = render_prompt(self.decision_name, candidates, context, iteration)
        digest = prompt_hash(self.system_prompt, prompt, self.config.model)

        response = self._get_client().invoke(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        if not isinstance(response, JudgeResponse):
            response = JudgeResponse.model_validate(response)

        provenance = ModelProvenance(model=self.config.model, prompt_hash=digest)
        return _to_decision(response, candidates, iteration), provenance


class RecordedJudge:
    """Replay a loop from a provenance record instead of calling the model.

    Used to reproduce a historical output exactly. If the recording runs out of
    decisions the loop is allowed to exhaust normally and fall back, rather than
    silently switching to a live call.
    """

    def __init__(self, decisions: list[JudgeDecision], model: str = "replay") -> None:
        self.decisions = list(decisions)
        self.model = model

    def __call__(
        self, candidates: list[Candidate], context: dict[str, Any], iteration: int
    ) -> tuple[JudgeDecision, ModelProvenance]:
        for decision in self.decisions:
            if decision.iteration == iteration:
                return decision, ModelProvenance(
                    model=self.model, prompt_hash="replayed", replayed=True
                )
        raise LookupError(f"no recorded decision for iteration {iteration}")


class StaticJudge:
    """Deterministic judge for offline use.

    Defaults to the conservative choice -- the first candidate -- which for both
    of the pipeline's loops is the least-aggressive option.
    """

    def __init__(
        self,
        rule: Callable[[list[Candidate]], int] | None = None,
        rationale: str = "deterministic rule, no model consulted",
    ) -> None:
        self.rule = rule or (lambda candidates: candidates[0].index)
        self.rationale = rationale

    def __call__(
        self, candidates: list[Candidate], context: dict[str, Any], iteration: int
    ) -> tuple[JudgeDecision, ModelProvenance]:
        response = JudgeResponse(
            action="select",
            selected_index=self.rule(candidates),
            rationale=self.rationale,
        )
        provenance = ModelProvenance(
            model="static", prompt_hash="not-a-model-call", replayed=False
        )
        return _to_decision(response, candidates, iteration), provenance
