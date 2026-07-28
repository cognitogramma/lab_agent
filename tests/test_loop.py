"""The bounded loop's guarantees: caps, fallbacks, and no invented candidates."""

from __future__ import annotations

import pytest

from polyrmc.config import LoopConfig
from polyrmc.state import Candidate, JudgeDecision, ModelProvenance
from polyrmc.tier1.judge import RecordedJudge, StaticJudge, prompt_hash, render_prompt
from polyrmc.tier1.loop import run_bounded_loop


def _candidates(values=(11, 21, 31)):
    return [
        Candidate(index=i, value=value, features={"window": float(value)}, label=f"w={value}")
        for i, value in enumerate(values)
    ]


def _fallback(candidates):
    return min(candidates, key=lambda candidate: candidate.value)


def test_judge_selection_resolves_the_loop():
    loop = run_bounded_loop(
        "test",
        propose=lambda hint: _candidates(),
        judge=StaticJudge(rule=lambda candidates: candidates[-1].index),
        fallback=_fallback,
        config=LoopConfig(max_iterations=3),
    )

    assert loop.resolved_value == 31
    assert not loop.used_fallback
    assert loop.iterations_used == 1


def test_rejected_candidates_are_recorded():
    loop = run_bounded_loop(
        "test",
        propose=lambda hint: _candidates(),
        judge=StaticJudge(rule=lambda candidates: candidates[1].index),
        fallback=_fallback,
        config=LoopConfig(max_iterations=3),
    )

    assert loop.decisions[0].rejected_indices == [0, 2]


def test_loop_falls_back_when_the_cap_is_hit():
    def always_repropose(candidates, context, iteration):
        decision = JudgeDecision(
            iteration=iteration, action="repropose", direction_hint="wider", rationale="no"
        )
        return decision, ModelProvenance(model="test", prompt_hash="x")

    loop = run_bounded_loop(
        "test",
        propose=lambda hint: _candidates(),
        judge=always_repropose,
        fallback=_fallback,
        config=LoopConfig(max_iterations=2),
    )

    assert loop.used_fallback
    assert loop.resolved_value == 11, "fallback must take the conservative option"
    assert loop.iterations_used == 2
    assert "cap of 2" in loop.fallback_reason


def test_a_judge_cannot_construct_a_candidate():
    """Selecting an index outside the offered set is refused, not honored."""

    def out_of_range(candidates, context, iteration):
        decision = JudgeDecision(
            iteration=iteration,
            action="select",
            selected_index=99,
            rationale="a window that was never offered",
        )
        return decision, ModelProvenance(model="test", prompt_hash="x")

    loop = run_bounded_loop(
        "test",
        propose=lambda hint: _candidates(),
        judge=out_of_range,
        fallback=_fallback,
        config=LoopConfig(max_iterations=3),
    )

    assert loop.used_fallback
    assert loop.resolved_value == 11
    assert "not among the offered candidates" in loop.fallback_reason


def test_empty_candidate_set_falls_back_without_calling_the_judge():
    def exploding_judge(candidates, context, iteration):
        raise AssertionError("the judge must not be called with no candidates")

    loop = run_bounded_loop(
        "test",
        propose=lambda hint: [],
        judge=exploding_judge,
        fallback=_fallback,
        config=LoopConfig(max_iterations=3),
    )

    assert loop.used_fallback
    assert "no candidate passing its safety gates" in loop.fallback_reason


def test_direction_hint_is_passed_to_the_next_proposal():
    seen: list[str | None] = []

    def repropose_once(candidates, context, iteration):
        if iteration == 0:
            decision = JudgeDecision(
                iteration=iteration,
                action="repropose",
                direction_hint="narrower",
                rationale="too wide",
            )
        else:
            decision = JudgeDecision(
                iteration=iteration, action="select", selected_index=0, rationale="ok"
            )
        return decision, ModelProvenance(model="test", prompt_hash="x")

    def propose(hint):
        seen.append(hint)
        return _candidates()

    run_bounded_loop(
        "test",
        propose=propose,
        judge=repropose_once,
        fallback=_fallback,
        config=LoopConfig(max_iterations=3),
    )

    assert seen == [None, "narrower"]


def test_recorded_judge_replays_without_a_model_call():
    recorded = [
        JudgeDecision(iteration=0, action="select", selected_index=2, rationale="recorded")
    ]
    loop = run_bounded_loop(
        "test",
        propose=lambda hint: _candidates(),
        judge=RecordedJudge(recorded),
        fallback=_fallback,
        config=LoopConfig(max_iterations=3),
    )

    assert loop.resolved_value == 31
    assert loop.provenance[0].replayed is True


def test_prompt_hash_is_stable_and_sensitive():
    prompt_a = render_prompt("fit_range", _candidates(), {"run": "a"}, 0)
    prompt_b = render_prompt("fit_range", _candidates(), {"run": "b"}, 0)

    assert prompt_hash("sys", prompt_a, "m") == prompt_hash("sys", prompt_a, "m")
    assert prompt_hash("sys", prompt_a, "m") != prompt_hash("sys", prompt_b, "m")
    assert prompt_hash("sys", prompt_a, "m1") != prompt_hash("sys", prompt_a, "m2")


def test_rendered_prompt_carries_features_not_raw_data():
    prompt = render_prompt("smoothing_window", _candidates(), {}, 0)

    assert "window=11" in prompt
    assert "[0]" in prompt and "[2]" in prompt
    assert len(prompt) < 2000, "the judge must never be handed a bulk series"


def test_recorded_judge_raises_when_the_recording_runs_out():
    with pytest.raises(LookupError):
        RecordedJudge([])( _candidates(), {}, 0)
