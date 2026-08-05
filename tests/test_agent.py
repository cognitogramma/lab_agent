"""The agent loop, exercised entirely offline.

Every test here runs with scripted providers: no network, no key, no
nondeterminism. The failure paths matter most -- an outage must cost judgment
and never correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.agent.graph import AgentGraph, Node
from polyrmc.agent.lanes import Shortlist, parse_verdict, reconcile, run_llm_lane
from polyrmc.agent.provider import ProviderError, Resilient, StubProvider, is_transient
from polyrmc.state import Candidate, JudgeVerdict, LoopState, RunState
from polyrmc.tools import ToolSurface


def _candidates(windows=(11, 15, 19)):
    return [
        Candidate(
            index=i,
            value=w,
            features={
                "residual_acf1": 0.05,
                "slope_retention": 0.95,
                "amplitude_retention": 0.98,
                "residual_sigma": 0.002,
            },
        )
        for i, w in enumerate(windows)
    ]


@pytest.fixture
def shortlist():
    return Shortlist(_candidates(), noise_floor=0.002, context={"run_id": "t"})


@pytest.fixture
def graph(run_config):
    state = RunState(run_id="agent-test", source_file="none")
    state.time_s = np.arange(0.0, 1000.0, 1.0)
    state.raw_signal = np.linspace(0.2, 0.8, 1000)
    state.spliced_signal = state.raw_signal
    state.noise_floor = 0.002
    state.loops["smoothing_window"] = LoopState(
        name="smoothing_window", max_iterations=3, candidates_by_iteration=[_candidates()]
    )
    tools = ToolSurface(state=state, config=run_config)
    return AgentGraph(tools=tools)


# --- resilience: I8, an outage costs judgment and never correctness -----------


def test_a_working_provider_is_used(shortlist):
    guard = Resilient(primary=StubProvider(['{"window_length": 15}']))
    verdict, errors = run_llm_lane(shortlist, guard)

    assert verdict.window_length == 15
    assert errors == []


def test_a_transient_failure_is_retried(shortlist):
    provider = StubProvider(
        [ProviderError("connection timeout"), '{"window_length": 19}']
    )
    guard = Resilient(primary=provider, sleep=lambda _s: None)
    verdict, _errors = run_llm_lane(shortlist, guard)

    assert verdict.window_length == 19
    assert len(provider.calls) == 2


def test_a_permanent_failure_is_not_retried(shortlist):
    """Retrying a deterministic refusal reproduces it and wastes the budget."""
    provider = StubProvider([ProviderError("invalid request: bad schema")] * 5)
    guard = Resilient(primary=provider, sleep=lambda _s: None)
    run_llm_lane(shortlist, guard)

    assert len(provider.calls) == 1


def test_failover_to_a_secondary_provider(shortlist):
    primary = StubProvider([ProviderError("503 unavailable")] * 3, name="primary")
    secondary = StubProvider(['{"window_length": 11}'], name="secondary")
    guard = Resilient(primary=primary, secondary=secondary, sleep=lambda _s: None)

    verdict, _errors = run_llm_lane(shortlist, guard)
    assert verdict.window_length == 11


def test_total_outage_falls_back_inside_the_safe_band(shortlist):
    guard = Resilient(
        primary=StubProvider([ProviderError("503 unavailable")] * 3),
        sleep=lambda _s: None,
    )
    llm, _errors = run_llm_lane(shortlist, guard)
    outcome = reconcile(llm, None, shortlist)

    assert llm is None
    assert outcome.fallback_taken
    # Smallest window: least smoothing, most faithful to the data.
    assert outcome.window_length == 11
    assert outcome.window_length in shortlist.windows


def test_no_provider_configured_is_not_an_error(shortlist):
    llm, _errors = run_llm_lane(shortlist, Resilient())
    outcome = reconcile(llm, None, shortlist)

    assert outcome.status == "fallback"
    assert "correctness" in outcome.message


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Connection reset", True),
        ("429 rate limit exceeded", True),
        ("503 Service Unavailable", True),
        ("invalid api key", False),
        ("content policy refusal", False),
    ],
)
def test_transient_classification(message, expected):
    assert is_transient(message) is expected


# --- the band is a hard boundary ---------------------------------------------


def test_a_choice_outside_the_band_is_refused(shortlist):
    guard = Resilient(primary=StubProvider(['{"window_length": 51}']))
    verdict, errors = run_llm_lane(shortlist, guard)

    assert verdict is None
    assert any("not in the safe band" in e for e in errors)


def test_free_text_answers_are_accepted_when_they_name_a_band_member():
    """A model that reasons well but formats badly should not cost the judgment."""
    verdict = parse_verdict(
        "I would take 15 here: residual_acf1 is already near zero at that width.",
        [11, 15, 19],
        "llm",
    )
    assert verdict.window_length == 15
    assert "residual_acf1" in verdict.cited_diagnostics


def test_an_answer_naming_no_band_member_is_refused():
    with pytest.raises(ValueError, match="named no window"):
        parse_verdict("hard to say, they all look fine", [11, 15, 19], "llm")


# --- reconciliation: section 6.7 stage 3 --------------------------------------


def _verdict(lane, window):
    return JudgeVerdict(lane=lane, window_length=window, rationale=f"{lane} says {window}")


def test_agreeing_lanes_proceed(shortlist):
    outcome = reconcile(_verdict("llm", 15), _verdict("vlm", 15), shortlist)

    assert outcome.status == "both_lanes"
    assert outcome.window_length == 15
    assert not outcome.needs_researcher
    # Both rationales are logged, not just the winner's.
    assert "numbers" in outcome.message and "picture" in outcome.message


def test_disagreeing_lanes_escalate_and_choose_nothing(shortlist):
    """Never averaged: a compromise window is one nobody argued for."""
    outcome = reconcile(_verdict("llm", 11), _verdict("vlm", 19), shortlist)

    assert outcome.status == "escalated"
    assert outcome.window_length is None
    assert outcome.disagreed
    assert outcome.needs_researcher
    assert "11" in outcome.message and "19" in outcome.message


def test_one_lane_down_proceeds_on_the_survivor(shortlist):
    outcome = reconcile(_verdict("llm", 15), None, shortlist)

    assert outcome.status == "llm_only"
    assert outcome.window_length == 15
    assert "vlm lane was unavailable" in outcome.message


def test_the_vlm_lane_is_skipped_without_a_plot(shortlist):
    from polyrmc.agent.lanes import run_vlm_lane

    verdict, errors = run_vlm_lane(shortlist, Resilient(), image=None)
    assert verdict is None
    assert any("no rendered plot" in e for e in errors)


# --- the graph ----------------------------------------------------------------


def test_intake_pauses_before_anything_is_processed(graph):
    steps = graph.run()

    assert steps[0].paused
    assert steps[0].interrupt.node is Node.INTAKE
    assert "declared, not inferred" in steps[0].interrupt.question
    # Nothing downstream ran.
    assert graph.node is Node.INTAKE


def test_a_paused_node_stays_current_rather_than_being_skipped(graph):
    graph.run()
    assert graph.pending is not None
    assert graph.node is Node.INTAKE
    graph.resume()
    assert graph.node is Node.INTAKE


def test_resume_without_a_pending_interrupt_is_refused(graph):
    with pytest.raises(RuntimeError, match="nothing is pending"):
        graph.resume()


def test_the_window_judgment_escalates_on_disagreement(graph):
    graph.node = Node.JUDGE_WINDOW
    graph.guard = Resilient(primary=StubProvider(['{"window_length": 11}']))
    graph.vlm_guard = Resilient(primary=StubProvider(['{"window_length": 19}']))
    graph.render_plot = lambda _state, _windows: b"fake-png"

    step = graph.advance()

    assert step.paused
    assert graph.state.lanes_disagreed
    assert graph.state.judge_status == "escalated"
    assert graph.state.window_length is None, "nothing committed while escalated"


def test_the_window_is_committed_when_the_lanes_agree(graph):
    graph.node = Node.JUDGE_WINDOW
    graph.guard = Resilient(primary=StubProvider(['{"window_length": 15}']))
    graph.vlm_guard = Resilient(primary=StubProvider(['{"window_length": 15}']))
    graph.render_plot = lambda _state, _windows: b"fake-png"

    step = graph.advance()

    assert not step.paused
    assert graph.state.window_length == 15
    assert graph.state.judge_status == "both_lanes"


def test_a_total_outage_still_completes_the_judgment(graph):
    """I8: the run finishes with a valid result and says a fallback was taken."""
    graph.node = Node.JUDGE_WINDOW
    step = graph.advance()

    assert not step.paused
    assert graph.state.fallback_taken
    assert graph.state.judge_status == "fallback"
    assert graph.state.window_length == 11


def test_baseline_review_asks_and_offers_no_correction(graph):
    from polyrmc.state import AnchorCandidate

    graph.state.anchor_candidates = [
        AnchorCandidate(
            id=i,
            index_range=(i * 100, i * 100 + 50),
            duration_fraction=0.05,
            mean_level=0.2,
            local_slope=0.0,
            local_variance=1e-6,
            position_fraction=0.1 * i,
        )
        for i in range(2)
    ]
    graph.node = Node.REVIEW_ANCHORS
    step = graph.advance()

    assert step.paused
    assert "no correction" in step.interrupt.options
    assert "yours" in step.interrupt.because


def test_few_candidates_means_no_correction_is_offerable(graph):
    graph.node = Node.REVIEW_ANCHORS
    step = graph.advance()

    assert not step.paused
    assert "No correction is offerable" in step.narration


# --- narration ----------------------------------------------------------------


def test_narration_is_generated_from_what_happened(graph):
    graph.run()
    text = graph.narrate()

    for entry in graph.state.execution_history:
        assert entry in text


def test_narration_surfaces_a_fallback(graph):
    graph.node = Node.JUDGE_WINDOW
    graph.advance()

    assert "conservative option" in graph.narrate()


def test_narration_surfaces_a_disagreement(graph):
    graph.node = Node.JUDGE_WINDOW
    graph.guard = Resilient(primary=StubProvider(['{"window_length": 11}']))
    graph.vlm_guard = Resilient(primary=StubProvider(['{"window_length": 19}']))
    graph.render_plot = lambda _state, _windows: b"fake-png"
    graph.advance()

    assert "disagreed" in graph.narrate()


def test_the_history_is_append_only_across_steps(graph):
    graph.run()
    first = list(graph.state.execution_history)
    graph.resume()

    assert graph.state.execution_history[: len(first)] == first


# --- rubrics ------------------------------------------------------------------


def test_the_rubrics_load_and_forbid_measuring():
    from polyrmc.agent.lanes import load_prompt

    assert "never measure" in load_prompt("vlm_lane.md").lower()
    assert "you do not compute" in load_prompt("system.md").lower()
    assert "shortest window that whitens" in load_prompt("llm_lane.md").lower()


def test_the_shortlist_table_carries_the_diagnostics(shortlist):
    table = shortlist.render_table()
    assert "residual_acf1" in table
    assert "slope_retention" in table
    for window in shortlist.windows:
        assert str(window) in table
