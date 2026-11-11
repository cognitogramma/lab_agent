"""The agent loop, driven offline.

Every test here runs against :class:`ScriptedProvider`: no network, no API key,
no nondeterminism. That is not a compromise. The failures worth testing in an
agentic loop -- a tool that refuses, a commit a researcher declines, a model
that keeps calling, a handler that raises -- are all reachable without a live
model, and each of them is a path the researcher's data travels down.

The end-to-end test at the bottom is the one that matters most. The old graph
tests exercised control flow against hand-built state, which is how a
disconnected graph once passed a full suite: every node tested, none of them
wired to the measurement code. This drives the loop against a real file and
asserts a CSV on disk with the right numbers in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from continuum_spectranalytics.agent.orchestrator import (
    Approval,
    ApprovalPolicy,
    Orchestrator,
    allow_all,
    default_system_prompt,
    deny_all,
)
from continuum_spectranalytics.agent.toolcall import AssistantTurn, Message, ScriptedProvider, ToolCall
from continuum_spectranalytics.agent.toolspec import Tool, ToolContext, build_registry
from continuum_spectranalytics.io_csv import read_processed_csv
from continuum_spectranalytics.memory.notes import NoteBook
from continuum_spectranalytics.state import RunState
from continuum_spectranalytics.tools import ToolSurface

ANSWERS = {
    "technique": "SLS",
    "instrument": "argen",
    "sample_class": "protein_peptide",
    "experiment_type": "acd_dilution",
    "c0_g_per_cm3": 0.004,
    "ls_units": "count_rate",
    "saturation_ceiling": 65535,
    "buffer_blank": 0.0,
    "toluene_ref": 5000.0,
    "r_toluene_cm_inv": 1.19e-5,
    "cell_volume_cm3": 1.0,
    "flow_rate_cm3_per_s": 0.001,
}


def call(name: str, **arguments) -> AssistantTurn:
    """One turn that asks for exactly one tool."""
    return AssistantTurn(tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=arguments)])


@pytest.fixture
def context(run_config, tmp_path) -> ToolContext:
    surface = ToolSurface(
        state=RunState(run_id="t", source_file=str(run_config.source_file)),
        config=run_config,
    )
    return ToolContext(
        surface=surface,
        notes=NoteBook(tmp_path / "memory"),
        runs_dir=tmp_path / "runs",
        author="tester",
    )


def agent_over(context: ToolContext, turns, **kwargs) -> Orchestrator:
    return Orchestrator(
        context=context,
        provider=ScriptedProvider(turns),
        approvals=kwargs.pop("approvals", ApprovalPolicy(ask=allow_all)),
        **kwargs,
    )


# --- the loop ----------------------------------------------------------------


def test_the_model_calls_a_tool_and_answers_from_its_result(context):
    agent = agent_over(
        context,
        [call("get_state"), AssistantTurn(text="The run is not loaded yet.")],
    )

    events = agent.ask("where are we?")

    # Every turn ends with a spend line, so assert on the sequence that precedes
    # it rather than on absolute positions.
    kinds = [event.kind for event in events]
    assert kinds == ["call", "result", "text", "spend"]
    assert "loaded: no" in events[1].text
    assert events[2].text == "The run is not loaded yet."


def test_research_ideas_save_and_retrieve_without_loading_an_experiment(context):
    assert context.surface.state.raw_signal is None
    agent = agent_over(context, [
        call("save_research_idea", title="Salt and aggregation", idea="Test salt-dependent aggregation.",
             evidence="A proposed comparison, not a measurement.", next_step="Read the related papers."),
        AssistantTurn(text="Saved the hypothesis and next reading task."),
    ])
    agent.ask("Remember this research idea")
    assert agent.trace[0][1].ok
    assert len(context.notes) == 1

    context.notes = NoteBook(context.notes.directory)
    later = agent_over(context, [call("read_research_ideas", query="salt"), AssistantTurn(text="Found your idea.")])
    events = later.ask("What was my salt idea?")
    result = next(e for e in events if e.kind == "result")
    assert "Salt and aggregation" in result.text
    assert "unverified hypothesis" in result.text
    assert "Read the related papers" in result.text
    assert context.surface.state.raw_signal is None


def test_the_tool_result_is_fed_back_to_the_model(context):
    provider = ScriptedProvider([call("get_state"), AssistantTurn(text="done")])
    agent = Orchestrator(context=context, provider=provider, approvals=ApprovalPolicy(ask=allow_all))

    agent.ask("status?")

    # The second call to the provider must carry the first call's result, or the
    # model is choosing its next move blind.
    _system, messages, _tools = provider.calls[1]
    tool_messages = [m for m in messages if m.role == "tool"]
    assert tool_messages, "the loop never handed the result back"
    assert "run_id: t" in tool_messages[0].results[0].content


def test_several_tools_in_one_turn_all_run(context):
    turn = AssistantTurn(
        tool_calls=[
            ToolCall("a", "get_state", {}),
            ToolCall("b", "get_history", {}),
        ]
    )
    agent = agent_over(context, [turn, AssistantTurn(text="both read")])

    events = agent.ask("read two things")

    assert [e.tool for e in events if e.kind == "call"] == ["get_state", "get_history"]
    assert len(agent.trace) == 2


def test_the_loop_is_capped_and_says_so(context):
    agent = agent_over(context, [call("get_state") for _ in range(10)], max_steps=3)

    events = agent.ask("loop forever")

    capped = next(e for e in events if e.kind == "capped")
    assert "3 tool rounds" in capped.text
    assert len(agent.trace) == 3


# --- refusals and faults -----------------------------------------------------


def test_a_rejection_reaches_the_model_intact(context):
    """The commit tier writes rejections as recovery instructions. They must arrive."""
    provider = ScriptedProvider([call("pick_window", window_length=9), AssistantTurn(text="ok")])
    agent = Orchestrator(context=context, provider=provider, approvals=ApprovalPolicy(ask=allow_all))

    events = agent.ask("pick a window")

    refusal = next(e for e in events if e.kind == "refused")
    assert "No safe band" in refusal.text
    _system, messages, _tools = provider.calls[1]
    result = [m for m in messages if m.role == "tool"][0].results[0]
    assert result.ok is False
    assert "REJECTED" in result.content


def test_an_unknown_tool_is_reported_rather_than_raised(context):
    agent = agent_over(context, [call("summon_a_number"), AssistantTurn(text="sorry")])

    events = agent.ask("do something impossible")

    error = next(e for e in events if e.kind == "error")
    assert "no tool named 'summon_a_number'" in error.text
    assert "get_state" in error.text, "the model needs to be told what does exist"


def test_a_tool_that_raises_does_not_end_the_session(context):
    def explode(ctx, args):
        raise RuntimeError("the detector caught fire")

    registry = build_registry(
        [Tool("boom", "read", "Always fails.", {"type": "object", "properties": {}}, explode)]
    )
    agent = Orchestrator(
        context=context,
        provider=ScriptedProvider([call("boom"), AssistantTurn(text="recovered")]),
        approvals=ApprovalPolicy(ask=allow_all),
        registry=registry,
    )

    events = agent.ask("break it")

    assert any(e.kind == "error" and "caught fire" in e.text for e in events)
    assert [e.text for e in events if e.kind == "text"] == ["recovered"]


# --- approval ----------------------------------------------------------------


def test_reads_run_without_asking_and_commits_do_not(context):
    asked: list[str] = []

    def ask(tool, call_):
        asked.append(tool.name)
        return Approval(True)

    agent = agent_over(
        context,
        [call("get_state"), call("set_config", key="channel", value="ls8"), AssistantTurn(text="done")],
        approvals=ApprovalPolicy(ask=ask),
    )

    agent.ask("read then write")

    assert asked == ["set_config"], "a read must not cost the researcher a prompt"


def test_a_declined_commit_does_not_run(context):
    agent = agent_over(
        context,
        [call("set_config", key="channel", value="ls3"), AssistantTurn(text="understood")],
        approvals=ApprovalPolicy(ask=lambda t, c: Approval(False, "ls8 only, house convention")),
    )

    events = agent.ask("switch channel")

    assert context.surface.config.channel == "ls8", "the declined call changed the run"
    denied = next(e for e in events if e.kind == "denied")
    assert "house convention" in denied.text
    assert agent.committed == []


def test_the_model_is_told_not_to_route_around_a_refusal(context):
    provider = ScriptedProvider(
        [call("set_config", key="channel", value="ls3"), AssistantTurn(text="ok")]
    )
    agent = Orchestrator(
        context=context,
        provider=provider,
        approvals=ApprovalPolicy(ask=lambda t, c: Approval(False, "no")),
    )

    agent.ask("switch channel")

    _system, messages, _tools = provider.calls[1]
    content = [m for m in messages if m.role == "tool"][0].results[0].content
    assert "Do not attempt the same change another way" in content


def test_always_allow_is_remembered_for_the_session(context):
    asked: list[str] = []

    def ask(tool, call_):
        asked.append(tool.name)
        return Approval(True, "fine", remember=True)

    policy = ApprovalPolicy(ask=ask)
    agent = agent_over(
        context,
        [
            call("set_config", key="channel", value="ls8"),
            call("set_config", key="channel", value="ls8"),
            AssistantTurn(text="done"),
        ],
        approvals=policy,
    )

    agent.ask("set it twice")

    assert asked == ["set_config"], "the second call asked again despite 'always'"
    assert "set_config" in policy.always


def test_nobody_at_the_terminal_means_no_commits(context):
    agent = agent_over(
        context,
        [call("set_config", key="channel", value="ls3"), AssistantTurn(text="ok")],
        approvals=ApprovalPolicy(ask=deny_all),
    )

    events = agent.ask("commit unattended")

    assert context.surface.config.channel == "ls8"
    assert any(e.kind == "denied" for e in events)


# --- memory ------------------------------------------------------------------


def test_the_agent_can_search_and_file_what_it_learned(context, tmp_path):
    agent = agent_over(
        context,
        [
            call("write_note", subject="liraglutide", body="Unstable above 3 M Gd.", tags=["gotcha"]),
            call("read_notes", query="Gd"),
            AssistantTurn(text="noted"),
        ],
    )

    events = agent.ask("remember that")

    results = [e for e in events if e.kind == "result"]
    assert "note filed under 'liraglutide'" in results[0].text
    assert "Unstable above 3 M Gd" in results[1].text
    # Written through immediately, so a session that crashes keeps its notes.
    assert NoteBook(tmp_path / "memory").search(query="Gd")


def test_past_runs_are_searchable_as_a_tool(context, tmp_path):
    import json

    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "bsa.provenance.json").write_text(
        json.dumps(
            {
                "run_id": "bsa-01",
                "written_at": "2026-08-01T10:00:00Z",
                "config": {"experiment_type": "acd_dilution"},
                "declarations": {"sample_name": {"value": "BSA"}},
                "parameters": {"a2_mw": 33.2, "k_d": -0.01},
            }
        ),
        encoding="utf-8",
    )

    agent = agent_over(
        context, [call("search_past_runs", query="bsa"), AssistantTurn(text="found it")]
    )
    events = agent.ask("have we run BSA before?")

    result = next(e for e in events if e.kind == "result")
    assert "bsa-01" in result.text and "BSA" in result.text


# --- what it spends ----------------------------------------------------------


def test_the_session_totals_what_it_spent(context):
    from continuum_spectranalytics.agent.budget import Budget, Usage

    turns = [
        AssistantTurn(
            tool_calls=[ToolCall("a", "get_state", {})],
            usage=Usage(requests=1, input_tokens=900, output_tokens=40),
        ),
        AssistantTurn(text="done", usage=Usage(requests=1, input_tokens=1400, output_tokens=90)),
    ]
    agent = Orchestrator(
        context=context,
        provider=ScriptedProvider(turns),
        approvals=ApprovalPolicy(ask=allow_all),
        budget=Budget(),
    )

    events = agent.ask("status?")

    assert agent.budget.spent.requests == 2
    assert agent.budget.spent.input_tokens == 2300
    assert agent.budget.spent.output_tokens == 130
    spend = next(e for e in events if e.kind == "spend")
    assert "2 request(s)" in spend.text


def test_the_request_budget_stops_the_loop_before_it_spends(context):
    """The Gemini free tier rations requests, not tokens. A cap that only
    triggered on tokens would never fire before the day's quota was gone."""
    from continuum_spectranalytics.agent.budget import Budget

    provider = ScriptedProvider([call("get_state") for _ in range(6)])
    agent = Orchestrator(
        context=context,
        provider=provider,
        approvals=ApprovalPolicy(ask=allow_all),
        budget=Budget(max_requests=2),
        max_steps=10,
    )

    events = agent.ask("keep going")

    assert len(provider.calls) == 2, "the loop spent past its own ceiling"
    capped = next(e for e in events if e.kind == "capped")
    assert "request budget" in capped.text


def test_a_provider_without_usage_metadata_still_counts_requests(context):
    """Losing the token figures must not lose the count the free tier rations."""
    from langchain_core.messages import AIMessage

    from continuum_spectranalytics.agent.toolcall import _usage_of

    usage = _usage_of(AIMessage(content="no usage metadata here"))

    assert usage.requests == 1
    assert usage.total_tokens == 0


def test_usage_is_read_off_a_real_reply():
    from langchain_core.messages import AIMessage

    from continuum_spectranalytics.agent.toolcall import _usage_of

    reply = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 812, "output_tokens": 44, "total_tokens": 856},
    )

    usage = _usage_of(reply)

    assert (usage.input_tokens, usage.output_tokens, usage.requests) == (812, 44, 1)


# --- the prompt --------------------------------------------------------------


def test_the_system_prompt_carries_the_rules_and_the_lab(context):
    prompt = default_system_prompt(__file__)

    assert "You do not compute" in prompt
    assert "You drive" in prompt, "the orchestrator rules are missing"
    assert "House conventions" in prompt, "LAB.md is missing"
    assert "no correction is a valid answer" in prompt.lower()


def test_every_tool_is_offered_to_the_model(context):
    agent = agent_over(context, [AssistantTurn(text="hi")])

    names = {schema["function"]["name"] for schema in agent.schemas}

    assert "load_run" in names and "emit_csv" in names and "search_past_runs" in names
    assert len(names) == len(build_registry())


def test_every_schema_binds_against_the_real_provider_converter():
    """Providers disagree about the corners of JSON Schema, and binding is
    all-or-nothing: one bad field takes all twenty-nine tools down at the first
    live call rather than at the call that used it.

    Gemini's converter is the strict one -- it rejects an integer ``enum``, for
    instance -- so the whole set is run through it here, with no key and no
    network, and the schemas stay plain enough for both providers.
    """
    genai = pytest.importorskip("langchain_google_genai._function_utils")
    from continuum_spectranalytics.agent.toolcall import as_openai_tools
    from continuum_spectranalytics.agent.toolspec import schemas_for

    registry = build_registry()
    converted = genai.convert_to_genai_function_declarations(
        as_openai_tools(schemas_for(registry))
    )

    declared = sum(len(tool.function_declarations) for tool in converted)
    assert declared == len(registry)


# --- end to end --------------------------------------------------------------


class PlannedProvider:
    """A provider whose next move is computed from live state.

    ``ScriptedProvider`` cannot express "pick a window from the band", because
    the band is not known until the sweep has run. This plays a fixed plan of
    steps, each of which resolves its own arguments at the moment it is reached
    -- which is what a real model does, and what makes the end-to-end assertion
    about real numbers rather than a rehearsed one.
    """

    name = "planned"
    model = "planned"

    def __init__(self, steps) -> None:
        self.steps = list(steps)

    def converse(self, system, messages, tools) -> AssistantTurn:
        if not self.steps:
            return AssistantTurn(text="finished")
        return self.steps.pop(0)()


def test_the_agent_drives_a_real_file_to_a_verified_csv(
    run_config, virial_argen_file, tmp_path
):
    config = run_config.model_copy(update={"source_file": virial_argen_file})
    surface = ToolSurface(
        state=RunState(run_id="e2e", source_file=str(virial_argen_file)), config=config
    )
    for key, value in ANSWERS.items():
        surface.set_declaration(key, value, author="tester")
        surface.declarations.refresh_form()

    out = tmp_path / "agent.csv"
    context = ToolContext(
        surface=surface,
        notes=NoteBook(tmp_path / "memory"),
        runs_dir=tmp_path / "runs",
        author="tester",
        output_csv=out,
    )

    def decide_pending():
        """What a researcher answers at the anomaly checkpoint: keep them."""
        pending = [a for a in surface.state.anomalies if a.decision == "pending"]
        if not pending:
            return AssistantTurn(tool_calls=[ToolCall("n", "get_state", {})])
        first = pending[0]
        return AssistantTurn(
            tool_calls=[
                ToolCall(
                    f"d{first.start_index}",
                    "set_anomaly_decision",
                    {
                        "start_index": first.start_index,
                        "end_index": first.end_index,
                        "decision": "keep",
                    },
                )
            ]
        )

    def choose_window():
        band = [int(c.value) for c in surface.get_shortlist()]
        assert band, "the sweep produced no safe band"
        return AssistantTurn(
            tool_calls=[ToolCall("w", "pick_window", {"window_length": band[0]})]
        )

    plan = [
        lambda: call("load_run"),
        lambda: call("check_declarations"),
        lambda: call("sweep_windows"),
        choose_window,
    ]
    # One decision step per region the detectors could flag, so the plan does
    # not have to know in advance how many there are.
    plan.extend([decide_pending] * 12)
    plan.extend(
        [
            lambda: call("emit_csv"),
            lambda: call("verify_output"),
            lambda: AssistantTurn(text="Run complete."),
        ]
    )

    agent = Orchestrator(
        context=context,
        provider=PlannedProvider(plan),
        approvals=ApprovalPolicy(ask=allow_all),
        max_steps=len(plan) + 2,
    )

    events = agent.ask("Process this run end to end.")

    assert out.exists(), "the agent never produced a file"
    frame, metadata = read_processed_csv(out)
    assert len(frame) > 0
    assert metadata["channel"] == "ls8", "only ls8 is analysed"
    assert surface.state.window_length in [int(c.value) for c in surface.get_shortlist()]

    verification = [e for e in events if e.kind == "result" and e.tool == "verify_output"]
    assert verification, "the agent emitted without verifying"
    assert "FAIL" not in verification[-1].text

    assert "emit_csv" in agent.committed and "pick_window" in agent.committed
    assert "load_run" not in agent.committed, "a compute tool must not count as a commit"


def test_extracting_parameters_recovers_the_known_answer(
    run_config, virial_argen_file, tmp_path
):
    """Part 2 through the tool surface, against a file with a known A2*Mw."""
    from tests.conftest import TRUE_A2, TRUE_MW

    config = run_config.model_copy(update={"source_file": virial_argen_file})
    surface = ToolSurface(
        state=RunState(run_id="p2", source_file=str(virial_argen_file)), config=config
    )
    for key, value in ANSWERS.items():
        surface.set_declaration(key, value, author="tester")
        surface.declarations.refresh_form()

    out = tmp_path / "p2.csv"
    context = ToolContext(
        surface=surface,
        notes=NoteBook(tmp_path / "memory"),
        runs_dir=tmp_path / "runs",
        author="tester",
        output_csv=out,
    )
    registry = build_registry()

    registry["load_run"].handler(context, {})
    registry["sweep_windows"].handler(context, {})
    surface.pick_window(int(surface.get_shortlist()[0].value))
    for record in surface.state.anomalies:
        if record.decision == "pending":
            surface.set_decision((record.start_index, record.end_index), "keep")
    registry["emit_csv"].handler(context, {})

    rendered = registry["extract_parameters"].handler(context, {})

    assert surface.state.parameters is not None
    recovered = surface.state.parameters.a2_mw
    expected = TRUE_A2 * TRUE_MW
    assert recovered == pytest.approx(expected, rel=0.05), (
        f"A2*Mw came back {recovered:.6g} against a known {expected:.6g}"
    )
    assert "withheld" in rendered, "an uncalibrated run must not report Mw"


# --- stopping ----------------------------------------------------------------
#
# A stop is cooperative and it has to be: a model call is a blocking request
# inside a vendor SDK and a tool is numpy over eighty thousand points, so the
# only safe places to end a turn are the boundaries between them.


def stopper(after_calls: int = 0):
    """A poll that goes true once a tripwire tool has run."""
    box = {"stop": after_calls == 0}
    return box, (lambda: box["stop"])


def test_a_stop_before_the_first_call_spends_nothing(context):
    """Asked to stop before it started, it does not call the model at all."""
    box, poll = stopper()
    provider = ScriptedProvider([AssistantTurn(text="never reached")])
    agent = Orchestrator(
        context=context, provider=provider, approvals=ApprovalPolicy(ask=allow_all),
        should_stop=poll,
    )

    events = agent.ask("go")
    assert [e.kind for e in events] == ["stopped", "spend"]
    assert provider.calls == [], "a stopped turn must not spend a request"
    assert agent.budget.spent.requests == 0


def test_a_stop_skips_the_rest_of_the_batch_but_answers_every_call(context):
    """The load-bearing one.

    A provider is handed an assistant turn that asked for three tools, and every
    one of them has to come back with a result or the next request is malformed
    and refused. So a stop costs the remaining calls -- and never the
    conversation, which is the thing the researcher would actually miss.
    """
    box = {"stop": False}

    def tripwire(ctx, args):
        box["stop"] = True
        return "the researcher hit stop while this was running"

    trip = Tool("tripwire", "read", "Test hook.", {"type": "object", "properties": {}}, tripwire)
    agent = Orchestrator(
        context=context,
        provider=ScriptedProvider(
            [
                AssistantTurn(
                    tool_calls=[
                        ToolCall("a", "tripwire", {}),
                        ToolCall("b", "get_state", {}),
                        ToolCall("c", "list_anomalies", {}),
                    ]
                ),
                AssistantTurn(text="never reached"),
            ]
        ),
        approvals=ApprovalPolicy(ask=allow_all),
        registry=build_registry(extra=[trip]),
        should_stop=lambda: box["stop"],
    )

    events = agent.ask("go")
    assert events[-2].kind == "stopped"

    # Only the tripwire ran; the two behind it were skipped, not executed.
    assert [c.name for c, r in agent.trace] == ["tripwire"]

    tool_message = agent.messages[-1]
    assert tool_message.role == "tool"
    assert len(tool_message.results) == 3, (
        "every tool call the model made must come back with a result, or the "
        "next request is malformed"
    )
    assert [r.call_id for r in tool_message.results] == ["a", "b", "c"]
    skipped = [r for r in tool_message.results if not r.ok]
    assert len(skipped) == 2
    assert "stopped the session" in skipped[0].content
    assert "Do not retry it" in skipped[0].content


def test_a_stop_between_rounds_ends_before_the_next_model_call(context):
    box = {"stop": False}

    def tripwire(ctx, args):
        box["stop"] = True
        return "ok"

    trip = Tool("tripwire", "read", "Test hook.", {"type": "object", "properties": {}}, tripwire)
    provider = ScriptedProvider(
        [
            AssistantTurn(tool_calls=[ToolCall("a", "tripwire", {})]),
            AssistantTurn(text="this second round must never happen"),
        ]
    )
    agent = Orchestrator(
        context=context, provider=provider, approvals=ApprovalPolicy(ask=allow_all),
        registry=build_registry(extra=[trip]), should_stop=lambda: box["stop"],
    )

    events = agent.ask("go")
    assert [c.name for c, r in agent.trace] == ["tripwire"]
    assert len(provider.calls) == 1, "the round after the stop was still requested"
    assert any(e.kind == "stopped" for e in events)
    # A stop is the researcher's decision, not a failure of the session.
    assert not any(e.kind in {"failed", "capped"} for e in events)


def test_the_run_keeps_whatever_finished_before_the_stop(context, tmp_path):
    """A stop is not a rollback. What ran, ran, and the state says so."""
    box = {"stop": False}

    def tripwire(ctx, args):
        box["stop"] = True
        return "ok"

    trip = Tool("tripwire", "read", "Test hook.", {"type": "object", "properties": {}}, tripwire)
    agent = Orchestrator(
        context=context,
        provider=ScriptedProvider(
            [
                AssistantTurn(
                    tool_calls=[
                        ToolCall("a", "set_declaration", {"key": "c0_g_per_cm3", "value": 0.004}),
                        ToolCall("b", "tripwire", {}),
                        ToolCall("c", "set_declaration", {"key": "ls_units", "value": "count_rate"}),
                    ]
                ),
                AssistantTurn(text="never reached"),
            ]
        ),
        approvals=ApprovalPolicy(ask=allow_all),
        registry=build_registry(extra=[trip]),
        should_stop=lambda: box["stop"],
    )
    agent.ask("go")

    declared = context.surface.state.declarations
    assert "c0_g_per_cm3" in declared, "a commit that completed before the stop stands"
    assert "ls_units" not in declared, "a commit behind the stop never ran"


def test_ctrl_c_ends_the_turn_and_the_second_one_ends_the_wait():
    """The terminal's stop, which is the same flag the window's button sets.

    Ctrl+C used to raise straight out through the generator: no spend figure, no
    pointer to the log, and a conversation left holding an assistant turn whose
    tool calls had no results. Now the first one asks the turn to end and the
    second one is the escape hatch, for when the provider is hung and the
    researcher wants their terminal back.
    """
    from continuum_spectranalytics.cli import Interruptible

    seen = []
    interrupts = Interruptible(on_notice=seen.append)

    with interrupts:
        assert interrupts() is False
        interrupts._handle(2, None)
        assert interrupts() is True, "the orchestrator polls this to end the turn"
        assert seen and "stopping" in seen[0]
        assert "again to give up" in seen[0], "the way out has to be on screen"

        with pytest.raises(KeyboardInterrupt):
            interrupts._handle(2, None)

    # Re-entering clears it: a stop ends one question, not the session.
    with interrupts:
        assert interrupts() is False


def test_an_interruptible_turn_off_the_main_thread_is_not_an_error():
    """Signals only exist on the main thread; a worker just runs to completion."""
    import threading

    from continuum_spectranalytics.cli import Interruptible

    outcome = []

    def worker():
        interrupts = Interruptible(on_notice=lambda _m: None)
        with interrupts:
            outcome.append(interrupts())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(10)
    assert outcome == [False]
