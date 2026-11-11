"""Exercise saved specialists through the ordinary assistant loop without a live model."""

from __future__ import annotations

import json
from threading import Event

import pytest

from continuum_spectranalytics.agent.budget import Budget, Usage
from continuum_spectranalytics.agent.orchestrator import Approval, ApprovalPolicy, Orchestrator
from continuum_spectranalytics.agent.subagents import SubagentStore
from continuum_spectranalytics.agent.toolcall import AssistantTurn, ScriptedProvider, ToolCall
from continuum_spectranalytics.agent.toolspec import ToolContext
from continuum_spectranalytics.memory.notes import NoteBook
from continuum_spectranalytics.state import RunState
from continuum_spectranalytics.tools import Rejected, ToolSurface


def tool(name, /, **arguments):
    return ToolCall(id="", name=name, arguments=arguments)


def turn(*calls):
    return AssistantTurn(tool_calls=list(calls), usage=Usage(requests=1))


@pytest.fixture
def context(run_config, tmp_path):
    surface = ToolSurface(
        state=RunState(run_id="current-run", source_file=str(run_config.source_file)),
        config=run_config,
    )
    return ToolContext(surface=surface, notes=NoteBook(tmp_path / "memory"), runs_dir=tmp_path / "runs")


@pytest.mark.parametrize("approved", [False, True])
def test_create_subagent_uses_approval_and_saved_profiles_survive_reopening(context, approved):
    arguments = dict(name="Run Reviewer (QA)", instructions="Identify caveats.", scope="run_review")
    reviewed = []

    def review(spec, call):
        reviewed.append((spec.name, call.arguments))
        return Approval(approved)

    provider = ScriptedProvider([turn(tool("create_subagent", **arguments)), "Done."])
    agent = Orchestrator(context, provider, approvals=ApprovalPolicy(ask=review))
    agent.ask("Create a reviewer.")

    assert reviewed == [("create_subagent", arguments)]
    assert agent.trace[0][1].ok == approved
    store = SubagentStore(context.notes.directory)
    if not approved:
        assert store.list() == []
        return

    assert store.get("run reviewer (qa)").instructions == arguments["instructions"]
    with pytest.raises(Rejected, match="already exists"):
        store.create("RUN REVIEWER (QA)", "Replace it.", "general")
    later = Orchestrator(context, ScriptedProvider([turn(tool("list_subagents")), "Listed."]))
    later.ask("List my specialists.")
    assert json.loads(later.trace[0][1].content) == [dict(name=arguments["name"], scope="run_review")]
    assert store.get(arguments["name"]).instructions == arguments["instructions"]


@pytest.mark.parametrize("scope", ["general", "run_review"])
def test_delegation_is_isolated_and_returns_a_report_to_the_parent(context, scope):
    SubagentStore(context.notes.directory).create("Reviewer", "Identify caveats.", scope)
    provider = ScriptedProvider([
        turn(tool("run_subagent", name="Reviewer", task="Check for missing declarations.")),
        turn(
            tool("get_state"),
            tool("write_note", subject="Forbidden", body="Do not store this."),
            tool("load_run"),
            tool("run_subagent", name="Reviewer", task="Delegate again."),
        ),
        "Report: review the intake form.",
        "The reviewer recommends checking the intake form.",
    ])
    agent = Orchestrator(context, provider)
    events = agent.ask("Parent-only context. Run my reviewer.")

    system, messages, schemas = provider.calls[1]
    assert "Identify caveats." in system
    assert [(m.role, m.text) for m in messages] == [("user", "Check for missing declarations.")]
    names = {schema["function"]["name"] for schema in schemas}
    if scope == "general":
        assert names == set()
    else:
        assert "get_state" in names
    assert not names & {"write_note", "load_run", "run_subagent"}

    parent_result = provider.calls[3][1][-1].results[0]
    assert parent_result.ok
    report = json.loads(parent_result.content)
    assert report["report"] == "Report: review the intake form."
    assert {item["name"]: item["ok"] for item in report["tools"]} == {
        "get_state": scope == "run_review", "write_note": False, "load_run": False, "run_subagent": False,
    }
    assert len(context.notes) == 0
    assert context.surface.state.raw_signal is None
    assert [e.text for e in events if e.kind == "text"] == ["The reviewer recommends checking the intake form."]


def test_child_requests_share_the_parent_budget_and_appear_in_spend(context):
    SubagentStore(context.notes.directory).create("Reviewer", "Identify caveats.", "run_review")
    provider = ScriptedProvider([
        turn(tool("run_subagent", name="Reviewer", task="Check the run.")),
        turn(tool("get_state")),
    ])
    budget = Budget(max_requests=2)
    agent = Orchestrator(context, provider, budget=budget)

    events = agent.ask("Run my reviewer.")

    assert budget.spent.requests == 2
    assert len(provider.calls) == 2
    assert not agent.trace[0][1].ok
    assert "session request budget reached" in agent.trace[0][1].content
    assert any(e.kind == "capped" for e in events)
    assert "this turn: 2 request(s)" in events[-1].text


def test_parent_and_child_share_one_web_allowance(context, monkeypatch):
    from continuum_spectranalytics import research

    searches = []

    def search(**kwargs):
        searches.append(kwargs["query"])
        return {"results": []}

    monkeypatch.setattr(research, "search_internet", search)
    SubagentStore(context.notes.directory).create("Literature Reader", "Find public sources.", "literature")
    provider = ScriptedProvider([
        turn(
            tool("search_internet", query="parent search"),
            tool("run_subagent", name="Literature Reader", task="Find public references."),
            tool("search_internet", query="parent after child"),
        ),
        turn(tool("search_internet", query="child first"), tool("search_internet", query="child second")),
        "I found no references before the limit.",
        "Finished.",
    ])
    agent = Orchestrator(context, provider, max_web_calls=2)

    agent.ask("Search and delegate.")

    assert searches == ["parent search", "child first"]
    assert "web_budget_exhausted" in provider.calls[2][1][-1].results[1].content
    assert "web_budget_exhausted" in agent.trace[-1][1].content


def test_stopping_during_delegation_discards_the_childs_report(context, monkeypatch):
    SubagentStore(context.notes.directory).create("Reviewer", "Identify caveats.", "general")
    provider = ScriptedProvider([
        turn(tool("run_subagent", name="Reviewer", task="Check for caveats.")),
        "This report arrived after Stop.",
    ])
    stop = Event()
    converse = provider.converse

    def stop_after_reply(*args):
        reply = converse(*args)
        if len(provider.calls) == 2:
            stop.set()
        return reply

    monkeypatch.setattr(provider, "converse", stop_after_reply)
    agent = Orchestrator(context, provider, should_stop=stop.is_set)

    events = agent.ask("Run my reviewer.")

    assert not agent.trace[0][1].ok
    assert "stopped the subagent task" in agent.trace[0][1].content
    assert any(e.kind == "stopped" for e in events)
    assert not any("This report arrived after Stop" in e.text for e in events)
