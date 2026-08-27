"""Is the route actually the model's, or is it a pipeline with a chat box?

The claim this file checks is not "the model is consulted". A fixed graph that
calls a model at one node also consults a model. The claim is that *nothing
decides what happens next except the model*, and it is checkable three ways:

* the same loop runs whatever order it is given, including orders that skip
  most of the pipeline entirely;
* the constraints it does enforce are about **validity**, not sequence -- a
  refusal names a precondition and what would be valid, and the model is free
  to go somewhere else entirely afterwards;
* the gates that matter hold against the model, not through it.
"""

from __future__ import annotations

import pytest

from polyrmc.agent.orchestrator import ApprovalPolicy, Orchestrator, allow_all, deny_all
from polyrmc.agent.toolcall import AssistantTurn, ScriptedProvider, ToolCall
from polyrmc.agent.toolspec import ToolContext, build_registry
from polyrmc.memory.notes import NoteBook
from polyrmc.state import RunState
from polyrmc.tools import ToolSurface


def call(*names_and_args) -> AssistantTurn:
    return AssistantTurn(
        tool_calls=[
            ToolCall(id=f"c{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(names_and_args)
        ]
    )


@pytest.fixture
def context(run_config, tmp_path) -> ToolContext:
    surface = ToolSurface(
        state=RunState(run_id="route", source_file=str(run_config.source_file)),
        config=run_config,
    )
    return ToolContext(
        surface=surface, notes=NoteBook(tmp_path / "m"), runs_dir=tmp_path / "r",
        author="tester",
    )


def drive(context, turns, policy=None):
    agent = Orchestrator(
        context=context,
        provider=ScriptedProvider(turns),
        approvals=policy or ApprovalPolicy(ask=allow_all),
        registry=build_registry(),
    )
    return agent, agent.ask("go")


def ran(agent) -> list[str]:
    return [c.name for c, r in agent.trace if r.ok]


def test_a_route_that_never_touches_the_run_at_all(context):
    """A question about history loads nothing. There is no obligatory first step."""
    agent, _ = drive(
        context,
        [call(("search_past_runs", {"query": "lysozyme"}), ("read_notes", {})), "nothing like it."],
    )
    assert ran(agent) == ["search_past_runs", "read_notes"]
    assert context.surface.state.raw_signal is None, "nothing was loaded, and nothing had to be"


def test_the_same_loop_runs_a_completely_different_route(context):
    """No shared setup, no common prefix -- the loop carries no state about order."""
    agent, _ = drive(
        context,
        [call(("load_run", {})), call(("sweep_windows", {})), "the band is computed."],
    )
    assert ran(agent) == ["load_run", "sweep_windows"]
    assert context.surface.state.raw_signal is not None


def test_a_precondition_is_not_a_route(context):
    """Out of order is refused by name, and the model may then go anywhere.

    This is the distinction the whole architecture turns on. A refusal says
    *this call is invalid and here is what would be valid* -- it does not say
    *do the next step of the pipeline*. The model answers a refused sweep by
    searching past runs, which no pipeline would have let it do, and the loop
    does not care.
    """
    agent, _ = drive(
        context,
        [
            call(("sweep_windows", {})),                    # nothing is loaded yet
            call(("search_past_runs", {"query": "bsa"})),   # ... so go somewhere else
            "I could not sweep, so I looked at what we ran before instead.",
        ],
    )
    refused = [(c.name, r.content) for c, r in agent.trace if not r.ok]
    assert refused and refused[0][0] == "sweep_windows"
    assert "load_run" in refused[0][1], "a refusal has to name what would be valid"
    assert ran(agent) == ["search_past_runs"]


def test_the_gates_hold_against_the_model_not_through_it(context):
    """Asked to emit a run that is not fit to leave, the tool refuses.

    Not the prompt, and not the loop: the model asked, and code said no. The
    reasons are the run's, and they are returned to the model to act on.
    """
    agent, _ = drive(context, [call(("emit_csv", {})), "it refused, and it was right."])
    refused = [r.content for c, r in agent.trace if not r.ok]
    assert refused, "emit_csv on an empty run must not succeed"
    assert "REJECTED" in refused[0]
    assert "no smoothing window has been chosen" in refused[0]


def test_a_commit_the_researcher_declines_is_not_routed_around(context):
    """The loop offers no way to get the effect another way."""
    agent, _ = drive(
        context,
        [call(("pick_window", {"window_length": 31})), "understood, I will not retry."],
        policy=ApprovalPolicy(ask=deny_all),
    )
    assert ran(agent) == []
    assert context.surface.state.window_length is None
