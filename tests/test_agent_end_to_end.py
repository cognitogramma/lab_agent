"""The agent graph, driving a real file from raw data to a verified CSV.

The other agent tests exercise the control flow against hand-built state, which
is how a disconnected graph passed a full test suite: every node was tested,
and none of them was connected to the measurement code. These tests assert the
thing that was missing -- that driving the graph produces a file on disk with
the right numbers in it.
"""

from __future__ import annotations

import pytest

from polyrmc.agent.graph import AgentGraph, Node
from polyrmc.agent.provider import Resilient, StubProvider
from polyrmc.io_csv import COLUMNS, read_processed_csv
from polyrmc.state import RunState
from polyrmc.tools import ToolSurface

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


@pytest.fixture
def graph(run_config, virial_argen_file, tmp_path):
    config = run_config.model_copy(update={"source_file": virial_argen_file})
    tools = ToolSurface(
        state=RunState(run_id="e2e", source_file=str(virial_argen_file)), config=config
    )
    return AgentGraph(tools=tools, output_csv=tmp_path / "agent.csv")


def declare(graph):
    """Answer the intake interview the way a researcher would."""
    for key, value in ANSWERS.items():
        graph.tools.set_declaration(key, value, author="tester")
        graph.tools.declarations.refresh_form()
    return graph


def decide_anomalies(graph, decision="keep"):
    """Answer the section 6.4 checkpoint: what to do with unnamed regions."""
    for record in graph.state.anomalies:
        if record.decision == "pending":
            graph.tools.set_decision(
                (record.start_index, record.end_index), decision
            )
    return graph


def drive(graph, max_steps=40):
    """Run the graph, answering each checkpoint as a researcher would."""
    for _ in range(max_steps):
        graph.run(max_steps=max_steps)
        if graph.pending is None or graph.node is Node.DONE:
            break
        if graph.node is Node.REVIEW_ANOMALIES:
            decide_anomalies(graph)
        elif graph.node is Node.REVIEW_ANCHORS:
            graph.state.anchor_selected_by = "researcher"  # "no correction"
        else:
            break
        graph.resume()
    return graph


def test_the_graph_stops_at_intake_before_touching_the_data(graph):
    steps = graph.run()

    assert steps[-1].paused
    assert steps[-1].interrupt.node is Node.INTAKE
    assert graph.state.raw_signal is None, "nothing was loaded before it was declared"


def test_a_declared_run_goes_all_the_way_to_a_verified_csv(graph, tmp_path):
    """The whole point: declare, then let the graph drive."""
    drive(declare(graph))

    assert graph.node is Node.DONE, graph.narrate()
    assert graph.emitted is not None
    assert graph.emitted.exists()

    frame, metadata = read_processed_csv(graph.emitted)
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) > 100
    assert metadata["run_id"] == graph.config.run_id

    failures = [c for c in graph.checks if c.failed]
    assert not failures, [c.detail for c in failures]


def test_the_graph_actually_processed_rather_than_narrating(graph):
    drive(declare(graph))

    # State the measurement path fills in, which a disconnected graph left empty.
    assert graph.state.raw_signal is not None
    assert graph.state.smoothed_signal is not None
    assert graph.state.window_length is not None
    assert graph.state.noise_floor is not None
    assert graph.state.m0 is not None


def test_the_two_lane_judge_chose_the_window_that_was_used(graph):
    """The lanes are on the live path now, not a parallel one."""
    declare(graph)
    graph.guard = Resilient(primary=StubProvider(['{"window_length": 0}'] * 4))
    graph.vlm_guard = Resilient()
    graph.render_plot = lambda _s, _w: b"png"

    # Advance to the point the band exists, then answer with a real band member.
    graph.node = Node.PROCESS
    graph.advance()  # load and sweep
    decide_anomalies(graph)
    while graph.node is not Node.JUDGE_WINDOW:
        graph.advance()  # declaration check, then the section 6.4 checkpoint

    band = [int(c.value) for c in graph.tools.get_shortlist()]
    assert band

    chosen = band[-1]
    graph.guard = Resilient(primary=StubProvider([f'{{"window_length": {chosen}}}']))
    graph.vlm_guard = Resilient(primary=StubProvider([f'{{"window_length": {chosen}}}']))
    graph.advance()

    assert graph.state.judge_status == "both_lanes"
    assert graph.state.window_length == chosen

    drive(graph)
    _frame, metadata = read_processed_csv(graph.emitted)
    assert int(float(metadata["smoothing_window"])) == chosen


def test_an_outage_still_produces_a_file(graph):
    """I8 end to end: no judge reachable, run completes, fallback recorded."""
    drive(declare(graph))

    assert graph.state.fallback_taken
    assert graph.state.judge_status == "fallback"
    assert graph.emitted.exists()
    assert "conservative option" in graph.narrate()


def test_emit_is_refused_while_a_conflict_is_unresolved(graph):
    from polyrmc.state import Conflict

    declare(graph)
    graph.node = Node.PROCESS
    graph.advance()
    decide_anomalies(graph)
    graph.state.declaration_conflicts = [
        Conflict(check="experiment_type_vs_trace", declared="a", observed="b")
    ]
    graph.node = Node.EMIT
    step = graph.advance()

    assert step.paused
    assert "unresolved declaration conflict" in step.interrupt.question
    assert graph.emitted is None


def test_narration_reports_the_real_counts(graph):
    drive(declare(graph))
    text = graph.narrate()

    assert "window(s) in the safe band" in text
    assert "verification:" in text
    assert f"emitted {graph.emitted}" in text
