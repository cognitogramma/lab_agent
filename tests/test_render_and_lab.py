"""The two pieces that turn the agent from plausible into working.

Without a renderer the visual lane has nothing to read, so the two-lane judge
silently becomes a one-lane judge while still reporting a two-lane status.
Without LAB.md loaded the agent answers from general knowledge of light
scattering rather than this lab's conventions.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyrmc.agent.lab import MISSING_NOTICE, find_lab_file, load_lab_context, system_prompt_with_lab
from polyrmc.agent.render import default_render_plot, render_window_overlays
from polyrmc.state import RunState


@pytest.fixture
def state() -> RunState:
    time_s = np.arange(0.0, 4000.0, 1.0)
    rng = np.random.default_rng(3)
    run = RunState(run_id="render-test", source_file="none")
    run.time_s = time_s
    run.spliced_signal = (
        0.20
        + 0.60 / (1.0 + np.exp(-(time_s - 2000.0) / 200.0))
        + rng.normal(0, 0.002, time_s.size)
    )
    return run


# --- the renderer -------------------------------------------------------------


def test_a_png_comes_out(state):
    image = render_window_overlays(state, [11, 31, 101])

    assert image is not None
    assert image[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    assert len(image) > 5000, "suspiciously small for a plot"


def test_nothing_to_draw_returns_none_rather_than_a_blank(state):
    """A blank image would be read as a picture; None is read as no reading."""
    assert render_window_overlays(state, []) is None
    assert render_window_overlays(RunState(run_id="x", source_file="y"), [11]) is None


def test_an_all_nan_trace_returns_none(state):
    state.spliced_signal = np.full(state.time_s.size, np.nan)
    assert render_window_overlays(state, [11]) is None


def test_a_window_the_trace_cannot_take_is_skipped_not_faked(state):
    """Drawing a substitute under a window's label would misinform the lane."""
    image = render_window_overlays(state, [11, 999999])
    assert image is not None  # the usable window still renders


def test_the_default_hook_matches_the_graph_signature(state):
    state.polyorder = 2
    assert default_render_plot(state, [11, 31]) is not None


def test_rendering_is_bounded_regardless_of_series_length():
    """Eighty thousand points cost seconds and buy nothing over three thousand."""
    import time

    time_s = np.arange(0.0, 79383.0, 1.0)
    big = RunState(run_id="big", source_file="none")
    big.time_s = time_s
    big.spliced_signal = np.linspace(0.2, 0.8, time_s.size)

    start = time.time()
    image = render_window_overlays(big, [11, 51, 201])
    elapsed = time.time() - start

    assert image is not None
    assert elapsed < 20.0, f"rendering took {elapsed:.1f}s"


# --- LAB.md -------------------------------------------------------------------


def test_the_projects_lab_file_is_found():
    path = find_lab_file(__file__)
    assert path is not None, "LAB.md should be found by walking up from tests/"
    assert path.name == "LAB.md"


def test_the_lab_context_loads_whole_not_in_fragments():
    context, path = load_lab_context(__file__)

    assert path is not None
    # Whole means whole: the unconfirmed values and the house conventions are
    # both present, and a top-k retrieval would have returned only one of them.
    assert "UNCONFIRMED" in context
    assert "House conventions" in context
    assert "R_toluene" in context
    assert len(context) > 1000


def test_a_missing_lab_file_is_reported_not_tolerated(tmp_path):
    context, path = load_lab_context(tmp_path)

    assert path is None
    assert context == MISSING_NOTICE
    assert "do not substitute general knowledge" in context


def test_the_system_prompt_carries_the_lab_context():
    prompt = system_prompt_with_lab("BASE PROMPT", __file__)

    assert prompt.startswith("BASE PROMPT")
    assert "# Lab context" in prompt
    assert "House conventions" in prompt


def test_the_graph_defaults_to_a_real_renderer_and_real_lab_context(run_config):
    from polyrmc.agent.graph import AgentGraph
    from polyrmc.tools import ToolSurface

    graph = AgentGraph(
        tools=ToolSurface(
            state=RunState(run_id="g", source_file="none"), config=run_config
        )
    )

    assert graph.render_plot is not None, "the visual lane would be blind"
    assert graph.lab_path is not None
    assert "House conventions" in graph.system_prompt
