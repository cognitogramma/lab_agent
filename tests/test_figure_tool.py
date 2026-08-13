"""The figure tool: what it draws, and what it refuses to draw.

Most of these are refusals. A plotting tool is unusually good at producing a
confident-looking wrong answer -- two runs on one axis in units they do not
share, or a figure whose axes are labelled from a configuration nobody could
read -- and the refusals are where that is prevented.
"""

from __future__ import annotations

import pytest

from polyrmc.agent.toolspec import ToolContext, build_registry
from polyrmc.memory.notes import NoteBook
from polyrmc.state import RunState
from polyrmc.tools import Rejected, ToolSurface
from tests.test_rehydrate import write_run

import json


@pytest.fixture
def runs_dir(tmp_path, run_config) -> object:
    """Three drawable runs, all with configurations that still validate."""
    config = json.loads(run_config.model_dump_json())
    directory = tmp_path / "runs"
    for name in ("alpha", "beta", "gamma"):
        write_run(directory, f"{name}_ls8", name, config=config)
    return directory


@pytest.fixture
def context(run_config, runs_dir, tmp_path) -> ToolContext:
    surface = ToolSurface(
        state=RunState(run_id="alpha", source_file=str(run_config.source_file)),
        config=run_config,
    )
    return ToolContext(
        surface=surface,
        notes=NoteBook(tmp_path / "memory"),
        runs_dir=runs_dir,
        author="tester",
    )


def draw(context: ToolContext, **arguments) -> str:
    return build_registry()["plot_figure"].handler(context, arguments)


def test_the_tool_is_compute_tier(context):
    """Drawing is not a commit. It writes a file but decides nothing."""
    assert build_registry()["plot_figure"].tier == "compute"


def test_index_overlays_several_runs_into_one_png(context, runs_dir):
    result = draw(context, kind="index", run_ids=["alpha", "beta"])

    written = runs_dir / "figures" / "index_alpha_ls8__vs__beta_ls8.png"
    assert written.exists()
    assert written.stat().st_size > 5_000
    assert "alpha_ls8" in result and "beta_ls8" in result


def test_omitting_run_ids_draws_the_current_run(context, runs_dir):
    draw(context, kind="index")

    assert (runs_dir / "figures" / "index_alpha_ls8.png").exists()


def test_a_single_run_diagnostic_writes_its_own_file(context, runs_dir):
    draw(context, kind="derived", run_ids=["beta"])

    assert (runs_dir / "figures" / "derived_beta_ls8.png").exists()


def test_drawing_is_recorded_in_the_run_history(context):
    draw(context, kind="index", run_ids=["alpha"])

    assert any("drew index figure" in h for h in context.surface.state.execution_history)


# --- refusals ---------------------------------------------------------------


def test_an_unknown_kind_is_refused_with_the_list(context):
    with pytest.raises(Rejected, match="kind must be one of"):
        draw(context, kind="scatter")


def test_count_figures_refuse_to_carry_a_comparison(context):
    """Two runs' detector counts are not on one scale, so they get one axis each."""
    with pytest.raises(Rejected) as error:
        draw(context, kind="raw", run_ids=["alpha", "beta"])

    message = str(error.value)
    assert "detector counts" in message
    assert "kind='index'" in message  # names the figure that can compare


def test_too_many_runs_is_refused_rather_than_drawn_illegibly(context):
    with pytest.raises(Rejected, match="unreadable"):
        draw(context, kind="index", run_ids=["a", "b", "c", "d", "e", "f", "g"])


def test_an_unknown_run_is_refused_with_what_is_available(context):
    with pytest.raises(Rejected) as error:
        draw(context, kind="index", run_ids=["nonesuch"])

    message = str(error.value)
    assert "nonesuch" in message
    assert "drawable" in message and "alpha" in message


def test_a_figure_needing_an_unreadable_config_is_refused(tmp_path, run_config):
    """The 65 legacy runs can be overlaid, but not labelled from their config."""
    config = json.loads(run_config.model_dump_json())
    directory = tmp_path / "runs"
    write_run(directory, "legacy_ls8", "legacy",
              config=dict(config, experiment_type="fixed_concentration"))
    surface = ToolSurface(
        state=RunState(run_id="legacy", source_file="raw.txt"), config=run_config
    )
    context = ToolContext(
        surface=surface, notes=NoteBook(tmp_path / "memory"), runs_dir=directory
    )

    with pytest.raises(Rejected) as error:
        draw(context, kind="derived", run_ids=["legacy"])
    assert "does not validate" in str(error.value)

    # ... but the comparable figure reads no config and still works.
    draw(context, kind="index", run_ids=["legacy"])
    assert (directory / "figures" / "index_legacy_ls8.png").exists()


def test_a_gap_in_the_record_is_reported_and_printed_on_the_figure(tmp_path, run_config):
    config = json.loads(run_config.model_dump_json())
    directory = tmp_path / "runs"
    write_run(directory, "old_ls8", "old", config=config, ingest={"dropped_rows": []})
    surface = ToolSurface(
        state=RunState(run_id="old", source_file="raw.txt"), config=run_config
    )
    context = ToolContext(
        surface=surface, notes=NoteBook(tmp_path / "memory"), runs_dir=directory
    )

    result = draw(context, kind="index", run_ids=["old"])

    assert "printed on the figure" in result
    assert "ND-filter" in result


def test_an_irrelevant_gap_is_not_printed_on_the_figure(tmp_path, run_config):
    """An unreadable config says nothing about a figure that never reads one."""
    config = json.loads(run_config.model_dump_json())
    directory = tmp_path / "runs"
    write_run(directory, "legacy_ls8", "legacy",
              config=dict(config, experiment_type="fixed_concentration"))
    surface = ToolSurface(
        state=RunState(run_id="legacy", source_file="raw.txt"), config=run_config
    )
    context = ToolContext(
        surface=surface, notes=NoteBook(tmp_path / "memory"), runs_dir=directory
    )

    result = draw(context, kind="index", run_ids=["legacy"])

    assert "printed on the figure" not in result


# --- working with a file, rather than with the archive ----------------------


def open_file(context: ToolContext, **arguments) -> str:
    return build_registry()["open_data_file"].handler(context, arguments)


def plot_cols(context: ToolContext, **arguments) -> str:
    return build_registry()["plot_columns"].handler(context, arguments)


def test_opening_a_file_is_read_tier_and_plotting_it_is_compute(context):
    """Looking costs nothing and decides nothing. Drawing writes a file."""
    registry = build_registry()
    assert registry["open_data_file"].tier == "read"
    assert registry["plot_columns"].tier == "compute"


def test_opening_a_raw_file_reports_its_columns(context, argen_file):
    result = open_file(context, path=str(argen_file))

    assert "argen" in result
    assert "ls16" in result
    assert "not plottable: Elapsed Time" in result


def test_plotting_columns_from_a_raw_file(context, argen_file, runs_dir):
    result = plot_cols(context, path=str(argen_file), x="seconds", y=["ls8", "ls9"])

    assert "wrote" in result
    assert "exploratory" in result
    written = list((runs_dir / "figures").glob("cols_*.png"))
    assert len(written) == 1 and written[0].stat().st_size > 5_000


def test_a_mistyped_column_comes_back_correctable(context, argen_file):
    """The refusal names the columns that do exist, so the next call can work."""
    with pytest.raises(Rejected) as error:
        plot_cols(context, path=str(argen_file), x="seconds", y=["ls99"])

    assert "ls99" in str(error.value)
    assert "seconds" in str(error.value)


def test_a_kaleidagraph_report_is_refused_by_both_tools(context, tmp_path):
    path = tmp_path / "run.qdr"
    path.write_bytes(b"\x00binary")

    for call in (open_file, plot_cols):
        with pytest.raises(Rejected, match="KaleidaGraph"):
            call(context, path=str(path), x="a", y=["b"])


def test_plotting_needs_columns_named(context, argen_file):
    with pytest.raises(Rejected, match="x column and at least one y"):
        plot_cols(context, path=str(argen_file), x="seconds", y=[])
