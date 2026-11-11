"""Reading finished runs back off disk, and drawing them again.

The cases that matter here are all about what a record does *not* contain.
This lab's archive is 120 runs written under an older schema, and the failure
mode worth testing is not a crash -- it is a figure that looks complete while
quietly standing in for something nobody recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from continuum_spectranalytics import figures
from continuum_spectranalytics.io_csv import write_processed_csv
from continuum_spectranalytics.rehydrate import find_run, list_past_runs, load_past_run


@pytest.fixture
def config_dict(run_config) -> dict:
    """A configuration that validates against today's schema."""
    return json.loads(run_config.model_dump_json())


def write_run(
    runs_dir: Path,
    stem: str,
    run_id: str | None = None,
    *,
    config: dict | None = None,
    ingest: dict | None = None,
    n: int = 60,
    with_csv: bool = True,
) -> Path:
    """Write one processed CSV and its sidecar, the way the pipeline does."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    seconds = np.arange(n, dtype=float)
    if with_csv:
        write_processed_csv(
            runs_dir / f"{stem}.csv",
            {
                "seconds": seconds,
                "concentration": np.full(n, np.nan),
                "ls8": np.linspace(1.0, 2.0, n),
                "ls8_smoothed": np.linspace(1.0, 2.0, n),
                "I_rayleigh": np.linspace(0.1, 0.2, n),
                "Mw_over_M0": np.linspace(1.0, 1.8, n),
            },
            metadata={"run_id": run_id or stem, "channel": "ls8"},
        )
    record = {
        "schema_version": 2,
        "written_at": "2026-08-01T00:00:00Z",
        "run_id": run_id or stem,
        "source_file": "raw.txt",
        "config": config if config is not None else {},
        "ingest": ingest if ingest is not None else {"nd_filter_changes": []},
        "anomalies": [],
        "splice": {"excised_indices": []},
    }
    path = runs_dir / f"{stem}.provenance.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_round_trip_reads_the_trace_and_its_marks(tmp_path, config_dict):
    runs = tmp_path / "runs"
    write_run(runs, "sample_ls8", "sample", config=config_dict,
              ingest={"nd_filter_changes": [12, 30]})

    run = load_past_run("sample", runs)

    assert run.run_id == "sample"
    assert len(run.frame) == 60
    assert list(run.frame.columns[:2]) == ["seconds", "concentration"]
    assert run.state.nd_filter_changes == [12, 30]
    assert run.config is not None
    assert run.unrecorded == {}


def test_absent_nd_register_is_reported_not_read_as_zero(tmp_path, config_dict):
    """A schema-1 run has no ND record. That is not the same as a clean run."""
    runs = tmp_path / "runs"
    write_run(runs, "old_ls8", "old", config=config_dict, ingest={"dropped_rows": []})

    run = load_past_run("old", runs)

    assert run.state.nd_filter_changes == []
    assert "nd_filter_changes" in run.unrecorded
    assert "cannot be ruled out" in run.unrecorded["nd_filter_changes"]


def test_config_that_predates_the_schema_is_kept_readable(tmp_path, config_dict):
    """65 real runs declare an experiment_type the current model rejects.

    They stay loadable, the config is reported as unreadable rather than
    guessed at, and nothing translates it to the nearest current spelling.
    """
    runs = tmp_path / "runs"
    legacy = dict(config_dict, experiment_type="fixed_concentration")
    write_run(runs, "legacy_ls8", "legacy", config=legacy)

    run = load_past_run("legacy", runs)

    assert run.config is None
    assert "config" in run.unrecorded
    assert "experiment_type" in run.unrecorded["config"]
    assert len(run.frame) == 60  # the data is still there


def test_a_run_id_naming_several_channels_is_refused(tmp_path, config_dict):
    """example-001 is three files here. Picking one silently draws the wrong one."""
    runs = tmp_path / "runs"
    for channel in ("ls8", "Cell_1", "Cell_8"):
        write_run(runs, f"multi_{channel}", "multi", config=config_dict)

    with pytest.raises(FileNotFoundError) as error:
        load_past_run("multi", runs)

    message = str(error.value)
    assert "3 processed files" in message
    assert "multi_Cell_1" in message and "multi_ls8" in message


def test_the_file_stem_resolves_what_the_run_id_cannot(tmp_path, config_dict):
    runs = tmp_path / "runs"
    for channel in ("ls8", "Cell_1"):
        write_run(runs, f"multi_{channel}", "multi", config=config_dict)

    assert find_run("multi_Cell_1", runs).name == "multi_Cell_1.provenance.json"


def test_provenance_without_data_is_refused_by_name(tmp_path, config_dict):
    runs = tmp_path / "runs"
    write_run(runs, "gone_ls8", "gone", config=config_dict, with_csv=False)

    with pytest.raises(FileNotFoundError, match="no boundary CSV"):
        load_past_run("gone", runs)


def test_listing_offers_only_what_can_actually_be_drawn(tmp_path, config_dict):
    runs = tmp_path / "runs"
    write_run(runs, "good_ls8", "good", config=config_dict)
    write_run(runs, "orphan_ls8", "orphan", config=config_dict, with_csv=False)

    listed = list_past_runs(runs)

    assert listed == ["good"]


def test_everything_listed_can_be_loaded(tmp_path, config_dict):
    """The invariant the model depends on: the list is not a list of traps."""
    runs = tmp_path / "runs"
    for channel in ("ls8", "Cell_1"):
        write_run(runs, f"multi_{channel}", "multi", config=config_dict)
    write_run(runs, "single_ls8", "single", config=config_dict)

    listed = list_past_runs(runs)

    assert len(listed) == 3
    for name in listed:
        load_past_run(name, runs)  # must not raise


def test_missing_runs_directory_lists_nothing(tmp_path):
    assert list_past_runs(tmp_path / "nowhere") == []


# --- the overlay ------------------------------------------------------------


def test_overlay_writes_a_png_for_several_runs(tmp_path, config_dict):
    runs = tmp_path / "runs"
    write_run(runs, "a_ls8", "a", config=config_dict)
    write_run(runs, "b_ls8", "b", config=config_dict)
    loaded = [load_past_run(name, runs) for name in ("a", "b")]

    out = figures.plot_overlay(
        [(r.state, r.frame) for r in loaded],
        tmp_path / "overlay.png",
        caveats=["one caveat"],
    )

    assert out.exists()
    assert out.stat().st_size > 5_000  # a real drawing, not an empty canvas


def test_overlay_refuses_an_empty_set(tmp_path):
    with pytest.raises(ValueError, match="no runs"):
        figures.plot_overlay([], tmp_path / "nothing.png")
