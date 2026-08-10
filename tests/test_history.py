"""Tests for historical run memory registry, sidecar indexer, and history search."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from polyrmc.config import RunConfig
from polyrmc.memory.registry import RunRegistry, RunSummary, summary_from_sidecar
from polyrmc.state import RunState
from polyrmc.tools import ToolSurface


@pytest.fixture
def mock_provenance_dir(tmp_path: Path) -> Path:
    """Create a temporary directory containing synthetic .provenance.json files."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    record_bsa = {
        "run_id": "bsa_test_01",
        "source_file": "data/raw/bsa_01.csv",
        "written_at": "2026-08-01T10:00:00Z",
        "config": {
            "experiment_type": "continuous_dilution",
            "sample_name": "BSA",
            "smoothing": {"window_length": 15},
        },
        "declarations": {
            "sample_name": {"value": "BSA"},
            "polymer_type": {"value": "Protein"},
        },
        "parameters": {
            "mw": 66430.0,
            "a2": 0.0005,
            "a2_mw": 33.215,
            "k_d": -0.01,
            "calibrated": True,
        },
        "anomalies": [{"start_index": 10, "end_index": 15, "anomaly_type": "stirrer_spike"}],
        "warnings": [],
    }

    record_dextran = {
        "run_id": "dextran_test_02",
        "source_file": "data/raw/dextran_02.csv",
        "written_at": "2026-08-05T14:30:00Z",
        "config": {
            "experiment_type": "continuous_dilution",
            "sample_name": "Dextran",
            "smoothing": {"window_length": 25},
        },
        "declarations": {
            "sample_name": {"value": "Dextran"},
            "polymer_type": {"value": "Polysaccharide"},
        },
        "parameters": {
            "mw": 150000.0,
            "a2": 0.0002,
            "a2_mw": 30.0,
            "k_d": 0.05,
            "calibrated": True,
        },
        "anomalies": [],
        "warnings": [],
    }

    (runs_dir / "bsa_01.provenance.json").write_text(json.dumps(record_bsa), encoding="utf-8")
    (runs_dir / "dextran_02.provenance.json").write_text(json.dumps(record_dextran), encoding="utf-8")
    return runs_dir


def test_registry_scan_and_search(mock_provenance_dir: Path):
    registry = RunRegistry(mock_provenance_dir)
    assert len(registry.summaries) == 2

    # Test search by query
    bsa_results = registry.search(query="bsa")
    assert len(bsa_results) == 1
    assert bsa_results[0].run_id == "bsa_test_01"
    assert bsa_results[0].sample_name == "BSA"
    assert bsa_results[0].mw == 66430.0

    # Test search by sample_name
    dex_results = registry.search(sample_name="Dextran")
    assert len(dex_results) == 1
    assert dex_results[0].run_id == "dextran_test_02"


def test_registry_comparison(mock_provenance_dir: Path):
    registry = RunRegistry(mock_provenance_dir)
    target = {
        "sample_name": "BSA",
        "a2_mw": 34.0,
        "k_d": -0.01,
    }
    comparison = registry.compare(target)
    assert comparison["historical_sample_count"] == 1
    assert "target_a2_mw" in comparison
    assert comparison["target_a2_mw"] == 34.0


def test_tool_surface_historical_methods(mock_provenance_dir: Path):
    from polyrmc.cli import load_config
    state = RunState(run_id="current_run", source_file="data/raw/current.csv")
    config = load_config("examples/run_config.example.json")
    tools = ToolSurface(state=state, config=config)

    past_runs = tools.search_past_runs(query="BSA", runs_dir=mock_provenance_dir)
    assert len(past_runs) == 1
    assert past_runs[0]["run_id"] == "bsa_test_01"

    comparison = tools.compare_current_run(runs_dir=mock_provenance_dir)
    assert "historical_sample_count" in comparison
