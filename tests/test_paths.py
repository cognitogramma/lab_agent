"""Persistent files and bundled defaults use different roots in the executable."""

import sys
from pathlib import Path

import pytest

from continuum_spectranalytics import APP_NAME, paths


@pytest.fixture
def frozen_layout(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "LAB.md").write_text("Shipped lab context", encoding="utf-8")
    examples = bundle / "examples"
    examples.mkdir()
    example = examples / "run_config.example.json"
    example.write_text("{}", encoding="utf-8")
    local = tmp_path / "local"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.delenv(paths.ROOT_ENV, raising=False)
    monkeypatch.delenv("POLYRMC_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "LAB.md").write_text("Unrelated working-directory file", encoding="utf-8")
    return local / APP_NAME, bundle, example


def test_frozen_first_run_writes_to_user_data_and_reads_bundle(frozen_layout):
    from continuum_spectranalytics.agent.lab import load_lab_context
    from continuum_spectranalytics.agent.toolcall import SPECS
    from continuum_spectranalytics.agent.transcript import SessionLog
    from continuum_spectranalytics.cli import _persist_key
    from continuum_spectranalytics.filesystem import policy_path
    from continuum_spectranalytics.memory.notes import NoteBook

    user_data, bundle, example = frozen_layout
    assert not user_data.exists()
    assert paths.example_config() == example
    assert load_lab_context() == ("Shipped lab context", bundle / "LAB.md")
    assert paths.runs_dir() == user_data / "data" / "runs"
    assert policy_path() == user_data / "filesystem-permissions.json"

    assert _persist_key(SPECS["openai"], "test-value")
    assert (user_data / ".env").read_text(encoding="utf-8").strip() == "OPENAI_API_KEY=test-value"
    book = NoteBook()
    book.add_research_idea("Durable idea", "Test the source comparison.", "Tester")
    assert book.path == user_data / "data" / "memory" / "notes.json"
    assert NoteBook().research_ideas()[0].subject == "Durable idea"

    log = SessionLog()
    try:
        log.write("test record")
        assert log.path is not None
        assert log.path.parent == user_data / "data" / "logs"
        assert "test record" in log.path.read_text(encoding="utf-8")
    finally:
        log.close()

    local_lab = user_data / "config" / "LAB.md"
    local_lab.parent.mkdir()
    local_lab.write_text("Custom lab context", encoding="utf-8")
    assert load_lab_context() == ("Custom lab context", local_lab)
    assert not (bundle / ".env").exists()
    assert not (bundle / "data").exists()


def test_frozen_root_overrides_do_not_move_bundled_resources(frozen_layout, tmp_path, monkeypatch):
    _, _, example = frozen_layout
    legacy = tmp_path / "legacy"
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("POLYRMC_ROOT", str(legacy))
    assert paths.project_root() == legacy
    monkeypatch.setenv(paths.ROOT_ENV, str(workspace))
    assert paths.project_root() == workspace
    assert paths.example_config() == example


def test_frozen_fallback_uses_local_app_data_under_the_user_home(frozen_layout, tmp_path, monkeypatch):
    user_home = tmp_path / "profile"
    monkeypatch.delenv("LOCALAPPDATA")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: user_home))
    assert paths.project_root() == user_home / "AppData" / "Local" / APP_NAME


def test_source_checkout_ignores_an_unrelated_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ROOT_ENV, raising=False)
    monkeypatch.delenv("POLYRMC_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    checkout = Path(__file__).resolve().parents[1]
    assert paths.project_root() == checkout
    assert paths.resource_root() == checkout
    assert paths.example_config().is_file()
