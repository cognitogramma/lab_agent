"""Offline tests for the agent's tools (no API key or network required)."""

import pytest

from src import tools
from src.tools import (
    _resolve_data_path,
    list_data_files,
    read_data_file,
    read_notes,
    save_note,
    summarize_csv,
)


@pytest.fixture(autouse=True)
def sandbox_data_dir(tmp_path, monkeypatch):
    """Point DATA_DIR/NOTES_DIR at a temp directory for every test."""
    data_dir = tmp_path / "data"
    notes_dir = data_dir / "notes"
    data_dir.mkdir()
    monkeypatch.setattr(tools, "DATA_DIR", data_dir)
    monkeypatch.setattr(tools, "NOTES_DIR", notes_dir)
    return data_dir


def test_resolve_rejects_path_escape():
    with pytest.raises(ValueError):
        _resolve_data_path("../secrets.txt")


def test_list_data_files_empty():
    assert "empty" in list_data_files.invoke({})


def test_read_data_file(sandbox_data_dir):
    (sandbox_data_dir / "notes.txt").write_text("hello lab", encoding="utf-8")
    assert read_data_file.invoke({"filename": "notes.txt"}) == "hello lab"


def test_read_data_file_missing():
    assert "not found" in read_data_file.invoke({"filename": "nope.txt"}).lower()


def test_read_data_file_rejects_binary_suffix(sandbox_data_dir):
    (sandbox_data_dir / "sim.bin").write_bytes(b"\x00\x01")
    assert "Cannot read" in read_data_file.invoke({"filename": "sim.bin"})


def test_read_data_file_truncates(sandbox_data_dir):
    (sandbox_data_dir / "big.txt").write_text("x" * 10_000, encoding="utf-8")
    out = read_data_file.invoke({"filename": "big.txt", "max_chars": 100})
    assert "truncated" in out


def test_summarize_csv(sandbox_data_dir):
    (sandbox_data_dir / "m.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    out = summarize_csv.invoke({"filename": "m.csv"})
    assert "2 rows x 2 columns" in out
    assert "mean" in out


def test_notes_roundtrip():
    assert "No notes" in read_notes.invoke({})
    result = save_note.invoke({"title": "Test Finding!", "content": "It converged."})
    assert "Saved note" in result
    notes = read_notes.invoke({})
    assert "Test Finding!" in notes
    assert "It converged." in notes
