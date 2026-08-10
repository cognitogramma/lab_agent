"""The durable notebook: what the agent concluded, kept between sessions."""

from __future__ import annotations

from pathlib import Path

import pytest

from polyrmc.memory.notes import NoteBook


def test_a_note_survives_the_session(tmp_path: Path):
    NoteBook(tmp_path).add(
        "liraglutide", "Goes unstable above 3 M Gd.", author="tester", tags=["gotcha"]
    )

    reopened = NoteBook(tmp_path)

    assert len(reopened) == 1
    assert reopened.notes[0].subject == "liraglutide"
    assert reopened.notes[0].author == "tester"


def test_a_note_needs_an_author(tmp_path: Path):
    with pytest.raises(ValueError, match="questioned later"):
        NoteBook(tmp_path).add("subject", "body", author="  ")


def test_a_note_needs_a_subject_and_a_body(tmp_path: Path):
    book = NoteBook(tmp_path)
    with pytest.raises(ValueError, match="subject"):
        book.add("", "body", author="tester")
    with pytest.raises(ValueError, match="body"):
        book.add("subject", "   ", author="tester")


def test_search_filters_by_text_run_and_tag(tmp_path: Path):
    book = NoteBook(tmp_path)
    book.add("BSA", "Blank was taken a day late.", author="a", run_id="r1", tags=["blank"])
    book.add("dextran", "Header count needed a second spelling.", author="b", run_id="r2")

    assert [n.subject for n in book.search(query="blank")] == ["BSA"]
    assert [n.subject for n in book.search(run_id="r2")] == ["dextran"]
    assert [n.subject for n in book.search(tag="blank")] == ["BSA"]
    assert len(book.search()) == 2


def test_a_corrupt_notebook_opens_empty_rather_than_failing(tmp_path: Path):
    """A stray comma in a notes file must not stop a session from starting."""
    (tmp_path / "notes.json").write_text("{not json", encoding="utf-8")

    book = NoteBook(tmp_path)

    assert len(book) == 0
    book.add("recovery", "Still usable.", author="tester")
    assert len(NoteBook(tmp_path)) == 1


def test_notes_are_kept_apart_from_run_provenance(tmp_path: Path):
    """A note is an opinion. Filing one inside the audit trail would make the
    audit trail unfalsifiable, so the store is its own file."""
    book = NoteBook(tmp_path)
    book.add("BSA", "Looked fine.", author="tester")

    assert book.path.name == "notes.json"
    assert not list(tmp_path.glob("*.provenance.json"))
