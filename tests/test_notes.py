"""The durable notebook: what the agent concluded, kept between sessions."""

from __future__ import annotations

from pathlib import Path

import pytest

from continuum_spectranalytics.memory.notes import NoteBook


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


def test_research_idea_survives_restart_with_evidence_and_next_step(tmp_path: Path):
    book = NoteBook(tmp_path)
    book.add("BSA observation", "A measured observation.", author="tester")
    book.add_research_idea(
        "Salt-dependent aggregation", "Salt may shift the aggregation threshold.", "Arnold",
        evidence="A question prompted by the cited source; not a measured result.",
        source_urls=["https://example.org/paper", "https://example.org/paper"],
        next_step="Compare a salt concentration series.", tags=["BSA"],
    )

    reopened = NoteBook(tmp_path)
    ideas = reopened.research_ideas()
    assert len(ideas) == 1
    idea = ideas[0]
    assert idea.author == "Arnold"
    assert idea.run_id is None
    assert "unverified hypothesis" in idea.body
    assert "Salt may shift" in idea.body
    assert "not a measured result" in idea.body
    assert idea.body.count("https://example.org/paper") == 1
    assert "Next step: Compare a salt concentration series." in idea.body
    assert reopened.research_ideas("concentration series") == [idea]
    assert reopened.search(query="aggregation") == [idea]
    assert len(reopened.notes) == 2


def test_an_empty_idea_is_not_saved(tmp_path: Path):
    book = NoteBook(tmp_path)
    with pytest.raises(ValueError, match="idea or hypothesis"):
        book.add_research_idea("Title", "   ", "Arnold")
    assert len(NoteBook(tmp_path)) == 0


def test_a_failed_idea_save_does_not_remain_in_memory(tmp_path: Path, monkeypatch):
    book = NoteBook(tmp_path)
    book.add("Existing note", "Keep this.", "Arnold")

    def fail_save():
        raise OSError("disk full")

    monkeypatch.setattr(book, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        book.add_research_idea("Unsaved", "A new hypothesis.", "Arnold")
    assert len(book) == 1
    assert len(NoteBook(tmp_path)) == 1
    assert not book.research_ideas()


@pytest.mark.parametrize("failure", ["partial_write", "replace"])
def test_failed_persistence_preserves_prior_ideas(tmp_path: Path, monkeypatch, failure):
    from continuum_spectranalytics.memory import notes as notes_module

    book = NoteBook(tmp_path)
    book.add_research_idea("Prior idea", "An existing hypothesis.", "Arnold")
    original = book.path.read_bytes()

    with monkeypatch.context() as patch:
        if failure == "partial_write":
            def partial_write(_value, stream, **_kwargs):
                stream.write('[{"subject": "partially written')
                stream.flush()
                raise OSError("interrupted write")

            patch.setattr(notes_module.json, "dump", partial_write)
        else:
            def failed_replace(_source, _target):
                raise OSError("replacement refused")

            patch.setattr(Path, "replace", failed_replace)

        with pytest.raises(OSError):
            book.add_research_idea("Unsaved idea", "A second hypothesis.", "Arnold")

    assert book.path.read_bytes() == original
    assert [idea.subject for idea in NoteBook(tmp_path).research_ideas()] == ["Prior idea"]
    assert [idea.subject for idea in book.research_ideas()] == ["Prior idea"]
    assert not list(tmp_path.glob(".notes.json.*.tmp"))

    book.add_research_idea("Later idea", "Saving works after the failure.", "Arnold")
    assert len(NoteBook(tmp_path).research_ideas()) == 2
