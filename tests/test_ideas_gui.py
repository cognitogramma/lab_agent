"""Research idea capture persists offline and keeps drafts when saving fails."""

from __future__ import annotations

from continuum_spectranalytics.gui.ideas import IdeaComposer
from continuum_spectranalytics.memory.notes import NoteBook


def test_manual_idea_save_persists_and_refreshes_the_notebook(gui_app, tmp_path, monkeypatch):
    notebook = NoteBook(tmp_path / "ideas")
    monkeypatch.setattr(gui_app.session, "notes", notebook)
    monkeypatch.setattr(gui_app.session.context, "notes", notebook)
    monkeypatch.setattr(gui_app.session.surface.state, "source_file", "none")
    before = gui_app.budget.spent.requests
    panel = gui_app.notes_panel
    panel.new_button.invoke()
    composer = panel.composer
    try:
        composer.idea_title.insert(0, "Compare buffer conditions")
        composer.idea_body.insert("1.0", "A salt sweep may separate screening from aggregation.")
        composer.sources.insert("1.0", "https://example.org/source\nhttps://example.org/source")
        composer.next_step.insert(0, "Find a published salt series")
        composer.save_button.invoke()
        restored = NoteBook(notebook.directory).research_ideas()
        assert len(restored) == 1
        note = restored[0]
        assert note.subject == "Compare buffer conditions"
        assert note.author == gui_app.session.context.author
        assert note.run_id is None
        assert "screening from aggregation" in note.body
        assert note.body.count("https://example.org/source") == 1
        assert "Next step: Find a published salt series" in note.body
        assert "unverified hypothesis" in note.body
        assert "Compare buffer conditions" in panel.out.get("1.0", "end")
        assert gui_app.state_panel.empty.library_counts["notes"] == 1
        assert gui_app.budget.spent.requests == before
        assert not composer.winfo_exists()
    finally:
        if composer.winfo_exists():
            composer.destroy()


def test_empty_idea_is_refused_and_draft_survives_closing(gui_app):
    called = []
    composer = IdeaComposer(gui_app, gui_app.fonts, called.append, lambda: None)
    try:
        composer.idea_title.insert(0, "A question to revisit")
        composer.save_button.invoke()
        assert not called
        assert "Add a title and an idea" in composer.status.cget("text")
        composer.withdraw()
        composer.deiconify()
        assert composer.idea_title.get() == "A question to revisit"
    finally:
        composer.destroy()


def test_storage_failure_reports_the_error_and_keeps_the_idea_draft(gui_app):
    def cannot_save(_arguments):
        raise OSError("disk unavailable")

    composer = IdeaComposer(gui_app, gui_app.fonts, cannot_save, lambda: None)
    try:
        composer.idea_title.insert(0, "Important idea")
        composer.idea_body.insert("1.0", "A hypothesis worth retaining")
        composer.save_button.invoke()
        assert "disk unavailable" in composer.status.cget("text")
        assert composer.idea_title.get() == "Important idea"
        assert composer.idea_body.get("1.0", "end-1c") == "A hypothesis worth retaining"
        assert composer.winfo_exists()
    finally:
        composer.destroy()
