"""Literature reading, local context and persistence without network calls."""

from __future__ import annotations

import json
import threading
import time

import pytest

from continuum_spectranalytics.gui.literature import LiteraturePanel
from continuum_spectranalytics.gui.theme import build_fonts


def pump(root, until, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if until():
            return
        time.sleep(.005)
    raise AssertionError("Literature workspace did not reach the expected state")


def paper(title="A relevant paper", url="https://example.org/first", **extra):
    return {"title": title, "url": url, "origin": "web", "domain": "example.org",
            "published": "2026-11-01", "snippet": "Discovery summary, not a full-text review.",
            "relevance": ["Shares the active experiment’s measurement technique."],
            "memory_refs": [{"kind": "note", "id": "n-1", "title": "Earlier buffer observation"}], **extra}


@pytest.fixture
def panel_factory(gui_app):
    created = []

    def make(**callbacks):
        defaults = dict(
            search=lambda _args: json.dumps({"results": [], "memory": []}),
            context=lambda: {"run_id": "run-1", "summary": "Protein light-scattering experiment.",
                             "suggested_query": "protein light scattering", "terms": ["protein"]},
            read=lambda args: json.dumps({"url": args["url"], "text": "Extracted source paragraphs.", "truncated": False}),
            load_saved=lambda: [], save_source=lambda item: dict(item),
        )
        defaults.update(callbacks)
        panel = LiteraturePanel(gui_app, build_fonts(gui_app), **defaults)
        created.append(panel)
        pump(gui_app, lambda: panel._io_active == 0 and not panel._io_pending)
        return panel

    yield make
    for panel in created:
        panel.destroy()
    gui_app.update()


def test_context_refresh_preserves_user_query_and_use_experiment_does_not_search(panel_factory, gui_app):
    context = {"run_id": "first", "summary": "First experiment", "suggested_query": "first safe topic"}
    searches = []
    panel = panel_factory(context=lambda: dict(context), search=lambda args: searches.append(args) or {"results": []})
    panel.query.insert(0, "my editable research question")
    context.update(run_id="next", summary="Second experiment", suggested_query="new safe topic")
    panel.refresh_context()
    pump(gui_app, lambda: panel.experiment.get("run_id") == "next")
    assert panel.query.get() == "my editable research question"
    assert "Second experiment" in panel.context_label.cget("text")
    panel.use_experiment()
    assert panel.query.get() == "new safe topic"
    assert searches == []


def test_relevance_and_memory_are_readable_even_when_internet_results_are_empty(panel_factory):
    panel = panel_factory()
    panel.show_results({"results": [paper()], "memory": []})
    shown = panel.detail.get("1.0", "end")
    assert "Why this may be relevant" in shown
    assert "Earlier buffer observation" in shown
    assert "{'kind'" not in shown
    assert "Abstract / search summary" in shown
    panel.show_results({"results": [], "memory": [{"kind": "note", "title": "Buffer note",
                        "excerpt": "Earlier dilution was unstable.", "author": "A. Researcher",
                        "created_at": "2026-10-30"}], "warnings": ["Internet lookup unavailable"]})
    assert panel.view == "Memory"
    shown = panel.detail.get("1.0", "end")
    assert "Lab memory" in shown and "Earlier dilution was unstable" in shown
    assert "A. Researcher" in shown and "2026-10-30" in shown
    assert "Internet lookup unavailable" in panel.status.cget("text")
    assert panel.read_button.cget("state") == "disabled"


def test_page_read_is_background_work_and_stale_selection_cannot_replace_reader(panel_factory, gui_app):
    gate = threading.Event()
    entered = threading.Event()
    thread_ids = []

    def read(args):
        thread_ids.append(threading.get_ident())
        assert args["max_chars"] == 24000
        entered.set()
        assert gate.wait(5)
        return {"url": args["url"], "text": "Stale full text that belongs to the first paper."}

    panel = panel_factory(read=read)
    panel.show_results({"results": [paper("First"), paper("Second", "https://example.org/second")]})
    panel.read_page()
    ticks = []
    gui_app.after(0, lambda: ticks.append(True))
    try:
        pump(gui_app, lambda: entered.is_set() and bool(ticks))
        assert panel._reading
        panel.listing.selection_set("1")
        panel._select()
        assert "Second" in panel.detail.get("1.0", "end")
        assert thread_ids[0] != threading.get_ident()
    finally:
        gate.set()
    pump(gui_app, lambda: panel._io_active == 0)
    assert "Stale full text" not in panel.detail.get("1.0", "end")
    assert "Second" in panel.detail.get("1.0", "end")


def test_local_paper_result_can_be_read_and_pdf_failure_keeps_original_available(panel_factory, gui_app):
    calls = []

    def read(args):
        calls.append(args)
        return {"url": args["url"], "error": "PDF extraction is unsupported", "text": ""}

    panel = panel_factory(read=read)
    panel.show_results({"results": [paper(origin="memory", kind="paper")]})
    assert panel.read_button.cget("state") == "normal"
    panel.read_page()
    pump(gui_app, lambda: not panel._reading)
    assert len(calls) == 1
    shown = panel.detail.get("1.0", "end")
    assert "Page unavailable" in shown and "Open original" in shown
    assert panel.open_button.cget("state") == "normal"


def test_read_paper_can_be_explicitly_saved_and_reopened_from_metadata_library(panel_factory, gui_app):
    storage = []
    gate = threading.Event()
    save_started = threading.Event()
    save_threads = []

    def save(item):
        save_threads.append(threading.get_ident())
        save_started.set()
        assert gate.wait(5)
        # The production reading library persists metadata, not downloaded
        # article text. Returning that record must still yield a usable source.
        metadata = {key: value for key, value in item.items() if key != "text"}
        storage.append(metadata)
        return metadata

    panel = panel_factory(load_saved=lambda: list(storage), save_source=save)
    panel.show_results({"query": "protein virial references", "results": [paper()]})
    panel.read_page()
    pump(gui_app, lambda: not panel._reading)
    assert "Extracted source paragraphs" in panel.detail.get("1.0", "end")
    panel.save_paper()
    try:
        pump(gui_app, save_started.is_set)
        assert panel._saving and storage == []
        assert save_threads[0] != threading.get_ident()
    finally:
        gate.set()
    pump(gui_app, lambda: not panel._saving)
    assert "text" not in storage[0]
    assert storage[0]["query"] == "protein virial references"
    assert storage[0]["run_id"] == "run-1"
    reopened = panel_factory(load_saved=lambda: list(storage))
    reopened.set_view("Saved")
    assert reopened.current()["title"] == "A relevant paper"
    assert "Abstract / search summary" in reopened.detail.get("1.0", "end")
    assert reopened.read_button.cget("state") == "normal"


def test_context_change_invalidates_pending_search_without_replacing_query(panel_factory, gui_app):
    gate = threading.Event()
    entered = threading.Event()

    def search(_args):
        entered.set()
        assert gate.wait(5)
        return {"results": [paper("Old experiment result")]}

    panel = panel_factory(search=search)
    panel.query.insert(0, "question I want to keep")
    panel.search()
    try:
        assert entered.wait(1)
        panel.context_changed()
        assert panel.query.get() == "question I want to keep"
    finally:
        gate.set()
    pump(gui_app, lambda: panel._workers == 0)
    assert panel.results == []
    assert "Old experiment result" not in panel.detail.get("1.0", "end")


def test_reader_font_focus_and_column_width_preserve_reading_content(panel_factory, monkeypatch):
    panel = panel_factory()
    panel.show_results({"results": [paper()]})
    content = panel.detail.get("1.0", "end")
    assert panel._reader_font.actual("size") == 15
    assert panel.detail.cget("wrap") == "word"
    monkeypatch.setattr(panel.reader_area, "winfo_width", lambda: 4000)
    panel.adjust_font(2)
    assert panel._reader_font.actual("size") == 17
    assert int(panel.reader_column.place_info()["width"]) < 4000
    assert panel.detail.get("1.0", "end") == content
    panel.toggle_reading_focus()
    assert panel._reading_focus
    assert panel.result_frame.winfo_manager() == ""
    assert panel.context_frame.winfo_manager() == ""
    panel.toggle_reading_focus()
    assert panel.result_frame.winfo_manager() == "grid"
    assert panel.current()["title"] == "A relevant paper"


def test_explicit_assistant_review_uses_context_and_renders_markdown_in_reader(panel_factory):
    prompts = []
    panel = panel_factory(on_review=prompts.append)
    panel.show_results({"query": "virial interpretation", "results": [paper()]})
    panel.ask_assistant()
    assert "run-1" in prompts[0] and "https://example.org/first" in prompts[0]
    assert "Do not alter the experiment" in prompts[0]
    panel.show_review("### Evidence\n**A supported observation** with [a source](https://example.org/source).\n- A limitation")
    shown = panel.detail.get("1.0", "end")
    assert "Evidence" in shown and "A supported observation" in shown
    assert "###" not in shown and "**" not in shown
    assert any(tag.startswith("review_link_") for tag in panel.detail.tag_names())
    assert panel._reading_focus and len(panel.results) == 1
    assert panel.save_button.cget("state") == panel.open_button.cget("state") == "disabled"


def test_saved_library_failure_is_actionable_instead_of_silent_empty(panel_factory):
    panel = panel_factory(load_saved=lambda: {"error": "Local library file is unreadable"})
    assert "Saved papers could not be loaded" in panel.status.cget("text")
    assert "unreadable" in panel.status.cget("text")


def test_search_finishing_does_not_interrupt_a_saved_source_chosen_while_waiting(panel_factory, gui_app):
    gate = threading.Event()
    entered = threading.Event()

    def search(_args):
        entered.set()
        assert gate.wait(5)
        return {"results": [paper("Newly found source")]}

    saved = paper("A saved reading choice", "https://example.org/saved", text="Previously saved page text.")
    panel = panel_factory(search=search, load_saved=lambda: [saved])
    panel.query.insert(0, "a new question")
    panel.search()
    try:
        assert entered.wait(1)
        panel.set_view("Saved")
        assert "Previously saved page text" in panel.detail.get("1.0", "end")
    finally:
        gate.set()
    pump(gui_app, lambda: not panel.busy)
    assert panel.view == "Saved"
    assert "Previously saved page text" in panel.detail.get("1.0", "end")
    panel.set_view("Found")
    assert panel.current()["title"] == "Newly found source"
    panel.set_view("Saved")
    panel.show_results({"results": [paper("Explicitly supplied new results")]})
    assert panel.view == "Found"
    assert panel.current()["title"] == "Explicitly supplied new results"


def test_save_finishing_after_selecting_memory_keeps_memory_unsavable(panel_factory, gui_app):
    gate = threading.Event()
    entered = threading.Event()
    saved = []

    def save(item):
        saved.append(dict(item))
        entered.set()
        assert gate.wait(5)
        return dict(item)

    panel = panel_factory(save_source=save)
    panel.show_results({"results": [paper()], "memory": [{
        "kind": "note", "title": "A local observation", "excerpt": "Local lab memory, not a paper.",
        "url": "https://example.org/a-reference-mentioned-in-a-note",
    }]})
    panel.save_paper()
    try:
        assert entered.wait(1)
        panel.set_view("Memory")
        assert panel.save_button.cget("state") == "disabled"
    finally:
        gate.set()
    pump(gui_app, lambda: not panel._saving)
    assert panel.view == "Memory"
    assert panel.save_button.cget("state") == "disabled"
    assert "Local lab memory" in panel.detail.get("1.0", "end")
    panel.save_paper()
    assert len(saved) == 1 and saved[0]["title"] == "A relevant paper"


def test_new_search_cannot_restore_old_results_or_save_under_an_edited_query(panel_factory, gui_app):
    saved = []
    gate = threading.Event()
    entered = threading.Event()

    def search(_args):
        entered.set()
        assert gate.wait(5)
        return {"results": [paper("A later result")]}

    panel = panel_factory(search=search, save_source=lambda item: saved.append(dict(item)) or dict(item))
    panel.show_results({"query": "original public query", "results": [paper("Original result")]})
    panel.query.delete(0, "end")
    panel.query.insert(0, "unsent edited question")
    panel.save_paper()
    pump(gui_app, lambda: not panel._saving)
    assert saved[0]["query"] == "original public query"
    assert saved[0]["run_id"] == "run-1"
    panel.search()
    try:
        assert entered.wait(1)
        panel.set_view("Found")
        assert panel.results == []
        panel.cancel()
        panel.set_view("Found")
        assert panel.results == []
    finally:
        gate.set()
    pump(gui_app, lambda: panel._workers == 0)
    assert panel.results == []


def test_reopening_saved_retries_a_failed_library_load(panel_factory, gui_app):
    replies = iter([{"error": "Library is temporarily unavailable"}, [paper("Recovered saved paper")]])
    panel = panel_factory(load_saved=lambda: next(replies))
    assert panel._saved_load_error
    panel.set_view("Saved")
    pump(gui_app, lambda: panel._io_active == 0)
    assert panel.current()["title"] == "Recovered saved paper"
