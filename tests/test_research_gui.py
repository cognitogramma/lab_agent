"""Research UI latency, cancellation and source handling without network calls."""

from __future__ import annotations

import json
import threading
import time

import pytest

from continuum_spectranalytics.gui.research import ResearchPanel, safe_web_url


@pytest.fixture(scope="module")
def tk_root(gui_app):
    return gui_app


@pytest.fixture
def panel_factory(tk_root):
    from continuum_spectranalytics.gui.theme import apply

    panels = []

    def create(callback):
        panel = ResearchPanel(tk_root, apply(tk_root), callback)
        panels.append(panel)
        return panel

    yield create
    for panel in panels:
        panel.destroy()
    tk_root.update()


def pump(root, until, timeout=4.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if until():
            return
        time.sleep(0.005)
    raise AssertionError("Research panel did not reach the expected state")


def response(title="Source", url="https://example.org/paper"):
    return json.dumps({"results": [{"title": title, "url": url,
                                   "domain": "example.org", "published": "2026-10-30",
                                   "snippet": "A useful source summary.", "authors": ["A. Researcher"]}]})


def test_search_does_not_block_tk_and_passes_filters(panel_factory, tk_root):
    gate = threading.Event()
    entered = threading.Event()
    arguments = []

    def backend(args):
        arguments.append((args, threading.current_thread().ident))
        entered.set()
        assert gate.wait(4)
        return response()

    panel = panel_factory(backend)
    panel.query.insert(0, "light scattering")
    panel.domain.insert(0, "example.org")
    panel.period.set("Past month")
    panel.search()
    ticks = []
    tk_root.after(0, lambda: ticks.append(True))
    try:
        pump(tk_root, lambda: entered.is_set() and bool(ticks))
        assert panel.busy
        assert not panel.results
        assert arguments[0][0] == {"query": "light scattering", "source": "web", "max_results": 8,
                                   "recency": "m", "site": "example.org"}
        assert arguments[0][1] != threading.current_thread().ident
    finally:
        gate.set()
    pump(tk_root, lambda: not panel.busy)
    assert panel.results[0]["title"] == "Source"


def test_late_cancelled_request_cannot_replace_new_results(panel_factory, tk_root):
    first_gate = threading.Event()
    first_entered = threading.Event()

    def backend(args):
        if args["query"] == "old":
            first_entered.set()
            assert first_gate.wait(4)
            return response("Stale result")
        return response("Current result")

    panel = panel_factory(backend)
    panel.query.insert(0, "old")
    panel.search()
    assert first_entered.wait(1)
    panel.cancel()
    assert "ignored" in panel.status.cget("text")
    panel.query.delete(0, "end")
    panel.query.insert(0, "new")
    panel.search()
    try:
        pump(tk_root, lambda: not panel.busy)
        assert panel.results[0]["title"] == "Current result"
    finally:
        first_gate.set()
    pump(tk_root, lambda: panel._workers == 0)
    assert panel.results[0]["title"] == "Current result"


def test_repeated_queries_bound_workers_and_keep_only_latest_pending_query(panel_factory, tk_root):
    gate = threading.Event()
    first_two_started = threading.Event()
    lock = threading.Lock()
    captured = []

    def backend(args):
        with lock:
            captured.append(args["query"])
            if len(captured) == 2:
                first_two_started.set()
        if args["query"] in {"first", "second"}:
            assert gate.wait(4)
        return response(args["query"])

    panel = panel_factory(backend)
    for query in ("first", "second", "discard this", "latest"):
        panel.query.delete(0, "end")
        panel.query.insert(0, query)
        panel.search()
    try:
        assert first_two_started.wait(1)
        assert panel._workers == 2
        assert panel._pending[1]["query"] == "latest"
    finally:
        gate.set()
    pump(tk_root, lambda: panel._workers == 0 and not panel.busy)
    assert set(captured) == {"first", "second", "latest"}
    assert panel.results[0]["title"] == "latest"


@pytest.mark.parametrize("payload", [json.dumps({"error": "Service timed out"}), "not JSON"])
def test_errors_replace_stale_sources_and_leave_search_available(panel_factory, tk_root, payload):
    replies = iter([response(), payload])
    panel = panel_factory(lambda _args: next(replies))
    panel.query.insert(0, "a topic")
    panel.search()
    pump(tk_root, lambda: not panel.busy)
    assert panel.results
    panel.search()
    assert panel.results == []
    pump(tk_root, lambda: not panel.busy)
    assert "could not complete" in panel.status.cget("text")
    assert panel.search_button.cget("text") == "Search"
    assert str(panel.open_button.cget("state")) == "disabled"


def test_result_selection_opens_original_and_copies_citation(panel_factory, tk_root, monkeypatch):
    opened = []
    monkeypatch.setattr("continuum_spectranalytics.gui.research.open_web_url", lambda url: opened.append(url) or True)
    panel = panel_factory(lambda _args: response())
    panel.query.insert(0, "a paper")
    panel.source.set("arXiv papers")
    panel._source_changed()
    panel.search()
    pump(tk_root, lambda: not panel.busy)
    assert "A useful source summary" in panel.detail.get("1.0", "end")
    panel.open_source()
    assert opened == ["https://example.org/paper"]
    panel.copy_citation()
    citation = panel.clipboard_get()
    assert "A. Researcher" in citation and "2026-10-30" in citation
    assert "https://example.org/paper" in citation
    assert str(panel.domain.cget("state")) == "disabled"


def test_agent_results_sync_query_and_source_without_another_request(panel_factory):
    def unexpected(_args):
        raise AssertionError("Displaying completed results must not issue a new request")

    panel = panel_factory(unexpected)
    panel.period.set("Past day")
    panel.domain.insert(0, "previous.example.org")
    payload = json.loads(response("Agent source"))
    payload.update(query="second virial coefficient", source="arxiv")
    assert panel.show_results(payload)
    assert panel.query.get() == "second virial coefficient"
    assert panel.source.get() == "arXiv papers"
    assert str(panel.domain.cget("state")) == "disabled"
    assert str(panel.period_choice.cget("state")) == "readonly"
    assert panel.period.get() == "Any time"
    assert panel.domain.get() == ""
    assert panel.results[0]["title"] == "Agent source"


def test_agent_results_display_their_returned_filters(panel_factory):
    panel = panel_factory(lambda _args: response())
    payload = json.loads(response())
    payload.update(query="protein scattering", source="web", recency="m", site="example.org")
    assert panel.show_results(payload)
    assert panel.period.get() == "Past month"
    assert panel.domain.get() == "example.org"


def test_wide_and_narrow_layouts_preserve_selected_source_and_preview(panel_factory):
    panel = panel_factory(lambda _args: response())
    panel.show_results(json.loads(response("A selected paper")))
    listing, detail = panel.listing, panel.detail
    preview = detail.get("1.0", "end")
    panel._set_wide_layout(True)
    assert panel._listing_frame.grid_info()["row"] == panel._preview_frame.grid_info()["row"]
    assert panel._preview_frame.grid_info()["column"] > panel._listing_frame.grid_info()["column"]
    assert panel.current()["title"] == "A selected paper"
    assert panel.detail.get("1.0", "end") == preview
    panel._set_wide_layout(False)
    assert panel._preview_frame.grid_info()["row"] > panel._listing_frame.grid_info()["row"]
    assert panel.listing is listing and panel.detail is detail
    assert panel.current()["title"] == "A selected paper"
    assert panel.detail.get("1.0", "end") == preview


def test_arxiv_date_filter_is_sent_and_domain_filter_is_omitted(panel_factory, tk_root):
    captured = []

    def backend(args):
        captured.append(args)
        return response()

    panel = panel_factory(backend)
    panel.query.insert(0, "polymer scattering")
    panel.domain.insert(0, "example.org")
    panel.source.set("arXiv papers")
    panel.period.set("Past year")
    panel._source_changed()
    panel.search()
    assert not panel.show_results(json.loads(response("Unrelated agent result")))
    pump(tk_root, lambda: not panel.busy)
    assert captured == [{"query": "polymer scattering", "source": "arxiv", "max_results": 8, "recency": "y"}]


@pytest.mark.parametrize("url", ["file:///C:/private.txt", "javascript:alert(1)", "https://user:pass@example.org",
                                 "https://", "https://example.org\ninvalid", None])
def test_only_public_web_link_schemes_can_open(url):
    assert safe_web_url(url) is None


def test_closing_panel_discards_queued_response(panel_factory, tk_root):
    gate = threading.Event()
    entered = threading.Event()

    def backend(_args):
        entered.set()
        assert gate.wait(4)
        return response()

    panel = panel_factory(backend)
    panel.query.insert(0, "a topic")
    panel.search()
    assert entered.wait(1)
    panel.close()
    gate.set()
    pump(tk_root, lambda: not panel._responses.empty())
    assert panel._closed and panel._poll_id is None
    assert panel.results == []
