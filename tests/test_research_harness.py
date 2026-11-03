"""Offline integration of model-selected search and reading in the real loop.

The provider below makes its next choice from the preceding tool response. It
has no source URL before search returns, and cannot write its cited answer
until page text is delivered. This exercises tool wiring and stop/budget gates;
it does not measure a live model's research quality or make network requests.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import threading
from urllib.parse import urlsplit

import numpy as np
import pytest

from polyrmc.agent.budget import Budget
from polyrmc.agent.orchestrator import Approval, ApprovalPolicy, Orchestrator
from polyrmc.agent.toolcall import AssistantTurn, ScriptedProvider, ToolCall
from polyrmc.agent.toolspec import ToolContext, build_registry
from polyrmc.memory.notes import NoteBook
from polyrmc.state import Declaration, RunState
from polyrmc.tools import ToolSurface


QUERY = "polymer light scattering second virial coefficient"
PAGE_TEXT = "This reference explains how concentration-dependent scattering is used to estimate a second virial coefficient."


@pytest.fixture
def context(run_config, tmp_path):
    state = RunState(
        run_id="research-harness",
        source_file=str(run_config.source_file),
        time_s=np.array([0.0, 1.0, 2.0]),
        raw_signal=np.array([51.0, 49.0, 46.0]),
        declarations={"c0_g_per_cm3": Declaration(value=0.004, author="researcher")},
    )
    return ToolContext(
        surface=ToolSurface(state=state, config=run_config),
        notes=NoteBook(tmp_path / "notes"),
        runs_dir=tmp_path / "runs",
        author="researcher",
    )


class EvidenceDrivenProvider:
    """A deterministic provider which reads actual preceding tool messages."""

    name = "offline-research-provider"
    model = "offline"

    def __init__(self, before_read=None, initial_url=None, follow_link=False):
        self.before_read = before_read
        self.initial_url = initial_url
        self.follow_link = follow_link
        self.calls = []
        self.discovered_url = None
        self.followed_url = None

    def converse(self, system, messages, tools):
        offered = {tool["function"]["name"] for tool in tools}
        assert {"search_internet", "read_web_page"} <= offered
        self.calls.append((list(messages), offered))
        previous = [result for message in messages if message.role == "tool" for result in message.results]
        if not previous:
            if self.initial_url:
                return AssistantTurn(tool_calls=[ToolCall(
                    "read-user-source", "read_web_page", {"url": self.initial_url, "max_chars": 6000},
                )])
            return AssistantTurn(tool_calls=[ToolCall(
                "find-public-source", "search_internet",
                {"query": QUERY, "source": "web", "max_results": 4},
            )])
        latest = previous[-1]
        assert latest.ok, latest.content
        payload = json.loads(latest.content)
        if latest.name == "search_internet":
            self.discovered_url = payload["results"][0]["url"]
            if self.before_read is not None:
                self.before_read()
            return AssistantTurn(tool_calls=[ToolCall(
                "read-discovered-source", "read_web_page",
                {"url": self.discovered_url, "max_chars": 6000},
            )])
        assert latest.name == "read_web_page"
        assert payload["text"], "The provider cannot answer from a search snippet alone"
        if self.follow_link and self.followed_url is None and payload.get("links"):
            self.followed_url = payload["links"][0]["url"]
            return AssistantTurn(tool_calls=[ToolCall(
                "read-linked-source", "read_web_page", {"url": self.followed_url, "max_chars": 6000},
            )])
        return AssistantTurn(text=f'{payload["text"]} [{payload["title"]}]({payload["url"]})')


def mock_public_sources(monkeypatch, source_url, after_search=None, page_links=None,
                        page_texts=None, first_search_error=None):
    """Patch only external lookup functions, leaving registry/loop intact."""
    search_calls = []
    page_calls = []
    research = importlib.import_module("polyrmc.research")
    reader = importlib.import_module("polyrmc.web_reader")

    def search_internet(**arguments):
        search_calls.append(arguments)
        if after_search is not None:
            after_search()
        if first_search_error and len(search_calls) == 1:
            return {"query": arguments["query"], "source": arguments.get("source", "web"),
                    "results": [], "error": first_search_error, "code": "timeout"}
        return {
            "query": arguments["query"], "source": arguments.get("source", "web"),
            "results": [{"title": "A public scattering reference", "url": source_url,
                         "domain": urlsplit(source_url).hostname,
                         "snippet": "A short discovery snippet, not the full source.",
                         "published": "2026-10-31"}],
        }

    def read_web_page(url, max_chars=12000):
        page_calls.append({"url": url, "max_chars": max_chars})
        text = (page_texts or {}).get(url, PAGE_TEXT)
        return {
            "url": url, "requested_url": url, "title": "A public scattering reference",
            "text": text, "links": (page_links or {}).get(url, []), "content_type": "text/html",
            "retrieved_at": "2026-11-01T12:00:00Z",
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "truncated": False, "untrusted": True,
        }

    monkeypatch.setattr(research, "search_internet", search_internet)
    monkeypatch.setattr(reader, "read_web_page", read_web_page)
    return search_calls, page_calls


def approval_recorder():
    asked = []

    def ask(tool, call):
        asked.append((tool.name, call.arguments))
        return Approval(False, "Research tools should be read-only")

    return asked, ApprovalPolicy(ask=ask)


def assert_scientific_state_unchanged(context, state, config):
    assert context.surface.state.model_dump(mode="json") == state.model_dump(mode="json")
    assert context.surface.config.model_dump(mode="json") == config.model_dump(mode="json")
    for name in ("time_s", "raw_signal", "spliced_signal", "baseline", "corrected_signal",
                 "smoothed_signal", "concentration", "kc_over_i"):
        before = getattr(state, name)
        after = getattr(context.surface.state, name)
        if before is None:
            assert after is None
        else:
            np.testing.assert_array_equal(after, before)
    assert context.surface.state.m0 is state.m0 is None
    assert context.output_csv is None
    assert not list(context.runs_dir.glob("*"))


@pytest.mark.parametrize("source_url", [
    "https://journals.example.org/scattering/reference-a",
    "https://preprints.example.net/2026/polymer-reference-b?version=2",
])
def test_provider_searches_then_reads_discovered_url_and_answers_with_citation(context, monkeypatch, source_url):
    search_calls, page_calls = mock_public_sources(monkeypatch, source_url)
    asked, approvals = approval_recorder()
    provider = EvidenceDrivenProvider()
    registry = build_registry()
    state = context.surface.state.model_copy(deep=True)
    config = context.surface.config.model_copy(deep=True)
    agent = Orchestrator(context=context, provider=provider, registry=registry, approvals=approvals)

    events = agent.ask("Find a reference, read it, and cite the explanation.")

    assert search_calls == [{"query": QUERY, "source": "web", "max_results": 4}]
    assert page_calls == [{"url": source_url, "max_chars": 6000}]
    assert provider.discovered_url == source_url
    assert len(provider.calls) == 3
    assert [call.name for call, _result in agent.trace] == ["search_internet", "read_web_page"]
    assert all(result.ok for _call, result in agent.trace)
    assert [event.tool for event in events if event.kind == "result"] == ["search_internet", "read_web_page"]
    answers = [event.text for event in events if event.kind == "text"]
    assert len(answers) == 1 and PAGE_TEXT in answers[0] and f"]({source_url})" in answers[0]
    assert registry["search_internet"].tier == registry["read_web_page"].tier == "read"
    assert asked == [] and agent.committed == []
    assert_scientific_state_unchanged(context, state, config)


def test_request_budget_stops_after_search_before_another_research_round(context, monkeypatch):
    source_url = "https://example.org/a-discovered-source"
    search_calls, page_calls = mock_public_sources(monkeypatch, source_url)
    asked, approvals = approval_recorder()
    provider = EvidenceDrivenProvider()
    agent = Orchestrator(context=context, provider=provider, approvals=approvals,
                         budget=Budget(max_requests=1))

    events = agent.ask("Research this until the request budget is reached.")

    assert len(search_calls) == 1 and page_calls == []
    assert len(provider.calls) == 1
    assert agent.budget.spent.requests == 1
    assert any(event.kind == "capped" and "request budget" in event.text for event in events)
    assert [call.name for call, _result in agent.trace] == ["search_internet"]
    assert asked == [] and agent.committed == []


def test_stop_between_provider_and_web_tool_prevents_fetch_and_pairs_result(context, monkeypatch):
    source_url = "https://example.net/another-discovered-source"
    search_calls, page_calls = mock_public_sources(monkeypatch, source_url)
    stopped = threading.Event()
    provider = EvidenceDrivenProvider(before_read=stopped.set)
    asked, approvals = approval_recorder()
    agent = Orchestrator(context=context, provider=provider, approvals=approvals,
                         should_stop=stopped.is_set)

    events = agent.ask("Find and read a source, unless I stop the session.")

    assert len(search_calls) == 1 and page_calls == []
    assert provider.discovered_url == source_url and len(provider.calls) == 2
    assert [call.name for call, _result in agent.trace] == ["search_internet"]
    result = agent.messages[-1].results[0]
    assert result.name == "read_web_page" and result.call_id == "read-discovered-source"
    assert not result.ok and "stopped the session" in result.content
    assert any(event.kind == "stopped" for event in events)
    assert not any(event.kind in {"failed", "capped"} for event in events)
    assert asked == [] and agent.committed == []


def test_provider_can_read_a_user_supplied_url_without_searching(context, monkeypatch):
    source_url = "https://example.org/reference-supplied-by-researcher"
    search_calls, page_calls = mock_public_sources(monkeypatch, source_url)
    asked, approvals = approval_recorder()
    provider = EvidenceDrivenProvider(initial_url=source_url)
    agent = Orchestrator(context=context, provider=provider, approvals=approvals)

    events = agent.ask(f"Read {source_url} and cite the explanation.")

    assert search_calls == []
    assert page_calls == [{"url": source_url, "max_chars": 6000}]
    assert [call.name for call, _result in agent.trace] == ["read_web_page"]
    assert len(provider.calls) == 2
    assert any(event.kind == "text" and PAGE_TEXT in event.text and source_url in event.text for event in events)
    assert asked == [] and agent.committed == []


def test_provider_follows_a_link_from_read_page_before_answering(context, monkeypatch):
    source_url = "https://example.org/discovered-article"
    linked_url = "https://supplement.example.net/methods/measurement"
    linked_text = "The linked methods section supplies the additional evidence requested."
    search_calls, page_calls = mock_public_sources(
        monkeypatch, source_url,
        page_links={source_url: [{"title": "Measurement methods", "url": linked_url}]},
        page_texts={linked_url: linked_text},
    )
    provider = EvidenceDrivenProvider(follow_link=True)
    asked, approvals = approval_recorder()
    agent = Orchestrator(context=context, provider=provider, approvals=approvals)

    events = agent.ask("Find a source and follow its methods link before answering.")

    assert len(search_calls) == 1
    assert page_calls == [{"url": source_url, "max_chars": 6000}, {"url": linked_url, "max_chars": 6000}]
    assert provider.followed_url == linked_url and len(provider.calls) == 4
    assert [call.name for call, _result in agent.trace] == ["search_internet", "read_web_page", "read_web_page"]
    answer = next(event.text for event in events if event.kind == "text")
    assert linked_text in answer and f"]({linked_url})" in answer
    assert asked == [] and agent.committed == []


def test_web_limit_applies_inside_a_batch_and_every_call_receives_a_result(context, monkeypatch):
    source_url = "https://example.org/first-source"
    search_calls, page_calls = mock_public_sources(monkeypatch, source_url)
    batch = AssistantTurn(tool_calls=[
        ToolCall("find-0", "search_internet", {"query": QUERY}),
        ToolCall("read-1", "read_web_page", {"url": source_url}),
        ToolCall("find-2", "search_internet", {"query": "a second public topic"}),
        ToolCall("read-3", "read_web_page", {"url": "https://example.net/third"}),
        ToolCall("read-4", "read_web_page", {"url": "https://example.net/fourth"}),
    ])
    provider = ScriptedProvider([batch, "The internet limit was reached; I will use the collected sources."])
    asked, approvals = approval_recorder()
    agent = Orchestrator(context=context, provider=provider, approvals=approvals, max_web_calls=2)

    events = agent.ask("Consult these public references.")

    assert len(search_calls) == len(page_calls) == 1
    results = next(message.results for message in agent.messages if message.role == "tool")
    assert [result.call_id for result in results] == [call.id for call in batch.tool_calls]
    assert [result.ok for result in results] == [True, True, False, False, False]
    assert all(json.loads(result.content)["code"] == "web_budget_exhausted" for result in results[2:])
    assert len(agent.trace) == 5
    assert len([event for event in events if event.kind == "error"]) == 3
    assert asked == [] and agent.committed == []


def test_web_limit_resets_on_the_next_researcher_question(context, monkeypatch):
    source_url = "https://example.org/first-question"
    second_url = "https://example.net/next-question"
    _search_calls, page_calls = mock_public_sources(monkeypatch, source_url)
    provider = ScriptedProvider([
        AssistantTurn(tool_calls=[
            ToolCall("first-allowed", "read_web_page", {"url": source_url}),
            ToolCall("first-over-limit", "read_web_page", {"url": second_url}),
        ]),
        "First question finished.",
        AssistantTurn(tool_calls=[ToolCall("next-allowed", "read_web_page", {"url": second_url})]),
        "Second question finished.",
    ])
    agent = Orchestrator(context=context, provider=provider, max_web_calls=1)

    agent.ask("Read the first reference.")
    second_events = agent.ask("Now read the other reference.")

    assert [call["url"] for call in page_calls] == [source_url, second_url]
    assert [result.ok for _call, result in agent.trace] == [True, False, True]
    assert any(event.kind == "result" and event.tool == "read_web_page" for event in second_events)
    assert not any(event.kind == "error" for event in second_events)


def test_zero_web_budget_disables_both_search_and_page_requests(context, monkeypatch):
    source_url = "https://example.org/reference"
    search_calls, page_calls = mock_public_sources(monkeypatch, source_url)
    provider = ScriptedProvider([
        AssistantTurn(tool_calls=[
            ToolCall("blocked-search", "search_internet", {"query": QUERY}),
            ToolCall("blocked-page", "read_web_page", {"url": source_url}),
        ]),
        "Internet access is disabled for this question.",
    ])
    agent = Orchestrator(context=context, provider=provider, max_web_calls=0)

    events = agent.ask("Search for a public reference.")

    assert search_calls == page_calls == []
    assert [result.ok for _call, result in agent.trace] == [False, False]
    assert len([event for event in events if event.kind == "error"]) == 2


def test_tool_failure_reaches_provider_and_it_recovers_with_a_revised_search(context, monkeypatch):
    source_url = "https://example.org/recovered-reference"
    search_calls, page_calls = mock_public_sources(monkeypatch, source_url, first_search_error="Search timed out")

    class RecoveringProvider(EvidenceDrivenProvider):
        observed_failure = None

        def converse(self, system, messages, tools):
            previous = [result for message in messages if message.role == "tool" for result in message.results]
            if previous and not previous[-1].ok:
                self.observed_failure = json.loads(previous[-1].content)
                assert self.observed_failure["error"] == "Search timed out"
                offered = {tool["function"]["name"] for tool in tools}
                assert {"search_internet", "read_web_page"} <= offered
                self.calls.append((list(messages), offered))
                return AssistantTurn(tool_calls=[ToolCall(
                    "revised-search", "search_internet", {"query": "polymer scattering review"},
                )])
            return super().converse(system, messages, tools)

    provider = RecoveringProvider()
    asked, approvals = approval_recorder()
    agent = Orchestrator(context=context, provider=provider, approvals=approvals)

    events = agent.ask("Find a reference, recover from a failed lookup, and cite the source you read.")

    assert [call["query"] for call in search_calls] == [QUERY, "polymer scattering review"]
    assert page_calls == [{"url": source_url, "max_chars": 6000}]
    assert provider.observed_failure["code"] == "timeout"
    assert [result.ok for _call, result in agent.trace] == [False, True, True]
    assert any(event.kind == "error" and event.tool == "search_internet" for event in events)
    assert any(event.kind == "text" and f"]({source_url})" in event.text for event in events)
    assert asked == [] and agent.committed == []


def test_literature_review_result_drives_source_reading_in_the_real_agent(context, monkeypatch):
    """Combined discovery reaches the model before it chooses what to read."""
    from polyrmc import literature

    source_url = "https://example.org/experiment-related-paper"
    _search_calls, page_calls = mock_public_sources(monkeypatch, source_url)
    reviews = []

    def review(ctx, **arguments):
        assert ctx is context
        reviews.append(arguments)
        return {
            "query": QUERY,
            "experiment": {"summary": "A dilution scattering experiment"},
            "results": [{"title": "Related reference", "url": source_url,
                         "relevance": ["Topic match: scattering"], "origin": "web"}],
            "memory": [{"kind": "note", "title": "Earlier dilution observation",
                        "excerpt": "Check concentration-dependent scattering."}],
        }

    monkeypatch.setattr(literature, "review_literature", review)

    class ReviewProvider(EvidenceDrivenProvider):
        def converse(self, system, messages, tools):
            previous = [result for message in messages if message.role == "tool" for result in message.results]
            if not previous:
                assert "review_literature" in {tool["function"]["name"] for tool in tools}
                return AssistantTurn(tool_calls=[ToolCall("review", "review_literature", {})])
            if previous[-1].name == "review_literature":
                payload = json.loads(previous[-1].content)
                assert payload["memory"][0]["title"] == "Earlier dilution observation"
                self.discovered_url = payload["results"][0]["url"]
                return AssistantTurn(tool_calls=[ToolCall("read", "read_web_page", {"url": self.discovered_url})])
            return super().converse(system, messages, tools)

    asked, approvals = approval_recorder()
    state, config = context.surface.state.model_copy(deep=True), context.surface.config.model_copy(deep=True)
    agent = Orchestrator(context=context, provider=ReviewProvider(), approvals=approvals, max_web_calls=2)
    events = agent.ask("Review sources related to this experiment and our earlier observations.")

    assert reviews == [{}]
    assert page_calls == [{"url": source_url, "max_chars": 12000}]
    assert any(event.kind == "text" and f"]({source_url})" in event.text for event in events)
    assert [result.ok for _call, result in agent.trace] == [True, True]
    assert asked == [] and agent.committed == []
    assert_scientific_state_unchanged(context, state, config)


@pytest.mark.parametrize("allowance", [0, 1])
def test_literature_discovery_shares_the_internet_budget(context, monkeypatch, allowance):
    from polyrmc import literature

    called = []

    def review(ctx, **arguments):
        called.append(arguments)
        return {"results": [], "memory": [{"kind": "note", "title": "Local observation"}]}

    monkeypatch.setattr(literature, "review_literature", review)
    provider = ScriptedProvider([
        AssistantTurn(tool_calls=[ToolCall("review-1", "review_literature", {}),
                                  ToolCall("review-2", "review_literature", {})]),
        "Use the evidence available within the budget.",
    ])
    agent = Orchestrator(context=context, provider=provider, max_web_calls=allowance)
    agent.ask("Find related papers.")
    assert len(called) == allowance
    assert [result.ok for _call, result in agent.trace] == ([False, False] if allowance == 0 else [True, False])
    assert json.loads(agent.trace[-1][1].content)["code"] == "web_budget_exhausted"
