"""Search contracts against fake services: no key, model call or live network."""

from __future__ import annotations

import io
import json
import sys
import types
import urllib.error
import urllib.parse

import pytest

from continuum_spectranalytics import research
from continuum_spectranalytics.agent.toolspec import build_registry


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", None)
    monkeypatch.setattr(research, "_arxiv_last_request", float("-inf"))

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected live network request")

    monkeypatch.setattr(research.urllib.request, "urlopen", no_network)


@pytest.fixture
def fake_web(monkeypatch):
    calls = []
    rows = []

    class FakeDDGS:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def text(self, query, **kwargs):
            calls.append((query, kwargs))
            return list(rows)

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    return rows, calls


def arxiv_response(monkeypatch, payload):
    calls = []

    def open_response(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(research.urllib.request, "urlopen", open_response)
    return calls


FEED = b'''<feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2601.12345v1</id>
<title>  Polymer\n scattering </title><summary> A useful\n abstract. </summary>
<published>2026-01-12T10:00:00Z</published>
<author><name>A. Researcher</name></author>
<link rel="alternate" href="http://arxiv.org/abs/2601.12345v1"/>
</entry></feed>'''


def test_web_passes_filters_and_returns_a_shared_source_record(fake_web):
    rows, calls = fake_web
    rows.append({"title": "Scattering reference", "href": "https://example.edu/paper?utm_source=web#section",
                 "body": "A   short\n description."})
    response = research.search_internet("  polymer  scattering ", max_results=3, recency="m", site="EXAMPLE.edu")

    assert calls == [("init", {"timeout": research.TIMEOUT}),
                     ("polymer scattering site:example.edu", {"max_results": 6, "timelimit": "m", "backend": research.WEB_BACKENDS})]
    assert response == {
        "query": "polymer scattering", "source": "web", "site": "example.edu", "recency": "m", "results": [{
            "title": "Scattering reference", "url": "https://example.edu/paper",
            "snippet": "A short description.", "domain": "example.edu", "published": None,
        }], "warning": "Date filtering is supplied by search engines; verify publication dates at the source.",
    }


def test_web_deduplicates_tracking_fragment_scheme_and_query_order(fake_web):
    rows, _ = fake_web
    rows.extend([
        {"title": "First", "href": "https://www.example.org/paper?a=1&b=2&utm_source=search#s"},
        {"title": "Duplicate", "href": "http://example.org/paper/?b=2&a=1&gclid=abc"},
        {"title": "Different paper", "href": "https://example.org/other"},
    ])
    response = research.search_internet("polymer")
    assert [row["title"] for row in response["results"]] == ["First", "Different paper"]


def test_returned_links_preserve_nontracking_query_bytes_and_order(fake_web):
    rows, _ = fake_web
    rows.append({"title": "Signed reference", "href": "https://example.org/ref?value=a%20b&signature=X%2FY%3D&literal=%2B&flag&utm_source=web#s"})
    response = research.search_internet("polymer")
    assert response["results"][0]["url"] == "https://example.org/ref?value=a%20b&signature=X%2FY%3D&literal=%2B&flag"


def test_web_filters_unsafe_links_and_wrong_sites_and_bounds_content(fake_web):
    rows, _ = fake_web
    rows.extend([
        {"title": "Script", "href": "javascript:alert(1)"},
        {"title": "File", "href": "file:///secret"},
        {"title": "Credentials", "href": "https://user:password@example.org/x"},
        {"title": "Wrong site", "href": "https://example.org.evil.test/x"},
        {"title": "x" * 500, "href": "https://journals.example.org/x", "body": "y" * 5000},
        {"title": "Second", "href": "https://example.org/second"},
    ])
    response = research.search_internet("polymer", max_results=1, site="example.org")
    assert len(response["results"]) == 1
    row = response["results"][0]
    assert len(row["title"]) == 240
    assert len(row["snippet"]) == 800
    assert row["domain"] == "journals.example.org"


@pytest.mark.parametrize("kwargs", [
    {"query": ""}, {"query": " "}, {"query": None}, {"query": "a" * 501},
    {"source": "news"}, {"max_results": 0}, {"max_results": 21},
    {"max_results": True}, {"max_results": "8"}, {"max_results": 1.5},
    {"recency": "decade"}, {"site": "https://example.org"},
    {"site": "example.org/path"}, {"site": "example.org OR site:evil.test"},
    {"site": "-bad.example"}, {"site": "example..org"},
])
def test_invalid_inputs_are_rejected_before_search(kwargs, fake_web):
    _, calls = fake_web
    response = research.search_internet(**({"query": "polymer"} | kwargs))
    assert response["error"]
    assert response["results"] == []
    assert calls == []


def test_empty_results_are_distinct_from_failed_search(fake_web):
    response = research.search_internet("very specific nonexistent result")
    assert response["results"] == []
    assert "No matching" in response["warning"]
    assert "error" not in response


def test_missing_optional_dependency_is_actionable():
    response = research.search_internet("polymer")
    assert "pip install -e .[search]" in response["error"]
    assert "arXiv" in response["error"]


@pytest.mark.parametrize("error, message", [
    (TimeoutError("request detail"), "timed out"),
    (urllib.error.URLError("dns details"), "could not connect"),
    (RuntimeError("proxy-secret-value"), "temporarily unavailable"),
])
def test_backend_errors_are_helpful_without_exposing_request_details(monkeypatch, error, message):
    def fail(*args):
        raise error

    monkeypatch.setattr(research, "_web_rows", fail)
    response = research.search_internet("polymer")
    assert message in response["error"]
    assert str(error) not in response["error"]
    assert response["results"] == []


def test_arxiv_uses_https_author_metadata_and_explicit_term_fields(monkeypatch):
    calls = arxiv_response(monkeypatch, FEED)
    response = research.search_internet('polymer "light scattering"', source="arxiv")

    request, timeout = calls[0]
    assert request.full_url.startswith("https://export.arxiv.org/api/query?")
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
    assert params["search_query"] == ['all:"polymer" AND all:"light scattering"']
    assert params["sortBy"] == ["relevance"]
    assert timeout == research.TIMEOUT
    assert response["results"] == [{
        "title": "Polymer scattering", "url": "https://arxiv.org/abs/2601.12345v1",
        "snippet": "A useful abstract.", "domain": "arxiv.org", "published": "2026-01-12",
        "authors": ["A. Researcher"],
    }]
    assert "peer-review" in response["warning"]


def test_arxiv_recency_is_a_submission_date_filter(monkeypatch):
    calls = arxiv_response(monkeypatch, FEED)
    response = research.search_internet("polymer", source="arxiv", recency="w")
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0][0].full_url).query)
    assert "AND submittedDate:[" in params["search_query"][0]
    assert " TO " in params["search_query"][0]
    assert params["sortBy"] == ["submittedDate"]
    assert response["results"]


def test_arxiv_preserves_apostrophes_in_natural_language(monkeypatch):
    calls = arxiv_response(monkeypatch, FEED)
    response = research.search_internet("Reed's scattering method", source="arxiv")
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0][0].full_url).query)
    assert params["search_query"] == ['all:"Reed\'s" AND all:"scattering" AND all:"method"']
    assert response["results"]


def test_arxiv_unclosed_phrase_has_a_helpful_error_without_network():
    response = research.search_internet('polymer "light scattering', source="arxiv")
    assert "Close the double quotation marks" in response["error"]


def test_arxiv_empty_result_preserves_preprint_warning(monkeypatch):
    arxiv_response(monkeypatch, b'<feed xmlns="http://www.w3.org/2005/Atom"/>')
    response = research.search_internet("polymer", source="arxiv")
    assert "peer-review" in response["warning"]
    assert "No matching" in response["warning"]


def test_external_authors_are_bounded_and_malformed_values_do_not_break_results(fake_web):
    rows, _ = fake_web
    rows.extend([
        {"title": "One", "href": "https://example.org/1", "authors": "A. Researcher"},
        {"title": "Two", "href": "https://example.org/2", "authors": 123},
    ])
    response = research.search_internet("polymer")
    assert response["results"][0]["authors"] == ["A. Researcher"]
    assert "authors" not in response["results"][1]


def test_arxiv_respects_request_spacing(monkeypatch):
    arxiv_response(monkeypatch, FEED)
    sleeps = []
    monkeypatch.setattr(research, "_arxiv_last_request", 99.0)
    monkeypatch.setattr(research.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(research.time, "sleep", sleeps.append)
    research.search_internet("polymer", source="arxiv")
    assert sleeps == [2.0]


@pytest.mark.parametrize("payload, message", [
    (b"not xml", "unreadable"),
    (b"<html/>", "unexpected response"),
    (b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <id>http://arxiv.org/api/errors#incorrect_query</id><title>Error</title>
    </entry></feed>''', "rejected the query"),
    (b"x" * (research.MAX_RESPONSE + 1), "oversized"),
])
def test_arxiv_bad_responses_are_errors_instead_of_paper_results(monkeypatch, payload, message):
    arxiv_response(monkeypatch, payload)
    response = research.search_internet("polymer", source="arxiv")
    assert message in response["error"]
    assert response["results"] == []


def test_arxiv_rejects_other_site_filter_without_a_request():
    response = research.search_internet("polymer", source="arxiv", site="example.org")
    assert "Choose Web" in response["error"]


def test_arxiv_http_error_reports_status_without_backend_details(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://export.arxiv.org", 503, "internal details", {}, None)

    monkeypatch.setattr(research.urllib.request, "urlopen", fail)
    response = research.search_internet("polymer", source="arxiv")
    assert "HTTP 503" in response["error"]
    assert "internal details" not in response["error"]


def test_search_registry_is_read_only_and_serializes_the_shared_contract(fake_web):
    rows, _ = fake_web
    rows.append({"title": "Reference", "href": "https://example.org/ref", "body": "Public snippet"})
    tool = build_registry()["search_internet"]
    assert tool.tier == "read"
    assert not tool.mutating
    schema = tool.schema()["parameters"]
    assert schema["required"] == ["query"]
    assert schema["properties"]["source"]["enum"] == ["web", "arxiv"]
    assert "untrusted" in tool.description
    result = json.loads(tool.handler(None, {"query": "polymer"}))
    assert result["results"][0]["url"] == "https://example.org/ref"
