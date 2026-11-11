"""A reading list preserves source metadata without network access or page bodies."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import socket

import pytest

from continuum_spectranalytics import literature_store as store
from continuum_spectranalytics.literature_store import ReadingLibrary, ReadingLibraryError


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Saving a reading list must not access the network")
    monkeypatch.setattr(socket, "getaddrinfo", unexpected)


def paper(**fields):
    return {"title": "Polymer scattering reference", "url": "https://example.org/paper",
            "authors": ["A. Researcher"], "published": "2026-10-30", "snippet": "A useful summary.",
            "origin": "arxiv", "relevance": ["Matches light scattering"],
            "memory_refs": [{"kind": "run", "id": "run-23", "title": "BSA dilution"}], **fields}


def test_listing_a_missing_library_does_not_write(tmp_path):
    directory = tmp_path / "missing" / "literature"
    assert ReadingLibrary(directory).list() == []
    assert not directory.exists()


def test_saved_sources_survive_new_instances_and_keep_only_metadata(tmp_path):
    directory = tmp_path / "literature"
    saved = ReadingLibrary(directory).save(paper(title="Scattering of β polymers", domain="spoofed.example",
                                                query="polymer scattering", run_id="run-23", annotation="Read the methods.",
                                                text="Full web body", html="<script>do_something()</script>",
                                                content={"raw": "private"}, saved_at="not trusted"))
    assert ReadingLibrary(directory).list() == [saved]
    assert saved["domain"] == "example.org"
    assert saved["id"].startswith("paper-")
    assert datetime.fromisoformat(saved["saved_at"]).tzinfo is not None
    assert saved["memory_refs"] == [{"kind": "run", "id": "run-23", "title": "BSA dilution"}]
    assert not {"text", "html", "content"} & saved.keys()
    assert "Full web body" not in (directory / "reading-list.json").read_text(encoding="utf-8")


def test_duplicate_url_updates_metadata_preserves_saved_time_and_annotation(tmp_path):
    first = ReadingLibrary(tmp_path).save(paper(url="https://WWW.example.org/paper?b=2&a=1&utm_source=email#one",
                                               annotation="My original reading note."))
    updated = ReadingLibrary(tmp_path).save({"title": "Revised title",
                                           "url": "http://example.org/paper?a=1&b=2&fbclid=tracking#two"})
    assert updated["id"] == first["id"]
    assert updated["saved_at"] == first["saved_at"]
    assert updated["annotation"] == first["annotation"]
    assert updated["authors"] == first["authors"]
    assert updated["title"] == "Revised title"
    assert updated["url"] == "http://example.org/paper?a=1&b=2"
    assert ReadingLibrary(tmp_path).list() == [updated]


def test_nontracking_query_bytes_are_preserved(tmp_path):
    url = "https://example.org/paper?name=a%20b&signature=A%2fb%2Bc&utm_medium=mail"
    saved = ReadingLibrary(tmp_path).save(paper(url=url))
    assert saved["url"] == "https://example.org/paper?name=a%20b&signature=A%2fb%2Bc"


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "file:///C:/private.txt", "data:text/html,<script>x</script>",
    "https://user:password@example.org", "https://@example.org", "https://example.org:8443/paper",
    "http://localhost/paper", "http://instrument.local/paper", "http://10.0.0.1/paper",
    "http://127.0.0.1/paper", "http://169.254.169.254/", "http://2130706433/", "http://127.1/",
    "http://[::1]/paper", "http://[::ffff:127.0.0.1]/paper", "https://example.org\\private",
    "https://example.org/\nprivate", "https://example.org:bad/", "https://exa mple.org/",
])
def test_invalid_or_local_urls_are_not_saved(tmp_path, url):
    directory = tmp_path / "literature"
    with pytest.raises(ReadingLibraryError, match="public|web ports"):
        ReadingLibrary(directory).save(paper(url=url))
    assert not directory.exists()


def test_public_numeric_address_is_metadata_only_without_dns(tmp_path):
    saved = ReadingLibrary(tmp_path).save(paper(url="https://8.8.8.8/reference"))
    assert saved["domain"] == "8.8.8.8"


@pytest.mark.parametrize("payload", [b"{not JSON", b"[]", b'{"version":2,"papers":[]}',
                                    b'{"version":1,"papers":[{"title":"broken"}]}'])
def test_malformed_library_is_reported_and_never_overwritten(tmp_path, payload):
    path = tmp_path / "reading-list.json"
    path.write_bytes(payload)
    library = ReadingLibrary(tmp_path)
    with pytest.raises(ReadingLibraryError, match="malformed|unsupported"):
        library.list()
    with pytest.raises(ReadingLibraryError, match="No data was changed"):
        library.save(paper())
    assert path.read_bytes() == payload


def test_oversized_existing_file_is_not_overwritten(tmp_path):
    path = tmp_path / "reading-list.json"
    payload = b" " * (store.MAX_BYTES + 1)
    path.write_bytes(payload)
    with pytest.raises(ReadingLibraryError, match="limit"):
        ReadingLibrary(tmp_path).save(paper())
    assert path.read_bytes() == payload


def test_long_snippets_are_bounded_and_raw_body_is_dropped(tmp_path):
    saved = ReadingLibrary(tmp_path).save(paper(snippet="s" * 10000, body="private raw body" * 10000))
    assert len(saved["snippet"]) == store.MAX_SNIPPET
    assert saved["snippet"].endswith("…")
    assert "body" not in saved


@pytest.mark.parametrize("fields", [
    {"title": "t" * (store.MAX_TITLE + 1)}, {"annotation": "a" * 2001},
    {"authors": ["name"] * (store.MAX_AUTHORS + 1)},
    {"relevance": ["reason"] * (store.MAX_RELEVANCE + 1)},
    {"memory_refs": [{"id": "r"}] * (store.MAX_MEMORY_REFS + 1)},
    {"memory_refs": [{"kind": {"raw": "object"}}]},
    {"memory_refs": [{"kind": "run"}]},
])
def test_oversized_or_invalid_metadata_fails_without_writing(tmp_path, fields):
    with pytest.raises(ReadingLibraryError):
        ReadingLibrary(tmp_path / "literature").save(paper(**fields))
    assert not (tmp_path / "literature").exists()


def test_entry_limit_allows_updates_but_preserves_file_on_new_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MAX_ENTRIES", 2)
    library = ReadingLibrary(tmp_path)
    library.save(paper(url="https://example.org/one"))
    library.save(paper(url="https://example.org/two"))
    before = library.path.read_bytes()
    with pytest.raises(ReadingLibraryError, match="already holds 2"):
        library.save(paper(url="https://example.org/three"))
    assert library.path.read_bytes() == before
    library.save(paper(url="https://example.org/one", title="Updated"))
    assert len(library.list()) == 2


def test_output_size_limit_preserves_previous_library(tmp_path, monkeypatch):
    library = ReadingLibrary(tmp_path)
    library.save(paper())
    before = library.path.read_bytes()
    monkeypatch.setattr(store, "MAX_BYTES", len(before) + 200)
    with pytest.raises(ReadingLibraryError, match="would exceed"):
        library.save(paper(url="https://example.org/two", annotation="x" * 1000))
    assert library.path.read_bytes() == before


def test_failed_atomic_replace_keeps_previous_file_and_removes_temp(tmp_path, monkeypatch):
    library = ReadingLibrary(tmp_path)
    library.save(paper())
    before = library.path.read_bytes()

    def denied(*_args):
        raise PermissionError("file is in use")

    monkeypatch.setattr(store.os, "replace", denied)
    with pytest.raises(ReadingLibraryError, match="previous reading list was preserved"):
        library.save(paper(title="A revised title"))
    assert library.path.read_bytes() == before
    assert not list(tmp_path.glob(".reading-list-*.tmp"))


def test_multiple_instances_do_not_lose_concurrent_thread_saves(tmp_path):
    def save(index):
        return ReadingLibrary(tmp_path).save(paper(url=f"https://example.org/paper-{index}"))

    with ThreadPoolExecutor(max_workers=4) as workers:
        saved = list(workers.map(save, range(12)))
    papers = ReadingLibrary(tmp_path).list()
    assert len(papers) == 12
    assert {row["id"] for row in papers} == {row["id"] for row in saved}
    assert json.loads((tmp_path / "reading-list.json").read_text(encoding="utf-8"))["version"] == 1
