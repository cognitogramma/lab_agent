"""Experiment-aware discovery joins real local stores to one fake search."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from continuum_spectranalytics import literature, paths
from continuum_spectranalytics.agent.toolspec import ToolContext
from continuum_spectranalytics.literature_store import ReadingLibrary
from continuum_spectranalytics.memory.notes import NoteBook
from continuum_spectranalytics.state import AnomalyRecord, AnomalyType, Declaration, RunState
from continuum_spectranalytics.tools import ToolSurface


@pytest.fixture
def context(run_config, tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path / "project"))
    state = RunState(run_id="private-run-771", source_file="C:/private/research/secret-file.csv")
    surface = ToolSurface(state=state, config=run_config.model_copy(deep=True))
    return ToolContext(surface=surface, notes=NoteBook(tmp_path / "notes"), runs_dir=tmp_path / "history", author="private-author")


@pytest.fixture(autouse=True)
def external(monkeypatch):
    fake = SimpleNamespace(calls=[], response={"results": []})

    def search(query, **kwargs):
        fake.calls.append((query, kwargs))
        return fake.response

    monkeypatch.setattr(literature, "search_internet", search)
    return fake


def declare(context, **values):
    for key, value in values.items():
        context.surface.state.declarations[key] = Declaration(value=value, author="private-author")


def paper(title="Protein light scattering", url="https://example.org/paper", **extra):
    return {"title": title, "url": url, "snippet": "Second virial coefficient in protein solutions.",
            "published": "2026-10-30", "authors": ["A. Researcher"], **extra}


def past_run(context, *, run_id="past-bsa", sample="BSA", material="protein", basis="declared"):
    context.runs_dir.mkdir(parents=True, exist_ok=True)
    path = context.runs_dir / (run_id + ".provenance.json")
    record = {"run_id": run_id, "source_file": "C:/private/archive/hidden.csv", "written_at": "2026-10-29T10:00:00Z",
              "config": {"experiment_type": "acd_dilution"},
              "declarations": {"sample_name": {"value": sample}}, "parameters": None}
    if basis == "declared":
        record["declarations"]["polymer_type"] = {"value": material}
    elif basis == "inferred":
        record["config"]["optical"] = {"dn_dc": 0.185}
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_context_suggests_scientific_topics_without_private_identifiers(context):
    declare(context, sample_name="PRIVATE_BATCH_771 BSA", sample_class="protein_peptide", technique="SLS")
    context.surface.state.raw_signal = np.array([1.0, 2.0])
    context.surface.state.anomalies.append(AnomalyRecord(start_index=0, end_index=0, channel="ls8", detected_by=["hampel"],
                                                       anomaly_type=AnomalyType.BASELINE_DRIFT))
    result = literature.experiment_context(context)
    assert result["run_id"] == "private-run-771"
    assert "bovine serum albumin" in result["suggested_query"]
    assert "static light scattering" in result["suggested_query"]
    assert "second virial coefficient" in result["suggested_query"]
    assert "baseline drift" in result["suggested_query"]
    assert "research papers" in result["suggested_query"]
    for private in ("PRIVATE_BATCH", "771", "secret-file", "private-author", "C:/"):
        assert private not in result["suggested_query"]
    assert result["summary"].startswith("Experiment loaded")
    assert "PRIVATE_BATCH_771 BSA" in result["summary"]


def test_context_updates_with_current_declarations_and_anomalies(context):
    declare(context, sample_name="BSA", technique="SLS")
    first = literature.experiment_context(context)
    declare(context, sample_name="dextran", technique="DLS", experiment_type="fixed_conc_aggregation")
    second = literature.experiment_context(context)
    assert "dextran" in second["terms"]
    assert "dynamic light scattering" in second["terms"]
    assert "aggregation kinetics" in second["terms"]
    assert "bovine serum albumin" not in second["terms"]
    assert first != second


def test_context_summary_does_not_show_file_paths_or_credentials(context):
    declare(context, sample_name="C:\\private\\secret.csv", sample_class="sk-ant-private-key-123456789012345")
    result = literature.experiment_context(context)
    assert "private\\" not in result["summary"]
    assert "sk-ant" not in result["summary"]
    assert "secret.csv" not in result["suggested_query"]


def test_no_run_has_an_explicit_broad_search_status(context, external):
    context.surface.config.experiment_type = None
    result = literature.review_literature(context)
    assert result["experiment"]["summary"].startswith("No experiment is loaded or declared")
    assert result["query"] == "light scattering research papers"
    assert len(external.calls) == 1
    assert any("No experiment is loaded" in warning for warning in result["warnings"])


def test_explicit_query_is_sent_unchanged_exactly_once_with_filters(context, external):
    query = '  "protein aggregation"  second virial coefficient  '
    result = literature.review_literature(context, query=query, source="arxiv", max_results=3, recency="y", site="arxiv.org")
    assert external.calls == [(query, {"source": "arxiv", "max_results": 3, "recency": "y", "site": "arxiv.org"})]
    assert result["query"] == query
    assert result["source"] == "arxiv"
    assert result["recency"] == "y"
    assert result["site"] == "arxiv.org"


def test_private_notes_and_past_run_data_never_become_external_query(context, external):
    declare(context, sample_name="BSA")
    context.notes.add("BSA private note", "INTERNAL_SENTENCE DO_NOT_SEND; run private-run-771 and key sk-secret-123456789012345",
                      author="confidential-author", run_id="private-run-771", tags=["unreleased-compound"])
    past_run(context)
    result = literature.review_literature(context)
    sent_query = external.calls[0][0]
    for private in ("INTERNAL_SENTENCE", "DO_NOT_SEND", "771", "sk-secret", "unreleased", "confidential", "hidden.csv"):
        assert private not in sent_query
    assert any("INTERNAL_SENTENCE" in item["excerpt"] for item in result["memory"])


def test_matching_notes_without_urls_are_retained_with_attribution(context, external):
    declare(context, sample_name="BSA")
    note = context.notes.add("BSA blank sensitivity", "The buffer blank was recorded later.", author="Researcher A", tags=["blank"])
    result = literature.review_literature(context)
    memory = next(item for item in result["memory"] if item["kind"] == "note")
    assert memory["title"] == "BSA blank sensitivity"
    assert memory["author"] == "Researcher A"
    assert memory["created_at"] == note.created_at
    assert memory["tags"] == ["blank"]
    assert memory["id"].startswith("note:")
    assert result["results"] == []


def test_web_and_note_reference_are_deduplicated_without_fabricating_metadata(context, external):
    declare(context, sample_name="BSA")
    context.notes.add("BSA reference", "See [this paper](https://example.org/paper?utm_source=note#s) for protein scattering.", author="tester")
    external.response = {"results": [paper()]}
    result = literature.review_literature(context)
    assert len(result["results"]) == 1
    candidate = result["results"][0]
    assert candidate["title"] == "Protein light scattering"
    assert candidate["origin"] == "web"
    assert candidate["published"] == "2026-10-30"
    assert candidate["authors"] == ["A. Researcher"]
    assert candidate["memory_refs"][0]["kind"] == "note"
    assert candidate["memory_refs"][0]["title"] == "BSA reference"
    assert len(external.calls) == 1


def test_note_only_reference_is_labelled_as_local_not_an_abstract(context):
    context.notes.add("Protein scattering", "See https://example.org/paper for the original report.", author="tester")
    result = literature.review_literature(context, query="protein scattering")
    candidate = result["results"][0]
    assert candidate["origin"] == "memory"
    assert candidate["title"] == "Reference in note: Protein scattering"
    assert candidate["published"] is None
    assert candidate["snippet"].startswith("Local note excerpt (not a paper abstract):")


def test_arxiv_pdf_and_abstract_versions_are_one_candidate(context, external):
    context.notes.add("Protein scattering", "Read https://arxiv.org/abs/2601.12345v1#s", author="tester")
    external.response = {"results": [paper(url="https://arxiv.org/pdf/2601.12345v1.pdf")]}
    result = literature.review_literature(context, query="protein scattering", source="arxiv")
    assert len(result["results"]) == 1
    assert result["results"][0]["origin"] == "arxiv"
    assert result["results"][0]["memory_refs"]


def test_saved_reading_library_is_a_third_memory_source_preserving_metadata(context, external):
    saved = ReadingLibrary(paths.project_root() / "data" / "literature").save(paper())
    result = literature.review_literature(context, query="protein scattering")
    candidate = result["results"][0]
    assert candidate["origin"] == "memory"
    assert candidate["saved_at"] == saved["saved_at"]
    assert candidate["id"] == saved["id"]
    assert candidate["authors"] == ["A. Researcher"]
    assert candidate["memory_refs"][0] == {"kind": "paper", "id": saved["id"], "title": saved["title"]}
    assert next(item for item in result["memory"] if item["kind"] == "paper")["saved_at"] == saved["saved_at"]


def test_saved_and_external_paper_share_one_candidate_and_memory_reference(context, external):
    saved = ReadingLibrary(paths.project_root() / "data" / "literature").save(paper())
    external.response = {"results": [paper(url="https://example.org/paper?utm_source=search")]}
    result = literature.review_literature(context, query="protein scattering")
    assert len(result["results"]) == 1
    assert result["results"][0]["origin"] == "web"
    assert result["results"][0]["saved_at"] == saved["saved_at"]
    assert result["results"][0]["memory_refs"][0]["id"] == saved["id"]


def test_saved_note_and_run_references_survive_review_while_relevance_is_recomputed(context):
    refs = [{"kind": "note", "id": "note-previous", "title": "Original lab conclusion"},
            {"kind": "run", "id": "run-previous", "title": "Earlier protein experiment"}]
    saved = ReadingLibrary(paths.project_root() / "data" / "literature").save(
        paper(memory_refs=refs, relevance=["Historical match for an unrelated quantum question."]))
    result = literature.review_literature(context, query="protein scattering")
    candidate = result["results"][0]
    assert all(ref in candidate["memory_refs"] for ref in refs)
    assert {"kind": "paper", "id": saved["id"], "title": saved["title"]} in candidate["memory_refs"]
    assert len(candidate["memory_refs"]) <= 8
    assert not any("quantum" in reason for reason in candidate["relevance"])
    assert any("topic match" in reason.lower() for reason in candidate["relevance"])
    assert any("filters apply to the external search" in warning for warning in result["warnings"])


def test_past_run_memory_preserves_material_inference_basis(context):
    past_run(context, basis="inferred")
    result = literature.review_literature(context, query="protein scattering")
    item = next(item for item in result["memory"] if item["kind"] == "run")
    assert "inferred from dn_dc=0.185" in item["excerpt"]
    assert "declared material" not in item["excerpt"].lower()
    assert "hidden.csv" not in item["excerpt"]


def test_deterministic_topic_ranking_explains_lexical_matches_only(context, external):
    external.response = {"results": [paper("Other subject", "https://example.org/z", snippet="Unrelated documentation."),
                                      paper("Protein scattering virial coefficient", "https://example.org/a")]}
    result = literature.review_literature(context, query="protein scattering virial coefficient")
    candidate = result["results"][0]
    assert candidate["url"] == "https://example.org/a"
    assert any("topic match" in reason.lower() for reason in candidate["relevance"])
    assert any("lexical" in reason.lower() for reason in candidate["relevance"])
    assert any("not evidence of scientific agreement" in warning for warning in result["warnings"])


def test_external_failure_keeps_useful_local_results_and_memory(context, external):
    context.notes.add("Protein scattering", "See https://example.org/paper", author="tester")
    external.response = {"results": [], "error": "Search is offline."}
    result = literature.review_literature(context, query="protein scattering")
    assert result["results"]
    assert result["memory"]
    assert "error" not in result
    assert "Search is offline." in result["warnings"]


def test_external_failure_with_note_but_no_url_is_still_useful(context, external):
    context.notes.add("Protein scattering", "A relevant remembered observation.", author="tester")
    external.response = {"results": [], "error": "Search is offline."}
    result = literature.review_literature(context, query="protein scattering")
    assert not result["results"]
    assert result["memory"]
    assert "error" not in result


def test_external_failure_without_local_matches_is_an_error(context, external):
    external.response = {"results": [], "error": "Search is offline."}
    result = literature.review_literature(context, query="protein scattering")
    assert result["error"] == "Search is offline."
    assert result["memory"] == []


def test_invalid_external_data_does_not_hide_memory(context, external):
    context.notes.add("Protein scattering", "A local observation.", author="tester")
    external.response = None
    result = literature.review_literature(context, query="protein scattering")
    assert result["memory"]
    assert "error" not in result
    assert any("unreadable" in warning for warning in result["warnings"])


@pytest.mark.parametrize("url", ["file:///secret", "javascript:alert(1)", "http://127.0.0.1/", "http://metadata.google.internal/",
                                 "http://intranet/paper", "http://2130706433/paper", "http://0177.0.0.1/paper",
                                 "https://user:password@example.org/paper", "https://example.org/paper?access_token=private"])
def test_local_reference_urls_must_be_public_without_access_credentials(context, url):
    context.notes.add("Protein scattering", "Read " + url, author="tester")
    result = literature.review_literature(context, query="protein scattering")
    assert result["memory"]
    assert result["results"] == []


@pytest.mark.parametrize("kwargs", [{"query": "q" * 501}, {"query": None}, {"source": "news"}, {"max_results": 0},
                                   {"max_results": 21}, {"max_results": True}, {"recency": "decade"}, {"site": []}])
def test_invalid_inputs_never_start_an_external_search(context, external, kwargs):
    result = literature.review_literature(context, **kwargs)
    assert result["error"]
    assert external.calls == []


def test_memory_paper_and_text_counts_are_bounded(context, external):
    for index in range(35):
        context.notes.add(f"Protein scattering {index}", "x" * 3000 + f" https://example.org/{index}", author="tester")
    external.response = {"results": [paper(f"Protein scattering {index}", f"https://external.org/{index}", snippet="x" * 5000) for index in range(40)]}
    result = literature.review_literature(context, query="protein scattering", max_results=5)
    assert len(result["results"]) == 5
    assert len(result["memory"]) == literature.MAX_MEMORY
    assert all(len(item["excerpt"]) <= literature.MAX_EXCERPT for item in result["memory"])
    assert all(len(item["snippet"]) <= 800 for item in result["results"])
    assert all(len(item["memory_refs"]) <= 8 for item in result["results"])


def test_review_never_mutates_measurements_declarations_or_stores(context, external):
    declare(context, sample_name="BSA")
    context.surface.state.raw_signal = np.array([1., 2., 3.])
    context.notes.add("BSA scattering", "A remembered observation.", author="tester")
    run_file = past_run(context)
    state_before = context.surface.state.model_dump_json()
    config_before = context.surface.config.model_dump_json()
    notes_before = context.notes.path.read_bytes()
    run_before = run_file.read_bytes()
    raw_before = context.surface.state.raw_signal.copy()
    literature.review_literature(context)
    assert context.surface.state.model_dump_json() == state_before
    assert context.surface.config.model_dump_json() == config_before
    assert context.notes.path.read_bytes() == notes_before
    assert run_file.read_bytes() == run_before
    np.testing.assert_array_equal(context.surface.state.raw_signal, raw_before)
    assert not (paths.project_root() / "data" / "literature").exists()


def test_unknown_material_is_not_presented_as_research_knowledge(context):
    past_run(context, material="unknown")
    result = literature.review_literature(context, query="BSA dilution")
    run = next(item for item in result["memory"] if item["kind"] == "run")
    assert "Material metadata" not in run["excerpt"]
