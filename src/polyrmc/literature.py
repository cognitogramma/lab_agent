"""Experiment-aware source discovery with local research memory kept local.

Suggestions use a fixed scientific vocabulary, never file names, run IDs,
authors or note sentences. Each review makes one external search; all memory
matching and ranking is deterministic, local, and explicitly lexical.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import urllib.parse
from itertools import islice
from typing import TYPE_CHECKING, Any

from polyrmc import paths
from polyrmc.research import search_internet
from polyrmc.web_reader import _target

if TYPE_CHECKING:
    from polyrmc.agent.toolspec import ToolContext

MAX_MEMORY = 20
MAX_SCAN = 500
MAX_TERMS = 12
MAX_EXCERPT = 600

# Canonical public phrases, not free-form excerpts from declarations. Unknown
# compounds and private sample codes are intentionally absent from suggestions.
_VOCABULARY = (
    (r"\bbsa\b|\bbovine serum albumin\b", "bovine serum albumin"),
    (r"\blysozyme\b", "lysozyme"), (r"\bdextran\b", "dextran"),
    (r"\bliraglutide\b", "liraglutide"), (r"\bpolystyrene\b", "polystyrene"),
    (r"\bpeg\b|\bpolyethylene glycol\b", "polyethylene glycol"),
    (r"\bcellulose\b", "cellulose"), (r"\bhyaluronic acid\b", "hyaluronic acid"),
    (r"\balginate\b", "alginate"), (r"\bpolysaccharides?\b", "polysaccharide"),
    (r"\bproteins?\b", "protein"), (r"\bpeptides?\b", "peptide"),
    (r"\bpolymers?\b", "polymer"), (r"\bcolloids?\b", "colloid"),
    (r"\bnanoparticles?\b", "nanoparticle"),
    (r"\bdls\b|\bdynamic light scattering\b", "dynamic light scattering"),
    (r"\bsls\b|\bstatic light scattering\b", "static light scattering"),
)
_EXPERIMENTS = {
    "acd_dilution": ("automatic continuous dilution", "second virial coefficient"),
    "fixed_conc_aggregation": ("aggregation kinetics",),
}
_ANOMALIES = {
    "transient_spike": "signal spikes", "baseline_drift": "baseline drift",
    "saturation_artifact": "detector saturation", "structural_noise": "structural noise",
}
_CONTEXT_FIELDS = ("sample_name", "sample_class", "sample_type", "polymer_type", "technique")
_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "from", "by",
         "as", "is", "are", "was", "were", "this", "that", "our", "their", "your", "we", "it",
         "research", "paper", "papers", "study", "studies", "using", "use", "find", "about", "review"}
_SECRET_QUERY = {"token", "access_token", "api_key", "apikey", "key", "password", "secret", "signature", "credential", "authorization"}


def _text(value: Any, limit: int) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ""
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit - 1] + "…"


def _value(declarations: dict, key: str):
    declaration = declarations.get(key)
    if isinstance(declaration, dict):
        return declaration.get("value")
    return getattr(declaration, "value", declaration)


def _tokens(value: str) -> set[str]:
    value = value.lower().replace("_", " ").replace("a₂", "a2")
    words = set(re.findall(r"[a-z][a-z0-9-]{1,}", value))
    for pattern, canonical in _VOCABULARY:
        if re.search(pattern, value, re.I):
            words.update(re.findall(r"[a-z][a-z0-9-]{1,}", canonical))
    words -= _STOP
    return {word[:-1] if word.endswith("s") and len(word) > 4 else word for word in words}


def _label(value: Any) -> str:
    text = _text(value, 100)
    if (not text or re.search(r"[/\\]|https?:|\b(?:sk|pk)[_-][a-z0-9_-]{8,}|\bAIza[a-z0-9_-]{20,}|\b(?:AKIA|ASIA)[A-Z0-9]{16}", text, re.I)
            or (len(text) >= 40 and re.fullmatch(r"[a-zA-Z0-9_-]+", text) and any(char.isdigit() for char in text))
            or re.search(r"(?:api.?key|access.?token|password|secret)\s*[:=]", text, re.I)):
        return ""
    return text


def experiment_context(ctx: "ToolContext") -> dict[str, Any]:
    """Summarise current local context and suggest public scientific terms only."""
    surface = getattr(ctx, "surface", None)
    state = getattr(surface, "state", None)
    config = getattr(surface, "config", None)
    declarations = dict(getattr(state, "declarations", {}) or {})
    texts = [_text(_value(declarations, key), 500).replace("_", " ") for key in _CONTEXT_FIELDS]
    blob = " ".join(texts)
    terms = [canonical for pattern, canonical in _VOCABULARY if re.search(pattern, blob, re.I)]
    technique = _text(_value(declarations, "technique"), 50).lower()
    if technique == "both":
        terms.extend(["static light scattering", "dynamic light scattering"])
    experiment = _value(declarations, "experiment_type") or getattr(config, "experiment_type", None)
    if not isinstance(experiment, str):
        experiment = None
    terms.extend(_EXPERIMENTS.get(experiment, ()))
    if not any("light scattering" in term for term in terms):
        terms.append("light scattering")
    anomaly_names = []
    for anomaly in list(getattr(state, "anomalies", ()) or ())[:100]:
        kind = getattr(anomaly, "anomaly_type", None)
        name = getattr(kind, "value", kind)
        if name in _ANOMALIES and _ANOMALIES[name] not in anomaly_names:
            anomaly_names.append(_ANOMALIES[name])
    terms.extend(anomaly_names[:2])
    terms = list(dict.fromkeys(terms))[:MAX_TERMS]
    loaded = getattr(state, "raw_signal", None) is not None
    summary = ["Experiment loaded." if loaded else "No experiment is loaded; using available declarations and configuration."]
    sample = next((_label(_value(declarations, key)) for key in ("sample_name", "sample_class", "polymer_type", "sample_type")
                   if _label(_value(declarations, key))), "")
    if sample:
        summary.append("Declared sample: " + sample + ".")
    if experiment in _EXPERIMENTS:
        summary.append("Configured experiment: " + _EXPERIMENTS[experiment][0] + ".")
    if anomaly_names:
        summary.append("Recorded signal topics: " + ", ".join(anomaly_names[:2]) + ".")
    if not loaded and not sample and experiment not in _EXPERIMENTS:
        summary = ["No experiment is loaded or declared. This is a broad literature search; open an experiment to tailor it."]
    suggested = " ".join(terms[:8] + ["research papers"])
    return {
        "run_id": _text(getattr(state, "run_id", ""), 100),
        "summary": _text(" ".join(summary), 600),
        "suggested_query": suggested[:500], "terms": terms,
    }


def _paper_url(value: Any):
    if not isinstance(value, str):
        return None
    try:
        target = _target(value)
    except ValueError:
        return None
    try:
        ipaddress.ip_address(target.host)
    except ValueError:
        # Reference extraction performs no DNS. Single-label intranet names
        # and numeric alternate IPv4 spellings are not public paper hosts.
        if "." not in target.host or not re.search(r"[a-z]", target.host.rsplit(".", 1)[-1]):
            return None
    parts = urllib.parse.urlsplit(target.url)
    components = []
    for component in parts.query.split("&"):
        key = urllib.parse.unquote_plus(component.partition("=")[0]).lower()
        if key in _SECRET_QUERY or key.startswith(("x-amz-", "x-goog-")):
            return None
        if key and not key.startswith("utm_") and key not in {"gclid", "fbclid", "msclkid"}:
            components.append(component)
    url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "&".join(components), ""))
    canonical_host = target.host.removeprefix("www.")
    canonical_path = parts.path.rstrip("/")
    if canonical_host == "arxiv.org" and canonical_path.startswith("/pdf/"):
        canonical_path = "/abs/" + canonical_path[5:].removesuffix(".pdf")
    key = (canonical_host, canonical_path,
           tuple(sorted(urllib.parse.parse_qsl("&".join(components), keep_blank_values=True))))
    return url, target.host, key


def _note_urls(body: str):
    bounded = body[:12000]
    for match in re.finditer(r"https?://[^\s<>\"']+", bounded):
        if len(body) > len(bounded) and match.end() == len(bounded):
            continue  # A cut-off URL is not a source record.
        url = match.group().rstrip(".,;!?")
        for closing, opening in ((")", "("), ("]", "["), ("}", "{")):
            while url.endswith(closing) and url.count(closing) > url.count(opening):
                url = url[:-1]
        parsed = _paper_url(url)
        if parsed is not None:
            yield parsed


def _relevance(blob: str, query_terms: set[str], context_terms: set[str]):
    tokens = _tokens(blob)
    query_overlap = sorted(tokens & query_terms)
    context_overlap = sorted(tokens & context_terms)
    reasons = []
    if query_overlap:
        reasons.append("Query topic match: " + ", ".join(query_overlap[:7]) + " (lexical overlap only).")
    if context_overlap:
        reasons.append("Experiment topic match: " + ", ".join(context_overlap[:7]) + " (not scientific validation).")
    return len(query_overlap) * 4 + len(context_overlap) * 2, reasons


def _references(value: Any) -> list[dict[str, str]]:
    """Keep bounded saved source links without treating them as new evidence."""
    if not isinstance(value, (list, tuple)):
        return []
    references = []
    for raw in value[:20]:
        if isinstance(raw, str):
            ref = {"kind": "memory", "title": _text(raw, 240)}
        elif isinstance(raw, dict):
            kind = _text(raw.get("kind"), 40)
            if kind and kind not in {"note", "run", "paper", "memory"}:
                continue
            ref = {key: text for key, limit in (("kind", 40), ("id", 160), ("title", 240), ("run_id", 160))
                   if (text := _text(raw.get(key), limit))}
        else:
            continue
        if not any(ref.get(key) for key in ("id", "title", "run_id")):
            continue
        if ref not in references:
            references.append(ref)
        if len(references) == 8:
            break
    return references


def _paper(raw: dict, origin: str):
    parsed = _paper_url(raw.get("url") or raw.get("href"))
    title = _text(raw.get("title"), 240)
    if parsed is None or not title:
        return None
    url, domain, key = parsed
    paper = {
        "title": title, "url": url, "snippet": _text(raw.get("snippet") or raw.get("abstract"), 800),
        "domain": domain, "published": _text(raw.get("published"), 40) or None,
        "origin": origin, "relevance": [],
        "memory_refs": _references(raw.get("memory_refs")) if origin == "memory" else [],
    }
    authors = raw.get("authors")
    if isinstance(authors, str):
        authors = [authors]
    if isinstance(authors, (list, tuple)):
        names = [_text(author, 100) for author in authors[:12] if isinstance(author, str)]
        if names:
            paper["authors"] = names
    for field in ("id", "saved_at"):
        if raw.get(field):
            paper[field] = _text(raw[field], 100)
    return paper, key


def _local_memory(ctx, query_terms, context_terms, run_id, warnings):
    records = []
    papers = []
    try:
        notes = ctx.notes.search(limit=MAX_SCAN)
    except Exception:
        notes = []
        warnings.append("Local notes could not be read; other sources are still available.")
    for note in notes[:MAX_SCAN]:
        subject = _text(getattr(note, "subject", ""), 240)
        raw_body = getattr(note, "body", "")
        raw_body = raw_body if isinstance(raw_body, str) else ""
        body = _text(raw_body, 12000)
        tags = [_text(tag, 80) for tag in list(getattr(note, "tags", ()) or ())[:12]]
        score, reasons = _relevance(subject + " " + body + " " + " ".join(tags), query_terms, context_terms)
        same_run = bool(run_id and getattr(note, "run_id", None) == run_id)
        if not score and not same_run:
            continue
        identifier = "note:" + hashlib.sha256((str(getattr(note, "created_at", "")) + "\0" + subject + "\0" + body).encode()).hexdigest()[:12]
        record = {"kind": "note", "id": identifier, "title": subject or "Lab note",
                  "excerpt": _text(body, MAX_EXCERPT), "tags": tags,
                  "author": _text(getattr(note, "author", ""), 120),
                  "created_at": _text(getattr(note, "created_at", ""), 40)}
        if getattr(note, "run_id", None):
            record["run_id"] = _text(note.run_id, 100)
        ref = {key: record[key] for key in ("kind", "id", "title")}
        records.append((score + (2 if same_run else 0), record))
        for url, domain, key in islice(_note_urls(raw_body), 8):
            paper = {"title": _text("Reference in note: " + subject, 240), "url": url, "domain": domain,
                     "snippet": "Local note excerpt (not a paper abstract): " + _text(body, 600),
                     "published": None, "origin": "memory", "relevance": reasons, "memory_refs": [ref]}
            papers.append((paper, key))
    try:
        runs = ctx.surface.search_past_runs(runs_dir=ctx.runs_dir)
    except Exception:
        runs = []
        warnings.append("Past-run memory could not be read; other sources are still available.")
    for run in runs[:MAX_SCAN]:
        if not isinstance(run, dict):
            continue
        label = _text(run.get("sample_name") or run.get("run_id"), 180)
        experiment = _text(run.get("experiment_type"), 100).replace("_", " ")
        material = _text(run.get("polymer_type"), 100)
        material_basis = _text(run.get("polymer_type_basis"), 120) or "not recorded"
        known_material = material if material.lower() not in {"unknown", "unrecorded", "none"} else ""
        blob = label + " " + experiment + " " + known_material
        score, _ = _relevance(blob, query_terms, context_terms)
        if not score:
            continue
        identifier = _text(run.get("run_id"), 100)
        material_detail = f" Material metadata: {known_material} (basis: {material_basis})." if known_material else ""
        records.append((score, {"kind": "run", "id": identifier, "run_id": identifier,
                               "title": label or "Past experiment",
                               "excerpt": _text(f"Recorded experiment: {experiment or 'not declared'}. Sample: {label}.{material_detail} Topic overlap is not a comparison of measured parameters.", MAX_EXCERPT)}))
    try:
        from polyrmc.literature_store import ReadingLibrary

        saved = ReadingLibrary(paths.project_root() / "data" / "literature").list()
    except Exception:
        saved = []
        warnings.append("The saved reading library could not be read; other sources are still available.")
    for raw in saved[:MAX_SCAN]:
        if not isinstance(raw, dict):
            continue
        normalized = _paper(raw, "memory")
        if normalized is None:
            continue
        paper, key = normalized
        score, reasons = _relevance(paper["title"] + " " + paper["snippet"], query_terms, context_terms)
        if not score:
            continue
        identifier = _text(raw.get("id"), 100) or "paper:" + hashlib.sha256(paper["url"].encode()).hexdigest()[:12]
        record = {"kind": "paper", "id": identifier, "title": paper["title"],
                  "excerpt": _text(paper["snippet"], MAX_EXCERPT)}
        if paper.get("saved_at"):
            record["saved_at"] = paper["saved_at"]
        ref = {key: record[key] for key in ("kind", "id", "title")}
        if ref not in paper["memory_refs"]:
            paper["memory_refs"] = paper["memory_refs"][:7] + [ref]
        paper["relevance"] = reasons
        records.append((score, record))
        papers.append((paper, key))
    records.sort(key=lambda item: (-item[0], item[1]["kind"], item[1]["title"].casefold(), item[1]["id"]))
    return [record for _, record in records[:MAX_MEMORY]], papers


def review_literature(
    ctx: "ToolContext", query: str = "", source: str = "web", max_results: int = 8,
    recency: str = "any", site: str = "",
) -> dict[str, Any]:
    """Combine one external search with notes, prior runs and saved sources."""
    context = experiment_context(ctx)
    response: dict[str, Any] = {"query": "", "source": source, "recency": recency, "site": site or None, "experiment": context,
                                "results": [], "memory": [], "warnings": []}
    warnings = response["warnings"]
    if not isinstance(query, str) or len(query) > 500:
        response["error"] = "Use a query of at most 500 characters."
        return response
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 20:
        response["error"] = "Result count must be an integer from 1 to 20."
        return response
    if not isinstance(source, str) or source not in {"web", "arxiv"}:
        response["error"] = "Choose web or arxiv as the search source."
        return response
    if not isinstance(recency, str) or recency not in {"any", "d", "w", "m", "y"}:
        response["error"] = "Recency must be any, d, w, m, or y."
        return response
    if site is not None and (not isinstance(site, str) or len(site) > 253):
        response["error"] = "Site must be a hostname of at most 253 characters."
        return response
    chosen = query if query.strip() else context["suggested_query"]
    response["query"] = chosen
    query_terms = _tokens(chosen)
    context_terms = _tokens(" ".join(context["terms"]))
    if context["summary"].startswith("No experiment"):
        warnings.append("No experiment is loaded. Relevance uses available public topics, not a measured-data comparison.")
    warnings.append("Topic matches are lexical suggestions, not evidence of scientific agreement. Web results are source candidates; publication and peer-review status are not verified.")
    memory, local_papers = _local_memory(ctx, query_terms, context_terms, context["run_id"], warnings)
    response["memory"] = memory
    if local_papers:
        warnings.append("Source, date and site filters apply to the external search. Local memory candidates remain visible and may fall outside those filters.")
    try:
        external = search_internet(chosen, source=source, max_results=max_results, recency=recency, site=site or None)
    except Exception:
        external = {"results": [], "error": "External search is unavailable. Local research memory is still shown when matched."}
    if not isinstance(external, dict):
        external = {"results": [], "error": "External search returned unreadable results."}
    response["site"] = external.get("site", site or None)
    response["recency"] = external.get("recency", recency)
    if external.get("warning"):
        warnings.append(_text(external["warning"], 500))
    external_error = _text(external.get("error"), 500)
    if external_error:
        warnings.append(external_error)
    combined = {}
    rows = external.get("results", [])
    if isinstance(rows, list):
        for raw in rows[:40]:
            if isinstance(raw, dict):
                normalized = _paper(raw, source)
                if normalized is not None:
                    paper, key = normalized
                    combined.setdefault(key, paper)
    for paper, key in local_papers:
        if key not in combined:
            combined[key] = paper
            continue
        existing = combined[key]
        for ref in paper["memory_refs"]:
            if ref not in existing["memory_refs"] and len(existing["memory_refs"]) < 8:
                existing["memory_refs"].append(ref)
        for field in ("authors", "published", "saved_at", "id"):
            if not existing.get(field) and paper.get(field):
                existing[field] = paper[field]
    ranked = []
    for key, paper in combined.items():
        score, reasons = _relevance(paper["title"] + " " + paper["snippet"], query_terms, context_terms)
        paper["relevance"] = reasons or ["Returned by source search; no lexical overlap with the current public topics was detected."]
        if paper["memory_refs"]:
            paper["relevance"].append("Referenced in local research memory; this does not verify the source's claims.")
        paper["memory_refs"] = paper["memory_refs"][:8]
        ranked.append((score, key, paper))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    response["results"] = [paper for _, _, paper in ranked[:max_results]]
    response["warnings"] = list(dict.fromkeys(warnings))[:8]
    if external_error and not response["results"] and not memory:
        response["error"] = external_error
    if not response["results"] and not memory and not external_error:
        response["warnings"].append("No matching sources or local research memories were found. Try broader public topic terms.")
    return response


__all__ = ["experiment_context", "review_literature"]
