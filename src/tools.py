"""Tools available to the research assistant agent."""

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from src.config import DATA_DIR, NOTES_DIR

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# File types the read tool will open as text.
READABLE_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".json", ".dat", ".log", ".xyz", ".pdb"}


def _resolve_data_path(filename: str) -> Path:
    """Resolve a user/model-supplied filename inside DATA_DIR, rejecting escapes."""
    path = (DATA_DIR / filename).resolve()
    if not path.is_relative_to(DATA_DIR.resolve()):
        raise ValueError(f"Path escapes the data directory: {filename}")
    return path


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers matching a query.

    Args:
        query: Search terms (topic, author names, or keywords), e.g.
            "reverse monte carlo polymer structure".
        max_results: Number of results to return (1-20).
    """
    max_results = max(1, min(int(max_results), 20))
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
    )
    with urllib.request.urlopen(f"{ARXIV_API_URL}?{params}", timeout=30) as resp:
        feed = ET.fromstring(resp.read())

    entries = feed.findall("atom:entry", ATOM_NS)
    if not entries:
        return f"No arXiv results for query: {query}"

    results = []
    for entry in entries:
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", "", ATOM_NS)).strip()
        summary = re.sub(r"\s+", " ", entry.findtext("atom:summary", "", ATOM_NS)).strip()
        published = entry.findtext("atom:published", "", ATOM_NS)[:10]
        arxiv_id = entry.findtext("atom:id", "", ATOM_NS).rsplit("/", 1)[-1]
        authors = ", ".join(
            a.findtext("atom:name", "", ATOM_NS)
            for a in entry.findall("atom:author", ATOM_NS)
        )
        results.append(
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"arXiv ID: {arxiv_id} ({published})\n"
            f"Abstract: {summary[:800]}"
        )
    return "\n\n---\n\n".join(results)


@tool
def list_data_files() -> str:
    """List all files in the lab's data/ directory (recursively), with sizes."""
    DATA_DIR.mkdir(exist_ok=True)
    files = [p for p in sorted(DATA_DIR.rglob("*")) if p.is_file()]
    if not files:
        return "The data/ directory is empty."
    lines = []
    for p in files:
        rel = p.relative_to(DATA_DIR)
        lines.append(f"{rel}  ({p.stat().st_size:,} bytes)")
    return "\n".join(lines)


@tool
def read_data_file(filename: str, max_chars: int = 8000) -> str:
    """Read a text-based file from the data/ directory.

    Args:
        filename: Path relative to data/, e.g. "results.csv" or "runs/run1.log".
        max_chars: Maximum characters to return (content is truncated beyond this).
    """
    path = _resolve_data_path(filename)
    if not path.exists():
        return f"File not found: {filename}. Use list_data_files to see what exists."
    if path.suffix.lower() not in READABLE_SUFFIXES:
        return (
            f"Cannot read {filename} as text (unsupported type '{path.suffix}'). "
            f"Readable types: {', '.join(sorted(READABLE_SUFFIXES))}"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[... truncated, {len(text):,} chars total]"
    return text


@tool
def summarize_csv(filename: str) -> str:
    """Compute summary statistics for a CSV file in the data/ directory.

    Returns shape, column names/dtypes, descriptive statistics for numeric
    columns, and a count of missing values.

    Args:
        filename: Path relative to data/, e.g. "measurements.csv".
    """
    path = _resolve_data_path(filename)
    if not path.exists():
        return f"File not found: {filename}. Use list_data_files to see what exists."
    df = pd.read_csv(path)
    parts = [
        f"File: {filename}",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "",
        "Columns and dtypes:",
        df.dtypes.to_string(),
        "",
        "Summary statistics (numeric columns):",
        df.describe().to_string(),
    ]
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        parts += ["", "Missing values:", missing.to_string()]
    return "\n".join(parts)


@tool
def save_note(title: str, content: str) -> str:
    """Save a persistent lab note (survives across sessions).

    Args:
        title: Short descriptive title, e.g. "RMC convergence findings".
        content: The note body (markdown is fine).
    """
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "note"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = NOTES_DIR / f"{stamp}-{slug}.md"
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return f"Saved note: {path.relative_to(DATA_DIR)}"


@tool
def read_notes() -> str:
    """Read all saved lab notes."""
    if not NOTES_DIR.exists():
        return "No notes saved yet."
    notes = sorted(NOTES_DIR.glob("*.md"))
    if not notes:
        return "No notes saved yet."
    parts = []
    for p in notes:
        parts.append(f"--- {p.name} ---\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


ALL_TOOLS = [
    search_arxiv,
    list_data_files,
    read_data_file,
    summarize_csv,
    save_note,
    read_notes,
]
