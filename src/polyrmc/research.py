"""Bounded internet lookups, separate from the measurement layer.

No provider or network work happens on import. Web search uses the optional
``ddgs`` package; arXiv uses its public Atom API. Both entry points return the
same small JSON-compatible record so the model and the window see the same
sources. Snippets are external, untrusted text, not instructions or evidence
about the current experiment.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import Any

TIMEOUT = 15
MAX_RESULTS = 20
MAX_QUERY = 500
MAX_RESPONSE = 2 * 1024 * 1024
ARXIV_URL = "https://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
RECENCY_DAYS = {"d": 1, "w": 7, "m": 31, "y": 365}
WEB_BACKENDS = "brave,duckduckgo,startpage,yahoo"
_TRACKING = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid"}
_arxiv_lock = threading.Lock()
_arxiv_last_request = float("-inf")


def _text(value: Any, limit: int) -> str:
    """Collapse display whitespace and bound every externally supplied field."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _site(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or len(value) > 253:
        raise ValueError("Site must be a hostname such as arxiv.org, without a URL or path.")
    try:
        host = value.strip().rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise ValueError("Site must be a valid hostname.") from error
    if not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
               for label in host.split(".")):
        raise ValueError("Site must be a hostname such as arxiv.org, without a URL or path.")
    return host


def _url(value: Any) -> tuple[str, str] | None:
    """Reject active URL schemes and strip tracking and fragments for deduping."""
    if not isinstance(value, str) or len(value) > 4096 or re.search(r"[\x00-\x20\x7f]", value):
        return None
    try:
        parts = urllib.parse.urlsplit(value)
        host = (parts.hostname or "").encode("idna").decode("ascii").lower()
        if parts.scheme.lower() not in {"http", "https"} or not host or parts.username or parts.password:
            return None
        port = parts.port
        netloc = f"[{host}]" if ":" in host else host
        if port is not None and (parts.scheme.lower(), port) not in {("http", 80), ("https", 443)}:
            netloc += f":{port}"
        # Preserve non-tracking bytes: re-encoding can invalidate signed links
        # or change a literal query value. Only the separate dedup key is sorted.
        query_parts = []
        for component in parts.query.split("&"):
            key = urllib.parse.unquote_plus(component.partition("=")[0]).lower()
            if not key.startswith("utm_") and key not in _TRACKING:
                query_parts.append(component)
        query = "&".join(query_parts)
        path = parts.path or "/"
        clean = urllib.parse.urlunsplit((parts.scheme.lower(), netloc, path, query, ""))
        return clean, host
    except (ValueError, UnicodeError):
        return None


def _normalize(rows: Any, limit: int, site: str | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in islice(rows, MAX_RESULTS * 3):
        if not isinstance(row, dict):
            continue
        target = _url(row.get("url") or row.get("href"))
        if target is None:
            continue
        url, domain = target
        if site and domain != site and not domain.endswith("." + site):
            continue
        parts = urllib.parse.urlsplit(url)
        key = (parts.netloc.removeprefix("www."), parts.path.rstrip("/"),
               urllib.parse.urlencode(sorted(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))))
        if key in seen:
            continue
        title = _text(row.get("title"), 240)
        if not title:
            continue
        seen.add(key)
        result = {
            "title": title,
            "url": url,
            "snippet": _text(row.get("snippet") or row.get("body"), 800),
            "domain": domain,
            "published": _text(row.get("published") or row.get("date"), 40) or None,
        }
        authors = row.get("authors")
        if isinstance(authors, str):
            authors = [authors]
        if isinstance(authors, (list, tuple)):
            names = [_text(author, 100) for author in authors[:12] if isinstance(author, str) and author.strip()]
            if names:
                result["authors"] = names
        results.append(result)
        if len(results) >= limit:
            break
    return results


def _web_rows(query: str, limit: int, recency: str) -> Any:
    from ddgs import DDGS

    return DDGS(timeout=TIMEOUT).text(
        query, max_results=min(limit * 2, MAX_RESULTS * 3),
        timelimit=None if recency == "any" else recency,
        # Auto can prioritize encyclopedia backends that ignore date filters.
        backend=WEB_BACKENDS,
    )


def _arxiv_rows(query: str, limit: int, recency: str) -> list[dict[str, Any]]:
    global _arxiv_last_request
    # Quoted phrases remain phrases; plain words each get their own field.
    if query.count('"') % 2:
        raise ValueError("Close the double quotation marks around a search phrase.")
    terms = [phrase or word for phrase, word in re.findall(r'"([^"]*)"|([^\s"]+)', query)
             if (phrase or word).strip()]
    expression = " AND ".join('all:"' + term.replace('"', '\\"') + '"' for term in terms)
    if not expression:
        raise ValueError("Enter at least one search term.")
    if recency != "any":
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=RECENCY_DAYS[recency])
        expression += f" AND submittedDate:[{start:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"
    params = urllib.parse.urlencode({
        "search_query": expression, "start": 0, "max_results": min(limit * 2, 40),
        "sortBy": "relevance" if recency == "any" else "submittedDate",
        "sortOrder": "descending",
    })
    request = urllib.request.Request(
        f"{ARXIV_URL}?{params}",
        headers={"User-Agent": "PolyRMC/0.1 (research literature lookup)",
                 "Accept": "application/atom+xml"},
    )
    # arXiv asks clients to leave three seconds between serial API calls.
    with _arxiv_lock:
        delay = 3.0 - (time.monotonic() - _arxiv_last_request)
        if delay > 0:
            time.sleep(delay)
        _arxiv_last_request = time.monotonic()
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = response.read(MAX_RESPONSE + 1)
    if len(payload) > MAX_RESPONSE:
        raise ValueError("arXiv returned an oversized response; narrow the query.")
    feed = ET.fromstring(payload)
    if feed.tag != "{" + ATOM["atom"] + "}feed":
        raise ValueError("arXiv returned an unexpected response. Try again later.")
    rows = []
    for entry in feed.findall("atom:entry", ATOM):
        identifier = entry.findtext("atom:id", "", ATOM)
        if "/api/errors" in identifier:
            raise ValueError("arXiv rejected the query. Try simpler search terms.")
        # The API commonly emits http links; use HTTPS for arXiv itself.
        url = identifier
        for link in entry.findall("atom:link", ATOM):
            if link.get("rel") == "alternate":
                url = link.get("href", identifier)
                break
        if urllib.parse.urlsplit(url).hostname in {"arxiv.org", "www.arxiv.org"}:
            url = re.sub(r"^http:", "https:", url)
        rows.append({
            "title": entry.findtext("atom:title", "", ATOM), "url": url,
            "snippet": entry.findtext("atom:summary", "", ATOM),
            "published": entry.findtext("atom:published", "", ATOM)[:10],
            "authors": [author.findtext("atom:name", "", ATOM)
                        for author in entry.findall("atom:author", ATOM)],
        })
    return rows


def search_internet(
    query: str, source: str = "web", max_results: int = 8,
    recency: str = "any", site: str | None = None,
) -> dict[str, Any]:
    """Search the web or arXiv and return bounded, deduplicated source records.

    ``recency`` is any/d/w/m/y; a month is 31 days and a year 365 for arXiv.
    arXiv searches plain words and quoted phrases, not raw API expressions.
    Failures return ``error`` and an empty result list, never fabricated results.
    """
    response: dict[str, Any] = {
        "query": _text(query, MAX_QUERY), "source": _text(source, 20), "results": [],
        "recency": _text(recency, 10), "site": None,
    }
    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Enter a search query.")
        if len(query) > MAX_QUERY:
            raise ValueError(f"Keep the search query to {MAX_QUERY} characters or fewer.")
        if source not in {"web", "arxiv"}:
            raise ValueError("Choose web or arxiv as the search source.")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= MAX_RESULTS:
            raise ValueError(f"Result count must be an integer from 1 to {MAX_RESULTS}.")
        if recency not in {"any", *RECENCY_DAYS}:
            raise ValueError("Recency must be any, d, w, m, or y.")
        site = _site(site)
        response["site"] = site
        query = " ".join(query.split())
        response["query"] = query
        if source == "arxiv":
            if site and site not in {"arxiv.org", "www.arxiv.org"}:
                raise ValueError("arXiv searches arxiv.org only. Choose Web to search another site.")
            site = "arxiv.org"
            rows = _arxiv_rows(query, max_results, recency)
            response["warning"] = "arXiv records are preprints; peer-review and publication status are not verified."
        else:
            rows = _web_rows(query + (f" site:{site}" if site else ""), max_results, recency)
            if recency != "any":
                response["warning"] = "Date filtering is supplied by search engines; verify publication dates at the source."
        response["results"] = _normalize(rows, max_results, site)
        if not response["results"]:
            empty = "No matching sources were returned. Try fewer terms or a wider date/site filter."
            response["warning"] = " ".join(part for part in (response.get("warning"), empty) if part)
    except ImportError:
        response["error"] = "Web search needs the optional search package. Install it with: pip install -e .[search]. arXiv search works without it."
    except (ValueError, TypeError) as error:
        response["error"] = _text(str(error), 300)
    except urllib.error.HTTPError as error:
        response["error"] = f"Search service returned HTTP {error.code}. Try again later or choose another source."
    except (TimeoutError, urllib.error.URLError):
        response["error"] = "Search could not connect or timed out. Check the internet connection and try again."
    except ET.ParseError:
        response["error"] = "arXiv returned unreadable data. Try again later."
    except Exception:
        # Backends have their own timeout/rate-limit exception hierarchies.
        # Do not expose their request URLs, proxy settings or opaque traceback.
        response["error"] = "Search is temporarily unavailable or rate limited. Try again later or choose another source."
    return response


__all__ = ["search_internet"]
