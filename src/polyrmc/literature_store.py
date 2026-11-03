"""A bounded local reading list containing source metadata, never page bodies.

All instances share an in-process lock. Writes replace a complete JSON file in
the same directory, so a failed save leaves the preceding library intact.
Reading a missing library creates nothing and performs no network lookups.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, unquote_plus, urlencode, urlsplit, urlunsplit

MAX_ENTRIES = 500
MAX_BYTES = 2 * 1024 * 1024
MAX_TITLE = 300
MAX_SNIPPET = 1200
MAX_AUTHORS = 20
MAX_RELEVANCE = 12
MAX_MEMORY_REFS = 20
_LOCK = threading.RLock()
_TRACKING = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid"}
_SCALARS = {"published": 80, "snippet": MAX_SNIPPET, "origin": 80,
            "query": 500, "run_id": 160, "annotation": 2000}
_FIELDS = {"id", "title", "url", "domain", "authors", "relevance", "memory_refs", *_SCALARS}
_STORED_FIELDS = _FIELDS | {"saved_at", "updated_at"}


class ReadingLibraryError(ValueError):
    """An invalid save or unreadable library, with a useful recovery message."""


def _text(value: Any, field: str, limit: int, *, clip: bool = False) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReadingLibraryError(f"{field} must be text.")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        if clip:
            return text[:limit - 1] + "…"
        raise ReadingLibraryError(f"{field} must be {limit} characters or fewer.")
    return text


def _public_address(host: str) -> bool:
    address = ipaddress.ip_address(host)
    if not address.is_global or any((address.is_private, address.is_loopback,
                                    address.is_link_local, address.is_multicast,
                                    address.is_reserved, address.is_unspecified)):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.is_site_local or (address.ipv4_mapped and not _public_address(str(address.ipv4_mapped))):
            return False
    return True


def _url(value: Any) -> tuple[str, str, str]:
    """Return a safe canonical URL, hostname, and stable deduplication key.

    Hostnames are checked syntactically, without DNS. A later page read still
    validates its resolved address. Non-tracking query bytes are preserved so
    signed source URLs are not re-encoded.
    """
    message = "Save a public HTTP or HTTPS source URL without credentials or local addresses."
    if not isinstance(value, str) or not 1 <= len(value) <= 4096:
        raise ReadingLibraryError(message)
    if re.search(r"[\x00-\x20\x7f\\]", value):
        raise ReadingLibraryError(message)
    try:
        parts = urlsplit(value)
        if (parts.scheme.lower() not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None):
            raise ReadingLibraryError(message)
        host = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        port = parts.port
        if port is not None and port not in {80, 443}:
            raise ReadingLibraryError("Saved source URLs support web ports 80 and 443 only.")
        if not host or "%" in host:
            raise ReadingLibraryError(message)
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if (len(host) > 253 or "." not in host or not re.search(r"[a-z]", host.rsplit(".", 1)[-1])
                    or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                               for label in host.split("."))
                    or host.endswith((".localhost", ".local", ".internal", ".lan", ".home"))):
                raise ReadingLibraryError(message)
        else:
            if not _public_address(host):
                raise ReadingLibraryError(message)
        scheme = parts.scheme.lower()
        netloc = f"[{host}]" if ":" in host else host
        if port is not None and (scheme, port) not in {("http", 80), ("https", 443)}:
            netloc += f":{port}"
        components = [component for component in parts.query.split("&")
                      if (key := unquote_plus(component.partition("=")[0]).lower()) not in _TRACKING
                      and not key.startswith("utm_")]
        query = "&".join(components)
        path = parts.path or "/"
        canonical = urlunsplit((scheme, netloc, path, query, ""))
        key = json.dumps((netloc.removeprefix("www."), path.rstrip("/"),
                          urlencode(sorted(parse_qsl(query, keep_blank_values=True)))),
                         ensure_ascii=True, separators=(",", ":"))
        return canonical, host, key
    except (UnicodeError, ValueError) as error:
        if isinstance(error, ReadingLibraryError):
            raise
        raise ReadingLibraryError(message) from error


def _strings(value: Any, field: str, count: int, length: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or len(value) > count:
        raise ReadingLibraryError(f"{field} must contain at most {count} text entries.")
    return list(dict.fromkeys(text for item in value if (text := _text(item, field, length))))


def _memory_refs(value: Any) -> list[dict[str, str] | str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or len(value) > MAX_MEMORY_REFS:
        raise ReadingLibraryError(f"memory_refs must contain at most {MAX_MEMORY_REFS} references.")
    refs: list[dict[str, str] | str] = []
    for item in value:
        if isinstance(item, str):
            text = _text(item, "memory_refs", 240)
            if text:
                refs.append(text)
        elif isinstance(item, dict):
            ref = {key: text for key, length in (("kind", 40), ("id", 160), ("title", 300), ("run_id", 160))
                   if key in item and (text := _text(item[key], f"memory_refs.{key}", length))}
            if not any(ref.get(key) for key in ("id", "title", "run_id")):
                raise ReadingLibraryError("Each memory reference needs an id, title or run_id.")
            refs.append(ref)
        else:
            raise ReadingLibraryError("memory_refs must contain text or reference metadata objects.")
    return refs


def _metadata(paper: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if not isinstance(paper, dict):
        raise ReadingLibraryError("Choose a paper or source to save.")
    url, domain, key = _url(paper.get("url"))
    title = _text(paper.get("title"), "title", MAX_TITLE)
    if not title:
        raise ReadingLibraryError("A saved source needs a title.")
    result: dict[str, Any] = {
        "id": "paper-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20],
        "title": title, "url": url, "domain": domain,
        "authors": _strings(paper.get("authors"), "authors", MAX_AUTHORS, 120),
        "relevance": _strings(paper.get("relevance"), "relevance", MAX_RELEVANCE, 500),
        "memory_refs": _memory_refs(paper.get("memory_refs")),
    }
    for field, limit in _SCALARS.items():
        if field in paper:
            result[field] = _text(paper[field], field, limit, clip=field == "snippet")
    result.setdefault("origin", "saved")
    result.setdefault("published", "")
    result.setdefault("snippet", "")
    return result, key


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 40:
        raise ReadingLibraryError("Saved timestamps must be ISO dates with a timezone.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError
    except ValueError as error:
        raise ReadingLibraryError("Saved timestamps must be ISO dates with a timezone.") from error
    return value


class ReadingLibrary:
    """Persistent source metadata, shared safely by this process's GUI workers."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / "reading-list.json"

    def list(self) -> list[dict[str, Any]]:
        """Read saved papers without creating directories or modifying files."""
        with _LOCK:
            return self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("rb") as source:
                payload = source.read(MAX_BYTES + 1)
        except FileNotFoundError:
            return []
        except OSError as error:
            raise ReadingLibraryError(f"Cannot read {self.path}. Check its permissions and try again.") from error
        if len(payload) > MAX_BYTES:
            raise ReadingLibraryError(f"{self.path} exceeds the {MAX_BYTES // 1024} KiB reading-list limit. Preserve a copy and reduce its size before saving.")
        try:
            document = json.loads(payload.decode("utf-8"))
            if (not isinstance(document, dict) or set(document) != {"version", "papers"}
                    or type(document["version"]) is not int or document["version"] != 1
                    or not isinstance(document["papers"], list) or len(document["papers"]) > MAX_ENTRIES):
                raise ValueError("Unsupported reading-list structure")
            papers, seen = [], set()
            for raw in document["papers"]:
                if not isinstance(raw, dict) or set(raw) - _STORED_FIELDS:
                    raise ValueError("Unsupported source metadata")
                paper, key = _metadata(raw)
                if raw.get("id") != paper["id"] or key in seen:
                    raise ValueError("Invalid or duplicate source id")
                paper["saved_at"] = _timestamp(raw.get("saved_at"))
                paper["updated_at"] = _timestamp(raw.get("updated_at", paper["saved_at"]))
                papers.append(paper)
                seen.add(key)
            return papers
        except (UnicodeError, ValueError, TypeError, RecursionError) as error:
            raise ReadingLibraryError(f"{self.path} is malformed or unsupported. No data was changed; restore a valid copy or choose another reading-list directory.") from error

    def save(self, paper: dict[str, Any]) -> dict[str, Any]:
        """Add or update one source, preserving its original saved timestamp."""
        clean, key = _metadata(paper)
        with _LOCK:
            papers = self._load()
            now = datetime.now(timezone.utc).isoformat()
            for index, previous in enumerate(papers):
                if _url(previous["url"])[2] == key:
                    for field in _FIELDS - {"id", "title", "url", "domain"}:
                        if field not in paper and field in previous:
                            clean[field] = previous[field]
                    clean["saved_at"] = previous["saved_at"]
                    clean["updated_at"] = now
                    papers[index] = clean
                    break
            else:
                if len(papers) >= MAX_ENTRIES:
                    raise ReadingLibraryError(f"The reading list already holds {MAX_ENTRIES} sources. Use another library directory to save more.")
                clean.update(saved_at=now, updated_at=now)
                papers.insert(0, clean)
            self._write(papers)
            return clean

    def _write(self, papers: list[dict[str, Any]]) -> None:
        payload = (json.dumps({"version": 1, "papers": papers}, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if len(payload) > MAX_BYTES:
            raise ReadingLibraryError(f"The reading list would exceed {MAX_BYTES // 1024} KiB. Shorten annotations or use another library directory; no data was changed.")
        temporary: Path | None = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.directory, prefix=".reading-list-", suffix=".tmp", delete=False) as target:
                temporary = Path(target.name)
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
        except OSError as error:
            raise ReadingLibraryError(f"Cannot save {self.path}. Check folder permissions or close any program holding the file; the previous reading list was preserved.") from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


__all__ = ["ReadingLibrary", "ReadingLibraryError"]
