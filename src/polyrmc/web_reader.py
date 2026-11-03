"""Read a bounded public page without credentials, proxies or browser state.

Every destination (including redirects) is resolved and checked before a socket
is created. The socket connects to that exact numeric address; HTTPS still
verifies the original hostname and supplies it as TLS SNI. Returned prose is
external, untrusted material, never an instruction to the agent.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import queue
import re
import socket
import ssl
import threading
import time
import urllib.parse
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from html.parser import HTMLParser
from typing import Any

TIMEOUT = 15.0
MAX_BYTES = 2 * 1024 * 1024
MAX_CHARS = 24000
MAX_LINKS = 30
MAX_REDIRECTS = 4
MAX_HTML_DEPTH = 128
_dns_slots = threading.BoundedSemaphore(4)
_REDIRECTS = {301, 302, 303, 307, 308}


class _ReaderError(ValueError):
    pass


@dataclass(frozen=True)
class _Target:
    url: str
    host: str
    port: int
    secure: bool
    path: str


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    return remaining


def _public(address: str) -> bool:
    try:
        value = ipaddress.ip_address(address)
    except ValueError:
        return False
    if not value.is_global or any((value.is_private, value.is_loopback, value.is_link_local,
                                  value.is_multicast, value.is_reserved, value.is_unspecified)):
        return False
    if isinstance(value, ipaddress.IPv6Address):
        if value.is_site_local or (value.ipv4_mapped and not _public(str(value.ipv4_mapped))):
            return False
    return True


def _target(url: str) -> _Target:
    if not isinstance(url, str) or not 1 <= len(url) <= 4096:
        raise _ReaderError("Supply an HTTP or HTTPS URL of at most 4096 characters.")
    if re.search(r"[\x00-\x20\x7f\\]", url):
        raise _ReaderError("The URL contains spaces, control characters or backslashes.")
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise _ReaderError("Only public HTTP and HTTPS pages can be read.")
        if parsed.username is not None or parsed.password is not None:
            raise _ReaderError("URLs containing usernames or passwords are not supported.")
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        secure = parsed.scheme.lower() == "https"
        port = parsed.port if parsed.port is not None else (443 if secure else 80)
        if port not in {80, 443}:
            raise _ReaderError("Only web ports 80 and 443 are supported.")
        if not host or "%" in host:
            raise _ReaderError("Supply a valid public hostname.")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if (len(host) > 253 or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part)
                                           for part in host.split("."))):
                raise _ReaderError("Supply a valid public hostname.")
            if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home")):
                raise _ReaderError("Local and private network destinations cannot be read.")
        else:
            if not _public(host):
                raise _ReaderError("Local, private and reserved network addresses cannot be read.")
        netloc = f"[{host}]" if ":" in host else host
        default_port = 443 if secure else 80
        if port != default_port:
            netloc += f":{port}"
        path = urllib.parse.quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
        query = urllib.parse.quote(parsed.query, safe="=&?/:@+%!$'()*,-._~;")
        normalized = urllib.parse.urlunsplit(("https" if secure else "http", netloc, path, query, ""))
        request_path = path + ("?" + query if query else "")
        return _Target(normalized, host, port, secure, request_path)
    except (UnicodeError, ValueError) as error:
        if isinstance(error, _ReaderError):
            raise
        raise _ReaderError("Supply a valid public HTTP or HTTPS URL.") from error


def _resolve(target: _Target, deadline: float) -> list[tuple[int, tuple]]:
    try:
        literal = ipaddress.ip_address(target.host)
    except ValueError:
        literal = None
    if literal is not None:
        family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
        address = (str(literal), target.port, 0, 0) if literal.version == 6 else (str(literal), target.port)
        return [(family, address)]
    if not _dns_slots.acquire(timeout=_remaining(deadline)):
        raise TimeoutError
    result: queue.Queue = queue.Queue(maxsize=1)

    def lookup():
        try:
            result.put(socket.getaddrinfo(target.host, target.port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP))
        except Exception as error:
            result.put(error)
        finally:
            _dns_slots.release()

    # DNS is bounded too; an OS resolver that stalls must not hold the worker
    # or process shutdown indefinitely. At most four such lookups can exist.
    threading.Thread(target=lookup, name="polyrmc-web-dns", daemon=True).start()
    try:
        records = result.get(timeout=_remaining(deadline))
    except queue.Empty as error:
        raise TimeoutError from error
    if isinstance(records, Exception):
        raise _ReaderError("The public hostname could not be resolved. Check the URL and connection.")
    addresses = []
    for family, _kind, _proto, _name, address in records:
        if family not in {socket.AF_INET, socket.AF_INET6} or not _public(address[0]):
            raise _ReaderError("The hostname resolves to a private or reserved network address and cannot be read.")
        pinned = (address[0], target.port, 0, 0) if family == socket.AF_INET6 else (address[0], target.port)
        entry = (family, pinned)
        if entry not in addresses:
            addresses.append(entry)
    if not addresses:
        raise _ReaderError("No public address was found for this hostname.")
    return addresses[:8]


class _PinnedConnection(http.client.HTTPConnection):
    def __init__(self, target: _Target, address: tuple[int, tuple], deadline: float):
        self.default_port = 443 if target.secure else 80
        super().__init__(target.host, target.port, timeout=_remaining(deadline))
        self.target = target
        self.address = address
        self.deadline = deadline
        self.transport = None

    def connect(self):
        family, address = self.address
        raw = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.sock = raw
        self.transport = raw
        try:
            raw.settimeout(_remaining(self.deadline))
            raw.connect(address)  # Numeric, already checked: no second DNS lookup.
            if self.target.secure:
                context = ssl.create_default_context()
                raw.settimeout(_remaining(self.deadline))
                self.sock = context.wrap_socket(raw, server_hostname=self.target.host)
                self.transport = self.sock
        except BaseException:
            raw.close()
            self.sock = None
            raise


def _abort(connection: _PinnedConnection):
    # HTTPConnection detaches a connection-close socket after response headers;
    # its makefile remains alive during the body read. Retain that transport so
    # the deadline can still interrupt a peer slowly trickling the body.
    sock = connection.transport
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    connection.close()


def _fetch(target: _Target, addresses: list[tuple[int, tuple]], deadline: float):
    last_error = None
    for address in addresses:
        connection = _PinnedConnection(target, address, deadline)
        timer = threading.Timer(_remaining(deadline), _abort, args=(connection,))
        timer.daemon = True
        timer.start()
        response = None
        try:
            connection.request("GET", target.path, headers={
                "User-Agent": "PolyRMC/0.1 (public research page reader)",
                "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9",
                "Accept-Encoding": "gzip, identity", "Connection": "close",
            })
            if connection.transport is not None:
                connection.transport.settimeout(_remaining(deadline))
            response = connection.getresponse()
            headers = {key.lower(): value for key, value in response.getheaders()}
            status = response.status
            if status in _REDIRECTS or status < 200 or status >= 300:
                return status, headers, b""
            declared = headers.get("content-length", "")
            if declared.isdigit() and int(declared) > MAX_BYTES:
                raise _ReaderError("This page exceeds the 2 MiB response limit. Read a smaller page.")
            chunks = []
            length = 0
            while True:
                if connection.transport is not None:
                    connection.transport.settimeout(_remaining(deadline))
                chunk = response.read1(min(65536, MAX_BYTES + 1 - length))
                _remaining(deadline)
                if not chunk:
                    break
                length += len(chunk)
                if length > MAX_BYTES:
                    raise _ReaderError("This page exceeds the 2 MiB response limit. Read a smaller page.")
                chunks.append(chunk)
            return status, headers, b"".join(chunks)
        except TimeoutError:
            raise
        except (OSError, http.client.HTTPException) as error:
            last_error = error
            _remaining(deadline)
        finally:
            timer.cancel()
            if response is not None:
                response.close()
            connection.close()
    if isinstance(last_error, ssl.SSLError):
        raise _ReaderError("The site's HTTPS certificate or secure connection could not be verified.")
    raise _ReaderError("The public page could not be reached. Check the connection or try another source.")


def _decode_payload(payload: bytes, headers: dict[str, str]) -> tuple[str, str, bytes]:
    encoding = headers.get("content-encoding", "identity").strip().lower()
    if encoding == "gzip":
        try:
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            decoded = decompressor.decompress(payload, MAX_BYTES + 1)
            if len(decoded) > MAX_BYTES or decompressor.unconsumed_tail:
                raise _ReaderError("The decompressed page exceeds the 2 MiB limit.")
            decoded += decompressor.flush(MAX_BYTES + 1 - len(decoded))
            if len(decoded) > MAX_BYTES:
                raise _ReaderError("The decompressed page exceeds the 2 MiB limit.")
            if not decompressor.eof or decompressor.unused_data:
                raise _ReaderError("The site returned incomplete or unsupported compressed data.")
            payload = decoded
        except zlib.error as error:
            raise _ReaderError("The site returned unreadable compressed data.") from error
    elif encoding not in {"", "identity"}:
        raise _ReaderError("The site uses an unsupported content encoding. Try another page.")
    message = Message()
    message["content-type"] = headers.get("content-type", "application/octet-stream")
    content_type = message.get_content_type().lower()
    supported = content_type in {"text/html", "application/xhtml+xml", "text/plain", "text/markdown", "application/json"} or content_type.endswith("+json")
    if not supported:
        if content_type == "application/pdf" or payload.startswith(b"%PDF-"):
            raise _ReaderError("PDF reading is not supported by this page tool. Use the paper's HTML abstract or a local PDF reader.")
        raise _ReaderError("This URL is not a supported HTML, plain-text or JSON page. Binary files cannot be read here.")
    if payload.startswith(b"%PDF-"):
        raise _ReaderError("The response contains PDF or binary data rather than readable page text.")
    charset = message.get_content_charset()
    if not charset and content_type in {"text/html", "application/xhtml+xml"}:
        match = re.search(br"<meta\b[^>]*\bcharset\s*=\s*['\"]?\s*([a-zA-Z0-9._-]+)", payload[:8192], re.I)
        if match:
            charset = match.group(1).decode("ascii")
    try:
        text = payload.decode(charset or "utf-8-sig", errors="replace")
    except LookupError:
        text = payload.decode("utf-8-sig", errors="replace")
    sample = text[:2048]
    controls = sum(ord(char) < 32 and char not in "\n\r\t" for char in sample)
    if "\x00" in sample or controls > max(2, len(sample) // 100):
        raise _ReaderError("The response contains binary data rather than readable page text.")
    return text, content_type, payload


class _PageParser(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg", "template", "iframe", "object", "button", "select", "textarea", "head"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    BREAK = {"p", "div", "section", "article", "main", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br", "blockquote", "pre", "hr"}

    def __init__(self, base: str, deadline: float | None = None):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.deadline = deadline
        self._events = 0
        self.stack: list[tuple[str, bool]] = []
        self.parts: dict[str, list[str]] = {"all": [], "main": [], "article": []}
        self.links: dict[str, list[dict]] = {"all": [], "main": [], "article": []}
        self.title: list[str] = []
        self._title_finished = False
        self.anchor = None

    def _tick(self):
        self._events += 1
        if self.deadline is not None and self._events % 128 == 1:
            _remaining(self.deadline)

    def _scopes(self):
        tags = {tag for tag, _skip in self.stack}
        return ["all"] + [scope for scope in ("main", "article") if scope in tags]

    def _hidden(self):
        return any(skip for _tag, skip in self.stack)

    def _document_title_active(self):
        tags = {tag for tag, _skip in self.stack}
        return (not self._title_finished and "title" in tags
                and not tags.intersection({"svg", "math", "template", "iframe", "object", "embed"}))

    def _append(self, text: str):
        for scope in self._scopes():
            self.parts[scope].append(text)

    def handle_starttag(self, tag, attributes):
        self._tick()
        attrs = {key: value or "" for key, value in attributes}
        hidden = (tag in self.SKIP or "hidden" in attrs or attrs.get("aria-hidden", "").lower() == "true"
                  or re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", attrs.get("style", ""), re.I))
        if tag not in self.VOID:
            if len(self.stack) >= MAX_HTML_DEPTH:
                raise _ReaderError("The page's HTML nesting exceeds the supported limit. Try a simpler source page.")
            self.stack.append((tag, bool(hidden)))
        if self._hidden() or hidden:
            return
        if tag in self.BREAK:
            self._append("\n")
        if tag == "a" and attrs.get("href"):
            self.anchor = {"href": attrs["href"], "parts": [], "scopes": self._scopes()}

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        self._tick()
        if tag == "title" and self._document_title_active():
            self._title_finished = True
        if tag == "a" and self.anchor is not None:
            anchor, self.anchor = self.anchor, None
            try:
                target = _target(urllib.parse.urljoin(self.base, anchor["href"]))
            except (ValueError, TypeError):
                pass
            else:
                title = _inline(" ".join(anchor["parts"]))[:180]
                if title:
                    for scope in anchor["scopes"]:
                        if len(self.links[scope]) < MAX_LINKS * 3:
                            self.links[scope].append({"title": title, "url": target.url})
        if tag in self.BREAK and not self._hidden():
            self._append("\n")
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self._tick()
        if self._document_title_active():
            self.title.append(data)
        if self._hidden():
            return
        self._append(data)
        if self.anchor is not None:
            self.anchor["parts"].append(data)

    def extract(self):
        scope = next((key for key in ("main", "article", "all") if _prose("".join(self.parts[key]))), "all")
        text = _prose("".join(self.parts[scope]))
        links, seen = [], set()
        for link in self.links[scope]:
            if link["url"] not in seen:
                links.append(link)
                seen.add(link["url"])
            if len(links) >= MAX_LINKS:
                break
        return _inline(" ".join(self.title))[:300], text, links


def _inline(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _prose(value: str) -> str:
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in value.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def read_web_page(url: str, max_chars: int = 12000) -> dict[str, Any]:
    """Retrieve public prose, with a SHA-256 of the decompressed source bytes.

    The 15-second budget covers DNS, redirects, connection and response reads.
    No cookies, authentication, proxies or page scripts are used. A returned
    link is only a reference; following it requires another validated read.
    """
    requested = url if isinstance(url, str) else ""
    result: dict[str, Any] = {
        "url": requested[:4096], "requested_url": requested[:4096], "title": "", "text": "", "links": [],
        "content_type": "", "retrieved_at": None, "sha256": None, "truncated": False, "untrusted": True,
    }
    deadline = time.monotonic() + TIMEOUT
    try:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= MAX_CHARS:
            raise _ReaderError(f"max_chars must be an integer from 1 to {MAX_CHARS}.")
        target = _target(url)
        for redirect in range(MAX_REDIRECTS + 1):
            result["url"] = target.url
            addresses = _resolve(target, deadline)
            status, headers, payload = _fetch(target, addresses, deadline)
            if status in _REDIRECTS:
                if redirect == MAX_REDIRECTS:
                    raise _ReaderError("This page redirected too many times. Use its final public URL.")
                location = headers.get("location")
                if not location:
                    raise _ReaderError("The site redirected without supplying another URL.")
                target = _target(urllib.parse.urljoin(target.url, location))
                continue
            if status < 200 or status >= 300:
                raise _ReaderError(f"The site returned HTTP {status}. The page may be unavailable or require browser access.")
            result["content_type"] = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            decoded, content_type, raw = _decode_payload(payload, headers)
            if content_type in {"text/html", "application/xhtml+xml"}:
                parser = _PageParser(target.url, deadline=deadline)
                parser.feed(decoded)
                parser.close()
                title, text, links = parser.extract()
            else:
                title, links = target.host, []
                if content_type == "application/json" or content_type.endswith("+json"):
                    try:
                        text = json.dumps(json.loads(decoded), ensure_ascii=False, indent=2)
                    except (ValueError, RecursionError):
                        raise _ReaderError("The site returned invalid or overly nested JSON.")
                else:
                    text = _prose(decoded)
            if not text:
                raise _ReaderError("No readable page text was found. The site may require JavaScript, sign-in or another page URL.")
            _remaining(deadline)
            result.update(title=title or target.host, text=text[:max_chars], links=links,
                          content_type=content_type, retrieved_at=datetime.now(timezone.utc).isoformat(),
                          sha256=hashlib.sha256(raw).hexdigest(), truncated=len(text) > max_chars)
            return result
    except _ReaderError as error:
        result["error"] = str(error)
    except (TimeoutError, socket.timeout):
        result["error"] = "The page read timed out. Try a smaller or different public page."
    except (OSError, http.client.HTTPException, UnicodeError):
        result["error"] = "The page could not be retrieved securely. Check the URL and connection or try another source."
    except Exception:
        result["error"] = "The page could not be read. Try another public HTML or text source."
    return result


__all__ = ["read_web_page"]
