"""Public destination checks and page extraction against fake DNS and sockets."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import socket
import time

import pytest

from polyrmc import web_reader as reader


PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def no_live_network(monkeypatch):
    def public_dns(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC_IP, port))]

    def no_socket(*args, **kwargs):
        raise AssertionError("A test attempted a live socket")

    monkeypatch.setattr(reader.socket, "getaddrinfo", public_dns)
    monkeypatch.setattr(reader.socket, "socket", no_socket)


def mock_pages(monkeypatch, pages):
    calls = []

    def fetch(target, addresses, deadline):
        calls.append((target, addresses))
        return pages[len(calls) - 1]

    monkeypatch.setattr(reader, "_fetch", fetch)
    return calls


def html_page(text):
    return 200, {"content-type": "text/html; charset=utf-8"}, text.encode()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "javascript:alert(1)", "ftp://example.org/file", "https://",
    "http://user:password@example.org/", "http://@example.org/", "http://example.org:8080/",
    "http://example.org:0/", "https://example.org/\nattack", "https://example.org/with space",
    "https://example.org\\@other.org/", "http://localhost/", "http://a.local/", "http://metadata.google.internal/",
    "http://127.0.0.1/", "http://10.1.1.1/", "http://172.16.0.2/", "http://192.168.1.1/",
    "http://169.254.169.254/", "http://100.100.100.200/", "http://0.0.0.0/", "http://224.0.0.1/",
    "http://192.0.2.10/", "http://[::1]/", "http://[fc00::1]/", "http://[fe80::1]/",
    "http://[::ffff:127.0.0.1]/", "http://[ff02::1]/",
])
def test_nonpublic_or_unsafe_url_is_refused_before_a_connection(url, monkeypatch):
    calls = mock_pages(monkeypatch, [html_page("<main>Should not be read</main>")])
    result = reader.read_web_page(url)
    assert result["error"]
    assert result["text"] == ""
    assert result["untrusted"] is True
    assert calls == []


@pytest.mark.parametrize("addresses", [["127.0.0.1"], [PUBLIC_IP, "10.0.0.1"], ["::1"], ["169.254.169.254"]])
def test_dns_rejects_any_private_answer_including_mixed_answers(monkeypatch, addresses):
    monkeypatch.setattr(reader.socket, "getaddrinfo", lambda host, port, **kwargs: [
        (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
         "", (ip, port, 0, 0) if ":" in ip else (ip, port)) for ip in addresses
    ])
    calls = mock_pages(monkeypatch, [html_page("<main>Should not be read</main>")])
    result = reader.read_web_page("https://example.org/")
    assert "private or reserved" in result["error"]
    assert calls == []


def test_redirect_destination_is_validated_before_following(monkeypatch):
    calls = mock_pages(monkeypatch, [(302, {"location": "http://169.254.169.254/latest/meta-data/"}, b"")])
    result = reader.read_web_page("https://example.org/")
    assert result["error"]
    assert len(calls) == 1


def test_redirect_hostname_is_resolved_again_and_cannot_rebind_private(monkeypatch):
    answers = iter([PUBLIC_IP, "127.0.0.1"])
    monkeypatch.setattr(reader.socket, "getaddrinfo", lambda host, port, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (next(answers), port))
    ])
    calls = mock_pages(monkeypatch, [(302, {"location": "/next"}, b"")])
    result = reader.read_web_page("https://example.org/")
    assert "private or reserved" in result["error"]
    assert len(calls) == 1


def test_redirect_limit_is_bounded(monkeypatch):
    calls = mock_pages(monkeypatch, [(302, {"location": "/again"}, b"")] * 5)
    result = reader.read_web_page("https://example.org/")
    assert "redirected too many" in result["error"]
    assert len(calls) == 5


def test_html_prefers_main_and_removes_chrome_hidden_elements_and_active_links(monkeypatch):
    body = '''<html><head><title> A useful &amp; precise paper </title><style>bad css</style></head>
    <body><nav>Site navigation <a href="/navigation">Menu</a></nav><p>Outside main</p>
    <main><h1>Polymer scattering</h1><p>Research <strong>evidence</strong> matters.</p>
    <script>steal_credentials()</script><aside>Advertisement</aside>
    <p hidden>Hidden data</p><p aria-hidden="true">Also hidden</p><div style="display:none">Hidden style</div>
    <p><a href="/paper#one">Full paper</a> and <a href="/paper#two">Same paper</a></p>
    <a href="javascript:alert(1)">Bad link</a><a href="file:///secret">Local link</a>
    <a href="http://127.0.0.1/">Private link</a></main><footer>Copyright navigation</footer></body></html>'''
    mock_pages(monkeypatch, [html_page(body)])
    result = reader.read_web_page("https://example.org/abstract")
    assert result["title"] == "A useful & precise paper"
    assert "Research evidence matters." in result["text"]
    for omitted in ("Site navigation", "Outside main", "steal_credentials", "Advertisement", "Hidden", "Copyright"):
        assert omitted not in result["text"]
    assert result["links"] == [{"title": "Full paper", "url": "https://example.org/paper"}]
    assert result["sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert result["retrieved_at"].endswith("+00:00")
    assert result["untrusted"] is True


def test_article_is_used_when_no_main_is_present(monkeypatch):
    mock_pages(monkeypatch, [html_page("<p>Outside</p><article><h2>Paper</h2><p>Readable source.</p></article>")])
    result = reader.read_web_page("https://example.org/paper")
    assert result["text"] == "Paper\n\nReadable source."


def test_document_title_excludes_accessibility_titles_in_body_svg(monkeypatch):
    body = '''<html><head><title>Optical scattering from surfaces | NIST</title></head>
    <body><header><svg><title>Lock</title><path/></svg></header>
    <main><h1>Optical scattering from surfaces</h1><p>Research programme.</p>
    <svg><title>Download icon</title></svg></main></body></html>'''
    mock_pages(monkeypatch, [html_page(body)])
    result = reader.read_web_page("https://www.nist.gov/programs-projects/optical-scattering-surfaces")
    assert result["title"] == "Optical scattering from surfaces | NIST"
    assert "Lock" not in result["text"]
    assert "Download icon" not in result["text"]


def test_fragment_title_ignores_preceding_hidden_embedded_titles(monkeypatch):
    body = "<svg><title>Icon</title></svg><template><title>Template</title></template><title>Page title</title><main>Public source.</main>"
    mock_pages(monkeypatch, [html_page(body)])
    result = reader.read_web_page("https://example.org/")
    assert result["title"] == "Page title"


def test_final_url_and_requested_url_are_distinct_after_redirect(monkeypatch):
    calls = mock_pages(monkeypatch, [(301, {"location": "/paper?ref=signed%20value#section"}, b""),
                                      html_page("<main>Useful source.</main>")])
    result = reader.read_web_page("https://example.org/old")
    assert result["requested_url"] == "https://example.org/old"
    assert result["url"] == "https://example.org/paper?ref=signed%20value"
    assert len(calls) == 2


def test_text_is_truncated_explicitly_and_source_hash_remains_complete(monkeypatch):
    payload = b"An entire source with much more text."
    mock_pages(monkeypatch, [(200, {"content-type": "text/plain"}, payload)])
    result = reader.read_web_page("http://example.org/", max_chars=9)
    assert result["text"] == "An entire"
    assert result["truncated"] is True
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_json_is_readable_and_charset_is_honoured(monkeypatch):
    payload = json.dumps({"author": "René"}, ensure_ascii=False).encode("iso-8859-1")
    mock_pages(monkeypatch, [(200, {"content-type": "application/json; charset=iso-8859-1"}, payload)])
    result = reader.read_web_page("https://example.org/paper.json")
    assert json.loads(result["text"]) == {"author": "René"}
    assert result["content_type"] == "application/json"


def test_html_meta_charset_and_gzip_are_supported(monkeypatch):
    payload = '<meta charset="windows-1252"><main>Un café scientifique.</main>'.encode("cp1252")
    mock_pages(monkeypatch, [(200, {"content-type": "text/html", "content-encoding": "gzip"}, gzip.compress(payload))])
    result = reader.read_web_page("https://example.org/")
    assert result["text"] == "Un café scientifique."
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_gzip_expansion_is_limited(monkeypatch):
    payload = gzip.compress(b"x" * (reader.MAX_BYTES + 1))
    mock_pages(monkeypatch, [(200, {"content-type": "text/plain", "content-encoding": "gzip"}, payload)])
    result = reader.read_web_page("https://example.org/")
    assert "decompressed page exceeds" in result["error"]


@pytest.mark.parametrize("kind,payload,fragment", [
    ("application/pdf", b"%PDF-1.7 fake", "PDF"),
    ("image/png", b"PNG fake", "Binary files"),
    ("text/plain", b"\x00\x01\x02\x03", "binary data"),
    ("text/html", b"%PDF-1.7 fake", "PDF or binary"),
    ("application/json", b"{not json", "invalid"),
])
def test_binary_pdf_and_invalid_json_have_helpful_errors(monkeypatch, kind, payload, fragment):
    mock_pages(monkeypatch, [(200, {"content-type": kind}, payload)])
    result = reader.read_web_page("https://example.org/")
    assert fragment in result["error"]
    assert result["text"] == ""


def test_empty_script_only_page_explains_browser_requirement(monkeypatch):
    mock_pages(monkeypatch, [html_page("<script>render_app()</script>")])
    result = reader.read_web_page("https://example.org/")
    assert "JavaScript" in result["error"]


def test_deeply_nested_html_is_rejected_before_unbounded_stack_scans(monkeypatch):
    mock_pages(monkeypatch, [html_page("<div>" * 1000 + "Text" + "</div>" * 1000)])
    result = reader.read_web_page("https://example.org/")
    assert "HTML nesting exceeds" in result["error"]


def test_html_parser_observes_its_deadline_during_events():
    parser = reader._PageParser("https://example.org/", deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        parser.feed("<main>" + "<p>Text</p>" * 200)


def test_timeout_returns_an_error_without_backend_details(monkeypatch):
    def timeout(*args):
        raise TimeoutError("internal request details")

    monkeypatch.setattr(reader, "_fetch", timeout)
    result = reader.read_web_page("https://example.org/")
    assert "timed out" in result["error"]
    assert "internal" not in result["error"]


@pytest.mark.parametrize("value", [0, 24001, True, "100", 1.5])
def test_invalid_text_limits_are_refused_before_network(monkeypatch, value):
    calls = mock_pages(monkeypatch, [])
    result = reader.read_web_page("https://example.org/", max_chars=value)
    assert "max_chars" in result["error"]
    assert calls == []


class FakeSocket:
    def __init__(self, response):
        self.response = response
        self.connected = []
        self.sent = []
        self.timeouts = []
        self.closed = False
        self.shutdowns = []

    def settimeout(self, value):
        self.timeouts.append(value)

    def connect(self, address):
        self.connected.append(address)

    def sendall(self, data):
        self.sent.append(data)

    def makefile(self, mode):
        return io.BytesIO(self.response)

    def close(self):
        self.closed = True

    def shutdown(self, how):
        self.shutdowns.append(how)


def wire_response(body=b"Public source text", content_type="text/plain", extra=b""):
    return b"HTTP/1.1 200 OK\r\nContent-Type: " + content_type.encode() + b"\r\nContent-Length: " + str(len(body)).encode() + b"\r\n" + extra + b"Connection: close\r\n\r\n" + body


def test_actual_http_transport_connects_only_to_the_validated_numeric_ip(monkeypatch):
    dns_calls = []

    def dns(host, port, **kwargs):
        dns_calls.append(host)
        # If code wrongly resolves again, the second answer is private.
        ip = PUBLIC_IP if len(dns_calls) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]

    fake = FakeSocket(wire_response())
    monkeypatch.setattr(reader.socket, "getaddrinfo", dns)
    monkeypatch.setattr(reader.socket, "socket", lambda *args: fake)
    result = reader.read_web_page("http://example.org/paper")
    assert "error" not in result
    assert result["text"] == "Public source text"
    assert dns_calls == ["example.org"]
    assert fake.connected == [(PUBLIC_IP, 80)]
    request = b"".join(fake.sent)
    assert b"GET /paper HTTP/1.1\r\n" in request
    assert b"Host: example.org\r\n" in request
    assert b"Cookie:" not in request
    assert b"Authorization:" not in request
    assert fake.closed


def test_https_pins_ip_but_verifies_and_sends_the_original_hostname(monkeypatch):
    fake = FakeSocket(wire_response())
    tls_calls = []

    class VerifiedContext:
        def wrap_socket(self, sock, server_hostname):
            tls_calls.append((sock, server_hostname))
            return sock

    contexts = []

    def default_context():
        contexts.append(True)
        return VerifiedContext()

    monkeypatch.setattr(reader.socket, "socket", lambda *args: fake)
    monkeypatch.setattr(reader.ssl, "create_default_context", default_context)
    result = reader.read_web_page("https://example.org/paper")
    assert "error" not in result
    assert fake.connected == [(PUBLIC_IP, 443)]
    assert contexts == [True]
    assert tls_calls == [(fake, "example.org")]
    assert b"Host: example.org\r\n" in b"".join(fake.sent)


def test_transport_rejects_oversized_response_before_reading_body(monkeypatch):
    fake = FakeSocket(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: "
                      + str(reader.MAX_BYTES + 1).encode() + b"\r\n\r\n")
    monkeypatch.setattr(reader.socket, "socket", lambda *args: fake)
    result = reader.read_web_page("http://example.org/")
    assert "2 MiB" in result["error"]
    assert fake.closed


def test_deadline_can_shutdown_a_detached_connection_close_transport():
    connection = reader._PinnedConnection(reader._target("http://example.org/"),
                                           (socket.AF_INET, (PUBLIC_IP, 80)), time.monotonic() + 15)
    fake = FakeSocket(b"")
    connection.transport = fake
    connection.sock = None  # HTTPConnection has detached it after headers.
    reader._abort(connection)
    assert fake.shutdowns == [socket.SHUT_RDWR]
