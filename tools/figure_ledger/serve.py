"""Figure labelling tool. Python standard library only.

    python tools/figure_ledger/serve.py /path/to/figures

Opens a browser on a local page that shows one image at a time with a labelling
form, and writes ``ledger.json`` into the image folder.

**Why a local server rather than a single HTML file.** A single file using the
File System Access API would have been the nicer answer, and it is the first
thing I tried to justify. It does not work: ``showDirectoryPicker`` is gated
behind a secure context, and a page opened by double-clicking an HTML file runs
on ``file://``, where the call throws. The workarounds all end in serving the
page from somewhere -- which is a server. It is also Chromium-only, and decoding
Netpbm in the browser would mean shipping a decoder in JavaScript to work around
a problem Python solves in forty lines.

So: a server, but a small one. No dependencies, no build step, no framework. It
serves exactly three things -- the page, the images, and the ledger -- and shuts
down cleanly on Ctrl-C after flushing to disk.

The page arrives with its data already in it. Nothing is fetched after load, so
the first paint has content rather than an empty shell that fills in later and
looks broken while it does.
"""

from __future__ import annotations

import argparse
import html
import json
import signal
import socketserver
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import images as image_tools  # noqa: E402
import ledger as ledger_store  # noqa: E402

# Extensions worth opening. The extension is only a filter for what to *try* --
# every candidate is then identified by its magic bytes, because these files
# routinely lie about what they are.
CANDIDATE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff",
    ".ppm", ".pgm", ".pbm", ".pnm", ".pam",
}

TEMPLATE = Path(__file__).resolve().parent / "app.html"


class Store:
    """The folder, its images, and the ledger. Guarded by a lock.

    ``http.server`` handles each request on its own thread, so two saves racing
    could interleave a read-modify-write and lose one of them.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder.resolve()
        self.lock = threading.Lock()
        self.records = ledger_store.load(self.folder)
        self.images = self._scan()
        self.dirty = False

    def _scan(self) -> list[dict[str, object]]:
        """Every image under the folder, recursively, identified by content."""
        found: list[dict[str, object]] = []
        for path in sorted(self.folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in CANDIDATE_SUFFIXES:
                continue
            relative = path.relative_to(self.folder).as_posix()
            try:
                identified = image_tools.identify(image_tools.read_header(path))
            except OSError as error:
                identified = image_tools.Identified(
                    "unreadable", displayable=False, detail=str(error)
                )
            found.append(
                {
                    "image": relative,
                    "filename": path.name,
                    "format": identified.format,
                    "displayable": identified.displayable,
                    "note": identified.detail,
                }
            )
        return found

    def resolve(self, relative: str) -> Path:
        """Resolve a request path inside the folder, refusing to escape it."""
        candidate = (self.folder / unquote(relative)).resolve()
        if candidate != self.folder and self.folder not in candidate.parents:
            raise PermissionError(f"{relative!r} is outside the image folder")
        if not candidate.is_file():
            raise FileNotFoundError(relative)
        return candidate

    def record_for(self, image: str) -> dict[str, object]:
        return self.records.get(image) or ledger_store.blank_record(image)

    def put(self, image: str, payload: dict) -> dict[str, object]:
        with self.lock:
            record = ledger_store.normalise(payload, image)
            self.records[image] = record
            ledger_store.save(self.folder, self.records)
            self.dirty = False
        return record

    def flush(self) -> Path | None:
        """Write on the way out, so a Ctrl-C mid-edit does not lose the last save."""
        with self.lock:
            if not self.records:
                return None
            return ledger_store.save(self.folder, self.records)


def build_page(store: Store) -> bytes:
    """Render the page with the folder state already embedded in it."""
    template = TEMPLATE.read_text(encoding="utf-8")
    state = {
        "folder": str(store.folder),
        "images": store.images,
        "records": {
            entry["image"]: store.record_for(str(entry["image"]))
            for entry in store.images
        },
    }
    # Escaping the closing tag keeps a filename containing "</script>" from
    # ending the block early. json.dumps alone does not do this.
    payload = json.dumps(state, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__INITIAL_STATE__", payload).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    store: Store

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # noqa: D102 -- quiet by default
        pass

    def handle_one_request(self) -> None:
        """Serve one request, treating a dropped connection as ordinary.

        Browsers close keep-alive sockets whenever they feel like it, and the
        default handler prints a full traceback when they do. A researcher
        watching this terminal should see a stack trace only when something is
        actually wrong.
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802 -- required name
        route = urlparse(self.path).path

        if route in ("/", "/index.html"):
            self._send(HTTPStatus.OK, build_page(self.store), "text/html; charset=utf-8")
            return

        if route == "/export.csv":
            body = ledger_store.to_csv(self.store.records).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header(
                "Content-Disposition", 'attachment; filename="ledger.csv"'
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route.startswith("/image/"):
            self._serve_image(route[len("/image/") :])
            return

        self._json(HTTPStatus.NOT_FOUND, {"error": f"no route {route}"})

    def _serve_image(self, relative: str) -> None:
        try:
            path = self.store.resolve(relative)
        except PermissionError as error:
            self._json(HTTPStatus.FORBIDDEN, {"error": str(error)})
            return
        except FileNotFoundError:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"no such image: {relative}"})
            return

        try:
            body, content_type, _found = image_tools.to_displayable(
                path.read_bytes()
            )
        except ValueError as error:
            # The message names the detected format, which is the actionable part.
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": str(error)})
            return
        except OSError as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            return

        self._send(HTTPStatus.OK, body, content_type)

    def do_POST(self) -> None:  # noqa: N802 -- required name
        route = urlparse(self.path).path
        if route != "/save":
            self._json(HTTPStatus.NOT_FOUND, {"error": f"no route {route}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            image = str(payload.get("image") or "")
            if not image:
                raise ledger_store.ValidationError("no image named in the request")
            record = self.store.put(image, payload.get("record") or {})
        except ledger_store.ValidationError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except (json.JSONDecodeError, ValueError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": f"bad request: {error}"})
            return
        except OSError as error:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"could not write the ledger: {error}"},
            )
            return

        self._json(HTTPStatus.OK, {"saved": True, "record": record})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def open_browser(url: str) -> None:
    """Open a browser, and carry on if that fails.

    ``webbrowser.open`` returns False rather than raising when it cannot find
    one, and on Linux it can also fail asynchronously in a subprocess. Either
    way the tool is still usable -- the URL is printed -- so a failure here must
    never take the server down with it.
    """
    try:
        if not webbrowser.open(url):
            print(f"  (could not open a browser automatically -- visit {url})")
    except Exception as error:  # any backend, any platform
        print(f"  (could not open a browser: {error} -- visit {url})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="figure-ledger", description=__doc__.splitlines()[0]
    )
    parser.add_argument("folder", help="Folder of figure images. Subfolders included.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 2

    try:
        store = Store(folder)
    except ledger_store.ValidationError as error:
        print(f"Could not start: {error}")
        return 2

    Handler.store = store
    displayable = sum(1 for entry in store.images if entry["displayable"])
    labelled = sum(1 for r in store.records.values() if ledger_store.is_logged(r))

    print(f"\nFigure ledger  --  {folder}")
    print(f"  {len(store.images)} image(s), {displayable} displayable")
    print(f"  {len(store.records)} existing record(s), {labelled} already logged")
    print(f"  ledger: {folder / ledger_store.LEDGER_NAME}")

    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as error:
        print(f"\nCould not listen on port {args.port}: {error}")
        print("Another copy may already be running. Try --port 8766.")
        return 2

    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  open {url}")
    print("  press Ctrl-C when you are done\n")

    def shutdown(*_args) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    if not args.no_browser:
        threading.Timer(0.4, open_browser, args=(url,)).start()

    try:
        server.serve_forever()
    finally:
        server.server_close()
        written = store.flush()
        if written:
            print(f"\nSaved {len(store.records)} record(s) to {written}")
        else:
            print("\nNothing to save.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
