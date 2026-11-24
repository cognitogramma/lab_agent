"""Files the researcher hands to the assistant: recognised, summarised, read.

A file dropped on the window (or attached with "Attach a file...", or ``/attach``
at the terminal) goes on the session's :class:`AttachmentShelf`. Three things
happen, none of which costs a model request:

1. **Access.** The exact file is granted to the session through
   :class:`~continuum_spectranalytics.filesystem.FileAccess`, so the normal
   protections (credential files, system folders, links) still apply.
2. **Recognition.** :func:`inspect` works out what the file *is* -- an ARGEN
   acquisition, a processed CSV, a KaleidaGraph file, an Excel workbook, a PDF
   paper, a Word or PowerPoint document, an image, plain text -- and writes a
   short summary of it.
3. **Hand-over.** With the researcher's next message the assistant receives
   that summary, with the file's id and the call that reads it, plus the image
   itself for image files. It does not have to guess the format or pick among
   tools: ``read_attachment`` reads every kind, in parts.

Pictures that the assistant asks to see -- a PDF page, a figure embedded in a
Word or PowerPoint file -- cannot travel back inside a tool result on every
provider, so they are queued here and the orchestrator shows them in a message
straight after the tool round. See :meth:`AttachmentShelf.take_pending_images`.
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from continuum_spectranalytics.filesystem import FileAccess
from continuum_spectranalytics.tools import Rejected

PART_CHARS = 8000
"""Characters per ``read_attachment`` part. Small enough to keep requests quick."""

MAX_ATTACH_BYTES = 200 * 1024 * 1024
MAX_IMAGE_SIDE = 1568
"""Longest image side sent to a model. Larger images are resized: providers
downscale them anyway, and the full-size upload only costs time."""
MAX_IMAGE_BYTES = 3_500_000
AUTO_IMAGES_PER_MESSAGE = 4

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"})
TEXT_SUFFIXES = frozenset({
    ".md", ".markdown", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log",
    ".xml", ".html", ".htm", ".tex", ".bib", ".ris", ".py", ".m", ".r", ".ipynb", ".rtf",
})
DATA_SUFFIXES = frozenset({".csv", ".tsv", ".txt", ".dat", ".asc", ".prn", ".qda", ".exp", ""})
LEGACY_OFFICE = {
    ".doc": "an old-format Word document (.doc). Open it in Word and use File > Save As > "
            "Word Document (.docx), then attach the .docx.",
    ".ppt": "an old-format PowerPoint file (.ppt). Save it as .pptx in PowerPoint and attach that.",
}

KIND_LABELS = {
    "argen": "ARGEN acquisition (raw instrument export)",
    "processed": "processed Continuum CSV",
    "kaleidagraph": "KaleidaGraph data (.QDA)",
    "delimited": "data table",
    "sls_logger": "SLS logger export (polarized/depolarized camera intensities)",
    "excel": "Excel workbook",
    "kaleidagraph_plot": "KaleidaGraph plot (.QPC)",
    "pdf": "PDF document",
    "word": "Word document",
    "powerpoint": "PowerPoint presentation",
    "image": "image",
    "text": "text file",
    "unsupported": "unsupported file",
}


@dataclass(frozen=True)
class ImagePart:
    """One picture on its way to the model, already sized and encoded."""

    media_type: str
    data: str  # base64
    label: str = ""


@dataclass
class Attachment:
    id: str
    path: Path
    size: int
    kind: str
    summary: str
    pages: int = 0  # PDF pages or slides
    figures: int = 0  # embedded pictures (Word, PowerPoint)
    reason: str = ""  # why an unsupported file cannot be read
    sent: bool = False
    _text: str | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def is_table(self) -> bool:
        return self.kind in {"argen", "processed", "kaleidagraph", "delimited", "excel", "sls_logger"}

    def text(self) -> str:
        if self._text is None:
            self._text = _extract_text(self)
        return self._text

    def parts(self) -> int:
        if self.kind in {"image", "unsupported"}:
            return 0
        return max(1, -(-len(self.text()) // PART_CHARS))

    def brief(self) -> str:
        """What the assistant is told about this file when it is handed over."""
        lines = [f"[{self.id}] {self.name} -- {self.label}, {_size(self.size)}",
                 f"  path: {self.path}"]
        lines.extend("  " + line for line in self.summary.splitlines() if line.strip())
        lines.append("  how to use it: " + self.how_to_use())
        return "\n".join(lines)

    def how_to_use(self) -> str:
        ref = f'read_attachment(id="{self.id}")'
        if self.kind == "unsupported":
            return f"it cannot be read: {self.reason}"
        if self.kind == "image":
            return (f"the image is shown to you with this message; {ref} shows it again later.")
        if self.kind == "argen":
            return (f"{ref} for an overview and the numbers; plot_columns(path=..., x=..., y=...) to "
                    "draw any channel. To process it as a measurement (load_run), ask the "
                    "researcher first: it replaces the run that is open.")
        if self.is_table:
            return f"{ref} for an overview and the numbers; plot_columns(path=..., x=..., y=...) to plot."
        if self.kind == "kaleidagraph_plot":
            return f"{ref} for the axis titles, series, and curve fit. It holds no data rows."
        if self.kind == "pdf":
            view = f'; {ref[:-1]}, view=N) shows page N as an image (figures, equations, scans)'
            return f"{ref} reads the text in parts (part=2, 3, ...){view}."
        if self.kind in {"word", "powerpoint"}:
            extra = (f'; {ref[:-1]}, view=N) shows embedded picture N of {self.figures}'
                     if self.figures else "")
            return f"{ref} reads the text and tables in parts{extra}."
        return f"{ref} reads it in parts (part=2, 3, ...)."


class AttachmentShelf:
    """The session's attached files, in the order they arrived."""

    def __init__(self, files: FileAccess) -> None:
        self.files = files
        self._items: dict[str, Attachment] = {}
        self._counter = 0
        self._pending_images: list[ImagePart] = []

    # --- adding and removing -------------------------------------------------

    def add(self, raw: str | Path) -> Attachment:
        """Grant, recognise and keep one file. Raises Rejected with the reason."""
        path = self.files.grant(raw)
        if path.is_dir():
            raise Rejected(
                f"{path.name} is a folder. Attach the files inside it, or drop the folder "
                "only to let the assistant browse it with list_directory."
            )
        for existing in self._items.values():
            if existing.path == path:
                existing.sent = False  # dropped again: hand it over again
                return existing
        size = path.stat().st_size
        if size > MAX_ATTACH_BYTES:
            raise Rejected(f"{path.name} is {_size(size)}; attachments are limited to {_size(MAX_ATTACH_BYTES)}.")
        self._counter += 1
        attachment = inspect(path, f"a{self._counter}", size)
        self._items[attachment.id] = attachment
        return attachment

    def remove(self, attachment_id: str) -> None:
        self._items.pop(attachment_id, None)

    def get(self, attachment_id: str) -> Attachment:
        key = str(attachment_id).strip().strip("[]")
        if key in self._items:
            return self._items[key]
        for item in self._items.values():  # a model sometimes passes the file name
            if item.name == key:
                return item
        known = ", ".join(f"{a.id} ({a.name})" for a in self._items.values()) or "none"
        raise Rejected(f"No attachment {attachment_id!r}. Attached this session: {known}.")

    def __iter__(self):
        return iter(list(self._items.values()))

    def __len__(self) -> int:
        return len(self._items)

    def unsent(self) -> list[Attachment]:
        return [a for a in self._items.values() if not a.sent]

    # --- handing over --------------------------------------------------------

    def take_for_message(self) -> tuple[str, list[ImagePart]]:
        """The block and pictures that go with the researcher's next message."""
        fresh = self.unsent()
        if not fresh:
            return "", []
        images: list[ImagePart] = []
        for attachment in fresh:
            attachment.sent = True
            if attachment.kind == "image" and len(images) < AUTO_IMAGES_PER_MESSAGE:
                try:
                    images.append(_image_part(_load_image(attachment.path), attachment.name))
                except Exception:  # noqa: BLE001 - the brief still says what it is
                    pass
        block = "Files attached to this message (already readable; no permission check needed):\n"
        block += "\n".join(a.brief() for a in fresh)
        return block, images

    def listing(self) -> str:
        if not self._items:
            return "No files are attached. The researcher can drag files onto the window."
        return "\n".join(a.brief() for a in self._items.values())

    # --- reading -------------------------------------------------------------

    def read(self, attachment_id: str, part: int = 1, view: int | None = None) -> str:
        attachment = self.get(attachment_id)
        if attachment.kind == "unsupported":
            raise Rejected(f"{attachment.name} cannot be read: {attachment.reason}")
        if attachment.kind == "image" or view is not None:
            image, label = _view(attachment, view)
            self._pending_images.append(_image_part(image, label))
            return (f"{label}: {image.width}x{image.height} pixels. The picture is shown to you in "
                    "the message right after these tool results.")
        total = attachment.parts()
        if not isinstance(part, int) or part < 1 or part > total:
            raise Rejected(f"{attachment.name} has {total} part(s); ask for part 1 to {total}.")
        chunk = attachment.text()[(part - 1) * PART_CHARS: part * PART_CHARS]
        header = f"{attachment.name} ({attachment.label}) -- part {part} of {total}"
        footer = (f"\n[part {part} of {total}; read_attachment(id=\"{attachment.id}\", part={part + 1}) "
                  "continues]" if part < total else "\n[end of file]")
        return f"{header}\n\n{chunk}{footer}"

    def take_pending_images(self) -> list[ImagePart]:
        images, self._pending_images = self._pending_images, []
        return images


# --- recognising ------------------------------------------------------------


def inspect(path: Path, attachment_id: str = "a1", size: int | None = None) -> Attachment:
    """Work out what a file is and summarise it. Never raises for a bad file."""
    size = path.stat().st_size if size is None else size
    suffix = path.suffix.lower()

    def make(kind: str, summary: str, **extra: Any) -> Attachment:
        return Attachment(id=attachment_id, path=path, size=size, kind=kind, summary=summary, **extra)

    def unsupported(reason: str) -> Attachment:
        return make("unsupported", "", reason=reason)

    try:
        if suffix in LEGACY_OFFICE:
            return unsupported(LEGACY_OFFICE[suffix])
        if suffix in IMAGE_SUFFIXES:
            image = _load_image(path)
            return make("image", f"{image.width}x{image.height} pixels")
        if suffix == ".pdf":
            return _inspect_pdf(path, make)
        if suffix in {".docx", ".docm"}:
            return _inspect_word(path, make)
        if suffix in {".pptx", ".pptm"}:
            return _inspect_powerpoint(path, make)
        if suffix == ".qpc":
            from continuum_spectranalytics.kaleidagraph import read_qpc

            plot = read_qpc(path)
            labels = "; ".join(plot.labels[:8]) or "no recoverable text"
            summary = f"labels: {labels}"
            if plot.fit_equations:
                summary += f"\ncurve fit: {plot.fit_equations[0]}"
            if plot.source_references:
                summary += "\nbuilt from: " + "; ".join(plot.source_references)
            return make("kaleidagraph_plot", summary)
        if suffix in TEXT_SUFFIXES:
            return _inspect_text(path, make)
        from continuum_spectranalytics.tabular import EXCEL_SUFFIXES, UnsupportedFile, open_table

        if suffix in EXCEL_SUFFIXES or suffix in DATA_SUFFIXES:
            try:
                table = open_table(path)
            except UnsupportedFile as error:
                return unsupported(str(error))
            except (ValueError, OSError):
                if suffix in EXCEL_SUFFIXES:
                    raise
                return _inspect_text(path, make)  # prose with a .txt name
            return make(table.kind, _table_summary(table))
        if _looks_like_text(path):
            return _inspect_text(path, make)
        from continuum_spectranalytics.tabular import UNSUPPORTED

        if suffix in UNSUPPORTED:
            return unsupported(UNSUPPORTED[suffix])
        return unsupported(
            f"the {suffix or 'extensionless'} format is not one Continuum reads. Supported: data "
            "exports (CSV/TXT/ARGEN, KaleidaGraph .QDA/.QPC, Excel), PDF, Word .docx, "
            "PowerPoint .pptx, images, and text files."
        )
    except ImportError as error:
        package = getattr(error, "name", None) or str(error)
        return unsupported(f"reading this needs a package that is not installed ({package}). "
                           "Reinstall Continuum with its [files] extra.")
    except Rejected:
        raise
    except Exception as error:  # noqa: BLE001 - a broken file gets a reason, not a crash
        return unsupported(f"it could not be opened ({type(error).__name__}: {str(error)[:200]}). "
                           "It may be damaged or password-protected.")


def _table_summary(table: Any) -> str:
    from continuum_spectranalytics.tabular import column_summary

    columns = table.columns
    shown = ", ".join(columns[:20]) + (f", ... ({len(columns)} in all)" if len(columns) > 20 else "")
    numeric = table.numeric_columns()
    frame = table.frame
    lines = [f"{len(frame):,} rows x {len(columns)} columns: {shown}"]
    if "time_s" in frame and len(frame) > 1:
        span = float(frame["time_s"].iloc[-1] - frame["time_s"].iloc[0])
        lines.append(f"covers {span:,.0f} s ({span / 60:.1f} min)")
        planned = table.metadata.get("record_duration_s")
        try:
            if planned and float(planned) > 1.5 * span:
                lines.append(f"WARNING: recording stopped early: {span:,.0f} s of a planned {float(planned):,.0f} s")
        except ValueError:
            pass
    if numeric and numeric != columns:
        lines.append("plottable: " + ", ".join(numeric[:20]) + (" ..." if len(numeric) > 20 else ""))
    lines.extend(f"note: {note}" for note in table.notes[:10])
    if table.kind != "argen":  # ARGEN already gets a dedicated pipeline and per-channel checks
        lines.extend(column_summary(frame))
    return "\n".join(lines)


def _inspect_pdf(path: Path, make) -> Attachment:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        pages = len(document)
        metadata = document.get_metadata_dict() or {}
        sample = ""
        for index in range(min(pages, 3)):
            textpage = document[index].get_textpage()
            sample += textpage.get_text_range() + "\n"
            if len(sample) > 1500:
                break
    finally:
        document.close()
    title = (metadata.get("Title") or "").strip() or _first_line(sample)
    lines = [f"{pages} page(s)"]
    if title:
        lines.append(f"title: {title[:200]}")
    if metadata.get("Author"):
        lines.append(f"author: {metadata['Author'][:200]}")
    if len(sample.strip()) < 40 * min(pages, 3):
        lines.append("little or no text layer -- probably scanned. View pages as images to read it.")
    else:
        lines.append("opening: " + _squash(sample)[:300])
    return make("pdf", "\n".join(lines), pages=pages)


def _inspect_word(path: Path, make) -> Attachment:
    import docx

    document = docx.Document(str(path))
    headings = [p.text.strip() for p in document.paragraphs
                if p.style is not None and str(p.style.name).lower().startswith(("heading", "title")) and p.text.strip()]
    words = sum(len(p.text.split()) for p in document.paragraphs)
    figures = len(_embedded_images(path, "word/media/"))
    lines = [f"about {words:,} words, {len(document.tables)} table(s), {figures} embedded picture(s)"]
    if headings:
        lines.append("headings: " + "; ".join(headings[:8]))
    else:
        first = next((p.text for p in document.paragraphs if p.text.strip()), "")
        if first:
            lines.append("opening: " + _squash(first)[:300])
    return make("word", "\n".join(lines), figures=figures)


def _inspect_powerpoint(path: Path, make) -> Attachment:
    from pptx import Presentation

    deck = Presentation(str(path))
    titles = []
    for slide in deck.slides:
        title = slide.shapes.title
        if title is not None and title.has_text_frame and title.text_frame.text.strip():
            titles.append(title.text_frame.text.strip())
    figures = len(_embedded_images(path, "ppt/media/"))
    lines = [f"{len(deck.slides)} slide(s), {figures} embedded picture(s)"]
    if titles:
        lines.append("slide titles: " + "; ".join(titles[:10]))
    return make("powerpoint", "\n".join(lines), pages=len(deck.slides), figures=figures)


def _inspect_text(path: Path, make) -> Attachment:
    text = _read_text(path)
    lines = text.count("\n") + 1
    summary = f"{lines:,} lines"
    first = _first_line(text)
    if first:
        summary += f"\nopening: {first[:200]}"
    attachment = make("text", summary)
    attachment._text = text
    return attachment


# --- extracting text ----------------------------------------------------------


def _extract_text(attachment: Attachment) -> str:
    path = attachment.path
    kind = attachment.kind
    if kind == "pdf":
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(path))
        try:
            pieces = []
            for index in range(len(document)):
                text = document[index].get_textpage().get_text_range().strip()
                pieces.append(f"--- page {index + 1} ---\n{text or '[no text on this page; view it as an image]'}")
        finally:
            document.close()
        return "\n\n".join(pieces)
    if kind == "word":
        return _word_text(path)
    if kind == "powerpoint":
        return _powerpoint_text(path)
    if kind == "kaleidagraph_plot":
        return attachment.summary
    if kind == "excel":
        from continuum_spectranalytics.tabular import read_excel_sheets

        pieces = []
        for name, frame in read_excel_sheets(path).items():
            frame = frame.dropna(how="all").dropna(axis=1, how="all")
            pieces.append(f"--- sheet {name!r}: {len(frame)} rows x {frame.shape[1]} columns ---\n"
                          + frame.to_csv(index=False))
        return "\n\n".join(pieces)
    if attachment.is_table:
        from continuum_spectranalytics.tabular import open_table

        table = open_table(path)
        return (attachment.summary + "\n\nfirst rows:\n" + table.preview(15)
                + "\n\nall rows as CSV:\n" + table.frame.to_csv(index=False))
    return _read_text(path)


def _word_text(path: Path) -> str:
    """Paragraphs and tables in document order, headings marked with #."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    out: list[str] = []
    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(element, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = str(paragraph.style.name).lower() if paragraph.style is not None else ""
            if style.startswith("heading"):
                level = re.sub(r"\D", "", style) or "1"
                out.append("#" * min(int(level), 6) + " " + text)
            elif style == "title":
                out.append("# " + text)
            else:
                out.append(text)
        elif tag == "tbl":
            table = Table(element, document)
            rows = []
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
            out.append("[table]\n" + "\n".join(rows))
    return "\n\n".join(out)


def _powerpoint_text(path: Path) -> str:
    from pptx import Presentation

    deck = Presentation(str(path))
    out = []
    for number, slide in enumerate(deck.slides, start=1):
        pieces = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                pieces.append(shape.text_frame.text.strip())
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    pieces.append("| " + " | ".join(c.text.strip() for c in row.cells) + " |")
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                pieces.append(f"[speaker notes] {notes}")
        out.append(f"--- slide {number} ---\n" + ("\n".join(pieces) or "[no text]"))
    return "\n\n".join(out)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp1252", errors="replace")


def _looks_like_text(path: Path) -> bool:
    with path.open("rb") as stream:
        head = stream.read(4096)
    if not head:
        return True
    if b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return False
    printable = sum(32 <= b < 127 or b in (9, 10, 13) or b >= 128 for b in head)
    return printable / len(head) > 0.9


# --- pictures -------------------------------------------------------------------


def _load_image(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        image.seek(0)  # first frame of a GIF or multi-page TIFF
        image.load()
        return image.copy()


def _view(attachment: Attachment, view: int | None):
    """The picture to show: the image itself, a PDF page, or an embedded figure."""
    from PIL import Image

    if attachment.kind == "image":
        return _load_image(attachment.path), attachment.name
    if not isinstance(view, int) or view < 1:
        raise Rejected("view must be a page or picture number starting at 1.")
    if attachment.kind == "pdf":
        if view > attachment.pages:
            raise Rejected(f"{attachment.name} has {attachment.pages} page(s).")
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(attachment.path))
        try:
            page = document[view - 1]
            width, height = page.get_size()
            scale = min(2.0, MAX_IMAGE_SIDE / max(width, height, 1))
            image = page.render(scale=scale).to_pil()
        finally:
            document.close()
        return image, f"{attachment.name}, page {view}"
    if attachment.kind in {"word", "powerpoint"}:
        prefix = "word/media/" if attachment.kind == "word" else "ppt/media/"
        members = _embedded_images(attachment.path, prefix)
        if not members:
            raise Rejected(f"{attachment.name} has no embedded pictures.")
        if view > len(members):
            raise Rejected(f"{attachment.name} has {len(members)} embedded picture(s).")
        import zipfile

        with zipfile.ZipFile(attachment.path) as archive:
            data = archive.read(members[view - 1])
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as error:  # noqa: BLE001 - EMF/WMF vector figures
            raise Rejected(
                f"picture {view} is in a format that cannot be shown ({Path(members[view - 1]).suffix}); "
                "often a vector chart. Ask the researcher for a PNG export of it."
            ) from error
        return image, f"{attachment.name}, picture {view}"
    raise Rejected(f"{attachment.name} is a {attachment.label}; there is nothing to view as an image. "
                   "Use part= to read it.")


def _embedded_images(path: Path, prefix: str) -> list[str]:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.startswith(prefix)]
    except zipfile.BadZipFile:
        return []
    return sorted(names, key=lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)])


def _image_part(image, label: str) -> ImagePart:
    """Resize and encode for a model: PNG for plots and line art, JPEG if too big."""
    from PIL import Image

    if max(image.size) > MAX_IMAGE_SIDE:
        image = image.copy()
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    media_type = "image/png"
    if buffer.tell() > MAX_IMAGE_BYTES:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        media_type = "image/jpeg"
    return ImagePart(media_type, base64.b64encode(buffer.getvalue()).decode("ascii"), label)


# --- small helpers ----------------------------------------------------------------


def _size(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} bytes"


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return _squash(line)
    return ""


__all__ = ["Attachment", "AttachmentShelf", "ImagePart", "PART_CHARS", "inspect"]
