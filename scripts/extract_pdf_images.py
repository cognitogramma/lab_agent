"""Extract images from a PDF.

Two extraction paths:

  embedded  Pull the raster XObjects stored in the file. Lossless: the
            compressed stream is written out exactly as stored, unless an
            alpha mask or an odd colorspace forces a decode.

  render    Rasterize whole pages. Needed for vector figures (matplotlib,
            Illustrator), which contain no embedded image at all.

  figures   Find each figure's bounding box by clustering the page's ink,
            then clip-render just that box. Gives cropped figures rather
            than whole pages, and works for vector and raster alike.

Default mode is `auto`: embedded extraction, plus a rendered page for any
page that yielded nothing.

The input may be a single PDF or a directory. Given a directory, every PDF
in it is processed into its own subfolder, plus one combined manifest.

Usage:
    python scripts/extract_pdf_images.py paper.pdf -o figures/
    python scripts/extract_pdf_images.py paper.pdf --mode render --dpi 600
    python scripts/extract_pdf_images.py paper.pdf --mode figures --dpi 600
    python scripts/extract_pdf_images.py paper.pdf --pages 3-7,12
    python scripts/extract_pdf_images.py C:\\papers --recursive
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pymupdf

# Raw stream formats that are valid standalone files. Anything else
# (jpx, jb2, raw separations) goes through the pixmap decode path.
DIRECT_WRITE_EXTS = {"png", "jpeg", "jpg", "tiff", "tif", "bmp", "gif"}

# Occupancy-grid resolution for figure detection, in points. Finer than this
# buys nothing: the gap tolerance, not the cell size, sets the precision.
CELL_PT = 2.0

# Lines closer than this are one run of text. Wide enough to join wrapped
# lines (~1pt apart), tight enough to keep a two-column gutter (~18pt) apart.
TEXT_GAP_PT = 6.0

# A run of text longer than this is prose, not an axis label or legend entry.
# This is what stops body copy being pulled into a figure by proximity.
MAX_LABEL_CHARS = 120

CAPTION_RE = re.compile(
    r"^\s*\(?(fig(?:ure)?|table|scheme|chart|plate|exhibit|panel)\s*\.?\s*"
    r"(\d+|[ivxlcdm]+|[a-z])\b",
    re.IGNORECASE,
)


@dataclass
class Record:
    """One row of the manifest."""

    page: int
    index: int
    xref: int
    path: str
    width: int
    height: int
    bpc: int
    colorspace: str
    has_alpha: bool
    ext: str
    bytes: int
    placed_w_pt: float = 0.0
    placed_h_pt: float = 0.0
    dpi_x: float = 0.0
    dpi_y: float = 0.0
    source: str = "embedded"
    note: str = ""
    bbox: str = ""
    caption: str = ""
    pdf: str = ""


@dataclass
class Stats:
    kept: int = 0
    skipped_small: int = 0
    skipped_duplicate: int = 0
    rendered_pages: int = 0
    figures: int = 0
    skipped_text_panel: int = 0
    pages_scanned: int = 0
    failed: list[str] = field(default_factory=list)


def parse_pages(spec: str | None, n_pages: int) -> list[int]:
    """Parse "1-5,8,11-" into 0-based page indices."""
    if not spec:
        return list(range(n_pages))

    wanted: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            start = int(lo) if lo.strip() else 1
            end = int(hi) if hi.strip() else n_pages
        else:
            start = end = int(chunk)
        wanted.update(range(start - 1, end))

    return sorted(p for p in wanted if 0 <= p < n_pages)


def placement(page: pymupdf.Page, xref: int, px_w: int, px_h: int) -> tuple[float, float, float, float]:
    """Size of the image on the page, and the DPI that implies.

    An image placed more than once reports its first placement; the effective
    DPI is what matters and it is the same for identical rects.
    """
    try:
        rects = page.get_image_rects(xref)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0
    if not rects:
        return 0.0, 0.0, 0.0, 0.0

    r = rects[0]
    w_pt, h_pt = r.width, r.height
    dpi_x = px_w / (w_pt / 72.0) if w_pt > 0 else 0.0
    dpi_y = px_h / (h_pt / 72.0) if h_pt > 0 else 0.0
    return round(w_pt, 2), round(h_pt, 2), round(dpi_x, 1), round(dpi_y, 1)


def save_via_pixmap(doc: pymupdf.Document, xref: int, smask: int, out_base: Path) -> tuple[Path, bytes, bool]:
    """Decode to a pixmap, recombine alpha, convert to RGB, save PNG.

    Used when the stored bytes would not stand alone as a file: soft-masked
    images (dumping raw loses the transparency and figures come out on a
    black field) and CMYK / JPX / separation colorspaces.
    """
    pix = pymupdf.Pixmap(doc, xref)

    if pix.colorspace is not None and pix.colorspace.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

    has_alpha = False
    if smask:
        if pix.alpha:  # Pixmap(pix, mask) requires an alpha-free base
            pix = pymupdf.Pixmap(pix, 0)
        pix = pymupdf.Pixmap(pix, pymupdf.Pixmap(doc, smask))
        has_alpha = True

    data = pix.tobytes("png")
    out = out_base.with_suffix(".png")
    out.write_bytes(data)
    return out, data, has_alpha


def extract_embedded(
    doc: pymupdf.Document,
    pages: list[int],
    outdir: Path,
    min_size: int,
    min_bytes: int,
    stats: Stats,
) -> tuple[list[Record], set[int]]:
    """Walk the pages and pull out every distinct raster XObject."""
    records: list[Record] = []
    seen_xrefs: set[int] = set()
    seen_hashes: dict[str, str] = {}
    productive_pages: set[int] = set()

    for pno in pages:
        page = doc[pno]
        for idx, info in enumerate(page.get_images(full=True)):
            xref, smask = info[0], info[1]
            width, height, bpc = info[2], info[3], info[4]
            cs_name = info[5] or "?"

            # Same object referenced from several pages: extract once.
            if xref in seen_xrefs:
                stats.skipped_duplicate += 1
                productive_pages.add(pno)
                continue
            seen_xrefs.add(xref)

            # Rules, spacers and gradient strips are technically images.
            # Both tests matter: a 2000x3 divider passes a width-only check.
            if width < min_size or height < min_size:
                stats.skipped_small += 1
                continue

            base = outdir / f"p{pno + 1:03d}_i{idx:02d}_x{xref}"
            try:
                meta = doc.extract_image(xref)
                ext = (meta.get("ext") or "").lower()
                data = meta.get("image") or b""

                if smask or ext not in DIRECT_WRITE_EXTS:
                    out, data, has_alpha = save_via_pixmap(doc, xref, smask, base)
                    ext = "png"
                    note = "decoded: alpha mask" if smask else f"decoded: {meta.get('ext')}"
                else:
                    out = base.with_suffix(f".{ext}")
                    out.write_bytes(data)
                    has_alpha = False
                    note = ""
            except Exception as exc:  # keep going; report at the end
                stats.failed.append(f"page {pno + 1} xref {xref}: {exc}")
                continue

            if len(data) < min_bytes:
                out.unlink(missing_ok=True)
                stats.skipped_small += 1
                continue

            # Second dedupe pass: identical picture stored under two xrefs.
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes:
                out.unlink(missing_ok=True)
                stats.skipped_duplicate += 1
                productive_pages.add(pno)
                continue
            seen_hashes[digest] = out.name

            w_pt, h_pt, dpi_x, dpi_y = placement(page, xref, width, height)
            records.append(
                Record(
                    page=pno + 1,
                    index=idx,
                    xref=xref,
                    path=out.name,
                    width=width,
                    height=height,
                    bpc=bpc,
                    colorspace=cs_name,
                    has_alpha=has_alpha,
                    ext=ext,
                    bytes=len(data),
                    placed_w_pt=w_pt,
                    placed_h_pt=h_pt,
                    dpi_x=dpi_x,
                    dpi_y=dpi_y,
                    note=note,
                )
            )
            stats.kept += 1
            productive_pages.add(pno)

    return records, productive_pages


def render_pages(
    doc: pymupdf.Document,
    pages: list[int],
    outdir: Path,
    dpi: int,
    stats: Stats,
) -> list[Record]:
    """Rasterize whole pages. The only way to capture vector figures."""
    records: list[Record] = []

    for pno in pages:
        page = doc[pno]
        try:
            pix = page.get_pixmap(dpi=dpi)
            out = outdir / f"p{pno + 1:03d}_render_{dpi}dpi.png"
            data = pix.tobytes("png")
            out.write_bytes(data)
        except Exception as exc:
            stats.failed.append(f"render page {pno + 1}: {exc}")
            continue

        rect = page.rect
        records.append(
            Record(
                page=pno + 1,
                index=0,
                xref=-1,
                path=out.name,
                width=pix.width,
                height=pix.height,
                bpc=8,
                colorspace="RGB",
                has_alpha=False,
                ext="png",
                bytes=len(data),
                placed_w_pt=round(rect.width, 2),
                placed_h_pt=round(rect.height, 2),
                dpi_x=float(dpi),
                dpi_y=float(dpi),
                source="render",
                note="full page raster",
            )
        )
        stats.rendered_pages += 1

    return records


# --------------------------------------------------------------------------
# Figure-region detection
#
# A "figure" is a connected island of non-body-text ink. The page's vector
# paths and placed images are painted into a coarse occupancy grid, the grid
# is dilated so that neighbouring marks bridge (a scatter plot is thousands
# of disjoint dots; an axis is separate from its tick labels), and the
# connected components of that dilated mask are the figures. Text blocks
# sitting inside a component -- axis labels, legends, panel letters -- get
# absorbed and the components recomputed, since absorbing text can merge two
# components that share an annotation.
# --------------------------------------------------------------------------


def graphic_rects(page: pymupdf.Page, keep_rules: bool) -> list[pymupdf.Rect]:
    """Vector-path and placed-image rects, minus obvious page furniture."""
    page_rect = page.rect
    page_area = page_rect.get_area() or 1.0
    rects: list[pymupdf.Rect] = []

    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []

    for d in drawings:
        r = pymupdf.Rect(d["rect"]) & page_rect
        if r.is_empty and r.width <= 0 and r.height <= 0:
            continue

        # A full-bleed fill is the page background, not a figure.
        if r.get_area() >= 0.90 * page_area:
            continue

        # Header/footer rules and column separators: hairlines spanning much
        # of the page. Dropping them stops a rule from welding a figure to
        # the running head. Axis lines are shorter than half a page.
        if not keep_rules:
            if r.height <= 2.5 and r.width >= 0.5 * page_rect.width:
                continue
            if r.width <= 2.5 and r.height >= 0.5 * page_rect.height:
                continue

        rects.append(r)

    for info in page.get_images(full=True):
        try:
            for r in page.get_image_rects(info[0]):
                r = pymupdf.Rect(r) & page_rect
                if not r.is_empty:
                    rects.append(r)
        except Exception:
            continue

    return rects


def text_units(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
    """Text grouped into visually contiguous runs, as (rect, text).

    PyMuPDF's own blocks are not usable here: it will put two side-by-side
    captions, or a running head and its page number, into one block because
    they share a baseline -- which then hands one figure both captions and
    the next figure none. So the block structure is discarded down to lines,
    and lines are re-agglomerated by proximity. Wrapped lines of one caption
    sit ~1pt apart and merge; a caption in the facing column is ~180pt away
    and does not.
    """
    lines: list[tuple[pymupdf.Rect, str]] = []
    try:
        data = page.get_text("dict")
    except Exception:
        return []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if text.strip():
                lines.append((pymupdf.Rect(line["bbox"]), text.strip()))

    if not lines:
        return []

    canvas = _Canvas(page.rect)
    units: list[tuple[pymupdf.Rect, str]] = []
    for group in _cluster([r for r, _t in lines], canvas, TEXT_GAP_PT):
        ordered = sorted(group, key=lambda i: (round(lines[i][0].y0, 1), lines[i][0].x0))

        # A caption typically sits a few points below the plot's axis label,
        # which is close enough to cluster with it. Proximity cannot tell
        # them apart, so force a break at every line that opens a caption:
        # "Figure 3." is the start of a new unit no matter what precedes it.
        chunks: list[list[int]] = []
        for i in ordered:
            if not chunks or CAPTION_RE.match(lines[i][1]):
                chunks.append([i])
            else:
                chunks[-1].append(i)

        for chunk in chunks:
            rect = _union([lines[i][0] for i in chunk])
            units.append((rect, " ".join(lines[i][1] for i in chunk)))

    return units


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Grow the mask by k cells in each direction (8-connected)."""
    out = mask
    for _ in range(k):
        nxt = out.copy()
        nxt[:-1, :] |= out[1:, :]
        nxt[1:, :] |= out[:-1, :]
        nxt[:, :-1] |= out[:, 1:]
        nxt[:, 1:] |= out[:, :-1]
        out = nxt
    return out


def _label(mask: np.ndarray) -> np.ndarray:
    """8-connected component labelling. Zero means background."""
    height, width = mask.shape
    labels = np.zeros((height, width), np.int32)
    current = 0
    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    for start in np.argwhere(mask):
        sy, sx = int(start[0]), int(start[1])
        if labels[sy, sx]:
            continue
        current += 1
        labels[sy, sx] = current
        stack = [(sy, sx)]
        while stack:
            y, x = stack.pop()
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    stack.append((ny, nx))

    return labels


class _Canvas:
    """Maps page points to grid cells and back."""

    def __init__(self, page_rect: pymupdf.Rect):
        self.x0, self.y0 = page_rect.x0, page_rect.y0
        self.width = max(1, int(np.ceil(page_rect.width / CELL_PT)))
        self.height = max(1, int(np.ceil(page_rect.height / CELL_PT)))

    def cells(self, r: pymupdf.Rect) -> tuple[int, int, int, int]:
        c0 = int((r.x0 - self.x0) // CELL_PT)
        c1 = int((r.x1 - self.x0) // CELL_PT)
        r0 = int((r.y0 - self.y0) // CELL_PT)
        r1 = int((r.y1 - self.y0) // CELL_PT)
        c0 = min(max(c0, 0), self.width - 1)
        c1 = min(max(c1, 0), self.width - 1)
        r0 = min(max(r0, 0), self.height - 1)
        r1 = min(max(r1, 0), self.height - 1)
        return r0, c0, r1, c1

    def blank(self) -> np.ndarray:
        return np.zeros((self.height, self.width), bool)


def _cluster(rects: list[pymupdf.Rect], canvas: _Canvas, gap_pt: float) -> list[list[int]]:
    """Group rect indices into connected islands."""
    if not rects:
        return []

    mask = canvas.blank()
    spans = []
    for r in rects:
        r0, c0, r1, c1 = canvas.cells(r)
        mask[r0:r1 + 1, c0:c1 + 1] = True
        spans.append((r0, c0))

    # Dilation grows *both* sides of a gap, so half the tolerance each way
    # makes two islands meet at exactly gap_pt. Grid discretisation makes
    # this accurate to about one cell.
    k = max(1, int(round(gap_pt / (2 * CELL_PT))))
    labels = _label(_dilate(mask, k))

    groups: dict[int, list[int]] = {}
    for i, (r0, c0) in enumerate(spans):
        lab = int(labels[r0, c0])
        if lab:
            groups.setdefault(lab, []).append(i)

    return list(groups.values())


def _union(rects: list[pymupdf.Rect]) -> pymupdf.Rect:
    box = pymupdf.Rect(rects[0])
    for r in rects[1:]:
        box |= r
    return box


def _contained(inner: pymupdf.Rect, outer: pymupdf.Rect) -> float:
    """Fraction of `inner` that lies within `outer`."""
    area = inner.get_area()
    if area <= 0:
        return 1.0 if outer.contains(inner) else 0.0
    return ((inner & outer).get_area()) / area


def _belongs_to_figure(
    rect: pymupdf.Rect, text: str, box: pymupdf.Rect, label_gap: float
) -> bool:
    """Is this run of text part of the figure in `box`?

    Containment alone is not enough. When a plot is stored as a bitmap of
    just the plot area -- very common in typeset papers -- the tick values,
    axis titles and legend are drawn as PDF text *beside* that bitmap, so
    they are 0% contained and would be left behind, giving a bare plot frame
    with no numbers on it.

    So text just outside the box also counts, provided it lines up with the
    box's rows or columns (an axis label runs alongside the plot; a sentence
    in the next column does not) and is short. The length cap is what keeps
    body text out: `text_units` merges a paragraph into one long run, while
    a tick value or legend entry stays well under the limit.
    """
    if _contained(rect, box) >= 0.5:
        return True
    if len(text) > MAX_LABEL_CHARS:
        return False

    halo = pymupdf.Rect(box) + (-label_gap, -label_gap, label_gap, label_gap)
    if (rect & halo).is_empty:
        return False

    h_overlap = min(box.x1, rect.x1) - max(box.x0, rect.x0)
    v_overlap = min(box.y1, rect.y1) - max(box.y0, rect.y0)
    return h_overlap > 0.25 * rect.width or v_overlap > 0.25 * rect.height


def find_captions(
    boxes: list[pymupdf.Rect],
    blocks: list[tuple[pymupdf.Rect, str]],
    max_gap: float,
) -> list[tuple[str, pymupdf.Rect | None]]:
    """Match each figure box to the nearest caption-looking text block.

    Looks below first (the convention for figures), then above (tables).
    Requires horizontal overlap so a caption in the other column is not
    stolen.
    """
    candidates = [(r, t) for r, t in blocks if CAPTION_RE.match(t)]
    results: list[tuple[str, pymupdf.Rect | None]] = []
    taken: set[int] = set()

    for box in boxes:
        best: tuple[float, int] | None = None
        for i, (rect, _text) in enumerate(candidates):
            if i in taken:
                continue

            overlap = min(box.x1, rect.x1) - max(box.x0, rect.x0)
            if overlap <= 0.3 * min(box.width, rect.width):
                continue

            if rect.y0 >= box.y1 - 2:
                distance = rect.y0 - box.y1
            elif rect.y1 <= box.y0 + 2:
                distance = (box.y0 - rect.y1) + 0.5  # tie-break toward below
            else:
                continue

            if 0 <= distance <= max_gap and (best is None or distance < best[0]):
                best = (distance, i)

        if best is None:
            results.append(("", None))
        else:
            taken.add(best[1])
            rect, text = candidates[best[1]]
            results.append((text, rect))

    return results


def find_figure_regions(
    page: pymupdf.Page,
    blocks: list[tuple[pymupdf.Rect, str]],
    gap_pt: float,
    min_pt: float,
    keep_rules: bool,
    label_gap: float,
) -> list[tuple[pymupdf.Rect, int, int, float]]:
    """Return (bbox, n_graphics, n_absorbed_text, text_area_fraction) per figure."""
    graphics = graphic_rects(page, keep_rules)
    if not graphics:
        return []

    canvas = _Canvas(page.rect)
    members = list(graphics)
    absorbed: set[int] = set()
    groups = _cluster(members, canvas, gap_pt)

    # Absorbing text can bridge two components (a shared legend, a panel
    # label between subplots), so re-cluster until the set stops growing.
    for _ in range(3):
        boxes = [_union([members[i] for i in g]) for g in groups]
        added = False
        for bi, (rect, text) in enumerate(blocks):
            # A caption describes the figure; it is not part of it. Absorbing
            # one would both inflate the crop and consume the text that
            # identifies the figure downstream.
            if bi in absorbed or CAPTION_RE.match(text):
                continue
            if any(_belongs_to_figure(rect, text, box, label_gap) for box in boxes):
                absorbed.add(bi)
                members.append(rect)
                added = True
        if not added:
            break
        groups = _cluster(members, canvas, gap_pt)

    n_graphics = len(graphics)
    out: list[tuple[pymupdf.Rect, int, int, float]] = []
    for g in groups:
        box = _union([members[i] for i in g])
        if box.width < min_pt or box.height < min_pt:
            continue
        graphics_in = sum(1 for i in g if i < n_graphics)
        text_in = len(g) - graphics_in
        text_area = sum(members[i].get_area() for i in g if i >= n_graphics)
        text_frac = text_area / (box.get_area() or 1.0)
        out.append((box, graphics_in, text_in, min(text_frac, 1.0)))

    out.sort(key=lambda item: (round(item[0].y0, 1), round(item[0].x0, 1)))
    return out


def extract_figures(
    doc: pymupdf.Document,
    pages: list[int],
    outdir: Path,
    dpi: int,
    gap_pt: float,
    min_pt: float,
    pad_pt: float,
    keep_rules: bool,
    with_caption: bool,
    caption_gap: float,
    drop_text_panels: bool,
    label_gap: float,
    stats: Stats,
) -> list[Record]:
    records: list[Record] = []
    seen_hashes: set[str] = set()

    for pno in pages:
        page = doc[pno]
        try:
            units = text_units(page)
            regions = find_figure_regions(
                page, units, gap_pt, min_pt, keep_rules, label_gap
            )
        except Exception as exc:
            stats.failed.append(f"detect page {pno + 1}: {exc}")
            continue

        if not regions:
            continue

        boxes = [box for box, _g, _t, _f in regions]
        captions = find_captions(boxes, units, caption_gap)

        for idx, ((box, n_graphics, n_text, text_frac), (caption, cap_rect)) in enumerate(
            zip(regions, captions), start=1
        ):
            # A shaded callout or pull-quote is a filled rectangle wrapped
            # around prose: almost no drawing, mostly text by area. Real
            # figures are either drawing-rich or image-backed with little
            # text, so both tests have to fire before anything is dropped.
            if drop_text_panels and n_graphics <= 3 and text_frac >= 0.40:
                stats.skipped_text_panel += 1
                continue

            clip = pymupdf.Rect(box)
            if with_caption and cap_rect is not None:
                clip |= cap_rect
            clip += (-pad_pt, -pad_pt, pad_pt, pad_pt)
            clip &= page.rect
            if clip.is_empty:
                continue

            try:
                pix = page.get_pixmap(clip=clip, dpi=dpi)
                data = pix.tobytes("png")
            except Exception as exc:
                stats.failed.append(f"render page {pno + 1} figure {idx}: {exc}")
                continue

            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes:
                stats.skipped_duplicate += 1
                continue
            seen_hashes.add(digest)

            out = outdir / f"p{pno + 1:03d}_fig{idx:02d}.png"
            out.write_bytes(data)

            note = f"{n_graphics} path/image rect(s), {n_text} text block(s)"
            if clip.get_area() >= 0.90 * page.rect.get_area():
                note += "; covers the whole page"

            records.append(
                Record(
                    page=pno + 1,
                    index=idx,
                    xref=-1,
                    path=out.name,
                    width=pix.width,
                    height=pix.height,
                    bpc=8,
                    colorspace="RGB",
                    has_alpha=False,
                    ext="png",
                    bytes=len(data),
                    placed_w_pt=round(clip.width, 2),
                    placed_h_pt=round(clip.height, 2),
                    dpi_x=float(dpi),
                    dpi_y=float(dpi),
                    source="figure",
                    note=note,
                    bbox=f"{clip.x0:.1f},{clip.y0:.1f},{clip.x1:.1f},{clip.y1:.1f}",
                    caption=" ".join(caption.split())[:300],
                )
            )
            stats.figures += 1

    return records


def write_manifest(records: list[Record], path: Path) -> None:
    fields = ["pdf"] + [f for f in Record.__dataclass_fields__ if f != "pdf"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r.pdf, r.page, r.index)):
            writer.writerow(asdict(rec))


def find_pdfs(root: Path, recursive: bool) -> list[Path]:
    """Every PDF under `root`, case-insensitively, in a stable order."""
    walk = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (p for p in walk if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: str(p).lower(),
    )


def process_pdf(path: Path, outdir: Path, args: argparse.Namespace) -> tuple[list[Record], Stats]:
    """Run the selected mode over one PDF. Never raises; failures land in Stats."""
    stats = Stats()

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        stats.failed.append(f"cannot open: {exc}")
        return [], stats

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            stats.failed.append("password-protected")
            return [], stats

        pages = parse_pages(args.pages, doc.page_count)
        if not pages:
            stats.failed.append("page range selected no pages")
            return [], stats

        outdir.mkdir(parents=True, exist_ok=True)
        records: list[Record] = []

        if args.mode == "figures":
            records += extract_figures(
                doc, pages, outdir, args.dpi, args.gap, args.min_figure, args.pad,
                args.keep_rules, args.with_caption, args.caption_gap,
                not args.keep_text_panels, args.label_gap, stats,
            )
        elif args.mode == "render":
            records += render_pages(doc, pages, outdir, args.dpi, stats)
        else:
            recs, productive = extract_embedded(
                doc, pages, outdir, args.min_size, args.min_bytes, stats
            )
            records += recs
            if args.mode == "auto":
                barren = [p for p in pages if p not in productive]
                if barren:
                    records += render_pages(doc, barren, outdir, args.dpi, stats)

        stats.pages_scanned = len(pages)
    finally:
        doc.close()

    for rec in records:
        rec.pdf = path.name

    write_manifest(records, outdir / "manifest.csv")
    return records, stats


def summarise(label: str, stats: Stats, mode: str, indent: str = "  ") -> None:
    print(f"{label}")
    if mode == "figures":
        print(f"{indent}figures cropped         : {stats.figures}")
        print(f"{indent}skipped (text panel)    : {stats.skipped_text_panel}")
    else:
        print(f"{indent}embedded images written : {stats.kept}")
        print(f"{indent}pages rendered          : {stats.rendered_pages}")
        print(f"{indent}skipped (too small)     : {stats.skipped_small}")
    print(f"{indent}skipped (duplicate)     : {stats.skipped_duplicate}")
    if stats.failed:
        print(f"{indent}failed                  : {len(stats.failed)}")
        for line in stats.failed[:10]:
            print(f"{indent}    {line}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract images from a PDF, or from every PDF in a directory.",
    )
    ap.add_argument("input", type=Path, help="a PDF file, or a directory of PDFs")
    ap.add_argument("-o", "--outdir", type=Path, default=None,
                    help="output directory (default: <name>_images/ beside the input)")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="when the input is a directory, also search subdirectories")
    ap.add_argument("--mode", choices=["embedded", "render", "figures", "auto"], default="auto",
                    help="auto = embedded, plus render any page that yielded nothing")
    ap.add_argument("--pages", default=None, help='page range, e.g. "3-7,12" (1-based)')
    ap.add_argument("--dpi", type=int, default=300, help="render DPI (default 300)")
    ap.add_argument("--min-size", type=int, default=64,
                    help="skip images narrower or shorter than this, in pixels")
    ap.add_argument("--min-bytes", type=int, default=2048,
                    help="skip images smaller than this on disk")

    fig = ap.add_argument_group("figures mode")
    fig.add_argument("--gap", type=float, default=12.0,
                     help="ink closer than this many points is one figure (default 12)")
    fig.add_argument("--min-figure", type=float, default=40.0,
                     help="discard figures smaller than this, in points (default 40)")
    fig.add_argument("--pad", type=float, default=6.0,
                     help="padding around each figure, in points (default 6)")
    fig.add_argument("--label-gap", type=float, default=24.0,
                     help="pull in axis labels and legends within this many points "
                          "of the figure (default 24)")
    fig.add_argument("--with-caption", action="store_true",
                     help="include the caption in the crop (always in the manifest)")
    fig.add_argument("--caption-gap", type=float, default=48.0,
                     help="max distance from figure to its caption, in points")
    fig.add_argument("--keep-rules", action="store_true",
                     help="keep page-spanning hairlines instead of treating them as furniture")
    fig.add_argument("--keep-text-panels", action="store_true",
                     help="keep shaded callouts and pull-quotes, which are dropped by default")
    args = ap.parse_args(argv)

    src = args.input
    if src.is_dir():
        pdfs = find_pdfs(src, args.recursive)
        if not pdfs:
            where = "or any subdirectory" if args.recursive else "(try --recursive)"
            print(f"error: no PDFs in {src} {where}", file=sys.stderr)
            return 1
        root = args.outdir or src / "extracted_images"
    elif src.is_file():
        pdfs = [src]
        root = args.outdir or src.with_name(f"{src.stem}_images")
    else:
        print(f"error: no such file or directory: {src}", file=sys.stderr)
        return 1

    root.mkdir(parents=True, exist_ok=True)
    batch = len(pdfs) > 1 or src.is_dir()

    all_records: list[Record] = []
    total = Stats()
    unreadable: list[str] = []

    for n, pdf in enumerate(pdfs, start=1):
        # One folder per PDF, mirroring any subdirectory structure so two
        # papers with the same filename cannot overwrite each other.
        if batch:
            rel = pdf.relative_to(src).parent if args.recursive else Path()
            outdir = root / rel / pdf.stem
        else:
            outdir = root

        records, stats = process_pdf(pdf, outdir, args)
        all_records += records

        total.kept += stats.kept
        total.figures += stats.figures
        total.rendered_pages += stats.rendered_pages
        total.skipped_small += stats.skipped_small
        total.skipped_duplicate += stats.skipped_duplicate
        total.skipped_text_panel += stats.skipped_text_panel
        total.pages_scanned += stats.pages_scanned
        total.failed += [f"{pdf.name}: {line}" for line in stats.failed]

        if not records and stats.failed:
            unreadable.append(f"{pdf.name}: {stats.failed[0]}")

        prefix = f"[{n}/{len(pdfs)}] " if batch else ""
        summarise(f"{prefix}{pdf.name}: {stats.pages_scanned} page(s) -> {outdir}",
                  stats, args.mode)

    if batch:
        combined = root / "manifest.csv"
        write_manifest(all_records, combined)
        print()
        summarise(f"TOTAL across {len(pdfs)} PDF(s):", total, args.mode)
        print(f"  combined manifest       : {combined}")
    else:
        print(f"  manifest                : {root / 'manifest.csv'}")

    lowres = [r for r in all_records if r.source == "embedded" and 0 < r.dpi_x < 150]
    if lowres:
        print(f"  note: {len(lowres)} image(s) placed below 150 DPI (see manifest)")

    # A PDF that produced nothing at all is worth a non-zero exit for scripting,
    # but only if every single one failed -- a partial batch still succeeded.
    return 1 if unreadable and len(unreadable) == len(pdfs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
