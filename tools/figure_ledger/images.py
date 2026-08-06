"""Identify and decode image files by what they contain, not what they are called.

Files out of ``pdfimages`` routinely carry a ``.png`` extension over Netpbm
data: the tool writes PPM/PGM/PBM unless given ``-png``, and renaming a file
converts nothing. So every file here is identified by its **magic bytes**. The
extension is treated as a hint at best and a lie at worst.

Netpbm is then decoded and re-encoded as PNG so a browser can display it, rather
than being rejected. Rejecting it would leave a third of an archive unlabelled
for a reason that is entirely fixable.

Where a file genuinely cannot be shown, the detected format is named. "Cannot
display this file" tells a researcher nothing; "this is TIFF" tells them exactly
what to run.

Standard library only -- zlib and struct are enough to write a PNG, and adding a
dependency to this tool would defeat its purpose.
"""

from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

# Magic byte signatures, longest first so a prefix never shadows a longer match.
SIGNATURES: tuple[tuple[bytes, str, bool], ...] = (
    (b"\x89PNG\r\n\x1a\n", "PNG", True),
    (b"\xff\xd8\xff", "JPEG", True),
    (b"GIF87a", "GIF", True),
    (b"GIF89a", "GIF", True),
    (b"RIFF", "WebP", True),  # refined below by the WEBP tag at offset 8
    (b"II*\x00", "TIFF (little-endian)", False),
    (b"MM\x00*", "TIFF (big-endian)", False),
    (b"BM", "BMP", True),
    (b"%PDF", "PDF", False),
    (b"\x00\x00\x01\x00", "ICO", False),
)

# P1-P6: plain/binary bitmap, graymap, pixmap.
NETPBM_KINDS = {
    b"P1": "Netpbm PBM (plain)",
    b"P2": "Netpbm PGM (plain)",
    b"P3": "Netpbm PPM (plain)",
    b"P4": "Netpbm PBM (binary)",
    b"P5": "Netpbm PGM (binary)",
    b"P6": "Netpbm PPM (binary)",
}

# Browsers display these natively, so they are served through untouched.
BROWSER_NATIVE = {"PNG", "JPEG", "GIF", "WebP", "BMP"}


@dataclass
class Identified:
    """What a file actually is."""

    format: str
    displayable: bool
    """True if a browser can render it, with or without conversion here."""

    needs_conversion: bool = False
    detail: str = ""

    @property
    def summary(self) -> str:
        return f"{self.format}{f' -- {self.detail}' if self.detail else ''}"


def identify(data: bytes) -> Identified:
    """Name the format from the leading bytes.

    Reads only the header, so this stays cheap on a folder of thousands.
    """
    if len(data) < 2:
        return Identified("empty or truncated", displayable=False)

    magic = data[:2]
    if magic in NETPBM_KINDS:
        # A third byte that is not whitespace means this is something else that
        # happens to start "P5" -- Netpbm always separates magic from width.
        if len(data) > 2 and data[2:3].isspace():
            return Identified(
                NETPBM_KINDS[magic],
                displayable=True,
                needs_conversion=True,
                detail="converted to PNG for display",
            )

    for signature, name, displayable in SIGNATURES:
        if data.startswith(signature):
            if name == "WebP":
                if data[8:12] != b"WEBP":
                    continue
            return Identified(name, displayable=displayable)

    return Identified(
        "unrecognised", displayable=False, detail=f"first bytes {data[:8]!r}"
    )


# --- Netpbm ------------------------------------------------------------------


def _tokens(data: bytes, start: int, count: int) -> tuple[list[bytes], int]:
    """Read exactly ``count`` whitespace-separated header tokens.

    Netpbm allows comments anywhere in the header, including between the width
    and the height, which is why this cannot be a simple split.

    Reading *exactly* the number asked for matters: the header is followed
    immediately by binary sample data, and a tokeniser that reads one token too
    many will happily parse pixel bytes as a number.
    """
    tokens: list[bytes] = []
    index = start
    while len(tokens) < count and index < len(data):
        char = data[index : index + 1]
        if char == b"#":
            while index < len(data) and data[index : index + 1] not in (b"\n", b"\r"):
                index += 1
        elif char.isspace():
            index += 1
        else:
            end = index
            while (
                end < len(data)
                and not data[end : end + 1].isspace()
                and data[end : end + 1] != b"#"
            ):
                end += 1
            tokens.append(data[index:end])
            index = end
    return tokens, index


def _header_number(token: bytes, name: str) -> int:
    try:
        return int(token)
    except ValueError as error:
        raise ValueError(
            f"Netpbm {name} is not a number: {token[:16]!r}. The header is "
            "malformed or the file is truncated."
        ) from error


def decode_netpbm(data: bytes) -> tuple[int, int, bytes, int]:
    """Decode P1-P6 into raw samples.

    Returns ``(width, height, samples, channels)`` with one byte per sample,
    which is what the PNG writer wants.
    """
    magic = data[:2]
    if magic not in NETPBM_KINDS:
        raise ValueError(f"not a Netpbm file: magic {magic!r}")

    plain = magic in (b"P1", b"P2", b"P3")
    channels = 3 if magic in (b"P3", b"P6") else 1
    wants_maxval = magic not in (b"P1", b"P4")

    header, index = _tokens(data, 2, 2)
    if len(header) < 2:
        raise ValueError("Netpbm header is truncated: no width and height")
    width = _header_number(header[0], "width")
    height = _header_number(header[1], "height")
    if width <= 0 or height <= 0:
        raise ValueError(f"Netpbm dimensions are not positive: {width}x{height}")

    maxval = 1
    if wants_maxval:
        more, index = _tokens(data, index, 1)
        if not more:
            raise ValueError("Netpbm header is truncated: no maxval")
        maxval = _header_number(more[0], "maxval")
        if maxval <= 0:
            raise ValueError(f"Netpbm maxval is not positive: {maxval}")

    count = width * height * channels

    if plain:
        body = data[index:]
        # Strip comments from the body too; plain formats permit them.
        body = re.sub(rb"#[^\n\r]*", b" ", body)
        values = [int(token) for token in body.split()[:count]]
        if len(values) < count:
            raise ValueError(
                f"Netpbm data is short: {len(values)} of {count} expected samples"
            )
        if magic == b"P1":
            # In PBM, 1 means black. Inverted relative to every other format.
            samples = bytes(0 if value else 255 for value in values)
        else:
            samples = _rescale(values, maxval)
        return width, height, samples, channels

    # Binary. Exactly one whitespace byte separates the header from the data.
    body = data[index + 1 :]

    if magic == b"P4":
        row_bytes = (width + 7) // 8
        needed = row_bytes * height
        if len(body) < needed:
            raise ValueError(
                f"Netpbm P4 data is short: {len(body)} of {needed} expected bytes"
            )
        out = bytearray(width * height)
        for row in range(height):
            base = row * row_bytes
            for column in range(width):
                bit = (body[base + (column >> 3)] >> (7 - (column & 7))) & 1
                out[row * width + column] = 0 if bit else 255
        return width, height, bytes(out), 1

    if maxval < 256:
        if len(body) < count:
            raise ValueError(
                f"Netpbm data is short: {len(body)} of {count} expected bytes"
            )
        values = list(body[:count])
    else:
        # 16-bit samples, big-endian per the spec.
        needed = count * 2
        if len(body) < needed:
            raise ValueError(
                f"Netpbm 16-bit data is short: {len(body)} of {needed} expected bytes"
            )
        values = [
            (body[i] << 8) | body[i + 1] for i in range(0, needed, 2)
        ]
    return width, height, _rescale(values, maxval), channels


def _rescale(values: list[int], maxval: int) -> bytes:
    """Map samples onto 0-255, which is what an 8-bit PNG carries."""
    if maxval == 255:
        return bytes(min(255, max(0, value)) for value in values)
    scale = 255.0 / maxval
    return bytes(min(255, max(0, int(round(value * scale)))) for value in values)


# --- PNG ---------------------------------------------------------------------


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_png(width: int, height: int, samples: bytes, channels: int) -> bytes:
    """Write an 8-bit PNG from raw samples.

    Filter type 0 on every scanline. Compression is what matters for transfer
    size here, and a per-row filter search would cost more time than it saves on
    a plot with large flat areas.
    """
    colour_type = {1: 0, 3: 2}.get(channels)
    if colour_type is None:
        raise ValueError(f"unsupported channel count: {channels}")

    stride = width * channels
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        raw += samples[row * stride : (row + 1) * stride]

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _chunk(b"IEND", b"")
    )


def to_displayable(data: bytes) -> tuple[bytes, str, Identified]:
    """Return bytes a browser can render, the content type, and what it was.

    Raises ValueError naming the detected format when the file cannot be shown,
    so the message a researcher sees says what the thing actually is.
    """
    found = identify(data)

    if found.needs_conversion:
        width, height, samples, channels = decode_netpbm(data)
        return encode_png(width, height, samples, channels), "image/png", found

    if found.format in BROWSER_NATIVE:
        content_type = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "GIF": "image/gif",
            "WebP": "image/webp",
            "BMP": "image/bmp",
        }[found.format]
        return data, content_type, found

    raise ValueError(
        f"This is {found.summary}. It is not a format a browser will display; "
        "convert it first (for example with ImageMagick or `pdfimages -png`)."
    )


def read_header(path: Path, size: int = 64) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


__all__ = [
    "BROWSER_NATIVE",
    "NETPBM_KINDS",
    "SIGNATURES",
    "Identified",
    "decode_netpbm",
    "encode_png",
    "identify",
    "read_header",
    "to_displayable",
]
