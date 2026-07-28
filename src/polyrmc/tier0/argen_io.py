"""Ingest ARGEN files: dynamic header, time reconstruction, scoped null handling.

Three things here are not negotiable, and each exists because the naive version
is wrong on real files:

1. The header length is **declared in the file** and read from it. It is never
   hard-coded, because it varies between acquisitions.
2. The reconstructed monotonic time is the authoritative x-axis, **not the row
   index**. Interior row drops make sampling non-uniform, so row index and
   elapsed time diverge.
3. Null handling is **scoped to the scattering channels**. A missing
   temperature, stir speed, or ND filter reading never discards a row.

.. note::
   ``HEADER_COUNT_PATTERNS`` is matched against the file preamble to find the
   declared header length. The patterns cover the common spellings; if this
   instrument writes a form not listed, add it there rather than falling back
   to a fixed row count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from polyrmc.state import TimeFormat

# Candidate spellings of the declared header length, most specific first.
HEADER_COUNT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"number\s+of\s+header\s+(?:rows|lines)\s*[:=,\t]\s*(\d+)", re.I),
    re.compile(r"header\s+(?:rows|lines|length)\s*[:=,\t]\s*(\d+)", re.I),
    re.compile(r"#\s*header\s*[:=,\t]\s*(\d+)", re.I),
    re.compile(r"data\s+(?:starts|begins)\s+(?:at\s+)?(?:row|line)\s*[:=,\t]\s*(\d+)", re.I),
)

_NULL_TOKENS = {"", "nan", "na", "n/a", "null", "none", "-", "--", "#n/a"}

# Time tokens: "MM:SS(.s)" wraps hourly; "HH:MM:SS(.s)" is absolute.
_CLOCK_RE = re.compile(r"^\s*(\d+):(\d{1,2})(?::(\d{1,2}))?(\.\d+)?\s*$")


@dataclass
class ArgenFile:
    """A parsed ARGEN acquisition."""

    path: Path
    metadata: dict[str, str]
    n_header_rows: int
    data: pd.DataFrame
    time_s: np.ndarray
    time_format: TimeFormat
    scattering_columns: list[str] = field(default_factory=list)
    dropped_rows: list[int] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(self.time_s)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if self.n_points else 0.0

    @property
    def is_uniformly_sampled(self) -> bool:
        """True when sample spacing is effectively constant.

        Almost always False on real files, which is exactly why time -- not row
        index -- is the x-axis.
        """
        if self.n_points < 3:
            return True
        dt = np.diff(self.time_s)
        return bool(np.allclose(dt, dt[0], rtol=1e-3))


def find_declared_header_length(lines: list[str]) -> int | None:
    """Read the header length declared in the file preamble.

    Returns ``None`` if no declaration is found, which the caller must treat as
    an error rather than silently guessing.
    """
    for line in lines:
        for pattern in HEADER_COUNT_PATTERNS:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    return None


def parse_header_metadata(lines: list[str], sep: str = "\t") -> dict[str, str]:
    """Pull ``key: value`` and ``key<sep>value`` pairs out of the preamble."""
    metadata: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped and (sep not in stripped or stripped.index(":") < stripped.index(sep)):
            key, _, value = stripped.partition(":")
        elif sep in stripped:
            key, _, value = stripped.partition(sep)
        else:
            continue
        key = key.strip().lstrip("#").strip()
        value = value.strip()
        if key and value:
            metadata.setdefault(key, value)
    return metadata


def detect_time_format(tokens: list[str]) -> TimeFormat:
    """Classify the elapsed-time encoding from token shape.

    ``MM:SS`` wraps to zero every hour and needs cumulative de-wrapping;
    ``HH:MM:SS`` and bare seconds are parsed directly.
    """
    for token in tokens:
        if token is None:
            continue
        text = str(token).strip()
        if not text or text.lower() in _NULL_TOKENS:
            continue
        match = _CLOCK_RE.match(text)
        if match:
            return (
                TimeFormat.ABSOLUTE_HHMMSS
                if match.group(3) is not None
                else TimeFormat.WRAPPING_MMSS
            )
        try:
            float(text)
        except ValueError:
            continue
        return TimeFormat.SECONDS
    raise ValueError("could not classify the elapsed-time format from any token")


def _clock_to_seconds(text: str) -> float:
    match = _CLOCK_RE.match(text)
    if not match:
        raise ValueError(f"not a clock-formatted time: {text!r}")
    first, second, third, frac = match.groups()
    fraction = float(frac) if frac else 0.0
    if third is None:  # MM:SS
        return int(first) * 60 + int(second) + fraction
    return int(first) * 3600 + int(second) * 60 + int(third) + fraction


def reconstruct_time(tokens: list[str], time_format: TimeFormat) -> np.ndarray:
    """Build a monotonic elapsed-time axis in seconds.

    For the wrapping format, a decrease relative to the previous sample means
    the field rolled over, and a full hour is added to the running offset. The
    result is the authoritative x-axis for every downstream calculation.
    """
    values = np.empty(len(tokens), dtype=float)

    if time_format is TimeFormat.SECONDS:
        for i, token in enumerate(tokens):
            values[i] = float(str(token).strip())
    else:
        for i, token in enumerate(tokens):
            values[i] = _clock_to_seconds(str(token).strip())

    if time_format is TimeFormat.WRAPPING_MMSS and len(values) > 1:
        offset = 0.0
        dewrapped = np.empty_like(values)
        dewrapped[0] = values[0]
        for i in range(1, len(values)):
            if values[i] < values[i - 1]:
                offset += 3600.0
            dewrapped[i] = values[i] + offset
        values = dewrapped

    if np.any(np.diff(values) < 0):
        raise ValueError("reconstructed time is not monotonic; format detection failed")
    return values


def identify_scattering_columns(
    columns: list[str], expected: int = 16, pattern: str = r"(cell|ch(?:an(?:nel)?)?|det)\s*_?\d+"
) -> list[str]:
    """Find the scattering-channel columns among the data columns."""
    regex = re.compile(pattern, re.I)
    found = [column for column in columns if regex.search(column)]
    if not found:
        raise ValueError(
            f"no scattering channels matched {pattern!r} in columns {columns[:12]}..."
        )
    if len(found) != expected:
        # Not fatal -- some acquisitions run fewer than 16 cells -- but the
        # caller should know the count differed from the instrument default.
        pass
    return found


def scoped_null_mask(frame: pd.DataFrame, scattering_columns: list[str]) -> np.ndarray:
    """Rows to drop: those with a null in **any** scattering channel.

    Nulls in auxiliary columns (temperature, stir speed, ND filter) are
    tolerated and never trigger a drop.
    """
    subset = frame[scattering_columns]
    return subset.isna().any(axis=1).to_numpy()


def load_argen_file(
    path: str | Path,
    *,
    sep: str = "\t",
    time_column: str | None = None,
    expected_channels: int = 16,
    channel_pattern: str = r"(cell|ch(?:an(?:nel)?)?|det)\s*_?\d+",
    encoding: str = "utf-8",
) -> ArgenFile:
    """Load an ARGEN acquisition into a validated :class:`ArgenFile`.

    Parameters
    ----------
    path:
        Raw instrument file.
    sep:
        Column delimiter; ARGEN exports are tab-delimited by default.
    time_column:
        Elapsed-time column name. Auto-detected when omitted.
    expected_channels:
        Instrument default is 16 independent cells.

    Raises
    ------
    ValueError
        If the file does not declare its header length, or if the time axis
        cannot be reconstructed monotonically.
    """
    path = Path(path)
    with path.open("r", encoding=encoding, errors="replace") as handle:
        preamble = [handle.readline() for _ in range(200)]

    n_header_rows = find_declared_header_length(preamble)
    if n_header_rows is None:
        raise ValueError(
            f"{path.name} does not declare its header length. Header length is read "
            "from the file, never assumed -- add the file's spelling to "
            "HEADER_COUNT_PATTERNS rather than hard-coding a row count."
        )

    metadata = parse_header_metadata(preamble[:n_header_rows], sep=sep)

    frame = pd.read_csv(
        path,
        sep=sep,
        skiprows=n_header_rows,
        encoding=encoding,
        engine="python",
        na_values=sorted(_NULL_TOKENS - {""}),
    )
    frame.columns = [str(column).strip() for column in frame.columns]

    if time_column is None:
        candidates = [c for c in frame.columns if re.search(r"time|elapsed", c, re.I)]
        if not candidates:
            raise ValueError(f"no time-like column in {list(frame.columns)[:12]}...")
        time_column = candidates[0]

    scattering_columns = identify_scattering_columns(
        list(frame.columns), expected=expected_channels, pattern=channel_pattern
    )

    drop_mask = scoped_null_mask(frame, scattering_columns)
    dropped_rows = np.flatnonzero(drop_mask).tolist()
    kept = frame.loc[~drop_mask].reset_index(drop=True)

    raw_time_tokens = kept[time_column].astype(str).tolist()
    time_format = detect_time_format(raw_time_tokens)
    time_s = reconstruct_time(raw_time_tokens, time_format)

    return ArgenFile(
        path=path,
        metadata=metadata,
        n_header_rows=n_header_rows,
        data=kept,
        time_s=time_s,
        time_format=time_format,
        scattering_columns=scattering_columns,
        dropped_rows=dropped_rows,
    )
