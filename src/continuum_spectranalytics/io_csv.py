"""Layout A: the validated CSV that joins Part 1 to Part 2.

Design section 5. This file is the **only** interface between the two halves of
the pipeline, and Part 2 re-reads it using the same ``Header Rows: N``
convention Part 1 applied to the raw instrument file -- the header length is
declared in the file and read back from it, never assumed.

The layout is deliberate:

*The original metadata header block sits on top, verbatim.* An output that
carries its own provenance of origin can be handed to anyone -- or read back
years later -- without a second file to explain what instrument produced it,
what the sample was, or when it ran. Only line 1 is rewritten, because it
declares the header length and the emitted file's header is a different length
from the raw file's. Everything below it is byte-for-byte what the instrument
wrote.

*Pipeline lines are appended below that block, still inside the header.* They
record what this stage did -- the chosen window, the branch taken for each open
question -- so the file is self-describing. Per-run provenance in full does not
live here: it goes to the sidecar, which keeps this table lean without losing
auditability. The prohibition is on provenance in the CSV **body**, and the
header is not the body.

*Six columns, in dependency order.* Each depends only on those to its left,
which is what makes the file checkable in one pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_spectranalytics.tier0.argen_io import find_declared_header_length

COLUMNS = (
    "seconds",
    "concentration",
    "ls8",
    "ls8_smoothed",
    "I_rayleigh",
    "Mw_over_M0",
)

# Column names as they appeared before the layout was pinned to the design.
# Files written by earlier versions are still readable, which matters because
# reprocessing a run to read its own output would defeat the point of having a
# stable boundary artifact.
_LEGACY_ALIASES = {
    "time_s": "seconds",
    "concentration_g_per_cm3": "concentration",
    "signal_raw": "ls8",
    "signal_smoothed": "ls8_smoothed",
    "rayleigh_ratio_cm_inv": "I_rayleigh",
    "mw_over_m0": "Mw_over_M0",
}

HEADER_DECLARATION = "Header Rows: {n}"


def write_processed_csv(
    path: str | Path,
    columns: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    source_header: list[str] | None = None,
    channel: str = "ls8",
) -> Path:
    """Write the Layout A boundary CSV.

    Parameters
    ----------
    columns:
        The six columns, keyed by their Layout A names.
    metadata:
        Pipeline lines appended to the header block, below the source header.
    source_header:
        The raw file's metadata lines, verbatim. Line 1 of those lines is
        replaced with a corrected header-length declaration; the rest are
        written unchanged.
    channel:
        The scattering channel, used to name the raw and smoothed columns when
        it is not ``ls8``. The design fixes the names for the working channel;
        naming them after whatever channel was actually processed keeps the file
        honest about its own contents.
    """
    missing = [name for name in COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"processed CSV is missing required columns: {missing}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The source block, minus its own header-length line, which no longer
    # describes this file.
    source = list(source_header or [])
    carried = [
        line for line in source if find_declared_header_length([line]) is None
    ]
    pipeline = [f"{key}: {value}" for key, value in (metadata or {}).items()]

    # Declared count includes the column row, matching the instrument's own
    # convention: "Header Rows: 10" means nine metadata lines and column names
    # on line ten.
    n_header = 1 + len(carried) + len(pipeline) + 1

    names = {name: name for name in COLUMNS}
    if channel != "ls8":
        names["ls8"] = channel
        names["ls8_smoothed"] = f"{channel}_smoothed"

    frame = pd.DataFrame({names[name]: columns[name] for name in COLUMNS})
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER_DECLARATION.format(n=n_header) + "\n")
        for line in carried:
            handle.write(line + "\n")
        for line in pipeline:
            handle.write(line + "\n")
        frame.to_csv(handle, index=False)
    return path


def read_processed_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read the boundary CSV back, using the length it declares.

    Returns the frame under Layout A column names and the parsed header. Files
    written under the earlier column names are renamed on read rather than
    rejected.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        preamble = [handle.readline() for _ in range(200)]

    n_header = find_declared_header_length(preamble)
    if n_header is None:
        raise ValueError(
            f"{path.name} does not declare its header length, so where the data "
            "starts cannot be established. It was not written by this pipeline."
        )

    column_row_index = n_header - 1
    metadata: dict[str, str] = {}
    for line in preamble[1:column_row_index]:
        stripped = line.strip().lstrip("#").strip()
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            metadata[key.strip()] = value.strip()

    frame = pd.read_csv(path, skiprows=column_row_index)
    frame = frame.rename(columns=_LEGACY_ALIASES)

    # A run on another channel names its columns after that channel.
    channel = metadata.get("channel")
    if channel and channel != "ls8":
        frame = frame.rename(
            columns={channel: "ls8", f"{channel}_smoothed": "ls8_smoothed"}
        )

    absent = [name for name in COLUMNS if name not in frame.columns]
    if absent:
        raise ValueError(
            f"{path.name} is missing columns {absent}; found {list(frame.columns)}"
        )
    return frame, metadata


__all__ = [
    "COLUMNS",
    "HEADER_DECLARATION",
    "read_processed_csv",
    "write_processed_csv",
]
