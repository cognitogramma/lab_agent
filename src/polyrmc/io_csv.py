"""The validated CSV that joins Part 1 to Part 2.

This file is the **only** interface between the two halves of the pipeline.
Part 2 re-reads it with the same parsing convention Part 1 applied to the raw
instrument file -- the header length is declared in the file and read back from
it, never assumed.

Six columns in dependency order. Three are new derivations: the concentration
axis from the dilution trajectory, the absolute Rayleigh ratio from toluene
calibration, and the aggregation index Mw/M0.

Per-run provenance does not live here. It goes to the sidecar
(:mod:`polyrmc.provenance`), which keeps this table lean without losing
auditability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polyrmc.tier0.argen_io import find_declared_header_length

COLUMNS = (
    "time_s",
    "concentration_g_per_cm3",
    "signal_raw",
    "signal_smoothed",
    "rayleigh_ratio_cm_inv",
    "mw_over_m0",
)

_HEADER_DECLARATION = "# Number of header rows: {n}"


def write_processed_csv(
    path: str | Path,
    columns: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the boundary CSV with a self-describing metadata header.

    The header declares its own length on the first line, so the reader never
    has to guess where the data starts.
    """
    missing = [name for name in COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"processed CSV is missing required columns: {missing}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata_lines = [f"# {key}: {value}" for key, value in (metadata or {}).items()]
    # +1 for the declaration line itself, +1 for the column-name row that
    # pandas writes; skiprows must land exactly on the header row.
    n_header = len(metadata_lines) + 1

    frame = pd.DataFrame({name: columns[name] for name in COLUMNS})
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_HEADER_DECLARATION.format(n=n_header) + "\n")
        for line in metadata_lines:
            handle.write(line + "\n")
        frame.to_csv(handle, index=False)
    return path


def read_processed_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read the boundary CSV back, using the declared header length.

    Returns the data frame and the parsed metadata header.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        preamble = [handle.readline() for _ in range(100)]

    n_header = find_declared_header_length(preamble)
    if n_header is None:
        raise ValueError(
            f"{path.name} does not declare its header length; it was not written "
            "by write_processed_csv"
        )

    metadata: dict[str, str] = {}
    for line in preamble[1:n_header]:
        stripped = line.strip().lstrip("#").strip()
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            metadata[key.strip()] = value.strip()

    frame = pd.read_csv(path, skiprows=n_header)
    unexpected = [name for name in COLUMNS if name not in frame.columns]
    if unexpected:
        raise ValueError(f"processed CSV is missing columns: {unexpected}")
    return frame, metadata
