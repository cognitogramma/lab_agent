"""Opening a data file as a table, whatever kind of file it is.

This is the part of KaleidaGraph this package was missing: point at a file, see
what columns are in it, plot one against another. Everything else here works on
runs the pipeline has already processed, which is the right shape for producing
a reported number and the wrong shape entirely for looking at data. A researcher
with a file in front of them wants to see it before deciding anything, and
sending them to another application to do that is how the other application
ends up owning the figures that get published.

Three kinds of file are recognised, tried in this order because the specific
ones declare themselves and the general one cannot:

1. A boundary CSV this pipeline wrote. It declares its own header length and
   its columns are fixed, so it is identified by being readable as one.
2. An ARGEN acquisition. Also declares its header length; every scattering
   channel is kept rather than just ls8, because the point of opening a file
   by hand is to look at what you did not already decide to analyse.
3. Anything else delimited. Sniffed, and honest about having sniffed.

What comes back is a frame and a description of how it was read. Nothing is
cleaned, censored, spliced, or smoothed on the way through -- a table opened
here is the file, not a run. That distinction is the whole reason this module
is separate from :mod:`continuum_spectranalytics.rehydrate`, which reads *processed* runs back
along with the record of every decision made about them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Formats a researcher will reasonably try, that this package cannot read yet.
# Named explicitly so the failure is "not supported" rather than whatever a
# delimiter sniffer says about a binary file, which is usually a wall of
# nonsense that reads like a bug in the tool.
# Each entry completes the sentence "<name> is ...", so an entry is a whole
# explanation rather than a label. They were labels once, slotted into a fixed
# "is a {} file, which this package cannot read yet" sentence, and the moment an
# entry needed to say something other than "a format we lack a reader for" the
# result was "x.qdr is a not a KaleidaGraph format ... file". The advice differs
# per format too: exporting to text is the answer for a spreadsheet and no help
# at all for a file that never held a table.
UNSUPPORTED = {
    # A .QPC is a KaleidaGraph plot. Refused here rather than read because it
    # holds no rows at all -- the examples are tens of kilobytes against
    # multi-megabyte data files, and the numbers live in the .QDA it was built
    # from. What it does say about itself is read elsewhere.
    ".qpc": "a KaleidaGraph plot, which records how a graph was drawn and not "
    "the data it was drawn from -- there are no rows in it. Open the .QDA it "
    "was built from for the numbers, or read the plot itself with "
    "`describe_plot_file` (in a session) or `continuum-spectranalytics plot --file <it>` (at a "
    "terminal).",
    # Not a KaleidaGraph format at all, despite being asked for by name often
    # enough to be worth answering precisely rather than generically.
    ".qdr": "not a KaleidaGraph format. The application writes .QDA (data), "
    ".QPC (graph), .QPL (layout), .QSC (script), .QST (style) and .QCT (colour "
    "table), and nothing named .QDR. You probably want the .QDA for data or the "
    ".QPC for a graph.",
    ".xls": "a legacy Excel workbook, which this package cannot read yet. "
    "Export it to text from the application that wrote it -- a delimited file "
    "with a header row opens here.",
    ".xlsx": "an Excel workbook, which this package cannot read yet. Export it "
    "to text from the application that wrote it -- a delimited file with a "
    "header row opens here.",
    ".opj": "an Origin project, which this package cannot read yet. Export it "
    "to text from the application that wrote it -- a delimited file with a "
    "header row opens here.",
}

PREVIEW_ROWS = 5


class UnsupportedFile(ValueError):
    """A file whose format this package does not read.

    Separate from a parse failure because the answer is different: a parse
    failure is worth retrying with different options, and this is not.
    """


@dataclass
class Table:
    """One data file, opened.

    ``notes`` records everything about *how* it was read that a reader would
    want to know before trusting a plot of it -- a sniffed delimiter, a header
    guessed at, columns that are not numeric. It is meant to be shown, not
    logged.
    """

    path: Path
    frame: pd.DataFrame
    kind: str
    metadata: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    def numeric_columns(self) -> list[str]:
        """Columns that can go on an axis."""
        return [
            str(c)
            for c in self.frame.columns
            if pd.api.types.is_numeric_dtype(self.frame[c])
        ]

    def preview(self, rows: int = PREVIEW_ROWS) -> str:
        return self.frame.head(rows).to_string(max_cols=12, show_dimensions=False)


def _try_kaleidagraph(path: Path) -> Table | None:
    from continuum_spectranalytics.kaleidagraph import NotAQdaFile, read_qda

    try:
        parsed = read_qda(path)
    except (NotAQdaFile, ValueError, OSError):
        return None
    return Table(
        path=path,
        frame=parsed.frame,
        kind="kaleidagraph",
        metadata={
            "columns_declared": str(parsed.n_columns_declared),
            "rows_declared": str(parsed.n_rows_declared),
        },
        notes=list(parsed.notes),
    )


def _try_processed(path: Path) -> Table | None:
    from continuum_spectranalytics.io_csv import read_processed_csv

    try:
        frame, metadata = read_processed_csv(path)
    except Exception:  # noqa: BLE001 - not this kind of file, try the next
        return None
    return Table(path=path, frame=frame, kind="processed", metadata=metadata)


def _try_argen(path: Path) -> Table | None:
    from continuum_spectranalytics.tier0.argen_io import load_argen_file

    try:
        # channel=None keeps every scattering column. The pipeline narrows to
        # ls8 on purpose; opening a file by hand is the one moment where the
        # other fifteen channels are the point.
        parsed = load_argen_file(path, channel=None)
    except Exception:  # noqa: BLE001
        return None

    frame = parsed.data.copy()
    # The reconstructed elapsed time, as a plottable column. The file's own time
    # column is text in instrument format (MM:SS.s, wrapping every hour), which
    # no axis can use and which is exactly the trap this reconstruction exists
    # to close.
    if len(parsed.time_s) == len(frame):
        frame.insert(0, "seconds", parsed.time_s)

    notes = []
    if parsed.available_channels:
        notes.append(
            f"{len(parsed.available_channels)} scattering channels available: "
            + ", ".join(parsed.available_channels)
        )
    notes.append(
        "raw acquisition: nothing is spliced, censored, smoothed, or "
        "baseline-corrected in this table"
    )
    return Table(
        path=path,
        frame=frame,
        kind="argen",
        metadata=dict(parsed.metadata),
        notes=notes,
    )


def _try_delimited(path: Path) -> Table | None:
    """Last resort: let pandas sniff it, and say that is what happened."""
    attempts = (
        ({"sep": None, "engine": "python", "comment": "#"}, "sniffed, # treated as comment"),
        ({"sep": None, "engine": "python"}, "sniffed"),
        ({"sep": "\t"}, "read as tab-delimited"),
        ({}, "read as comma-delimited"),
    )
    for options, description in attempts:
        try:
            frame = pd.read_csv(path, **options)
        except Exception:  # noqa: BLE001
            continue
        # At least one column has to hold actual numbers. Without this the
        # sniffer accepts prose: a sentence splits on whitespace into as many
        # "columns" as it has words, and the caller is handed a table of English
        # to plot. A recognised format is allowed to be all text -- that is a
        # real file with a real answer -- but a guess is not.
        #
        # Values, not dtype. Ragged lines leave trailing columns that are empty
        # in every row, and pandas types an all-empty column as float64, so a
        # dtype check calls two lines of prose numeric on the strength of a
        # column that contains nothing at all.
        has_numeric = any(
            pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any()
            for column in frame.columns
        )
        if frame.shape[1] >= 2 and len(frame) and has_numeric:
            return Table(
                path=path,
                frame=frame,
                kind="delimited",
                notes=[
                    f"format not recognised; {description}. Check the column "
                    "names below match what you expect before plotting."
                ],
            )
    return None


def open_table(path: str | Path) -> Table:
    """Open any supported data file as a table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no file at {path}")
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory, not a data file")

    suffix = path.suffix.lower()
    if suffix in UNSUPPORTED:
        raise UnsupportedFile(
            f"{path.name} is {UNSUPPORTED[suffix]}"
        )

    # KaleidaGraph first: it identifies itself by a signature, so it is the one
    # format here that can be recognised rather than attempted.
    for reader in (_try_kaleidagraph, _try_processed, _try_argen, _try_delimited):
        table = reader(path)
        if table is not None:
            return table

    raise ValueError(
        f"{path.name} could not be read as a table. It is not a processed CSV, "
        "not an ARGEN acquisition, and no delimiter produced more than one "
        "column with numbers in it. If it is a text export, check it has a "
        "header row and that the numeric columns are not quoted."
    )


__all__ = [
    "PREVIEW_ROWS",
    "UNSUPPORTED",
    "Table",
    "UnsupportedFile",
    "open_table",
]
