"""Reading KaleidaGraph ``.QDA`` data files.

KaleidaGraph writes a closed binary format with no published specification, so
what follows was worked out from the lab's own files and is described here in
full -- both because the next person to touch it has nothing else to go on, and
because a reader built from inference should say plainly which parts are
inferred.

The format is big-endian throughout, which is the giveaway that this began life
on 68k Macs. Values are IEEE 754 **single** precision, four bytes each. That
matters more than it looks: a column read back from a ``.QDA`` carries about
seven significant digits, not the fifteen a float64 pipeline will happily print
after it, and every number this module returns has already been rounded once by
the application that wrote it.

Layout, by offset::

    0x000  u16   unused (0x000c observed)
    0x002  u16   column count
    0x200  ...   column table: 6 bytes per column, opening with a u32 row count
    ...          column name table: 40 bytes per column, NUL-padded, beginning
                 at ``0x200 + columns * 6``

Neither the names nor the data sit at a fixed offset: both tables are sized by
the column count, so a 21-column file puts its names 66 bytes further along than
a 10-column one. The data begins at ``0x200 + columns * 46``, and each column is
then stored as one contiguous block::

    data    n * 4 bytes, big-endian float32
    flags   n * 2 bytes, one 0x0001 word per value
    record  136 bytes: the signature 0e 02 01 00 05 00 00 01, then the column
            name again, NUL-padded

The record sits *after* the data it describes, and the next column starts
immediately after it. So the records delimit the columns, and because a block is
exactly six bytes per row, the row count follows from the gap rather than from
any stored length: ``n = (record_offset - block_start) / 6``. That division
coming out exact is also the check that a block really is a column -- a region
that is not one almost never divides evenly and never lands within the declared
row count.

That check matters because these files hold more than a table. A file with plots
in it carries them between the last data column and the trailing empty ones, and
a reader that assumed every gap between records was column data would read a
megabyte of plot layout as a column of zeros.

Two things are known to be unknown, and are treated as such rather than guessed:

* The flag words are 0x0001 for every value in every file examined. They are
  most likely a per-cell "filled" mask, but nothing here has ever been observed
  set to anything else, so this module does not use them to mask anything. If a
  file ever shows a different value, that is worth looking at before trusting
  the column.
* ``.QDR`` files are KaleidaGraph *reports*, a different format holding plot
  layouts rather than a data table. They are refused by name rather than read
  hopefully, since a plot layout parsed as a data table would produce a
  perfectly plausible table of nonsense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# The 8 bytes that begin every per-column record. Long enough that a false
# positive inside float data is not a practical concern.
COLUMN_RECORD_SIGNATURE = bytes.fromhex("0e02010005000001")
COLUMN_RECORD_SIZE = 136

# The first two bytes tell the two formats apart without reference to the
# extension, which matters because these files are decades old and get renamed.
QDA_HEADER = bytes.fromhex("000c")

N_COLUMNS_OFFSET = 0x002

# At 0x200 begins a table of six bytes per column, then the name table of forty
# bytes per column, then the data. Both of those tables are sized by the column
# count, so neither the names nor the data sit at a fixed offset -- the only
# reason a constant appeared to work is that the first files examined all had
# ten columns. A twenty-one column file puts its names 66 bytes further along,
# and reading from the ten-column offset lands mid-table, where the bytes decode
# as small tidy floats and yield a table that looks entirely reasonable.
COLUMN_TABLE_OFFSET = 0x200
COLUMN_TABLE_ENTRY_SIZE = 6
NAME_FIELD_SIZE = 40

# Four bytes of value plus two of flag, per row. The whole reader turns on this
# number being exact.
BYTES_PER_ROW = 6


class NotAQdaFile(ValueError):
    """A file that is not a KaleidaGraph data table."""


@dataclass
class QdaFile:
    """One KaleidaGraph data file, read."""

    path: Path
    frame: pd.DataFrame
    n_columns_declared: int
    n_rows_declared: int
    notes: list[str] = field(default_factory=list)


def _decode_name(raw: bytes) -> str:
    return raw.split(b"\x00")[0].decode("mac-roman", errors="replace").strip()


def _name_at(data: bytes, table_at: int, index: int) -> str:
    """The name at one slot of the header's name table."""
    start = table_at + index * NAME_FIELD_SIZE
    return _decode_name(data[start : start + NAME_FIELD_SIZE])


def _record_name(data: bytes, record_at: int) -> str:
    """The name a column record carries for itself, which is authoritative."""
    start = record_at + len(COLUMN_RECORD_SIGNATURE)
    return _decode_name(data[start : start + NAME_FIELD_SIZE])


def _flags(data: bytes, record_at: int, rows: int) -> np.ndarray | None:
    """The per-cell filled mask for a column of ``rows``, or ``None``.

    Returns ``None`` when the bytes where the mask should be are not a mask,
    which is what separates a column from a picture. Every word is 0 or 1 --
    1 for a cell holding a value, 0 for one left blank -- and a stretch of
    stored plot layout that happens to be the right length is not remotely that.
    Without this check a megabyte of plot reads back as a smooth column of
    numbers.

    The zeros are real and rare: one column in the file this was built from has
    a single blank cell near its end. Reading the float sitting in a blank slot
    would produce a value nobody entered, which is why the mask is applied
    rather than merely checked.
    """
    if rows <= 0:
        return np.empty(0, dtype=bool)
    start = record_at - rows * 2
    if start < 0:
        return None
    words = np.frombuffer(data, dtype=">u2", count=rows, offset=start)
    if not bool((words <= 1).all()):
        return None
    return words == 1


def _resolve_column(
    data: bytes, block_start: int, record_at: int, declared_rows: int
) -> tuple[int, np.ndarray] | None:
    """How many rows the column ending at ``record_at`` holds.

    The data is anchored to the *end* of the block, not the start, because a
    block does not always hold one column. A column type this reader does not
    handle -- ``elapsedtime`` is stored at ten bytes per row, not six -- leaves
    no record of its own, so its bytes sit at the front of the next column's
    block. Measuring backwards from the record finds the column that record
    names; measuring forwards finds the wrong one and renames it.
    """
    available = record_at - block_start
    if available < 0:
        return None

    # The exact fit first, then what the header allocated. Both are checked
    # against the flags, so a wrong guess is rejected rather than returned.
    candidates = []
    if available % BYTES_PER_ROW == 0:
        candidates.append(available // BYTES_PER_ROW)
    candidates.append(declared_rows)

    for rows in candidates:
        if rows < 0 or rows > declared_rows or rows * BYTES_PER_ROW > available:
            continue
        mask = _flags(data, record_at, rows)
        if mask is not None:
            return rows, mask
    return None


def read_qda(path: str | Path) -> QdaFile:
    """Read a KaleidaGraph ``.QDA`` file into a frame."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < COLUMN_TABLE_OFFSET + COLUMN_TABLE_ENTRY_SIZE:
        raise NotAQdaFile(f"{path.name} is too short to be a KaleidaGraph data file")

    records = [m.start() for m in re.finditer(re.escape(COLUMN_RECORD_SIGNATURE), data)]
    if not records:
        raise NotAQdaFile(
            f"{path.name} holds no KaleidaGraph column records. If it is a .QDR "
            "it is a report layout, not a data table."
        )

    n_columns = int(np.frombuffer(data, dtype=">u2", count=1, offset=N_COLUMNS_OFFSET)[0])
    if n_columns <= 0:
        raise NotAQdaFile(f"{path.name} declares {n_columns} columns")

    # Both tables are sized by the column count, so both the names and the data
    # move with it. The row count is the first field of the column table.
    name_table_at = COLUMN_TABLE_OFFSET + n_columns * COLUMN_TABLE_ENTRY_SIZE
    first_data_at = name_table_at + n_columns * NAME_FIELD_SIZE
    if len(data) < first_data_at:
        raise NotAQdaFile(
            f"{path.name} declares {n_columns} columns but ends before its "
            "column table does"
        )
    n_rows = int(np.frombuffer(data, dtype=">u4", count=1, offset=COLUMN_TABLE_OFFSET)[0])

    notes: list[str] = []
    if len(records) != n_columns:
        # Reported, not corrected. The records are what actually delimit the
        # data, so they win -- but a header disagreeing with the body is worth
        # knowing about before the numbers are used for anything.
        notes.append(
            f"header declares {n_columns} columns; {len(records)} column records "
            "found. The records were used."
        )

    columns: dict[str, np.ndarray] = {}
    lengths: list[int] = []
    unreadable: list[str] = []
    blanks = 0
    for index, record_at in enumerate(records):
        start = first_data_at if index == 0 else records[index - 1] + COLUMN_RECORD_SIZE

        # The name in the record, not the name at this position in the table.
        # They diverge the moment a column has no record: everything after it
        # shifts by one, and the numbers end up under a neighbour's name. That
        # failure is invisible in the output -- a temperature column labelled
        # "stirring speed" plots perfectly well.
        name = _record_name(data, record_at) or f"column_{index + 1}"

        resolved = _resolve_column(data, start, record_at, n_rows)
        if resolved is None:
            # Reported under the column's own name, because "this column could
            # not be read" and "this column is empty" are different answers.
            unreadable.append(name)
            values = np.empty(0, dtype=np.float64)
        else:
            rows, mask = resolved
            values = np.frombuffer(
                data, dtype=">f4", count=rows, offset=record_at - rows * BYTES_PER_ROW
            ).astype(np.float64)
            if rows and not mask.all():
                # A blank cell holds whatever was last in that slot. It is not a
                # measurement and must not arrive as one.
                values = np.where(mask, values, np.nan)
                blanks += int((~mask).sum())

        # An unnamed, empty column is a spare slot in the application's grid,
        # not a measurement. Carrying eight of them into a frame gives every
        # plot a column list mostly made of nothing.
        if values.size == 0 and not _record_name(data, record_at):
            continue
        while name in columns:
            name = f"{name}_{index + 1}"
        columns[name] = values
        lengths.append(values.size)

    if not columns:
        return QdaFile(
            path=path,
            frame=pd.DataFrame(),
            n_columns_declared=n_columns,
            n_rows_declared=n_rows,
            notes=notes + ["the file holds no named or filled columns"],
        )

    # Columns of unequal length are normal here: KaleidaGraph lets a column stop
    # short. They are padded rather than truncated, so a shorter column loses
    # nothing and the gap is visible as NaN instead of silently shortening every
    # other column to match.
    height = max(lengths) if lengths else 0
    if len(set(lengths)) > 1:
        notes.append(
            "columns have different lengths ("
            + ", ".join(f"{name}={values.size}" for name, values in columns.items())
            + "); shorter ones are padded with NaN"
        )
    padded = {
        name: np.concatenate([values, np.full(height - values.size, np.nan)])
        for name, values in columns.items()
    }

    if blanks:
        notes.append(
            f"{blanks} cell(s) were blank in KaleidaGraph and are NaN here, not "
            "the number that happened to be sitting in the slot"
        )
    if unreadable:
        notes.append(
            "could not be read as columns: " + ", ".join(unreadable)
            + ". The bytes where their values would be are not laid out as a "
            "column -- a stored plot, or a column type this reader does not "
            "handle, such as the text elapsedtime column an ARGEN export puts "
            "second."
        )
    notes.append(
        "values are single precision as stored by KaleidaGraph: about seven "
        "significant digits, whatever is printed downstream"
    )
    return QdaFile(
        path=path,
        frame=pd.DataFrame(padded),
        n_columns_declared=n_columns,
        n_rows_declared=n_rows,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# .QPC -- the plot file
# ---------------------------------------------------------------------------
#
# A .QPC is the graph, and it holds no data at all. The examples shipped with
# the application are 16-28 KB against multi-megabyte .QDA files, they contain
# none of the column records above, and they name their source by filename. So
# what can be recovered is what was plotted and how -- axis titles, series
# labels, the curve fit with the initial guesses it was given, and which data
# file it came from. The numbers are in the .QDA, which reads.
#
# Text is stored two different ways in the same file, and it took a wrong
# reader to notice:
#
# * Labels are **Pascal strings**: a length byte, then that many characters.
#   They are not NUL-terminated, and a short value written over a longer one
#   leaves the tail of the old value in place -- which is why a naive scan
#   reports "Displacement (mm)ress (MPa)", the second half being what is left
#   of "Remote Applied Stress (MPa)". The length byte is authoritative.
# * Fit equations are plain C strings, so the length-byte scan finds only
#   shifted fragments of them and never the whole thing. They are matched by
#   shape instead.
#
# Classification below is best-effort and says so. Anything not confidently
# placed is returned verbatim rather than relabelled or dropped.

QPC_HEADER = bytes.fromhex("000f")

# Fonts are named in every file and are never what anyone is asking about.
_FONT_NAMES = {
    "symbol", "helvetica", "arial", "geneva", "times", "times new roman",
    "courier", "courier new", "palatino", "monaco",
}

# A fit as KaleidaGraph writes it: parameters m0..m9, and the initial guesses
# appended after semicolons.
_FIT_SHAPE = re.compile(r"m\d")
_FIT_GUESS = re.compile(r"m\d\s*=")

# An equation begins with a term, never with the operator or semicolon that
# joins it to the one before. Runs that start mid-expression are fragments of a
# neighbour -- "; m3=1; m4=1" is the tail of a fit, not a fit.
_EQUATION_START = re.compile(r"^[A-Za-z0-9(]")

# A label is text somebody typed. Either it has a space in it, or it is a single
# plain word; anything else with punctuation woven through it is a fragment of
# compiled 68k, of which these files carry a great deal.
_LABEL_SHAPE = re.compile(r"^[A-Za-z0-9]")

# "Housing Starts 2:55:06 PM 3/14/00" -- the source table and when it was read.
_SOURCE_STAMP = re.compile(r"\d{1,2}:\d{2}:\d{2}\s*[AP]M\s*\d{1,2}/\d{1,2}/\d{2,4}")

# Menu items copied into the file alongside a saved script. Dropped by shape
# rather than by a list of every command the application has.
_UI_CHROME = re.compile(r"^(Undo|Redo)\s|\.\.\.$|scroll bar$")

MIN_TEXT = 4
MAX_TEXT = 200


class NotAQpcFile(ValueError):
    """A file that is not a KaleidaGraph plot."""


@dataclass
class QpcFile:
    """One KaleidaGraph plot file, read for what it says about itself.

    ``labels`` holds axis titles and series names together, in the order the
    file stores them, because nothing found so far distinguishes the two
    reliably. In the files examined the axis titles come first, but three
    examples is not a rule, and inventing the distinction would put a series
    name on an axis in whatever writes this up.
    """

    path: Path
    labels: list[str] = field(default_factory=list)
    fit_equations: list[str] = field(default_factory=list)
    source_references: list[str] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)
    other_text: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.labels or self.fit_equations or self.source_references)


def _pascal_strings(data: bytes) -> list[tuple[int, str]]:
    """Every length-prefixed string in the file, without the overlaps.

    A length byte followed by that many printable bytes occurs by chance inside
    binary, and inside a long C string it occurs at almost every offset -- one
    fit equation yields thirty shifted fragments of itself. Candidates are
    therefore taken longest-first and a candidate overlapping one already taken
    is discarded, which collapses each cluster to its single longest reading.
    """
    candidates: list[tuple[int, int, str]] = []
    for index in range(len(data) - 1):
        length = data[index]
        if not MIN_TEXT <= length <= MAX_TEXT:
            continue
        raw = data[index + 1 : index + 1 + length]
        if len(raw) != length or not all(32 <= byte < 127 for byte in raw):
            continue
        text = raw.decode("ascii")
        if re.search(r"[A-Za-z]{3}", text):
            candidates.append((index + 1, length, text))

    kept: list[tuple[int, int, str]] = []
    for start, length, text in sorted(candidates, key=lambda item: (-item[1], item[0])):
        if any(start < k_start + k_len and k_start < start + length
               for k_start, k_len, _ in kept):
            continue
        kept.append((start, length, text))
    return [(start, text) for start, _length, text in sorted(kept)]


def _c_strings(data: bytes) -> list[str]:
    """Printable runs, used for the fit equations the length scan cannot see."""
    return [
        match.group().decode("ascii")
        for match in re.finditer(rb"[\x20-\x7e]{8,}", data)
    ]


def read_qpc(path: str | Path) -> QpcFile:
    """Read what a KaleidaGraph ``.QPC`` says about the plot it holds."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 4:
        raise NotAQpcFile(f"{path.name} is too short to be a KaleidaGraph plot")

    # Told apart by what a data file has and a plot file does not: columns. The
    # header version was the obvious candidate and is wrong -- 13 of the 15
    # example plots open with 000f but two older ones open with 000c, which is
    # also what every data file opens with. Column records are categorical:
    # every .QDA examined has them, no .QPC has any.
    if re.search(re.escape(COLUMN_RECORD_SIGNATURE), data):
        raise NotAQpcFile(
            f"{path.name} holds column records, so it is a KaleidaGraph *data* "
            "file rather than a plot. Open it as a table -- it has the numbers "
            "in it."
        )
    if data[:2] not in (QPC_HEADER, QDA_HEADER):
        raise NotAQpcFile(f"{path.name} is not a KaleidaGraph file")

    found = QpcFile(path=path)
    seen: set[str] = set()
    chrome = 0

    # Fits first, from the whole-run scan, so the complete equation is taken
    # rather than the fragment the length scan finds.
    for text in _c_strings(data):
        stripped = text.strip()
        if not (_FIT_GUESS.search(stripped) and _EQUATION_START.match(stripped)):
            continue
        if stripped not in seen:
            seen.add(stripped)
            found.fit_equations.append(stripped)

    for _offset, text in _pascal_strings(data):
        stripped = text.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)

        if stripped.lower() in _FONT_NAMES:
            found.fonts.append(stripped)
        elif _SOURCE_STAMP.search(stripped):
            found.source_references.append(stripped)
        elif _FIT_GUESS.search(stripped) and _EQUATION_START.match(stripped):
            found.fit_equations.append(stripped)
        elif _FIT_SHAPE.search(stripped):
            # Fit-shaped but carrying no initial guesses: the analytic partial
            # derivatives KaleidaGraph stores beside an equation, or a fragment
            # of one. Kept, because they are part of how the fit was defined,
            # but never offered as "the equation" and never as a label.
            found.other_text.append(stripped)
        elif _UI_CHROME.search(stripped):
            chrome += 1
        elif (
            _LABEL_SHAPE.match(stripped)
            and re.fullmatch(r"[A-Za-z0-9 ()\[\]{}#/.,:;%+*=&'\"?!|-]+", stripped)
            and (" " in stripped or stripped.isalnum())
        ):
            found.labels.append(stripped)
        else:
            found.other_text.append(stripped)

    # Fullest first. A stray non-printable byte inside an equation splits the
    # run that recovers it, leaving a tail that is still a valid-looking fit --
    # "m4*m4)); m1=1; m2=1" reads like one until you notice it begins mid-term.
    # Ordering by length puts the complete reading at the top without throwing
    # the fragment away.
    found.fit_equations.sort(key=len, reverse=True)

    found.notes.append(
        "a .QPC holds the plot, not the data it plots: the numbers are in the "
        ".QDA it was built from"
    )
    found.notes.append(
        "axis titles and series names are not distinguished -- the format does "
        "not separate them anywhere this reader can see, so they are listed in "
        "file order"
    )
    if chrome:
        found.notes.append(
            f"{chrome} application menu string(s) stored alongside a saved "
            "script were left out"
        )
    if found.other_text:
        found.notes.append(
            "some recovered text did not classify and is listed verbatim rather "
            "than guessed at"
        )
    return found


__all__ = [
    "COLUMN_RECORD_SIGNATURE",
    "COLUMN_RECORD_SIZE",
    "NotAQdaFile",
    "NotAQpcFile",
    "QDA_HEADER",
    "QPC_HEADER",
    "QdaFile",
    "QpcFile",
    "read_qda",
    "read_qpc",
]
