"""The ledger: schema, validation, and durable writes.

One record per image, written to ``ledger.json`` in the image folder. Real files
on disk, never browser storage.

Two things here are defensive on purpose.

**Writes are atomic.** Every save goes to a temporary file in the same directory
and is then renamed over the target. ``os.replace`` is atomic on POSIX and on
Windows, so a crash or a power cut mid-write leaves either the old ledger or the
new one, never a truncated file. A labelling session is hours of work that
exists nowhere else.

**Dates keep their precision.** Archive material often carries only a year, and
forcing a full date would mean inventing a month and a day that then look like
data. ``2021``, ``2021-06`` and ``2021-06-14`` are all valid and the precision
that was entered is recorded alongside the value.
"""

from __future__ import annotations

import calendar
import csv
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEDGER_NAME = "ledger.json"

# Field order for the CSV. Fixed so exports diff cleanly between sessions.
COLUMNS = (
    "image",
    "filename",
    "date",
    "datePrecision",
    "dateKind",
    "classification",
    "organization",
    "projectType",
    "methods",
    "measured",
    "measuredOther",
    "sampleType",
    "proteinClass",
    "sampleName",
    "concentration",
    "ph",
    "buffer",
    "temp",
    "goals",
    "goalDetails",
    "notes",
    "updated",
)

LIST_FIELDS = frozenset({"methods", "measured", "goals", "goalDetails"})

_DATE_PATTERNS = (
    (re.compile(r"^\d{4}$"), "year"),
    (re.compile(r"^\d{4}-\d{2}$"), "month"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "day"),
)


class ValidationError(ValueError):
    """A record was refused, with the reason a person needs to fix it."""


@dataclass
class DateCheck:
    precision: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def check_date(value: str) -> DateCheck:
    """Validate a partial or full date and report which precision it carries.

    Rejects dates that cannot exist -- ``2021-02-30`` parses as a shape and is
    still not a day -- because a bad date is worse than a blank one: it will sort
    and filter as though it were real.
    """
    text = (value or "").strip()
    if not text:
        return DateCheck("none")

    precision = ""
    for pattern, name in _DATE_PATTERNS:
        if pattern.match(text):
            precision = name
            break
    if not precision:
        return DateCheck(
            "none",
            f"{text!r} is not a date. Use YYYY, YYYY-MM, or YYYY-MM-DD.",
        )

    parts = [int(part) for part in text.split("-")]
    year = parts[0]
    if not 1800 <= year <= 2200:
        return DateCheck("none", f"year {year} is outside 1800-2200")
    if len(parts) >= 2 and not 1 <= parts[1] <= 12:
        return DateCheck("none", f"month {parts[1]:02d} does not exist")
    if len(parts) == 3:
        last = calendar.monthrange(year, parts[1])[1]
        if not 1 <= parts[2] <= last:
            return DateCheck(
                "none",
                f"{text} does not exist: {calendar.month_name[parts[1]]} {year} "
                f"has {last} days",
            )
    return DateCheck(precision)


def blank_record(image: str) -> dict[str, Any]:
    """An empty record for an image that has not been labelled yet."""
    record: dict[str, Any] = {field: "" for field in COLUMNS}
    record["image"] = image
    record["filename"] = Path(image).name
    for field in LIST_FIELDS:
        record[field] = []
    record["updated"] = ""
    return record


def normalise(record: dict[str, Any], image: str) -> dict[str, Any]:
    """Coerce an incoming record into the schema, and validate it.

    Unknown keys are dropped rather than stored: this file is going to be read
    by agents later, and a schema that quietly accumulates stray fields stops
    being a schema.
    """
    clean = blank_record(image)

    for field in COLUMNS:
        if field not in record:
            continue
        value = record[field]
        if field in LIST_FIELDS:
            if isinstance(value, str):
                value = [part for part in (v.strip() for v in value.split(",")) if part]
            clean[field] = [str(item) for item in (value or [])]
        elif field not in ("image", "filename", "datePrecision", "updated"):
            clean[field] = "" if value is None else str(value).strip()

    checked = check_date(clean["date"])
    if not checked.ok:
        raise ValidationError(checked.error)
    clean["datePrecision"] = checked.precision

    # proteinClass only means anything for a protein. Storing one against an RNA
    # sample would make a later query for globular proteins return RNA.
    if clean["sampleType"] != "Protein":
        clean["proteinClass"] = ""

    clean["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return clean


def is_logged(record: dict[str, Any]) -> bool:
    """Whether a record counts as done: a method, a sample type, and a goal."""
    return bool(record.get("methods")) and bool(record.get("sampleType")) and bool(
        record.get("goals")
    )


def load(folder: Path) -> dict[str, dict[str, Any]]:
    """Read an existing ledger, keyed by image path.

    A ledger from an earlier attempt is resumed rather than replaced. A corrupt
    one raises instead of being silently discarded -- overwriting hours of work
    because a file would not parse is not a recovery.
    """
    path = folder / LEDGER_NAME
    if not path.is_file():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValidationError(
            f"{path} exists but is not valid JSON ({error}). Move it aside if you "
            "want to start over; it is not being overwritten."
        ) from error

    records = data.get("records", data) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValidationError(f"{path} does not contain a list of records")

    out: dict[str, dict[str, Any]] = {}
    for entry in records:
        if not isinstance(entry, dict):
            continue
        image = str(entry.get("image") or entry.get("filename") or "").strip()
        if not image:
            continue
        merged = blank_record(image)
        for field in COLUMNS:
            if field in entry and entry[field] is not None:
                merged[field] = entry[field]
        merged["filename"] = Path(image).name
        out[image] = merged
    return out


def save(folder: Path, records: dict[str, dict[str, Any]]) -> Path:
    """Write the ledger atomically: temp file in the same directory, then rename.

    Same directory matters -- ``os.replace`` is only atomic within a filesystem,
    and a temp file in the system temp directory may be on another one.
    """
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / LEDGER_NAME
    payload = json.dumps(
        [records[key] for key in sorted(records)], indent=2, ensure_ascii=False
    )

    handle, temporary = tempfile.mkstemp(
        dir=str(folder), prefix=".ledger-", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(payload + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def to_csv(records: dict[str, dict[str, Any]]) -> str:
    """Flatten to CSV. Lists become semicolon-joined, since commas are the delimiter."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    for key in sorted(records):
        row = dict(records[key])
        for field in LIST_FIELDS:
            row[field] = "; ".join(row.get(field) or [])
        writer.writerow({field: row.get(field, "") for field in COLUMNS})
    return buffer.getvalue()


__all__ = [
    "COLUMNS",
    "LEDGER_NAME",
    "LIST_FIELDS",
    "DateCheck",
    "ValidationError",
    "blank_record",
    "check_date",
    "is_logged",
    "load",
    "normalise",
    "save",
    "to_csv",
]
