"""Reading a finished run back off disk, well enough to draw it again.

A processed run leaves two files side by side: the boundary CSV and the
provenance sidecar. Between them they already hold everything the figure set
needs -- the trace itself, and the marks that say which parts of it are not
measurements. Until now nothing read them back, so a figure could only be drawn
during the run that produced it. Asking about a run from last month meant
reprocessing it, which is both slow and a quiet lie: reprocessing under today's
code is not the same act as reading what was recorded then.

Nothing here recomputes anything. The CSV is read as written and the sidecar is
read as written; where they disagree with what today's pipeline would produce,
the file wins, because the file is the record.

The one thing this module will not do is invent the marks it cannot find.
Sidecars written under schema 1 predate the ND-filter register, so an older run
has no record of whether the filter moved. That is reported as unknown -- see
:attr:`PastRun.unrecorded` -- and never as zero. A figure claiming a clean run
because the file predates the check is worse than one that says it does not
know.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from polyrmc import paths
from polyrmc.config import RunConfig
from polyrmc.io_csv import read_processed_csv
from polyrmc.state import AnomalyRecord, RunState, SpliceRecord

SIDECAR_SUFFIX = ".provenance.json"


@dataclass
class PastRun:
    """One finished run, read back from its own files.

    ``state`` carries only what the figures read: the run id, the splice, the
    anomaly records, and the ND-filter samples. The rest of a live ``RunState``
    is left at its defaults, because the arrays are in ``frame`` and nothing
    downstream of a figure should be reading a rehydrated state for anything
    else.
    """

    run_id: str
    frame: pd.DataFrame
    state: RunState
    config: RunConfig | None
    sidecar: dict
    csv_path: Path
    sidecar_path: Path
    # Keyed by what is missing, valued by how to say so to a reader. A caller
    # drawing a figure keeps only the keys that figure depends on: a config
    # that will not parse is worth reporting, but printing it under a figure
    # that never reads one just teaches people to ignore the small red text.
    unrecorded: dict[str, str] = field(default_factory=dict)

    @property
    def schema_version(self) -> int:
        return int(self.sidecar.get("schema_version", 1))

    @property
    def label(self) -> str:
        """What to call this run when its id is not unique.

        A run id can name several sidecars, one per channel, so the file stem is
        the only identifier guaranteed to point at one figure's worth of data.
        """
        name = self.sidecar_path.name[: -len(SIDECAR_SUFFIX)]
        return name if name != self.run_id else self.run_id


def _error_fields(error: Exception) -> list[str]:
    """Which fields a validation failure actually named.

    All of them, not just the first. A record can drift from the schema in
    several places at once, and reporting only the first turns a two-line fix
    into two rounds of guessing.
    """
    getter = getattr(error, "errors", None)
    if not callable(getter):
        return []
    try:
        raw = getter()
    except Exception:  # noqa: BLE001 - a diagnostic must not raise
        return []
    names: list[str] = []
    for item in raw:
        name = ".".join(str(part) for part in (item.get("loc") or ()))
        if name and name not in names:
            names.append(name)
    return names[:4]


def _csv_for(sidecar_path: Path) -> Path:
    """The boundary CSV beside a sidecar.

    Derived from the sidecar's own name rather than from the ``source_file``
    field: that field records the *raw* instrument file, which is usually not
    on this machine at all, and following it would look up the wrong thing on
    the runs that matter most.
    """
    name = sidecar_path.name
    if name.endswith(SIDECAR_SUFFIX):
        name = name[: -len(SIDECAR_SUFFIX)]
    return sidecar_path.with_name(f"{name}.csv")


def find_run(run_id: str, runs_dir: str | Path | None = None) -> Path:
    """Locate the sidecar for one run id.

    The file stem is tried first and is the only identifier that is reliably
    unique. A recorded ``run_id`` is not: one run processed on three channels
    writes three sidecars that all record the same id, and ``example-001``
    genuinely names three different files here. Resolving that to whichever
    happened to sort first would draw one channel under a title claiming the
    run, which is the kind of wrong figure nobody checks.
    """
    directory = Path(runs_dir) if runs_dir else paths.runs_dir()
    if not directory.is_dir():
        raise FileNotFoundError(f"no runs directory at {directory}")

    exact_stem = directory / f"{run_id}{SIDECAR_SUFFIX}"
    if exact_stem.exists():
        return exact_stem

    id_matches: list[Path] = []
    for candidate in sorted(directory.glob(f"*{SIDECAR_SUFFIX}")):
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("run_id") == run_id:
            id_matches.append(candidate)

    if len(id_matches) == 1:
        return id_matches[0]
    if len(id_matches) > 1:
        names = ", ".join(p.name[: -len(SIDECAR_SUFFIX)] for p in id_matches)
        raise FileNotFoundError(
            f"{run_id!r} names {len(id_matches)} processed files, one per channel "
            f"({names}). Name one of those exactly -- they are different "
            "measurements and only one of them is the figure you meant."
        )
    raise FileNotFoundError(f"no run {run_id!r} under {directory}")


def load_past_run(run_id: str, runs_dir: str | Path | None = None) -> PastRun:
    """Read one finished run back into a frame and the marks that annotate it."""
    sidecar_path = find_run(run_id, runs_dir)
    record = json.loads(sidecar_path.read_text(encoding="utf-8"))

    csv_path = _csv_for(sidecar_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{sidecar_path.name} has no boundary CSV beside it ({csv_path.name}). "
            "The run's provenance survived and its data did not, so there is "
            "nothing to draw."
        )
    frame, _ = read_processed_csv(csv_path)

    ingest = record.get("ingest") or {}
    unrecorded: dict[str, str] = {}

    # Absent is not zero. A schema-1 sidecar has no ND-filter register at all,
    # which is a different fact from a run where the filter held still, and the
    # figure has to be able to tell a reader which one it is looking at.
    if "nd_filter_changes" in ingest:
        nd_filter_changes = list(ingest["nd_filter_changes"] or [])
    else:
        nd_filter_changes = []
        unrecorded["nd_filter_changes"] = (
            "no ND-filter record, so an unmarked scale break cannot be ruled out"
        )

    anomalies: list[AnomalyRecord] = []
    unreadable = 0
    for raw in record.get("anomalies") or []:
        try:
            anomalies.append(AnomalyRecord.model_validate(raw))
        except Exception:  # noqa: BLE001 - one unreadable record is not a dead run
            unreadable += 1
    if unreadable:
        unrecorded["anomalies"] = f"{unreadable} anomaly record(s) could not be read"

    try:
        splice = SpliceRecord.model_validate(record.get("splice") or {})
    except Exception:  # noqa: BLE001
        splice = SpliceRecord()
        unrecorded["splice"] = "excision record unreadable, so gaps are not drawn"

    state = RunState(
        run_id=record.get("run_id", run_id),
        source_file=record.get("source_file", ""),
        nd_filter_changes=nd_filter_changes,
        anomalies=anomalies,
        splice=splice,
    )
    state.time_s = frame["seconds"].to_numpy(dtype=float)
    state.spliced_signal = frame["ls8"].to_numpy(dtype=float)
    state.smoothed_signal = frame["ls8_smoothed"].to_numpy(dtype=float)
    state.concentration = frame["concentration"].to_numpy(dtype=float)

    # A recorded config need not validate against today's model, and 65 of the
    # runs in this lab's own history do not: they declare an experiment_type of
    # "fixed_concentration", which the current schema does not define. That is
    # not a corrupt file, it is a record written before a rename, and refusing
    # to read it would make more than half the archive undrawable.
    #
    # It is deliberately not translated to the nearest current spelling. Which
    # experiment a past run actually was is a statement about the experiment,
    # and this module reads records rather than deciding what they meant. The
    # figures that need a config say so and are refused; the overlay does not
    # read one and works either way.
    config: RunConfig | None
    try:
        config = RunConfig.model_validate(record.get("config") or {})
    except Exception as error:  # noqa: BLE001 - any schema drift lands here
        config = None
        fields = _error_fields(error)
        unrecorded["config"] = (
            "the recorded configuration does not validate against the current "
            "schema" + (f" ({', '.join(fields)})" if fields else "")
        )

    return PastRun(
        run_id=state.run_id,
        frame=frame,
        state=state,
        config=config,
        sidecar=record,
        csv_path=csv_path,
        sidecar_path=sidecar_path,
        unrecorded=unrecorded,
    )


def list_past_runs(runs_dir: str | Path | None = None) -> list[str]:
    """Every run id that can actually be drawn, newest first.

    Two invariants, both of which exist because the caller is usually a model
    choosing what to draw from this list. Every string returned can be passed
    straight back to :func:`find_run` -- so a run id shared by several channels
    is reported by file stem, which is unique, rather than by an id that would
    be refused as ambiguous the moment it was used. And a sidecar with no CSV
    beside it is left out rather than listed and then refused on load: offering
    something undrawable wastes a turn and reads as a bug.
    """
    directory = Path(runs_dir) if runs_dir else paths.runs_dir()
    if not directory.is_dir():
        return []

    found: list[tuple[str, str, str]] = []
    for candidate in directory.glob(f"*{SIDECAR_SUFFIX}"):
        if not _csv_for(candidate).exists():
            continue
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stem = candidate.name[: -len(SIDECAR_SUFFIX)]
        run_id = str(record.get("run_id") or stem)
        found.append((str(record.get("written_at", "")), run_id, stem))

    counts: dict[str, int] = {}
    for _, run_id, _stem in found:
        counts[run_id] = counts.get(run_id, 0) + 1

    found.sort(reverse=True)
    return [stem if counts[run_id] > 1 else run_id for _, run_id, stem in found]


__all__ = [
    "PastRun",
    "SIDECAR_SUFFIX",
    "find_run",
    "list_past_runs",
    "load_past_run",
]
