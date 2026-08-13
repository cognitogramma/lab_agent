"""Historical run memory registry and sidecar indexer.

Design section: Cross-run research memory.
Scans and indexes `.provenance.json` sidecar files from past processing runs,
allowing researchers to search prior results, retrieve historical benchmarks,
and compare a current trace against past measurements.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from polyrmc.provenance import SIDECAR_SUFFIX, load_sidecar


@dataclass
class RunSummary:
    """Compact summary of a historical run for rapid search and comparison."""

    run_id: str
    source_file: str
    written_at: str
    provenance_path: str
    experiment_type: str = "unknown"
    sample_name: str = "unknown"
    polymer_type: str = "unknown"
    mw: float | None = None
    a2: float | None = None
    a2_mw: float | None = None
    k_d: float | None = None
    window_length: int | None = None
    anomalies_count: int = 0
    warnings_count: int = 0
    declarations: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    dn_dc: float | None = None
    polymer_type_basis: str = "unrecorded"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The two values this lab uses, and what each one says about the material.
# Deliberately exact rather than nearest-match: a run at 0.165 was configured
# for something these two entries do not describe, and rounding it to whichever
# is closer would put a material in a search result that nobody ever declared.
DN_DC_CLASS = {
    0.185: "protein",
    0.150: "polysaccharide",
}

_DN_DC_TOLERANCE = 1e-6


def _class_for_dn_dc(value: Any) -> str | None:
    """The material class implied by a refractive-index increment, if any."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    for known, name in DN_DC_CLASS.items():
        if abs(number - known) <= _DN_DC_TOLERANCE:
            return name
    return None


def summary_from_sidecar(record: dict[str, Any], path: str | Path) -> RunSummary:
    """Parse a `.provenance.json` sidecar dict into a RunSummary."""
    run_id = record.get("run_id", Path(path).stem.replace(SIDECAR_SUFFIX, ""))
    source_file = record.get("source_file", "")
    written_at = record.get("written_at", "")

    config = record.get("config", {})
    experiment_type = config.get("experiment_type", "unknown")

    declarations_raw = record.get("declarations", {})
    declarations: dict[str, Any] = {}
    for k, v in declarations_raw.items():
        if isinstance(v, dict) and "value" in v:
            declarations[k] = v["value"]
        else:
            declarations[k] = v

    # Falls back to the run id, which in practice is where the sample actually
    # lives: these sidecars are named for what was measured
    # ("lira-2000mMGd", "liraglutide_sample_2_(unstable)_500mM_NaCl") and almost
    # none of them carry a sample_name declaration. Reporting "unknown" for
    # every historical run makes the search that this index exists for useless,
    # and the run id is a worse answer than a declaration but a far better one
    # than nothing.
    sample_name = str(
        declarations.get("sample_name")
        or config.get("sample_name")
        or declarations.get("polymer_type")
        or run_id
        or "unknown"
    )
    # The refractive-index increment a run was processed under, which is the
    # only signal in this archive that bears on what the material was: none of
    # the 120 historical sidecars carries a declaration at all, so without this
    # every search by material returns nothing.
    #
    # It is recorded as an inference and labelled as one. dn_dc is a property of
    # the analysis, not of the sample -- it is what somebody typed in -- so a
    # run configured at 0.150 is a run somebody treated as a polysaccharide,
    # which is a weaker claim than the run being one. That distinction is not
    # pedantry here: 39 runs in this archive sit at 0.150 while being named for
    # a peptide, and exactly one of those two facts is wrong. Callers that
    # report a material to a person are expected to report the basis with it.
    dn_dc = (config.get("optical") or {}).get("dn_dc")
    declared_type = declarations.get("polymer_type") or declarations.get("sample_type")
    if declared_type:
        polymer_type = str(declared_type)
        polymer_type_basis = "declared"
    elif (implied := _class_for_dn_dc(dn_dc)) is not None:
        polymer_type = implied
        polymer_type_basis = f"inferred from dn_dc={float(dn_dc):.3f}"
    else:
        polymer_type = "unknown"
        polymer_type_basis = "unrecorded"

    parameters = record.get("parameters") or {}
    mw = parameters.get("mw")
    a2 = parameters.get("a2")
    a2_mw = parameters.get("a2_mw")
    k_d = parameters.get("k_d")

    m0_data = record.get("m0") or {}
    window_length = config.get("smoothing", {}).get("window_length")
    if window_length is None and m0_data:
        window_length = m0_data.get("window_length")

    anomalies = record.get("anomalies", [])
    warnings = record.get("warnings", [])

    return RunSummary(
        run_id=run_id,
        source_file=source_file,
        written_at=written_at,
        provenance_path=str(path),
        experiment_type=experiment_type,
        sample_name=sample_name,
        polymer_type=polymer_type,
        mw=float(mw) if mw is not None else None,
        a2=float(a2) if a2 is not None else None,
        a2_mw=float(a2_mw) if a2_mw is not None else None,
        k_d=float(k_d) if k_d is not None else None,
        window_length=int(window_length) if window_length is not None else None,
        anomalies_count=len(anomalies),
        warnings_count=len(warnings),
        declarations=declarations,
        warnings=warnings,
        dn_dc=float(dn_dc) if dn_dc is not None else None,
        polymer_type_basis=polymer_type_basis,
    )


class RunRegistry:
    """In-memory index of past runs scanned from sidecar files."""

    def __init__(self, runs_dir: str | Path | None = None) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir else None
        self.summaries: list[RunSummary] = []
        if self.runs_dir and self.runs_dir.exists():
            self.scan(self.runs_dir)

    def scan(self, directory: str | Path) -> list[RunSummary]:
        """Scan a directory recursively for `.provenance.json` sidecar files."""
        dir_path = Path(directory)
        new_summaries: list[RunSummary] = []
        if not dir_path.exists():
            return new_summaries

        # Match *.provenance.json
        for path in dir_path.glob(f"**/*{SIDECAR_SUFFIX}"):
            try:
                record = load_sidecar(path)
                summary = summary_from_sidecar(record, path)
                new_summaries.append(summary)
            except (json.JSONDecodeError, OSError, KeyError):
                continue

        # Sort by written_at descending (newest first)
        new_summaries.sort(key=lambda s: s.written_at, reverse=True)
        self.summaries = new_summaries
        return self.summaries

    def search(
        self,
        query: str | None = None,
        sample_name: str | None = None,
        experiment_type: str | None = None,
        min_mw: float | None = None,
        max_mw: float | None = None,
    ) -> list[RunSummary]:
        """Filter indexed run summaries by query parameters."""
        results: list[RunSummary] = []
        for s in self.summaries:
            if sample_name and sample_name.lower() not in s.sample_name.lower() and sample_name.lower() not in s.polymer_type.lower():
                continue
            if experiment_type and experiment_type.lower() != s.experiment_type.lower():
                continue
            if min_mw is not None and (s.mw is None or s.mw < min_mw):
                continue
            if max_mw is not None and (s.mw is None or s.mw > max_mw):
                continue
            if query:
                q = query.lower()
                text_blob = f"{s.run_id} {s.sample_name} {s.polymer_type} {s.experiment_type} {s.source_file} {s.declarations}".lower()
                if q not in text_blob:
                    continue
            results.append(s)
        return results

    def compare(
        self, target: RunSummary | dict[str, Any], candidates: Sequence[RunSummary] | None = None
    ) -> dict[str, Any]:
        """Compare a target run against a benchmark set of historical runs."""
        pool = list(candidates) if candidates is not None else self.summaries
        if not pool:
            return {"status": "no_historical_runs", "message": "No past runs available for comparison."}

        target_name = (
            target.sample_name
            if isinstance(target, RunSummary)
            else target.get("sample_name", "unknown")
        )
        target_a2_mw = (
            target.a2_mw
            if isinstance(target, RunSummary)
            else target.get("a2_mw")
        )
        target_kd = (
            target.k_d
            if isinstance(target, RunSummary)
            else target.get("k_d")
        )

        # Filter pool for same or similar sample if possible
        matched = [s for s in pool if target_name.lower() in s.sample_name.lower() or target_name.lower() in s.polymer_type.lower()]
        if not matched:
            matched = pool  # fallback to all runs

        valid_a2_mw = [s.a2_mw for s in matched if s.a2_mw is not None]
        valid_kd = [s.k_d for s in matched if s.k_d is not None]

        summary_stats: dict[str, Any] = {
            "historical_sample_count": len(matched),
            "matched_by_sample_name": len(matched) if matched != pool else 0,
        }

        if valid_a2_mw:
            mean_a2_mw = sum(valid_a2_mw) / len(valid_a2_mw)
            summary_stats["mean_a2_mw"] = mean_a2_mw
            if target_a2_mw is not None:
                summary_stats["target_a2_mw"] = target_a2_mw
                summary_stats["a2_mw_diff_pct"] = (
                    ((target_a2_mw - mean_a2_mw) / mean_a2_mw) * 100.0 if mean_a2_mw != 0 else 0.0
                )

        if valid_kd:
            mean_kd = sum(valid_kd) / len(valid_kd)
            summary_stats["mean_kd"] = mean_kd
            if target_kd is not None:
                summary_stats["target_kd"] = target_kd
                summary_stats["kd_diff_pct"] = (
                    ((target_kd - mean_kd) / mean_kd) * 100.0 if mean_kd != 0 else 0.0
                )

        return summary_stats


@dataclass
class FigureRecord:
    """A labeled figure from a past publication."""
    image: str
    filename: str
    date: str
    dateKind: str
    classification: str
    organization: str
    projectType: str
    methods: list[str]
    measured: list[str]
    sampleType: str
    proteinClass: str
    sampleName: str
    goals: list[str]
    notes: str


class FigureRegistry:
    """In-memory index of labeled historical figures from ledger.json."""

    def __init__(self, ledger_path: str | Path | None = None) -> None:
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.records: list[FigureRecord] = []
        if self.ledger_path and self.ledger_path.exists():
            self.load()

    def load(self) -> None:
        """Load records from ledger.json."""
        if not self.ledger_path or not self.ledger_path.exists():
            return
        
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return
            
            for item in data:
                # Fallback to string split if it was stored as semi-colon delimited or list
                def _parse_list(val: Any) -> list[str]:
                    if isinstance(val, list):
                        return val
                    if isinstance(val, str):
                        return [x.strip() for x in val.split(";") if x.strip()]
                    return []
                
                record = FigureRecord(
                    image=item.get("image", ""),
                    filename=item.get("filename", ""),
                    date=item.get("date", ""),
                    dateKind=item.get("dateKind", ""),
                    classification=item.get("classification", ""),
                    organization=item.get("organization", ""),
                    projectType=item.get("projectType", ""),
                    methods=_parse_list(item.get("methods", [])),
                    measured=_parse_list(item.get("measured", [])),
                    sampleType=item.get("sampleType", ""),
                    proteinClass=item.get("proteinClass", ""),
                    sampleName=item.get("sampleName", ""),
                    goals=_parse_list(item.get("goals", [])),
                    notes=item.get("notes", ""),
                )
                self.records.append(record)
        except Exception:
            pass

    def search(
        self,
        query: str | None = None,
        method: str | None = None,
        sample_type: str | None = None,
        goal: str | None = None
    ) -> list[FigureRecord]:
        """Search labeled figures."""
        results: list[FigureRecord] = []
        for r in self.records:
            if method and method.lower() not in [m.lower() for m in r.methods]:
                continue
            if sample_type and sample_type.lower() != r.sampleType.lower():
                continue
            if goal:
                goal_lower = goal.lower()
                if not any(goal_lower in g.lower() for g in r.goals):
                    continue
            
            if query:
                q = query.lower()
                text_blob = f"{r.filename} {r.sampleName} {r.projectType} {r.notes} {' '.join(r.measured)}".lower()
                if q not in text_blob:
                    continue
            results.append(r)
        return results


__all__ = ["RunRegistry", "RunSummary", "summary_from_sidecar", "FigureRegistry", "FigureRecord"]
