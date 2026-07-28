"""Per-run provenance sidecar.

Reproducibility here is *managed*, not claimed away. Two runs may select
different smoothing windows. What makes that acceptable is that the model id
and prompt hash are pinned per run, the full loop trajectory including rejected
candidates is recorded, and :func:`replay_decisions` reproduces any historical
output from the record rather than by re-calling the model.

The sidecar carries the bookkeeping -- excised indices, chosen window,
judge-versus-fallback status, calibration constants actually used -- so the
data table stays lean while every number in it remains reconstructible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polyrmc.config import RunConfig
from polyrmc.state import JudgeDecision, LoopState, RunState

SIDECAR_SUFFIX = ".provenance.json"


def sidecar_path(output_csv: str | Path) -> Path:
    """Path of the sidecar that accompanies a processed CSV."""
    output_csv = Path(output_csv)
    return output_csv.with_suffix("").with_suffix(SIDECAR_SUFFIX)


def build_record(state: RunState, config: RunConfig) -> dict[str, Any]:
    """Assemble the full provenance record for a run."""
    return {
        "schema_version": 1,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "run_id": state.run_id,
        "source_file": state.source_file,
        "config": json.loads(config.model_dump_json()),
        "ingest": {
            "n_header_rows": state.n_header_rows,
            "time_format": state.time_format.value if state.time_format else None,
            "dropped_rows": state.dropped_rows,
            "nd_filter_changes": state.nd_filter_changes,
        },
        "anomalies": [json.loads(a.model_dump_json()) for a in state.anomalies],
        "splice": json.loads(state.splice.model_dump_json()),
        "loops": {
            name: json.loads(loop.model_dump_json())
            for name, loop in state.loops.items()
        },
        "fit": json.loads(state.fit.model_dump_json()) if state.fit else None,
        "parameters": (
            json.loads(state.parameters.model_dump_json()) if state.parameters else None
        ),
        "warnings": state.warnings,
    }


def write_sidecar(state: RunState, config: RunConfig, output_csv: str | Path) -> Path:
    """Write the sidecar next to the processed CSV."""
    path = sidecar_path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_record(state, config), indent=2, default=str), encoding="utf-8"
    )
    return path


def load_sidecar(path: str | Path) -> dict[str, Any]:
    """Read a provenance record back."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replay_decisions(record: dict[str, Any], loop_name: str) -> list[JudgeDecision]:
    """Recover the decisions a named loop made, for replay without a model call."""
    loop = record.get("loops", {}).get(loop_name)
    if loop is None:
        raise KeyError(f"no recorded loop named {loop_name!r} in this provenance record")
    return [JudgeDecision.model_validate(decision) for decision in loop["decisions"]]


def summarize_loop(loop: LoopState) -> str:
    """One-line human summary of how a loop resolved.

    Reported in run output so the judge-versus-fallback distinction is visible
    without opening the sidecar.
    """
    if loop.used_fallback:
        return (
            f"{loop.name}: fell back to the conservative option "
            f"({loop.resolved_value!r}) after {loop.iterations_used} iteration(s) "
            f"-- {loop.fallback_reason}"
        )
    n_rejected = sum(len(d.rejected_indices) for d in loop.decisions)
    return (
        f"{loop.name}: judge selected {loop.resolved_value!r} in "
        f"{loop.iterations_used} iteration(s), {n_rejected} candidate(s) rejected"
    )
