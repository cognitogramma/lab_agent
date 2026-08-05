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
        # 2: adds declarations, declaration_conflicts, paths, blocked, and
        # censored_indices. Version 1 records predate the path register, so an
        # absent "paths" key means "not recorded", not "no branches taken".
        "schema_version": 2,
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
        # What a person declared this run to be, and where the data disagreed.
        # A conflict is recorded whether or not it was resolved -- an
        # unresolved one is the reason the run should not have been emitted.
        "declarations": {
            key: json.loads(d.model_dump_json())
            for key, d in state.declarations.items()
        },
        "declaration_conflicts": [
            json.loads(c.model_dump_json()) for c in state.declaration_conflicts
        ],
        # The branch taken at every question the design leaves open. Without
        # this, an unanswered question is indistinguishable from an answered one
        # in the finished record.
        "paths": [json.loads(p.model_dump_json()) for p in state.paths],
        "blocked": state.blocked,
        "censored_indices": state.censored_indices,
        "anomalies": [json.loads(a.model_dump_json()) for a in state.anomalies],
        "splice": json.loads(state.splice.model_dump_json()),
        # Enough to regenerate exactly what was subtracted, or to see that
        # nothing was. The candidate set is recorded whether or not any of it
        # was selected: what was on offer is part of how the decision was made.
        "baseline": {
            "transition_region": state.transition_region,
            "candidates": [
                json.loads(c.model_dump_json()) for c in state.anchor_candidates
            ],
            "anchors_selected": state.anchors_selected,
            "selected_by": state.anchor_selected_by,
            "drift_model": (
                json.loads(state.drift_model.model_dump_json())
                if state.drift_model
                else None
            ),
            "applied": state.baseline_applied,
        },
        # M0 divides every value in the aggregation index, so which window was
        # used -- and how much the choice mattered -- has to be reconstructible.
        "m0": (
            {
                "method": state.m0.method,
                "value": state.m0.value,
                "index_range": list(state.m0.index_range),
                "n_points": state.m0.n_points,
                "first_point_value": state.m0.first_point_value,
                "sensitivity": state.m0.sensitivity,
                "blocked": state.m0.blocked,
                "escalations": state.m0.escalations,
            }
            if state.m0 is not None
            else None
        ),
        "judge": {
            "status": state.judge_status,
            "llm_lane": (
                json.loads(state.llm_lane.model_dump_json()) if state.llm_lane else None
            ),
            "vlm_lane": (
                json.loads(state.vlm_lane.model_dump_json()) if state.vlm_lane else None
            ),
            "lanes_disagreed": state.lanes_disagreed,
            "fallback_taken": state.fallback_taken,
        },
        "execution_history": state.execution_history,
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
