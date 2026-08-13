"""The tools the orchestrator can call, and what their results look like to it.

:mod:`polyrmc.tools` is the enforcement boundary -- it validates, and it
rejects. This module is the *description* of that boundary in the only
vocabulary a model can act on: a name, a schema, and a sentence saying when to
reach for it. Nothing here validates anything. Adding a guard in this file would
be a mistake worth naming, because a guard written next to a schema is a guard a
different caller walks straight past.

Three things live here and nowhere else.

**Schemas.** Deliberately plain -- object, properties, required, enum, items,
and nothing more. Providers disagree about the exotic corners of JSON Schema,
and a tool that binds against one model and silently fails to bind against
another is a portability claim that was never true.

**Tiers.** Every tool is ``read``, ``compute``, or ``commit``, matching the
three tiers the tool surface already documents. The tier is what the approval
policy reads, so the question "can this change my data" has one answer, declared
once, rather than a judgement call made at each call site.

**Rendering.** A tool returns Python; a model reads text. ``RunState`` alone is
a pydantic model wrapping eight numpy arrays, and handing it over verbatim would
spend the entire context window on a trace the agent is explicitly forbidden to
do arithmetic over. Every renderer here answers the same question: what is the
smallest thing that would let a competent colleague decide what to do next?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from polyrmc import paths
from polyrmc.memory.notes import NoteBook
from polyrmc.tools import Rejected, ToolSurface

Tier = Literal["read", "compute", "commit"]

# Where run history and figure labels are looked up when nothing overrides it.
# Both were hardcoded at their call sites before, to two different paths, which
# meant the chat session and the tool surface searched different ledgers. They
# now hang off the resolved project root, so the answer no longer depends on
# which directory the researcher started the session from.
DEFAULT_LEDGER_ENV = "POLYRMC_FIGURE_LEDGER"

MAX_ROWS = 25
"""Cap on rows in any list a tool renders.

There are already over a hundred sidecars in ``data/runs``. Rendering all of
them turns one search into most of a context window, and a model given a
hundred rows reasons over the first few anyway. The renderers say how many were
withheld so the agent can narrow the query rather than assume it saw everything.
"""


@dataclass
class ToolContext:
    """Everything the handlers act on."""

    surface: ToolSurface
    notes: NoteBook
    runs_dir: Path = field(default_factory=paths.runs_dir)
    ledger_path: Path | None = None
    author: str = "researcher"
    output_csv: Path | None = None


@dataclass
class Tool:
    """One callable action, as the model sees it."""

    name: str
    tier: Tier
    description: str
    parameters: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], str]

    @property
    def mutating(self) -> bool:
        return self.tier == "commit"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _object(properties: dict[str, Any] | None = None, required: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required),
    }


NO_ARGS = _object()


# --- renderers ---------------------------------------------------------------


def _truncated(rows: list[str], total: int) -> str:
    text = "\n".join(rows)
    if total > len(rows):
        text += f"\n... {total - len(rows)} more not shown; narrow the query."
    return text or "(none)"


def _number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def render_state(surface: ToolSurface) -> str:
    """The run at a glance: what is loaded, what is decided, what is blocking."""
    state = surface.state
    config = surface.config
    lines = [
        f"run_id: {state.run_id}",
        f"source_file: {state.source_file}",
        f"experiment_type: {config.experiment_type}",
        f"channel: {config.channel}",
    ]

    if state.raw_signal is None:
        lines.append("loaded: no -- call load_run before anything can be computed")
    else:
        lines.append(
            f"loaded: yes, {state.raw_signal.size} samples"
            + (f", time format {state.time_format.value}" if state.time_format else "")
        )
        lines.append(f"noise_floor: {_number(state.noise_floor)}")

    declared = ", ".join(sorted(state.declarations)) or "none"
    lines.append(f"declarations: {declared}")

    outstanding = [q.key for q in surface.declarations.unanswered()]
    if outstanding:
        lines.append(f"undeclared (required): {', '.join(outstanding)}")

    unresolved = state.unresolved_conflicts
    if unresolved:
        lines.append(
            "unresolved conflicts: "
            + ", ".join(c.check for c in unresolved)
            + "  <- these block emit_csv"
        )

    pending = [a for a in state.anomalies if a.decision == "pending"]
    lines.append(
        f"anomalies: {len(state.anomalies)} region(s), {len(pending)} still pending a decision"
    )
    lines.append(f"censored samples: {len(state.censored_indices)}")
    lines.append(
        f"smoothing window: {state.window_length if state.window_length is not None else 'not chosen'}"
    )
    lines.append(
        f"baseline: {'applied' if state.baseline_applied else 'none'}, "
        f"{len(state.anchor_candidates)} candidate(s), "
        f"selected by {state.anchor_selected_by}"
    )

    if state.fit is not None:
        lines.append(
            f"fit: slope {_number(state.fit.slope)}, intercept {_number(state.fit.intercept)}, "
            f"R^2 {_number(state.fit.r_squared, 4)}, n={state.fit.n_points}"
        )
    if state.parameters is not None:
        p = state.parameters
        lines.append(f"A2*Mw: {_number(p.a2_mw)} cm^3/g (calibration-free)")
        lines.append(f"k_D: {_number(p.k_d)}")
        if p.calibrated:
            lines.append(f"Mw: {_number(p.mw)} g/mol, A2: {_number(p.a2)}")
        else:
            lines.append("Mw, A2, A3: withheld -- alpha is not calibrated on this run")

    for reason in state.blocked:
        lines.append(f"blocked: {reason}")
    for warning in state.warnings[-5:]:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


def render_anomalies(records: Sequence[Any]) -> str:
    rows = [
        f"({r.start_index}, {r.end_index}) {r.n_points} sample(s) "
        f"type={r.anomaly_type.value if r.anomaly_type else 'unclassified'} "
        f"by={r.classified_by or '-'} decision={r.decision} "
        f"detected_by={','.join(r.detected_by)}"
        for r in records[:MAX_ROWS]
    ]
    return _truncated(rows, len(records))


def render_shortlist(candidates: Sequence[Any], noise_floor: float | None) -> str:
    if not candidates:
        return (
            "(no safe band computed -- call sweep_windows, which needs a loaded run)"
        )
    rows = [f"noise_floor: {_number(noise_floor)}"]
    for c in candidates[:MAX_ROWS]:
        features = ", ".join(f"{k}={_number(v)}" for k, v in sorted(c.features.items()))
        rows.append(f"  [{c.index}] window={c.value} {c.label}  {features}")
    rows.append(
        "Every candidate above already passed the numerical gates. pick_window "
        "accepts only these values."
    )
    return "\n".join(rows)


def render_conflicts(conflicts: Sequence[Any]) -> str:
    if not conflicts:
        return "(no conflicts: the data supports every declaration tested)"
    rows = []
    for c in conflicts:
        status = (
            f"resolved by {c.resolved_by}: {c.resolution_reason}"
            if c.resolved
            else "UNRESOLVED -- blocks emit_csv"
        )
        rows.append(
            f"{c.check}: declared {c.declared}, observed {c.observed}\n"
            f"    {c.detail}\n    {status}"
        )
    return "\n".join(rows)


def render_anchors(candidates: Sequence[Any]) -> str:
    if not candidates:
        return "(no anchor candidates were proposed)"
    rows = [
        f"[{c.id}] samples {c.index_range[0]}-{c.index_range[1]} "
        f"({c.n_points} pts, {c.duration_fraction:.1%} of run, at {c.position_fraction:.1%}) "
        f"mean={_number(c.mean_level)} slope={_number(c.local_slope)} var={_number(c.local_variance)}"
        for c in candidates[:MAX_ROWS]
    ]
    rows.append(
        "Anchors are selected by id from this list. Free-form index ranges are refused."
    )
    return _truncated(rows, len(candidates))


def render_intake(surface: ToolSurface) -> str:
    outstanding = surface.declarations.unanswered()
    if not outstanding:
        return "(every required declaration is recorded)"
    rows = []
    for q in outstanding[:MAX_ROWS]:
        line = f"{q.key}: {q.text}"
        if q.options:
            line += "\n    options: " + ", ".join(
                f"{o.value!r} ({o.source})" for o in q.options
            )
        if q.suggestion is not None:
            line += f"\n    suggested: {q.suggestion.value!r} from {q.suggestion.source} (not applied)"
        if q.disputed:
            line += "\n    DISPUTED: sources disagree; no default is offered, a person must settle it"
        if q.blocks:
            line += f"\n    blocks: {', '.join(q.blocks)}"
        rows.append(line)
    return _truncated(rows, len(outstanding))


def render_run_summaries(summaries: Sequence[dict[str, Any]]) -> str:
    if not summaries:
        return "(no past runs matched)"
    rows = []
    for s in summaries[:MAX_ROWS]:
        run_id = str(s.get("run_id") or "?")
        sample = str(s.get("sample_name") or "")
        # Most historical sidecars carry no sample_name, so the registry falls
        # back to the run id. Printing it twice wastes a line and reads as two
        # facts when it is one.
        head = run_id if sample in ("", run_id) else f"{run_id}  ({sample})"
        rows.append(
            f"{head}  [{s.get('experiment_type')}]  "
            f"A2*Mw={_number(s.get('a2_mw'))} k_D={_number(s.get('k_d'))} "
            f"Mw={_number(s.get('mw'))} window={s.get('window_length')} "
            f"anomalies={s.get('anomalies_count')} at {str(s.get('written_at'))[:10]}"
        )
    return _truncated(rows, len(summaries))


# --- handlers ----------------------------------------------------------------
#
# Each takes the context and the model's arguments, and returns text. A handler
# never catches Rejected: the orchestrator turns it into a failed tool result so
# the refusal, which is written as a recovery instruction, reaches the model
# intact.


def _h_get_state(ctx: ToolContext, args: dict[str, Any]) -> str:
    return render_state(ctx.surface)


def _h_list_anomalies(ctx: ToolContext, args: dict[str, Any]) -> str:
    return render_anomalies(ctx.surface.list_anomalies(type=args.get("type")))


def _h_get_shortlist(ctx: ToolContext, args: dict[str, Any]) -> str:
    return render_shortlist(ctx.surface.get_shortlist(), ctx.surface.state.noise_floor)


def _h_list_conflicts(ctx: ToolContext, args: dict[str, Any]) -> str:
    return render_conflicts(ctx.surface.list_conflicts())


def _h_list_anchor_candidates(ctx: ToolContext, args: dict[str, Any]) -> str:
    return render_anchors(ctx.surface.list_anchor_candidates())


def _h_get_intake_form(ctx: ToolContext, args: dict[str, Any]) -> str:
    return render_intake(ctx.surface)


def _h_list_blocked(ctx: ToolContext, args: dict[str, Any]) -> str:
    reasons = ctx.surface.list_blocked()
    paths = ctx.surface.list_paths()
    lines = [f"blocked: {r}" for r in reasons]
    for p in paths:
        lines.append(
            f"path {p.question}: took '{p.decision}' because {p.because}"
            + (f" (blocks {', '.join(p.blocked)})" if p.blocked else "")
        )
    return "\n".join(lines) or "(nothing is blocked)"


def _h_get_history(ctx: ToolContext, args: dict[str, Any]) -> str:
    history = ctx.surface.state.execution_history
    if not history:
        return "(nothing has run yet on this run)"
    return "\n".join(f"{i + 1}. {entry}" for i, entry in enumerate(history[-40:]))


def _h_search_past_runs(ctx: ToolContext, args: dict[str, Any]) -> str:
    matches = ctx.surface.search_past_runs(
        query=args.get("query"),
        sample_name=args.get("sample_name"),
        runs_dir=ctx.runs_dir,
    )
    experiment = args.get("experiment_type")
    if experiment:
        matches = [
            m for m in matches if str(m.get("experiment_type", "")).lower() == experiment.lower()
        ]
    return render_run_summaries(matches)


def _h_get_past_run(ctx: ToolContext, args: dict[str, Any]) -> str:
    run_id = str(args.get("run_id", "")).strip()
    matches = [
        m
        for m in ctx.surface.search_past_runs(runs_dir=ctx.runs_dir)
        if m.get("run_id") == run_id
    ]
    if not matches:
        return (
            f"no indexed run has id {run_id!r}. Use search_past_runs to list what "
            "is available."
        )
    record = matches[0]
    lines = [f"{k}: {v}" for k, v in record.items() if k not in {"warnings", "declarations"}]
    declarations = record.get("declarations") or {}
    if declarations:
        lines.append("declarations:")
        lines.extend(f"    {k} = {v!r}" for k, v in sorted(declarations.items()))
    warnings = record.get("warnings") or []
    if warnings:
        lines.append("warnings:")
        lines.extend(f"    {w}" for w in warnings[:10])
    return "\n".join(lines)


def _h_compare_to_past_runs(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.surface.state.parameters is None:
        return (
            "the current run has no derived parameters yet, so there is nothing to "
            "compare. Run extract_parameters first."
        )
    report = ctx.surface.compare_current_run(runs_dir=ctx.runs_dir)
    return "\n".join(f"{k}: {_number(v) if isinstance(v, float) else v}" for k, v in report.items())


def _h_search_figures(ctx: ToolContext, args: dict[str, Any]) -> str:
    try:
        matches = ctx.surface.search_historical_figures(
            query=args.get("query"),
            method=args.get("method"),
            sample_type=args.get("sample_type"),
            goal=args.get("goal"),
            ledger_path=ctx.ledger_path,
        )
    except Exception as error:  # a missing or malformed ledger is not fatal
        return f"(figure ledger unavailable: {error})"
    if not matches:
        return "(no labelled figures matched; the ledger may be empty)"
    rows = [
        f"{m.get('filename')} | {m.get('sampleName') or 'unnamed'} | "
        f"{', '.join(m.get('methods') or []) or 'no methods'} | "
        f"{m.get('projectType')} | {m.get('date') or 'undated'}"
        + (f" | {m.get('notes')}" if m.get("notes") else "")
        for m in matches[:MAX_ROWS]
    ]
    return _truncated(rows, len(matches))


def _h_read_notes(ctx: ToolContext, args: dict[str, Any]) -> str:
    found = ctx.notes.search(
        query=args.get("query"),
        subject=args.get("subject"),
        run_id=args.get("run_id"),
        tag=args.get("tag"),
    )
    if not found:
        return f"(no notes matched; the notebook holds {len(ctx.notes)} note(s))"
    return "\n".join(n.render() for n in found)


def _h_write_note(ctx: ToolContext, args: dict[str, Any]) -> str:
    note = ctx.notes.add(
        subject=str(args.get("subject", "")),
        body=str(args.get("body", "")),
        author=str(args.get("author") or ctx.author),
        run_id=args.get("run_id") or (ctx.surface.state.run_id or None),
        tags=list(args.get("tags") or []),
    )
    return f"note filed under {note.subject!r} by {note.author}; notebook now holds {len(ctx.notes)}"


def _h_load_run(ctx: ToolContext, args: dict[str, Any]) -> str:
    state = ctx.surface.load_run(path=args.get("path"))
    ctx.surface.state.execution_history.append(
        f"loaded {state.source_file}: {state.raw_signal.size if state.raw_signal is not None else 0} samples"
    )
    return render_state(ctx.surface)


def _h_sweep_windows(ctx: ToolContext, args: dict[str, Any]) -> str:
    band = ctx.surface.sweep_windows()
    ctx.surface.state.execution_history.append(
        f"swept smoothing windows: {len(band)} in the safe band"
    )
    return render_shortlist(ctx.surface.get_shortlist(), ctx.surface.state.noise_floor)


def _h_check_declarations(ctx: ToolContext, args: dict[str, Any]) -> str:
    conflicts = ctx.surface.check_declarations()
    ctx.surface.state.execution_history.append(
        f"checked declarations against the trace: {len(conflicts)} conflict(s)"
    )
    return render_conflicts(ctx.surface.list_conflicts())


def _h_fit_drift(ctx: ToolContext, args: dict[str, Any]) -> str:
    ids = [int(i) for i in args.get("anchor_ids") or []]
    drift = ctx.surface.fit_drift(ids, kind=str(args.get("kind") or "poly"))
    return (
        f"proposed {drift.kind} drift over anchors {ids}: "
        f"max_slope_ratio={_number(drift.max_slope_ratio)} "
        f"fold_change_impact={_number(drift.fold_change_impact)} "
        f"reversible={drift.reversible}\n"
        "Nothing has been subtracted. apply_baseline_correction commits it, and "
        "needs select_anchors first."
    )


def _h_extract_parameters(ctx: ToolContext, args: dict[str, Any]) -> str:
    """Part 2: the virial fit over an emitted CSV."""
    from polyrmc.pipeline import run_part2
    from polyrmc.tier1.judge import StaticJudge

    csv = args.get("processed_csv") or (
        str(ctx.output_csv) if ctx.output_csv else None
    )
    if not csv:
        raise Rejected(
            "REJECTED. extract_parameters needs a processed CSV. Emit one with "
            "emit_csv first, or pass processed_csv explicitly."
        )
    if not Path(csv).exists():
        raise Rejected(f"REJECTED. {csv} does not exist.")

    order = int(args.get("fit_order") or 1)
    if order not in (1, 2):
        raise Rejected(f"REJECTED. fit_order must be 1 or 2; got {order}.")

    part2 = run_part2(csv, ctx.surface.config, judge=StaticJudge(), fit_order=order)
    # Part 2 builds its own state from the CSV. Only the results are carried
    # back, never the arrays: the current state's trace is Part 1's, and
    # overwriting it would leave the session describing a run it no longer holds.
    ctx.surface.state.fit = part2.fit
    ctx.surface.state.parameters = part2.parameters
    ctx.surface.state.loops["fit_range"] = part2.loops.get("fit_range")
    ctx.surface.state.warnings.extend(part2.warnings)
    ctx.surface.state.blocked.extend(part2.blocked)
    ctx.surface.state.execution_history.append(f"extracted parameters from {csv}")
    return render_state(ctx.surface)


def _h_verify_output(ctx: ToolContext, args: dict[str, Any]) -> str:
    csv = args.get("processed_csv") or (str(ctx.output_csv) if ctx.output_csv else None)
    if not csv:
        raise Rejected("REJECTED. verify_output needs a processed CSV path.")
    checks = ctx.surface.verify_output(csv)
    rows = [
        f"{'FAIL' if c.failed else 'ok  '} {c.number}. {c.name}: {c.detail}"
        for c in checks
    ]
    failures = sum(1 for c in checks if c.failed)
    rows.append(f"{len(checks) - failures} of {len(checks)} passed or skipped, {failures} failed")
    if failures:
        rows.append(
            "A failed check names an invariant that broke. It is not a warning to "
            "read past -- report it."
        )
    return "\n".join(rows)


def _h_set_declaration(ctx: ToolContext, args: dict[str, Any]) -> str:
    result = ctx.surface.set_declaration(
        key=str(args["key"]),
        value=args["value"],
        author=str(args.get("author") or ctx.author),
    )
    return result.message


def _h_resolve_conflict(ctx: ToolContext, args: dict[str, Any]) -> str:
    return ctx.surface.resolve_conflict(
        check=str(args["check"]),
        reason=str(args.get("reason", "")),
        author=str(args.get("author") or ctx.author),
    ).message


def _h_pick_window(ctx: ToolContext, args: dict[str, Any]) -> str:
    result = ctx.surface.pick_window(int(args["window_length"]))
    ctx.surface.state.execution_history.append(result.message)
    return result.message


def _h_set_anomaly_decision(ctx: ToolContext, args: dict[str, Any]) -> str:
    result = ctx.surface.set_decision(
        (int(args["start_index"]), int(args["end_index"])), str(args["decision"])
    )
    ctx.surface.state.execution_history.append(result.message)
    return result.message


def _h_select_anchors(ctx: ToolContext, args: dict[str, Any]) -> str:
    return ctx.surface.select_anchors(
        [int(i) for i in args.get("anchor_ids") or []],
        selected_by=str(args.get("selected_by") or ctx.author),
    ).message


def _h_apply_baseline(ctx: ToolContext, args: dict[str, Any]) -> str:
    result = ctx.surface.apply_baseline_correction(
        kind=str(args.get("kind") or "poly"), override=bool(args.get("override") or False)
    )
    ctx.surface.state.execution_history.append(result.message)
    return result.message


def _h_revert_baseline(ctx: ToolContext, args: dict[str, Any]) -> str:
    result = ctx.surface.revert_baseline_correction()
    ctx.surface.state.execution_history.append(result.message)
    return result.message


def _h_set_config(ctx: ToolContext, args: dict[str, Any]) -> str:
    return ctx.surface.set_config(str(args["key"]), args["value"]).message


def _h_emit_csv(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = args.get("path") or (str(ctx.output_csv) if ctx.output_csv else None)
    result = ctx.surface.emit_csv(path)
    ctx.output_csv = Path(result.changed["path"])
    ctx.surface.state.execution_history.append(result.message)
    return result.message + "\nRun verify_output next; an emit is not finished until it is checked."


# --- the registry ------------------------------------------------------------

_TOOLS: list[Tool] = [
    # read ---------------------------------------------------------------
    Tool(
        "get_state",
        "read",
        "The current run at a glance: what is loaded, what has been decided, what "
        "is blocking an emit. Call this first when you are unsure where the run "
        "stands.",
        NO_ARGS,
        _h_get_state,
    ),
    Tool(
        "list_anomalies",
        "read",
        "Every flagged region of the trace, with the detector that fired, the "
        "assigned type, and whether a decision is still pending. A pending region "
        "blocks emit_csv.",
        _object(
            {
                "type": {
                    "type": "string",
                    "description": "Optional filter.",
                    "enum": [
                        "transient_spike",
                        "baseline_drift",
                        "saturation_artifact",
                        "structural_noise",
                        "splice_gap",
                    ],
                }
            }
        ),
        _h_list_anomalies,
    ),
    Tool(
        "get_shortlist",
        "read",
        "The safe band of smoothing windows with the computed features of each. "
        "Every candidate has already passed the numerical gates; pick_window "
        "accepts nothing outside this list.",
        NO_ARGS,
        _h_get_shortlist,
    ),
    Tool(
        "list_conflicts",
        "read",
        "Declarations the trace does not appear to support. A conflict is never "
        "resolved by code and never silently; each unresolved one blocks emit_csv.",
        NO_ARGS,
        _h_list_conflicts,
    ),
    Tool(
        "list_anchor_candidates",
        "read",
        "Baseline anchor candidates enumerated by code, with level, slope and "
        "variance for each. Anchors are chosen by id from this set.",
        NO_ARGS,
        _h_list_anchor_candidates,
    ),
    Tool(
        "get_intake_form",
        "read",
        "Required declarations still outstanding, with their options and what each "
        "one blocks. These are answered by a person, never inferred from the trace.",
        NO_ARGS,
        _h_get_intake_form,
    ),
    Tool(
        "list_blocked",
        "read",
        "Outputs that cannot be computed, the reason for each, and which branch the "
        "run took at every unresolved design question.",
        NO_ARGS,
        _h_list_blocked,
    ),
    Tool(
        "get_history",
        "read",
        "What has actually been run on this run, in order.",
        NO_ARGS,
        _h_get_history,
    ),
    # read: memory over past work ----------------------------------------
    Tool(
        "search_past_runs",
        "read",
        "Search previously processed experiments by sample, type, or free text. "
        "Use this before analysing a new run of a familiar sample: what the same "
        "material did last time is the cheapest sanity check available.",
        _object(
            {
                "query": {"type": "string", "description": "Free text over id, sample, source file."},
                "sample_name": {"type": "string", "description": "Sample or polymer name."},
                "experiment_type": {"type": "string", "description": "e.g. acd_dilution."},
            }
        ),
        _h_search_past_runs,
    ),
    Tool(
        "get_past_run",
        "read",
        "Everything indexed about one past run: declarations, parameters, window, "
        "warnings. Use it after search_past_runs narrows to a candidate.",
        _object({"run_id": {"type": "string", "description": "Run id from search_past_runs."}}, ["run_id"]),
        _h_get_past_run,
    ),
    Tool(
        "compare_to_past_runs",
        "read",
        "Compare this run's derived parameters against the historical set for the "
        "same sample. Needs parameters, so run extract_parameters first.",
        NO_ARGS,
        _h_compare_to_past_runs,
    ),
    Tool(
        "search_figures",
        "read",
        "Search labelled figures from past publications and reports by method, "
        "sample type, or research goal.",
        _object(
            {
                "query": {"type": "string"},
                "method": {"type": "string", "description": "e.g. SLS, DLS."},
                "sample_type": {"type": "string"},
                "goal": {"type": "string"},
            }
        ),
        _h_search_figures,
    ),
    Tool(
        "read_notes",
        "read",
        "Read notes filed in earlier sessions: conclusions, gotchas, and context "
        "that no sidecar records. Check these when a sample or instrument quirk "
        "comes up.",
        _object(
            {
                "query": {"type": "string"},
                "subject": {"type": "string"},
                "run_id": {"type": "string"},
                "tag": {"type": "string"},
            }
        ),
        _h_read_notes,
    ),
    Tool(
        "write_note",
        "read",
        "File a note for future sessions. Record a conclusion, a gotcha, or a "
        "decision and its reason -- something a later reader could not recover "
        "from the run's own provenance. Do not file numbers a sidecar already "
        "holds.",
        _object(
            {
                "subject": {"type": "string", "description": "Sample, run, or instrument the note is about."},
                "body": {"type": "string", "description": "The conclusion, in full sentences."},
                "author": {"type": "string", "description": "Who concluded this. Defaults to the session's researcher."},
                "run_id": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            ["subject", "body"],
        ),
        _h_write_note,
    ),
    # compute ------------------------------------------------------------
    Tool(
        "load_run",
        "compute",
        "Ingest and clean the raw instrument file: time reconstruction, channel "
        "selection, detectors, splice, censoring, and anchor proposals. Stops "
        "before the smoothing window, which is a judgment.",
        _object({"path": {"type": "string", "description": "Raw file. Defaults to the configured source."}}),
        _h_load_run,
    ),
    Tool(
        "sweep_windows",
        "compute",
        "Compute the safe band of smoothing windows. Does not choose one.",
        NO_ARGS,
        _h_sweep_windows,
    ),
    Tool(
        "check_declarations",
        "compute",
        "Test every declaration against the loaded trace and record what conflicts. "
        "Run this after loading and before emitting.",
        NO_ARGS,
        _h_check_declarations,
    ),
    Tool(
        "fit_drift",
        "compute",
        "Fit a drift model over the named anchor candidates and report what "
        "applying it would do. Subtracts nothing.",
        _object(
            {
                "anchor_ids": {"type": "array", "items": {"type": "integer"}},
                "kind": {"type": "string", "enum": ["poly", "piecewise_linear"]},
            },
            ["anchor_ids"],
        ),
        _h_fit_drift,
    ),
    Tool(
        "extract_parameters",
        "compute",
        "Part 2: fit the virial expansion over an emitted CSV to get A2*Mw, k_D, "
        "and -- only on a calibrated run -- Mw and A2.",
        _object(
            {
                "processed_csv": {"type": "string", "description": "Defaults to the CSV this session emitted."},
                # No enum: Gemini's schema validator accepts only string enums,
                # and an integer one fails at bind time rather than at call
                # time -- taking every other tool down with it. The handler
                # rejects anything but 1 or 2 and says so.
                "fit_order": {"type": "integer", "description": "1 (linear) or 2 (quadratic)."},
            }
        ),
        _h_extract_parameters,
    ),
    Tool(
        "verify_output",
        "compute",
        "Run the pipeline's own checks against an emitted CSV.",
        _object({"processed_csv": {"type": "string"}}),
        _h_verify_output,
    ),
    # commit -------------------------------------------------------------
    Tool(
        "set_declaration",
        "commit",
        "Record what a person says this run is. Only call this to write down what "
        "the researcher has actually told you. Never infer a declaration from the "
        "shape of the trace.",
        _object(
            {
                "key": {"type": "string"},
                "value": {"type": "string", "description": "The declared value, as text; it is coerced and validated."},
                "author": {"type": "string", "description": "Who is declaring it."},
            },
            ["key", "value"],
        ),
        _h_set_declaration,
    ),
    Tool(
        "resolve_conflict",
        "commit",
        "Mark a declaration conflict as understood, with a recorded reason from the "
        "researcher. The reason is mandatory: an unexplained dismissal is "
        "indistinguishable from not having looked.",
        _object(
            {
                "check": {"type": "string"},
                "reason": {"type": "string", "description": "The physical explanation."},
                "author": {"type": "string"},
            },
            ["check", "reason"],
        ),
        _h_resolve_conflict,
    ),
    Tool(
        "pick_window",
        "commit",
        "Choose the smoothing window. Only a value from the safe band is accepted.",
        _object({"window_length": {"type": "integer"}}, ["window_length"]),
        _h_pick_window,
    ),
    Tool(
        "set_anomaly_decision",
        "commit",
        "Decide what happens to a flagged region: excise it, keep it, or correct it "
        "(drift only). Regions the rules could not name should be kept unless the "
        "researcher says otherwise -- excising an unnamed region deletes data on a guess.",
        _object(
            {
                "start_index": {"type": "integer"},
                "end_index": {"type": "integer"},
                "decision": {"type": "string", "enum": ["excise", "keep", "correct"]},
            },
            ["start_index", "end_index", "decision"],
        ),
        _h_set_anomaly_decision,
    ),
    Tool(
        "select_anchors",
        "commit",
        "Choose baseline anchors by candidate id. Validated against slope and "
        "coverage ceilings.",
        _object(
            {
                "anchor_ids": {"type": "array", "items": {"type": "integer"}},
                "selected_by": {"type": "string"},
            },
            ["anchor_ids"],
        ),
        _h_select_anchors,
    ),
    Tool(
        "apply_baseline_correction",
        "commit",
        "Subtract the fitted drift. Anchor selection decides what counts as signal "
        "and nothing downstream catches getting it wrong, so this needs the "
        "researcher's explicit agreement. No correction is a valid answer.",
        _object(
            {
                "kind": {"type": "string", "enum": ["poly", "piecewise_linear"]},
                "override": {
                    "type": "boolean",
                    "description": "Proceed past a failed ceiling. Recorded on the run. "
                    "Only with explicit researcher approval.",
                },
            }
        ),
        _h_apply_baseline,
    ),
    Tool(
        "revert_baseline_correction",
        "commit",
        "Undo the applied correction, recovering the uncorrected values exactly.",
        NO_ARGS,
        _h_revert_baseline,
    ),
    Tool(
        "set_config",
        "commit",
        "Change one configuration value, validated against its schema. Use a dotted "
        "path, e.g. smoothing.polyorder.",
        _object({"key": {"type": "string"}, "value": {"type": "string"}}, ["key", "value"]),
        _h_set_config,
    ),
    Tool(
        "emit_csv",
        "commit",
        "Write the boundary CSV. Refuses a run carrying an unresolved conflict, no "
        "chosen window, or an undecided anomaly region.",
        _object({"path": {"type": "string"}}),
        _h_emit_csv,
    ),
]


def build_registry(extra: Sequence[Tool] = ()) -> dict[str, Tool]:
    """The tool table, keyed by name."""
    registry = {tool.name: tool for tool in _TOOLS}
    for tool in extra:
        registry[tool.name] = tool
    return registry


def schemas_for(registry: dict[str, Tool]) -> list[dict[str, Any]]:
    """Schemas in the plain form the provider adapter normalises."""
    return [tool.schema() for tool in registry.values()]


def describe_registry(registry: dict[str, Tool]) -> str:
    """A one-line-per-tool listing, for the `/tools` command."""
    rows = []
    for tier in ("read", "compute", "commit"):
        members = [t for t in registry.values() if t.tier == tier]
        if not members:
            continue
        rows.append(f"{tier}:")
        for tool in members:
            head = tool.description.split(".")[0]
            rows.append(f"  {tool.name:<28} {head}")
    return "\n".join(rows)


def default_ledger_path() -> Path | None:
    """Where labelled figures are looked up, if anywhere.

    Read from the environment rather than hardcoded. Two call sites used to
    carry two different absolute paths into one person's Downloads folder, so
    the chat session and the tool surface searched different ledgers and
    neither said so.
    """
    import os

    raw = os.environ.get(DEFAULT_LEDGER_ENV)
    return Path(raw) if raw else None


__all__ = [
    "DEFAULT_LEDGER_ENV",
    "MAX_ROWS",
    "NO_ARGS",
    "Tier",
    "Tool",
    "ToolContext",
    "build_registry",
    "default_ledger_path",
    "describe_registry",
    "render_state",
    "schemas_for",
]
