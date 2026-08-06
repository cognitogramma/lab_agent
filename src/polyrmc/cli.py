"""Command-line entry point.

    python -m polyrmc.cli intake --raw data/raw/<file> --declarations run.decl.json
    python -m polyrmc.cli part1 --config run.json
    python -m polyrmc.cli part2 --config run.json --csv data/runs/<run>.csv
    python -m polyrmc.cli run   --config run.json

``intake`` is the section 6.0 interview: it asks what the run is, records each
answer with an author and a timestamp, then checks those answers against the
trace. It makes no model call and needs no API key.

The judge defaults to the deterministic :class:`StaticJudge` unless
``--judge model`` is passed and an API key is configured, so the pipeline is
fully runnable offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from polyrmc.config import RunConfig
from polyrmc.intake import (
    DeclarationSet,
    MissingDeclarations,
    build_intake_form,
    render_question,
    to_config,
)
from polyrmc.pipeline import new_run_id, run_part1, run_part2
from polyrmc.provenance import summarize_loop
from polyrmc.state import Declaration
from polyrmc.tier0.argen_io import DEFAULT_CHANNEL, load_argen_file
from polyrmc.tier0.consistency import check_declarations
from polyrmc.tier1.judge import ModelJudge, StaticJudge
from polyrmc.verify import summarize, verify_output
from polyrmc.tier1.loop import Judge


def read_json(path: str | Path) -> dict:
    """Read a JSON file written by anything, including a Windows editor.

    Decoded as utf-8-sig rather than utf-8: PowerShell's ``Out-File`` and
    Notepad both prepend a byte-order mark, which the plain utf-8 decoder
    reports as a syntax error on line 1 of an otherwise valid file.
    """
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_config(path: str | Path) -> RunConfig:
    """Load a run configuration from JSON."""
    return RunConfig.model_validate(read_json(path))


def make_judge(kind: str, decision_name: str, config: RunConfig) -> Judge:
    """Build the judge for a loop, refusing to pretend a model call happened."""
    if kind == "static":
        return StaticJudge()
    # ModelJudge speaks to Gemini through langchain_google_genai, which reads
    # either of these. Naming the wrong provider's variable here made
    # `--judge model` unusable no matter how the key was set.
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit(
            "--judge model requires GOOGLE_API_KEY (or GEMINI_API_KEY). Set it in "
            ".env, or use --judge static to run the deterministic conservative rule "
            "instead."
        )
    return ModelJudge(decision_name, config=config.loop)


def _ask(question, author: str) -> object | None:
    """Put one question to the person at the terminal.

    An empty answer accepts the suggestion where one is offered and skips the
    question otherwise. A disputed question never has a suggestion, so it cannot
    be answered by pressing return -- which is the point of marking it.
    """
    print()
    print(render_question(question))
    while True:
        raw = input("  > ").strip()
        if not raw:
            if question.suggestion is not None:
                return question.suggestion.value
            print("    skipped -- what depends on it stays blocked")
            return None
        try:
            return question.validate_answer(raw)
        except ValueError as error:
            print(f"    {error}")


def run_intake(args) -> int:
    """Section 6.0 interview, then the section 6.0.1 checks against the trace."""
    argen = load_argen_file(args.raw, channel=args.channel or DEFAULT_CHANNEL)
    channel = argen.scattering_columns[0]
    signal = argen.data[channel].to_numpy(dtype=float)

    declarations = DeclarationSet(build_intake_form(metadata=argen.metadata))

    # Answers supplied on the command line are validated exactly as typed ones
    # are, so a batch run cannot slip past a check an interactive one would hit.
    supplied: dict[str, object] = {}
    if args.answers:
        supplied = read_json(args.answers)
    if args.declarations and Path(args.declarations).exists() and not args.answers:
        stored = read_json(args.declarations)
        supplied = {key: entry["value"] for key, entry in stored.items()}

    interactive = sys.stdin.isatty() and not args.no_prompt

    print(f"{Path(args.raw).name}: {argen.n_points} samples, channel {channel}")
    if argen.metadata:
        print("file metadata:")
        for key, value in argen.metadata.items():
            print(f"  {key}: {value}")

    # The form is rebuilt after every answer: the instrument fixes which toluene
    # candidates apply, and the experiment type decides whether the dilution
    # geometry is asked about at all.
    asked: set[str] = set()
    while True:
        pending = [q for q in declarations.unanswered() if q.key not in asked]
        if not pending:
            break
        question = pending[0]
        asked.add(question.key)

        if question.key in supplied:
            declarations.record(
                question.key, supplied[question.key], author=args.author
            )
        elif interactive:
            answer = _ask(question, args.author)
            if answer is not None:
                declarations.record(question.key, answer, author=args.author)
        declarations.refresh_form(metadata=argen.metadata)

    if not interactive and not supplied:
        print("\nintake questions (no answers supplied, nothing recorded):")
        for question in declarations.unanswered():
            print(render_question(question))

    declared = {key: entry.value for key, entry in declarations.declarations.items()}

    stir_column = next(
        (c for c in argen.data.columns if "stir" in c.lower()), None
    )
    stir_speed = (
        argen.data[stir_column].to_numpy(dtype=float) if stir_column else None
    )

    report = check_declarations(
        argen.time_s,
        signal,
        declared,
        metadata=argen.metadata,
        stir_speed=stir_speed,
    )

    print("\ndeclarations:")
    for key, entry in sorted(declarations.declarations.items()):
        origin = f" (derived from {entry.derived_from})" if entry.derived_from else ""
        print(f"  {key} = {entry.value!r}  by {entry.author}{origin}")

    print("\nconsistency checks:")
    for line in report.passed:
        print(f"  ok       {line}")
    for line in report.skipped:
        print(f"  skipped  {line}")
    for conflict in report.conflicts:
        print(f"  CONFLICT {conflict.check}")
        print(f"           declared: {conflict.declared}")
        print(f"           observed: {conflict.observed}")
        print(f"           {conflict.detail}")

    blocked = declarations.blocked_reasons()
    if blocked:
        print("\nblocked:")
        for reason in blocked:
            print(f"  {reason}")

    if args.declarations:
        path = Path(args.declarations)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    key: json.loads(entry.model_dump_json())
                    for key, entry in declarations.declarations.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\ndeclarations written to {path}")

    if report.unresolved:
        print(
            f"\n{len(report.unresolved)} unresolved conflict(s). These block emit_csv "
            "until a person resolves each one with a recorded reason. Code does not "
            "resolve them, because the data can disagree with a correct declaration "
            "for real physical reasons."
        )
        return 2
    return 0


def run_process(args) -> int:
    """Declarations in, verified CSV out. One command, no hand-written config.

    This is the path a real run takes. The configuration is *derived* from what
    a person declared rather than maintained beside it, so a run cannot be
    declared one thing and processed as another -- which was possible while the
    two lived in separate files.
    """
    stored = read_json(args.declarations)
    declarations = {
        key: Declaration.model_validate(entry) for key, entry in stored.items()
    }

    run_id = args.run_id or new_run_id(Path(args.raw).stem[:16])
    try:
        config = to_config(
            declarations,
            source_file=args.raw,
            run_id=run_id,
            channel=args.channel or DEFAULT_CHANNEL,
        )
    except MissingDeclarations as error:
        print(f"cannot process this run: {error}")
        print(f"\nRun intake first:\n  polyrmc intake --raw {args.raw} "
              f"--declarations {args.declarations} --author <you>")
        return 2

    # The declaration-versus-data check runs before anything is computed. A run
    # processed under a declaration the trace contradicts is the one error that
    # leaves every number looking normal.
    argen = load_argen_file(args.raw, channel=config.channel)
    channel = argen.scattering_columns[0]
    stir_column = next((c for c in argen.data.columns if "stir" in c.lower()), None)
    report = check_declarations(
        argen.time_s,
        argen.data[channel].to_numpy(dtype=float),
        {key: d.value for key, d in declarations.items()},
        metadata=argen.metadata,
        stir_speed=(
            argen.data[stir_column].to_numpy(dtype=float) if stir_column else None
        ),
    )
    for line in report.passed:
        print(f"  ok       {line}")
    for line in report.skipped:
        print(f"  skipped  {line}")
    for conflict in report.conflicts:
        print(f"  CONFLICT {conflict.check}: declared {conflict.declared}, "
              f"observed {conflict.observed}")
        print(f"           {conflict.detail}")
    if report.unresolved and not args.force:
        print(
            f"\nRefusing to process: {len(report.unresolved)} unresolved conflict(s). "
            "Resolve them with a recorded reason, or pass --force if you have a "
            "physical explanation and want it on the record."
        )
        return 2

    judge = make_judge(args.judge, "smoothing_window", config)
    state, csv_path = run_part1(config, judge=judge, output_csv=args.out)
    state.declarations = declarations
    state.declaration_conflicts = report.conflicts
    if report.unresolved and args.force:
        state.warnings.append(
            f"processed over {len(report.unresolved)} unresolved conflict(s) by "
            "explicit --force"
        )

    _report(state, f"processed -> {csv_path}")
    return _verify(state, csv_path, config)


def run_agent(args) -> int:
    """Drive the agent graph, answering its checkpoints at the terminal.

    The graph pauses wherever a judgment belongs to a person. Nothing here
    decides on your behalf: an unanswered checkpoint stops the run rather than
    proceeding on a default.
    """
    from polyrmc.agent.graph import GREETING, AgentGraph, Node
    from polyrmc.state import RunState
    from polyrmc.tools import Rejected, ToolSurface

    stored = read_json(args.declarations)
    declarations = {
        key: Declaration.model_validate(entry) for key, entry in stored.items()
    }
    run_id = args.run_id or new_run_id(Path(args.raw).stem[:16])
    try:
        config = to_config(
            declarations,
            source_file=args.raw,
            run_id=run_id,
            channel=args.channel or DEFAULT_CHANNEL,
        )
    except MissingDeclarations as error:
        print(f"cannot start: {error}")
        return 2

    state = RunState(run_id=run_id, source_file=str(args.raw))
    state.declarations = declarations
    tools = ToolSurface(state=state, config=config)
    for key, declaration in declarations.items():
        tools.declarations.declarations[key] = declaration
    tools.declarations.refresh_form()

    graph = AgentGraph(tools=tools, output_csv=args.out)
    print(GREETING)
    print(
        f"lab context: {graph.lab_path or 'NONE FOUND -- running without house conventions'}"
    )

    interactive = sys.stdin.isatty() and not args.no_prompt
    for _ in range(40):
        for step in graph.run():
            print(f"  {step.narration}")
        if graph.pending is None:
            break

        print()
        print(graph.pending)
        if not interactive:
            print("\nStopped at a checkpoint. Re-run without --no-prompt to answer it.")
            return 2

        if graph.node is Node.REVIEW_ANOMALIES:
            answer = input("  decide all as [keep/excise] or 'stop': ").strip().lower()
            if answer not in {"keep", "excise"}:
                return 2
            for record in graph.state.anomalies:
                if record.decision == "pending":
                    tools.set_decision(
                        (record.start_index, record.end_index), answer
                    )
        elif graph.node is Node.REVIEW_ANCHORS:
            answer = input("  anchor ids, comma separated, or 'none': ").strip()
            if answer and answer.lower() != "none":
                try:
                    ids = [int(part) for part in answer.split(",")]
                    tools.select_anchors(ids, selected_by="researcher")
                    tools.apply_baseline_correction()
                except (ValueError, Rejected) as error:
                    print(f"  {error}")
                    return 2
            else:
                graph.state.anchor_selected_by = "researcher"
        elif graph.node is Node.JUDGE_WINDOW:
            answer = input("  window: ").strip()
            try:
                tools.pick_window(int(answer))
            except (ValueError, Rejected) as error:
                print(f"  {error}")
                return 2
        else:
            print("\nThis one needs fixing outside the agent.")
            return 2
        graph.resume()

    print()
    print(graph.narrate())
    if graph.checks:
        print("\nverification")
        print(summarize(graph.checks))
    failures = [c for c in graph.checks if c.failed]
    return 3 if failures else (0 if graph.node is Node.DONE else 2)


def _report(state, label: str) -> None:
    print(f"\n{label}")
    for loop in state.loops.values():
        print(f"  {summarize_loop(loop)}")
    for warning in state.warnings:
        print(f"  warning: {warning}")
    for reason in state.blocked:
        print(f"  blocked: {reason}")


def _verify(state, csv_path, config) -> int:
    """Run the pipeline's own checks against what it just produced."""
    results = verify_output(
        csv_path,
        state,
        m0_sensitivity_ceiling=config.m0_sensitivity_ceiling,
        drift_slope_ceiling=config.baseline.drift_slope_ceiling,
    )
    print("\nverification")
    print(summarize(results))
    failures = [r for r in results if r.failed]
    if failures:
        print(f"\n{len(failures)} check(s) FAILED.")
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="polyrmc", description=__doc__)
    parser.add_argument(
        "stage", choices=["intake", "process", "agent", "part1", "part2", "run"]
    )
    parser.add_argument("--config", help="Run configuration JSON.")
    parser.add_argument("--csv", help="Processed CSV (part2 only).")
    # No default: part1 falls back to config.channel, and overriding that with a
    # CLI default would silently ignore a config that names another channel.
    parser.add_argument("--channel", help="Scattering channel.")
    parser.add_argument("--raw", help="Raw instrument file (intake only).")
    parser.add_argument(
        "--declarations",
        help="Where intake answers are read from and written to (JSON).",
    )
    parser.add_argument(
        "--answers", help="Pre-supplied intake answers as a flat JSON object."
    )
    parser.add_argument(
        "--author", default=os.environ.get("USER") or os.environ.get("USERNAME") or "",
        help="Who is making these declarations. Required: an unattributed "
        "declaration is not auditable.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Never prompt; print the outstanding questions and stop.",
    )
    parser.add_argument(
        "--judge",
        choices=["static", "model"],
        default="static",
        help="Tier-1 judge. 'static' takes the conservative option without a model.",
    )
    parser.add_argument("--out", help="Where to write the processed CSV.")
    parser.add_argument("--run-id", help="Identifier for this run.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process despite unresolved conflicts. Recorded on the run.",
    )
    parser.add_argument("--fit-order", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the section 13 checks against the emitted CSV and fail on any.",
    )
    args = parser.parse_args(argv)

    if args.stage == "intake":
        if not args.raw:
            parser.error("intake requires --raw")
        if not str(args.author).strip():
            parser.error(
                "intake requires --author. A declaration is a statement by a named "
                "person about their own experiment; an unattributed one cannot be "
                "questioned later by whoever finds it wrong."
            )
        return run_intake(args)

    if args.stage in {"process", "agent"}:
        if not (args.raw and args.declarations):
            parser.error(f"{args.stage} requires --raw and --declarations")
        return run_process(args) if args.stage == "process" else run_agent(args)

    if not args.config:
        parser.error(f"{args.stage} requires --config")
    config = load_config(args.config)

    if args.stage in {"part1", "run"}:
        judge = make_judge(args.judge, "smoothing_window", config)
        state, csv_path = run_part1(config, channel=args.channel, judge=judge)
        _report(state, f"part 1 complete -> {csv_path}")
        if args.verify and _verify(state, csv_path, config) != 0:
            return 3
        csv_for_part2 = csv_path
    else:
        if not args.csv:
            parser.error("part2 requires --csv")
        csv_for_part2 = Path(args.csv)

    if args.stage in {"part2", "run"}:
        judge = make_judge(args.judge, "fit_range", config)
        state = run_part2(csv_for_part2, config, judge=judge, fit_order=args.fit_order)
        _report(state, f"part 2 complete <- {csv_for_part2}")
        parameters = state.parameters
        if parameters:
            print(f"\n  A2*Mw = {parameters.a2_mw:.6g} cm^3/g (calibration-free)")
            print(f"  k_D   = {parameters.k_d:.6g}")
            if parameters.calibrated:
                print(f"  Mw    = {parameters.mw:.6g} g/mol")
                print(f"  A2    = {parameters.a2:.6g} cm^3 mol/g^2")
            else:
                print("  Mw, A2: not reported (alpha not calibrated)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
