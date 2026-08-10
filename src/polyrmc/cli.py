"""Command-line entry point.

    python -m polyrmc.cli agent
    python -m polyrmc.cli agent --raw data/raw/<file> --declarations run.decl.json
    python -m polyrmc.cli intake --raw data/raw/<file> --declarations run.decl.json
    python -m polyrmc.cli process --raw data/raw/<file> --declarations run.decl.json
    python -m polyrmc.cli part1 --config run.json
    python -m polyrmc.cli part2 --config run.json --csv data/runs/<run>.csv
    python -m polyrmc.cli run   --config run.json

``agent`` is the interactive session, and the one to reach for by default. The
model drives: it reads the run, searches what the lab did before, and calls the
tools it needs, stopping to ask you before anything commits. Started with no
arguments it is still useful -- searching past experiments does not need a file
in hand.

``intake`` is the section 6.0 interview: it asks what the run is, records each
answer with an author and a timestamp, then checks those answers against the
trace. ``process`` takes those declarations straight through to a verified CSV.
Neither makes a model call or needs an API key, and neither has changed: the
deterministic path is the same path whether a model chose to walk it or a
researcher typed the command.

The tier-1 judge defaults to :class:`StaticJudge` unless ``--judge model`` is
passed, so every non-agent stage is fully runnable offline.
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


def _session_config(args) -> tuple[RunConfig, dict[str, Declaration], str]:
    """Build the run configuration an agent session starts from.

    Two entry shapes. With ``--raw`` and ``--declarations``, the configuration
    is *derived* from what a person declared, exactly as ``process`` derives it,
    so a run cannot be declared one thing and analysed as another. Without them
    the session still starts -- a researcher asking about past work should not
    have to nominate a run first -- and the agent reports the run as unloaded
    until it is pointed at a file.
    """
    if args.raw and args.declarations:
        stored = read_json(args.declarations)
        declarations = {
            key: Declaration.model_validate(entry) for key, entry in stored.items()
        }
        run_id = args.run_id or new_run_id(Path(args.raw).stem[:16])
        config = to_config(
            declarations,
            source_file=args.raw,
            run_id=run_id,
            channel=args.channel or DEFAULT_CHANNEL,
        )
        return config, declarations, run_id

    run_id = args.run_id or "session"
    if args.config:
        return load_config(args.config), {}, run_id

    # No run nominated. The session still needs a configuration object to hang
    # the tool surface on, and it is not this command's place to invent one:
    # dn_dc alone is 0.185 for a protein and 0.150 for a polysaccharide, and a
    # default would be the exact failure the intake interview exists to prevent
    # -- a run processed under a physical model nobody chose. The reviewed
    # example config in the repo is used as scaffolding, and nothing is computed
    # against it until a real run is loaded.
    example = Path("examples/run_config.example.json")
    if example.exists():
        config = load_config(example)
        return config.model_copy(update={"run_id": run_id, "source_file": args.raw or "none"}), {}, run_id

    raise MissingDeclarations(
        "an agent session needs either a run (--raw with --declarations) or a "
        "configuration (--config). There is no default: dn_dc is 0.185 for a "
        "protein and 0.150 for a polysaccharide, and guessing it would process "
        "the run under a physical model nobody chose."
    )


def _ask_approval(tool, call):
    """Put one commit to the researcher.

    The arguments are shown in full. An approval prompt that hides what is about
    to change trains the reader to approve without looking, which is worse than
    not asking at all.
    """
    from polyrmc.agent.orchestrator import Approval

    print(f"\n  commit: {call.render()}")
    print(f"    {tool.description.splitlines()[0]}")
    try:
        answer = input("    allow? [y]es / [n]o / [a]lways / reason: ").strip()
    except (KeyboardInterrupt, EOFError):
        return Approval(False, "interrupted")

    lowered = answer.lower()
    if lowered in {"y", "yes"}:
        return Approval(True)
    if lowered in {"a", "always"}:
        return Approval(True, "always allowed this session", remember=True)
    if lowered in {"n", "no", ""}:
        return Approval(False, "declined")
    # Anything else is treated as a refusal with a stated reason, which is more
    # useful to the agent than a bare no.
    return Approval(False, answer)


REPL_HELP = """\
  /state            where the run stands
  /tools            every tool the agent can call
  /history          what the agent has run this session
  /runs [query]     search past experiments directly, no model call
  /notes [query]    read notes filed in earlier sessions
  /always <tool>    pre-approve a commit tool for this session
  /auto             approve every commit for the rest of the session
  /cost             requests and tokens spent on your key this session
  /help             this list
  /exit             end the session"""


def run_agent(args) -> int:
    """The agent session: the model drives, and commits go through you.

    Nothing here decides the route. The agent reads the state, calls what it
    needs, and stops to ask when a change belongs to a person.
    """
    from polyrmc.agent.budget import Budget, rates_from_env
    from polyrmc.agent.orchestrator import (
        GREETING,
        ApprovalPolicy,
        Orchestrator,
        allow_all,
        deny_all,
    )
    from polyrmc.agent.toolcall import ProviderUnavailable, build_provider
    from polyrmc.agent.toolspec import (
        ToolContext,
        build_registry,
        default_ledger_path,
        describe_registry,
        render_state,
    )
    from polyrmc.memory.notes import NoteBook
    from polyrmc.memory.registry import RunRegistry
    from polyrmc.state import RunState
    from polyrmc.tools import ToolSurface

    try:
        config, declarations, run_id = _session_config(args)
    except MissingDeclarations as error:
        print(f"cannot start: {error}")
        print(
            f"\nRun intake first:\n  polyrmc intake --raw {args.raw} "
            f"--declarations {args.declarations} --author <you>"
        )
        return 2

    state = RunState(run_id=run_id, source_file=str(args.raw or "none"))
    state.declarations = dict(declarations)
    surface = ToolSurface(state=state, config=config)
    for key, declaration in declarations.items():
        surface.declarations.declarations[key] = declaration
    surface.declarations.refresh_form()

    runs_dir = Path(args.runs_dir or "data/runs")
    notes = NoteBook(args.notes_dir)
    context = ToolContext(
        surface=surface,
        notes=notes,
        runs_dir=runs_dir,
        ledger_path=default_ledger_path(),
        author=str(args.author or "researcher"),
        output_csv=Path(args.out) if args.out else None,
    )

    interactive = sys.stdin.isatty() and not args.no_prompt
    policy = ApprovalPolicy(ask=_ask_approval if interactive else deny_all)
    if args.auto_approve:
        policy.ask = allow_all

    registry = build_registry()
    try:
        provider = build_provider(args.provider, args.model)
    except ProviderUnavailable as error:
        print(f"\n{error}")
        print(
            "\nThe tools below still work through the slash commands, and every "
            "deterministic stage is available through `polyrmc process`."
        )
        provider = None

    registry_size = len(registry)
    indexed = len(RunRegistry(runs_dir).summaries)
    budget = Budget(
        max_requests=args.budget_requests,
        max_tokens=args.budget_tokens,
        rates=rates_from_env(),
    )

    print(f"\n{GREETING}")
    if provider is not None:
        agent = Orchestrator(
            context=context,
            provider=provider,
            approvals=policy,
            registry=registry,
            max_steps=args.max_steps,
            budget=budget,
        )
        print(f"model: {provider.name} / {provider.model}   tools: {registry_size}")
        print(
            "lab context: "
            + str(agent.lab_path or "NONE FOUND -- running without house conventions")
        )
    else:
        agent = None
    print(f"memory: {indexed} past run(s) in {runs_dir}, {len(notes)} note(s)")
    if not args.auto_approve:
        print("commits need your approval. /help for commands.\n")
    else:
        print("--auto-approve: every commit runs unattended and is recorded.\n")

    if not interactive:
        if agent is None:
            return 2
        question = args.prompt or "Summarise where this run stands and what it needs."
        try:
            for event in agent.ask(question):
                print(event.render())
        except Exception as error:
            print(f"the model call failed: {type(error).__name__}: {error}")
            return 2
        return 0

    while True:
        try:
            line = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nsession ended.")
            break
        if not line:
            continue

        if line.startswith("/"):
            command, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            if command in {"exit", "quit", "q"}:
                print("session ended.")
                break
            if command == "help":
                print(REPL_HELP)
            elif command == "state":
                print(render_state(surface))
            elif command == "tools":
                print(describe_registry(registry))
            elif command == "history":
                print(agent.narrate() if agent else "(no model session)")
            elif command == "runs":
                print(registry["search_past_runs"].handler(context, {"query": rest or None}))
            elif command == "notes":
                print(registry["read_notes"].handler(context, {"query": rest or None}))
            elif command == "always":
                if rest in registry and registry[rest].mutating:
                    policy.always.add(rest)
                    print(f"{rest} approved for this session")
                else:
                    print(f"{rest!r} is not a commit tool. /tools lists them.")
            elif command == "auto":
                policy.ask = allow_all
                print("every commit will now run without asking.")
            elif command == "cost":
                print(budget.render())
            else:
                print(f"unknown command {command!r}. /help for the list.")
            print()
            continue

        if agent is None:
            print("no model is configured, so there is nothing to ask. /help\n")
            continue

        try:
            for event in agent.ask(line):
                print(event.render())
        except Exception as error:  # a provider fault must not end the session
            print(f"  model call failed: {error}")
        print()

    # Printed unconditionally on the way out, including after an interrupt. A
    # spend figure that only appears on a clean exit is missing from exactly the
    # sessions that ran away.
    if agent is not None:
        print(budget.render("spent this session"))
    return 0


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


def run_chat(args) -> int:
    """``chat`` and ``agent`` are the same session under two names.

    They were different things once: one drove a fixed graph over a nominated
    run, the other answered questions about past ones. A session that can call
    tools does both, and keeping two commands would only invite the question of
    which one to start.
    """
    return run_agent(args)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    # A Windows console is cp1252 unless told otherwise, and a model's reply
    # routinely carries a subscript or a Greek letter. Losing a whole session --
    # including the record of what it just committed -- to an encoding error on
    # the way to the screen is not a trade worth making, so unrepresentable
    # characters degrade instead.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # not a reconfigurable stream
            pass

    parser = argparse.ArgumentParser(prog="polyrmc", description=__doc__)
    parser.add_argument(
        "stage", choices=["intake", "process", "agent", "chat", "part1", "part2", "run"]
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
    parser.add_argument(
        "--provider",
        choices=["gemini", "claude"],
        default=os.environ.get("POLYRMC_PROVIDER", "gemini"),
        help="Model that drives the agent session. Gemini is the default and is "
        "the one keyed in .env; Claude needs ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="Override the provider's model id.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Tool-calling rounds allowed per question before the agent stops "
        "and reports where it got to.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Run commits without asking. Every one is still validated and "
        "recorded; only the confirmation is skipped.",
    )
    parser.add_argument(
        "--notes-dir",
        help="Where session notes are kept. Defaults to data/memory.",
    )
    parser.add_argument(
        "--prompt",
        help="A single question for a non-interactive agent session.",
    )
    parser.add_argument(
        "--budget-requests",
        type=int,
        default=int(os.environ.get("POLYRMC_BUDGET_REQUESTS", "0")),
        help="Stop the session after this many model requests. The Gemini free "
        "tier rations requests, not tokens, so this is the cap that matters "
        "there. 0 means no ceiling.",
    )
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=int(os.environ.get("POLYRMC_BUDGET_TOKENS", "0")),
        help="Stop the session after this many tokens. 0 means no ceiling.",
    )
    parser.add_argument("--out", help="Where to write the processed CSV.")
    parser.add_argument("--run-id", help="Identifier for this run.")
    parser.add_argument(
        "--runs-dir",
        help="Directory containing historical .provenance.json run sidecars.",
    )
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

    if args.stage == "chat":
        return run_chat(args)

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

    if args.stage == "process":
        if not (args.raw and args.declarations):
            parser.error("process requires --raw and --declarations")
        return run_process(args)

    if args.stage == "agent":
        # No required arguments. An agent session with no run nominated is a
        # normal way to start -- searching what the lab did before does not need
        # a file in hand -- and the agent reports the run as unloaded until it
        # is pointed at one.
        if bool(args.raw) != bool(args.declarations):
            parser.error(
                "--raw and --declarations go together: a run is analysed under "
                "what a person declared it to be, never under one without the other"
            )
        return run_agent(args)

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
