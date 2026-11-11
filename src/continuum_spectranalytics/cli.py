"""Command-line entry point.

    python -m continuum_spectranalytics.cli agent
    python -m continuum_spectranalytics.cli gui
    python -m continuum_spectranalytics.cli agent --raw data/raw/<file> --declarations run.decl.json
    python -m continuum_spectranalytics.cli intake --raw data/raw/<file> --declarations run.decl.json
    python -m continuum_spectranalytics.cli process --raw data/raw/<file> --declarations run.decl.json
    python -m continuum_spectranalytics.cli part1 --config run.json
    python -m continuum_spectranalytics.cli part2 --config run.json --csv data/runs/<run>.csv
    python -m continuum_spectranalytics.cli run   --config run.json

``agent`` is the interactive session, and the one to reach for by default. The
model drives: it reads the run, searches what the lab did before, and calls the
tools it needs, stopping to ask you before anything commits. Started with no
arguments it is still useful -- searching past experiments does not need a file
in hand. ``gui`` is that same session in a window, with the run state and the
figures beside the conversation instead of recalled on demand.

``intake`` is the section 6.0 interview: it asks what the run is, records each
answer with an author and a timestamp, then checks those answers against the
trace. ``process`` takes those declarations straight through to a verified CSV.
Neither makes a model call or needs an API key, and neither has changed: the
deterministic path is the same path whether a model chose to walk it or a
researcher typed the command.

No stage but ``agent`` consults a model at all: ``intake``, ``process``,
``part1``, ``part2`` and ``plot`` are pure measurement and need no key.
"""

from __future__ import annotations

import argparse
import json
import os
import signal as _signal
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from continuum_spectranalytics.config import RunConfig
from continuum_spectranalytics.intake import (
    DeclarationSet,
    MissingDeclarations,
    build_intake_form,
    render_question,
    to_config,
)
from continuum_spectranalytics import APP_NAME, env_setting, paths
from continuum_spectranalytics.pipeline import new_run_id, run_part1, run_part2
from continuum_spectranalytics.provenance import summarize_loop
from continuum_spectranalytics.state import Declaration
from continuum_spectranalytics.tier0.argen_io import DEFAULT_CHANNEL, load_argen_file
from continuum_spectranalytics.tier0.consistency import check_declarations
from continuum_spectranalytics.verify import summarize, verify_output


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
        print(f"\nRun intake first:\n  continuum-spectranalytics intake --raw {args.raw} "
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
    state, csv_path = run_part1(config, output_csv=args.out)
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
    """Build the run configuration a session starts from.

    Three entry shapes, and the important one is the third.

    With ``--raw`` and ``--declarations``, the configuration is *derived* from
    what a person declared, exactly as ``process`` derives it, so a run cannot
    be declared one thing and analysed as another.

    With ``--config``, it is whatever that file says.

    With **just a raw file, or nothing at all**, the configuration is bare:
    every physical constant is ``None``, meaning nobody has answered yet. That
    is a real state rather than a failure. The trace loads, the anomalies are
    found, the smoothing band is computed and the figures draw, because none of
    that needs optics -- and the columns that do need an answer are blocked with
    the reason recorded, which ``concentration_column`` and the rest have always
    done. Declarations arrive through ``set_declaration`` as the researcher
    supplies them, attributed and timestamped, and land in the run's own record.

    What used to happen instead was worse than requiring the file. A session
    with no declarations was scaffolded from ``examples/run_config.example.json``
    -- so ``dn_dc`` was silently 0.150, and a protein run taken all the way to
    ``extract_parameters`` would have reported a molar mass a third out with a
    clean fit and no warning anywhere. An invented constant is the one thing
    this pipeline must never carry, and it was carrying one by default.
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

    if args.config:
        return load_config(args.config), {}, args.run_id or "session"

    if args.raw:
        run_id = args.run_id or new_run_id(Path(args.raw).stem[:16])
    else:
        run_id = args.run_id or "session"

    return (
        RunConfig(
            run_id=run_id,
            source_file=args.raw or "none",
            channel=args.channel or DEFAULT_CHANNEL,
        ),
        {},
        run_id,
    )


@dataclass
class Session:
    """The run surface one agent session acts on, and what it was built from."""

    config: "RunConfig"
    declarations: dict
    run_id: str
    surface: object
    context: object
    notes: object
    runs_dir: Path


def open_session(args) -> Session:
    """Build the run surface from the arguments a session was started with.

    Shared by the terminal session and the window, deliberately. Two
    constructions of the same run would eventually disagree -- a declaration
    applied in one and not the other, a runs directory resolved differently --
    and the disagreement would show up as a run analysed one way at a prompt and
    another way in a window. There is one way to open a run and this is it.
    """
    from continuum_spectranalytics.agent.toolspec import ToolContext, default_ledger_path
    from continuum_spectranalytics.memory.notes import NoteBook
    from continuum_spectranalytics.state import RunState
    from continuum_spectranalytics.tools import ToolSurface

    config, declarations, run_id = _session_config(args)

    state = RunState(run_id=run_id, source_file=str(args.raw or "none"))
    state.declarations = dict(declarations)
    surface = ToolSurface(state=state, config=config)
    for key, declaration in declarations.items():
        surface.declarations.declarations[key] = declaration
    surface.declarations.refresh_form()

    runs_dir = Path(args.runs_dir) if args.runs_dir else paths.runs_dir()
    notes = NoteBook(args.notes_dir)
    return Session(
        config=config,
        declarations=declarations,
        run_id=run_id,
        surface=surface,
        context=ToolContext(
            surface=surface,
            notes=notes,
            runs_dir=runs_dir,
            ledger_path=default_ledger_path(),
            author=str(args.author or "researcher"),
            output_csv=Path(args.out) if args.out else None,
        ),
        notes=notes,
        runs_dir=runs_dir,
    )


def _ask_approval(tool, call):
    """Put one commit to the researcher.

    The arguments are shown in full, one per line. An approval prompt that hides
    what is about to change trains the reader to approve without looking, which
    is worse than not asking at all -- so this is the one place in the session
    where nothing is summarised.

    What it no longer shows is the *call*. ``pick_window(window_length=31)`` is
    the shape of the request going over the wire, and a person deciding whether
    to allow a change is not reviewing a request; they are reviewing a change.
    Same facts, named the way the rest of the session names them.
    """
    from continuum_spectranalytics.agent.orchestrator import Approval

    # The line above this one already named the activity, because the call event
    # is reported before the tool runs. Repeating the phrase here made the same
    # sentence appear twice in a row and read like two separate things
    # happening; what this block has to add is that it *commits*, and what it
    # will write.
    print("\n      COMMIT -- needs your approval")
    for key, value in call.arguments.items():
        print(f"        {key.replace('_', ' ')}: {value}")
    print(f"      {tool.description.splitlines()[0]}")
    try:
        answer = input("      allow? [y]es / [n]o / [a]lways / reason: ").strip()
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
  /providers        which model providers are keyed on this machine
  /log [n]          the session log: where it is, and its last n lines
  /help             this list
  /exit             end the session"""


def _persist_key(spec, value: str, env_path: Path | None = None) -> bool:
    """Write one key into ``.env``, replacing any existing line for it.

    ``.env`` is gitignored, which is the only reason offering to write here is
    defensible. The file is rewritten whole rather than appended to, so signing
    in twice leaves one line rather than two that disagree.

    The path comes from :func:`continuum_spectranalytics.paths.env_file`, not from the working
    directory. Writing to ``Path(".env")`` put the key wherever the researcher
    happened to be standing, while the loader looked beside the package -- so a
    key saved from a data folder was silently never read again.
    """
    env_path = Path(env_path) if env_path is not None else paths.env_file()
    try:
        lines = (
            env_path.read_text(encoding="utf-8-sig").splitlines()
            if env_path.exists()
            else []
        )
        kept = [
            line
            for line in lines
            if not line.strip().startswith(f"{spec.canonical_key}=")
        ]
        kept.append(f"{spec.canonical_key}={value}")
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return True
    except OSError as error:
        print(f"  could not write {env_path}: {error}")
        return False


def _sign_in(spec) -> bool:
    """Take a key for one provider at the terminal.

    Read with ``getpass`` so the key never reaches the scrollback, and never
    echoed back for confirmation. Offering to store it is a separate question
    from accepting it, because a shared or teaching machine is a normal place to
    run this and a key written there outlives the session.
    """
    from getpass import getpass

    print(f"\n{spec.name} needs a key you supply. Get one from {spec.key_url}")
    print(f"It will be read as {spec.canonical_key}. Leave blank to skip.")
    try:
        value = getpass(f"  paste {spec.canonical_key}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    if not value:
        return False

    os.environ[spec.canonical_key] = value
    env_path = paths.env_file()
    try:
        answer = input(f"  save it to {env_path} for next time? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        # The key is already accepted for this session. Losing the terminal
        # while deciding whether to store it should not also lose the session.
        answer = ""
        print()
    if answer in {"y", "yes"}:
        if _persist_key(spec, value):
            print(f"  written to {env_path} ({spec.canonical_key}). It is gitignored.")
    else:
        print("  kept for this session only.")
    return True


def _resolve_provider(args, interactive: bool, on_wait=None):
    """Pick the brain for this session, offering to take a key if none is set.

    Order: what was asked for, else whatever is already keyed, else -- at a
    terminal -- ask. A researcher who has not chosen should not have to know
    which provider this package happens to list first.
    """
    from continuum_spectranalytics.agent.toolcall import (
        SPECS,
        ProviderUnavailable,
        build_provider,
        describe_providers,
        key_for,
        keyed_providers,
    )

    kind = args.provider or (keyed_providers() or [None])[0]

    if kind is None:
        if not interactive:
            raise ProviderUnavailable(
                "No model provider is keyed on this machine. This is a "
                "buy-your-own-tokens setup: pick a provider, get a key, and set "
                "it.\n" + describe_providers()
            )
        print("\nNo provider is keyed yet. You supply the tokens for this agent.")
        print(describe_providers())
        choice = input(f"  which provider? [{'/'.join(SPECS)}] ").strip().lower()
        if choice not in SPECS:
            raise ProviderUnavailable(
                f"{choice!r} is not one of {sorted(SPECS)}. Start again once you "
                "have a key."
            )
        kind = choice

    spec = SPECS.get(kind)
    if spec is not None and not key_for(spec) and interactive:
        _sign_in(spec)

    return build_provider(
        kind,
        args.model,
        min_interval=float(getattr(args, "min_request_interval", 0.0) or 0.0),
        on_wait=on_wait,
    )


def _check_key(kind: str, model: str | None = None) -> tuple[bool, str]:
    """Spend one small call to find out whether a key actually works.

    Worth the token. The alternative is telling a researcher their key is
    stored, letting them start a session, and having the first real question
    fail on a key that was pasted short or belongs to another provider -- by
    which point the failure looks like the agent rather than the setup.

    The prompt is deliberately tiny, and no tools are bound: this is checking
    that a key opens a door, not that a model can think.
    """
    from continuum_spectranalytics.agent.toolcall import (
        Message,
        ProviderCallFailed,
        ProviderUnavailable,
        build_provider,
    )

    try:
        provider = build_provider(kind, model)
    except ProviderUnavailable as error:
        return False, str(error)

    try:
        provider.converse(
            "Answer with the single word: ok",
            [Message(role="user", text="ping")],
            [],
        )
    except ProviderCallFailed as error:
        return False, str(error)
    except Exception as error:  # noqa: BLE001 - any fault here is a failed check
        return False, f"{type(error).__name__}: {error}"
    return True, f"{provider.name} / {provider.model}"


def run_setup(args) -> int:
    """Take a key and store it, before anything else is attempted.

    This exists because the sign-in prompt it wraps was unreachable. It lived
    inside the agent session, behind the check that a run had been declared, so
    the first thing a new user saw was a demand for declarations about an
    experiment they had not loaded -- and nothing anywhere told them a key was
    needed, or where to get one. A thing you must do first has to be runnable
    first.

    Needs no run, no config, and no declarations, and it is the one command in
    this package that is allowed to make a model call without a run in hand.
    """
    from continuum_spectranalytics.agent.toolcall import SPECS, key_for

    interactive = sys.stdin.isatty() and not args.no_prompt
    env_path = paths.env_file()

    print(f"\n{APP_NAME} needs a model provider. You supply the tokens.")
    print(f"project root: {paths.project_root()}")
    print(f"keys are read from and written to: {env_path}\n")

    order = list(SPECS)
    for index, name in enumerate(order, start=1):
        spec = SPECS[name]
        status = "keyed" if key_for(spec) else "no key"
        print(f"  {index}  {spec.name:<8} {status:<7} {spec.key_url}")

    # The measurement path is the part that must never depend on this, and a
    # researcher who wanted to process data rather than talk to a model should
    # be told they are already finished.
    print(
        "\nNone of this is needed for `continuum-spectranalytics intake`, `continuum-spectranalytics process`, or "
        "`continuum-spectranalytics process`,\nwhich runs the whole measurement path with no model "
        "and no key."
    )

    if not interactive:
        keyed = [name for name in order if key_for(SPECS[name])]
        print(f"\nkeyed here: {', '.join(keyed) if keyed else 'none'}")
        if not keyed:
            print("Run `continuum-spectranalytics setup` at a terminal to paste one in.")
            return 1
        return 0

    choice = (args.provider or "").strip().lower()
    if not choice:
        try:
            choice = input(f"\n  which provider? [1-{len(order)}] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nnothing changed.")
            return 1
    if choice.isdigit() and 1 <= int(choice) <= len(order):
        choice = order[int(choice) - 1]
    if choice not in SPECS:
        print(f"\n{choice!r} is not one of {order}. Nothing changed.")
        return 1

    spec = SPECS[choice]
    replacing = bool(key_for(spec))
    if replacing:
        print(f"\n{spec.name} is already keyed ({spec.canonical_key}).")
        try:
            answer = input("  replace it? [y/N] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = ""
            print()
        replacing = answer in {"y", "yes"}
        if not replacing:
            print("  keeping the key already set.")

    # Asked for whenever there is no key, and whenever the researcher chose to
    # replace one. Keeping an existing key is the only path that skips straight
    # to the check.
    if not key_for(spec) or replacing:
        if not _sign_in(spec):
            print("\nno key taken. Nothing changed.")
            return 1

    print("\n  checking...", end=" ", flush=True)
    ok, detail = _check_key(spec.name, args.model)
    if ok:
        print(f"ok ({detail})")
        print("\nReady. Start a session with `continuum-spectranalytics agent`.")
        return 0

    # A failed check does not unwrite the key. The paste may be fine and the
    # provider merely down, and making a researcher re-enter a 40-character
    # secret to find that out is a worse trade than leaving a key that says
    # plainly it did not answer.
    print("failed")
    print(f"\n{detail}")
    print(
        f"\nThe key is still stored in {env_path}. Fix what the provider named "
        "and run\n`continuum-spectranalytics setup` again, or pick another provider."
    )
    return 1


def run_plot(args) -> int:
    """Open a data file and plot columns from it. No model, no key, no run.

    The gap this fills is the reason a lab keeps a second application open. The
    rest of this package works on runs it has processed, which is right for
    producing a number and useless for the ordinary act of looking at a file
    somebody just handed you. That act should not require declaring an
    experiment, and it certainly should not require an API key.

    With no ``--x`` it describes the file instead of plotting it, because the
    first thing anyone needs is the column names, and guessing them from an
    instrument export is how you plot the wrong channel.
    """
    from continuum_spectranalytics import figures
    from continuum_spectranalytics.kaleidagraph import NotAQpcFile, read_qpc
    from continuum_spectranalytics.tabular import UnsupportedFile, open_table

    if not args.file:
        print("plot needs --file <path>.")
        return 2

    # A KaleidaGraph plot is described rather than opened: it has no rows, so
    # there is nothing to put on an axis, but it does say what was plotted and
    # which fit was used -- which is usually why somebody opened it.
    if Path(args.file).suffix.lower() == ".qpc":
        try:
            plot = read_qpc(args.file)
        except (NotAQpcFile, OSError) as error:
            print(f"\n{error}")
            return 1
        print(f"\n{Path(args.file).name}: KaleidaGraph plot")
        for title, values in (
            ("text in the plot (axis titles and series names, in file order)", plot.labels),
            ("curve fits, fullest reading first", plot.fit_equations),
            ("data it was built from", plot.source_references),
        ):
            if values:
                print(f"\n{title}:")
                for value in values:
                    print(f"  {value}")
        if plot.other_text:
            print("\nrecovered but not classified:")
            for value in plot.other_text[:10]:
                print(f"  {value}")
        for note in plot.notes:
            print(f"\nnote: {note}")
        return 0

    try:
        table = open_table(args.file)
    except UnsupportedFile as error:
        print(f"\n{error}")
        return 2
    except (FileNotFoundError, IsADirectoryError, ValueError) as error:
        print(f"\n{error}")
        return 1

    source = Path(args.file)
    numeric = table.numeric_columns()

    if not args.x:
        print(f"\n{source.name}: {table.kind}, {len(table.frame)} rows, "
              f"{len(table.columns)} columns")
        print("\nplottable columns:")
        for name in numeric:
            print(f"  {name}")
        other = [c for c in table.columns if c not in numeric]
        if other:
            print("\nnot plottable (not numeric):")
            for name in other:
                print(f"  {name}")
        for note in table.notes:
            print(f"\nnote: {note}")
        print(f"\n{table.preview()}")
        print(
            f"\nPlot it with:  continuum-spectranalytics plot --file {source.name} --x "
            f"{numeric[0] if numeric else 'X'} --y "
            f"{numeric[1] if len(numeric) > 1 else 'Y'}"
        )
        return 0

    if not args.y:
        print("plot needs --y <column> [<column> ...] once --x is given.")
        return 2

    # Beside the file it came from, which is where somebody looking at a file
    # expects the picture of it to land. The agent's own plot_columns writes
    # into the runs directory instead: a session may be pointed at a file in a
    # folder it cannot write to, and a known-writable location is worth more
    # there than matching this convention.
    out_path = Path(args.out) if args.out else source.with_name(
        f"{source.stem}_{args.x}_vs_{'_'.join(args.y)}.png"
    )

    try:
        written = figures.plot_columns(
            table.frame,
            args.x,
            args.y,
            out_path,
            title=args.title,
            style=args.style,
            logx=args.logx,
            logy=args.logy,
            source=f"{source.name} ({table.kind})",
        )
    except ValueError as error:
        print(f"\n{error}")
        return 1

    print(f"\nwrote {written}")
    for note in table.notes:
        print(f"note: {note}")
    return 0


def _log_warnings_to(log) -> None:
    """Send Python warnings to the session log instead of the terminal.

    Provider SDKs warn about their own internals -- one of them announces that
    the model ignores ``temperature``, once per request -- and those lines
    arrive interleaved with the conversation, look like the session complaining
    about the run, and are about neither. They are still worth keeping, so they
    go where the rest of the background work goes.
    """
    import warnings

    def show(message, category, filename, lineno, file=None, line=None):
        log.write(
            f"warning {category.__name__}",
            f"{message}\n  from {filename}:{lineno}",
        )

    warnings.showwarning = show


def _tail_log(log, argument: str = "") -> str:
    """The log's location, and its last few lines.

    Reading the file back rather than replaying a buffer, so what ``/log`` shows
    is what a colleague opening the file would see.
    """
    try:
        count = max(1, int(argument))
    except (TypeError, ValueError):
        count = 20
    if log.path is None or not log.path.exists():
        return f"session log: {log.location}"
    try:
        lines = log.path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        return f"session log: {log.location} (cannot be read back: {error})"
    body = "\n".join(lines[-count:])
    return f"session log: {log.location}\n\n{body}"


class Interruptible:
    """Ctrl+C ends the turn, not the process.

    The terminal had no way to interrupt a running turn. Ctrl+C raised straight
    out through the generator, which lost the spend figure, lost the pointer to
    the session log, and left the conversation holding an assistant turn whose
    tool calls had no results -- so a session that survived it could not have
    asked anything else anyway.

    Now it sets the same cooperative flag the window's Stop button sets, and the
    turn ends at its next boundary. It has to be cooperative for the same reason
    there: a model call is a blocking request inside a vendor SDK and a tool is
    numpy over eighty thousand points, and neither can be cut off without
    leaving the run in a state nobody can describe.

    The second Ctrl+C is the escape hatch. If the researcher has already asked
    once and nothing has come back, the provider is hung and they are entitled
    to their terminal back.

    Used as a context manager so the handler is only installed for the duration
    of a turn: at the prompt, Ctrl+C should end the session as it always did.
    """

    NOTICE = (
        "\n  stopping -- waiting for the model call or tool already running to "
        "return.\n  Ctrl+C again to give up on it."
    )

    def __init__(self, on_notice=print) -> None:
        self.requested = False
        self._previous = None
        self._on_notice = on_notice

    def __call__(self) -> bool:
        """The poll the orchestrator holds."""
        return self.requested

    def _handle(self, signum, frame):
        if self.requested:
            self._restore()
            raise KeyboardInterrupt
        self.requested = True
        self._on_notice(self.NOTICE)

    def _restore(self) -> None:
        if self._previous is not None:
            _signal.signal(_signal.SIGINT, self._previous)
            self._previous = None

    def __enter__(self) -> "Interruptible":
        self.requested = False
        try:
            self._previous = _signal.signal(_signal.SIGINT, self._handle)
        except ValueError:
            # Not the main thread, so there is no signal to take. The turn runs
            # to completion, which is what happened before any of this existed.
            self._previous = None
        return self

    def __exit__(self, *exc) -> None:
        self._restore()


def run_agent(args) -> int:
    """The agent session: the model drives, and commits go through you.

    Nothing here decides the route. The agent reads the state, calls what it
    needs, and stops to ask when a change belongs to a person.
    """
    from continuum_spectranalytics.agent.budget import Budget, rates_from_env
    from continuum_spectranalytics.agent.orchestrator import (
        GREETING,
        ApprovalPolicy,
        Orchestrator,
        allow_all,
        deny_all,
    )
    # ProviderUnavailable only: a session that cannot start is this function's
    # problem, while a provider that dies mid-session is now the orchestrator's
    # -- it comes back as a ``failed`` event so the work already done comes back
    # with it, rather than as an exception that unwinds past the events.
    from continuum_spectranalytics.agent.toolcall import ProviderUnavailable
    from continuum_spectranalytics.agent.toolspec import build_registry, describe_registry, render_state
    from continuum_spectranalytics.agent.transcript import Narrator, SessionLog, error_block
    from continuum_spectranalytics.memory.registry import RunRegistry

    try:
        session = open_session(args)
    except MissingDeclarations as error:
        print(f"cannot start: {error}")
        print(
            f"\nRun intake first:\n  continuum-spectranalytics intake --raw {args.raw} "
            f"--declarations {args.declarations} --author <you>"
        )
        return 2

    surface = session.surface
    context = session.context
    notes = session.notes
    runs_dir = session.runs_dir
    run_id = session.run_id

    interactive = sys.stdin.isatty() and not args.no_prompt
    policy = ApprovalPolicy(ask=_ask_approval if interactive else deny_all)
    if args.auto_approve:
        policy.ask = allow_all

    log = SessionLog()
    narrator = Narrator(log, verbose=bool(getattr(args, "verbose", False)))
    _log_warnings_to(log)

    def announce_wait(seconds: float, reason: str) -> None:
        """Say that the session is waiting, rather than appearing to hang.

        A rate limit can ask for the better part of a minute. Silence for that
        long, in the middle of a run that has been printing a line every few
        seconds, reads as a crash -- and a researcher who kills it loses a
        conversation that cost real tokens to build.
        """
        print(f"  ... waiting {seconds:.0f}s: {reason}")
        log.write("rate limit", f"waiting {seconds:.1f}s -- {reason}")

    registry = build_registry()
    try:
        provider = _resolve_provider(args, interactive, on_wait=announce_wait)
    except ProviderUnavailable as error:
        print(f"\n{error}")
        print(
            "\nThe tools below still work through the slash commands, and every "
            "deterministic stage is available through `continuum-spectranalytics process`."
        )
        provider = None

    registry_size = len(registry)
    indexed = len(RunRegistry(runs_dir).summaries)
    budget = Budget(
        max_requests=args.budget_requests,
        max_tokens=args.budget_tokens,
        rates=rates_from_env(),
    )

    interrupts = Interruptible()

    print(f"\n{GREETING}")
    if provider is not None:
        agent = Orchestrator(
            context=context,
            provider=provider,
            approvals=policy,
            registry=registry,
            max_steps=args.max_steps,
            max_web_calls=getattr(args, "max_web_calls", 8),
            budget=budget,
            should_stop=interrupts,
        )
        print(f"model: {provider.name} / {provider.model}   tools: {registry_size}")
        print(
            "lab context: "
            + str(agent.lab_path or "NONE FOUND -- running without house conventions")
        )
    else:
        agent = None
    print(f"memory: {indexed} past run(s) in {runs_dir}, {len(notes)} note(s)")
    # Named on the way in as well as on the way out, so a researcher who wants
    # the detail knows where it is before they need it rather than after.
    print(f"working notes: {log.location}")
    log.banner(
        run=run_id,
        model=f"{provider.name} / {provider.model}" if provider else "none configured",
        tools=registry_size,
        lab_context=str(agent.lab_path) if agent else "",
        runs_dir=str(runs_dir),
    )
    if not args.auto_approve:
        print("commits need your approval. /help for commands.\n")
    else:
        print("--auto-approve: every commit runs unattended and is recorded.\n")

    def report(events) -> str:
        """Show one turn, and say how it ended.

        Three outcomes, because two different questions are asked of the answer.
        ``"dead"`` means the brain is gone and the session is over. ``"capped"``
        means this turn stopped without answering -- a budget or the step limit
        -- which a person at a terminal recovers from by raising the ceiling and
        asking again, and a script cannot recover from at all. ``"ok"`` means it
        answered. Collapsing the last two loses the case a batch caller most
        needs to detect: a question that returned quietly having answered
        nothing.

        A stop is not narrated as though the model had said it. It comes out as
        a banner with the reason, what to do about it, and where the provider's
        own words are -- the four hundred characters of JSON that used to land
        in the middle of a transcript.
        """
        outcome = "ok"
        for event in events:
            narrator.show(event)
            if event.kind == "failed":
                failure = event.detail
                print(
                    error_block(
                        failure.headline if failure else "the model call failed",
                        (failure.meaning if failure else str(event.text))
                        + f" {budget.render('Spent before it stopped', nudge=False)}.",
                        failure.remedy if failure else "",
                        log.location,
                    )
                )
                outcome = "dead"
            elif event.kind == "capped":
                print(
                    error_block(
                        "stopped without answering",
                        f"{event.text} {budget.render('Spent', nudge=False)}.",
                        "Raise the ceiling and ask again, or run `continuum-spectranalytics "
                        "process` for the whole deterministic path with no "
                        "model at all.",
                        log.location,
                    )
                )
                if outcome == "ok":
                    outcome = "capped"
        return outcome

    if not interactive:
        if agent is None:
            log.close()
            return 2
        question = args.prompt or "Summarise where this run stands and what it needs."
        narrator.user(question)
        try:
            outcome = report(agent.stream(question))
        except Exception as error:  # a bug here must not cost the record
            log.write("session error", f"{type(error).__name__}: {error}")
            print(error_block("the session failed", f"{type(error).__name__}: {error}",
                              "", log.location))
            log.close()
            return 2
        log.close()
        # Anything short of an answer is a non-zero exit. A script that asked a
        # question and got a banner instead has not been answered, and telling
        # it otherwise is how a caller ends up treating a spent key as a result.
        return 0 if outcome == "ok" else 2

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
            elif command == "log":
                print(_tail_log(log, rest))
            elif command == "providers":
                from continuum_spectranalytics.agent.toolcall import describe_providers

                print(describe_providers())
                if provider is not None:
                    print(f"\nthis session: {provider.name} / {provider.model}")
            else:
                print(f"unknown command {command!r}. /help for the list.")
            print()
            continue

        if agent is None:
            print("no model is configured, so there is nothing to ask. /help\n")
            continue

        narrator.user(line)
        try:
            with interrupts:
                outcome = report(agent.stream(line))
            if interrupts.requested:
                log.write("session", "stopped by Ctrl+C")
        except Exception as error:  # a bug here must not cost the transcript
            log.write("session error", f"{type(error).__name__}: {error}")
            print(
                error_block(
                    "the session hit a bug",
                    f"{type(error).__name__}: {error}",
                    "Nothing committed is lost. /state shows where the run "
                    "stands, and `continuum-spectranalytics process` runs the whole deterministic "
                    "path with no model at all.",
                    log.location,
                )
            )
            outcome = "ok"  # a fault in this loop is not a dead provider
        # A cap is recoverable at a terminal -- raise it and ask again -- so
        # only a provider that stopped answering ends the session here.
        if outcome == "dead":
            # The brain is gone, so the session is over. Tool faults are handled
            # inside the orchestrator and fed back for the model to correct;
            # nothing can correct a provider that will not answer, and looping
            # here would re-ask on a key that just refused. There is no failover
            # onto a second provider on purpose: these are the researcher's own
            # tokens, and moving their session to another of their keys is a
            # spending decision that is theirs to make.
            print(
                "\nsession ended. Nothing committed is lost -- /state on the next\n"
                "session shows where this run stands."
            )
            break
        print()

    # Printed unconditionally on the way out, including after an interrupt. A
    # spend figure that only appears on a clean exit is missing from exactly the
    # sessions that ran away, and the same goes for where the detail went.
    if agent is not None:
        print(budget.render("spent this session"))
    print(f"working notes: {log.location}")
    log.close()
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


def run_gui(args) -> int:
    """The same session, in a window.

    Everything the terminal session enforces is enforced there too, because it
    is the same orchestrator over the same tool surface: commits go through the
    approval policy, refusals arrive whole, the loop is capped, and the log
    under ``data/logs`` is written by the same narrator. What the window adds is
    the run state permanently on screen and the figures beside the conversation
    that produced them.
    """
    from continuum_spectranalytics.gui import launch

    return launch(args)


def run_chat(args) -> int:
    """``chat`` and ``agent`` are the same session under two names.

    They were different things once: one drove a fixed graph over a nominated
    run, the other answered questions about past ones. A session that can call
    tools does both, and keeping two commands would only invite the question of
    which one to start.
    """
    return run_agent(args)


def main(argv: list[str] | None = None) -> int:
    # Loaded from the resolved project root rather than by search. python-dotenv
    # walks up from whichever frame calls it, which meant the file found here
    # depended on the working directory and on whether the package was installed
    # as a checkout or a wheel -- from a data folder it resolved to a `.env` in
    # the user's home directory that nothing had ever written.
    load_dotenv(paths.env_file())

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

    parser = argparse.ArgumentParser(
        prog="continuum-spectranalytics", description=f"{APP_NAME}\n\n{__doc__}"
    )
    parser.add_argument(
        "stage",
        choices=[
            "setup",
            "intake",
            "process",
            "plot",
            "agent",
            "gui",
            "chat",
            "part1",
            "part2",
            "run",
        ],
        help="setup takes an API key and stores it; intake, process and plot are "
        "the measurement path and need no key; agent is the session, and gui is "
        "the same session in a window with the figures beside it.",
    )
    parser.add_argument("--file", help="Data file to open or plot (plot only).")
    parser.add_argument("--x", help="Column for the x axis (plot only).")
    parser.add_argument(
        "--y", nargs="+", help="One or more columns for the y axis (plot only)."
    )
    parser.add_argument(
        "--style", choices=["line", "scatter"], default="line",
        help="How to draw the series (plot only).",
    )
    parser.add_argument("--logx", action="store_true", help="Log x axis (plot only).")
    parser.add_argument("--logy", action="store_true", help="Log y axis (plot only).")
    parser.add_argument("--title", help="Figure title (plot only).")
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
        "--provider",
        choices=["gemini", "claude", "openai"],
        default=env_setting("PROVIDER") or None,
        help="Brain that drives the agent session, on your own key. Defaults to "
        "whichever provider is keyed on this machine. GOOGLE_API_KEY or "
        "GEMINI_API_KEY for gemini, ANTHROPIC_API_KEY for claude, "
        "OPENAI_API_KEY for openai.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="Override the provider's model id. The defaults are pins and can go "
        "stale; this is how you move off one without editing the package.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Tool-calling rounds allowed per question before the agent stops "
        "and reports where it got to.",
    )
    parser.add_argument(
        "--max-web-calls", type=int,
        default=int(env_setting("MAX_WEB_CALLS", "8")),
        help="Internet tool calls allowed per question, including batched calls. Default 8; 0 disables agent web requests.",
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
        "--min-request-interval",
        type=float,
        default=float(env_setting("MIN_REQUEST_INTERVAL", "0") or 0),
        help="Seconds to leave between model requests. A free tier rations "
        "requests per minute and an agent spends one per tool round, so a "
        "session doing real work trips that ceiling within a few rounds; 13 "
        "paces a 5-per-minute limit. 0 (the default) does not wait.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print the session's background work -- full tool output and "
        "per-turn token counts -- which otherwise only goes to the session log.",
    )
    parser.add_argument(
        "--prompt",
        help="A single question for a non-interactive agent session.",
    )
    parser.add_argument(
        "--budget-requests",
        type=int,
        default=int(env_setting("BUDGET_REQUESTS", "0")),
        help="Stop the session after this many model requests. The Gemini free "
        "tier rations requests, not tokens, so this is the cap that matters "
        "there. 0 means no ceiling.",
    )
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=int(env_setting("BUDGET_TOKENS", "0")),
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
    if args.max_web_calls < 0:
        parser.error("--max-web-calls must be zero or a positive integer")

    if args.stage == "setup":
        return run_setup(args)

    if args.stage == "plot":
        return run_plot(args)

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

    if args.stage in {"agent", "gui"}:
        # No required arguments. An agent session with no run nominated is a
        # normal way to start -- searching what the lab did before does not need
        # a file in hand -- and the agent reports the run as unloaded until it
        # is pointed at one.
        if args.declarations and not args.raw:
            parser.error(
                "--declarations names what a run is, so it needs the run: pass "
                "--raw as well. A raw file on its own is fine -- it opens "
                "undeclared, and what is undeclared is blocked rather than guessed."
            )
        return run_gui(args) if args.stage == "gui" else run_agent(args)

    if not args.config:
        parser.error(f"{args.stage} requires --config")
    config = load_config(args.config)

    if args.stage in {"part1", "run"}:
        state, csv_path = run_part1(config, channel=args.channel)
        _report(state, f"part 1 complete -> {csv_path}")
        if args.verify and _verify(state, csv_path, config) != 0:
            return 3
        csv_for_part2 = csv_path
    else:
        if not args.csv:
            parser.error("part2 requires --csv")
        csv_for_part2 = Path(args.csv)

    if args.stage in {"part2", "run"}:
        state = run_part2(csv_for_part2, config, fit_order=args.fit_order)
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
