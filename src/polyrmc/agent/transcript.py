"""What a researcher reads, and where everything else goes.

The session used to print its own mechanics. Every tool call arrived as
``pick_window(window_length=31)`` -- Python call syntax, which is the shape of
the API request and not the shape of a thing that happened -- followed by the
tool's entire output, followed by a token-accounting line after every turn. A
question that took six rounds produced four screens, of which perhaps five lines
concerned the measurement. The information was all true and almost none of it
was for a person.

So this module splits one stream into two.

**The terminal gets the account of what happened.** The model's own prose, and
one plain line per action: *sweeping the smoothing windows*, *choosing the
smoothing window*. Nothing in call syntax, no argument dumps, no token counts.
Refusals and declined commits stay on screen in full, because those are
decisions about the run rather than background work.

**The log gets everything.** Arguments as JSON, complete tool output, per-turn
usage, warnings raised inside provider SDKs, and the provider's raw error text.
It is a file, written as the session runs, so a session that dies still has it.

The split is presentation only. :class:`~polyrmc.agent.orchestrator.Event` still
carries the full text of everything; this module decides what a person is shown.
Anything hidden here is one ``/log`` away, and the log is named on the way in
and on the way out so nobody has to know that in advance.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from polyrmc import paths

# ASCII only, everywhere in this module. A Windows console is cp1252 by default
# and a session that dies encoding a bullet after the model has committed its
# work loses the record of what it did.
BULLET = "  . "
RULE = "-" * 66

# What each tool is doing, in the words a researcher would use. Keyed by tool
# name because the tool's own description is written for the model -- it states
# preconditions and refusal behaviour, which is the right thing to tell a caller
# deciding whether to call and the wrong thing to tell a reader watching it run.
# A tool missing from here degrades to its own name with the underscores knocked
# out, which is plain enough to be honest and plain enough to notice.
ACTIVITY: dict[str, str] = {
    "apply_baseline_correction": "applying the baseline correction",
    "check_declarations": "checking the declarations against the trace",
    "compare_to_past_runs": "comparing this run against past ones",
    "describe_plot_file": "reading a KaleidaGraph plot file",
    "emit_csv": "writing the boundary CSV",
    "extract_parameters": "fitting the virial expansion",
    "fit_drift": "fitting the baseline drift",
    "get_history": "reading what has been run so far",
    "get_intake_form": "checking which declarations are outstanding",
    "get_past_run": "looking up a past run",
    "get_shortlist": "reading the safe band of windows",
    "get_state": "checking where the run stands",
    "list_anchor_candidates": "listing the baseline anchor candidates",
    "list_anomalies": "listing the flagged regions of the trace",
    "list_blocked": "listing what cannot be computed, and why",
    "list_conflicts": "listing declarations the trace disputes",
    "load_run": "loading and cleaning the raw file",
    "open_data_file": "opening a data file",
    "pick_window": "choosing the smoothing window",
    "plot_columns": "plotting columns to a PNG",
    "plot_figure": "drawing a figure",
    "read_notes": "reading notes from earlier sessions",
    "resolve_conflict": "recording a conflict as understood",
    "revert_baseline_correction": "reverting the baseline correction",
    "search_figures": "searching labelled figures",
    "search_past_runs": "searching past runs",
    "search_internet": "searching internet sources",
    "read_web_page": "reading a public source",
    "review_literature": "finding literature related to this experiment",
    "select_anchors": "selecting baseline anchors",
    "set_anomaly_decision": "deciding what happens to a flagged region",
    "set_config": "changing a configuration value",
    "set_declaration": "recording a declaration",
    "sweep_windows": "sweeping the smoothing windows",
    "verify_output": "verifying the emitted CSV",
    "write_note": "filing a note for later sessions",
}

# Argument hint length. Long enough for a window length, a column name or a
# short path; short enough that it never becomes the argument dump this module
# exists to remove.
HINT_LIMIT = 56


def activity(name: str) -> str:
    """The human phrase for one tool."""
    return ACTIVITY.get(name, name.replace("_", " "))


def hint(arguments: dict[str, Any]) -> str:
    """A short phrase naming what one call is about, or "" if nothing fits.

    Values, not the call. ``window length 31`` tells a reader which window was
    chosen; ``pick_window(window_length=31)`` tells them the same thing in the
    shape of a function call they did not write and cannot run.

    It returns the phrase bare, without the parentheses the terminal sets it in.
    Deciding what is worth saying and deciding how it sits in a line are
    different jobs, and the window sets the same phrase in a different way --
    smaller and greyer, after the activity, where a console has only punctuation
    to work with.
    """
    if not arguments:
        return ""
    parts = []
    for key, value in arguments.items():
        if isinstance(value, (dict, list, tuple)):
            continue  # structure belongs in the log, not in a parenthetical
        text = str(value).strip()
        if not text or text.lower() in {"none", "null"}:
            continue
        if len(text) > HINT_LIMIT:
            text = text[: HINT_LIMIT - 3] + "..."
        parts.append(f"{key.replace('_', ' ')} {text}")
    if not parts:
        return ""
    joined = ", ".join(parts)
    if len(joined) > HINT_LIMIT:
        joined = joined[: HINT_LIMIT - 3] + "..."
    return joined


def first_sentence(text: str, limit: int = 200) -> str:
    """The opening line of a longer message, for a one-line report."""
    stripped = str(text).strip()
    line = stripped.splitlines()[0] if stripped else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def default_log_path(now: datetime | None = None) -> Path:
    """One file per session, named for when it started.

    Not one rolling ``agent.log``. Two terminals open at once is normal here --
    one processing a run, one asking about an old one -- and interleaving them
    into a single file makes the record of both unreadable.
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return paths.logs_dir() / f"session-{stamp}.log"


class SessionLog:
    """The background record of one session, written as it happens.

    Opened in append mode and flushed after every entry, because the sessions
    whose detail is most worth having are the ones that ended badly -- a
    provider that stopped answering, a console closed mid-run -- and a buffer
    held until a clean exit is empty in exactly those cases.

    A log that cannot be opened is not an error. The terminal view is complete
    on its own and the log is the extra; refusing to start a session because a
    data directory is read-only would trade the thing that matters for the thing
    that does not.
    """

    def __init__(self, path: Path | None = None, stream: TextIO | None = None) -> None:
        if stream is not None:
            self.path = Path(path) if path is not None else None
            self._stream: TextIO | None = stream
            self._owned = False
            return
        self.path = Path(path) if path is not None else default_log_path()
        self._owned = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8")
        except OSError:
            self.path = None
            self._stream = None

    # --- writing ---------------------------------------------------------

    def write(self, label: str, text: str = "") -> None:
        if self._stream is None:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{stamp}] {label}"
        body = str(text).rstrip()
        if body:
            indented = "\n".join("    " + line for line in body.splitlines())
            entry = f"{entry}\n{indented}"
        self._emit(entry + "\n")

    def banner(self, **facts: Any) -> None:
        """Open the log with what this session is."""
        self.write("session started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for key, value in facts.items():
            if value not in (None, ""):
                self.write(key.replace("_", " "), str(value))
        self._emit(RULE + "\n")

    def _emit(self, text: str) -> None:
        if self._stream is None:
            return
        try:
            self._stream.write(text)
            self._stream.flush()
        except (OSError, ValueError):  # closed console, full disk
            self._stream = None

    def close(self) -> None:
        if self._stream is not None and self._owned:
            try:
                self._stream.close()
            except (OSError, ValueError):
                pass
        self._stream = None

    @property
    def location(self) -> str:
        """Where the log is, shortened against the project root for reading."""
        if self.path is None:
            return "(none: the log directory could not be written)"
        return paths.relative_to_root(self.path)


class Narrator:
    """Turns events into the two streams: lines for a person, detail for a log.

    Holds no state about the run. It is asked about one event and answers with
    what to print, having already written the whole of it to the log.
    """

    def __init__(self, log: SessionLog, verbose: bool = False) -> None:
        self.log = log
        self.verbose = verbose

    def user(self, text: str) -> None:
        self.log.write("you", text)

    def note(self, text: str) -> None:
        """Something the CLI itself did, worth recording but not narrating."""
        self.log.write("session", text)

    def lines(self, event: Any) -> list[str]:
        """What to show for one event. Everything is logged either way."""
        kind = event.kind
        call = getattr(event, "call", None)
        arguments = dict(call.arguments) if call is not None else {}

        if kind == "call":
            self.log.write(
                f"call {event.tool}",
                json.dumps(arguments, default=str, indent=2, sort_keys=True)
                if arguments
                else "(no arguments)",
            )
            aside = hint(arguments)
            return [f"{BULLET}{activity(event.tool)}" + (f" ({aside})" if aside else "")]

        if kind == "result":
            self.log.write(f"result {event.tool}", event.text)
            return [_indent(event.text)] if self.verbose else []

        if kind == "text":
            self.log.write("model", event.text)
            return ["", _wrapped(event.text), ""]

        if kind == "refused":
            # A refusal is a decision about the run, and the tool surface writes
            # its rejections as recovery instructions a person can act on. This
            # is the one place the tool's own words are the right words.
            self.log.write(f"refused {event.tool}", event.text)
            return [_indent(f"refused: {event.text}", "      ")]

        if kind == "denied":
            self.log.write(f"declined {event.tool}", event.text)
            return [_indent(f"you declined this: {event.text}", "      ")]

        if kind == "error":
            self.log.write(f"error {event.tool}", event.text)
            # The full text is written for the model to correct itself with --
            # it lists every available tool when a name was wrong -- and reading
            # thirty tool names is not how a person finds out a step failed.
            return [_indent(f"failed: {first_sentence(event.text)}", "      ")]

        if kind in {"capped", "failed"}:
            raw = getattr(event.detail, "raw", "")
            # ``event.text`` already quotes the provider inside this package's
            # own explanation of it, so logging both verbatim writes the same
            # four hundred characters of JSON twice. The classification is the
            # summary; the raw response is the evidence; one of each.
            summary = event.detail.one_line() if raw else event.text
            self.log.write(kind, summary)
            if raw:
                # The entire reason the banner can afford to be four lines: the
                # JSON it replaces has to land somewhere, or "full detail:
                # <log>" is a promise the log does not keep.
                self.log.write("provider said", raw)
            return []  # the caller renders these as an error block

        if kind == "spend":
            self.log.write("spend", event.text)
            return [_indent(event.text)] if self.verbose else []

        self.log.write(kind, event.text)
        return [event.text] if event.text else []

    def show(self, event: Any) -> None:
        for line in self.lines(event):
            print(line)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def _wrapped(text: str, width: int = 88, prefix: str = "  ") -> str:
    """The model's prose, indented to sit under the conversation.

    Wrapped by paragraph rather than by line, because a reply arrives with its
    own line breaks that were chosen for no particular terminal width.
    """
    blocks = []
    for paragraph in str(text).strip().split("\n\n"):
        stripped = paragraph.strip()
        if not stripped:
            continue
        blocks.append(
            textwrap.fill(
                stripped,
                width=width,
                initial_indent=prefix,
                subsequent_indent=prefix,
                replace_whitespace=True,
            )
        )
    return "\n\n".join(blocks)


def error_block(title: str, body: str, remedy: str = "", detail: str = "") -> str:
    """A stop, rendered so it cannot be mistaken for part of the conversation.

    Running out of quota is not a message from the model and must not read like
    one. It gets a rule above and below, the reason in plain words, what to do
    about it, and a pointer to the provider's own text in the log -- which is
    where a four-hundred-character JSON error belongs.
    """
    lines = ["", RULE, f"  {title.upper()}", RULE, _wrapped(body)]
    if remedy:
        lines.extend(["", _wrapped(f"What to do: {remedy}")])
    if detail:
        lines.extend(["", f"  full detail: {detail}"])
    lines.append(RULE)
    return "\n".join(lines)


__all__ = [
    "ACTIVITY",
    "BULLET",
    "HINT_LIMIT",
    "RULE",
    "Narrator",
    "SessionLog",
    "activity",
    "default_log_path",
    "error_block",
    "first_sentence",
    "hint",
]
