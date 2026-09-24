"""The window: one agent session, with the run visible beside it.

This is the terminal session with a second dimension spent on three things it
could not do.

**The figures are in the room.** They were always drawn -- every processed run
writes them, and ``plot_figure`` writes more -- and they were always in a folder
somebody had to go and open. A decision about a smoothing window taken while
looking at the number, with the picture of that window unopened two directories
away, is the decision this whole pipeline exists to stop being taken by eye.
Now it is beside the conversation that produced it.

**The run is always on screen.** ``/state`` was a thing you remembered to type.
The panel is rendered from the same :func:`~polyrmc.agent.toolspec.render_state`
the model gets from ``get_state``, refreshed after every tool, so what the
researcher is looking at and what the agent is reasoning over are one text.

**The commit gate is a real gate.** A terminal prompt with three sensible
answers sits in the same stream as everything else and is answered by muscle
memory. A dialog that has to be dismissed cannot be typed past.

Nothing here weakens what was already true. Commits go through
:class:`~polyrmc.agent.orchestrator.ApprovalPolicy`, refusals arrive whole, the
loop is capped, and every mechanic -- arguments, whole tool output, token
counts, the provider's own words -- goes to the same session log under
``data/logs`` that the terminal writes. The window is a second reader of one
account, not a second account.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from types import SimpleNamespace
from typing import Any

from polyrmc import paths
from polyrmc.gui import theme
from polyrmc.gui.session import Bridge, Signal
from polyrmc.gui.widgets import (
    ApprovalDialog,
    FigurePanel,
    OpenRunDialog,
    SearchPanel,
    SignInDialog,
    StatePanel,
    ToolsPanel,
    Transcript,
    open_externally,
)

POLL_MS = 60
"""How often the window looks for work from the turn thread.

Fast enough that a line appears as it happens rather than in batches, slow
enough to be free. The poll is also what keeps the transcript live behind a
modal dialog, which is why it reschedules itself before doing anything.
"""

OPENING = (
    "Point me at a run, ask about one we processed before, or ask a question "
    "about the measurement. Commits stop and ask you first."
)


class Workbench(tk.Tk):
    """One session, one window."""

    def __init__(self, args) -> None:
        super().__init__()
        from polyrmc.agent.budget import Budget, rates_from_env
        from polyrmc.agent.orchestrator import ApprovalPolicy, allow_all, deny_all
        from polyrmc.agent.toolspec import build_registry
        from polyrmc.agent.transcript import Narrator, SessionLog
        from polyrmc.cli import open_session

        self.args = args
        self.fonts = theme.apply(self)
        self.title("PolyRMC")
        self.geometry("1280x820")
        self.minsize(940, 620)

        self.session = open_session(args)
        self.registry = build_registry()
        self.log = SessionLog()
        self.narrator = Narrator(self.log)
        self.policy = ApprovalPolicy(ask=allow_all if args.auto_approve else deny_all)
        self.budget = Budget(
            max_requests=args.budget_requests,
            max_tokens=args.budget_tokens,
            rates=rates_from_env(),
        )
        self.provider: Any = None
        self.agent: Any = None
        self.bridge = Bridge(agent=None, surface=self.session.surface)
        self.verbose = tk.BooleanVar(value=bool(getattr(args, "verbose", False)))
        self.dead = False

        self._build_menu()
        self._build_layout()
        self._log_warnings()

        self.protocol("WM_DELETE_WINDOW", self.close)
        # Escape is the reflex when something is running away with itself, and
        # it costs nothing when nothing is: `stop` returns immediately if the
        # bridge is idle.
        self.bind("<Escape>", lambda _e: self.stop())
        self._pump_id = self.after(POLL_MS, self._pump)
        self.after(60, self._start_session)

    # --- construction -----------------------------------------------------

    def _build_menu(self) -> None:
        bar = tk.Menu(self)
        session = tk.Menu(bar, tearoff=0)
        session.add_command(label="Open a run...", command=self.open_run)
        session.add_command(label="Provider and key...", command=self.sign_in)
        session.add_command(label="File permissions...", command=self.show_file_permissions)
        session.add_separator()
        session.add_command(label="Copy the transcript", command=self.copy_transcript)
        session.add_command(label="Open the session log", command=self.open_log)
        session.add_separator()
        session.add_command(label="Close", command=self.close)
        bar.add_cascade(label="Session", menu=session)

        view = tk.Menu(bar, tearoff=0)
        view.add_checkbutton(
            label="Show full tool output", variable=self.verbose,
            onvalue=True, offvalue=False,
        )
        view.add_command(label="Rescan figures", command=lambda: self.figures.refresh())
        view.add_command(label="What was run this session", command=self.show_history)
        bar.add_cascade(label="View", menu=view)

        help_menu = tk.Menu(bar, tearoff=0)
        help_menu.add_command(label="Commands", command=self.show_help)
        help_menu.add_command(label="What this session costs", command=self.show_cost)
        help_menu.add_command(label="Providers on this machine", command=self.show_providers)
        bar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=bar)

    def show_file_permissions(self) -> None:
        """Show the effective general-file policy without invoking a model."""
        import json
        from textwrap import fill

        from polyrmc.tools import Rejected

        try:
            policy = json.loads(self.session.context.files.permissions())
            details = (
                "Read: " + ", ".join(policy["read_roots"]) + "\n\n"
                "Write: " + ", ".join(policy["write_roots"]) + "\n\n"
                "Excluded folders:\n" + fill(", ".join(policy["protected_folders"]), 85) + "\n\n"
                "Additional exclusions: " + (", ".join(policy["denied_paths"]) or "None") + "\n\n"
                "System and credential files (.env, Windows hives), and this policy are protected.\n\n"
                + fill(policy["limits"], 85) + "\n\n"
                "Existing-file replacements require a current hash and save a backup.\n"
                "Read content is sent to your model provider and recorded in the session log.\n\n"
                "Local settings: " + policy["policy_file"]
            )
        except Rejected as error:
            details = str(error)
        messagebox.showinfo("Agent file permissions", details, parent=self)

    def _build_layout(self) -> None:
        p = theme.PALETTE

        header = ttk.Frame(self, padding=(16, 10, 16, 8))
        header.pack(fill="x")
        self.run_label = ttk.Label(header, text="", style="Heading.TLabel")
        self.run_label.pack(side="left")
        self.model_label = ttk.Label(header, text="", style="Quiet.TLabel")
        self.model_label.pack(side="right")
        ttk.Separator(self, orient="horizontal").pack(fill="x")

        split = ttk.PanedWindow(self, orient="horizontal")
        split.pack(fill="both", expand=True)

        self.tabs = ttk.Notebook(split)
        self.state_panel = StatePanel(self.tabs, self.fonts)
        self.tools_panel = ToolsPanel(
            self.tabs, self.fonts, self.registry, self.always_allow
        )
        self.runs_panel = SearchPanel(
            self.tabs, self.fonts,
            "sample, buffer or run id -- read straight from the provenance sidecars",
            lambda q: self._call_tool("search_past_runs", {"query": q or None}),
        )
        self.notes_panel = SearchPanel(
            self.tabs, self.fonts,
            "what earlier sessions concluded, kept apart from run provenance",
            lambda q: self._call_tool("read_notes", {"query": q or None}),
        )
        self.figures = FigurePanel(
            self.tabs, self.fonts,
            [self.session.runs_dir / "figures", paths.project_root() / "data" / "figures"],
        )
        for panel, name in (
            (self.state_panel, "state"),
            (self.figures, "figures"),
            (self.runs_panel, "past runs"),
            (self.notes_panel, "notes"),
            (self.tools_panel, "tools"),
        ):
            self.tabs.add(panel, text=name)
        split.add(self.tabs, weight=2)

        right = ttk.Frame(split, style="Paper.TFrame")
        self.transcript = Transcript(right, self.fonts)
        self.transcript.pack(fill="both", expand=True)

        composer = tk.Frame(right, background=p.panel, padx=14, pady=10)
        composer.pack(fill="x")
        box = tk.Frame(composer, background=p.paper, highlightthickness=1,
                       highlightbackground=p.rule)
        box.pack(fill="x")
        self.entry = tk.Text(
            box, height=3, wrap="word", background=p.paper, foreground=p.ink,
            font=self.fonts.body, borderwidth=0, highlightthickness=0,
            padx=10, pady=8, insertbackground=p.ink,
        )
        self.entry.pack(side="left", fill="both", expand=True)
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", lambda _e: None)
        self.ask_button = ttk.Button(
            composer, text="Ask", style="Accent.TButton", command=self.submit
        )
        self.ask_button.pack(side="right", pady=(8, 0))
        ttk.Label(
            composer, text="Enter sends  ·  Shift+Enter for a new line  ·  / for commands",
            style="Faint.TLabel",
        ).pack(side="left", pady=(8, 0))
        split.add(right, weight=5)
        # Placed on the first real <Configure> rather than from here or from a
        # timer. A ttk paned window has no geometry until it is mapped, and a
        # sash set before that is silently dropped -- which left the panels at
        # one pixel wide and the window looking like a chat box.
        self._split = split
        self._sash_placed = False
        split.bind("<Configure>", self._place_sash)

        status = ttk.Frame(self, padding=(14, 6))
        status.pack(fill="x", side="bottom")
        ttk.Separator(self, orient="horizontal").pack(fill="x", side="bottom")
        self.spend_label = ttk.Label(status, text="", style="Faint.TLabel")
        self.spend_label.pack(side="left")
        self.log_label = ttk.Label(status, text="", style="Faint.TLabel", cursor="hand2")
        self.log_label.pack(side="right")
        self.log_label.bind("<Button-1>", lambda _e: self.open_log())
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=110)
        self.entry.focus_set()

    SASH = 460
    """Where the panels end and the conversation begins, in pixels.

    Wide enough for the state panel's longest line -- a blocked reason, which is
    a sentence -- without wrapping it, since a wrapped reason is the one a
    researcher skims.
    """

    def _place_sash(self, _event=None) -> None:
        """Split the window once, the first time it has a width to split."""
        # The threshold is the sash plus a usable conversation, not twice the
        # sash: the window's own minimum width is 940, and a test of 2*460 would
        # sit close enough to it that a slightly narrow window never got split
        # at all.
        if self._sash_placed or self._split.winfo_width() < self.SASH + 220:
            return
        self._sash_placed = True
        try:
            self._split.sashpos(0, self.SASH)
        except tk.TclError:
            pass

    def _log_warnings(self) -> None:
        """Provider SDK warnings belong in the log, not on top of the run.

        One of them announces that the model ignores ``temperature``, once per
        request. In a terminal those lines interleave with the conversation; in a
        window they would go to a console nobody is looking at, which is worse --
        they would be lost rather than merely misplaced.
        """
        import warnings

        def show(message, category, filename, lineno, file=None, line=None):
            self.log.write(f"warning {category.__name__}", f"{message}\n  from {filename}:{lineno}")

        warnings.showwarning = show

    # --- starting ---------------------------------------------------------

    def _start_session(self) -> None:
        self._refresh_header()
        self.state_panel.show(self._render_state())
        self.transcript.note(OPENING)
        self._connect(announce=True)
        self.log.banner(
            run=self.session.run_id,
            model=f"{self.provider.name} / {self.provider.model}" if self.provider else "none configured",
            tools=len(self.registry),
            runs_dir=str(self.session.runs_dir),
            surface="gui",
        )
        indexed = self._past_run_count()
        self.transcript.note(
            f"memory: {indexed} past run(s) in {paths.relative_to_root(self.session.runs_dir)}, "
            f"{len(self.session.notes)} note(s)"
        )
        self._refresh_status()

    def _connect(self, announce: bool = False) -> None:
        """Attach a brain to the current run, if one is keyed.

        The window opens either way. A session with no provider is not a broken
        session: the panels read past runs, notes and figures with no model at
        all, and every deterministic stage is a ``polyrmc process`` away -- so a
        missing key disables the one thing that needs it and says so, rather
        than refusing to open.

        The bridge is rebuilt unconditionally, brain or no brain, because it is
        what carries the approval gate. It is also where ``policy.ask`` is
        pointed: the orchestrator's default refuses every commit outright on the
        grounds that nobody is at the terminal, which is the right default for a
        script and the wrong one for a window with a person sitting at it.
        """
        from polyrmc.agent.orchestrator import Orchestrator
        from polyrmc.agent.toolcall import ProviderUnavailable, build_provider, keyed_providers

        self.agent = None
        self.provider = None
        kind = self.args.provider or (keyed_providers() or [None])[0]

        if kind is None:
            if announce:
                self.transcript.note(
                    "No provider is keyed on this machine, so there is no brain to ask. "
                    "The panels and every deterministic stage still work. "
                    "Session > Provider and key... to sign in."
                )
        else:
            try:
                self.provider = build_provider(
                    kind,
                    self.args.model,
                    min_interval=float(getattr(self.args, "min_request_interval", 0.0) or 0.0),
                    on_wait=self._announce_wait,
                )
            except ProviderUnavailable as error:
                self.transcript.block("cannot use that provider", str(error), "declined")

        if self.provider is not None:
            self.agent = Orchestrator(
                context=self.session.context,
                provider=self.provider,
                approvals=self.policy,
                registry=self.registry,
                max_steps=self.args.max_steps,
                budget=self.budget,
            )
            if announce:
                self.transcript.note(
                    f"lab context: {self.agent.lab_path or 'NONE FOUND -- running without house conventions'}"
                )

        # The outgoing bridge is closed, not dropped. If it were holding a
        # commit nobody had answered, dropping it would leave a daemon thread
        # blocked on an event no one is left to set.
        self.bridge.close()
        self.bridge = Bridge(agent=self.agent, surface=self.session.surface)
        if self.agent is not None:
            self.agent.should_stop = self.bridge.should_stop
        if not self.args.auto_approve:
            self.policy.ask = self.bridge.ask
        self._refresh_header()

    def _announce_wait(self, seconds: float, reason: str) -> None:
        """Called on the turn thread when a provider asks to be left alone.

        Routed through the queue rather than drawn from here. Tk is not
        thread-safe, and a rate limit is precisely the moment when touching a
        widget from the wrong thread would take the window down with it.
        """
        self.bridge.signals.put(
            Signal("waiting", (f"waiting {seconds:.0f}s", reason, seconds))
        )

    # --- the pump ---------------------------------------------------------

    def _pump(self) -> None:
        # Rescheduled first, before anything that can open a nested event loop.
        # The approval dialog runs one; if the next poll were scheduled after it,
        # the transcript behind the dialog would freeze until it was answered --
        # including the very reasoning that explains what is being asked.
        self._pump_id = self.after(POLL_MS, self._pump)
        for signal in self.bridge.drain():
            try:
                self._handle(signal)
            except tk.TclError:  # the window went away mid-turn
                return

    def _handle(self, signal: Signal) -> None:
        kind = signal.kind
        if kind == "event":
            self._show_event(signal.payload)
        elif kind == "state":
            self.state_panel.show(signal.payload)
        elif kind == "figures":
            self.figures.show_newest()
            self.tabs.select(self.figures)
        elif kind == "ask":
            self._put_approval(signal.payload)
        elif kind == "waiting":
            headline, reason, seconds = signal.payload
            self.log.write("rate limit", f"waiting {seconds:.1f}s -- {reason}")
            self.transcript.action(headline, reason)
        elif kind == "crash":
            self.transcript.banner(
                "the session hit a bug",
                signal.payload,
                "Nothing committed is lost. The state panel shows where the run "
                "stands, and `polyrmc process` runs the whole deterministic path "
                "with no model at all.",
                self.log.location,
            )
            self.log.write("session error", signal.payload)
        elif kind == "done":
            self._finish(signal.payload)

    def _show_event(self, event) -> None:
        from polyrmc.agent.transcript import activity, first_sentence, hint

        # Logged through the Narrator, and only logged: the strings it hands back
        # are shaped for a console. One writer means the file a colleague opens
        # is the same file whichever surface produced it.
        self.narrator.lines(event)

        kind = event.kind
        if kind == "text":
            self.transcript.model(event.text)
        elif kind == "call":
            arguments = dict(event.call.arguments) if event.call else {}
            self.transcript.action(activity(event.tool), hint(arguments))
        elif kind == "result":
            if self.verbose.get():
                self.transcript.output(event.text)
        elif kind == "refused":
            self.transcript.block("refused", event.text, "refused")
        elif kind == "denied":
            self.transcript.block("you declined this", event.text, "declined")
        elif kind == "error":
            self.transcript.block("failed", first_sentence(event.text), "failed")
        elif kind == "failed":
            failure = event.detail
            self.transcript.banner(
                failure.headline if failure else "the model call failed",
                (failure.meaning if failure else str(event.text))
                + f" {self.budget.render('Spent before it stopped', nudge=False)}.",
                failure.remedy if failure else "",
                self.log.location,
            )
        elif kind == "stopped":
            self.transcript.block("you stopped this", event.text, "declined")
        elif kind == "capped":
            self.transcript.banner(
                "stopped without answering",
                f"{event.text} {self.budget.render('Spent', nudge=False)}.",
                "Raise the ceiling and ask again, or run `polyrmc process` for the "
                "whole deterministic path with no model at all.",
                self.log.location,
            )
        elif kind == "spend":
            if self.verbose.get():
                self.transcript.note(event.text)
            self._refresh_status()

    def _put_approval(self, request) -> None:
        from polyrmc.agent.transcript import activity

        answer = ApprovalDialog.put(
            self, self.fonts, request.tool, request.call, activity(request.tool.name)
        )
        if answer.allowed and answer.remember:
            self.policy.always.add(request.tool.name)
        self.log.write(
            f"approval {request.tool.name}",
            "allowed" if answer.allowed else f"declined: {answer.reason}",
        )
        request.reply(answer)

    def _finish(self, outcome: str) -> None:
        self._busy(False)
        self._refresh_status()
        self.state_panel.show(self._render_state())
        self.figures.refresh()
        if outcome == "dead":
            # The brain is gone. There is no failover onto another of the
            # researcher's keys on purpose: that is their money, and moving a
            # session onto a second key is a spending decision that is theirs.
            self.dead = True
            self.ask_button.configure(state="disabled")
            self.transcript.note(
                "The session is over -- nothing committed is lost. Reopen and the "
                "state panel shows where this run stands."
            )
        self.transcript.spacer()

    # --- asking -----------------------------------------------------------

    def _on_return(self, _event):
        self.submit()
        return "break"  # Enter sends; the newline it would have inserted does not

    def submit(self) -> None:
        line = self.entry.get("1.0", "end-1c").strip()
        if not line:
            return
        if line.startswith("/"):
            self.entry.delete("1.0", "end")
            self._slash(line)
            return
        if self.dead:
            # The button is disabled, but Enter in the box is not the button.
            # Re-asking here would spend another request on the key that just
            # refused one.
            self.transcript.note(
                "That provider stopped answering, so there is nothing left to ask. "
                "Session > Provider and key... to carry on with another one; the "
                "panels and every slash command still work as they are."
            )
            return
        if self.agent is None:
            self.transcript.note(
                "No model is configured, so there is nothing to ask. "
                "Session > Provider and key... to sign in."
            )
            return
        if self.bridge.busy:
            self.transcript.note("still working on the last question.")
            return
        self.entry.delete("1.0", "end")
        self.transcript.you(line)
        self.narrator.user(line)
        self._busy(True)
        if not self.bridge.submit(line):
            self._busy(False)

    def _busy(self, working: bool) -> None:
        """One button, two jobs.

        A separate greyed-out Stop beside a greyed-out Ask is two dead controls
        for the whole time nothing is running, and the live one is whichever is
        not currently useful. The button is always the thing you can do now.
        """
        if working:
            self.ask_button.configure(
                text="Stop", style="Commit.TButton", command=self.stop, state="normal"
            )
            self.entry.configure(state="disabled")
            self.progress.pack(side="right", padx=(0, 14))
            self.progress.start(14)
        else:
            self.ask_button.configure(
                text="Ask", style="Accent.TButton", command=self.submit,
                state="disabled" if self.dead else "normal",
            )
            self.entry.configure(state="normal")
            self.progress.stop()
            self.progress.pack_forget()
            if not self.dead:
                self.entry.focus_set()

    def stop(self) -> None:
        """Ask the running turn to end, and say honestly that it has not yet.

        The label goes to "stopping", not "stopped". A model call is a blocking
        request inside a vendor SDK and a tool is numpy over eighty thousand
        points; neither can be cut off without leaving the run in a state nobody
        can describe. So the stop lands at the next boundary, which is usually a
        second or two and occasionally the length of one model call -- and a
        button that claimed otherwise would have people pulling the plug when
        the window "ignored" them.
        """
        if not self.bridge.busy:
            return
        self.bridge.request_stop()
        self.ask_button.configure(text="stopping...", state="disabled")
        self.transcript.note(
            "stopping -- waiting for the model call or tool already running to "
            "return. Nothing further will be started."
        )
        self.log.write("session", "stop requested by the researcher")

    # --- slash commands ---------------------------------------------------

    HELP = """\
/state             show where the run stands
/tools             every tool the agent can call
/history           what the agent has run this session
/runs [query]      search past experiments, no model call
/notes [query]     read notes filed in earlier sessions
/always <tool>     pre-approve one commit tool for this session
/auto              approve every commit for the rest of the session
/cost              requests and tokens spent on your key this session
/providers         which model providers are keyed on this machine
/log [n]           the session log: where it is, and its last n lines
/open              point this session at a run
/help              this list"""

    def _slash(self, line: str) -> None:
        command, _, rest = line[1:].partition(" ")
        rest = rest.strip()
        if command == "help":
            self.show_help()
        elif command == "state":
            self.tabs.select(self.state_panel)
            self.state_panel.show(self._render_state())
        elif command == "tools":
            self.tabs.select(self.tools_panel)
        elif command == "history":
            self.show_history()
        elif command == "runs":
            self.tabs.select(self.runs_panel)
            self.runs_panel.query.delete(0, "end")
            self.runs_panel.query.insert(0, rest)
            self.runs_panel.search()
        elif command == "notes":
            self.tabs.select(self.notes_panel)
            self.notes_panel.query.delete(0, "end")
            self.notes_panel.query.insert(0, rest)
            self.notes_panel.search()
        elif command == "always":
            self.always_allow(rest)
        elif command == "auto":
            from polyrmc.agent.orchestrator import allow_all

            self.policy.ask = allow_all
            self.transcript.note("every commit will now run without asking.")
        elif command == "cost":
            self.show_cost()
        elif command == "providers":
            self.show_providers()
        elif command == "log":
            self.show_log_tail(rest)
        elif command == "open":
            self.open_run()
        elif command in {"exit", "quit", "q"}:
            self.close()
        else:
            self.transcript.note(f"unknown command {command!r}. /help for the list.")

    def always_allow(self, name: str) -> None:
        tool = self.registry.get(name)
        if tool is None or not tool.mutating:
            self.transcript.note(
                f"{name!r} is not a commit tool. The tools panel lists them."
            )
            return
        self.policy.always.add(name)
        self.transcript.note(f"{name} approved for the rest of this session.")

    # --- menu actions -----------------------------------------------------

    def show_help(self) -> None:
        self.transcript.note(self.HELP)

    def show_history(self) -> None:
        self.transcript.note(
            self.agent.narrate() if self.agent else "(no model session)"
        )

    def show_cost(self) -> None:
        self.transcript.note(self.budget.render())

    def show_providers(self) -> None:
        from polyrmc.agent.toolcall import describe_providers

        text = describe_providers()
        if self.provider is not None:
            text += f"\n\nthis session: {self.provider.name} / {self.provider.model}"
        self.transcript.note(text)

    def show_log_tail(self, argument: str = "") -> None:
        from polyrmc.cli import _tail_log

        self.transcript.note(_tail_log(self.log, argument))

    def open_log(self) -> None:
        if self.log.path is not None and self.log.path.exists():
            open_externally(self.log.path)
        else:
            self.transcript.note(f"session log: {self.log.location}")

    def copy_transcript(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.transcript.plain())
        self.transcript.note("transcript copied to the clipboard.")

    def sign_in(self) -> None:
        from polyrmc.agent.toolcall import SPECS, keyed_providers
        from polyrmc.cli import _persist_key

        chosen = SignInDialog.put(self, self.fonts, SPECS, keyed_providers())
        if chosen is None:
            return
        kind, value, save = chosen
        spec = SPECS[kind]
        import os

        os.environ[spec.canonical_key] = value
        if save and _persist_key(spec, value):
            self.transcript.note(
                f"key written to {paths.env_file()} as {spec.canonical_key}. It is gitignored."
            )
        self.args.provider = kind
        self._connect(announce=True)
        if self.provider is not None:
            # Signing in is how a session recovers from a spent key, so it has
            # to lift the stop that the spent key put there.
            self.dead = False
            self.ask_button.configure(state="normal")
            self.transcript.note(f"signed in: {self.provider.name} / {self.provider.model}")
        self._refresh_status()

    def open_run(self) -> None:
        from polyrmc.agent.orchestrator import ApprovalPolicy, allow_all, deny_all
        from polyrmc.cli import open_session
        from polyrmc.intake import MissingDeclarations

        if self.bridge.busy:
            self.transcript.note("a question is still running; wait for it to finish.")
            return
        chosen = OpenRunDialog.put(self, self.fonts, str(self.args.author or "researcher"))
        if not chosen:
            return

        nominated = SimpleNamespace(**vars(self.args))
        nominated.raw = chosen["raw"]
        nominated.declarations = chosen["declarations"]
        nominated.author = chosen["author"]
        nominated.run_id = None
        try:
            self.session = open_session(nominated)
        except (MissingDeclarations, OSError, ValueError) as error:
            messagebox.showerror("Cannot open that run", str(error), parent=self)
            return

        self.args = nominated
        # Standing approvals do not survive a change of subject. "Always allow
        # emit_csv" is a coherent decision about the run in front of you and an
        # incoherent one about the next run, which is a different measurement
        # with different declarations.
        self.policy = ApprovalPolicy(ask=allow_all if self.args.auto_approve else deny_all)
        self.transcript.clear()
        self.transcript.note(f"opened {chosen['raw']}")
        self._connect()
        self.figures.directories = [
            self.session.runs_dir / "figures",
            paths.project_root() / "data" / "figures",
        ]
        self.figures.refresh()
        self.dead = False
        self.ask_button.configure(state="normal")
        self.state_panel.show(self._render_state())
        self._refresh_header()
        self.log.write("session", f"opened run {self.session.run_id} from {chosen['raw']}")

    # --- odds and ends ----------------------------------------------------

    def _call_tool(self, name: str, arguments: dict) -> str:
        """Run one read tool straight, with no model in the way."""
        from polyrmc.tools import Rejected

        try:
            return self.registry[name].handler(self.session.context, arguments)
        except Rejected as error:
            return str(error)
        except Exception as error:  # noqa: BLE001 - a panel is not worth a dead window
            return f"{type(error).__name__}: {error}"

    def _render_state(self) -> str:
        from polyrmc.agent.toolspec import render_state

        try:
            return render_state(self.session.surface)
        except Exception as error:  # noqa: BLE001
            return f"(state could not be rendered: {error})"

    def _past_run_count(self) -> int:
        from polyrmc.memory.registry import RunRegistry

        try:
            return len(RunRegistry(self.session.runs_dir).summaries)
        except Exception:  # noqa: BLE001
            return 0

    def _refresh_header(self) -> None:
        source = Path(self.session.surface.state.source_file or "none").name
        self.run_label.configure(text=f"{self.session.run_id}   ·   {source}")
        if self.provider is not None:
            brain = f"{self.provider.name} / {self.provider.model}"
        else:
            brain = "no model keyed"
        approval = "auto-approve" if self.args.auto_approve else "commits need you"
        self.model_label.configure(
            text=f"{brain}   ·   {len(self.registry)} tools   ·   {approval}"
        )
        self.title(f"PolyRMC — {self.session.run_id}")

    def _refresh_status(self) -> None:
        self.spend_label.configure(text=self.budget.render("spent"))
        self.log_label.configure(text=f"working notes: {self.log.location}")

    def close(self) -> None:
        """Let go of the turn thread before the window goes.

        The thread is a daemon, so a model call in flight does not hold the
        process open. A commit waiting on approval is the case that needs
        handling, and :meth:`Bridge.close` refuses it -- an unanswered prompt has
        to end as a refusal, never as an approval nobody gave.
        """
        self.bridge.close()
        try:
            # Otherwise the next poll fires against a window that no longer
            # exists, and the last thing a clean exit prints is a Tk traceback.
            self.after_cancel(self._pump_id)
        except (tk.TclError, AttributeError):
            pass
        self.log.write("session", self.budget.render("spent this session"))
        self.log.close()
        self.destroy()


def launch(args) -> int:
    """Open the window and run until it closes."""
    from polyrmc.intake import MissingDeclarations

    try:
        window = Workbench(args)
    except MissingDeclarations as error:
        print(f"cannot start: {error}")
        return 2
    window.mainloop()
    return 0


__all__ = ["OPENING", "POLL_MS", "Workbench", "launch"]
