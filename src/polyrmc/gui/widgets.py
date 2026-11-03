"""The parts of the window, and what each of them refuses to abbreviate.

Two of these are not display at all. :class:`ApprovalDialog` is the commit gate
-- the same gate the terminal opens with ``allow? [y]es / [n]o / [a]lways`` --
and :class:`SignInDialog` is the key prompt. Everything else here shows what
already happened.

The rule the terminal session established holds in every one of them: what is
hidden must be one gesture away, and two things are never hidden at all. A
refusal is printed whole, because the tool surface writes its rejections as
recovery instructions and the band it names is the useful part. An approval
prompt is printed whole, because a prompt that hides what is about to change
trains you to approve without looking.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable, Sequence

from polyrmc.gui.links import open_web_url, safe_web_url
from polyrmc.gui.theme import PALETTE, Fonts

# Indents, in pixels. The account of what happened sits under the prose it
# belongs to, the way it does in the terminal.
PROSE_INDENT = 4
ACTION_INDENT = 22
BLOCK_INDENT = 20

VALUE_COLUMN = 172
"""Where the state panel's values start, in pixels.

Wide enough for ``undeclared (required)``, which is the longest label
:func:`~polyrmc.agent.toolspec.render_state` emits and the one whose value a
researcher most needs to read.
"""


_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")

# Inline spans, in one pass so that a later pattern cannot re-enter what an
# earlier one already claimed. ``**`` is tried before ``*`` for the obvious
# reason; backticks are tried first because their content is verbatim and must
# not be scanned for anything else at all.
_INLINE = re.compile(
    r"`(?P<code>[^`]+)`"
    r"|\[(?P<link_label>[^\]\n]+)\]\((?P<link_url>(?:[^()\s<>]|\([^()\s<>]*\))+)\)"
    r"|\*\*(?=\S)(?P<strong>.+?)(?<=\S)\*\*"
    r"|(?<![\w*])\*(?=\S)(?P<em>[^*\n]+?)(?<=\S)\*(?![\w*])"
)


def _inline(
    text: str, block: str, carried: tuple[str, ...] = ()
) -> list[tuple[str, tuple[str, ...]]]:
    """Split one line into runs, each carrying the block tag plus its own.

    Recursive, because spans nest: a model writing a declaration key in a list
    of required ones produces ``**`c0_g_per_cm3`**``, and a single pass renders
    the bold and leaves the backticks sitting in the middle of it. The body of a
    match is strictly shorter than the match -- the delimiters are consumed --
    so the recursion terminates on its own.

    Code is the exception and is never re-scanned. Its content is verbatim by
    definition, and a path or an expression is entitled to contain an asterisk.
    """
    chunks: list[tuple[str, tuple[str, ...]]] = []
    cursor = 0
    for found in _INLINE.finditer(text):
        if found.start() > cursor:
            chunks.append((text[cursor:found.start()], (block, *carried)))
        if found.group("code") is not None:
            chunks.append((found.group("code"), (block, *carried, "md_code")))
        elif found.group("link_label") is not None:
            target = safe_web_url(found.group("link_url"))
            if target:
                chunks += _inline(found.group("link_label"), block, (*carried, "md_link", f"link:{target}"))
            else:
                chunks.append((found.group(0), (block, *carried)))
        else:
            name = "strong" if found.group("strong") is not None else "em"
            chunks += _inline(found.group(name), block, (*carried, f"md_{name}"))
        cursor = found.end()
    if cursor < len(text):
        chunks.append((text[cursor:], (block, *carried)))
    return chunks


def open_externally(path: Path) -> None:
    """Hand a file to whatever the desktop opens it with.

    A PNG in a canvas is scaled to a panel; a figure destined for a paper gets
    read at full size in the viewer the researcher already uses.
    """
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606 - the whole point
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass


def _scrolled_text(parent, **kwargs) -> tuple[tk.Text, ttk.Frame]:
    frame = ttk.Frame(parent, style="Paper.TFrame")
    text = tk.Text(frame, **kwargs)
    bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.grid(row=0, column=0, sticky="nsew")
    bar.grid(row=0, column=1, sticky="ns")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return text, frame


class Transcript(ttk.Frame):
    """The account of the session, as a person reads it.

    Same split as the terminal: the model's prose and one plain line per action,
    with the mechanics in the log. What the window adds is not more text but
    less uniform text -- a refusal, a declined commit and a stop are each shaped
    unlike the conversation they interrupt, so none of them can be skimmed past
    as another line of narration.
    """

    def __init__(self, parent, fonts: Fonts) -> None:
        super().__init__(parent, style="Paper.TFrame")
        p = PALETTE
        self.text, frame = _scrolled_text(
            self,
            wrap="word",
            background=p.paper,
            foreground=p.ink,
            font=fonts.body,
            borderwidth=0,
            highlightthickness=0,
            padx=20,
            pady=16,
            spacing1=1,
            spacing3=2,
            selectbackground=p.wash_accent,
            selectforeground=p.ink,
            inactiveselectbackground=p.wash_accent,
            cursor="arrow",
        )
        frame.pack(fill="both", expand=True)
        self.text.configure(state="disabled")
        self._configure_tags(fonts)

    def _configure_tags(self, fonts: Fonts) -> None:
        p = PALETTE
        t = self.text
        t.tag_configure(
            "you",
            foreground=p.accent,
            font=fonts.bold,
            lmargin1=PROSE_INDENT,
            lmargin2=PROSE_INDENT,
            spacing1=12,
            spacing3=6,
        )
        t.tag_configure(
            "model",
            foreground=p.ink,
            font=fonts.body,
            lmargin1=PROSE_INDENT,
            lmargin2=PROSE_INDENT,
            spacing1=4,
            spacing3=8,
        )
        t.tag_configure(
            "action",
            foreground=p.quiet,
            font=fonts.small,
            lmargin1=ACTION_INDENT,
            lmargin2=ACTION_INDENT + 14,
            spacing1=1,
        )
        t.tag_configure("bullet", foreground=p.faint)
        t.tag_configure("hint", foreground=p.faint, font=fonts.small)
        t.tag_configure(
            "note", foreground=p.faint, font=fonts.small,
            lmargin1=PROSE_INDENT, lmargin2=PROSE_INDENT, spacing1=6,
        )
        t.tag_configure(
            "output",
            foreground=p.quiet,
            font=fonts.mono_small,
            lmargin1=ACTION_INDENT + 14,
            lmargin2=ACTION_INDENT + 14,
            spacing3=4,
        )
        for name, ink, wash in (
            ("refused", p.refused, p.wash_refused),
            ("declined", p.commit, p.wash_commit),
            ("failed", p.refused, p.wash_refused),
        ):
            t.tag_configure(
                name,
                foreground=ink,
                background=wash,
                font=fonts.small,
                lmargin1=BLOCK_INDENT,
                lmargin2=BLOCK_INDENT,
                rmargin=BLOCK_INDENT,
                spacing1=8,
                spacing3=8,
            )
            self._margin_stripe(name, ink)
        t.tag_configure("label", font=fonts.bold)
        t.tag_configure(
            "banner_title", font=fonts.heading, foreground=p.refused,
            background=p.wash_refused,
            lmargin1=BLOCK_INDENT, lmargin2=BLOCK_INDENT, rmargin=BLOCK_INDENT,
            spacing1=12, spacing3=4,
        )
        self._margin_stripe("banner_title", p.refused)
        t.tag_configure(
            "banner_body", font=fonts.body, foreground=p.ink,
            background=p.wash_refused,
            lmargin1=BLOCK_INDENT, lmargin2=BLOCK_INDENT, rmargin=BLOCK_INDENT,
            spacing3=6,
        )
        self._margin_stripe("banner_body", p.refused)
        t.tag_configure(
            "banner_detail", font=fonts.small, foreground=p.faint,
            background=p.wash_refused,
            lmargin1=BLOCK_INDENT, lmargin2=BLOCK_INDENT, rmargin=BLOCK_INDENT,
            spacing3=12,
        )
        self._margin_stripe("banner_detail", p.refused)

        # Markdown. Created after the block tags on purpose: Tk resolves a
        # clash between two tags on one range in favour of the one made later,
        # and bold inside a list item has to beat the list item's own font.
        t.tag_configure(
            "md_body", font=fonts.body, foreground=p.ink,
            lmargin1=PROSE_INDENT, lmargin2=PROSE_INDENT, spacing3=3,
        )
        t.tag_configure(
            "md_head", font=fonts.heading, foreground=p.ink,
            lmargin1=PROSE_INDENT, lmargin2=PROSE_INDENT, spacing1=12, spacing3=3,
        )
        t.tag_configure(
            "md_item", font=fonts.body, foreground=p.ink,
            lmargin1=PROSE_INDENT + 16, lmargin2=PROSE_INDENT + 32, spacing3=2,
        )
        t.tag_configure(
            "md_item2", font=fonts.body, foreground=p.ink,
            lmargin1=PROSE_INDENT + 38, lmargin2=PROSE_INDENT + 54, spacing3=2,
        )
        t.tag_configure("md_marker", foreground=p.faint)
        t.tag_configure("md_strong", font=fonts.bold)
        t.tag_configure("md_em", font=(fonts.body[0], fonts.body[1], "italic"))
        t.tag_configure("md_code", font=fonts.mono_small, foreground=p.accent)
        t.tag_configure("md_link", foreground=p.accent, underline=True)

    def _margin_stripe(self, tag: str, colour: str) -> None:
        """Colour the left margin of a tag, where Tk is new enough to do it.

        This is what makes an interruption read as a block rather than as more
        conversation in another colour. ``-lmargincolor`` arrived in Tk 8.6.6;
        on anything older the wash alone still separates it, so a missing option
        is not worth refusing to open a window over.
        """
        try:
            self.text.tag_configure(tag, lmargincolor=colour)
        except tk.TclError:
            pass

    # --- writing ---------------------------------------------------------

    def _write(self, chunks: Sequence[tuple[str, str | tuple[str, ...]]]) -> None:
        at_end = self.text.yview()[1] > 0.999
        self.text.configure(state="normal")
        for body, tags in chunks:
            for tag in (tags,) if isinstance(tags, str) else tags:
                if tag.startswith("link:") and tag not in self.text.tag_names():
                    target = tag.removeprefix("link:")
                    self.text.tag_configure(tag)
                    self.text.tag_bind(tag, "<Button-1>", lambda _e, url=target: open_web_url(url))
                    self.text.tag_bind(tag, "<Enter>", lambda _e: self.text.configure(cursor="hand2"))
                    self.text.tag_bind(tag, "<Leave>", lambda _e: self.text.configure(cursor="arrow"))
            self.text.insert("end", body, tags)
        self.text.configure(state="disabled")
        # Only follow the conversation if the reader was already at the bottom.
        # Yanking someone back from a refusal they were re-reading, because a
        # later tool finished, loses them the thing they stopped to read.
        if at_end:
            self.text.see("end")

    def you(self, text: str) -> None:
        self._write([(text.strip() + "\n", "you")])

    def model(self, text: str) -> None:
        """The model's reply, with its markdown rendered rather than displayed.

        A model writes markdown whatever you ask of it, and a console has no way
        to honour that -- so the terminal session prints the asterisks and the
        reader learns to see past them. A window has no such excuse. ``###`` and
        ``**`` are formatting instructions, and showing the instruction instead
        of obeying it is the same mistake as printing call syntax at somebody.

        Deliberately partial. Headings, list items, ``**strong**``, ``*emphasis*``
        and ``code`` plus web links are rendered; tables and LaTeX are left exactly as
        the model wrote them. Half-rendered maths would be worse than none:
        ``$M_w$`` at least says a symbol was meant, where a mangled attempt at
        one says something untrue about the measurement.

        Underscores are never emphasis here, whatever CommonMark says. This lab
        writes ``c0_g_per_cm3`` and ``ls_units`` in ordinary prose, and a
        renderer that italicised the middle of a declaration key would corrupt
        the one kind of word that has to survive verbatim.
        """
        chunks: list[tuple[str, str | tuple[str, ...]]] = []
        for raw in str(text).strip().splitlines():
            stripped = raw.strip()
            if not stripped:
                chunks.append(("\n", "md_body"))
                continue
            if len(stripped) >= 3 and set(stripped) <= {"-", "*", "_"}:
                continue  # a horizontal rule; the spacing already breaks the section
            heading = _HEADING.match(raw)
            if heading:
                chunks += _inline(heading.group(2), "md_head")
                chunks.append(("\n", "md_head"))
                continue
            item = _LIST_ITEM.match(raw.rstrip())
            if item:
                tag = "md_item2" if len(item.group(1)) >= 2 else "md_item"
                marker = item.group(2)
                chunks.append((f"{marker if marker[0].isdigit() else '-'}  ", (tag, "md_marker")))
                chunks += _inline(item.group(3), tag)
                chunks.append(("\n", tag))
                continue
            chunks += _inline(stripped, "md_body")
            chunks.append(("\n", "md_body"))
        if chunks:
            self._write(chunks)

    def action(self, phrase: str, hint: str = "") -> None:
        chunks: list[tuple[str, str | tuple[str, ...]]] = [
            ("•  ", ("action", "bullet")),
            (phrase, "action"),
        ]
        if hint:
            chunks.append((f"  {hint}", ("action", "hint")))
        chunks.append(("\n", "action"))
        self._write(chunks)

    def output(self, text: str) -> None:
        """Full tool output, shown only when the researcher asked for it."""
        self._write([(text.rstrip() + "\n", "output")])

    def note(self, text: str) -> None:
        self._write([(text.strip() + "\n", "note")])

    def block(self, label: str, body: str, kind: str) -> None:
        """A refusal or a declined commit, whole and unabbreviated."""
        self._write(
            [
                (f"{label}\n", (kind, "label")),
                (body.strip() + "\n", kind),
            ]
        )

    def banner(self, title: str, body: str, remedy: str = "", detail: str = "") -> None:
        """A stop. Not a message from the model, and shaped so it cannot read as one."""
        chunks: list[tuple[str, str | tuple[str, ...]]] = [
            (f"{title.upper()}\n", "banner_title"),
            (body.strip() + "\n", "banner_body"),
        ]
        if remedy:
            chunks.append((f"What to do: {remedy.strip()}\n", "banner_body"))
        chunks.append((f"full detail: {detail}\n" if detail else "\n", "banner_detail"))
        self._write(chunks)

    def spacer(self) -> None:
        self._write([("\n", "note")])

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for tag in self.text.tag_names():
            if tag.startswith("link:"):
                self.text.tag_delete(tag)
        self.text.configure(cursor="arrow")
        self.text.configure(state="disabled")

    def plain(self) -> str:
        return self.text.get("1.0", "end-1c")

    def copy_text(self) -> str:
        """Keep source URLs when copying a conversation with rendered links."""
        links = []
        for tag in self.text.tag_names():
            if not tag.startswith("link:"):
                continue
            ranges = self.text.tag_ranges(tag)
            if ranges:
                label = self.text.get(ranges[0], ranges[1])
                position = tuple(map(int, str(ranges[0]).split(".")))
                links.append((position, label, tag.removeprefix("link:")))
        if not links:
            return self.plain()
        sources = "\n".join(f"- {label}: {url}" for _, label, url in sorted(links))
        return self.plain().rstrip() + "\n\nSources:\n" + sources


# Which state lines are asking for something, and which are only reporting.
# Keyed on the label render_state emits, so the panel colours what the pipeline
# calls important rather than what this module guesses is.
_ALARM_KEYS = {"blocked", "unresolved conflicts", "undeclared (required)"}
_CAUTION_KEYS = {"warning", "anomalies"}


class StatePanel(ttk.Frame):
    """Where the run stands, refreshed from the worker between tool calls.

    Rendered from :func:`~polyrmc.agent.toolspec.render_state` -- the same text
    the model is given when it calls ``get_state``. Deliberately the same: a
    panel that computed its own summary would eventually disagree with the one
    the agent is reasoning over, and the researcher would have no way to tell
    which of the two was describing their run.
    """

    def __init__(self, parent, fonts: Fonts, on_open=None, on_research=None,
                 on_runs=None, on_notes=None) -> None:
        from polyrmc.gui.shell import HomePanel

        super().__init__(parent, style="Paper.TFrame")
        p = PALETTE
        self.fonts = fonts
        self._details_requested = False
        noop = lambda: None
        self.empty = HomePanel(self, fonts, on_open or noop, on_research or noop,
                               on_runs or noop, on_notes or noop)
        self.empty.pack(fill="both", expand=True)
        self.details_toggle = ttk.Button(self, text="View session details", style="Quiet.TButton", command=self._toggle_details)
        self.details_toggle.pack(side="bottom", anchor="w", padx=24, pady=(8, 14))
        self.text, frame = _scrolled_text(
            self,
            wrap="word",
            background=p.paper,
            foreground=p.ink,
            font=fonts.small,
            borderwidth=0,
            highlightthickness=0,
            padx=24,
            pady=18,
            spacing1=7,
            spacing3=7,
            cursor="arrow",
            tabs=(VALUE_COLUMN,),
            # A modest requested width so the panel can be dragged narrow. Left
            # at Tk's default of 80 characters it sets a floor the sash cannot
            # cross, and the researcher who wants the conversation wide finds the
            # divider stuck.
            width=40,
        )
        self.details_frame = frame
        t = self.text
        # One number for the tab stop and the wrap indent, in pixels. They were
        # a tab stop in points and a margin in pixels, which is the same distance
        # written two ways and therefore two different distances -- a wrapped
        # blocked reason restarted to the left of the column it belonged to.
        t.tag_configure("row", lmargin1=0, lmargin2=VALUE_COLUMN, tabs=(VALUE_COLUMN,))
        t.tag_configure("key", foreground=p.faint, font=fonts.small)
        t.tag_configure("value", foreground=p.ink, font=fonts.mono_small)
        t.tag_configure("alarm", foreground=p.refused, font=fonts.mono_small)
        t.tag_configure("caution", foreground=p.commit, font=fonts.mono_small)
        t.tag_configure("withheld", foreground=p.faint, font=fonts.mono_small)
        t.configure(state="disabled")
        self.show("(no run open)")

    def _toggle_details(self) -> None:
        self._details_requested = not self._details_requested
        self._place_content()

    def _place_content(self) -> None:
        if self._details_requested or self._loaded:
            self.empty.pack_forget()
            self.details_frame.pack(fill="both", expand=True)
            self.details_toggle.configure(text="Hide session details" if not self._loaded else "Run details")
        else:
            self.details_frame.pack_forget()
            self.empty.pack(fill="both", expand=True)
            self.details_toggle.configure(text="View session details")

    def show(self, rendered: str) -> None:
        p = PALETTE
        t = self.text
        yview = t.yview()[0]
        t.configure(state="normal")
        t.delete("1.0", "end")
        for line in str(rendered).splitlines():
            key, sep, value = line.partition(": ")
            if not sep:
                t.insert("end", line + "\n", ("row", "key"))
                continue
            style = "value"
            if key in _ALARM_KEYS or value.startswith("no --"):
                style = "alarm"
            elif key in _CAUTION_KEYS and not value.startswith("0 region"):
                style = "caution"
            elif "withheld" in value or "not chosen" in value or value == "none":
                style = "withheld"
            t.insert("end", key, ("row", "key"))
            t.insert("end", "\t", "row")
            t.insert("end", value + "\n", ("row", style))
        t.configure(state="disabled")
        t.yview_moveto(yview)
        self._loaded = "loaded: yes" in str(rendered)
        self._place_content()


class ToolsPanel(ttk.Frame):
    """Every tool, by tier, with the commits marked.

    The tiering is the whole of the approval design -- reads and computes run
    freely because they cannot change a run, and every commit stops to ask -- so
    the panel is grouped by it rather than alphabetically. A researcher deciding
    whether to hand over standing approval for a tool should be able to see what
    class of thing they are agreeing to.
    """

    TIER_NOTE = {
        "read": "runs freely; cannot change the run",
        "compute": "runs freely; computes into the run's own record",
        "commit": "stops and asks you first",
    }

    def __init__(self, parent, fonts: Fonts, registry: dict, on_always: Callable[[str], None]) -> None:
        super().__init__(parent, style="Paper.TFrame")
        p = PALETTE
        self.registry = registry
        self.on_always = on_always

        self.tree = ttk.Treeview(self, columns=("what",), show="tree headings", height=14)
        self.tree.heading("#0", text="tool", anchor="w")
        self.tree.heading("what", text="what it does", anchor="w")
        self.tree.column("#0", width=210, stretch=False)
        self.tree.column("what", width=340, stretch=True)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.tag_configure("tier", foreground=p.faint, font=fonts.bold)
        self.tree.tag_configure("commit", foreground=p.commit)
        self.tree.tag_configure("plain", foreground=p.ink)
        self.tree.bind("<Double-1>", self._double_click)
        self.refresh()

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for tier in ("read", "compute", "commit"):
            members = sorted(
                (t for t in self.registry.values() if t.tier == tier), key=lambda t: t.name
            )
            if not members:
                continue
            parent = self.tree.insert(
                "", "end", text=f"{tier}  ({len(members)})",
                values=(self.TIER_NOTE[tier],), open=True, tags=("tier",),
            )
            for tool in members:
                self.tree.insert(
                    parent, "end", iid=tool.name, text=f"   {tool.name}",
                    values=(tool.description.split(".")[0],),
                    tags=("commit" if tool.mutating else "plain",),
                )

    def _double_click(self, _event) -> None:
        name = self.tree.focus()
        tool = self.registry.get(name)
        if tool is not None and tool.mutating:
            self.on_always(name)


class SearchPanel(ttk.Frame):
    """A query box over one tool that needs no model.

    Past runs and the notebook are searched here directly. Both are things a
    researcher wants to look at *before* deciding what to ask, and routing that
    through a model would spend a request to read a file this process can open
    itself.
    """

    def __init__(self, parent, fonts: Fonts, placeholder: str, run: Callable[[str], str]) -> None:
        super().__init__(parent, style="Paper.TFrame")
        p = PALETTE
        self.run = run

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=(10, 6))
        self.query = ttk.Entry(bar, font=fonts.body)
        self.query.pack(side="left", fill="x", expand=True)
        self.query.bind("<Return>", lambda _e: self.search())
        ttk.Button(bar, text="Search", command=self.search, width=8).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(bar, text="Clear", command=self.clear, width=6).pack(side="left", padx=(4, 0))
        self.hint = ttk.Label(self, text=placeholder, style="Faint.TLabel", wraplength=360)
        self.hint.pack(
            anchor="w", padx=10, pady=(0, 6)
        )
        self.bind("<Configure>", lambda event: self.hint.configure(wraplength=max(160, event.width - 24)))

        self.out, frame = _scrolled_text(
            self, wrap="word", background=p.paper, foreground=p.ink,
            font=fonts.body, borderwidth=0, highlightthickness=0,
            padx=12, pady=8, cursor="arrow", width=40,
        )
        frame.pack(fill="both", expand=True)
        self.out.configure(state="disabled")
        self.show("Search by keyword, or leave the query blank and choose Search to browse all records.")

    def clear(self) -> None:
        self.query.delete(0, "end")
        self.show("Search by keyword, or leave the query blank and choose Search to browse all records.")
        self.query.focus_set()

    def search(self) -> None:
        self.show(self.run(self.query.get().strip()))

    def show(self, text: str) -> None:
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("end", str(text).rstrip() + "\n")
        self.out.configure(state="disabled")


class FigurePanel(ttk.Frame):
    """The PNGs this pipeline draws, where the decision about them is made.

    This is the one thing a window buys outright. The figures were always
    written -- to ``data/runs/figures`` and beside each processed CSV -- and
    they were always somewhere else, so a researcher asking whether a smoothing
    window tracked the trace answered it from a number in a transcript while the
    picture of it sat unopened in a folder.

    Every caveat these figures carry is printed *into the image* by
    :mod:`polyrmc.figures`, so what is shown here is the whole claim, not a
    thumbnail of it with the qualifications left in the tool output.
    """

    def __init__(self, parent, fonts: Fonts, directories: Sequence[Path]) -> None:
        super().__init__(parent, style="Paper.TFrame")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        p = PALETTE
        self.directories = [Path(d) for d in directories]
        self.paths: list[Path] = []

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Button(top, text="rescan", command=self.refresh, width=8).pack(side="left")
        self.open_button = ttk.Button(
            top, text="open full size", command=self._open, state="disabled"
        )
        self.open_button.pack(side="left", padx=(6, 0))
        self.caption = ttk.Label(self, text="", style="Faint.TLabel")
        self.caption.pack(anchor="w", padx=10)

        split = ttk.PanedWindow(self, orient="vertical")
        split.pack(fill="both", expand=True, padx=8, pady=8)

        holder = ttk.Frame(split)
        self.listing = tk.Listbox(
            holder, background=p.paper, foreground=p.ink, font=fonts.mono_small,
            borderwidth=0, highlightthickness=0, activestyle="none",
            selectbackground=p.wash_accent, selectforeground=p.ink, height=6,
        )
        bar = ttk.Scrollbar(holder, orient="vertical", command=self.listing.yview)
        self.listing.configure(yscrollcommand=bar.set)
        self.listing.pack(side="left", fill="both", expand=True)
        bar.pack(side="left", fill="y")
        self.listing.bind("<<ListboxSelect>>", lambda _e: self._select())
        self.listing.bind("<Double-1>", lambda _e: self._open())
        # weight 0: the listing keeps the height it asked for and every pixel
        # gained by widening the panel or dragging the divider goes to the
        # figure, which is the thing being looked at.
        split.add(holder, weight=0)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_facecolor(p.paper)
        self.axes = self.figure.add_axes((0, 0, 1, 1))
        self.axes.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, master=split)
        split.add(self.canvas.get_tk_widget(), weight=1)
        self._blank("no figure drawn yet")
        self.refresh()

    def refresh(self) -> None:
        found: list[Path] = []
        for directory in self.directories:
            if directory.is_dir():
                found.extend(directory.rglob("*.png"))
        seen: dict[Path, float] = {}
        for path in found:
            try:
                seen[path.resolve()] = path.stat().st_mtime
            except OSError:
                continue
        self.paths = sorted(seen, key=lambda q: seen[q], reverse=True)

        selected = self.current()
        self.listing.delete(0, "end")
        for path in self.paths:
            self.listing.insert("end", self._label(path))
        if selected in self.paths:
            index = self.paths.index(selected)
            self.listing.selection_set(index)
            self.listing.see(index)
        self.caption.configure(
            text=f"{len(self.paths)} figure(s), newest first"
            if self.paths
            else "nothing drawn yet -- ask the agent for a figure, or run `polyrmc process`"
        )

    def _label(self, path: Path) -> str:
        for directory in self.directories:
            try:
                return str(path.relative_to(directory.resolve()))
            except (ValueError, OSError):
                continue
        return path.name

    def current(self) -> Path | None:
        selection = self.listing.curselection()
        if not selection or selection[0] >= len(self.paths):
            return None
        return self.paths[selection[0]]

    def show_newest(self) -> None:
        self.refresh()
        if self.paths:
            self.listing.selection_clear(0, "end")
            self.listing.selection_set(0)
            self._select()

    def _select(self) -> None:
        path = self.current()
        if path is None:
            return
        self.open_button.configure(state="normal")
        try:
            from matplotlib.image import imread

            image = imread(str(path))
        except Exception as error:  # noqa: BLE001 - an unreadable PNG is not a crash
            self._blank(f"cannot draw {path.name}: {error}")
            return
        self.axes.clear()
        self.axes.set_axis_off()
        self.axes.imshow(image, interpolation="antialiased")
        # Held at the top of the pane rather than centred in it. These figures
        # are wide and the pane is tall, so an equal aspect leaves a band of
        # empty paper either way; put it all below the figure, where it reads as
        # the end of the panel rather than as a gap someone forgot to fill.
        self.axes.set_anchor("N")
        self.canvas.draw_idle()
        self.caption.configure(text=self._label(path))

    def _blank(self, message: str) -> None:
        self.axes.clear()
        self.axes.set_axis_off()
        self.axes.set_anchor("C")  # a message belongs in the middle; a figure does not
        self.axes.text(
            0.5, 0.5, message, ha="center", va="center",
            color=PALETTE.faint, fontsize=9, transform=self.axes.transAxes,
        )
        self.canvas.draw_idle()

    def _open(self) -> None:
        path = self.current()
        if path is not None:
            open_externally(path)


class ApprovalDialog(tk.Toplevel):
    """The commit gate, as a window.

    Everything about it is copied from the terminal prompt on purpose, including
    what it does *not* do. It does not summarise the arguments; they are listed
    one per line, in full, however many there are. It does not show the call --
    a person deciding whether to allow a change is reviewing a change, not a
    request going over a wire. And declining is a first-class answer with a
    reason attached, because "no, use the wider band" is worth more to the agent
    than a bare refusal.

    Closing the window is a refusal. There is no way to dismiss this dialog into
    an approval, because an approval nobody gave is the one outcome the audit
    trail exists to rule out.
    """

    def __init__(self, parent, fonts: Fonts, tool: Any, call: Any, activity: str) -> None:
        super().__init__(parent)
        from polyrmc.agent.orchestrator import Approval

        self.answer = Approval(False, "declined")
        p = PALETTE
        self.title("Commit -- needs your approval")
        self.configure(background=p.wash_commit)
        self.resizable(True, False)
        self.transient(parent)

        wrap = tk.Frame(self, background=p.wash_commit, padx=20, pady=16)
        wrap.pack(fill="both", expand=True)

        tk.Label(
            wrap, text="COMMIT -- NEEDS YOUR APPROVAL", font=fonts.heading,
            foreground=p.commit, background=p.wash_commit, anchor="w",
        ).pack(fill="x")
        tk.Label(
            wrap, text=activity, font=fonts.body, foreground=p.ink,
            background=p.wash_commit, anchor="w",
        ).pack(fill="x", pady=(2, 10))

        table = tk.Frame(wrap, background=p.paper, highlightthickness=1,
                         highlightbackground=p.rule, padx=14, pady=10)
        table.pack(fill="both", expand=True)
        arguments = dict(getattr(call, "arguments", {}) or {})
        if not arguments:
            tk.Label(
                table, text="(no arguments)", font=fonts.mono_small,
                foreground=p.faint, background=p.paper, anchor="w",
            ).pack(fill="x")
        for row, (key, value) in enumerate(arguments.items()):
            line = tk.Frame(table, background=p.paper)
            line.pack(fill="x", pady=1)
            tk.Label(
                line, text=key.replace("_", " "), font=fonts.small, width=20,
                foreground=p.faint, background=p.paper, anchor="w",
            ).pack(side="left", anchor="n")
            # wraplength rather than truncation: an argument long enough to need
            # wrapping is exactly the argument worth reading before allowing it.
            tk.Label(
                line, text=str(value), font=fonts.mono_small, foreground=p.ink,
                background=p.paper, anchor="w", justify="left", wraplength=420,
            ).pack(side="left", fill="x", expand=True)

        tk.Label(
            wrap, text=tool.description.splitlines()[0], font=fonts.small,
            foreground=p.quiet, background=p.wash_commit, anchor="w",
            justify="left", wraplength=560,
        ).pack(fill="x", pady=(10, 4))

        reason = tk.Frame(wrap, background=p.wash_commit)
        reason.pack(fill="x", pady=(4, 12))
        tk.Label(
            reason, text="if you decline, say why (optional):", font=fonts.small,
            foreground=p.faint, background=p.wash_commit,
        ).pack(side="left")
        self.reason = ttk.Entry(reason, font=fonts.small)
        self.reason.pack(side="left", fill="x", expand=True, padx=(8, 0))

        buttons = tk.Frame(wrap, background=p.wash_commit)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Decline", command=self._decline).pack(side="right")
        ttk.Button(
            buttons, text="Always allow this tool", command=self._always
        ).pack(side="right", padx=6)
        allow = ttk.Button(buttons, text="Allow", style="Accent.TButton", command=self._allow)
        allow.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._decline)
        self.bind("<Escape>", lambda _e: self._decline())
        self.bind("<Return>", lambda _e: self._allow())
        self.update_idletasks()
        self._centre(parent)
        allow.focus_set()
        self.grab_set()

    def _centre(self, parent) -> None:
        try:
            x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
            y = parent.winfo_rooty() + max(60, (parent.winfo_height() - self.winfo_height()) // 3)
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except tk.TclError:
            pass

    def _allow(self) -> None:
        from polyrmc.agent.orchestrator import Approval

        self.answer = Approval(True)
        self.destroy()

    def _always(self) -> None:
        from polyrmc.agent.orchestrator import Approval

        self.answer = Approval(True, "always allowed this session", remember=True)
        self.destroy()

    def _decline(self) -> None:
        from polyrmc.agent.orchestrator import Approval

        stated = self.reason.get().strip()
        self.answer = Approval(False, stated or "declined")
        self.destroy()

    @classmethod
    def put(cls, parent, fonts: Fonts, tool: Any, call: Any, activity: str) -> Any:
        dialog = cls(parent, fonts, tool, call, activity)
        parent.wait_window(dialog)
        return dialog.answer


class SignInDialog(tk.Toplevel):
    """Take a key for one provider, without it reaching the scrollback.

    The terminal reads keys with ``getpass``. The window's equivalent is a
    masked entry that is never echoed back for confirmation. Storing it is a
    separate question from accepting it, because a shared or teaching machine is
    a normal place to run this and a key written there outlives the session.
    """

    def __init__(self, parent, fonts: Fonts, specs: dict, keyed: Sequence[str]) -> None:
        super().__init__(parent)
        self.result: tuple[str, str, bool] | None = None
        p = PALETTE
        self.title("Bring your own tokens")
        self.configure(background=p.panel)
        self.resizable(False, False)
        self.transient(parent)
        self.specs = specs

        wrap = tk.Frame(self, background=p.panel, padx=22, pady=18)
        wrap.pack(fill="both", expand=True)
        tk.Label(
            wrap, text="Which brain drives this session?", font=fonts.heading,
            background=p.panel, foreground=p.ink, anchor="w",
        ).pack(fill="x")
        tk.Label(
            wrap,
            text="This package ships no key and brokers no model access. These are\n"
                 "API keys billed per token -- a Claude Pro or ChatGPT Plus\n"
                 "subscription is a separate product and does not grant API access.",
            font=fonts.small, background=p.panel, foreground=p.quiet,
            anchor="w", justify="left",
        ).pack(fill="x", pady=(4, 12))

        self.choice = tk.StringVar(value=(list(keyed) or list(specs))[0])
        for name, spec in specs.items():
            row = tk.Frame(wrap, background=p.panel)
            row.pack(fill="x", pady=1)
            ttk.Radiobutton(
                row, text=name, value=name, variable=self.choice,
                command=self._retarget, width=10,
            ).pack(side="left")
            status = "keyed" if name in keyed else spec.canonical_key
            tk.Label(
                row, text=status, font=fonts.mono_small, background=p.panel,
                foreground=p.settled if name in keyed else p.faint, anchor="w",
            ).pack(side="left")

        self.where = tk.Label(
            wrap, text="", font=fonts.small, background=p.panel,
            foreground=p.accent, anchor="w", cursor="hand2",
        )
        self.where.pack(fill="x", pady=(12, 2))
        self.where.bind("<Button-1>", lambda _e: self._browse())

        self.entry = ttk.Entry(wrap, show="•", font=fonts.mono, width=52)
        self.entry.pack(fill="x", pady=(2, 6))
        self.save = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            wrap, text="save it to .env for next time (gitignored)", variable=self.save
        ).pack(anchor="w")

        buttons = tk.Frame(wrap, background=p.panel)
        buttons.pack(fill="x", pady=(16, 0))
        ttk.Button(buttons, text="Skip", command=self._skip).pack(side="right")
        ttk.Button(
            buttons, text="Use this key", style="Accent.TButton", command=self._accept
        ).pack(side="right", padx=6)

        self._retarget()
        self.protocol("WM_DELETE_WINDOW", self._skip)
        self.bind("<Escape>", lambda _e: self._skip())
        self.bind("<Return>", lambda _e: self._accept())
        self.entry.focus_set()
        self.grab_set()

    def _retarget(self) -> None:
        spec = self.specs[self.choice.get()]
        self.where.configure(text=f"get a key: {spec.key_url}")

    def _browse(self) -> None:
        import webbrowser

        webbrowser.open(self.specs[self.choice.get()].key_url)

    def _accept(self) -> None:
        value = self.entry.get().strip()
        if not value:
            self._skip()
            return
        self.result = (self.choice.get(), value, bool(self.save.get()))
        self.destroy()

    def _skip(self) -> None:
        self.result = None
        self.destroy()

    @classmethod
    def put(cls, parent, fonts: Fonts, specs: dict, keyed: Sequence[str]):
        dialog = cls(parent, fonts, specs, keyed)
        parent.wait_window(dialog)
        return dialog.result


class OpenRunDialog(tk.Toplevel):
    """Point the session at a run.

    The declarations file is optional, and usually absent. A run opens
    undeclared: the trace loads, the anomalies are found, the smoothing band is
    computed and the figures draw, because none of that needs to know what the
    sample is. What does need to know -- the concentration axis, K, Mw -- is
    blocked with the reason on the record until somebody answers, and you answer
    by telling the agent, which records it against your name.

    Supplying one is still the fastest route for a run that has already been
    through intake. What is refused is a declarations file with no raw file to
    apply it to, which describes nothing.
    """

    def __init__(self, parent, fonts: Fonts, author: str) -> None:
        super().__init__(parent)
        self.result: dict[str, str] | None = None
        p = PALETTE
        self.title("Open a run")
        self.configure(background=p.panel)
        self.resizable(False, False)
        self.transient(parent)

        wrap = tk.Frame(self, background=p.panel, padx=22, pady=18)
        wrap.pack(fill="both", expand=True)
        tk.Label(
            wrap, text="Open a run", font=fonts.heading, background=p.panel,
            foreground=p.ink, anchor="w",
        ).pack(fill="x")
        tk.Label(
            wrap,
            text="A raw file is all you need. It opens undeclared, and anything\n"
                 "that depends on an unanswered question is blocked with the\n"
                 "reason shown, never guessed. Tell the agent what the run is and\n"
                 "it records each answer against your name.\n\n"
                 "The declarations file is optional: it is what `polyrmc intake`\n"
                 "wrote, if this run has already been through it.",
            font=fonts.small, background=p.panel, foreground=p.quiet,
            anchor="w", justify="left",
        ).pack(fill="x", pady=(4, 12))

        self.raw = self._picker(wrap, fonts, "raw instrument file", (("data files", "*.*"),))
        self.declarations = self._picker(
            wrap, fonts, "declarations (optional)", (("JSON", "*.json"), ("all", "*.*"))
        )

        fields = tk.Frame(wrap, background=p.panel)
        fields.pack(fill="x", pady=(10, 0))
        tk.Label(
            fields, text="author", font=fonts.small, width=18, anchor="w",
            background=p.panel, foreground=p.faint,
        ).pack(side="left")
        self.author = ttk.Entry(fields, font=fonts.small)
        self.author.insert(0, author)
        self.author.pack(side="left", fill="x", expand=True)

        buttons = tk.Frame(wrap, background=p.panel)
        buttons.pack(fill="x", pady=(18, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(
            buttons, text="Open", style="Accent.TButton", command=self._accept
        ).pack(side="right", padx=6)
        self.warning = tk.Label(
            wrap, text="", font=fonts.small, background=p.panel, foreground=p.refused,
            anchor="w",
        )
        self.warning.pack(fill="x", pady=(8, 0))

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda _e: self.destroy())
        self.grab_set()

    def _picker(self, parent, fonts: Fonts, label: str, kinds) -> ttk.Entry:
        from tkinter import filedialog

        row = tk.Frame(parent, background=PALETTE.panel)
        row.pack(fill="x", pady=2)
        tk.Label(
            row, text=label, font=fonts.small, width=18, anchor="w",
            background=PALETTE.panel, foreground=PALETTE.faint,
        ).pack(side="left")
        entry = ttk.Entry(row, font=fonts.mono_small, width=44)
        entry.pack(side="left", fill="x", expand=True)

        def browse() -> None:
            chosen = filedialog.askopenfilename(parent=self, title=label, filetypes=kinds)
            if chosen:
                entry.delete(0, "end")
                entry.insert(0, chosen)

        ttk.Button(row, text="...", width=3, command=browse).pack(side="left", padx=(4, 0))
        return entry

    def _accept(self) -> None:
        raw = self.raw.get().strip()
        declarations = self.declarations.get().strip()
        if declarations and not raw:
            self.warning.configure(
                text="A declarations file describes a run, so it needs the run. "
                     "Choose a raw file as well."
            )
            return
        if not raw:
            self.destroy()
            return
        self.result = {
            "raw": raw,
            "declarations": declarations,
            "author": self.author.get().strip() or "researcher",
        }
        self.destroy()

    @classmethod
    def put(cls, parent, fonts: Fonts, author: str):
        dialog = cls(parent, fonts, author)
        parent.wait_window(dialog)
        return dialog.result


__all__ = [
    "ApprovalDialog",
    "FigurePanel",
    "OpenRunDialog",
    "SearchPanel",
    "SignInDialog",
    "StatePanel",
    "ToolsPanel",
    "Transcript",
    "open_externally",
]
