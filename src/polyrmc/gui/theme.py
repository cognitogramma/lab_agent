"""How the window looks, in one place.

The terminal session solved its presentation problem by taking things away: no
call syntax, no argument dumps, no token counts, one plain line per action. A
window has a second axis the terminal does not, and the temptation is to spend
it on decoration. It is spent here on *distinction* instead -- the four things a
researcher must not skim are the four things that look unlike the rest:

* a **commit** waiting on them,
* a **refusal**, whose text names the band that would have been valid,
* a **stop**, which is not a message from the model and must never read as one,
* a **withheld** number, which is the pipeline declining to report rather than
  failing to compute.

Everything else -- the model's prose, the running account of what it did -- is
deliberately quiet. A palette where every event is emphatic is a palette where
none of them are.

The ground is paper rather than a dark editor theme, because the figures this
pipeline draws are white-background PNGs destined for a paper, and a dark frame
around them misrepresents what they will look like where they end up.
"""

from __future__ import annotations

import tkinter.font as tkfont
from dataclasses import dataclass
from tkinter import ttk

# Candidate families in preference order. The first one actually installed wins,
# because a font name Tk cannot resolve silently degrades to a default that is
# usually Times -- legible, and wrong in a way nobody thinks to report.
UI_FAMILIES = ("Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", "DejaVu Sans")
MONO_FAMILIES = ("Cascadia Mono", "Consolas", "SF Mono", "DejaVu Sans Mono", "Courier New")


@dataclass(frozen=True)
class Palette:
    """The colours, named for what they mean rather than what they are.

    ``commit`` and ``refused`` are different hues on purpose. They are both
    interruptions, and a reader who has learned that "the orange block is the
    one I answer" should not have to re-read to find out that this orange block
    is the one that already happened.
    """

    paper: str = "#FBFAF7"       # the conversation
    panel: str = "#F1EFEA"       # side panels, status bar
    sunken: str = "#E7E4DD"      # inputs, selected rows
    rule: str = "#D8D4CA"        # hairlines and borders
    ink: str = "#1B1A17"         # body text
    quiet: str = "#6B675E"       # the running account of actions
    faint: str = "#928D82"       # labels, units, timestamps
    accent: str = "#1D5C70"      # the researcher's own turn, links, focus
    commit: str = "#9A4B12"      # a change waiting on a person
    refused: str = "#98241F"     # a tool that said no, and a stop
    settled: str = "#2E6640"     # a check that passed
    wash_commit: str = "#FBF0E4"
    wash_refused: str = "#FBEAE7"
    wash_accent: str = "#E9F1F3"


PALETTE = Palette()


def _first_installed(root, families: tuple[str, ...], fallback: str) -> str:
    available = {name.lower() for name in tkfont.families(root)}
    for family in families:
        if family.lower() in available:
            return family
    return fallback


@dataclass(frozen=True)
class Fonts:
    body: tuple
    bold: tuple
    small: tuple
    mono: tuple
    mono_small: tuple
    heading: tuple


def build_fonts(root) -> Fonts:
    ui = _first_installed(root, UI_FAMILIES, "TkDefaultFont")
    mono = _first_installed(root, MONO_FAMILIES, "TkFixedFont")
    return Fonts(
        body=(ui, 10),
        bold=(ui, 10, "bold"),
        small=(ui, 9),
        mono=(mono, 10),
        mono_small=(mono, 9),
        heading=(ui, 11, "bold"),
    )


def apply(root) -> Fonts:
    """Dress the window and hand back the resolved fonts.

    ``clam`` rather than the native Windows theme, because the native one paints
    its own greys into notebook tabs and frame borders and ignores the
    background it is given -- which leaves a paper-coloured window with slate
    panels bolted to it. ``clam`` honours what it is told.
    """
    fonts = build_fonts(root)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:  # noqa: BLE001 - a missing theme is not worth a dead window
        pass

    p = PALETTE
    root.configure(background=p.panel)
    style.configure(".", background=p.panel, foreground=p.ink, font=fonts.body)
    style.configure("TFrame", background=p.panel)
    style.configure("Paper.TFrame", background=p.paper)
    style.configure("TLabel", background=p.panel, foreground=p.ink)
    style.configure("Quiet.TLabel", foreground=p.quiet, font=fonts.small)
    style.configure("Faint.TLabel", foreground=p.faint, font=fonts.small)
    style.configure("Heading.TLabel", font=fonts.heading, foreground=p.ink)
    style.configure("Commit.TLabel", foreground=p.commit, font=fonts.bold)
    style.configure("Refused.TLabel", foreground=p.refused)
    style.configure("Value.TLabel", font=fonts.mono_small, foreground=p.ink)

    style.configure(
        "TButton",
        background=p.sunken,
        foreground=p.ink,
        borderwidth=1,
        focusthickness=0,
        padding=(10, 4),
    )
    style.map(
        "TButton",
        background=[("pressed", p.rule), ("active", p.rule), ("disabled", p.panel)],
        foreground=[("disabled", p.faint)],
    )
    style.configure(
        "Accent.TButton", background=p.accent, foreground="#FFFFFF", padding=(14, 5)
    )
    style.map(
        "Accent.TButton",
        background=[("pressed", "#164A5A"), ("active", "#246E85"), ("disabled", p.rule)],
        foreground=[("disabled", p.faint)],
    )
    style.configure("Commit.TButton", background=p.commit, foreground="#FFFFFF")
    style.map("Commit.TButton", background=[("active", "#B25916")])

    style.configure(
        "TNotebook", background=p.panel, borderwidth=0, tabmargins=(4, 4, 0, 0)
    )
    style.configure(
        "TNotebook.Tab",
        background=p.panel,
        foreground=p.quiet,
        padding=(11, 5),
        borderwidth=0,
        font=fonts.small,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", p.paper)],
        foreground=[("selected", p.ink)],
        expand=[("selected", (0, 0, 0, 1))],
    )

    style.configure(
        "TEntry",
        fieldbackground=p.paper,
        bordercolor=p.rule,
        lightcolor=p.rule,
        darkcolor=p.rule,
        insertcolor=p.ink,
        padding=4,
    )
    style.configure(
        "Treeview",
        background=p.paper,
        fieldbackground=p.paper,
        foreground=p.ink,
        borderwidth=0,
        rowheight=21,
    )
    style.configure(
        "Treeview.Heading",
        background=p.panel,
        foreground=p.faint,
        font=fonts.small,
        borderwidth=0,
        padding=(6, 4),
    )
    style.map(
        "Treeview",
        background=[("selected", p.wash_accent)],
        foreground=[("selected", p.ink)],
    )
    style.configure(
        "TScrollbar",
        background=p.sunken,
        troughcolor=p.panel,
        bordercolor=p.panel,
        arrowcolor=p.faint,
        borderwidth=0,
    )
    style.map("TScrollbar", background=[("active", p.rule)])
    style.configure("TSeparator", background=p.rule)
    style.configure("TPanedwindow", background=p.rule)
    style.configure("TCheckbutton", background=p.panel, foreground=p.ink)
    style.map("TCheckbutton", background=[("active", p.panel)])
    return fonts


__all__ = ["MONO_FAMILIES", "PALETTE", "UI_FAMILIES", "Fonts", "Palette", "apply", "build_fonts"]
