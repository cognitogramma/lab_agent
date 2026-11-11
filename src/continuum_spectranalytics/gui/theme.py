"""An ink-blue research studio with copper actions and readable prose.

Run decisions keep their own semantic colours. The rest of the interface uses
spacing, type and surface contrast to establish hierarchy rather than a border
around every control. Copper marks actions, teal marks settled results, and
blue selections keep the interface distinct from the scientific decisions.
"""

from __future__ import annotations

import tkinter.font as tkfont
from dataclasses import dataclass
from tkinter import TclError, ttk

# Prefer Times New Roman and resolve an installed serif fallback explicitly;
# an unavailable family otherwise silently becomes Tk's platform default.
UI_FAMILIES = ("Times New Roman", "Liberation Serif", "Times", "Nimbus Roman", "DejaVu Serif")
HEADING_FAMILIES = UI_FAMILIES
MONO_FAMILIES = ("Cascadia Mono", "Consolas", "SF Mono", "DejaVu Sans Mono", "Courier New")


@dataclass(frozen=True)
class Palette:
    """The colours, named for what they mean rather than what they are.

    ``commit`` and ``refused`` are different hues on purpose. They are both
    interruptions, and a reader who has learned that "the orange block is the
    one I answer" should not have to re-read to find out that this orange block
    is the one that already happened.
    """

    paper: str = "#17263B"       # cards, figures and conversation
    panel: str = "#111E30"       # surrounding panels
    sunken: str = "#20334B"      # inputs and quiet controls
    rule: str = "#31445C"        # subtle dividers
    ink: str = "#EDF2F7"         # primary text
    quiet: str = "#AFBDCD"       # supporting text
    faint: str = "#AFBDCD"       # labels, units and timestamps
    accent: str = "#E7B47A"      # primary actions, links and focus
    accent_ink: str = "#171C26"  # text on a solid copper action
    accent_hover: str = "#F0C996"
    accent_pressed: str = "#D69C60"
    support: str = "#79C6C3"     # complementary scientific/status accent
    hover: str = "#2B415C"
    selection: str = "#29445E"
    disabled_ink: str = "#6E839B"
    commit: str = "#F0B558"      # a change waiting on a person
    refused: str = "#FF8C91"     # a refusal or failure
    settled: str = "#79C6C3"     # a check that passed
    wash_commit: str = "#33291C"
    wash_refused: str = "#351F23"
    wash_accent: str = "#243B53"
    nav: str = "#111E30"
    nav_active: str = "#20334B"
    nav_ink: str = "#EDF2F7"
    nav_muted: str = "#AFBDCD"
    canvas: str = "#0B1220"


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
    title: tuple


def build_fonts(root) -> Fonts:
    ui = _first_installed(root, UI_FAMILIES, "TkDefaultFont")
    heading = _first_installed(root, HEADING_FAMILIES, "TkDefaultFont")
    mono = _first_installed(root, MONO_FAMILIES, "TkFixedFont")
    return Fonts(
        body=(ui, 12),
        bold=(ui, 12, "bold"),
        small=(ui, 11),
        mono=(mono, 10),
        mono_small=(mono, 9),
        heading=(heading, 16, "bold"),
        title=(heading, 22, "bold"),
    )


def apply(root) -> Fonts:
    """Dress the window and hand back the resolved fonts.

    ``clam`` honours surface and state colours on the legacy ttk controls.
    The surrounding rounded shell can use those same palette values without
    changing the fonts or the measurement widgets.
    """
    fonts = build_fonts(root)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:  # noqa: BLE001 - a missing theme is not worth a dead window
        pass

    p = PALETTE
    try:
        root.configure(fg_color=p.canvas)
    except TclError:  # plain Tk/Toplevel test surfaces
        root.configure(background=p.canvas)
    root.option_add("*Menu.background", p.panel)
    root.option_add("*Menu.foreground", p.ink)
    root.option_add("*Menu.activeBackground", p.sunken)
    root.option_add("*Menu.activeForeground", p.ink)
    root.option_add("*Menu.font", fonts.small)
    style.configure(".", background=p.panel, foreground=p.ink, font=fonts.body)
    style.configure("TFrame", background=p.panel)
    style.configure("Paper.TFrame", background=p.paper)
    style.configure("Card.TFrame", background=p.paper, borderwidth=0, relief="flat")
    style.configure("Canvas.TFrame", background=p.canvas)
    style.configure("Toolbar.TFrame", background=p.paper)
    style.configure("TLabel", background=p.panel, foreground=p.ink)
    style.configure("Quiet.TLabel", foreground=p.quiet, font=fonts.small)
    style.configure("Faint.TLabel", foreground=p.faint, font=fonts.small)
    style.configure("Heading.TLabel", font=fonts.heading, foreground=p.ink)
    style.configure("Brand.TLabel", font=(fonts.heading[0], 18, "bold"), foreground=p.accent)
    style.configure("PageTitle.TLabel", font=fonts.title, foreground=p.ink)
    style.configure("PageSubtitle.TLabel", font=fonts.body, foreground=p.quiet)
    style.configure("Kicker.TLabel", font=(fonts.body[0], 9, "bold"), foreground=p.accent)
    style.configure("CardTitle.TLabel", font=fonts.heading, foreground=p.ink, background=p.paper)
    style.configure("CardBody.TLabel", font=fonts.body, foreground=p.ink, background=p.paper)
    style.configure("CardQuiet.TLabel", font=fonts.small, foreground=p.quiet, background=p.paper)
    style.configure("CardKicker.TLabel", font=(fonts.body[0], 9, "bold"), foreground=p.accent, background=p.paper)
    style.configure("Badge.TLabel", background=p.wash_accent, foreground=p.accent, font=(fonts.body[0], 9, "bold"), padding=(10, 5))
    style.configure("PaperHeading.TLabel", font=fonts.heading, foreground=p.ink, background=p.paper)
    style.configure("PaperQuiet.TLabel", font=fonts.small, foreground=p.quiet, background=p.paper)
    style.configure("Commit.TLabel", foreground=p.commit, font=fonts.bold)
    style.configure("Refused.TLabel", foreground=p.refused)
    style.configure("Value.TLabel", font=fonts.mono_small, foreground=p.ink)

    style.configure(
        "TButton",
        background=p.sunken,
        foreground=p.ink,
        borderwidth=1,
        bordercolor=p.rule,
        lightcolor=p.sunken,
        darkcolor=p.sunken,
        focuscolor=p.accent,
        focusthickness=1,
        relief="flat",
        padding=(14, 8),
    )
    style.map(
        "TButton",
        background=[("disabled", p.panel), ("pressed", p.selection), ("active", p.hover)],
        foreground=[("disabled", p.disabled_ink)],
        bordercolor=[("focus", p.accent), ("active", p.rule)],
    )
    style.configure(
        "Accent.TButton", background=p.accent, foreground=p.accent_ink, padding=(18, 9), bordercolor=p.accent,
        lightcolor=p.accent, darkcolor=p.accent, font=fonts.bold
    )
    style.map(
        "Accent.TButton",
        background=[("disabled", p.sunken), ("pressed", p.accent_pressed), ("active", p.accent_hover)],
        foreground=[("disabled", p.disabled_ink), ("active", p.accent_ink)],
        bordercolor=[("disabled", p.sunken), ("focus", p.accent_hover), ("active", p.accent_hover)],
    )
    style.configure("Quiet.TButton", background=p.panel, bordercolor=p.panel, foreground=p.quiet)
    style.configure(
        "TopNav.TButton", background=p.paper, foreground=p.quiet, borderwidth=0,
        bordercolor=p.paper, lightcolor=p.paper, darkcolor=p.paper,
        relief="flat", padding=(16, 10), font=fonts.body,
    )
    style.map(
        "TopNav.TButton", background=[("active", p.sunken), ("pressed", p.wash_accent)],
        foreground=[("active", p.ink)], bordercolor=[("focus", p.accent)],
    )
    style.configure(
        "Selected.TopNav.TButton", background=p.rule, foreground=p.ink,
        bordercolor=p.rule, lightcolor=p.rule, darkcolor=p.rule,
        font=fonts.bold,
    )
    style.map(
        "Selected.TopNav.TButton", background=[("pressed", p.selection), ("active", p.hover)],
        foreground=[("active", p.ink)], bordercolor=[("focus", p.accent)],
    )
    style.configure(
        "Toolbar.TButton", background=p.paper, bordercolor=p.rule,
        foreground=p.ink, padding=(14, 9),
    )
    style.configure("Commit.TButton", background=p.commit, foreground=p.accent_ink, bordercolor=p.commit)
    style.map("Commit.TButton", background=[("disabled", p.sunken), ("pressed", "#D69B42"), ("active", "#F6C576")],
              foreground=[("disabled", p.disabled_ink), ("active", p.accent_ink)])
    style.configure(
        "TMenubutton", background=p.sunken, foreground=p.ink, arrowcolor=p.quiet,
        bordercolor=p.rule, lightcolor=p.sunken, darkcolor=p.sunken,
        relief="flat", padding=(12, 9),
    )
    style.map("TMenubutton", background=[("active", p.hover)], foreground=[("disabled", p.disabled_ink)])

    style.configure(
        "TNotebook", background=p.panel, borderwidth=0, tabmargins=(0, 0, 0, 0)
    )
    style.configure(
        "TNotebook.Tab",
        background=p.panel,
        foreground=p.quiet,
        padding=(12, 9),
        borderwidth=0,
        font=fonts.small,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", p.paper)],
        foreground=[("selected", p.ink)],
        expand=[("selected", (0, 0, 0, 1))],
    )
    # Navigation owns page selection; keep the notebook as the content stack.
    # Clam's Notebook.client ignores borderwidth and paints its bevel from
    # these three colours. Match the content surface on every edge.
    style.configure(
        "Workspace.TNotebook", background=p.paper, borderwidth=0, tabmargins=0,
        bordercolor=p.paper, lightcolor=p.paper, darkcolor=p.paper, relief="flat",
    )
    style.layout("Workspace.TNotebook.Tab", [])

    style.configure(
        "TEntry",
        fieldbackground=p.sunken,
        foreground=p.ink,
        bordercolor=p.rule,
        lightcolor=p.rule,
        darkcolor=p.rule,
        insertcolor=p.ink,
        selectbackground=p.selection,
        selectforeground=p.ink,
        padding=(10, 9),
        relief="flat",
    )
    style.map(
        "TEntry", bordercolor=[("focus", p.accent)], lightcolor=[("focus", p.accent)], darkcolor=[("focus", p.accent)],
        fieldbackground=[("disabled", p.panel), ("readonly", p.sunken)],
        foreground=[("disabled", p.disabled_ink), ("readonly", p.ink)],
    )
    style.configure(
        "TCombobox", fieldbackground=p.sunken, background=p.sunken, foreground=p.ink,
        bordercolor=p.rule, lightcolor=p.rule, darkcolor=p.rule,
        arrowcolor=p.quiet, padding=(10, 7), arrowsize=13,
    )
    style.map(
        "TCombobox", fieldbackground=[("disabled", p.panel), ("readonly", p.sunken)],
        background=[("disabled", p.panel), ("active", p.hover), ("readonly", p.sunken)],
        foreground=[("disabled", p.disabled_ink), ("readonly", p.ink)],
        selectbackground=[("readonly", p.sunken)], selectforeground=[("readonly", p.ink)],
        bordercolor=[("focus", p.accent)],
    )
    root.option_add("*TCombobox*Listbox.background", p.sunken)
    root.option_add("*TCombobox*Listbox.foreground", p.ink)
    root.option_add("*TCombobox*Listbox.selectBackground", p.selection)
    root.option_add("*TCombobox*Listbox.selectForeground", p.ink)
    root.option_add("*TCombobox*Listbox.font", fonts.small)
    style.configure(
        "Treeview",
        background=p.paper,
        fieldbackground=p.paper,
        foreground=p.ink,
        borderwidth=0,
        rowheight=36,
        font=fonts.body,
        relief="flat",
    )
    style.configure(
        "Treeview.Heading",
        background=p.panel,
        foreground=p.quiet,
        font=(fonts.body[0], 9, "bold"),
        borderwidth=0,
        padding=(10, 10),
        relief="flat",
    )
    style.map(
        "Treeview",
        background=[("selected", p.selection)],
        foreground=[("selected", p.ink)],
    )
    style.configure(
        "TScrollbar",
        background=p.rule,
        troughcolor=p.paper,
        bordercolor=p.paper,
        lightcolor=p.paper,
        darkcolor=p.paper,
        arrowcolor=p.faint,
        borderwidth=0,
        arrowsize=11,
        relief="flat",
    )
    style.map("TScrollbar", background=[("pressed", p.disabled_ink), ("active", p.hover)])
    style.configure("TSeparator", background=p.rule)
    style.configure("TPanedwindow", background=p.rule)
    style.configure("Workspace.TPanedwindow", background=p.canvas, sashwidth=12)
    style.configure("TProgressbar", background=p.accent, troughcolor=p.sunken, borderwidth=0)
    style.configure("TCheckbutton", background=p.panel, foreground=p.ink, focuscolor=p.accent,
                    indicatorbackground=p.sunken, indicatorforeground=p.ink, bordercolor=p.rule)
    style.map("TCheckbutton", background=[("active", p.panel)], foreground=[("disabled", p.disabled_ink)],
              indicatorbackground=[("selected", p.accent)], indicatorforeground=[("selected", p.accent_ink)])
    style.configure("TRadiobutton", background=p.panel, foreground=p.ink, focuscolor=p.accent,
                    indicatorbackground=p.sunken, indicatorforeground=p.accent)
    style.map("TRadiobutton", background=[("active", p.panel)], foreground=[("disabled", p.disabled_ink)])
    return fonts


__all__ = [
    "HEADING_FAMILIES", "MONO_FAMILIES", "PALETTE", "UI_FAMILIES",
    "Fonts", "Palette", "apply", "build_fonts",
]
