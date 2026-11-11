"""Rounded library navigation and a useful, illustrated research home.

The illustrations are decorative vectors, never plots of a user's experiment.
The only library totals displayed here are supplied by the caller.
"""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk
from continuum_spectranalytics.gui.theme import PALETTE as P
from continuum_spectranalytics.gui.theme import build_fonts


SURFACE = P.paper
CARD = P.panel
RAISED = P.sunken
TEXT = P.ink
MUTED = P.quiet
ACCENT = P.accent


def _font(fonts, size: int = 15, *, heading: bool = False, bold: bool = False):
    family = fonts.heading[0] if heading else fonts.body[0]
    return ctk.CTkFont(family=family, size=size, weight="bold" if bold else "normal")


def _widget_scale(widget) -> float:
    """Match native artwork to the surrounding CTk logical-pixel geometry."""
    try:
        value = float(ctk.ScalingTracker.get_widget_scaling(widget))
    except (KeyError, AttributeError, TypeError, ValueError, tk.TclError):
        value = float(widget.winfo_fpixels("1i")) / 96.0
    return value if math.isfinite(value) and value > 0 else 1.0


class _Artwork(tk.Canvas):
    """Antialiased decorative vectors, with crisp native theme-font labels.

    Geometry is sampled above display resolution, then reduced once. Text stays
    native so Windows draws the chosen font at the final display size. The
    cached image has exactly one pixel per canvas pixel; it is never stretched.
    """

    SCALE = 3

    COLOURS = {
        "open": ("#23434D", "#17263B"),
        "research": ("#35465D", "#17263B"),
        "notes": ("#35465D", "#17263B"),
        "runs": ("#573E3A", "#17263B"),
        "hero": ("#223A50", "#17263B"),
    }

    def __init__(self, parent, kind: str, *, height: int = 120, width: int = 1, fonts=None):
        self._dpi_scale = _widget_scale(parent)
        self._logical_width, self._logical_height = width, height
        super().__init__(parent, width=self._pixels(width), height=self._pixels(height),
                         background=self.COLOURS[kind][0],
                         borderwidth=0, highlightthickness=0)
        self.kind = kind
        self.fonts = fonts or build_fonts(self)
        self._photo: ImageTk.PhotoImage | None = None
        self._painted_size: tuple[int, int] | None = None
        self._disposed = False
        self._redraw_id: str | None = None
        self.bind("<Configure>", self._schedule_draw)
        ctk.ScalingTracker.add_widget(self._set_scaling, self)

    def _pixels(self, value: float) -> int:
        return max(1, round(value * self._dpi_scale))

    def _set_scaling(self, widget_scaling: float, _window_scaling: float):
        if self._disposed or math.isclose(self._dpi_scale, widget_scaling):
            return
        self._dpi_scale = widget_scaling
        self._painted_size = None
        self.configure(width=self._pixels(self._logical_width), height=self._pixels(self._logical_height))
        self._schedule_draw()

    def _schedule_draw(self, _event=None):
        if not self._disposed and self._redraw_id is None:
            self._redraw_id = self.after_idle(self._redraw)

    def _redraw(self):
        self._redraw_id = None
        self._draw()

    def _draw(self, _event=None):
        if self._disposed:
            return
        width, height = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        if self._painted_size == (width, height):
            return
        image = self._render(width, height)
        photo = ImageTk.PhotoImage(image, master=self)
        self.delete("all")
        self.create_image(0, 0, image=photo, anchor="nw", tags=("illustration",))
        self._photo = photo
        self._painted_size = (width, height)
        if self.kind == "hero":
            self._hero_labels(width)
        else:
            label, colour = {
                "open": ("EXPERIMENTS", "#AFBDCD"),
                "research": ("IDEAS & SOURCES", "#D2C8EE"),
                "notes": ("RESEARCH MEMORY", "#D2C8EE"),
                "runs": ("YOUR LAB LIBRARY", "#E5B3BD"),
            }[self.kind]
            self.create_text(width * .1, height * .88, text=label, anchor="w",
                             fill=colour, font=(self.fonts.body[0], -self._pixels(10), "bold"), tags=("label",))

    def _render(self, width: int, height: int) -> Image.Image:
        """Render unchanged artwork geometry with smooth curves and circles."""
        display_width, display_height = width, height
        pixels_w, pixels_h = width * self.SCALE, height * self.SCALE
        # Fractional positions follow the allocated area; fixed radii, strokes,
        # offsets and responsive thresholds remain in logical units.
        width, height = width / self._dpi_scale, height / self._dpi_scale
        scale = self.SCALE * self._dpi_scale
        first, last = self.COLOURS[self.kind]
        start = tuple(int(first[n:n + 2], 16) for n in (1, 3, 5))
        end = tuple(int(last[n:n + 2], 16) for n in (1, 3, 5))
        # One continuous pixel gradient replaces the old 48 hard strip edges.
        row = Image.new("RGB", (pixels_w, 1))
        row.putdata([
            tuple(round(a + (b - a) * x / max(pixels_w - 1, 1)) for a, b in zip(start, end))
            for x in range(pixels_w)
        ])
        image = row.resize((pixels_w, pixels_h), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)

        def line(points, colour, thickness=1):
            draw.line([(round(x * scale), round(y * scale)) for x, y in points],
                      fill=colour, width=max(1, round(thickness * scale)), joint="curve")

        def oval(bounds, colour=None, outline=None, thickness=1):
            draw.ellipse(tuple(round(value * scale) for value in bounds), fill=colour,
                         outline=outline, width=max(1, round(thickness * scale)))

        def rectangle(bounds, colour):
            draw.rectangle(tuple(round(value * scale) for value in bounds), fill=colour)

        if self.kind == "hero":
            if width > 700:
                cx, cy = width - 72, 54
                oval((cx - 38, cy - 38, cx + 38, cy + 38), outline="#35634B")
                oval((cx - 25, cy - 25, cx + 25, cy + 25), outline="#3F7455")
                for x, y, radius in ((cx, cy, 7), (cx + 35, cy - 15, 4), (cx - 25, cy + 29, 5)):
                    line(((cx, cy), (x, y)), "#5A946C", 2)
                    oval((x - radius, y - radius, x + radius, y + radius), "#76C68F")
        elif self.kind == "open":
            # Decorative waveform: no measured values or axes.
            for offset in (-.1, .1):
                line(((width * .1, height * (.5 + offset)),
                      (width * .9, height * (.5 + offset))), "#39566B")
            points = []
            samples = max(140, min(round(width * 2), 4096))
            for step in range(samples):
                x = step / (samples - 1)
                y = .64 - .34 * math.exp(-((x - .37) / .11) ** 2) + .05 * math.sin(x * 24)
                points.append((width * (.08 + .84 * x), height * y))
            line(points, "#9EDBD7", 3)
            oval((width * .34 - 5, height * .31 - 5,
                  width * .34 + 5, height * .31 + 5), ACCENT)
        elif self.kind in ("research", "notes"):
            cx, cy = width * .5, height * .46
            paper_w, paper_h = min(width * .36, 80), height * .58
            for shift, fill in ((14, "#5A4F85"), (7, "#8E83B6"), (0, "#E2DCF5")):
                rectangle((cx - paper_w / 2 + shift, cy - paper_h / 2 - shift * .4,
                           cx + paper_w / 2 + shift, cy + paper_h / 2 - shift * .4), fill)
            for index, fraction in enumerate((.68, .83, .58, .74)):
                y = cy - paper_h * .24 + index * paper_h * .16
                line(((cx - paper_w * .32, y),
                      (cx - paper_w * .32 + paper_w * fraction * .7, y)), "#8E82AB", 2)
        else:
            cx, cy, radius = width * .53, height * .46, min(width * .22, height * .3)
            points = [(cx + radius * math.cos(n * math.tau / 6),
                       cy + radius * math.sin(n * math.tau / 6)) for n in range(6)]
            for index, (x, y) in enumerate(points):
                line(((x, y), points[(index + 1) % 6]), "#DC939C", 2)
                outer_x, outer_y = cx + (x - cx) * 1.45, cy + (y - cy) * 1.45
                line(((x, y), (outer_x, outer_y)), "#AD7383", 2)
                oval((outer_x - 3, outer_y - 3, outer_x + 3, outer_y + 3), "#B67586")
            for index, (x, y) in enumerate(points):
                radius = 6 if index % 2 == 0 else 4
                oval((x - radius, y - radius, x + radius, y + radius),
                     "#F4BFAD" if index % 2 == 0 else "#DF8F9B")
        return image.resize((display_width, display_height), Image.Resampling.LANCZOS)

    def _hero_labels(self, width: int):
        family, body = self.fonts.heading[0], self.fonts.body[0]
        self.create_text(self._pixels(22), self._pixels(23), text="THE RESEARCH DESK", anchor="w", fill="#A7C9B3",
                         font=(body, -self._pixels(10), "bold"), tags=("label",))
        self.create_text(self._pixels(22), self._pixels(62), text="From literature to new ideas.", anchor="w", fill=TEXT,
                         width=max(width - self._pixels(80), self._pixels(200)),
                         font=(family, -self._pixels(28), "bold"), tags=("label",))
        self.create_text(self._pixels(22), self._pixels(105), text="Review scientific sources and build a lasting record of research ideas.",
                         anchor="w", width=max(width - self._pixels(42), self._pixels(100)),
                         fill="#C4D3C9", font=(body, -self._pixels(14)),
                         tags=("label",))

    def destroy(self):
        self._disposed = True
        self.unbind("<Configure>")
        ctk.ScalingTracker.remove_widget(self._set_scaling, self)
        if self._redraw_id is not None:
            self.after_cancel(self._redraw_id)
            self._redraw_id = None
        super().destroy()
        self._photo = None


class LibraryRail(ctk.CTkFrame):
    """Compact page navigation; selecting programmatically never re-enters it."""

    PAGES = (
        ("run", "01", "Workbench"),
        ("research", "02", "Literature"),
        ("figures", "03", "Visuals"),
        ("runs", "04", "Experiments"),
        ("notes", "05", "Ideas & notes"),
        ("tools", "06", "Toolbox"),
    )

    def __init__(self, parent, fonts, on_select: Callable[[str], None],
                 on_open: Callable[[], None], on_settings: Callable[[], None], width: int = 220):
        super().__init__(parent, width=width, corner_radius=12, fg_color=SURFACE)
        self.grid_propagate(False)
        self.pack_propagate(False)
        self._on_select = on_select
        self.selected = "run"
        self.buttons: dict[str, ctk.CTkButton] = {}
        brand = ctk.CTkFrame(self, fg_color="transparent")
        brand.pack(fill="x", padx=18, pady=(23, 25))
        symbol = tk.Canvas(brand, width=30, height=30, highlightthickness=0, background=SURFACE)
        symbol.pack(side="left", padx=(0, 9))
        symbol.create_line(7, 21, 15, 8, 24, 20, 7, 21, fill=ACCENT, width=2)
        for x, y in ((7, 21), (15, 8), (24, 20)):
            symbol.create_oval(x - 3, y - 3, x + 3, y + 3, fill=ACCENT, outline="")
        ctk.CTkLabel(brand, text="STUDIO", font=_font(fonts, 18, heading=True, bold=True),
                     text_color=TEXT, anchor="w").pack(side="left")
        ctk.CTkLabel(self, text="WORKSPACES", text_color=MUTED, anchor="w",
                     font=_font(fonts, 11, bold=True)).pack(fill="x", padx=22, pady=(0, 11))
        for key, icon, label in self.PAGES:
            button = ctk.CTkButton(
                self, text=f"{icon}   {label}", anchor="w", height=45,
                corner_radius=8, border_width=0, fg_color="transparent", hover_color=P.hover,
                text_color=MUTED, font=_font(fonts),
                command=lambda selected=key: self._choose(selected),
            )
            button.pack(fill="x", padx=10, pady=2)
            self.buttons[key] = button
        self.settings_button = ctk.CTkButton(
            self, text="Model setup", anchor="w", fg_color="transparent", hover_color=P.hover,
            text_color=MUTED, height=38, font=_font(fonts, 13), command=on_settings,
        )
        self.settings_button.pack(side="bottom", fill="x", padx=14, pady=(0, 16))
        self.open_button = ctk.CTkButton(
            self, text="Import a trace", height=40, corner_radius=7,
            fg_color=P.sunken, hover_color=P.hover, text_color=TEXT,
            font=_font(fonts, 14, bold=True), command=on_open,
        )
        self.open_button.pack(side="bottom", fill="x", padx=16, pady=(18, 8))
        self.select("run")

    def _choose(self, key: str):
        self.select(key)
        self._on_select(key)

    def select(self, key: str):
        if key not in self.buttons:
            raise ValueError(f"Unknown library page: {key}")
        self.selected = key
        for candidate, button in self.buttons.items():
            button.configure(border_width=1 if candidate == key else 0, border_color=P.rule,
                             fg_color=RAISED if candidate == key else "transparent",
                             text_color=TEXT if candidate == key else MUTED)


class _ActionCard(ctk.CTkFrame):
    """An independent research task, presented as a compact workstream."""
    def __init__(self, parent, fonts, kind, title, description, action, command):
        super().__init__(parent, corner_radius=8, fg_color=CARD, border_width=1, border_color=P.rule)
        self.grid_columnconfigure(1, weight=1)
        self.art = _Artwork(self, kind, width=100, height=74, fonts=fonts)
        self.art.grid(row=0, column=0, rowspan=3, sticky="ns", padx=(14, 18), pady=14)
        self.title = ctk.CTkLabel(self, text=title, text_color=TEXT, anchor="w", justify="left",
                                 font=_font(fonts, 21, heading=True, bold=True))
        self.title.grid(row=0, column=1, sticky="ew", padx=(0, 14), pady=(12, 0))
        self.description = ctk.CTkLabel(self, text=description, text_color=MUTED, anchor="w", justify="left",
                                       font=_font(fonts, 13), wraplength=500)
        self.description.grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=(0, 6))
        self.button = ctk.CTkButton(self, text=action + "  →", width=145, height=30, corner_radius=6,
                                    fg_color=ACCENT if kind == "research" else P.sunken,
                                    hover_color=P.accent_hover if kind == "research" else P.hover,
                                    text_color=P.accent_ink if kind == "research" else TEXT,
                                    font=_font(fonts, 12, bold=True), command=command)
        self.button.grid(row=2, column=1, sticky="w", pady=(0, 12))
        for widget in (self.art, self.title, self.description):
            widget.bind("<Button-1>", lambda _event: command(), add="+")
        self.bind("<Configure>", self._resize_text, add="+")

    def _resize_text(self, event):
        if event.widget is self:
            width = max(event.width - 164, 130)
            self.title.configure(wraplength=width)
            self.description.configure(wraplength=width)


class HomePanel(ctk.CTkFrame):
    """Research actions and factual entry points into the local library."""

    def __init__(self, parent, fonts, on_open: Callable[[], None], on_research: Callable[[], None],
                 on_runs: Callable[[], None], on_notes: Callable[[], None]):
        super().__init__(parent, corner_radius=0, fg_color=SURFACE)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.content = ctk.CTkScrollableFrame(
            self, corner_radius=0, fg_color=SURFACE,
            scrollbar_fg_color=SURFACE, scrollbar_button_color=P.rule,
            scrollbar_button_hover_color=P.hover,
        )
        self.content.grid(row=0, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        hero_frame = ctk.CTkFrame(self.content, corner_radius=12, fg_color="#223A50")
        hero_frame.grid(row=0, column=0, sticky="ew", padx=22, pady=(20, 18))
        self.hero = _Artwork(hero_frame, "hero", height=155, fonts=fonts)
        self.hero.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.content, text="Choose a workspace", text_color=TEXT, anchor="w",
                     font=_font(fonts, 24, heading=True, bold=True)).grid(
                         row=1, column=0, sticky="ew", padx=24, pady=(0, 12))
        self.card_grid = ctk.CTkFrame(self.content, fg_color="transparent")
        self.card_grid.grid(row=2, column=0, sticky="ew", padx=22)
        definitions = (
            ("research", "Review the literature", "Find papers, read sources and compare the evidence behind a question.", "Find sources", on_research),
            ("notes", "Keep research ideas", "Revisit saved ideas, hypotheses and observations in your research notebook.", "Open ideas & notes", on_notes),
            ("open", "Open an experiment", "Inspect a trace, review the decisions, and follow the measurement.", "Open a file", on_open),
            ("runs", "Revisit an experiment", "Compare previous work and keep the context behind a result.", "Browse library", on_runs),
        )
        self.cards: dict[str, _ActionCard] = {}
        self.action_buttons: dict[str, ctk.CTkButton] = {}
        for key, title, description, action, callback in definitions:
            card = _ActionCard(self.card_grid, fonts, key, title, description, action, callback)
            self.cards[key] = card
            self.action_buttons[key] = card.button
        self._columns = 0
        self.card_grid.bind("<Configure>", self._arrange_cards, add="+")
        shelf = ctk.CTkFrame(self.content, fg_color="transparent")
        shelf.grid(row=3, column=0, sticky="ew", padx=22, pady=(19, 12))
        shelf.grid_columnconfigure((0, 1), weight=1, uniform="library")
        self.runs_button = ctk.CTkButton(
            shelf, text="Past experiments  →", anchor="w", height=42, corner_radius=8,
            fg_color=P.sunken, hover_color=P.hover, text_color=TEXT,
            font=_font(fonts, 14), command=on_runs,
        )
        self.runs_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.notes_button = ctk.CTkButton(
            shelf, text="Research ideas & notes  →", anchor="w", height=42, corner_radius=8,
            fg_color=P.sunken, hover_color=P.hover, text_color=TEXT,
            font=_font(fonts, 14), command=on_notes,
        )
        self.notes_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.library_counts: dict[str, int | None] = {"runs": None, "notes": None}

    def _arrange_cards(self, event=None):
        if self._columns == 1:
            return
        self.card_grid.grid_columnconfigure(0, weight=1)
        for index, card in enumerate(self.cards.values()):
            card.grid(row=index, column=0, sticky="ew", pady=(0, 10))
        self._columns = 1

    def set_library(self, runs: int, notes: int):
        for value in (runs, notes):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Library counts must be non-negative integers.")
        self.library_counts = {"runs": runs, "notes": notes}
        self.runs_button.configure(text=f"{runs:,} experiment{'s' if runs != 1 else ''} in your library  →")
        self.notes_button.configure(text=f"{notes:,} saved lab note{'s' if notes != 1 else ''}  →")


Launchpad = HomePanel

__all__ = ["HomePanel", "Launchpad", "LibraryRail"]
