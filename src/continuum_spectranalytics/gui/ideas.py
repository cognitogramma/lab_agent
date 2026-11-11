"""Capture research ideas directly, alongside the searchable notebook."""

from __future__ import annotations

from collections.abc import Callable

import customtkinter as ctk

from continuum_spectranalytics import APP_NAME
from continuum_spectranalytics.gui.theme import Fonts, PALETTE as P
from continuum_spectranalytics.gui.widgets import SearchPanel


class IdeasPanel(SearchPanel):
    """The existing notebook search plus an offline idea composer."""

    def __init__(self, parent, fonts: Fonts, run: Callable[[str], str],
                 save: Callable[[dict], None]):
        super().__init__(parent, fonts, "Find saved research ideas, observations and conclusions.", run)
        self._fonts = fonts
        self._save = save
        self.composer: IdeaComposer | None = None
        self.new_button = ctk.CTkButton(
            self, text="New research idea", height=34, corner_radius=7,
            fg_color=P.accent, hover_color=P.accent_hover, text_color=P.accent_ink,
            font=(fonts.body[0], 14, "bold"), command=self.new_idea,
        )
        self.new_button.pack(fill="x", padx=10, pady=(10, 0), before=self.query.master)

    def new_idea(self) -> None:
        if self.composer is not None and self.composer.winfo_exists():
            self.composer.deiconify()
            self.composer.lift()
            return
        self.composer = IdeaComposer(self, self._fonts, self._save, self._saved)

    def _saved(self) -> None:
        self.query.delete(0, "end")
        self.search()


class IdeaComposer(ctk.CTkToplevel):
    """A draft is cleared only after its save callback succeeds."""

    def __init__(self, parent, fonts: Fonts, save: Callable[[dict], None], on_saved: Callable[[], None]):
        super().__init__(parent)
        self.title(f"New research idea — {APP_NAME}")
        self.configure(fg_color=P.paper)
        self.transient(parent.winfo_toplevel())
        self._save = save
        self._on_saved = on_saved
        scale = ctk.ScalingTracker.get_window_scaling(self)
        width = min(580, max(420, int(self.winfo_screenwidth() / scale) - 80))
        height = min(620, max(420, int(self.winfo_screenheight() / scale) - 100))
        self.geometry(f"{width}x{height}")
        self.minsize(min(420, width), min(420, height))
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

        form = ctk.CTkScrollableFrame(self, fg_color=P.paper, corner_radius=0)
        form.pack(fill="both", expand=True, padx=18, pady=18)
        form.columnconfigure(0, weight=1)
        family = fonts.body[0]

        def label(text, row):
            ctk.CTkLabel(form, text=text, anchor="w", text_color=P.ink,
                         font=(family, 14)).grid(row=row, column=0, sticky="ew", pady=(10, 4))

        def entry(row):
            item = ctk.CTkEntry(form, height=36, font=(family, 15), fg_color=P.sunken,
                                text_color=P.ink, border_color=P.rule)
            item.grid(row=row, column=0, sticky="ew")
            return item

        label("Title", 0)
        self.idea_title = entry(1)
        label("Idea or hypothesis", 2)
        self.idea_body = ctk.CTkTextbox(form, height=150, wrap="word", font=(family, 16),
                                        fg_color=P.sunken, text_color=P.ink, border_width=1,
                                        border_color=P.rule)
        self.idea_body.grid(row=3, column=0, sticky="ew")
        label("Source URLs (optional; one per line)", 4)
        self.sources = ctk.CTkTextbox(form, height=60, wrap="word", font=(family, 14),
                                      fg_color=P.sunken, text_color=P.ink, border_width=1,
                                      border_color=P.rule)
        self.sources.grid(row=5, column=0, sticky="ew")
        label("Next step (optional)", 6)
        self.next_step = entry(7)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=22, pady=(0, 18))
        self.status = ctk.CTkLabel(footer, text="Ideas are stored locally as unverified hypotheses.",
                                    text_color=P.quiet, font=(family, 13), anchor="w",
                                    justify="left", wraplength=480)
        self.status.pack(fill="x", pady=(0, 10))
        footer.bind("<Configure>", lambda event: self.status.configure(wraplength=max(200, event.width)))
        self.save_button = ctk.CTkButton(footer, text="Save research idea", height=36,
                                         fg_color=P.accent, hover_color=P.accent_hover,
                                         text_color=P.accent_ink, font=(family, 14, "bold"),
                                         command=self.save)
        self.save_button.pack(side="right")
        ctk.CTkButton(footer, text="Keep draft", height=36, width=100,
                      fg_color=P.sunken, hover_color=P.hover, text_color=P.ink,
                      font=(family, 14), command=self.withdraw).pack(side="left")
        self.after(100, self.idea_title.focus_set)

    def save(self) -> None:
        title = self.idea_title.get().strip()
        idea = self.idea_body.get("1.0", "end-1c").strip()
        if not title or not idea:
            self.status.configure(text="Add a title and an idea or hypothesis before saving.", text_color=P.refused)
            return
        arguments = {
            "title": title,
            "idea": idea,
            "source_urls": self.sources.get("1.0", "end-1c").split(),
            "next_step": self.next_step.get().strip(),
        }
        try:
            self._save(arguments)
        except (OSError, ValueError) as error:
            self.status.configure(text=f"Could not save: {error}", text_color=P.refused)
            return
        self._on_saved()
        self.destroy()


__all__ = ["IdeasPanel"]
