"""An experiment-aware literature workspace with a comfortable source reader."""

from __future__ import annotations

import json
import queue
import re
import threading
import tkinter as tk
import tkinter.font as tkfont
from collections import OrderedDict
from collections.abc import Callable
from tkinter import ttk
from urllib.parse import urlsplit

import customtkinter as ctk

from continuum_spectranalytics.gui.links import open_web_url, safe_web_url
from continuum_spectranalytics.gui.research import ResearchPanel
from continuum_spectranalytics.gui.theme import Fonts, PALETTE as P
from continuum_spectranalytics.gui.widgets import _HEADING, _LIST_ITEM, _inline


class LiteraturePanel(ResearchPanel):
    """Reuse the bounded search engine, with a separate reading-oriented UI.

    Context, saved-library access, page reads and explicit saves run in a
    bounded background queue. Workers never access a Tk variable or widget.
    Reading is invalidated whenever selection or experiment context changes.
    """

    def __init__(self, parent, fonts: Fonts, search: Callable[[dict], str], *,
                 context: Callable[[], dict], read: Callable[[dict], str],
                 load_saved: Callable[[], list[dict]], save_source: Callable[[dict], dict],
                 on_review: Callable[[str], None] | None = None):
        # ResearchPanel's search/cancellation engine is independent of its UI.
        ttk.Frame.__init__(self, parent, style="Paper.TFrame")
        self.fonts = fonts
        self._search = search
        self._context = context
        self._read = read
        self._load_saved = load_saved
        self._save_source = save_source
        self._on_review = on_review
        self._responses = queue.Queue()
        self._generation = self._workers = 0
        self._pending = None
        self._closed = self.busy = False
        self._result_query = ""
        self._wide_layout = None
        self.results: list[dict] = []
        self._found: list[dict] = []
        self._memory: list[dict] = []
        self._saved: list[dict] = []
        self._saved_load_error = False
        self.experiment: dict = {}
        self.view = "Found"
        self.source = tk.StringVar(self, value="Web")
        self.period = tk.StringVar(self, value="Any time")
        self._io_queue = queue.Queue()
        self._io_active = 0
        self._io_pending: OrderedDict[str, tuple] = OrderedDict()
        self._io_tokens = {name: 0 for name in ("context", "saved", "read", "save")}
        self._reading = self._saving = False
        self._selected_key = None
        self._page_cache: OrderedDict[str, dict] = OrderedDict()
        self._reading_focus = False
        self.reader_font_size = 15
        self._reader_font = tkfont.Font(self, family=fonts.body[0], size=self.reader_font_size)
        self._build()
        self.bind("<Configure>", self._resize, add="+")
        self.bind("<Destroy>", self._destroyed, add="+")
        self._poll_id = self.after(self.POLL_MS, self._poll)
        self.refresh_context()
        self.refresh_saved()

    def _px(self, value):
        try:
            scale = ctk.ScalingTracker.get_widget_scaling(self)
        except KeyError:
            scale = self.winfo_fpixels("1i") / 96
        return max(1, round(value * scale))

    def _button(self, parent, text, command, *, primary=False, width=105):
        return ctk.CTkButton(parent, text=text, command=command, width=width, height=30,
                             corner_radius=7, font=(self.fonts.body[0], 14),
                             fg_color=P.accent if primary else P.sunken,
                             hover_color=P.accent_hover if primary else P.hover,
                             text_color=P.accent_ink if primary else P.ink,
                             text_color_disabled=P.disabled_ink, bg_color=P.paper)

    def _build(self):
        family = self.fonts.body[0]
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)
        header = ttk.Frame(self, style="Paper.TFrame", padding=(18, 13, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Literature review", style="PaperHeading.TLabel").pack(side="left")
        self.context_frame = ttk.Frame(self, style="Paper.TFrame", padding=(18, 0, 18, 10))
        self.context_frame.grid(row=1, column=0, sticky="ew")
        self.context_frame.columnconfigure(0, weight=1)
        self.context_label = tk.Label(self.context_frame, text="Loading experiment context…", anchor="w",
                                      justify="left", wraplength=700, background=P.paper,
                                      foreground=P.quiet, font=self.fonts.small)
        self.context_label.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.use_experiment_button = self._button(self.context_frame, "Use experiment", self.use_experiment, width=130)
        self.use_experiment_button.grid(row=0, column=1, sticky="e")
        self.use_experiment_button.configure(state="disabled")

        self.controls = ttk.Frame(self, style="Paper.TFrame", padding=(18, 0, 18, 8))
        self.controls.grid(row=2, column=0, sticky="ew")
        self.controls.columnconfigure(0, weight=1)
        self.query = ctk.CTkEntry(self.controls, height=35, corner_radius=8, border_width=1,
                                  border_color=P.rule, fg_color=P.sunken, text_color=P.ink,
                                  font=(family, 16), bg_color=P.paper)
        self.query.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.query.bind("<Return>", self._on_return)
        self.search_button = self._button(self.controls, "Find related papers", self.search, primary=True, width=165)
        self.search_button.grid(row=0, column=1, sticky="e")
        filters = ttk.Frame(self.controls, style="Paper.TFrame")
        filters.grid(row=1, column=0, columnspan=2, sticky="w", pady=(7, 0))
        combo = dict(height=28, corner_radius=7, border_width=0, fg_color=P.sunken, button_color=P.sunken,
                     button_hover_color=P.hover, dropdown_fg_color=P.panel, dropdown_hover_color=P.hover,
                     dropdown_text_color=P.ink, text_color=P.ink, bg_color=P.paper,
                     font=(family, 13), dropdown_font=(family, 13), state="readonly")
        self.source_choice = ctk.CTkComboBox(filters, variable=self.source, values=["Web", "arXiv papers"],
                                            width=140, command=lambda _value: self._source_changed(), **combo)
        self.source_choice.pack(side="left", padx=(0, 8))
        self.period_choice = ctk.CTkComboBox(filters, variable=self.period, values=list(self.PERIODS), width=135, **combo)
        self.period_choice.pack(side="left")
        # Retain the search engine's optional-domain interface without crowding
        # the review workspace with another permanent field.
        self.domain = ctk.CTkEntry(filters)
        tk.Label(filters, text="  + matching lab memory", background=P.paper, foreground=P.quiet,
                 font=self.fonts.small).pack(side="left")
        self.status = tk.Label(self, text="Search a public topic, or use the active experiment’s suggested query.",
                               background=P.paper, foreground=P.quiet, font=self.fonts.small,
                               anchor="w", justify="left", wraplength=850)
        self.status.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 8))

        self.split = ttk.Frame(self, style="Paper.TFrame")
        self.split.grid(row=4, column=0, sticky="nsew", padx=12, pady=(0, 10))
        self.result_frame = ttk.Frame(self.split, style="Paper.TFrame")
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(1, weight=1)
        self.scopes = ctk.CTkSegmentedButton(self.result_frame, values=["Found", "Memory", "Saved"],
                                            command=self.set_view, height=30,
                                            font=(family, 14), fg_color=P.panel,
                                            selected_color=P.selection, selected_hover_color=P.hover,
                                            unselected_color=P.panel, unselected_hover_color=P.hover,
                                            text_color=P.ink)
        self.scopes.set("Found")
        self.scopes.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        style = ttk.Style(self)
        style.configure("Literature.Treeview", rowheight=self._px(37), font=self.fonts.small)
        style.map("Literature.Treeview", background=[("selected", P.selection)], foreground=[("selected", P.ink)])
        self.listing = ttk.Treeview(self.result_frame, columns=("source", "date"), show="tree headings",
                                    selectmode="browse", style="Literature.Treeview", height=5)
        for column, label, width in (("#0", "Original title", 180), ("source", "Origin", 62), ("date", "Date", 78)):
            self.listing.heading(column, text=label, anchor="w")
            self.listing.column(column, width=self._px(width), minwidth=self._px(45), stretch=column == "#0")
        self.listing.grid(row=1, column=0, sticky="nsew")
        scrollbar = ctk.CTkScrollbar(self.result_frame, orientation="vertical", command=self.listing.yview,
                                     fg_color=P.paper, button_color=P.rule, button_hover_color=P.hover, width=10)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.listing.configure(yscrollcommand=scrollbar.set)
        self.listing.bind("<<TreeviewSelect>>", self._select)
        self.listing.bind("<Double-1>", lambda _event: self.read_page())
        self.list_message = tk.Label(self.result_frame, text="Found papers, lab memory and saved sources stay separate.",
                                     background=P.paper, foreground=P.quiet, font=self.fonts.small,
                                     justify="left", anchor="w", wraplength=320)
        self.list_message.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.reader_frame = ttk.Frame(self.split, style="Paper.TFrame")
        self.reader_frame.columnconfigure(0, weight=1)
        self.reader_frame.rowconfigure(2, weight=1)
        tools = ttk.Frame(self.reader_frame, style="Paper.TFrame")
        tools.grid(row=0, column=0, sticky="ew", padx=12, pady=(0, 7))
        tk.Label(tools, text="Reader", background=P.paper, foreground=P.quiet,
                 font=self.fonts.small).pack(side="left")
        self.focus_button = self._button(tools, "Focus reading", self.toggle_reading_focus, width=115)
        self.focus_button.pack(side="right", padx=(6, 0))
        self.font_up = self._button(tools, "A+", lambda: self.adjust_font(1), width=35)
        self.font_up.pack(side="right", padx=(4, 0))
        self.font_down = self._button(tools, "A−", lambda: self.adjust_font(-1), width=35)
        self.font_down.pack(side="right")
        actions = ttk.Frame(self.reader_frame, style="Paper.TFrame")
        self.reader_actions = actions
        actions.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        self.read_button = self._button(actions, "Read page", self.read_page, primary=True, width=95)
        self.open_button = self._button(actions, "Open original ↗", self.open_source, width=120)
        self.save_button = self._button(actions, "Save paper", self.save_paper, width=95)
        self.copy_button = self._button(actions, "Copy citation", self.copy_citation, width=105)
        self.review_button = self._button(actions, "Ask assistant", self.ask_assistant, width=105)
        self._action_buttons = (self.read_button, self.open_button, self.save_button, self.copy_button, self.review_button)
        for index, button in enumerate(self._action_buttons):
            button.grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 5), pady=(0, 5))
            button.configure(state="disabled")
        self.reader_frame.bind("<Configure>", self._reader_actions_layout)
        self.reader_area = tk.Frame(self.reader_frame, background=P.paper)
        self.reader_area.grid(row=2, column=0, sticky="nsew")
        self.reader_area.bind("<Configure>", lambda _event: self._wrap_reader())
        self.reader_column = tk.Frame(self.reader_area, background=P.paper)
        self.reader_column.place(relx=.5, rely=0, anchor="n", relheight=1, width=500)
        self.reader_column.columnconfigure(0, weight=1)
        self.reader_column.rowconfigure(0, weight=1)
        self.detail = tk.Text(self.reader_column, wrap="word", font=self._reader_font, width=78, height=15,
                              background=P.paper, foreground=P.ink, borderwidth=0, highlightthickness=0,
                              padx=self._px(16), pady=self._px(12), spacing1=self._px(4),
                              spacing2=self._px(4), spacing3=self._px(12), cursor="arrow",
                              selectbackground=P.selection, selectforeground=P.ink)
        self.detail.grid(row=0, column=0, sticky="nsew")
        bar = ctk.CTkScrollbar(self.reader_column, orientation="vertical", command=self.detail.yview,
                               fg_color=P.paper, button_color=P.rule, button_hover_color=P.hover, width=10)
        bar.grid(row=0, column=1, sticky="ns")
        self.detail.configure(yscrollcommand=bar.set)
        self.detail.tag_configure("title", font=(family, 20, "bold"), spacing3=self._px(12))
        self.detail.tag_configure("metadata", font=(family, 12), foreground=P.quiet, spacing3=self._px(12))
        self.detail.tag_configure("section", font=(family, 13, "bold"), foreground=P.accent,
                                  spacing1=self._px(12), spacing3=self._px(7))
        self.detail.tag_configure("link", font=(family, 12), foreground=P.accent, underline=True)
        self.detail.tag_configure("review_heading", font=(family, 18, "bold"), spacing1=self._px(12))
        self.detail.tag_configure("md_strong", font=(family, 15, "bold"))
        self.detail.tag_configure("md_em", font=(family, 15, "italic"))
        self.detail.tag_configure("md_code", font=self.fonts.mono, foreground=P.support)
        self.detail.tag_bind("link", "<Button-1>", lambda _event: self.open_source())
        self.detail.tag_bind("link", "<Enter>", lambda _event: self.detail.configure(cursor="hand2"))
        self.detail.tag_bind("link", "<Leave>", lambda _event: self.detail.configure(cursor="arrow"))
        self.detail.bind("<Tab>", self._next_focus)
        self.detail.bind("<Shift-Tab>", self._previous_focus)
        self._render_message("Read the evidence in context.", "Find related papers, inspect why they match this experiment, and read a source before asking for a review.")
        self._set_wide_layout(True)

    def _resize(self, event):
        if self._closed or event.widget is not self:
            return
        self.status.configure(wraplength=max(120, event.width - self._px(36)))
        self.context_label.configure(wraplength=max(120, event.width - self._px(200)))
        self._set_wide_layout(event.width >= self._px(850))
        self._wrap_reader()

    def _set_wide_layout(self, wide):
        if self._wide_layout == wide and not self._reading_focus:
            return
        self._wide_layout = wide
        for row in (0, 1):
            self.split.rowconfigure(row, weight=0)
        for column in (0, 1):
            self.split.columnconfigure(column, weight=0)
        if self._reading_focus:
            self.result_frame.grid_remove()
            self.reader_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=0)
            self.split.rowconfigure(0, weight=1)
            self.split.columnconfigure(0, weight=1)
        elif wide:
            self.result_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
            self.reader_frame.grid(row=0, column=1, columnspan=1, sticky="nsew", pady=0)
            self.split.columnconfigure(0, weight=1, minsize=self._px(270))
            self.split.columnconfigure(1, weight=3)
            self.split.rowconfigure(0, weight=1)
        else:
            self.result_frame.grid(row=0, column=0, sticky="nsew", padx=0)
            self.reader_frame.grid(row=1, column=0, columnspan=1, sticky="nsew", pady=(12, 0))
            self.split.columnconfigure(0, weight=1, minsize=0)
            self.split.rowconfigure(1, weight=3)
        self._wrap_reader()

    def _wrap_reader(self):
        if self._closed:
            return
        maximum = self._reader_font.measure("0" * 78) + self._px(40)
        available = max(1, self.reader_area.winfo_width())
        self.reader_column.place_configure(width=min(available, maximum))

    def _reader_actions_layout(self, event):
        if self._closed or event.widget is not self.reader_frame:
            return
        columns = 5 if event.width >= self._px(620) else (3 if event.width >= self._px(355) else 2)
        for index, button in enumerate(self._action_buttons):
            button.grid_configure(row=index // columns, column=index % columns)

    def adjust_font(self, delta):
        self.reader_font_size = max(12, min(24, self.reader_font_size + delta))
        self._reader_font.configure(size=self.reader_font_size)
        self.detail.tag_configure("md_strong", font=(self.fonts.body[0], self.reader_font_size, "bold"))
        self.detail.tag_configure("md_em", font=(self.fonts.body[0], self.reader_font_size, "italic"))
        self.detail.tag_configure("title", font=(self.fonts.body[0], max(20, self.reader_font_size + 4), "bold"))
        self.detail.tag_configure("review_heading", font=(self.fonts.body[0], self.reader_font_size + 3, "bold"))
        self.detail.configure(spacing2=self._px(max(4, self.reader_font_size / 3)))
        self._wrap_reader()

    def toggle_reading_focus(self):
        self._reading_focus = not self._reading_focus
        self.focus_button.configure(text="Exit focus" if self._reading_focus else "Focus reading")
        for widget in (self.context_frame, self.controls):
            widget.grid_remove() if self._reading_focus else widget.grid()
        wide = self._wide_layout
        self._wide_layout = None
        self._set_wide_layout(wide)

    def refresh_context(self):
        if not self._closed:
            self._submit_io("context", self._context)

    def refresh_saved(self):
        if not self._closed:
            self._saved_load_error = False
            self._submit_io("saved", self._load_saved)

    def _set_experiment(self, value):
        self.experiment = value if isinstance(value, dict) else {}
        summary = str(self.experiment.get("summary") or "No active experiment. Search a public topic of your choice.")
        run_id = self.experiment.get("run_id")
        self.context_label.configure(text=(f"Experiment {run_id} · " if run_id else "") + summary[:500])
        self.use_experiment_button.configure(state="normal" if self.experiment.get("suggested_query") else "disabled")

    def use_experiment(self):
        suggested = self.experiment.get("suggested_query")
        if isinstance(suggested, str) and suggested.strip():
            self.query.delete(0, "end")
            self.query.insert(0, suggested.strip()[:500])
            self.query.focus_set()

    def context_changed(self):
        if self._closed:
            return
        self.cancel()
        self._generation += 1
        self._pending = None
        self._io_tokens["context"] += 1
        self._io_pending.pop("context", None)
        self._invalidate_read()
        self.experiment = {}
        self._found = self._memory = []
        self._set_experiment({})
        self.set_view("Found")
        self._render_message("Experiment changed.", "Use the updated experiment context or enter a new research question.")

    invalidate_context = context_changed

    def search(self):
        if self._closed:
            return
        if self.query.get().strip():
            self._invalidate_read()
            self._found = []
            self._memory = []
            self.view = "Found"
            self.scopes.set("Found")
        super().search()
        if self.busy:
            self.search_button.configure(text="Cancel search")

    run = search

    def cancel(self):
        super().cancel()
        if not self._closed:
            self.search_button.configure(text="Find related papers")

    def _clear_results(self):
        self._invalidate_read()
        super()._clear_results()
        self._selected_key = None
        for button in (self.read_button, self.save_button, self.review_button):
            button.configure(state="disabled")

    def _show_empty(self, title, message):
        self._render_message(title, message)

    def _show_sources(self):
        pass  # the reader and library list remain visible throughout a search

    def _show_detail(self, message):
        self._render_message("Literature review", str(message))

    def _finish(self, payload):
        previous_view = self.view
        was_searching = self.busy
        self.busy = False
        self._enable_filters()
        self.search_button.configure(text="Find related papers", command=self.search)
        if isinstance(payload.get("experiment"), dict):
            self._set_experiment(payload["experiment"])
        self._found = [dict(item) for item in payload.get("results", []) if isinstance(item, dict)] if isinstance(payload.get("results", []), list) else []
        query = str(payload.get("query") or self._result_query)
        run_id = self.experiment.get("run_id")
        for item in self._found:
            item["query"] = query
            item["run_id"] = run_id
        self._memory = [dict(item, origin="memory") for item in payload.get("memory", []) if isinstance(item, dict)] if isinstance(payload.get("memory", []), list) else []
        warning = payload.get("warnings") or payload.get("warning") or []
        warning = " · ".join(map(str, warning)) if isinstance(warning, list) else str(warning)
        message = f"{len(self._found)} sources · {len(self._memory)} matching lab memories."
        if payload.get("error"):
            message += " " + str(payload["error"])
        if warning:
            message += " " + warning
        self.status.configure(text=message, foreground=P.refused if payload.get("error") else P.quiet)
        if previous_view == "Saved" and was_searching:
            # A finished lookup updates the found library without pulling the
            # reader away from a saved paper the person chose in the meantime.
            return
        self.set_view("Found" if self._found or not self._memory else "Memory")

    def set_view(self, view):
        if view not in {"Found", "Memory", "Saved"}:
            return
        if view == "Saved" and self._saved_load_error:
            self.refresh_saved()
        self._invalidate_read()
        self.view = view
        self.scopes.set(view)
        self.results = {"Found": self._found, "Memory": self._memory, "Saved": self._saved}[view]
        self._selected_key = None
        self.listing.delete(*self.listing.get_children())
        for index, item in enumerate(self.results):
            origin = item.get("origin") or ("saved" if view == "Saved" else "web")
            self.listing.insert("", "end", iid=str(index), text=item.get("title") or item.get("kind") or "Untitled source",
                                values=(origin, item.get("published") or item.get("updated_at") or "—"))
        self.list_message.configure(text=f"{len(self.results)} {view.lower()} item(s). " +
                                    ("Lab memories are local notes and run records, not publications." if view == "Memory" else "Select a source to inspect its relevance and read it."))
        if self.results:
            self.listing.selection_set("0")
            self._select()
        else:
            self._selected_key = None
            for button in (self.read_button, self.open_button, self.save_button, self.copy_button, self.review_button):
                button.configure(state="disabled")
            self._render_message(f"No {view.lower()} items yet.", "Try a related-paper search, check matching lab memory, or save a useful source for later reading.")

    def _select(self, _event=None):
        item = self.current()
        if item is None:
            return
        key = item.get("url") or f'memory:{item.get("id", item.get("title", ""))}'
        if key == self._selected_key:
            return
        self._invalidate_read()
        self._selected_key = key
        url = safe_web_url(item.get("url"))
        memory = self._is_memory_item(item)
        self.read_button.configure(state="normal" if url and not memory else "disabled")
        self.open_button.configure(state="normal" if url else "disabled")
        self.save_button.configure(state="normal" if url and not memory and not self._saving else "disabled")
        self.copy_button.configure(state="normal")
        self.review_button.configure(state="normal" if url and not memory and self._on_review else "disabled")
        page = self._page_cache.get(url) if url else None
        if page is None and item.get("text"):
            page = item
        self._render_source(item, page)

    def _render_message(self, title, message):
        self._clear_review_links()
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", str(title) + "\n\n", "title")
        self.detail.insert("end", str(message))
        self.detail.configure(state="disabled")
        self.detail.yview_moveto(0)

    def _render_source(self, item, page=None):
        self._clear_review_links()
        text = self.detail
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("end", str(item.get("title") or "Untitled source") + "\n", "title")
        authors = item.get("authors") or []
        authors = ", ".join(map(str, authors)) if isinstance(authors, list) else str(authors)
        metadata = [str(value) for value in (authors, item.get("origin"), item.get("domain"),
                    item.get("published") or item.get("created_at"), item.get("author")) if value]
        if metadata:
            text.insert("end", " · ".join(metadata) + "\n\n", "metadata")
        relevance = item.get("relevance") or []
        if relevance:
            text.insert("end", "Why this may be relevant\n", "section")
            text.insert("end", "\n".join("• " + str(value) for value in relevance) + "\n\n")
        if item.get("memory_refs"):
            text.insert("end", "Related lab context\n", "section")
            labels = []
            for ref in item["memory_refs"]:
                labels.append(str(ref.get("title") or " ".join(str(ref[key]) for key in ("kind", "id") if ref.get(key)))
                              if isinstance(ref, dict) else str(ref))
            text.insert("end", ", ".join(label for label in labels if label) + "\n\n", "metadata")
        if self._is_memory_item(item):
            text.insert("end", f'Lab memory · {item.get("kind", "record")}\n', "section")
            text.insert("end", str(item.get("excerpt") or item.get("snippet") or "No excerpt is available.") + "\n\n")
        elif page and page.get("error"):
            text.insert("end", "Page unavailable\n", "section")
            text.insert("end", str(page["error"]) + "\n\nUse Open original to read a PDF, protected page, or unsupported document in your browser.\n\n")
        elif page and page.get("text"):
            text.insert("end", "Extracted page text\n", "section")
            if page.get("truncated"):
                text.insert("end", "This extract is truncated. Open the original for the rest of the source.\n\n", "metadata")
            if page.get("retrieved_at"):
                text.insert("end", "Retrieved " + str(page["retrieved_at"]) + "\n\n", "metadata")
            text.insert("end", str(page["text"]) + "\n\n")
        else:
            text.insert("end", "Abstract / search summary\n", "section")
            text.insert("end", str(item.get("abstract") or item.get("snippet") or "No summary is available.") + "\n\n")
            text.insert("end", "Choose Read page to inspect the source text. A search summary is not a full-paper review.\n\n", "metadata")
        url = safe_web_url((page or {}).get("url") or item.get("url"))
        if url:
            text.insert("end", url, "link")
        text.configure(state="disabled")
        text.yview_moveto(0)

    def _is_memory_item(self, item):
        return self.view == "Memory" or (item.get("kind") in {"note", "run"} and not safe_web_url(item.get("url")))

    def _invalidate_read(self):
        self._io_tokens["read"] += 1
        self._io_pending.pop("read", None)
        self._reading = False
        if hasattr(self, "read_button"):
            self.read_button.configure(text="Read page", command=self.read_page)

    def read_page(self):
        if self._reading:
            self._invalidate_read()
            self.status.configure(text="Reading cancelled. Any page request already sent will finish under its timeout and be ignored.")
            return
        item = self.current()
        url = safe_web_url(item.get("url")) if item else None
        if not url or self._is_memory_item(item):
            return
        self._reading = True
        self.read_button.configure(text="Cancel reading")
        self.status.configure(text="Reading the selected source…", foreground=P.quiet)
        self._submit_io("read", self._read, {"url": url, "max_chars": 24000})

    def save_paper(self):
        item = self.current()
        if not item or self._saving or self._is_memory_item(item) or not safe_web_url(item.get("url")):
            return
        record = dict(item)
        page = self._page_cache.get(item["url"])
        if page and not page.get("error"):
            for key in ("text", "retrieved_at", "sha256", "truncated", "content_type"):
                if key in page:
                    record[key] = page[key]
        self._saving = True
        self.save_button.configure(text="Saving…", state="disabled")
        self._submit_io("save", self._save_source, record)

    def ask_assistant(self):
        item = self.current()
        url = safe_web_url(item.get("url")) if item else None
        if not url or not self._on_review:
            return
        prompt = (f'Review the scientific source "{item.get("title", "")}" at {url} in relation to '
                  f'the active experiment ({self.experiment.get("run_id", "no run selected")}).\n'
                  f'Experiment context: {self.experiment.get("summary", "No experiment context available.")}\n'
                  f'Research question: {self._result_query or self.query.get()}\n'
                  'Read the source, distinguish its claims from local lab memory, explain relevance and limitations, '
                  'and cite the source URLs. Do not alter the experiment.')
        self._on_review(prompt)

    def show_review(self, text):
        self._invalidate_read()
        self.listing.selection_remove(*self.listing.selection())
        self._selected_key = None
        for button in (self.read_button, self.open_button, self.save_button, self.copy_button, self.review_button):
            button.configure(state="disabled")
        self._clear_review_links()
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", "Literature review\n\n", "title")
        fenced = False
        for line in str(text).splitlines():
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                self.detail.insert("end", line + "\n", "md_code")
                continue
            heading = _HEADING.match(line)
            item = _LIST_ITEM.match(line)
            if heading:
                self._insert_review_line(heading.group(2), "review_heading")
            elif item:
                self._insert_review_line("• " + item.group(3), "review_body")
            else:
                self._insert_review_line(line, "review_body")
            self.detail.insert("end", "\n")
        self.detail.configure(state="disabled")
        self.detail.yview_moveto(0)
        if not self._reading_focus:
            self.toggle_reading_focus()

    def _clear_review_links(self):
        for tag in self.detail.tag_names():
            if tag.startswith("review_link_"):
                self.detail.tag_delete(tag)

    def _insert_review_line(self, line, block):
        cursor = 0
        pattern = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
        for match in pattern.finditer(line):
            for body, tags in _inline(line[cursor:match.start()], block):
                self.detail.insert("end", body, tags)
            target = safe_web_url(match.group(2))
            if target:
                tag = "review_link_" + str(len([name for name in self.detail.tag_names() if name.startswith("review_link_")]))
                self.detail.tag_configure(tag, foreground=P.accent, underline=True)
                self.detail.tag_bind(tag, "<Button-1>", lambda _event, url=target: open_web_url(url))
                self.detail.tag_bind(tag, "<Enter>", lambda _event: self.detail.configure(cursor="hand2"))
                self.detail.tag_bind(tag, "<Leave>", lambda _event: self.detail.configure(cursor="arrow"))
                for body, tags in _inline(match.group(1), block, (tag,)):
                    self.detail.insert("end", body, tags)
            else:
                self.detail.insert("end", match.group(1), block)
            cursor = match.end()
        for body, tags in _inline(line[cursor:], block):
            self.detail.insert("end", body, tags)

    def _submit_io(self, kind, callback, argument=None):
        self._io_tokens[kind] += 1
        job = (kind, self._io_tokens[kind], callback, argument)
        if self._io_active < 2:
            self._start_io(job)
        else:
            self._io_pending[kind] = job

    def _start_io(self, job):
        self._io_active += 1
        threading.Thread(target=self._fetch_io, args=(self._io_queue, *job), daemon=True).start()

    @staticmethod
    def _fetch_io(out, kind, token, callback, argument):
        try:
            payload = callback() if argument is None else callback(argument)
            if isinstance(payload, str):
                payload = json.loads(payload)
        except Exception as error:
            payload = {"error": str(error) or type(error).__name__}
        out.put((kind, token, payload))

    def _poll(self):
        super()._poll()
        if self._closed:
            return
        while True:
            try:
                kind, token, payload = self._io_queue.get_nowait()
            except queue.Empty:
                break
            self._io_active -= 1
            if token != self._io_tokens[kind]:
                continue
            self._handle_io(kind, payload)
        while self._io_pending and self._io_active < 2:
            _kind, job = self._io_pending.popitem(last=False)
            self._start_io(job)

    def _handle_io(self, kind, payload):
        if kind == "context":
            if isinstance(payload, dict) and not payload.get("error"):
                self._set_experiment(payload.get("experiment", payload))
            elif isinstance(payload, dict):
                self.context_label.configure(text="Experiment context unavailable: " + str(payload["error"]))
        elif kind == "saved":
            if not isinstance(payload, list):
                self._saved_load_error = True
                error = payload.get("error", "Invalid library response") if isinstance(payload, dict) else str(payload)
                self.status.configure(text="Saved papers could not be loaded: " + str(error) + ". Try reopening Saved after checking local storage.", foreground=P.refused)
                return
            self._saved = [dict(item) for item in payload if isinstance(item, dict)]
            if self.view == "Saved":
                self.set_view("Saved")
        elif kind == "read":
            self._reading = False
            self.read_button.configure(text="Read page")
            item = self.current()
            if item is None:
                return
            if not isinstance(payload, dict):
                payload = {"error": "The source reader returned an invalid response."}
            self._page_cache[item["url"]] = payload
            while len(self._page_cache) > 24:
                self._page_cache.popitem(last=False)
            self._render_source(item, payload)
            self.status.configure(text="Source could not be extracted. Open the original page." if payload.get("error") else
                                  "Source text loaded." + (" The extract is truncated." if payload.get("truncated") else ""),
                                  foreground=P.refused if payload.get("error") else P.quiet)
        elif kind == "save":
            self._saving = False
            selected = self.current()
            can_save = selected and not self._is_memory_item(selected) and safe_web_url(selected.get("url"))
            self.save_button.configure(text="Save paper", state="normal" if can_save else "disabled")
            if not isinstance(payload, dict) or payload.get("error"):
                self.status.configure(text="Paper was not saved: " + str(payload.get("error", "Invalid storage response") if isinstance(payload, dict) else payload),
                                      foreground=P.refused)
                return
            key = payload.get("url") or payload.get("id")
            self._saved = [item for item in self._saved if (item.get("url") or item.get("id")) != key]
            self._saved.insert(0, payload)
            self.status.configure(text="Paper saved to the local library.", foreground=P.support)
            if self.view == "Saved":
                self.set_view("Saved")

    def close(self):
        if self._closed:
            return
        for kind in self._io_tokens:
            self._io_tokens[kind] += 1
        self._io_pending.clear()
        super().close()


__all__ = ["LiteraturePanel"]
