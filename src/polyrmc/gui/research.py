"""Read-only internet research without tying network latency to Tk's event loop."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
import customtkinter as ctk
from collections.abc import Callable
from tkinter import ttk
from urllib.parse import urlsplit

from polyrmc.gui.links import open_web_url, safe_web_url
from polyrmc.gui.theme import PALETTE, Fonts


class ResearchPanel(ttk.Frame):
    """Search, inspect a source, and copy its citation; no model key required.

    Workers only call the backend and write to a queue. Cancellation detaches
    the viewer from a request; an HTTP request already in flight finishes under
    the backend's timeout. At most two run, with only the newest queued query
    retained when someone repeatedly changes their mind.
    """

    PERIODS = {"Any time": "any", "Past day": "d", "Past week": "w", "Past month": "m", "Past year": "y"}
    POLL_MS = 60
    WIDE_AT = 760

    def __init__(self, parent, fonts: Fonts, search: Callable[[dict], str]) -> None:
        super().__init__(parent, style="Paper.TFrame")
        self._search = search
        self._responses: queue.Queue = queue.Queue()
        self._generation = 0
        self._workers = 0
        self._pending: tuple[int, dict] | None = None
        self._closed = False
        self._result_query = ""
        self._wide_layout = False
        self.busy = False
        self.results: list[dict] = []
        self.source = tk.StringVar(self, value="Web")
        self.period = tk.StringVar(self, value="Any time")
        p = PALETTE
        family = fonts.body[0]
        heading_family = fonts.heading[0]
        style = ttk.Style(self)
        style.configure("Research.Treeview", rowheight=36, font=fonts.small)
        style.configure("Research.Treeview.Heading", font=(family, 9, "bold"), padding=(6, 9))
        style.map("Research.Treeview", background=[("selected", p.selection)],
                  foreground=[("selected", p.ink)])

        top = ttk.Frame(self, style="Paper.TFrame", padding=(20, 20, 20, 10))
        top.pack(fill="x")
        top.columnconfigure(0, weight=1)
        tk.Label(top, text="DISCOVER", background=p.paper, foreground=p.accent,
                 font=(family, 9, "bold"), anchor="w").grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Label(top, text="Explore the literature", background=p.paper, foreground=p.ink,
                 font=(heading_family, 20, "bold"), anchor="w").grid(
                     row=1, column=0, columnspan=2, sticky="w", pady=(5, 4))
        self.subtitle = tk.Label(top, text="Find the evidence behind your next idea.", background=p.paper,
                                 foreground=p.quiet, font=fonts.small, anchor="w", justify="left", wraplength=420)
        self.subtitle.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 18))
        tk.Label(top, text="Search a topic, question, or paper", background=p.paper,
                 foreground=p.ink, font=(family, 9, "bold"), anchor="w").grid(
                     row=3, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.query = ctk.CTkEntry(top, height=42, corner_radius=21, border_width=1,
                                  fg_color=p.sunken, bg_color=p.paper, border_color=p.rule,
                                  text_color=p.ink, font=(family, 15))
        self.query.grid(row=4, column=0, sticky="ew")
        self.query.bind("<Return>", self._on_return)
        self.search_button = ctk.CTkButton(top, text="Search", command=self.search,
                                          width=100, height=42, corner_radius=21,
                                          fg_color=p.accent, hover_color=p.accent_hover, bg_color=p.paper,
                                          text_color=p.accent_ink, font=(family, 14, "bold"))
        self.search_button.grid(row=4, column=1, sticky="ns", padx=(8, 0))

        suggestions = tk.Frame(top, background=p.paper)
        suggestions.grid(row=5, column=0, columnspan=2, sticky="w", pady=(9, 14))
        self.suggestion_buttons = []
        for label in ("Polymer scattering", "Virial coefficients"):
            chip = ctk.CTkButton(suggestions, text=label, command=lambda text=label: self._suggest(text),
                                font=(family, 12), width=140, height=28, corner_radius=14,
                                fg_color=p.sunken, hover_color=p.hover, text_color=p.quiet,
                                bg_color=p.paper, border_width=0)
            chip.pack(side="left", padx=(0, 7))
            self.suggestion_buttons.append(chip)

        filter_shell = ctk.CTkFrame(top, fg_color=p.panel, bg_color=p.paper,
                                    corner_radius=14, border_width=0)
        filter_shell.grid(row=6, column=0, columnspan=2, sticky="ew")
        filters = tk.Frame(filter_shell, background=p.panel, padx=12, pady=8)
        filters.pack(fill="both", expand=True, padx=4, pady=4)
        self._filters = filters
        filters.columnconfigure(0, weight=1)
        filters.columnconfigure(1, weight=1)
        for column, label in enumerate(("SOURCE", "PUBLISHED")):
            tk.Label(filters, text=label, background=p.panel, foreground=p.quiet,
                     font=(family, 8, "bold"), anchor="w").grid(
                         row=0, column=column, sticky="w", pady=(0, 5))
        combo_style = dict(width=130, height=34, corner_radius=17, border_width=0,
                           fg_color=p.sunken, button_color=p.sunken, button_hover_color=p.hover,
                           dropdown_fg_color=p.sunken, dropdown_hover_color=p.selection,
                           dropdown_text_color=p.ink, text_color=p.ink, text_color_disabled=p.faint,
                           font=(family, 13), dropdown_font=(family, 13), bg_color=p.panel)
        self.source_choice = ctk.CTkComboBox(filters, variable=self.source,
                                            values=["Web", "arXiv papers"], state="readonly",
                                            command=lambda _choice: self._source_changed(), **combo_style)
        self.source_choice.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.period_choice = ctk.CTkComboBox(filters, variable=self.period,
                                            values=list(self.PERIODS), state="readonly", **combo_style)
        self.period_choice.grid(row=1, column=1, sticky="ew")
        domain_row = tk.Frame(filters, background=p.panel)
        self._domain_row = domain_row
        domain_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(9, 0))
        domain_row.columnconfigure(1, weight=1)
        self._domain_label = tk.Label(domain_row, text="Domain · optional", background=p.panel,
                                      foreground=p.quiet, font=fonts.small)
        self._domain_label.grid(row=0, column=0, padx=(0, 8))
        self.domain = ctk.CTkEntry(domain_row, height=34, corner_radius=17, border_width=0,
                                   fg_color=p.sunken, bg_color=p.panel, text_color=p.ink,
                                   font=(family, 13))
        self.domain.grid(row=0, column=1, sticky="ew")
        self.domain.bind("<Return>", self._on_return)

        self.status = tk.Label(self, text="Search the web or find papers on arXiv.",
                               background=p.paper, foreground=p.quiet, font=fonts.small,
                               anchor="w", justify="left", wraplength=350)
        self.status.pack(fill="x", padx=20, pady=(3, 10))
        self.bind("<Configure>", self._resize, add="+")

        self.body = ttk.Frame(self, style="Paper.TFrame")
        self.body.pack(fill="both", expand=True, padx=18)
        self.split = ttk.Frame(self.body, style="Paper.TFrame")
        self.split.pack(fill="both", expand=True)
        self.split.columnconfigure(0, weight=1)
        self.split.rowconfigure(0, weight=1, uniform="research_rows")
        self.split.rowconfigure(2, weight=1, uniform="research_rows")
        listing = ttk.Frame(self.split, style="Paper.TFrame")
        self._listing_frame = listing
        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(0, weight=1)
        self.listing = ttk.Treeview(listing, columns=("source", "date"), show="tree headings",
                                    selectmode="browse", height=5, style="Research.Treeview")
        self.listing.heading("#0", text="Title", anchor="w")
        self.listing.heading("source", text="Source", anchor="w")
        self.listing.heading("date", text="Date", anchor="w")
        self.listing.column("#0", width=230, minwidth=100, stretch=True)
        self.listing.column("source", width=105, minwidth=60, stretch=True)
        self.listing.column("date", width=84, minwidth=65, stretch=False)
        self.listing.grid(row=0, column=0, sticky="nsew")
        scrollbar = ctk.CTkScrollbar(listing, orientation="vertical", command=self.listing.yview,
                                     width=10, fg_color=p.paper, bg_color=p.paper,
                                     button_color=p.rule, button_hover_color=p.disabled_ink)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listing.configure(yscrollcommand=scrollbar.set)
        self.listing.bind("<<TreeviewSelect>>", self._select)
        self.listing.bind("<Double-1>", lambda _event: self.open_source())
        self.listing.bind("<Return>", lambda _event: self.open_source())
        listing.grid(row=0, column=0, sticky="nsew")
        self._divider = tk.Frame(self.split, background=p.rule, height=1, width=1)
        self._divider.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        preview = ttk.Frame(self.split, style="Paper.TFrame")
        self._preview_frame = preview
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.detail = tk.Text(preview, wrap="word", background=p.paper, foreground=p.ink,
                              font=fonts.body, borderwidth=0, highlightthickness=0,
                              padx=4, pady=16, spacing1=2, spacing3=6, height=7, width=25, cursor="arrow",
                              selectbackground=p.wash_accent, selectforeground=p.ink,
                              inactiveselectbackground=p.wash_accent)
        self.detail.grid(row=0, column=0, sticky="nsew")
        detail_bar = ctk.CTkScrollbar(preview, orientation="vertical", command=self.detail.yview,
                                      width=10, fg_color=p.paper, bg_color=p.paper,
                                      button_color=p.rule, button_hover_color=p.disabled_ink)
        detail_bar.grid(row=0, column=1, sticky="ns")
        self.detail.configure(yscrollcommand=detail_bar.set)
        self.detail.tag_configure("title", font=(heading_family, 15, "bold"), spacing3=9)
        self.detail.tag_configure("metadata", font=fonts.small, foreground=p.quiet, spacing3=11)
        self.detail.tag_configure("link", foreground=p.accent, underline=True, spacing1=8)
        self.detail.tag_bind("link", "<Button-1>", lambda _event: self.open_source())
        self.detail.tag_bind("link", "<Enter>", lambda _event: self.detail.configure(cursor="hand2"))
        self.detail.tag_bind("link", "<Leave>", lambda _event: self.detail.configure(cursor="arrow"))
        self.detail.bind("<Tab>", self._next_focus)
        self.detail.bind("<Shift-Tab>", self._previous_focus)
        preview.grid(row=2, column=0, sticky="nsew")
        self.actions = ttk.Frame(self, style="Paper.TFrame", padding=(20, 12, 20, 20))
        self.open_button = ctk.CTkButton(self.actions, text="Open source ↗", command=self.open_source,
                                         state="disabled", width=140, height=36, corner_radius=18,
                                         fg_color=p.accent, hover_color=p.accent_hover, bg_color=p.paper,
                                         text_color=p.accent_ink, text_color_disabled=p.faint, font=(family, 13, "bold"))
        self.open_button.pack(side="left")
        self.copy_button = ctk.CTkButton(self.actions, text="Copy citation", command=self.copy_citation,
                                         state="disabled", width=130, height=36, corner_radius=18,
                                         fg_color=p.sunken, hover_color=p.hover, bg_color=p.paper,
                                         text_color=p.ink, text_color_disabled=p.faint, font=(family, 13))
        self.copy_button.pack(side="left", padx=(8, 0))

        self.empty = tk.Frame(self.body, background=p.paper)
        self.empty_content = tk.Frame(self.empty, background=p.paper)
        self.empty_content.place(relx=.5, rely=.44, anchor="center", relwidth=.9)
        mark = tk.Canvas(self.empty_content, width=64, height=64, background=p.paper, highlightthickness=0)
        mark.pack(pady=(0, 14))
        mark.create_oval(2, 2, 62, 62, fill=p.wash_accent, outline="")
        mark.create_rectangle(20, 16, 39, 43, fill=p.paper, outline=p.accent, width=1.5)
        for y in (23, 28, 33):
            mark.create_line(25, y, 34, y, fill=p.accent, width=1.2)
        mark.create_oval(32, 31, 44, 43, fill=p.wash_accent, outline=p.accent, width=2)
        mark.create_line(42, 42, 48, 48, fill=p.accent, width=2)
        self.empty_title = tk.Label(self.empty_content, background=p.paper, foreground=p.ink,
                                    font=(heading_family, 15, "bold"), justify="center", wraplength=300)
        self.empty_title.pack()
        self.empty_message = tk.Label(self.empty_content, background=p.paper, foreground=p.quiet,
                                      font=fonts.small, justify="center", wraplength=300)
        self.empty_message.pack(pady=(8, 0))
        self._show_empty("A good question starts here.", "Search the web or browse arXiv preprints.\nYour sources and summaries will appear here.")
        self._show_detail("Your sources will appear here.\n\nChoose a result to read its summary and open the original page.")
        self.bind("<Destroy>", self._destroyed, add="+")
        self._poll_id = self.after(self.POLL_MS, self._poll)

    def _resize(self, event) -> None:
        if not self._closed and event.widget is self:
            width = max(120, event.width - 44)
            self.status.configure(wraplength=width)
            self.subtitle.configure(wraplength=width)
            self.empty_title.configure(wraplength=width - 16)
            self.empty_message.configure(wraplength=width - 16)
            self._set_wide_layout(event.width / ctk.ScalingTracker.get_widget_scaling(self) >= self.WIDE_AT)

    def _set_wide_layout(self, wide: bool) -> None:
        """Reflow the same source and preview widgets, preserving selection."""
        if self._wide_layout == wide:
            return
        self._wide_layout = wide
        if wide:
            self.split.rowconfigure(0, weight=1, uniform="")
            self.split.rowconfigure(2, weight=0, uniform="")
            for column in (0, 2):
                self.split.columnconfigure(column, weight=1, uniform="research_columns")
            self._listing_frame.grid_configure(row=0, column=0, padx=(0, 16))
            self._divider.grid_configure(row=0, column=1, sticky="ns", padx=0, pady=0)
            self._preview_frame.grid_configure(row=0, column=2, padx=(16, 0))
            self.detail.configure(pady=8)
            self._filters.columnconfigure(2, weight=2)
            self._domain_row.grid_configure(row=0, column=2, rowspan=2, columnspan=1,
                                            padx=(16, 0), pady=0, sticky="nsew")
            self._domain_row.columnconfigure(0, weight=1)
            self._domain_row.columnconfigure(1, weight=0)
            self._domain_label.grid_configure(row=0, column=0, sticky="w", padx=0, pady=(0, 5))
            self.domain.grid_configure(row=1, column=0, sticky="ew")
        else:
            self.split.columnconfigure(0, weight=1, uniform="")
            self.split.columnconfigure(2, weight=0, uniform="")
            for row in (0, 2):
                self.split.rowconfigure(row, weight=1, uniform="research_rows")
            self._listing_frame.grid_configure(row=0, column=0, padx=0)
            self._divider.grid_configure(row=1, column=0, sticky="ew", padx=0, pady=(8, 0))
            self._preview_frame.grid_configure(row=2, column=0, padx=0)
            self.detail.configure(pady=16)
            self._filters.columnconfigure(2, weight=0)
            self._domain_row.grid_configure(row=2, column=0, rowspan=1, columnspan=2,
                                            padx=0, pady=(9, 0), sticky="ew")
            self._domain_row.columnconfigure(0, weight=0)
            self._domain_row.columnconfigure(1, weight=1)
            self._domain_label.grid_configure(row=0, column=0, sticky="", padx=(0, 8), pady=0)
            self.domain.grid_configure(row=0, column=1, sticky="ew")

    def _suggest(self, text: str) -> None:
        self.query.delete(0, "end")
        self.query.insert(0, text)
        self.search()

    def _show_empty(self, title: str, message: str) -> None:
        self.empty_title.configure(text=title)
        self.empty_message.configure(text=message)
        self.empty.place(x=0, y=0, relwidth=1, relheight=1)
        self.empty.lift()
        self.actions.pack_forget()

    def _show_sources(self) -> None:
        self.empty.place_forget()
        self.actions.pack(fill="x")

    def _source_changed(self, _event=None) -> None:
        paper = self.source.get() == "arXiv papers"
        self.domain.configure(state="disabled" if paper or self.busy else "normal")
        if paper:
            self.status.configure(text="Search arXiv preprints. Date filters use submission dates.")
        else:
            self.status.configure(text="Search the web. Use a domain to focus on a journal or institution.")

    @staticmethod
    def _next_focus(event) -> str:
        event.widget.tk_focusNext().focus_set()
        return "break"

    @staticmethod
    def _previous_focus(event) -> str:
        event.widget.tk_focusPrev().focus_set()
        return "break"

    def _on_return(self, _event=None) -> str:
        self.search()
        return "break"

    def search(self) -> None:
        if self._closed:
            return
        query = self.query.get().strip()
        if not query:
            self.status.configure(text="Enter a topic, question, or paper title to search.", foreground=PALETTE.quiet)
            self.query.focus_set()
            return
        source = "arxiv" if self.source.get() == "arXiv papers" else "web"
        arguments = {"query": query, "source": source, "max_results": 8,
                     "recency": self.PERIODS.get(self.period.get(), "any")}
        if source == "web" and self.domain.get().strip():
            arguments["site"] = self.domain.get().strip()
        self._generation += 1
        self._result_query = query
        self.busy = True
        self.source_choice.configure(state="disabled")
        self.period_choice.configure(state="disabled")
        self.domain.configure(state="disabled")
        self._clear_results()
        self.search_button.configure(text="Cancel", command=self.cancel)
        self.status.configure(text=f'Searching {"arXiv" if source == "arxiv" else "the web"} for “{query}”…',
                              foreground=PALETTE.quiet)
        self._show_empty("Finding your sources…", "Keep working in the conversation.\nWe’ll bring the evidence here when it arrives.")
        self._show_detail("Searching…\n\nYou can keep working in the conversation while sources are fetched.")
        job = (self._generation, arguments)
        if self._workers < 2:
            self._start(job)
        else:
            self._pending = job

    run = search

    def _start(self, job: tuple[int, dict]) -> None:
        self._workers += 1
        # The worker receives only ordinary Python objects. It never reads a Tk
        # variable, schedules an `after`, or calls a widget, even on failure.
        threading.Thread(target=self._fetch, args=(self._search, self._responses, *job), daemon=True).start()

    @staticmethod
    def _fetch(search: Callable[[dict], str], responses: queue.Queue, generation: int, arguments: dict) -> None:
        try:
            answer = search(arguments)
            payload = json.loads(answer) if isinstance(answer, str) else answer
            if not isinstance(payload, dict):
                raise ValueError("The search service returned an invalid response.")
        except Exception as error:  # a backend failure must not kill Tk's event loop
            payload = {"error": str(error) or type(error).__name__}
        responses.put((generation, payload))

    def _poll(self) -> None:
        self._poll_id = None
        if self._closed:
            return
        while True:
            try:
                generation, payload = self._responses.get_nowait()
            except queue.Empty:
                break
            self._workers -= 1
            if generation == self._generation and self.busy:
                self._finish(payload)
        if self._pending is not None and self._workers < 2:
            pending, self._pending = self._pending, None
            self._start(pending)
        self._poll_id = self.after(self.POLL_MS, self._poll)

    def _finish(self, payload: dict) -> None:
        self.busy = False
        self._enable_filters()
        self.search_button.configure(text="Search", command=self.search)
        error = payload.get("error")
        if error:
            self.status.configure(text="Search could not complete. Adjust the query or try again.", foreground=PALETTE.refused)
            self._show_detail(str(error))
            self._show_empty("The search couldn’t finish.", str(error)[:400])
            return
        raw = payload.get("results", [])
        if not isinstance(raw, list):
            self.status.configure(text="The search service returned an invalid result list.", foreground=PALETTE.refused)
            self._show_detail("Please try the search again.")
            self._show_empty("Let’s try that again.", "The source returned an unreadable response.\nTry another query or source.")
            return
        self.results = [result for result in raw if isinstance(result, dict)]
        for index, result in enumerate(self.results):
            url = safe_web_url(result.get("url"))
            domain = result.get("domain") or (urlsplit(url).hostname if url else "Unknown source")
            self.listing.insert("", "end", iid=str(index), text=result.get("title") or "Untitled source",
                                values=(domain, result.get("published") or "—"))
        count = len(self.results)
        searched = str(payload.get("query") or self._result_query)
        context = f' for “{searched}”' if searched else ""
        text = f'{count} source{"s" if count != 1 else ""} found{context}.' if count else "No sources found. Try broader keywords or remove filters."
        warning = payload.get("warning")
        if warning:
            text += " " + str(warning)
        self.status.configure(text=text, foreground=PALETTE.quiet)
        if self.results:
            self._show_sources()
            self.listing.selection_set("0")
            self._select()
        else:
            self._show_empty("No matches this time.", "Try broader keywords, another source,\nor fewer filters.")
            self._show_detail("No matching sources.\n\nTry fewer words, a related term, or a different source.")

    def show_results(self, payload: dict | str) -> bool:
        """Display an agent's completed search on the Tk thread.

        A direct search the researcher is currently waiting for takes priority.
        Return whether the supplied results were displayed, so callers can
        choose whether to bring the panel into view.
        """
        if self._closed or self.busy:
            return False
        try:
            parsed = json.loads(payload) if isinstance(payload, str) else payload
            if not isinstance(parsed, dict):
                raise ValueError("The search service returned an invalid response.")
        except (TypeError, ValueError) as error:
            parsed = {"error": str(error)}
        self._generation += 1
        self._clear_results()
        if parsed.get("query"):
            self._result_query = str(parsed["query"])
            self.query.delete(0, "end")
            self.query.insert(0, str(parsed["query"]))
        if parsed.get("source") in {"web", "arxiv"}:
            self.source.set("arXiv papers" if parsed["source"] == "arxiv" else "Web")
        period = str(parsed.get("recency") or "any")
        self.period.set(next((label for label, value in self.PERIODS.items() if value == period), "Any time"))
        self.domain.configure(state="normal")
        self.domain.delete(0, "end")
        if parsed.get("site"):
            self.domain.insert(0, str(parsed["site"]))
        self._source_changed()
        self._finish(parsed)
        return True

    def cancel(self) -> None:
        if not self.busy:
            return
        self._generation += 1
        self._pending = None
        self.busy = False
        self._enable_filters()
        self.search_button.configure(text="Search", command=self.search)
        self.status.configure(text="Search cancelled. Any request already sent may finish; its results will be ignored.",
                              foreground=PALETTE.quiet)
        self._show_empty("Ready when you are.", "Search cancelled. Try another question\nor choose a suggested topic above.")
        self._show_detail("Search cancelled. Enter a new query when you are ready.")

    def _enable_filters(self) -> None:
        self.source_choice.configure(state="readonly")
        self.period_choice.configure(state="readonly")
        self.domain.configure(state="disabled" if self.source.get() == "arXiv papers" else "normal")

    def _clear_results(self) -> None:
        self.results = []
        self.listing.delete(*self.listing.get_children())
        self.open_button.configure(state="disabled")
        self.copy_button.configure(state="disabled")

    def current(self) -> dict | None:
        selected = self.listing.selection()
        if not selected:
            return None
        try:
            return self.results[int(selected[0])]
        except (ValueError, IndexError):
            return None

    def _show_detail(self, message: str) -> None:
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", message)
        self.detail.configure(state="disabled")

    def _select(self, _event=None) -> None:
        result = self.current()
        if result is None:
            return
        url = safe_web_url(result.get("url"))
        self.open_button.configure(state="normal" if url else "disabled")
        self.copy_button.configure(state="normal")
        metadata = [str(result[key]) for key in ("domain", "published") if result.get(key)]
        authors = result.get("authors")
        if authors:
            metadata.append(", ".join(map(str, authors)) if isinstance(authors, list) else str(authors))
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", str(result.get("title") or "Untitled source") + "\n", "title")
        if metadata:
            self.detail.insert("end", " · ".join(metadata) + "\n\n", "metadata")
        self.detail.insert("end", str(result.get("snippet") or "No summary was provided. Open the original source for details."))
        if url:
            self.detail.insert("end", "\n\n" + url, "link")
        self.detail.configure(state="disabled")
        self.detail.yview_moveto(0)

    def open_source(self) -> None:
        result = self.current()
        url = safe_web_url(result.get("url")) if result else None
        if url and not open_web_url(url):
            self.status.configure(text="The browser could not be opened. Copy the citation to use its link.",
                                  foreground=PALETTE.refused)

    def copy_citation(self) -> None:
        result = self.current()
        if result is None:
            return
        authors = result.get("authors")
        author_text = ", ".join(map(str, authors)) if isinstance(authors, list) else str(authors or "")
        parts = [author_text, str(result.get("title") or "Untitled source"), str(result.get("published") or ""),
                 safe_web_url(result.get("url")) or ""]
        self.clipboard_clear()
        self.clipboard_append(". ".join(part for part in parts if part))
        self.status.configure(text="Citation copied to the clipboard.", foreground=PALETTE.quiet)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._pending = None
        self.busy = False
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None

    def destroy(self) -> None:
        # CTk children can process pending geometry work during their teardown.
        # Stop observing before labels disappear, rather than at the final
        # parent <Destroy> notification.
        self.close()
        self.unbind("<Configure>")
        super().destroy()

    def _destroyed(self, event) -> None:
        if event.widget is self:
            self.close()


__all__ = ["ResearchPanel", "safe_web_url"]
