"""The visual shell, independent of the analysis and search implementations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk

from polyrmc import APP_NAME, paths
from polyrmc.gui.ideas import IdeasPanel
from polyrmc.gui.literature import LiteraturePanel
from polyrmc.gui.shell import LibraryRail
from polyrmc.gui.theme import PALETTE as P
from polyrmc.gui.widgets import FigurePanel, SearchPanel, StatePanel, ToolsPanel, Transcript


def build_layout(app) -> None:
    ctk.set_appearance_mode("dark")
    body_font = (app.fonts.body[0], 16)
    small_font = (app.fonts.body[0], 14)
    heading_font = (app.fonts.heading[0], 24, "bold")
    app.configure(fg_color=P.canvas)
    shell = ctk.CTkFrame(app, fg_color=P.canvas, corner_radius=0)
    shell.pack(fill="both", expand=True)

    toolbar = ctk.CTkFrame(shell, fg_color=P.canvas, corner_radius=0, height=66)
    toolbar.pack(fill="x", padx=12, pady=(10, 8))
    toolbar.columnconfigure(1, weight=1)
    ctk.CTkLabel(toolbar, text=APP_NAME.replace(" ", "\n", 1), font=(app.fonts.heading[0], 22, "bold"),
                 text_color=P.ink, width=186, anchor="w", justify="left").grid(row=0, column=0, padx=(12, 18))
    ctk.CTkLabel(toolbar, text="RESEARCH STUDIO", font=(app.fonts.body[0], 12, "bold"),
                 text_color=P.support, anchor="w").grid(row=0, column=1, sticky="w")
    app.assistant_toggle = ctk.CTkButton(
        toolbar, text="Research assistant", width=148, height=44, corner_radius=8,
        fg_color=P.sunken, hover_color=P.hover, text_color=P.ink,
        font=body_font, command=app.toggle_assistant,
    )
    app.assistant_toggle.grid(row=0, column=2, padx=(0, 8))
    app.menu_button = ctk.CTkButton(
        toolbar, text="Menu", width=64, height=44, corner_radius=8,
        fg_color=P.sunken, hover_color=P.hover, text_color=P.ink,
        font=body_font, command=app.show_menu,
    )
    app.menu_button.grid(row=0, column=3)

    middle = ctk.CTkFrame(shell, fg_color=P.canvas, corner_radius=0)
    middle.pack(fill="both", expand=True, padx=8)
    app.navigation = LibraryRail(middle, app.fonts, app._navigate, app.open_run, app.sign_in)
    app.navigation.pack(side="left", fill="y", padx=(0, 8))
    app.content = ctk.CTkFrame(middle, fg_color=P.canvas, corner_radius=0)
    app.content.pack(side="left", fill="both", expand=True)
    app.content.columnconfigure(0, weight=1)
    app.content.rowconfigure(0, weight=1)
    app.workspace = ctk.CTkFrame(app.content, fg_color=P.paper, corner_radius=9)
    app.workspace.grid(row=0, column=0, sticky="nsew")
    header = ctk.CTkFrame(app.workspace, fg_color="transparent", corner_radius=0)
    header.pack(fill="x", padx=24, pady=(18, 4))
    header.columnconfigure(0, weight=1)
    app.page_title = ctk.CTkLabel(header, text="Workbench", font=heading_font, anchor="w", text_color=P.ink)
    app.page_title.grid(row=0, column=0, sticky="w")
    search = ctk.CTkFrame(header, fg_color="transparent", corner_radius=0)
    search.grid(row=0, column=1, sticky="e", padx=(18, 0))
    search.columnconfigure(0, weight=1)
    app.global_query = ctk.CTkEntry(search, height=36, width=240, corner_radius=7,
        border_width=1, border_color=P.rule, fg_color=P.panel, text_color=P.ink,
        placeholder_text_color=P.quiet, placeholder_text="Find papers or web sources",
        font=(app.fonts.body[0], 13))
    app.global_query.grid(row=0, column=0, sticky="ew")
    app.global_query.bind("<Return>", app.search_global)
    app.find_sources_button = ctk.CTkButton(search, text="Find sources", width=98, height=36, corner_radius=7,
        fg_color=P.sunken, hover_color=P.hover, text_color=P.ink,
        font=small_font, command=app.search_global)
    app.find_sources_button.grid(row=0, column=1, padx=(7, 0))

    def arrange_search(event) -> None:
        if event.widget is not header:
            return
        wide = event.width / ctk.ScalingTracker.get_window_scaling(app) >= 650
        app.page_title.grid_configure(columnspan=1 if wide else 2)
        search.grid_configure(row=0 if wide else 1, column=1 if wide else 0,
                               columnspan=1 if wide else 2, sticky="e" if wide else "ew",
                               padx=(18, 0) if wide else 0, pady=0 if wide else (10, 0))

    header.bind("<Configure>", arrange_search, add="+")
    app.tabs = ttk.Notebook(app.workspace, style="Workspace.TNotebook")
    app.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    app.state_panel = StatePanel(app.tabs, app.fonts, on_open=app.open_run, on_research=app.show_research,
                                 on_runs=lambda: app._navigate("runs"), on_notes=lambda: app._navigate("notes"))
    app.tools_panel = ToolsPanel(app.tabs, app.fonts, app.registry, app.always_allow)
    app.runs_panel = SearchPanel(app.tabs, app.fonts, "Find an experiment by sample, buffer or run id.",
                                 lambda q: app._call_tool("search_past_runs", {"query": q or None}))
    app.notes_panel = IdeasPanel(app.tabs, app.fonts,
                                 lambda q: app._call_tool("read_notes", {"query": q or None}),
                                 app._save_research_idea)
    app.research_panel = LiteraturePanel(
        app.tabs, app.fonts,
        lambda arguments: app._call_tool("review_literature", arguments),
        context=app._literature_context,
        read=lambda arguments: app._call_tool("read_web_page", arguments),
        load_saved=app.reading_library.list,
        save_source=app.reading_library.save,
        on_review=app._review_papers,
    )
    app.figures = FigurePanel(app.tabs, app.fonts,
                              [app.session.runs_dir / "figures", paths.project_root() / "data" / "figures"])
    app._panels = {"run": app.state_panel, "research": app.research_panel, "figures": app.figures,
                   "runs": app.runs_panel, "notes": app.notes_panel, "tools": app.tools_panel}
    for key, panel in app._panels.items():
        app.tabs.add(panel, text=key)
    app.tabs.bind("<<NotebookTabChanged>>", app._sync_navigation)

    app.assistant_visible = False
    app.drawer = ctk.CTkFrame(app.content, fg_color=P.paper, corner_radius=9, width=390)
    app.drawer.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    app.drawer.grid_propagate(False)
    app.drawer.pack_propagate(False)
    title = ctk.CTkFrame(app.drawer, fg_color="transparent", corner_radius=0)
    title.pack(fill="x", padx=18, pady=(18, 12))
    ctk.CTkLabel(title, text="Research assistant", font=(app.fonts.heading[0], 20, "bold"),
                 text_color=P.ink).pack(side="left")
    ctk.CTkButton(title, text="Close", width=60, height=30, corner_radius=7,
                  fg_color=P.sunken, hover_color=P.rule, text_color=P.quiet,
                  font=small_font, command=app.hide_assistant).pack(side="right")
    app.welcome = ctk.CTkFrame(app.drawer, fg_color="transparent", corner_radius=0)
    app.welcome.pack(fill="x", padx=18, pady=(0, 10))
    ctk.CTkLabel(app.welcome, text="Review sources. Develop an idea.", text_color=P.quiet,
                 font=body_font, anchor="w").pack(fill="x")
    ctk.CTkButton(app.welcome, text="Model settings", width=130, height=30, corner_radius=7,
                  fg_color=P.sunken, hover_color=P.rule, text_color=P.ink,
                  font=small_font, command=app.sign_in).pack(anchor="w", pady=(10, 0))
    composer = ctk.CTkFrame(app.drawer, fg_color="transparent", corner_radius=0)
    composer.pack(side="bottom", fill="x", padx=16, pady=16)
    ctk.CTkLabel(composer, text="Your question", text_color=P.quiet, font=small_font,
                 anchor="w").pack(fill="x", pady=(0, 6))
    box = ctk.CTkFrame(composer, fg_color=P.sunken, corner_radius=14)
    box.pack(fill="x")
    app.entry = tk.Text(box, height=3, wrap="word", background=P.sunken, foreground=P.ink,
                        font=app.fonts.body, borderwidth=0, highlightthickness=0,
                        padx=2, pady=4, insertbackground=P.accent, selectbackground=P.selection)
    app.entry.pack(fill="both", expand=True, padx=12, pady=8)
    app.entry.bind("<Return>", app._on_return)
    app.entry.bind("<Shift-Return>", lambda _e: None)
    app.entry.bind("<Control-k>", lambda _e: app.show_research())
    app.entry.bind("<Control-o>", app._open_run_shortcut)
    app.ask_button = ctk.CTkButton(composer, text="Ask", width=76, height=36, corner_radius=7,
                                   fg_color=P.accent, hover_color=P.accent_hover, text_color=P.accent_ink,
                                   font=(app.fonts.body[0], 14, "bold"), command=app.submit)
    app.ask_button.pack(side="right", pady=(8, 0))
    ctk.CTkLabel(composer, text="Enter to send", text_color=P.quiet, font=small_font).pack(side="left", pady=(8, 0))
    app.transcript = Transcript(app.drawer, app.fonts)
    app.transcript.pack(fill="both", expand=True, padx=2)
    app.drawer.grid_remove()

    status = ctk.CTkFrame(shell, fg_color=P.canvas, corner_radius=0, height=46)
    status.pack(side="bottom", fill="x", padx=20, pady=(6, 6), before=middle)
    status.columnconfigure(0, weight=1)
    status.columnconfigure(1, weight=1)
    status.columnconfigure(2, weight=1)
    left = ctk.CTkFrame(status, fg_color="transparent", corner_radius=0)
    left.grid(row=0, column=0, sticky="w")
    app.run_label = ctk.CTkLabel(left, text="", text_color=P.quiet, font=small_font,
                                 wraplength=320, anchor="w", justify="left")
    app.run_label.pack(anchor="w")
    center = ctk.CTkFrame(status, fg_color="transparent", corner_radius=0)
    center.grid(row=0, column=1)
    app.activity_label = ctk.CTkLabel(center, text="Ready", text_color=P.support,
                                      font=small_font)
    app.activity_label.pack()
    app.spend_label = ctk.CTkLabel(center, text="", text_color=P.quiet, font=small_font, wraplength=310)
    app.spend_label.pack()
    app.progress = ttk.Progressbar(center, mode="indeterminate", length=110)
    right = ctk.CTkFrame(status, fg_color="transparent", corner_radius=0)
    right.grid(row=0, column=2, sticky="e")
    app.model_label = ctk.CTkLabel(right, text="", text_color=P.quiet, font=small_font,
                                   wraplength=320, justify="right", anchor="e")
    app.model_label.pack(anchor="e")
    app.log_label = ctk.CTkLabel(right, text="", text_color=P.ink, font=small_font, cursor="hand2")
    app.log_label.pack(anchor="e")
    app.log_label.bind("<Button-1>", lambda _e: app.open_log())
    app.tabs.focus_set()
