"""Dropping files from Explorer onto the window.

Tk has no native drag-and-drop. ``tkinterdnd2`` ships the tkdnd extension for
Windows, macOS and Linux; when it is missing (an old install, an unusual Tk)
the window still opens and "Attach a file..." in the Session menu does the same
job through a file dialog.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable, Iterable, Sequence


def enable(root: tk.Misc) -> bool:
    """Load tkdnd into this Tk interpreter. False if it is unavailable."""
    try:
        from tkinterdnd2 import TkinterDnD
    except ImportError:
        return False
    try:
        TkinterDnD._require(root)
    except (RuntimeError, tk.TclError):
        return False
    return True


def split_paths(root: tk.Misc, data: str) -> list[str]:
    """Paths from a drop event. Tk braces paths with spaces: ``{C:/My Data/a.csv}``."""
    return [p for p in root.tk.splitlist(data) if p]


def register(root: tk.Misc, widgets: Iterable[tk.Misc], on_drop: Callable[[Sequence[str]], None]) -> None:
    """Make each widget accept dropped files and hand their paths to ``on_drop``."""
    from tkinterdnd2 import DND_FILES

    def handle(data, action="copy"):
        on_drop(split_paths(root, data))
        return action

    command = root.register(handle)
    for widget in widgets:
        try:
            root.tk.call("tkdnd::drop_target", "register", widget._w, (DND_FILES,))
            # tkinterdnd2's binding helper lives on its own widget classes, so
            # bind through Tk directly for plain tk and CustomTkinter widgets.
            root.tk.call("bind", widget._w, "<<Drop>>", f"{command} %D %A")
        except tk.TclError:
            continue
