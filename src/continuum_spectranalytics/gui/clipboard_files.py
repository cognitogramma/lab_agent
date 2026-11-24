"""Files copied in File Explorer (Ctrl+C), read off the clipboard for Ctrl+V.

Explorer puts copied files on the clipboard as a CF_HDROP list, which Tk's own
clipboard cannot see -- it only reads text. On Windows the list is read through
the Win32 API. Everywhere, clipboard *text* that is nothing but paths to files
also counts: Explorer's "Copy as path" ("C:\\...\\paper.pdf", quotes included)
and the file:// lines Linux file managers copy.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

CF_HDROP = 15


def files_on_clipboard(root) -> list[str]:
    """Paths of files on the clipboard, or [] when it holds ordinary text."""
    if os.name == "nt":
        listed = _windows_file_list()
        if listed:
            return listed
    try:
        text = root.clipboard_get()
    except Exception:  # noqa: BLE001 - empty clipboard or non-text content
        return []
    return paths_in_text(text)


def paths_in_text(text: str) -> list[str]:
    """Every line a path to an existing file or folder, or [] if any line is not."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if lines and lines[0] in {"copy", "cut"}:  # x-special/gnome-copied-files
        lines = lines[1:]
    found = []
    for line in lines:
        candidate = line.strip().strip('"').strip("'")
        if candidate.startswith("file://"):
            parsed = urlparse(candidate)
            candidate = unquote(parsed.path)
            if os.name == "nt" and candidate.startswith("/") and ":" in candidate[:4]:
                candidate = candidate[1:]
        path = Path(candidate).expanduser()
        if not path.is_absolute() or not path.exists():
            return []
        found.append(str(path))
    return found


def _windows_file_list() -> list[str]:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    shell32 = ctypes.windll.shell32
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT
    shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wintypes.UINT, ctypes.c_wchar_p, wintypes.UINT]

    if not user32.IsClipboardFormatAvailable(CF_HDROP):
        return []
    # Another program can hold the clipboard for a moment; try briefly.
    for _ in range(5):
        if user32.OpenClipboard(None):
            break
        time.sleep(0.03)
    else:
        return []
    try:
        handle = user32.GetClipboardData(CF_HDROP)
        if not handle:
            return []
        count = shell32.DragQueryFileW(handle, 0xFFFFFFFF, None, 0)
        files = []
        for index in range(count):
            length = shell32.DragQueryFileW(handle, index, None, 0)
            buffer = ctypes.create_unicode_buffer(length + 1)
            shell32.DragQueryFileW(handle, index, buffer, length + 1)
            files.append(buffer.value)
        return files
    finally:
        user32.CloseClipboard()
