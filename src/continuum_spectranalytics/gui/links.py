"""Web links opened by a deliberate click, shared by research and chat."""

from __future__ import annotations

import re
import webbrowser
from urllib.parse import urlsplit


def safe_web_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 4096:
        return None
    target = value.strip()
    if re.search(r"[\x00-\x20\x7f]", target):
        return None
    try:
        parts = urlsplit(target)
        if (parts.scheme.lower() not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None):
            return None
        _ = parts.port  # accessing the port validates malformed/out-of-range values
    except ValueError:
        return None
    return target


def open_web_url(value: object) -> bool:
    target = safe_web_url(value)
    if target is None:
        return False
    try:
        return bool(webbrowser.open_new_tab(target))
    except (OSError, webbrowser.Error):
        return False
