"""Locate persistent project files and read-only resources.

The Windows executable stores mutable files in the current user's local app
data, across launches and installs. PyInstaller's bundle holds the shipped
defaults. Source installations use the checkout, or the working directory
when no checkout exists. CONTINUUM_SPECTRANALYTICS_ROOT (formerly POLYRMC_ROOT)
overrides the persistent root. Writers create their own directories when needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from continuum_spectranalytics import APP_NAME, env_setting

ROOT_ENV = "CONTINUUM_SPECTRANALYTICS_ROOT"
SUBAGENTS_FILENAME = "subagents.sqlite3"

# A checkout is identified by what only a checkout has. ``data/`` is not on this
# list on purpose: it is created by the first run that writes a sidecar, so a
# researcher's own data folder would otherwise be mistaken for the project root.
_MARKERS = ("pyproject.toml", ".git")


def is_frozen() -> bool:
    """Whether the app is running from a packaged executable."""
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Return the workspace root for persistent project files."""
    override = (env_setting("ROOT") or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    if is_frozen():
        local = Path(os.environ.get("LOCALAPPDATA", "").strip())
        if not local.is_absolute():
            local = Path.home() / "AppData" / "Local"
        return (local / APP_NAME).resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if any((candidate / marker).exists() for marker in _MARKERS):
            return candidate

    return Path.cwd().resolve()


def resource_root() -> Path:
    """Return the root containing shipped examples and default lab context."""
    if is_frozen():
        bundle = getattr(sys, "_MEIPASS", None)
        return Path(bundle).resolve() if bundle else Path(sys.executable).resolve().parent
    return project_root()


def env_file() -> Path:
    """The ``.env`` this installation reads keys from and writes them to.

    One path for both directions. When these disagree -- and they did, with the
    writer using the working directory and the reader walking up from the module
    -- a researcher saves a key, relaunches somewhere else, and is asked for it
    again with no indication of where the first one went.
    """
    return project_root() / ".env"


def runs_dir() -> Path:
    """Where processed runs and their provenance sidecars live."""
    return project_root() / "data" / "runs"


def notes_dir() -> Path:
    """Where the agent's own notebook lives, apart from run provenance."""
    return project_root() / "data" / "memory"


def logs_dir() -> Path:
    """Where session logs go.

    Separate from ``runs_dir`` on purpose. A run directory is the audit trail of
    a measurement and every file in it is a result; a session log is the
    background noise of one conversation -- tool arguments, full tool output,
    token counts, the provider's raw complaint -- and mixing the two makes the
    directory that has to stay legible the one that fills up fastest.
    """
    return project_root() / "data" / "logs"


def example_config() -> Path:
    """The reviewed example configuration used as session scaffolding."""
    return resource_root() / "examples" / "run_config.example.json"


def relative_to_root(path: str | Path) -> str:
    """Render a path for a human, shortened against the root when it is inside.

    A message telling a researcher their key went to a 90-character absolute
    path is technically complete and practically unreadable.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(project_root()))
    except ValueError:
        return str(resolved)


__all__ = [
    "ROOT_ENV",
    "env_file",
    "example_config",
    "is_frozen",
    "logs_dir",
    "notes_dir",
    "project_root",
    "relative_to_root",
    "resource_root",
    "runs_dir",
]
