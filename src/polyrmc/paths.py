"""Where this installation keeps its files.

Every default path in this package used to be written relative to the current
working directory -- ``data/runs``, ``data/memory``, ``examples/``, ``.env``.
That works exactly once, in the checkout, with the terminal sitting in the
project root, which is where it was always tested and never where a researcher
actually is. Run ``polyrmc agent`` from the folder holding your raw files and
the session cannot find the example config, reports zero past runs next to a
hundred sidecars, and writes a saved key into a ``.env`` nothing will read back.

So the root is resolved once, here, and every default hangs off it. The rules,
in order:

1. ``POLYRMC_ROOT``, if set. This is the escape hatch for a deployment that
   keeps its data somewhere other than beside the code, and it is what the
   tests use rather than depending on where pytest was invoked.
2. The nearest ancestor of this file containing a project marker. In a checkout
   or an editable install that is the repo root, which is what a researcher
   means by "the project".
3. The current working directory. This is the honest answer for a wheel
   installed into site-packages: there is no checkout to find, so the project
   is wherever you are standing.

Nothing here creates a directory. Resolving where a thing belongs and deciding
to write it are different acts, and the writers already handle their own
``mkdir`` at the point they commit to writing.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "POLYRMC_ROOT"

# A checkout is identified by what only a checkout has. ``data/`` is not on this
# list on purpose: it is created by the first run that writes a sidecar, so a
# researcher's own data folder would otherwise be mistaken for the project root.
_MARKERS = ("pyproject.toml", ".git")


def project_root() -> Path:
    """The directory this installation treats as the project.

    Deliberately not cached. The cost is a handful of ``stat`` calls, and a
    cached root is a root that cannot be changed by a test, or by a researcher
    exporting ``POLYRMC_ROOT`` for one command.
    """
    override = os.environ.get(ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if any((candidate / marker).exists() for marker in _MARKERS):
            return candidate

    return Path.cwd().resolve()


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
    return project_root() / "examples" / "run_config.example.json"


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
    "logs_dir",
    "notes_dir",
    "project_root",
    "relative_to_root",
    "runs_dir",
]
