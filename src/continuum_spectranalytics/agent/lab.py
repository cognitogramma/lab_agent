"""LAB.md -- the instrument context, always in the agent's context, whole.

Design section 10.5. This is the hyperspecificity mechanism: constants,
calibration values, house conventions and authoritative references for one
instrument or project. Swapping the file swaps the lab, and nothing is
retrained.

It is loaded **whole**, never retrieved in fragments. The file is small and it
has to be *complete*: top-k retrieval over it would return the three chunks that
happened to match a query and leave the agent unable to know what it was not
shown. A missing house convention is not a gap the agent can notice.

A missing LAB.md is reported rather than silently tolerated. An agent running
without its lab context will still refuse the right things -- the guards live in
code -- but it will not know this lab's conventions, and it should say so rather
than answer as though it did.
"""

from __future__ import annotations

from pathlib import Path

from continuum_spectranalytics import paths

# Searched in order. The project root is where a single-instrument lab keeps it;
# the config directory is for a deployment serving several.
SEARCH_PATHS = (
    "LAB.md",
    "config/LAB.md",
    ".continuum-spectranalytics/LAB.md",
    ".polyrmc/LAB.md",
)

MISSING_NOTICE = (
    "No LAB.md was found. You do not have this lab's instrument context: "
    "constants, calibration values, house conventions, and which values are "
    "unconfirmed. Say so when it matters rather than answering as though you "
    "had it, and do not substitute general knowledge for a house convention."
)


def find_lab_file(start: str | Path | None = None) -> Path | None:
    """Find a lab file beside source data or in the installed app's workspace."""
    if start is None and paths.is_frozen():
        directories = (paths.project_root(), paths.resource_root())
    else:
        current = Path(start or Path.cwd()).resolve()
        if current.is_file():
            current = current.parent
        directories = (current, *current.parents)

    for directory in directories:
        for candidate in SEARCH_PATHS:
            path = directory / candidate
            if path.is_file():
                return path
    return None


def load_lab_context(start: str | Path | None = None) -> tuple[str, Path | None]:
    """Return the lab context and where it came from.

    The text is what goes into the agent's context. The path is returned
    alongside it so a run can record *which* lab file it was operating under --
    two instruments with different ceilings would otherwise be indistinguishable
    in the log.
    """
    path = find_lab_file(start)
    if path is None:
        return MISSING_NOTICE, None
    return path.read_text(encoding="utf-8"), path


PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt from disk.

    Prompts live in files rather than in code on purpose. What the agent is told
    it may and may not do is the reviewable part of this system, and a
    researcher who disagrees with the taste encoded there should be able to
    change it and see the effect without touching Python.
    """
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def system_prompt_with_lab(base: str, start: str | Path | None = None) -> str:
    """Compose the system prompt with the lab context appended.

    Appended rather than interleaved, so the base prompt stays a stable,
    reviewable document and the lab-specific part is visibly separable from it.
    """
    context, path = load_lab_context(start)
    origin = f" ({path})" if path else ""
    return f"{base}\n\n---\n\n# Lab context{origin}\n\n{context}\n"


__all__ = [
    "MISSING_NOTICE",
    "PROMPT_DIR",
    "SEARCH_PATHS",
    "find_lab_file",
    "load_lab_context",
    "load_prompt",
    "system_prompt_with_lab",
]
