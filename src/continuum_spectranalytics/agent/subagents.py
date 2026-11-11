"""Local storage for reusable subagent instructions and capability scopes."""

from __future__ import annotations

import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from continuum_spectranalytics import paths
from continuum_spectranalytics.tools import Rejected

SUBAGENT_SCOPES = ("general", "literature", "run_review")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subagents (
    name_key TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    instructions TEXT NOT NULL,
    scope TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class SubagentProfile:
    name: str
    instructions: str
    scope: str


def _text(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise Rejected(f"The {label} must be nonblank text of at most {limit} characters.")
    if any(not char.isprintable() and char not in "\t\n\r" for char in value):
        raise Rejected(f"The {label} contains unsupported control characters.")
    return value.strip()


def _name(value: object) -> str:
    name = unicodedata.normalize("NFC", _text(value, "subagent name", 60))
    if not name[0].isalnum() or any(not char.isalnum() and char not in " -_.()&'" for char in name):
        raise Rejected(
            "Start the subagent name with a letter or number. Use letters, numbers, "
            "spaces, or these punctuation marks: - _ . ( ) & '."
        )
    return name


def _profile(name: object, instructions: object, scope: object) -> SubagentProfile:
    if not isinstance(scope, str) or scope not in SUBAGENT_SCOPES:
        raise Rejected(f"Choose a subagent scope from {', '.join(SUBAGENT_SCOPES)}.")
    return SubagentProfile(_name(name), _text(instructions, "subagent instructions", 4000), scope)


def validate_task(task: str) -> str:
    return _text(task, "subagent task", 4000)


class SubagentStore:
    """Store plain-text profiles across sessions in the local notebook directory."""

    def __init__(self, directory: Path | None = None) -> None:
        directory = Path(directory) if directory is not None else paths.notes_dir()
        self.path = directory / paths.SUBAGENTS_FILENAME

    def create(self, name: str, instructions: str, scope: str) -> SubagentProfile:
        profile = _profile(name, instructions, scope)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute(_SCHEMA)
                inserted = db.execute(
                    "INSERT INTO subagents (name_key, name, instructions, scope) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(name_key) DO NOTHING",
                    (profile.name.casefold(), profile.name, profile.instructions, profile.scope),
                )
                if not inserted.rowcount:
                    raise Rejected(f"A subagent named {profile.name!r} already exists. Choose a different name.")
        except (OSError, sqlite3.DatabaseError) as error:
            raise RuntimeError(f"Could not save to the subagent database: {error}") from error
        return profile

    def get(self, name: str) -> SubagentProfile:
        normalized = _name(name)
        matches = self._load(normalized.casefold())
        if not matches:
            raise Rejected(f"No subagent named {normalized!r} is saved. List subagents to see available names.")
        return matches[0]

    def list(self) -> list[SubagentProfile]:
        return self._load()

    def _load(self, key: str | None = None) -> list[SubagentProfile]:
        if not self.path.exists():
            return []
        query = "SELECT name, instructions, scope FROM subagents"
        parameters = ()
        if key is not None:
            query += " WHERE name_key = ?"
            parameters = (key,)
        query += " ORDER BY name_key"
        try:
            uri = self.path.absolute().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as db:
                db.execute("BEGIN")
                # A newly opened database may not yet have its schema during
                # the first concurrent create. Read the schema and rows together.
                if not db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'subagents'"
                ).fetchone():
                    return []
                return [_profile(*row) for row in db.execute(query, parameters)]
        except (Rejected, sqlite3.DatabaseError) as error:
            raise RuntimeError(f"Could not read the subagent database: {error}") from error
