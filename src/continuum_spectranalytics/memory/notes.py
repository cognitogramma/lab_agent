"""What the agent learned, kept between sessions.

:mod:`continuum_spectranalytics.memory.registry` remembers what the *instrument* did -- it is an
index over provenance sidecars, and every field in it is derived from a real
run. This module is the other half: what was concluded, and by whom. A note
records that lira-2000mM went unstable above 3 M Gd, that a particular file
needs the header count read a certain way, that a sample's blank was taken on a
different day. None of that is recoverable from a sidecar, and all of it is the
kind of thing a researcher otherwise re-derives every few months.

Notes are kept deliberately separate from run provenance, in their own
directory, and never mixed into it. A sidecar is a record of what a measurement
did and has to stay reconstructible from the run itself; a note is somebody's
opinion. Filing an opinion inside the audit trail would make the audit trail
unfalsifiable, so the two live apart and a note always carries its author.

The store is plain JSON on disk, one file, human-readable and hand-editable. It
is small by nature -- these are notes, not a corpus -- and a researcher who
disagrees with one should be able to open it in an editor and delete the line.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from continuum_spectranalytics import paths

NOTES_FILENAME = "notes.json"
RESEARCH_IDEA_TAG = "research-idea"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Note:
    """One recorded conclusion.

    ``subject`` is what the note is about -- a sample name, a run id, an
    instrument quirk -- and is what search keys on first. ``author`` is
    mandatory for the same reason it is mandatory on a declaration: an
    unattributed conclusion cannot be questioned later by whoever finds it
    wrong.
    """

    subject: str
    body: str
    author: str
    created_at: str = field(default_factory=_now)
    run_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Note":
        return cls(
            subject=str(raw.get("subject", "")),
            body=str(raw.get("body", "")),
            author=str(raw.get("author", "unknown")),
            created_at=str(raw.get("created_at") or _now()),
            run_id=raw.get("run_id"),
            tags=[str(t) for t in raw.get("tags") or []],
        )

    def render(self) -> str:
        head = f"[{self.created_at[:10]}] {self.subject} -- {self.author}"
        if self.run_id:
            head += f" (run {self.run_id})"
        if self.tags:
            head += f" tags: {', '.join(self.tags)}"
        return f"{head}\n    {self.body}"


class NoteBook:
    """The durable note store, loaded from and written to one JSON file."""

    def __init__(self, directory: str | Path | None = None) -> None:
        # Resolved per instance rather than frozen into a module constant: the
        # default hangs off the project root, and a root read at import time is
        # a root no test and no CONTINUUM_SPECTRANALYTICS_ROOT can move.
        self.directory = Path(directory) if directory else paths.notes_dir()
        self.path = self.directory / NOTES_FILENAME
        self.notes: list[Note] = []
        self.load()

    def load(self) -> list[Note]:
        """Read the store. A missing or corrupt file is an empty notebook.

        Corruption is tolerated rather than raised: a session that cannot
        start because a notes file has a stray comma in it is a worse outcome
        than one that starts with no notes and says so.
        """
        self.notes = []
        if not self.path.exists():
            return self.notes
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            return self.notes
        if isinstance(raw, list):
            self.notes = [Note.from_dict(item) for item in raw if isinstance(item, dict)]
        return self.notes

    def save(self) -> Path:
        """Replace the notebook only after a complete sibling file is flushed."""
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            # A sibling lives on the same filesystem, so replacement is atomic.
            # Close before replacing: Windows does not allow replacing an open file.
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                    prefix=f".{NOTES_FILENAME}.", suffix=".tmp",
                                    delete=False) as stream:
                temporary = Path(stream.name)
                json.dump([note.to_dict() for note in self.notes], stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    # Cleanup must not hide the original persistence failure.
                    pass
        return self.path

    def add(
        self,
        subject: str,
        body: str,
        author: str,
        run_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Note:
        """File a note and persist immediately.

        Written through on every add rather than flushed at session end: a
        session that ends by crashing is exactly the one whose notes are worth
        keeping.
        """
        if not str(subject).strip():
            raise ValueError("a note needs a subject, or nothing will ever find it")
        if not str(body).strip():
            raise ValueError("a note needs a body")
        if not str(author).strip():
            raise ValueError(
                "a note needs an author. An unattributed conclusion cannot be "
                "questioned later by whoever finds it wrong."
            )
        note = Note(
            subject=str(subject).strip(),
            body=str(body).strip(),
            author=str(author).strip(),
            run_id=run_id,
            tags=[str(t).strip() for t in (tags or []) if str(t).strip()],
        )
        self.notes.append(note)
        try:
            self.save()
        except Exception:
            self.notes.remove(note)
            raise
        return note

    def search(
        self,
        query: str | None = None,
        subject: str | None = None,
        run_id: str | None = None,
        tag: str | None = None,
        limit: int = 20,
    ) -> list[Note]:
        """Filter notes, newest first."""
        results: list[Note] = []
        for note in self.notes:
            if subject and subject.lower() not in note.subject.lower():
                continue
            if run_id and note.run_id != run_id:
                continue
            if tag and tag.lower() not in [t.lower() for t in note.tags]:
                continue
            if query:
                blob = f"{note.subject} {note.body} {' '.join(note.tags)}".lower()
                if query.lower() not in blob:
                    continue
            results.append(note)
        results.sort(key=lambda n: n.created_at, reverse=True)
        return results[:limit]

    def add_research_idea(
        self,
        title: str,
        idea: str,
        author: str,
        *,
        evidence: str = "",
        source_urls: list[str] | None = None,
        next_step: str = "",
        run_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Note:
        """Keep a hypothesis and its context in the searchable notebook.

        Labels survive both JSON round trips and ordinary notebook rendering.
        A research idea is explicitly provisional, even when it cites a source;
        it is never added to measured run provenance.
        """
        if not str(idea).strip():
            raise ValueError("a research idea needs an idea or hypothesis")
        urls = list(dict.fromkeys(str(url).strip() for url in (source_urls or []) if str(url).strip()))
        body = ["Research idea — unverified hypothesis", f"Idea: {str(idea).strip()}"]
        if str(evidence).strip():
            body.append(f"Evidence / motivation: {str(evidence).strip()}")
        if urls:
            body.append("Sources:\n" + "\n".join(urls))
        if str(next_step).strip():
            body.append(f"Next step: {str(next_step).strip()}")
        idea_tags = [RESEARCH_IDEA_TAG, *(tags or [])]
        return self.add(title, "\n".join(body), author, run_id=run_id,
                        tags=list(dict.fromkeys(idea_tags)))

    def research_ideas(self, query: str | None = None, limit: int = 20) -> list[Note]:
        """Retrieve saved ideas without mixing them with measured observations."""
        return self.search(query=query, tag=RESEARCH_IDEA_TAG, limit=limit)

    def __len__(self) -> int:
        return len(self.notes)


__all__ = ["NOTES_FILENAME", "RESEARCH_IDEA_TAG", "Note", "NoteBook"]
