"""Exercise the disk boundary and actual orchestrator approval routing offline."""
import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from continuum_spectranalytics.filesystem import FileAccess, FilePolicy, MAX_READ_BYTES, MAX_WRITE_BYTES
from continuum_spectranalytics.tools import Rejected, ToolSurface
from continuum_spectranalytics.agent.toolspec import ToolContext, build_registry
from continuum_spectranalytics.agent.orchestrator import ApprovalPolicy, Orchestrator, allow_all
from continuum_spectranalytics.agent.toolcall import ScriptedProvider, ToolCall
from continuum_spectranalytics.memory.notes import NoteBook
from continuum_spectranalytics.state import RunState


@pytest.fixture
def files(tmp_path):
    return FileAccess(FilePolicy((tmp_path,), (tmp_path,)))


def test_create_read_replace_with_backup(files, tmp_path):
    folder = tmp_path / "new" / "nested"
    files.create_directory(str(folder))
    path = folder / "notes.txt"
    files.write_file(str(path), "first \u03b1")
    original = json.loads(files.read_file(str(path)))
    assert original["content"] == "first \u03b1"
    replaced = json.loads(files.write_file(str(path), "second", expected_sha256=original["sha256"]))
    assert path.read_text() == "second"
    assert Path(replaced["backup"]).read_text(encoding="utf-8") == "first \u03b1"
    assert replaced["sha256"] == hashlib.sha256(b"second").hexdigest()


def test_replacement_requires_current_hash(files, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("original")
    for digest in ("", "stale"):
        with pytest.raises(Rejected, match="Read it first"):
            files.write_file(str(path), "bad", expected_sha256=digest)
    assert path.read_text() == "original"
    assert not list(tmp_path.glob("*.continuum-spectranalytics-backup-*"))


def test_binary_round_trip(files, tmp_path):
    path = tmp_path / "sample.bin"
    data = bytes(range(256))
    files.write_file(str(path), base64.b64encode(data).decode(), "base64")
    read = json.loads(files.read_file(str(path), "base64"))
    assert base64.b64decode(read["content"]) == data
    with pytest.raises(Rejected, match="not UTF-8"):
        files.read_file(str(path))


@pytest.mark.parametrize("relative", ["Windows/system.ini", "Program Files/app/settings.txt",
    "AppData/Local/data.txt", ".ssh/id_rsa", ".git/config", ".env", ".env.local",
    "filesystem-permissions.json", "pagefile.sys"])
def test_protected_paths_refused_for_reads_and_writes(files, tmp_path, relative):
    for write in (False, True):
        with pytest.raises(Rejected, match="protected"):
            files.policy.check(tmp_path / relative, write=write)


def test_outside_and_explicit_denied_paths(tmp_path):
    allowed = tmp_path / "allowed"
    denied = allowed / "private"
    policy = FilePolicy((allowed,), (allowed,), (denied,))
    for write in (False, True):
        with pytest.raises(Rejected, match="outside"):
            policy.check(tmp_path / "outside.txt", write=write)
        with pytest.raises(Rejected, match="excluded"):
            policy.check(denied / "notes.txt", write=write)


@pytest.mark.parametrize("raw", ["relative.txt", "//server/share/file", "\\\\?\\C:\\file", "\\\\.\\NUL"])
def test_nonlocal_or_relative_paths(files, raw):
    with pytest.raises(Rejected):
        files.policy.check(raw)


@pytest.mark.parametrize("name", ["NUL", "COM1.txt", "notes.txt:secret", "trailing.", "trailing "])
def test_windows_path_aliases_refused(files, tmp_path, name):
    with pytest.raises(Rejected):
        files.policy.check(tmp_path / name)


def test_hardlink_refused(files, tmp_path):
    original = tmp_path / "original.txt"
    original.write_text("original")
    alias = tmp_path / "alias.txt"
    os.link(original, alias)
    with pytest.raises(Rejected, match="Hard-linked"):
        files.read_file(str(alias))
    with pytest.raises(Rejected, match="Hard-linked"):
        files.write_file(str(alias), "bad")
    assert original.read_text() == "original"


def test_symlink_refused(files, tmp_path):
    original = tmp_path / "original.txt"
    original.write_text("original")
    alias = tmp_path / "alias.txt"
    try:
        alias.symlink_to(original)
    except OSError:
        pytest.skip("Windows account cannot create symlinks")
    with pytest.raises(Rejected, match="Linked"):
        files.read_file(str(alias))


def test_list_omits_protected_files_and_paginates(files, tmp_path):
    folder = tmp_path / "listing"
    folder.mkdir()
    (folder / ".env").write_text("secret")
    for n in range(101):
        (folder / f"sample-{n:03}.txt").write_text("data")
    first = json.loads(files.list_directory(str(folder)))
    second = json.loads(files.list_directory(str(folder), first["next_offset"]))
    assert first["total"] == 101
    assert len(first["entries"]) == 100
    assert len(second["entries"]) == 1
    assert second["next_offset"] is None


def test_unicode_pagination_and_write_limit(files, tmp_path):
    path = tmp_path / "notes.txt"
    content = "\u03b1" * (MAX_READ_BYTES + 5)
    files.write_file(str(path), content)
    first = json.loads(files.read_file(str(path)))
    second = json.loads(files.read_file(str(path), offset=first["next_offset"]))
    assert first["content"] + second["content"] == content
    with pytest.raises(Rejected, match="1 MiB"):
        files.write_file(str(tmp_path / "huge.txt"), "a" * (MAX_WRITE_BYTES + 1))


def test_policy_missing_disabled_and_invalid_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTINUUM_SPECTRANALYTICS_ROOT", str(tmp_path))
    config = tmp_path / "filesystem-permissions.json"
    for content in (None, "{broken", '{"enabled": false}', '{"enabled": true, "read_roots": "C:/"}'):
        if content is not None:
            config.write_text(content)
        with pytest.raises(Rejected, match="unavailable"):
            FileAccess().read_file(str(tmp_path / "anything.txt"))
    config.write_text(json.dumps({"enabled": True, "read_roots": [str(tmp_path)], "write_roots": []}))
    file = tmp_path / "read.txt"
    file.write_text("allowed")
    assert json.loads(FileAccess().read_file(str(file)))["content"] == "allowed"
    with pytest.raises(Rejected, match="write roots"):
        FileAccess().write_file(str(tmp_path / "new.txt"), "no")


def test_agent_refuses_unapproved_write_then_accepts_approved_write(files, tmp_path, run_config):
    context = ToolContext(surface=ToolSurface(state=RunState(run_id="files", source_file=str(run_config.source_file)), config=run_config),
                          notes=NoteBook(tmp_path / "memory"), files=files)
    agent = Orchestrator(context=context, provider=ScriptedProvider([]), approvals=ApprovalPolicy())
    path = tmp_path / "summary.md"
    call = ToolCall("write", "write_file", {"path": str(path), "content": "summary"})
    result, event = agent._invoke(call)
    assert not result.ok and event.kind == "denied" and not path.exists()
    agent.approvals = ApprovalPolicy(ask=allow_all)
    result, event = agent._invoke(call)
    assert result.ok and event.kind == "result" and path.read_text() == "summary"
    registry = build_registry()
    assert registry["create_directory"].mutating
    assert not registry["read_file"].mutating
