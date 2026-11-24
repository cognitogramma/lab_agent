"""Using files on the researcher's own computer: OneDrive, folders, copy, open, paste."""
from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from continuum_spectranalytics import filesystem
from continuum_spectranalytics.filesystem import FileAccess, FilePolicy
from continuum_spectranalytics.gui.clipboard_files import paths_in_text
from continuum_spectranalytics.tools import Rejected


@pytest.fixture
def policy_file(monkeypatch, tmp_path):
    target = tmp_path / "appdata" / filesystem.POLICY_NAME
    monkeypatch.setattr(filesystem, "policy_path", lambda: target)
    return target


def _reparse(path: Path, tag: int):
    """Make ``path`` look like a Windows reparse point with this tag."""
    real = Path.lstat

    def lstat(self):
        info = real(self)
        if self == path:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400, st_reparse_tag=tag)
        return info

    return lstat


def test_onedrive_files_are_readable(monkeypatch, tmp_path):
    synced = tmp_path / "OneDrive" / "Desktop" / "run 12.csv"
    synced.parent.mkdir(parents=True)
    synced.write_text("t,ls8\n")
    monkeypatch.setattr(Path, "lstat", _reparse(synced, 0x9000701A))  # IO_REPARSE_TAG_CLOUD_7
    files = FileAccess(FilePolicy((tmp_path,), ()))
    assert json.loads(files.read_file(str(synced)))["content"] == "t,ls8\n"


def test_links_and_junctions_are_still_refused(monkeypatch, tmp_path):
    linked = tmp_path / "junction"
    linked.mkdir()
    monkeypatch.setattr(Path, "lstat", _reparse(linked, 0xA0000003))  # IO_REPARSE_TAG_MOUNT_POINT
    with pytest.raises(Rejected, match="reparse"):
        FileAccess(FilePolicy((tmp_path,), ())).list_directory(str(linked))


def test_the_dialogs_choice_is_saved_and_used(policy_file, tmp_path):
    desktop, documents = tmp_path / "Desktop", tmp_path / "Documents"
    desktop.mkdir(), documents.mkdir()
    (desktop / "a.txt").write_text("a")

    files = FileAccess()
    with pytest.raises(Rejected, match="unavailable"):
        files.list_directory(str(desktop))

    filesystem.save_policy([desktop, documents], [desktop])
    saved = json.loads(policy_file.read_text())
    assert saved["enabled"] is True
    assert saved["write_roots"] == [str(desktop)]
    assert [e["name"] for e in json.loads(files.list_directory(str(desktop)))["entries"]] == ["a.txt"]

    filesystem.save_policy([], [])
    assert json.loads(policy_file.read_text())["enabled"] is False


def test_saving_keeps_hand_written_exclusions(policy_file, tmp_path):
    policy_file.parent.mkdir(parents=True)
    policy_file.write_text(json.dumps({"enabled": True, "read_roots": [], "write_roots": [],
                                       "denied_paths": [str(tmp_path / "private")]}))
    filesystem.save_policy([tmp_path], [])
    assert json.loads(policy_file.read_text())["denied_paths"] == [str(tmp_path / "private")]


def test_known_folders_finds_the_desktop(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(filesystem, "_windows_known_folder", lambda guid: None)
    (tmp_path / "Desktop").mkdir()
    assert filesystem.known_folders() == {"Desktop": (tmp_path / "Desktop").resolve()}


@pytest.fixture
def desk(tmp_path):
    source, desktop = tmp_path / "runs", tmp_path / "Desktop"
    source.mkdir(), desktop.mkdir()
    figure = source / "kinetics.png"
    figure.write_bytes(b"\x89PNG fake")
    return FileAccess(FilePolicy((source, desktop), (desktop,))), figure, desktop


def test_a_figure_is_copied_onto_the_desktop(desk):
    files, figure, desktop = desk
    result = json.loads(files.copy_file(str(figure), str(desktop)))
    assert Path(result["to"]) == desktop / "kinetics.png"
    assert (desktop / "kinetics.png").read_bytes() == figure.read_bytes()
    assert not [p for p in desktop.iterdir() if p.name.startswith(".continuum")]


def test_copying_never_overwrites_without_being_told(desk):
    files, figure, desktop = desk
    (desktop / "kinetics.png").write_bytes(b"older")
    with pytest.raises(Rejected, match="already exists"):
        files.copy_file(str(figure), str(desktop))
    result = json.loads(files.copy_file(str(figure), str(desktop), overwrite=True))
    assert Path(result["backup"]).read_bytes() == b"older"
    assert (desktop / "kinetics.png").read_bytes() == figure.read_bytes()


def test_copying_out_of_an_allowed_folder_is_refused(desk, tmp_path):
    files, figure, _ = desk
    with pytest.raises(Rejected, match="write"):
        files.copy_file(str(figure), str(tmp_path / "runs" / "copy.png"))


def test_documents_open_in_their_application_but_programs_never_do(desk, monkeypatch):
    files, figure, desktop = desk
    launched = []
    monkeypatch.setattr(filesystem, "_launch", launched.append)
    assert "Opened" in files.open_file(str(figure))
    assert launched == [figure.resolve()]

    for name in ("setup.exe", "run.bat", "macro.xlsm", "shortcut.lnk", "script.ps1"):
        (desktop / name).write_bytes(b"x")
        with pytest.raises(Rejected, match="run code"):
            files.open_file(str(desktop / name))
    assert launched == [figure.resolve()]


def test_copy_and_open_ask_the_researcher_first():
    from continuum_spectranalytics.agent.toolspec import build_registry

    tools = build_registry()
    assert tools["copy_file"].tier == "commit"
    assert tools["open_file"].tier == "commit"


# --- Temp and the app's own folder, which Windows keeps under AppData ---------


def test_files_in_temp_can_be_attached_even_under_appdata(monkeypatch, tmp_path, policy_file):
    """Outlook attachments and files opened from a zip live in AppData\\Local\\Temp."""
    temp = tmp_path / "Users" / "me" / "AppData" / "Local" / "Temp"
    opened = temp / "Outlook" / "protocol.docx"
    opened.parent.mkdir(parents=True)
    opened.write_text("x")
    monkeypatch.setattr(filesystem.tempfile, "gettempdir", lambda: str(temp))
    files = FileAccess()
    assert files.grant(opened) == opened.resolve()

    secret = temp / ".ssh" / "id_rsa"
    secret.parent.mkdir()
    secret.write_text("key")
    with pytest.raises(Rejected, match="protected"):
        files.grant(secret)

    elsewhere = tmp_path / "Users" / "me" / "AppData" / "Roaming" / "Browser" / "cookies.txt"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("x")
    with pytest.raises(Rejected, match="protected"):
        files.grant(elsewhere)


def test_a_figure_the_app_drew_can_be_copied_to_the_desktop(monkeypatch, tmp_path, policy_file):
    data = tmp_path / "AppData" / "Local" / "Continuum Spectranalytics" / "data"
    figure = data / "runs" / "figures" / "index_run12.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"png")
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    monkeypatch.setattr(filesystem, "app_data_dir", lambda: data.resolve())
    filesystem.save_policy([desktop], [desktop])

    json.loads(FileAccess().copy_file(str(figure), str(desktop)))
    assert (desktop / "index_run12.png").read_bytes() == b"png"


# --- pasting ------------------------------------------------------------------


def test_copy_as_path_and_file_uris_are_files(tmp_path):
    first, second = tmp_path / "My Paper.pdf", tmp_path / "data.csv"
    first.write_text("x"), second.write_text("y")
    assert paths_in_text(f'"{first}"\n"{second}"') == [str(first), str(second)]
    assert paths_in_text(f"copy\n{first.as_uri()}") == [str(first)]


def test_ordinary_text_is_not_mistaken_for_files(tmp_path):
    real = tmp_path / "a.txt"
    real.write_text("x")
    assert paths_in_text("Compare these two runs") == []
    assert paths_in_text(f"{real}\nand some words") == []
    assert paths_in_text("") == []


# --- the window ------------------------------------------------------------------


def _settle(app, seconds=0.2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update()
        time.sleep(0.005)


@pytest.fixture
def app(gui_app, policy_file):
    yield gui_app
    for attachment in gui_app.session.context.attachments:
        gui_app.session.context.attachments.remove(attachment.id)
    gui_app.render_attachments()
    gui_app.entry.delete("1.0", "end")
    gui_app.transcript.clear()


def test_pasting_files_copied_in_explorer_attaches_them(app, tmp_path):
    paper = tmp_path / "paper notes.md"
    paper.write_text("# notes")
    app.clipboard_clear()
    app.clipboard_append(f'"{paper}"')
    assert app.paste_files() == "break"
    _settle(app)
    assert [a.name for a in app.session.context.attachments.unsent()] == ["paper notes.md"]


def test_pasting_text_still_pastes_text(app):
    app.clipboard_clear()
    app.clipboard_append("just a sentence")
    assert app.paste_files() is None
    assert not app.session.context.attachments.unsent()


def test_file_access_dialog_ties_saving_to_reading(app, tmp_path):
    from continuum_spectranalytics.gui.widgets import FileAccessDialog

    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    dialog = FileAccessDialog(app, app.fonts, {"Desktop": desktop}, read=[], write=[])
    dialog.write_vars["Desktop"].set(True)
    dialog._write_changed("Desktop")
    assert dialog.read_vars["Desktop"].get() is True
    dialog._accept()
    assert dialog.result == ([str(desktop)], [str(desktop)])


def test_the_built_app_self_test_passes(gui_app):
    from continuum_spectranalytics.selftest import run

    results = run()
    failed = {name: r for name, r in results.items() if r["ok"] != "yes"}
    assert not failed, failed
