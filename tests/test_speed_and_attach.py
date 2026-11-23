"""Choosing a model, keeping requests small, and attaching dropped files."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from continuum_spectranalytics import filesystem
from continuum_spectranalytics.agent.orchestrator import (
    STALE_RESULT_CHARS, ApprovalPolicy, Orchestrator, allow_all,
)
from continuum_spectranalytics.agent.toolcall import (
    SPECS, AssistantTurn, LangChainToolProvider, Message, ScriptedProvider, ToolCall,
)
from continuum_spectranalytics.agent.toolspec import Tool, ToolContext, build_registry
from continuum_spectranalytics.filesystem import FileAccess, FilePolicy
from continuum_spectranalytics.memory.notes import NoteBook
from continuum_spectranalytics.state import RunState
from continuum_spectranalytics.tools import Rejected, ToolSurface


# --- prompt caching -------------------------------------------------------


class RecordingClient:
    def __init__(self) -> None:
        self.payload = None
        self.kwargs = None

    def bind_tools(self, tools):
        return self

    def invoke(self, payload, **kwargs):
        from langchain_core.messages import AIMessage

        self.payload, self.kwargs = payload, kwargs
        return AIMessage(content="ok")


def test_claude_requests_cache_the_instructions_and_the_history():
    client = RecordingClient()
    provider = LangChainToolProvider(client, name="claude", model="claude-sonnet-5-5")
    provider.converse("RULES", [Message(role="user", text="hi")], [{"type": "function"}])

    system = client.payload[0].content
    assert system == [{"type": "text", "text": "RULES", "cache_control": {"type": "ephemeral"}}]
    assert client.kwargs == {"cache_control": {"type": "ephemeral"}}


@pytest.mark.parametrize("name", ["gemini", "openai"])
def test_other_providers_are_sent_plain_requests(name):
    client = RecordingClient()
    provider = LangChainToolProvider(client, name=name, model="m")
    provider.converse("RULES", [Message(role="user", text="hi")], [])
    assert client.payload[0].content == "RULES"
    assert client.kwargs == {}


def test_every_provider_offers_its_default_model_in_the_picker():
    for spec in SPECS.values():
        assert spec.suggested_models
        assert spec.default_model in spec.suggested_models


# --- history compaction ---------------------------------------------------


@pytest.fixture
def context(run_config, tmp_path) -> ToolContext:
    surface = ToolSurface(state=RunState(run_id="t", source_file=str(run_config.source_file)), config=run_config)
    return ToolContext(surface=surface, notes=NoteBook(tmp_path / "memory"), runs_dir=tmp_path / "runs")


def test_earlier_questions_tool_results_are_shortened_when_resent(context):
    big = "x" * (STALE_RESULT_CHARS * 4)
    registry = build_registry()
    registry["bulky"] = Tool("bulky", "read", "returns a lot", {"type": "object", "properties": {}},
                             lambda ctx, args: big)
    provider = ScriptedProvider([
        AssistantTurn(tool_calls=[ToolCall("c1", "bulky", {})]),
        AssistantTurn(text="first answer"),
        AssistantTurn(text="second answer"),
    ])
    agent = Orchestrator(context=context, provider=provider, registry=registry,
                         approvals=ApprovalPolicy(ask=allow_all))
    agent.ask("first question")
    # Within the question that produced it, the result goes out whole.
    within = provider.calls[1][1][-1].results[0].content
    assert within == big

    agent.ask("second question")
    resent = [m for m in provider.calls[2][1] if m.role == "tool"][0].results[0].content
    assert len(resent) < STALE_RESULT_CHARS + 200
    assert "omitted" in resent
    # The record itself is untouched.
    assert agent.messages[2].results[0].content == big


# --- attached files ---------------------------------------------------------


@pytest.fixture
def no_policy(monkeypatch, tmp_path):
    """The shipped default: no filesystem-permissions.json at all."""
    monkeypatch.setattr(filesystem, "policy_path", lambda: tmp_path / "absent" / filesystem.POLICY_NAME)


def test_a_dropped_file_is_readable_without_a_policy_file(no_policy, tmp_path):
    dropped = tmp_path / "My Data" / "trace.csv"
    dropped.parent.mkdir()
    dropped.write_text("t,ls8\n0,1\n")
    other = tmp_path / "other.txt"
    other.write_text("private")

    files = FileAccess()
    with pytest.raises(Rejected, match="File access unavailable"):
        files.read_file(str(dropped))

    files.grant(str(dropped))
    assert json.loads(files.read_file(str(dropped)))["content"].startswith("t,ls8")
    with pytest.raises(Rejected, match="outside"):
        files.read_file(str(other))
    # Attaching is read-only.
    with pytest.raises(Rejected):
        files.write_file(str(dropped), "overwritten")


def test_a_dropped_folder_can_be_listed_and_read(no_policy, tmp_path):
    folder = tmp_path / "runs"
    folder.mkdir()
    (folder / "a.txt").write_text("a")
    files = FileAccess()
    files.grant(folder)
    names = [e["name"] for e in json.loads(files.list_directory(str(folder)))["entries"]]
    assert names == ["a.txt"]


def test_credentials_cannot_be_attached(no_policy, tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("KEY=1")
    with pytest.raises(Rejected, match="protected"):
        FileAccess().grant(secret)


def test_attaching_extends_an_enabled_policy_instead_of_replacing_it(tmp_path):
    allowed, dropped_dir = tmp_path / "allowed", tmp_path / "elsewhere"
    allowed.mkdir(), dropped_dir.mkdir()
    (allowed / "a.txt").write_text("a")
    (dropped_dir / "b.txt").write_text("b")
    files = FileAccess(FilePolicy((allowed,), (allowed,)))
    files.grant(dropped_dir / "b.txt")
    assert json.loads(files.read_file(str(allowed / "a.txt")))["content"] == "a"
    assert json.loads(files.read_file(str(dropped_dir / "b.txt")))["content"] == "b"


# --- the window -------------------------------------------------------------


def _settle(app, seconds=0.2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update()
        time.sleep(0.005)


@pytest.fixture
def app(gui_app, no_policy):
    yield gui_app
    gui_app.entry.delete("1.0", "end")
    gui_app.transcript.clear()


def test_dropping_a_file_on_the_window_attaches_it(app, tmp_path):
    pytest.importorskip("tkinterdnd2")
    assert app.drops_enabled, "tkdnd should load wherever tkinterdnd2 is installed"
    dropped = tmp_path / "Run 12" / "raw trace.csv"
    dropped.parent.mkdir()
    dropped.write_text("t,ls8\n")

    # Run the exact script tkdnd runs on a drop, with Tk's brace quoting for spaces.
    script = app.tk.call("bind", app.entry._w, "<<Drop>>")
    assert script, "the message box should be a drop target"
    app.tk.eval(str(script).replace("%D", "{{%s}}" % dropped.as_posix()).replace("%A", "copy"))
    _settle(app)

    shelf = app.session.context.attachments
    assert [a.path for a in shelf.unsent()] == [dropped.resolve()]
    chips = " ".join(str(w.cget("text")) for chip in app.attach_bar.winfo_children()
                     for w in chip.winfo_children() if hasattr(w, "cget"))
    assert "raw trace.csv" in chips
    assert dropped.resolve() in app.session.context.files.granted
    assert json.loads(app.session.context.files.read_file(str(dropped)))["content"] == "t,ls8\n"
    for attachment in shelf:
        shelf.remove(attachment.id)
    app.render_attachments()


def test_a_protected_file_is_refused_in_the_transcript(app, tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("KEY=1")
    app.attach_files([str(secret)])
    _settle(app)
    assert not app.session.context.attachments.unsent()
    assert "cannot attach" in app.transcript.plain().lower()


def test_model_settings_returns_the_chosen_model(app, monkeypatch):
    from continuum_spectranalytics.gui.widgets import SignInDialog

    monkeypatch.delenv(SPECS["claude"].model_env, raising=False)
    dialog = SignInDialog(app, app.fonts, SPECS, keyed=["claude"], current=None)
    dialog.choice.set("claude")
    dialog._retarget()
    assert "claude-haiku-5-5" in dialog.model_box.cget("values")
    assert dialog.model.get() == SPECS["claude"].default_model
    dialog.model.set("claude-haiku-5-5")
    dialog._accept()  # no key typed: the saved one is kept
    assert dialog.result == ("claude", "", False, "claude-haiku-5-5")


def test_model_settings_needs_a_key_for_an_unkeyed_provider(app):
    from continuum_spectranalytics.gui.widgets import SignInDialog

    dialog = SignInDialog(app, app.fonts, SPECS, keyed=[], current=None)
    dialog.choice.set("openai")
    dialog._retarget()
    dialog._accept()
    assert dialog.result is None
    assert "needs an API key" in dialog.key_hint.cget("text")
    dialog._skip()
