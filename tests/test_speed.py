"""Speed: how long the model thinks, live text, and where each answer's time went."""
from __future__ import annotations

import pytest

from continuum_spectranalytics.agent.budget import Usage
from continuum_spectranalytics.agent.orchestrator import ApprovalPolicy, Orchestrator, allow_all
from continuum_spectranalytics.agent.toolcall import (
    SPECS, AssistantTurn, LangChainToolProvider, Message, ScriptedProvider, ToolCall,
    _usage_of, build_provider,
)
from continuum_spectranalytics.agent.toolspec import ToolContext, build_registry
from continuum_spectranalytics.memory.notes import NoteBook
from continuum_spectranalytics.state import RunState
from continuum_spectranalytics.tools import ToolSurface


# --- the Speed setting --------------------------------------------------------


@pytest.mark.parametrize("speed, effort", [("fast", "low"), ("balanced", "medium"),
                                           ("thorough", "high"), ("default", None)])
def test_speed_sets_how_long_each_provider_thinks(monkeypatch, speed, effort):
    seen = {}

    def record(spec, model, key, temp, effort=None):
        seen[spec.name] = effort
        return object()

    monkeypatch.setattr("continuum_spectranalytics.agent.toolcall._client_for", record)
    for spec in SPECS.values():
        monkeypatch.setenv(spec.canonical_key, "k")
        provider = build_provider(spec.name, speed=speed)
        assert provider.speed == speed
    assert seen == {name: effort for name in SPECS}


def test_fast_is_the_default(monkeypatch):
    monkeypatch.delenv("CONTINUUM_SPECTRANALYTICS_SPEED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr("continuum_spectranalytics.agent.toolcall._client_for",
                        lambda *a, **kw: object())
    assert build_provider("claude").speed == "fast"
    monkeypatch.setenv("CONTINUUM_SPECTRANALYTICS_SPEED", "thorough")
    assert build_provider("claude").speed == "thorough"


@pytest.mark.parametrize("name", sorted(SPECS))
def test_the_real_clients_accept_the_setting(monkeypatch, name):
    from continuum_spectranalytics.agent.toolcall import _client_for

    client = _client_for(SPECS[name], SPECS[name].default_model, "not-a-key", 0.2, effort="low")
    assert getattr(client, "reasoning_effort", None) == "low"


class RejectsEffort:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, payload, **kwargs):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return AIMessage(content="ok")


def test_a_model_that_rejects_the_setting_is_retried_without_it():
    strict, plain = RejectsEffort("400 invalid_request: 'effort' is not supported on this model"), RejectsEffort()
    provider = LangChainToolProvider(strict, name="claude", model="old-model", fallback_client=lambda: plain)
    provider.speed = "fast"
    turn = provider.converse("rules", [Message(role="user", text="hi")], [])
    assert turn.text == "ok"
    assert (strict.calls, plain.calls) == (1, 1)
    assert provider.speed == "default"
    provider.converse("rules", [Message(role="user", text="again")], [])
    assert (strict.calls, plain.calls) == (1, 2)  # remembered for the session


# --- streaming --------------------------------------------------------------------


class Streams:
    def bind_tools(self, tools):
        return self

    def stream(self, payload, **kwargs):
        from langchain_core.messages import AIMessageChunk

        yield AIMessageChunk(content="The run ")
        yield AIMessageChunk(content="looks clean.")
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "get_state", "args": "{}", "id": "c1", "index": 0}],
            usage_metadata={"input_tokens": 1200, "output_tokens": 30, "total_tokens": 1230,
                            "input_token_details": {"cache_read": 1000}},
        )


def test_reply_text_arrives_as_it_is_written():
    pieces = []
    provider = LangChainToolProvider(Streams(), name="openai", model="m")
    turn = provider.converse("rules", [Message(role="user", text="hi")], [{"type": "function"}],
                             on_text=pieces.append)
    assert pieces == ["The run ", "looks clean."]
    assert turn.text == "The run looks clean."
    assert [c.name for c in turn.tool_calls] == ["get_state"]
    assert turn.usage.cached_tokens == 1000


def test_cached_tokens_are_counted():
    from langchain_core.messages import AIMessage

    reply = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                                                    "input_token_details": {"cache_read": 7}})
    assert _usage_of(reply) == Usage(requests=1, input_tokens=10, output_tokens=2, cached_tokens=7)


# --- where the time went ---------------------------------------------------------------


@pytest.fixture
def context(run_config, tmp_path) -> ToolContext:
    surface = ToolSurface(state=RunState(run_id="t", source_file=str(run_config.source_file)), config=run_config)
    return ToolContext(surface=surface, notes=NoteBook(tmp_path / "memory"), runs_dir=tmp_path / "runs")


def test_each_answer_reports_where_its_time_went(context):
    provider = ScriptedProvider([
        AssistantTurn(text="Checking.", tool_calls=[ToolCall("c1", "get_state", {})],
                      usage=Usage(requests=1, input_tokens=11000, output_tokens=20, cached_tokens=9000)),
        AssistantTurn(text="Nothing is loaded yet.",
                      usage=Usage(requests=1, input_tokens=11500, output_tokens=40, cached_tokens=11000)),
    ])
    agent = Orchestrator(context=context, provider=provider, registry=build_registry(),
                         approvals=ApprovalPolicy(ask=allow_all))
    streamed = []
    agent.on_text_delta = streamed.append
    agent.ask("where does the run stand?")

    assert "".join(streamed) == "Checking.Nothing is loaded yet."
    timing = agent.last_timing
    assert len(timing.rounds) == 2
    assert timing.rounds[0].tools == ["get_state"]
    assert timing.total >= timing.model_seconds + timing.tool_seconds - 1e-6
    summary = timing.summary()
    assert "model" in summary and "2 calls" in summary
    assert "22,500 tokens in (88% from cache)" in summary
    report = timing.report()
    assert "get_state" in report and "(answer)" in report


def test_providers_without_streaming_still_work(context):
    class Plain:
        name, model = "plain", "plain"

        def converse(self, system, messages, tools):
            return AssistantTurn(text="done")

    agent = Orchestrator(context=context, provider=Plain(), registry=build_registry())
    agent.on_text_delta = lambda piece: None
    assert [e.text for e in agent.ask("hi") if e.kind == "text"] == ["done"]


# --- the window ---------------------------------------------------------------------------


def test_streamed_text_is_replaced_by_the_rendered_reply(gui_app):
    transcript = gui_app.transcript
    transcript.clear()
    transcript.stream("**Bold** draft ")
    transcript.stream("text")
    assert "**Bold** draft text" in transcript.plain()
    transcript.model("**Bold** final text")
    shown = transcript.plain()
    assert "draft" not in shown
    assert "Bold final text" in shown and "**" not in shown
    transcript.clear()


def test_the_bridge_forwards_text_as_it_is_written(context):
    import time

    from continuum_spectranalytics.gui.session import Bridge

    agent = Orchestrator(context=context, provider=ScriptedProvider([AssistantTurn(text="two words")]),
                         registry=build_registry())
    bridge = Bridge(agent=agent, surface=context.surface)
    bridge.submit("hi")
    deltas, deadline = [], time.monotonic() + 10
    while time.monotonic() < deadline:
        kinds = [(s.kind, s.payload) for s in bridge.drain()]
        deltas += [p for k, p in kinds if k == "delta"]
        if any(k == "done" for k, _ in kinds):
            break
        time.sleep(0.01)
    assert "".join(deltas) == "two words"


def test_openai_uses_the_responses_api():
    """gpt-5.5 refuses tools plus a reasoning setting on chat completions."""
    from continuum_spectranalytics.agent.toolcall import _client_for

    client = _client_for(SPECS["openai"], "gpt-5.5", "not-a-key", 0.2, effort="low")
    assert client.use_responses_api is True


def test_view_zero_means_read_the_text(tmp_path, monkeypatch):
    from continuum_spectranalytics import filesystem

    monkeypatch.setattr(filesystem, "policy_path", lambda: tmp_path / "absent.json")
    notes = tmp_path / "notes.md"
    notes.write_text("# Notes")
    context = ToolContext(surface=None, notes=NoteBook(tmp_path / "m"), runs_dir=tmp_path / "r")
    context.attachments.add(notes)
    read = build_registry()["read_attachment"].handler
    for view in (0, None, "", "0"):
        assert "# Notes" in read(context, {"id": "a1", "view": view})
    assert "# Notes" in read(context, {"id": "a1", "part": 0})
