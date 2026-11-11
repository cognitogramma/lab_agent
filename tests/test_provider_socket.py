"""The bring-your-own-tokens provider socket.

Every test here runs with no key, no package import from a provider SDK, and no
network. That is the point being protected: the socket has to be checkable
without anyone spending anything, because the researcher's own money is what a
live test would spend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from continuum_spectranalytics.agent.toolcall import (
    SPECS,
    LangChainToolProvider,
    Message,
    ProviderCallFailed,
    ProviderUnavailable,
    build_provider,
    describe_providers,
    explain_provider_error,
    key_for,
    keyed_providers,
    redact,
)

ALL_KEYS = sorted({name for spec in SPECS.values() for name in spec.env_keys})


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    """Start every test with an unkeyed machine.

    Without this the suite passes or fails depending on whose shell it runs in:
    a developer with GEMINI_API_KEY exported would never see the no-key paths,
    which are exactly the paths a new labmate hits first.
    """
    for name in ALL_KEYS:
        monkeypatch.delenv(name, raising=False)
    for spec in SPECS.values():
        monkeypatch.delenv(spec.model_env, raising=False)


class FakeClient:
    """Stands in for a LangChain chat client, without one being installed."""

    def __init__(self, reply=None, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, payload):
        if self.error is not None:
            raise self.error
        return self.reply


# --- key discovery -------------------------------------------------------


def test_gemini_key_is_found_under_either_spelling(monkeypatch):
    """Google's console says GEMINI_API_KEY; its SDK reads GOOGLE_API_KEY."""
    spec = SPECS["gemini"]
    assert key_for(spec) is None

    monkeypatch.setenv("GEMINI_API_KEY", "k-second-spelling")
    assert key_for(spec) == "k-second-spelling"

    # The canonical name wins when both are set, so a stale second spelling
    # cannot quietly shadow the one the docs told the researcher to use.
    monkeypatch.setenv("GOOGLE_API_KEY", "k-canonical")
    assert key_for(spec) == "k-canonical"


def test_keyed_providers_reports_only_what_is_set(monkeypatch):
    assert keyed_providers() == []
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert keyed_providers() == ["openai"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert keyed_providers() == ["claude", "openai"]


def test_describe_providers_never_prints_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secretvalue123456")
    rendered = describe_providers()

    assert "secretvalue" not in rendered
    assert "sk-ant" not in rendered
    assert "claude" in rendered and "keyed" in rendered
    # An unkeyed provider names the variable and where to buy one.
    assert "OPENAI_API_KEY" in rendered
    assert "platform.openai.com" in rendered


# --- resolution ----------------------------------------------------------


def test_unknown_provider_lists_the_real_ones():
    with pytest.raises(ProviderUnavailable) as caught:
        build_provider("llama")

    message = str(caught.value)
    assert "llama" in message
    for name in SPECS:
        assert name in message


def test_missing_key_names_the_variable_and_where_to_get_one():
    with pytest.raises(ProviderUnavailable) as caught:
        build_provider("claude")

    message = str(caught.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "console.anthropic.com" in message
    assert "No provider is keyed on this machine yet." in message


def test_missing_key_points_at_a_provider_that_is_keyed(monkeypatch):
    """A researcher holding one key should be told they hold it."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    with pytest.raises(ProviderUnavailable) as caught:
        build_provider("openai")

    assert "Keyed here already: gemini" in str(caught.value)


def test_missing_package_reports_the_pip_line(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(
        "continuum_spectranalytics.agent.toolcall._client_for",
        lambda *a, **kw: (_ for _ in ()).throw(ImportError("no module")),
    )

    with pytest.raises(ProviderUnavailable) as caught:
        build_provider("openai")

    assert "pip install langchain-openai" in str(caught.value)


@pytest.mark.parametrize("kind", sorted(SPECS))
def test_model_pin_resolution_order(monkeypatch, kind):
    """Explicit --model beats the environment, which beats the pinned default."""
    spec = SPECS[kind]
    monkeypatch.setenv(spec.canonical_key, "k")
    seen: list[str] = []
    monkeypatch.setattr(
        "continuum_spectranalytics.agent.toolcall._client_for",
        lambda s, model, key, temp: seen.append(model) or FakeClient(),
    )

    assert build_provider(kind).model == spec.default_model

    monkeypatch.setenv(spec.model_env, "from-environment")
    assert build_provider(kind).model == "from-environment"

    assert build_provider(kind, "from-argument").model == "from-argument"
    assert seen == [spec.default_model, "from-environment", "from-argument"]


def test_built_provider_carries_the_key_it_was_given(monkeypatch):
    """The key is passed explicitly rather than left to each SDK's env lookup."""
    monkeypatch.setenv("GEMINI_API_KEY", "k-only-under-the-second-name")
    captured = {}

    def record(spec, model, api_key, temperature):
        captured.update(spec=spec.name, model=model, api_key=api_key)
        return FakeClient()

    monkeypatch.setattr("continuum_spectranalytics.agent.toolcall._client_for", record)
    provider = build_provider("gemini")

    assert captured["api_key"] == "k-only-under-the-second-name"
    assert provider.name == "gemini"


# --- failure, and the absence of failover --------------------------------


def test_provider_fault_becomes_a_typed_failure():
    """A dead brain raises its own type, so a caller can end the session on it."""
    provider = LangChainToolProvider(
        FakeClient(error=RuntimeError("429 quota exceeded")),
        name="gemini",
        model="gemini-3.6-flash",
    )

    with pytest.raises(ProviderCallFailed) as caught:
        provider.converse("system", [Message(role="user", text="hi")], [])

    assert caught.value.provider == "gemini"
    assert caught.value.model == "gemini-3.6-flash"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("429 RESOURCE_EXHAUSTED", "out of quota"),
        ("insufficient_quota: check your billing", "out of quota"),
        ("401 Unauthorized", "was rejected"),
        ("invalid_api_key supplied", "was rejected"),
        ("404 model not found", "model id was rejected"),
        ("Connection timed out", "could not be reached"),
    ],
)
def test_failure_explanations_name_the_fix(raw, expected):
    message = explain_provider_error(RuntimeError(raw), "openai", "gpt-5")

    assert expected in message
    assert "openai (gpt-5) failed" in message
    # The provider's own words are kept: they are the only ground truth about
    # what actually happened.
    assert raw in message


def test_unrecognised_failure_still_names_the_provider_and_type():
    message = explain_provider_error(ValueError("something odd"), "claude", "m")

    assert "claude (m) failed" in message
    assert "ValueError" in message
    assert "something odd" in message


def test_a_key_in_an_error_is_redacted_before_it_reaches_a_terminal():
    """Session transcripts get pasted into lab channels."""
    leaked = RuntimeError("rejected key sk-ant-api03-ABCDEF123456 for model x")
    message = explain_provider_error(leaked, "claude", "claude-sonnet-5")

    assert "sk-ant-api03-ABCDEF123456" not in message
    assert "<redacted>" in message

    assert redact("AIzaSyABCDEFGHIJKLMNOP") == "<redacted>"
    assert redact("nothing secret here") == "nothing secret here"


def test_there_is_no_failover_between_providers(monkeypatch):
    """Two keys present must not mean a dead session silently spends the other.

    Failover would be a spending decision made on the researcher's behalf. The
    contract is that a fault raises and the caller stops.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    attempts: list[str] = []

    def record(spec, model, api_key, temperature):
        attempts.append(spec.name)
        return FakeClient(error=RuntimeError("429 quota exceeded"))

    monkeypatch.setattr("continuum_spectranalytics.agent.toolcall._client_for", record)
    provider = build_provider("gemini")

    with pytest.raises(ProviderCallFailed):
        provider.converse("system", [Message(role="user", text="hi")], [])

    assert attempts == ["gemini"]


# --- signing in ----------------------------------------------------------


def test_persist_key_replaces_rather_than_appends(tmp_path: Path):
    from continuum_spectranalytics.cli import _persist_key

    env = tmp_path / ".env"
    env.write_text("OTHER=keep-me\nANTHROPIC_API_KEY=old\n", encoding="utf-8")
    spec = SPECS["claude"]

    assert _persist_key(spec, "new", env)
    assert _persist_key(spec, "newer", env)

    lines = env.read_text(encoding="utf-8").splitlines()
    assert "OTHER=keep-me" in lines
    assert lines.count("ANTHROPIC_API_KEY=newer") == 1
    assert not any(line.endswith("=old") for line in lines)
    assert not any(line.endswith("=new") for line in lines)


def test_persist_key_creates_the_file_when_absent(tmp_path: Path):
    from continuum_spectranalytics.cli import _persist_key

    env = tmp_path / ".env"
    assert _persist_key(SPECS["openai"], "k-fresh", env)
    assert env.read_text(encoding="utf-8").strip() == "OPENAI_API_KEY=k-fresh"


def test_every_spec_is_self_consistent():
    """A spec with a blank field produces an error message that helps nobody."""
    for name, spec in SPECS.items():
        assert spec.name == name
        assert spec.env_keys and all(spec.env_keys)
        assert spec.canonical_key == spec.env_keys[0]
        assert spec.default_model and spec.model_env and spec.pip_name
        assert spec.key_url.startswith("https://")
