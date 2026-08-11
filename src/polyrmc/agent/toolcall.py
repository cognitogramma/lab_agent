"""Tool-calling transport: the model asks for actions, and gets results back.

This is what separates an orchestrator from a pipeline. The older
:mod:`polyrmc.agent.provider` speaks text in, text out -- adequate for a judge
that picks an index from a shortlist, and useless for a model that has to decide
*what to do next*. A judge is handed the question; an orchestrator has to find
it. That needs a transport where the model can name a tool, receive what it
returned, and decide again with the answer in hand.

The neutral shapes here -- :class:`Message`, :class:`ToolCall`,
:class:`ToolResult` -- exist so the loop in :mod:`polyrmc.agent.orchestrator`
never learns which provider is behind it. That is not portability for its own
sake. The whole premise of this system is that a model outage costs judgment and
not correctness, and that promise is only checkable if the thing being swapped
out has a boundary you can point at.

One live adapter serves every provider, over LangChain's chat interface, which
normalises tool calls into the same ``AIMessage.tool_calls`` shape whichever
provider answered. Which brains are on offer is declared in :data:`SPECS` --
Gemini, Claude, and OpenAI at present -- and a researcher selects one with
``--provider`` and pays for it with their own key. Nothing here ships a key,
brokers access, or prefers a vendor: the lab's value is in the measurement path
below this module, and the brain above it is a commodity the user buys.

A provider that fails mid-session raises :class:`ProviderCallFailed` and the
session ends. It does not fail over onto a second key, because the second key
is also the researcher's and spending it was never agreed to.

And one scripted adapter, :class:`ScriptedProvider`, which is what the test
suite runs against. The loop is exercised end to end with no network, no key,
and no nondeterminism, because the interesting failures in an agentic loop --
a tool that rejects, a model that repeats itself, a cap that has to hold -- are
all reachable offline and none of them should need a live provider to test.

No provider SDK is imported at module scope.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, Sequence

from polyrmc.agent.budget import Usage

@dataclass(frozen=True)
class ProviderSpec:
    """Everything provider-specific about one brain, in one declaration.

    This lab buys its own tokens. Nobody here sells model access, and the value
    of the system is the measurement path and its audit trail -- so which brain
    drives a session is the researcher's purchase, not this package's opinion.
    That makes the provider a socket, and this table is the whole of it: adding
    one is an entry here plus a branch in :func:`_client_for`, not a new adapter.

    ``env_keys`` is a tuple because a provider can be keyed under more than one
    name in the wild. The first entry is the canonical one -- what a sign-in
    writes, and what the docs name -- and the rest are accepted so a researcher
    who already exported the other spelling is not told they have no key while
    holding one.
    """

    name: str
    env_keys: tuple[str, ...]
    default_model: str
    model_env: str
    pip_name: str
    key_url: str

    @property
    def canonical_key(self) -> str:
        return self.env_keys[0]


# Model ids are pinned rather than discovered: a run has to be able to say which
# brain judged it. Override per session with the provider's ``*_MODEL`` variable
# or ``--model``; a rejected id is reported as such rather than retried.
SPECS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        name="gemini",
        env_keys=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        default_model="gemini-3.6-flash",
        model_env="GEMINI_MODEL",
        pip_name="langchain-google-genai",
        key_url="https://aistudio.google.com/apikey",
    ),
    "claude": ProviderSpec(
        name="claude",
        env_keys=("ANTHROPIC_API_KEY",),
        default_model="claude-sonnet-5",
        model_env="ANTHROPIC_MODEL",
        pip_name="langchain-anthropic",
        key_url="https://console.anthropic.com/settings/keys",
    ),
    "openai": ProviderSpec(
        name="openai",
        env_keys=("OPENAI_API_KEY",),
        default_model="gpt-5",
        model_env="OPENAI_MODEL",
        pip_name="langchain-openai",
        key_url="https://platform.openai.com/api-keys",
    ),
}

# Retained as names because other modules and the docs refer to them, and a
# researcher grepping for the key they need should land somewhere.
DEFAULT_GEMINI_MODEL = SPECS["gemini"].default_model
DEFAULT_CLAUDE_MODEL = SPECS["claude"].default_model
DEFAULT_OPENAI_MODEL = SPECS["openai"].default_model

GEMINI_KEYS = SPECS["gemini"].env_keys
CLAUDE_KEYS = SPECS["claude"].env_keys
OPENAI_KEYS = SPECS["openai"].env_keys


class ProviderUnavailable(RuntimeError):
    """No usable model client. Says which key or package is missing.

    Raised at construction rather than at first call, so a session that cannot
    reach a model says so before a researcher has typed a question into it.
    """


class ProviderCallFailed(RuntimeError):
    """A configured provider refused or could not be reached mid-session.

    Separate from :class:`ProviderUnavailable`, which means the session never
    started. This one ends a session that was already running, and it is a
    distinct type because the caller has to tell it apart from a tool fault: a
    tool that refuses is fed back to the model to correct, while a brain that
    will not answer means there is nothing left to correct with.

    There is deliberately no failover onto a second key. Every key here was
    bought by the person at the terminal, and quietly moving their session onto
    a different one spends money they did not agree to spend.
    """

    def __init__(self, message: str, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


@dataclass(frozen=True)
class ToolCall:
    """One action the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """One line, for the transcript a person reads."""
        if not self.arguments:
            return f"{self.name}()"
        inner = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({inner})"


@dataclass(frozen=True)
class ToolResult:
    """What a tool returned, on its way back to the model.

    ``ok`` is false for a refusal. A refusal is still a result and is still fed
    back: the commit tier writes its rejections as recovery instructions --
    what failed, by how much, what would be valid instead -- and a model that
    never sees them will retry the same invalid action instead of correcting.
    """

    call_id: str
    name: str
    content: str
    ok: bool = True


@dataclass
class Message:
    """One entry in the conversation, in provider-neutral form."""

    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)


@dataclass
class AssistantTurn:
    """What the model said, what it wants run, and what it cost."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = "unknown"
    usage: Usage = field(default_factory=lambda: Usage(requests=1))

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ToolCallingProvider(Protocol):
    """Anything that can continue a conversation, with tools available."""

    name: str
    model: str

    def converse(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantTurn: ...


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def _first_key(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


class LangChainToolProvider:
    """Adapter over any LangChain chat model that supports ``bind_tools``.

    One implementation serves both providers. LangChain already normalises the
    wire differences -- Gemini's ``functionCall`` parts and Claude's
    ``tool_use`` blocks both arrive as ``AIMessage.tool_calls`` -- so writing
    two adapters would mean maintaining two copies of the same translation for
    no gain.
    """

    def __init__(self, client: Any, name: str, model: str) -> None:
        self.client = client
        self.name = name
        self.model = model

    # --- translation ---------------------------------------------------------

    @staticmethod
    def _to_langchain(messages: Sequence[Message]) -> list[Any]:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        converted: list[Any] = []
        for message in messages:
            if message.role == "user":
                converted.append(HumanMessage(content=message.text))
            elif message.role == "assistant":
                converted.append(
                    AIMessage(
                        content=message.text,
                        tool_calls=[
                            {
                                "name": call.name,
                                "args": dict(call.arguments),
                                "id": call.id,
                                "type": "tool_call",
                            }
                            for call in message.tool_calls
                        ],
                    )
                )
            else:
                # Every result becomes its own ToolMessage. Both providers pair
                # results to calls by id and will error on a call left
                # unanswered, so the loop must return one per call even when a
                # tool refused.
                for result in message.results:
                    converted.append(
                        ToolMessage(
                            content=result.content,
                            tool_call_id=result.call_id,
                            name=result.name,
                            status="success" if result.ok else "error",
                        )
                    )
        return converted

    def converse(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantTurn:
        from langchain_core.messages import SystemMessage

        bound = self.client.bind_tools(list(tools)) if tools else self.client
        payload = [SystemMessage(content=system), *self._to_langchain(messages)]
        try:
            reply = bound.invoke(payload)
        except Exception as error:  # provider SDKs raise their own types
            raise ProviderCallFailed(
                explain_provider_error(error, self.name, self.model),
                provider=self.name,
                model=self.model,
            ) from error

        calls = [
            ToolCall(
                id=str(raw.get("id") or _new_call_id()),
                name=str(raw.get("name", "")),
                arguments=dict(raw.get("args") or {}),
            )
            for raw in getattr(reply, "tool_calls", []) or []
        ]
        return AssistantTurn(
            text=_text_of(reply),
            tool_calls=calls,
            model=self.model,
            usage=_usage_of(reply),
        )


def _usage_of(reply: Any) -> Usage:
    """Read the token counts off a reply.

    Absence is reported as zero tokens against one request rather than raising.
    A provider that stops populating ``usage_metadata`` should cost the session
    its accounting, not its answer -- and the request count, which is what the
    free tier actually rations, is known either way.
    """
    metadata = getattr(reply, "usage_metadata", None) or {}
    try:
        return Usage(
            requests=1,
            input_tokens=int(metadata.get("input_tokens") or 0),
            output_tokens=int(metadata.get("output_tokens") or 0),
        )
    except (AttributeError, TypeError, ValueError):
        return Usage(requests=1)


def _text_of(reply: Any) -> str:
    """Flatten a reply's content to text.

    Content is a plain string from Gemini and a list of typed blocks from
    Claude, and a turn that requests a tool often carries no text at all.
    """
    content = getattr(reply, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content or "")


# Credential shapes the three providers issue. An SDK error sometimes quotes the
# request it failed on, and a session transcript is something a researcher pastes
# into a lab channel when asking for help.
_KEY_SHAPES = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,}|AIza[A-Za-z0-9_\-]{8,})")

# Ordered because the first match wins and the categories overlap in the wild:
# a quota refusal is served as 403 by one provider and 429 by another.
_ERROR_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("insufficient_quota", "quota", "429", "rate limit", "resource_exhausted", "billing"),
        "That key is out of quota or over its rate limit. This is a "
        "buy-your-own-tokens setup, so the fix is on the account the key belongs "
        "to -- top it up, wait out the window, or start again on another "
        "--provider you hold a key for.",
    ),
    (
        ("401", "403", "unauthorized", "permission_denied", "invalid_api_key", "api key not valid"),
        "That key was rejected. Check it is current and pasted whole, and that "
        "it belongs to the provider you selected.",
    ),
    (
        ("404", "not found", "does not exist", "unknown model", "invalid model"),
        "That model id was rejected. Pin a different one with --model, or set "
        "the provider's *_MODEL variable; the default here is a pin, not a "
        "discovery, and pins go stale.",
    ),
    (
        ("timeout", "timed out", "connection", "unavailable", "503", "502", "504"),
        "The provider could not be reached. This one is usually transient -- the "
        "session stops rather than retrying, so nothing is spent twice.",
    ),
)


def redact(text: str) -> str:
    """Blank anything shaped like a credential before it reaches a terminal."""
    return _KEY_SHAPES.sub("<redacted>", str(text))


def explain_provider_error(error: Exception, provider: str, model: str) -> str:
    """Turn an SDK exception into something a researcher can act on.

    The raw text is kept -- it is the only ground truth about what the provider
    said -- but it is led with the category, because the useful distinction is
    whether the fix is a key, a model id, a bill, or a wait.
    """
    raw = redact(error)
    lowered = raw.lower()
    for markers, hint in _ERROR_HINTS:
        if any(marker in lowered for marker in markers):
            return f"{provider} ({model}) failed: {hint}\n  provider said: {raw}"
    return f"{provider} ({model}) failed: {type(error).__name__}: {raw}"


def key_for(spec: ProviderSpec) -> str | None:
    """The key this provider would use, under whichever name it is set."""
    return _first_key(spec.env_keys)


def keyed_providers() -> list[str]:
    """Providers a session could start on right now, in table order."""
    return [name for name, spec in SPECS.items() if key_for(spec)]


def describe_providers() -> str:
    """Which brains are keyed and which are not, for the CLI to print.

    Deliberately reports only presence. A key is a credential the researcher
    paid for; this package reads it, passes it to the provider, and never prints
    it, logs it, or writes it to a run sidecar.
    """
    rows = []
    for spec in SPECS.values():
        if key_for(spec):
            rows.append(f"  {spec.name:<8} keyed ({spec.canonical_key})")
        else:
            rows.append(f"  {spec.name:<8} no key -- set {spec.canonical_key}, from {spec.key_url}")
    return "\n".join(rows)


def _client_for(spec: ProviderSpec, model: str, api_key: str, temperature: float) -> Any:
    """Construct the LangChain client for one provider.

    All three accept the key as ``api_key`` and it is passed explicitly rather
    than left to each SDK's environment lookup. That is not redundancy: Gemini's
    client reads only ``GOOGLE_API_KEY``, so a researcher whose key sits in
    ``GEMINI_API_KEY`` -- the spelling Google's own console suggests -- would be
    told they had no key while holding one. That already happened here once.
    """
    if spec.name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)
    if spec.name == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)
    if spec.name == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    raise ProviderUnavailable(f"no client constructor for {spec.name!r}")


def build_provider(
    kind: str, model: str | None = None, temperature: float = 0.2
) -> ToolCallingProvider:
    """Resolve a provider by name, or say exactly what is missing.

    The two ways this fails are the two a researcher can act on: no key, or no
    package. Both name the fix in the message, because the alternative is a
    stack trace from inside an SDK that assumes the reader chose it.
    """
    spec = SPECS.get(kind)
    if spec is None:
        raise ProviderUnavailable(
            f"{kind!r} is not a known provider. Available: {sorted(SPECS)}."
        )

    api_key = key_for(spec)
    if not api_key:
        keyed = keyed_providers()
        alternative = (
            f" Keyed here already: {', '.join(keyed)}."
            if keyed
            else " No provider is keyed on this machine yet."
        )
        raise ProviderUnavailable(
            f"{spec.name} needs {spec.canonical_key}. Get one from {spec.key_url} "
            f"and put it in .env, or pick another --provider.{alternative}"
        )

    name = model or os.environ.get(spec.model_env) or spec.default_model
    try:
        client = _client_for(spec, name, api_key, temperature)
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ProviderUnavailable(
            f"{spec.pip_name} is not installed. pip install {spec.pip_name}"
        ) from error
    return LangChainToolProvider(client, name=spec.name, model=name)


def gemini_provider(model: str | None = None, temperature: float = 0.2) -> ToolCallingProvider:
    """Build the Gemini orchestrator client, or say what is missing."""
    return build_provider("gemini", model, temperature)


def claude_provider(model: str | None = None, temperature: float = 0.2) -> ToolCallingProvider:
    """Build the Claude orchestrator client, or say what is missing."""
    return build_provider("claude", model, temperature)


def openai_provider(model: str | None = None, temperature: float = 0.2) -> ToolCallingProvider:
    """Build the OpenAI orchestrator client, or say what is missing."""
    return build_provider("openai", model, temperature)


PROVIDERS = {
    "gemini": gemini_provider,
    "claude": claude_provider,
    "openai": openai_provider,
}


class ScriptedProvider:
    """A provider that replays a fixed list of turns.

    The whole orchestrator is testable through this: hand it a turn that calls
    a tool, then a turn that reads the result and answers, and the loop runs
    with no network and no nondeterminism. Running out of script raises rather
    than returning something plausible, because a loop that made one more call
    than the test expected is exactly the bug worth failing on.
    """

    name = "scripted"

    def __init__(self, turns: Sequence[AssistantTurn | str], model: str = "scripted") -> None:
        self.turns = [
            AssistantTurn(text=t) if isinstance(t, str) else t for t in turns
        ]
        self.model = model
        self.calls: list[tuple[str, list[Message], list[dict[str, Any]]]] = []

    def converse(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantTurn:
        self.calls.append((system, list(messages), list(tools)))
        if not self.turns:
            raise ProviderUnavailable(
                "scripted provider is exhausted: the loop asked for another turn "
                "than the script provides"
            )
        turn = self.turns.pop(0)
        # Filled in here so tests can write ToolCall(id="", ...) and still get
        # a well-formed pairing back.
        turn.tool_calls = [
            call if call.id else ToolCall(_new_call_id(), call.name, call.arguments)
            for call in turn.tool_calls
        ]
        return turn


def as_openai_tools(schemas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise plain ``{name, description, parameters}`` dicts for binding."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return [convert_to_openai_tool(schema) for schema in schemas]


def summarise_arguments(arguments: dict[str, Any], limit: int = 120) -> str:
    """Render call arguments compactly for a terminal line."""
    if not arguments:
        return ""
    text = json.dumps(arguments, default=str, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = [
    "CLAUDE_KEYS",
    "Usage",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "GEMINI_KEYS",
    "OPENAI_KEYS",
    "PROVIDERS",
    "SPECS",
    "AssistantTurn",
    "LangChainToolProvider",
    "Message",
    "ProviderCallFailed",
    "ProviderSpec",
    "ProviderUnavailable",
    "ScriptedProvider",
    "ToolCall",
    "ToolCallingProvider",
    "ToolResult",
    "as_openai_tools",
    "build_provider",
    "claude_provider",
    "describe_providers",
    "explain_provider_error",
    "gemini_provider",
    "key_for",
    "keyed_providers",
    "openai_provider",
    "redact",
    "summarise_arguments",
]
