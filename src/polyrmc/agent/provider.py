"""Model providers, and the wrapper that makes an outage survivable.

Design section 10.3, enforcing invariant I8: **a model outage MUST cost only
judgment, never correctness.**

That invariant is what the whole arrangement is for. Every model-dependent step
in this pipeline chooses among options that deterministic code has already
certified as safe, so when the model is unreachable the run does not stop and
does not degrade -- it takes the most conservative option in the band and
records that it did. The result stays scientifically valid. What is lost is
taste, not correctness.

The escalation order on failure is fixed:

1. Retry with backoff, for transient errors only. A rejected request is not
   transient and retrying it just wastes time.
2. Fail over to a secondary provider, if one is configured. The judge is
   provider-agnostic by construction: table or image in, choice out.
3. Fall back to the deterministic default **inside the already-safe band**, and
   log ``fallback_taken``.

A provider is anything callable that takes a rendered prompt and returns a
choice. No provider SDK is imported at module scope, so the offline path -- the
one the test suite and every deterministic run take -- needs neither a package
nor a key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# Errors worth retrying: the request never reached a model, or the service was
# briefly unavailable. A refusal or a malformed-request error is deterministic
# and will refuse identically on the next attempt.
TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "unavailable",
    "overloaded",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
)


class ProviderError(RuntimeError):
    """A provider call failed. Carries whether the failure looked transient."""

    def __init__(self, message: str, transient: bool | None = None) -> None:
        super().__init__(message)
        self.transient = (
            is_transient(message) if transient is None else bool(transient)
        )


def is_transient(message: str) -> bool:
    lowered = str(message).lower()
    return any(marker in lowered for marker in TRANSIENT_MARKERS)


class Provider(Protocol):
    """Anything that can answer a rendered prompt.

    ``image`` is optional so the same protocol serves both judge lanes: the
    numeric lane sends a table, the visual lane sends a rendered plot, and
    neither knows which provider is behind it.
    """

    name: str

    def __call__(
        self, system: str, prompt: str, image: bytes | None = None
    ) -> str: ...


@dataclass
class CallOutcome:
    """What happened on one guarded call."""

    value: Any
    provider: str
    attempts: int
    fell_back: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class Resilient:
    """Guards one model call, and never lets its failure reach the science.

    Parameters
    ----------
    primary:
        The provider to try first. ``None`` means no model is configured at all,
        which is a normal state, not an error -- the run goes straight to the
        deterministic fallback.
    secondary:
        Optional failover. Tried once, after the primary is exhausted.
    max_attempts:
        Attempts against the primary, including the first.
    backoff_seconds:
        Base delay, doubled per retry.
    """

    primary: Provider | None = None
    secondary: Provider | None = None
    max_attempts: int = 3
    backoff_seconds: float = 0.5
    sleep: Callable[[float], None] = time.sleep

    def call(
        self,
        system: str,
        prompt: str,
        fallback: Callable[[], Any],
        parse: Callable[[str], Any] | None = None,
        image: bytes | None = None,
    ) -> CallOutcome:
        """Try the model; on exhaustion, take the deterministic fallback.

        ``fallback`` is a zero-argument callable rather than a value so the
        conservative choice is computed from the safe band at the moment it is
        needed -- it can never be a stale answer from before the band was known.
        """
        errors: list[str] = []
        attempts = 0

        for provider in (self.primary, self.secondary):
            if provider is None:
                continue
            limit = self.max_attempts if provider is self.primary else 1
            for attempt in range(limit):
                attempts += 1
                try:
                    raw = provider(system, prompt, image)
                    value = parse(raw) if parse else raw
                    return CallOutcome(value, provider.name, attempts, False, errors)
                except Exception as error:  # provider SDKs raise their own types
                    message = f"{provider.name}: {error}"
                    errors.append(message)
                    if not is_transient(str(error)):
                        # Deterministic failure. Retrying reproduces it, so move
                        # on to the next provider rather than burning the budget.
                        break
                    if attempt < limit - 1:
                        self.sleep(self.backoff_seconds * (2**attempt))

        return CallOutcome(fallback(), "deterministic-fallback", attempts, True, errors)


class StubProvider:
    """A scripted provider, for tests and for offline demonstration.

    Exists so the graph can be exercised end to end with no network, no key, and
    no nondeterminism. Raising is scripted too, because the failure paths are
    the ones that matter most here.
    """

    def __init__(self, responses: list[str | Exception], name: str = "stub") -> None:
        self.responses = list(responses)
        self.name = name
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(
        self, system: str, prompt: str, image: bytes | None = None
    ) -> str:
        self.calls.append((system, prompt, image is not None))
        if not self.responses:
            raise ProviderError(f"{self.name}: no scripted response left")
        reply = self.responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


__all__ = [
    "TRANSIENT_MARKERS",
    "CallOutcome",
    "Provider",
    "ProviderError",
    "Resilient",
    "StubProvider",
    "is_transient",
]
