"""What a session costs, and the cap that stops it.

An orchestrator spends differently from a judge. The old design called a model
once per run, to pick a smoothing window from a shortlist, and the bill was a
rounding error nobody needed to see. This one spends a request per tool round
and re-sends the whole transcript each time, so a single question can cost six
requests and a growing prompt. That is the price of letting the model choose the
route, and it is worth paying -- but not blind.

Two things are counted, because two different limits bite.

**Requests.** The Gemini free tier caps ``gemini-3.6-flash`` at 5 per minute and
20 per day, and it is a *request* cap, not a token cap. A researcher who has
asked three questions has spent their day, and no token figure would have warned
them.

**Tokens.** What a paid key is billed on. Input dominates: the transcript is
resent every round, so a ten-round question pays for its own history ten times.
Reported separately from output for that reason -- a session that looks
expensive is usually long, not verbose.

Cost in currency is *derived*, and only when someone supplies a rate. Token
counts are measured and are always right; a price is a claim about a vendor's
pricing page on a particular day, and one hardcoded here would be wrong within a
quarter and would be believed anyway. Set ``POLYRMC_RATES`` to opt in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

RATES_ENV = "POLYRMC_RATES"
"""Per-million-token rates: ``POLYRMC_RATES="in=0.30,out=2.50"``.

Deliberately not defaulted. See the module docstring: an unset rate reports
tokens and no money, which is honest. A stale rate reports money and is not.
"""


@dataclass
class Usage:
    """Tokens and requests spent."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.requests + other.requests,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def add(self, other: "Usage") -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


@dataclass(frozen=True)
class Rates:
    """Price per million tokens, as supplied by whoever is paying."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


def rates_from_env(raw: str | None = None) -> Rates | None:
    """Parse ``in=0.30,out=2.50``, or return None if unset or unreadable.

    A malformed rate yields no cost line rather than a wrong one.
    """
    text = raw if raw is not None else os.environ.get(RATES_ENV, "")
    if not text.strip():
        return None
    values: dict[str, float] = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        try:
            values[key.strip().lower()] = float(value)
        except ValueError:
            return None
    if "in" not in values or "out" not in values:
        return None
    return Rates(values["in"], values["out"])


class BudgetExceeded(RuntimeError):
    """The session hit its own ceiling. Carries what was spent."""


@dataclass
class Budget:
    """A ceiling on one session, checked before each model call.

    Checked *before* rather than after, so the limit is a limit and not a
    notification that it was passed. Zero means no ceiling.
    """

    max_requests: int = 0
    max_tokens: int = 0
    spent: Usage = field(default_factory=Usage)
    rates: Rates | None = field(default_factory=rates_from_env)

    def check(self) -> None:
        if self.max_requests and self.spent.requests >= self.max_requests:
            raise BudgetExceeded(
                f"session request budget reached: {self.spent.requests} of "
                f"{self.max_requests}. Raise it with --budget-requests, or start "
                "a new session."
            )
        if self.max_tokens and self.spent.total_tokens >= self.max_tokens:
            raise BudgetExceeded(
                f"session token budget reached: {self.spent.total_tokens:,} of "
                f"{self.max_tokens:,}. Raise it with --budget-tokens, or start a "
                "new session."
            )

    def record(self, usage: Usage) -> None:
        self.spent.add(usage)

    def render(self, label: str = "session", nudge: bool = True) -> str:
        """One line a researcher paying for this can read at a glance.

        ``nudge`` suppresses the "set POLYRMC_RATES" hint. It is useful advice
        when a session ends normally and noise inside the banner that tells
        someone their key is spent -- the moment they are least interested in
        configuring a cost estimate for it.
        """
        spent = self.spent
        parts = [
            f"{label}: {spent.requests} request(s)",
            f"{spent.input_tokens:,} in",
            f"{spent.output_tokens:,} out",
        ]
        if self.rates is not None:
            parts.append(f"~${self.rates.cost(spent):.4f}")
        elif nudge:
            parts.append(f"set {RATES_ENV} for a cost estimate")
        line = ", ".join(parts)
        if self.max_requests:
            line += f"  [cap {self.max_requests} requests]"
        if self.max_tokens:
            line += f"  [cap {self.max_tokens:,} tokens]"
        return line


__all__ = [
    "RATES_ENV",
    "Budget",
    "BudgetExceeded",
    "Rates",
    "Usage",
    "rates_from_env",
]
