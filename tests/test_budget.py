"""Token and request accounting: the session runs on the researcher's key."""

from __future__ import annotations

import pytest

from polyrmc.agent.budget import (
    RATES_ENV,
    Budget,
    BudgetExceeded,
    Rates,
    Usage,
    rates_from_env,
)


def test_usage_adds_up():
    total = Usage(1, 100, 20) + Usage(1, 150, 30)

    assert total.requests == 2
    assert total.input_tokens == 250
    assert total.total_tokens == 300


def test_rates_are_opt_in(monkeypatch):
    """Token counts are measured. A price is a claim about a vendor's pricing
    page on a particular day, so it is only reported when someone supplies it."""
    monkeypatch.delenv(RATES_ENV, raising=False)
    assert rates_from_env() is None

    monkeypatch.setenv(RATES_ENV, "in=0.30,out=2.50")
    assert rates_from_env() == Rates(0.30, 2.50)


@pytest.mark.parametrize("raw", ["", "in=0.3", "out=2.5", "in=abc,out=1", "nonsense"])
def test_an_unreadable_rate_reports_no_cost_rather_than_a_wrong_one(raw):
    assert rates_from_env(raw) is None


def test_cost_is_per_million_tokens():
    rates = Rates(input_per_mtok=0.30, output_per_mtok=2.50)

    cost = rates.cost(Usage(requests=1, input_tokens=1_000_000, output_tokens=1_000_000))

    assert cost == pytest.approx(2.80)


def test_the_request_ceiling_stops_the_session():
    budget = Budget(max_requests=2)
    budget.record(Usage(requests=2))

    with pytest.raises(BudgetExceeded, match="request budget"):
        budget.check()


def test_the_ceiling_is_checked_before_spending_not_after():
    """A limit that only reports having been passed is not a limit."""
    budget = Budget(max_requests=1)
    budget.check()  # nothing spent yet, so this must pass

    budget.record(Usage(requests=1))

    with pytest.raises(BudgetExceeded):
        budget.check()


def test_the_token_ceiling_stops_the_session():
    budget = Budget(max_tokens=500)
    budget.record(Usage(requests=1, input_tokens=400, output_tokens=200))

    with pytest.raises(BudgetExceeded, match="token budget"):
        budget.check()


def test_zero_means_no_ceiling():
    budget = Budget()
    budget.record(Usage(requests=10_000, input_tokens=10**9))

    budget.check()  # must not raise


def test_the_render_says_what_to_set_when_no_rate_is_configured():
    line = Budget(rates=None).render()

    assert RATES_ENV in line
    assert "$" not in line, "a cost was reported without a rate to compute it from"


def test_the_render_shows_cost_once_a_rate_is_configured():
    budget = Budget(rates=Rates(0.30, 2.50))
    budget.record(Usage(requests=3, input_tokens=1_000_000, output_tokens=0))

    line = budget.render()

    assert "3 request(s)" in line
    assert "$0.3000" in line
