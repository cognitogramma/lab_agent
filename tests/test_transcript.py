"""What a researcher sees, and what goes to the log instead.

The split these tests defend is not cosmetic. A session that prints its own
mechanics buries the two lines that matter -- what the model concluded, and what
it changed -- under argument dumps and token counts, and a reader who has learnt
to skim a transcript is a reader who will skim the approval prompt too.

So: the terminal gets an account, the log gets everything, and nothing is only
in the terminal.
"""

from __future__ import annotations

import io
import types

import pytest

from polyrmc.agent.budget import Budget, Usage
from polyrmc.agent.orchestrator import (
    ApprovalPolicy,
    Event,
    Orchestrator,
    allow_all,
)
from polyrmc.agent.toolcall import (
    AssistantTurn,
    ProviderCallFailed,
    ScriptedProvider,
    ToolCall,
    classify_provider_error,
)
from polyrmc.agent.toolspec import ToolContext, build_registry
from polyrmc.agent.transcript import (
    Narrator,
    SessionLog,
    activity,
    error_block,
    first_sentence,
)
from polyrmc.memory.notes import NoteBook
from polyrmc.state import RunState
from polyrmc.tools import ToolSurface


@pytest.fixture
def log() -> SessionLog:
    """A log over a buffer, so a test can read back what a session recorded."""
    return SessionLog(stream=io.StringIO())


def written(log: SessionLog) -> str:
    return log._stream.getvalue()


# --- the human stream --------------------------------------------------------


def test_a_tool_call_is_shown_as_an_activity_not_as_a_call(log):
    narrator = Narrator(log)
    call = ToolCall("id", "sweep_windows", {})
    lines = narrator.lines(Event("call", tool="sweep_windows", call=call))

    assert lines == ["  . sweeping the smoothing windows"]
    # The shape of the API request must not survive into the terminal.
    assert "sweep_windows(" not in "".join(lines)
    assert "sweep_windows" not in "".join(lines)


def test_call_arguments_are_named_in_words_not_in_call_syntax(log):
    narrator = Narrator(log)
    call = ToolCall("id", "pick_window", {"window_length": 31})
    (line,) = narrator.lines(Event("call", tool="pick_window", call=call))

    assert line == "  . choosing the smoothing window (window length 31)"
    assert "=" not in line


def test_tool_output_goes_to_the_log_and_not_to_the_terminal(log):
    narrator = Narrator(log)
    output = "noise_floor: 5.28\n" + "\n".join(f"  [{i}] window={i}" for i in range(40))

    assert narrator.lines(Event("result", output, tool="sweep_windows")) == []
    assert "noise_floor: 5.28" in written(log)
    assert "[39] window=39" in written(log)


def test_verbose_puts_the_background_work_back_on_screen(log):
    narrator = Narrator(log, verbose=True)
    shown = narrator.lines(Event("result", "noise_floor: 5.28", tool="sweep_windows"))

    assert "noise_floor: 5.28" in "".join(shown)


def test_a_refusal_is_shown_in_full_because_it_is_a_decision(log):
    narrator = Narrator(log)
    rejection = (
        "REJECTED. Window 9999 is not in the safe band. Safe band is [5, 7, 11]."
    )
    (line,) = narrator.lines(Event("refused", rejection, tool="pick_window"))

    # The whole of it, including the band. This is the one message the tool
    # surface writes for a person to act on rather than for the model to parse.
    assert "Safe band is [5, 7, 11]" in line
    assert rejection in written(log)


def test_a_model_facing_error_is_cut_short_on_screen_and_kept_whole_in_the_log(log):
    narrator = Narrator(log)
    full = (
        "no tool named 'load_raw'. Available: " + ", ".join(sorted(build_registry()))
    )
    (line,) = narrator.lines(Event("error", full, tool="load_raw"))

    assert len(line) < len(full)
    assert full in written(log)


def test_token_accounting_never_interrupts_the_conversation(log):
    narrator = Narrator(log)
    spend = "this turn: 3 request(s), 14,600 in / 270 out"

    assert narrator.lines(Event("spend", spend)) == []
    assert spend in written(log)


def test_every_tool_in_the_registry_has_words_a_person_would_use():
    """A tool added without a phrase degrades; it must not read as code."""
    for name in build_registry():
        phrase = activity(name)
        assert "_" not in phrase, f"{name} has no human phrase"


def test_a_log_that_cannot_be_opened_does_not_stop_a_session(tmp_path):
    # A file where the directory should be: mkdir fails, and the session is
    # expected to carry on with a complete terminal view and no log.
    blocker = tmp_path / "logs"
    blocker.write_text("not a directory", encoding="utf-8")
    log = SessionLog(path=blocker / "session.log")

    log.write("call", "anything")  # must not raise
    assert log.path is None
    assert "could not be written" in log.location


# --- stopping ----------------------------------------------------------------


def test_a_stop_does_not_read_like_something_the_model_said():
    block = error_block(
        "out of quota",
        "That key is out of quota or over its rate limit.",
        "Top the account up, or start again on another --provider.",
        "data/logs/session-x.log",
    )

    assert "OUT OF QUOTA" in block
    assert "data/logs/session-x.log" in block
    assert block.count("-" * 20) >= 3  # ruled off above, below, and under the title


def test_the_provider_s_raw_complaint_is_logged_where_the_banner_points(log):
    narrator = Narrator(log)
    raw = "429 RESOURCE_EXHAUSTED {'error': {'code': 429, 'message': 'quota'}}"
    failure = classify_provider_error(RuntimeError(raw), "gemini", "gemini-3.6-flash")

    assert narrator.lines(Event("failed", "gemini refused", detail=failure)) == []
    assert raw in written(log)


@pytest.mark.parametrize(
    "text, headline",
    [
        ("429 RESOURCE_EXHAUSTED: quota exceeded", "out of quota"),
        ("401 Unauthorized: invalid_api_key", "key rejected"),
        ("404 model not found", "model id rejected"),
        ("connection timed out", "provider unreachable"),
        ("something nobody has seen before", "the model call failed"),
    ],
)
def test_a_failure_is_sorted_by_what_would_fix_it(text, headline):
    failure = classify_provider_error(RuntimeError(text), "gemini", "m")
    assert failure.headline == headline


def test_a_leaked_key_is_blanked_before_it_reaches_a_log_or_a_banner():
    failure = classify_provider_error(
        RuntimeError("401 rejected key=sk-abcdef1234567890"), "openai", "gpt-5"
    )
    assert "sk-abcdef1234567890" not in failure.raw
    assert "<redacted>" in failure.raw


def test_first_sentence_leaves_a_short_line_alone():
    assert first_sentence("one line") == "one line"
    assert first_sentence("first\nsecond") == "first"


# --- the loop ----------------------------------------------------------------


@pytest.fixture
def context(run_config, tmp_path) -> ToolContext:
    surface = ToolSurface(
        state=RunState(run_id="t", source_file=str(run_config.source_file)),
        config=run_config,
    )
    return ToolContext(
        surface=surface,
        notes=NoteBook(tmp_path / "memory"),
        runs_dir=tmp_path / "runs",
        author="tester",
    )


class DiesAfter(ScriptedProvider):
    """Answers its script, then refuses the way a spent key does."""

    name = "gemini"
    model = "gemini-3.6-flash"

    def converse(self, system, messages, tools):
        if not self.turns:
            inner = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
            raise ProviderCallFailed(
                "gemini refused",
                provider="gemini",
                model=self.model,
                failure=classify_provider_error(inner, "gemini", self.model),
            )
        return super().converse(system, messages, tools)


def test_work_already_done_survives_a_provider_that_stops_answering(context):
    """The bug this file was written for.

    ``ask`` collects events into a list and the caller prints the list, so an
    exception on round two threw away round one -- tools that really ran, on a
    run that really changed, reported as though the session had never started.
    A researcher whose key expired mid-question saw a stack of quota JSON and no
    record of the three tools that had already succeeded.
    """
    provider = DiesAfter(
        [
            AssistantTurn(
                text="Looking first.",
                tool_calls=[ToolCall("", "get_state", {})],
                usage=Usage(requests=1, input_tokens=100, output_tokens=10),
            )
        ]
    )
    agent = Orchestrator(
        context=context,
        provider=provider,
        approvals=ApprovalPolicy(ask=allow_all),
        budget=Budget(),
    )

    events = agent.ask("where are we?")  # must not raise

    kinds = [event.kind for event in events]
    assert "call" in kinds and "result" in kinds, "the work that ran is still reported"
    assert kinds[-2:] == ["failed", "spend"]

    failed = next(event for event in events if event.kind == "failed")
    assert failed.detail.headline == "out of quota"
    # Classified where the provider's own exception was in hand, so the raw text
    # is the provider's rather than this package's explanation of it.
    assert "RESOURCE_EXHAUSTED" in failed.detail.raw
    assert "failed:" not in failed.detail.raw


def test_history_reads_back_in_the_words_the_session_used(context):
    provider = ScriptedProvider(
        [
            AssistantTurn(tool_calls=[ToolCall("", "get_state", {})]),
            AssistantTurn(text="done"),
        ]
    )
    agent = Orchestrator(
        context=context,
        provider=provider,
        approvals=ApprovalPolicy(ask=allow_all),
    )
    agent.ask("status?")

    assert "checking where the run stands" in agent.narrate()
    assert "get_state(" not in agent.narrate()


def test_a_commit_is_narrated_before_it_is_put_to_the_researcher(context):
    """Ordering, which only streaming gets right.

    The approval question is asked from inside the loop. A caller that collects
    the whole turn before printing it therefore shows the prompt first and the
    reasoning afterwards -- the researcher is asked to allow a window they have
    not been told about yet, which is precisely the habit the approval gate
    exists to prevent.
    """
    seen: list[str] = []

    def ask_and_record(tool, call_):
        # Whatever has been yielded by now is what the researcher has read by
        # the time this question reaches them.
        from polyrmc.agent.orchestrator import Approval

        seen.append("PROMPT")
        return Approval(True)

    provider = ScriptedProvider(
        [
            AssistantTurn(
                text="I will sweep first, then propose a window.",
                tool_calls=[ToolCall("", "load_run", {})],
            ),
            AssistantTurn(tool_calls=[ToolCall("", "sweep_windows", {})]),
            AssistantTurn(tool_calls=[ToolCall("", "pick_window", {"window_length": 7})]),
            AssistantTurn(text="Window 7 is set."),
        ]
    )
    agent = Orchestrator(
        context=context,
        provider=provider,
        approvals=ApprovalPolicy(ask=ask_and_record),
    )

    for event in agent.stream("pick a window for me"):
        if event.kind == "text":
            seen.append("TEXT")
        elif event.kind == "call":
            seen.append(f"CALL {event.tool}")

    assert seen.index("TEXT") < seen.index("PROMPT")
    assert seen.index("CALL pick_window") < seen.index("PROMPT")


def test_ask_returns_the_same_events_stream_yields(context):
    """The list form is the streamed form, collected -- not a second path."""
    turns = [AssistantTurn(tool_calls=[ToolCall("", "get_state", {})]),
             AssistantTurn(text="done")]
    kinds = []
    for build in (lambda a, t: a.ask(t), lambda a, t: list(a.stream(t))):
        agent = Orchestrator(
            context=context,
            provider=ScriptedProvider([AssistantTurn(**vars(t)) for t in turns]),
            approvals=ApprovalPolicy(ask=allow_all),
        )
        kinds.append([event.kind for event in build(agent, "status?")])
    assert kinds[0] == kinds[1]


# --- rate limits, which are a wait and not a bill -----------------------------


class FakeClient:
    """A bound chat client that raises a scripted sequence, then answers."""

    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, payload):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return types.SimpleNamespace(content="done", tool_calls=[], usage_metadata={})


RATE_LIMIT = (
    "429 RESOURCE_EXHAUSTED: quotaId: "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', quotaValue: '5'. "
    "Please retry in 41.2s"
)
DAILY_QUOTA = (
    "429 RESOURCE_EXHAUSTED: quota exceeded. quotaId: "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', quotaValue: '20'"
)


def test_a_per_minute_limit_is_not_reported_as_a_bill_to_pay():
    """The two 429s want opposite things from the reader.

    A daily quota is spent and the fix costs money. A per-minute limit is an
    agent asking faster than a free tier allows, and the fix is to wait the
    number of seconds the provider just named. Telling the second researcher to
    top up their account is wrong twice: they need not spend, and the one thing
    that would work goes unsaid.
    """
    minute = classify_provider_error(RuntimeError(RATE_LIMIT), "gemini", "m")
    daily = classify_provider_error(RuntimeError(DAILY_QUOTA), "gemini", "m")

    assert minute.headline == "rate limited" and minute.rate_limited
    assert minute.retry_after == pytest.approx(41.2)
    assert "top it up" not in minute.remedy

    assert daily.headline == "out of quota" and not daily.rate_limited
    assert daily.retry_after is None


def test_a_rate_limit_is_waited_out_rather_than_ending_the_session():
    """The one place the session retries, and it costs nothing to do so.

    The no-retry rule protects a researcher from paying twice for one question.
    A rate-limited request was refused, not served: no tokens were counted and
    there is nothing to pay twice for. Ending a minutes-old conversation over it
    trades a real loss for an imaginary saving.
    """
    from polyrmc.agent.toolcall import LangChainToolProvider

    slept: list[float] = []
    client = FakeClient([RuntimeError(RATE_LIMIT)])
    provider = LangChainToolProvider(
        client, name="gemini", model="m", sleep=slept.append
    )

    turn = provider.converse("sys", [], [])

    assert turn.text == "done", "the session carried on"
    assert client.calls == 2, "it asked again after waiting"
    assert slept == [pytest.approx(41.2)], "it waited exactly as long as asked"


def test_a_daily_quota_ends_the_session_on_the_first_answer():
    """Waiting cannot fix a spent day, so nothing waits."""
    from polyrmc.agent.toolcall import LangChainToolProvider

    slept: list[float] = []
    client = FakeClient([RuntimeError(DAILY_QUOTA)] * 4)
    provider = LangChainToolProvider(
        client, name="gemini", model="m", sleep=slept.append
    )

    with pytest.raises(ProviderCallFailed) as caught:
        provider.converse("sys", [], [])

    assert client.calls == 1, "it did not ask again"
    assert slept == []
    assert caught.value.failure.headline == "out of quota"


def test_waiting_is_bounded_so_a_stuck_limit_still_stops():
    from polyrmc.agent.toolcall import LangChainToolProvider

    slept: list[float] = []
    client = FakeClient([RuntimeError(RATE_LIMIT)] * 10)
    provider = LangChainToolProvider(
        client, name="gemini", model="m", sleep=slept.append, max_rate_limit_waits=2
    )

    with pytest.raises(ProviderCallFailed):
        provider.converse("sys", [], [])

    assert client.calls == 3, "the first call plus two waits"
    assert len(slept) == 2


def test_pacing_leaves_the_asked_for_gap_between_requests():
    """Cheaper than being refused: a session that paces is never rate limited."""
    from polyrmc.agent.toolcall import LangChainToolProvider

    now = [100.0]
    slept: list[float] = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    provider = LangChainToolProvider(
        FakeClient([]),
        name="gemini",
        model="m",
        min_interval=13.0,
        sleep=sleep,
        clock=lambda: now[0],
    )
    provider.converse("sys", [], [])
    now[0] += 3.0  # the tools took three seconds
    provider.converse("sys", [], [])

    assert slept == [pytest.approx(10.0)], "it waited only the remaining gap"
