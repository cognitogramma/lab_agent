"""The window, driven without one.

Two halves, for two different reasons.

The half that matters runs headless. :mod:`polyrmc.gui.session` imports no Tk on
purpose, so the thing worth testing -- a turn crossing from a worker thread to a
window, and a commit stopping dead in the middle of it until a person answers --
is testable on any machine, including one with no display. Every failure mode
that costs a researcher something lives there: an approval that never arrives, a
window closed on a waiting commit, a turn that raises.

The other half constructs the real window and is skipped where Tk cannot open a
display. It asserts what a screenshot would: that each kind of event renders,
that a refusal arrives whole, and that closing the approval dialog is a refusal
rather than an approval nobody gave.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest

from polyrmc.agent.orchestrator import (
    Approval,
    ApprovalPolicy,
    Orchestrator,
    allow_all,
)
from polyrmc.agent.toolcall import AssistantTurn, ScriptedProvider, ToolCall
from polyrmc.agent.toolspec import ToolContext, build_registry
from polyrmc.gui.session import ApprovalRequest, Bridge, Signal
from polyrmc.memory.notes import NoteBook
from polyrmc.state import RunState
from polyrmc.tools import ToolSurface

TIMEOUT = 20.0


def call(name: str, **arguments) -> AssistantTurn:
    return AssistantTurn(tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=arguments)])


@pytest.fixture
def context(run_config, tmp_path) -> ToolContext:
    surface = ToolSurface(
        state=RunState(run_id="gui-test", source_file=str(run_config.source_file)),
        config=run_config,
    )
    return ToolContext(
        surface=surface,
        notes=NoteBook(tmp_path / "memory"),
        runs_dir=tmp_path / "runs",
        author="tester",
    )


def build(context, turns, policy: ApprovalPolicy | None = None) -> tuple[Bridge, Orchestrator]:
    agent = Orchestrator(
        context=context,
        provider=ScriptedProvider(turns),
        approvals=policy or ApprovalPolicy(ask=allow_all),
        registry=build_registry(),
    )
    return Bridge(agent=agent, surface=context.surface), agent


def collect(bridge: Bridge, until: str = "done", timeout: float = TIMEOUT) -> list[Signal]:
    """Poll the way the window polls, and stop when the turn ends."""
    gathered: list[Signal] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for signal in bridge.drain():
            gathered.append(signal)
            if signal.kind == until:
                return gathered
        time.sleep(0.01)
    raise AssertionError(f"no {until!r} signal within {timeout}s; got {[s.kind for s in gathered]}")


def wait_for(bridge: Bridge, kind: str, timeout: float = TIMEOUT) -> Signal:
    return collect(bridge, until=kind, timeout=timeout)[-1]


# --- the turn crossing threads ----------------------------------------------


def test_a_turn_reaches_the_window_in_the_order_it_happened(context):
    bridge, _ = build(context, [call("get_state"), "The run is not loaded yet."])

    assert bridge.submit("where does this stand?") is True
    signals = collect(bridge)

    assert signals[-1].kind == "done"
    assert signals[-1].payload == "ok"
    # A call is reported before its result. The window renders in arrival order,
    # and an approval prompt has to follow the reasoning that led to it rather
    # than arrive ahead of it.
    events = [s.payload.kind for s in signals if s.kind == "event"]
    assert events.index("call") < events.index("result")
    assert "text" in events


def test_the_state_panel_is_rendered_on_the_thread_that_writes_the_run(context):
    """Rendered text arrives, not a live surface for the window to walk.

    Iterating the declarations dict while a tool writes it raises. The window
    never touches the run for this reason: it is handed finished text.
    """
    bridge, _ = build(context, [call("get_state"), "done"])
    bridge.submit("state?")
    signals = collect(bridge)

    rendered = [s.payload for s in signals if s.kind == "state"]
    assert rendered, "the window was never told where the run stands"
    assert all(isinstance(text, str) for text in rendered)
    assert any("run_id: gui-test" in text for text in rendered)


def test_a_turn_that_raises_reports_a_crash_and_still_finishes(context):
    bridge, agent = build(context, ["never reached"])

    def explode(_text):
        raise RuntimeError("the loop came apart")

    agent.stream = explode
    bridge.submit("anything")
    signals = collect(bridge)

    kinds = [s.kind for s in signals]
    assert "crash" in kinds
    assert kinds[-1] == "done"
    crash = next(s for s in signals if s.kind == "crash")
    assert "the loop came apart" in crash.payload


class Stalling:
    """A provider that answers only when the test lets it.

    ``reply`` decides whether the turn it eventually gives back is a final
    answer or a request for tools -- which matters for stopping. A turn that
    answers is finished, and there is nothing left to cancel; only a turn that
    wants another round has a next step to be stopped before.
    """

    name = "stalling"
    model = "stalling"

    def __init__(self, gate: threading.Event, reply: AssistantTurn | None = None) -> None:
        self.gate = gate
        self.reply = reply or AssistantTurn(text="answered at last.")

    def converse(self, system, messages, tools):
        assert self.gate.wait(TIMEOUT), "the test never released the provider"
        return self.reply


def test_one_turn_at_a_time(context):
    gate = threading.Event()
    agent = Orchestrator(
        context=context,
        provider=Stalling(gate),
        approvals=ApprovalPolicy(ask=allow_all),
        registry=build_registry(),
    )
    bridge = Bridge(agent=agent, surface=context.surface)

    assert bridge.submit("first") is True
    # Two questions in flight would interleave into one transcript, and each
    # would be answered against a run the other was changing.
    assert bridge.submit("second") is False
    gate.set()
    collect(bridge)
    # Once it has answered, the next question is welcome.
    assert bridge.submit("third") is True
    collect(bridge)


def test_nothing_is_submitted_without_a_brain(context):
    assert Bridge(agent=None, surface=context.surface).submit("hello") is False


# --- the commit gate ---------------------------------------------------------


DECLARE = call("set_declaration", key="c0_g_per_cm3", value=0.004)


def gated(context, closing: str):
    """A session whose one commit is put to the window."""
    policy = ApprovalPolicy()
    bridge, _ = build(context, [DECLARE, closing], policy=policy)
    policy.ask = bridge.ask
    return bridge


def test_a_commit_stops_until_the_window_answers(context):
    """The tool does not run until a person has answered, and then it does."""
    bridge = gated(context, "recorded.")

    bridge.submit("record c0 as 0.004")
    request = wait_for(bridge, "ask").payload
    assert isinstance(request, ApprovalRequest)
    assert request.tool.name == "set_declaration"
    assert request.call.arguments == {"key": "c0_g_per_cm3", "value": 0.004}
    assert not request.answered
    assert "c0_g_per_cm3" not in context.surface.state.declarations

    request.reply(Approval(True))
    signals = collect(bridge)
    assert not any(s.kind == "event" and s.payload.kind == "denied" for s in signals)
    assert context.surface.state.declarations["c0_g_per_cm3"].value == 0.004


def test_declining_reaches_the_agent_as_the_researchers_own_words(context):
    bridge = gated(context, "understood.")

    bridge.submit("record c0")
    request = wait_for(bridge, "ask").payload
    request.reply(Approval(False, "that is the stock, not the cell"))
    signals = collect(bridge)

    denied = [s.payload for s in signals if s.kind == "event" and s.payload.kind == "denied"]
    assert denied, "the window's refusal never reached the transcript"
    assert "that is the stock" in denied[0].text
    assert "c0_g_per_cm3" not in context.surface.state.declarations


def test_closing_the_window_refuses_a_waiting_commit(context):
    """An unanswered prompt ends as a refusal, never as an approval nobody gave."""
    bridge = gated(context, "stopped.")

    bridge.submit("record c0")
    request = wait_for(bridge, "ask").payload
    bridge.close()

    assert request.answered
    assert request.await_reply().allowed is False
    assert "closed" in request.await_reply().reason
    collect(bridge)  # the worker was released and the turn finished
    assert "c0_g_per_cm3" not in context.surface.state.declarations


def test_the_first_answer_wins(context):
    """Two paths can answer -- the dialog and a closing window. Only one counts."""
    request = ApprovalRequest(tool=None, call=None)
    request.reply(Approval(True, "allowed"))
    request.reply(Approval(False, "the window was closed"))
    assert request.await_reply().allowed is True


def test_a_commit_blocks_its_worker_until_answered(context):
    """The gate is a real gate: the calling thread does not proceed."""
    bridge = Bridge(agent=None, surface=context.surface)
    tool = build_registry()["pick_window"]
    answered = threading.Event()
    outcome: list = []

    def worker():
        outcome.append(bridge.ask(tool, ToolCall("c1", "pick_window", {"window_length": 31})))
        answered.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    request = wait_for(bridge, "ask").payload
    assert not answered.wait(0.2), "the worker carried on without an answer"

    request.reply(Approval(True))
    assert answered.wait(TIMEOUT)
    assert outcome[0].allowed is True


def test_a_closing_session_refuses_later_asks(context):
    bridge = Bridge(agent=None, surface=context.surface)
    bridge.close()
    verdict = bridge.ask(build_registry()["pick_window"], ToolCall("c1", "pick_window", {}))
    assert verdict.allowed is False


# --- the window itself -------------------------------------------------------


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    """One window for this whole file, or a skip where Tk cannot open.

    One, not one per test, and it is the only Tk root any test here creates.
    Tk will happily run several interpreters in a process and Tcl will not: it
    re-searches for its own library on every root after the first, and on
    Windows that search fails often enough that a root-per-test turned six
    reliable tests into six that usually passed. Tests wanting an empty widget
    tree take a :class:`~tkinter.Toplevel` off this one instead.

    It is built with no provider keyed and a redirected project root, so nothing
    it writes -- session log, notes, run history -- lands in the real tree.
    """
    tk = pytest.importorskip("tkinter")
    from polyrmc import paths
    from polyrmc.agent.toolcall import SPECS
    from polyrmc.gui.app import Workbench

    example = paths.example_config()
    if not example.exists():
        pytest.skip("no example configuration to scaffold a session from")

    patch = pytest.MonkeyPatch()
    patch.setenv(paths.ROOT_ENV, str(tmp_path_factory.mktemp("root")))
    for spec in SPECS.values():
        for key in spec.env_keys:
            patch.delenv(key, raising=False)

    try:
        built = Workbench(gui_args(example, tmp_path_factory.mktemp("session")))
    except tk.TclError as error:  # no display, or no Tk libraries
        patch.undo()
        pytest.skip(f"no display: {error}")
    built.withdraw()
    _settle(built)
    yield built
    built.close()
    patch.undo()


def gui_args(example, tmp_path):
    from types import SimpleNamespace

    return SimpleNamespace(
        config=str(example), raw=None, declarations=None, run_id="windowed",
        channel=None, runs_dir=str(tmp_path / "runs"), notes_dir=str(tmp_path / "memory"),
        out=None, author="tester", provider=None, model=None, max_steps=4,
        auto_approve=False, budget_requests=0, budget_tokens=0,
        min_request_interval=0.0, verbose=False, no_prompt=False, prompt=None,
    )


def _settle(built, seconds: float = 0.3) -> None:
    """Pump the event loop, so scheduled callbacks actually run."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        built.update()
        time.sleep(0.005)


@pytest.fixture
def root(app):
    """A throwaway top level, so each test starts with an empty widget tree."""
    import tkinter as tk

    window = tk.Toplevel(app)
    window.withdraw()
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def window(app):
    """The shared window, put back afterwards for whoever gets it next."""
    yield app
    app.dead = False
    app.ask_button.configure(state="normal")
    app.policy.always.clear()
    app.transcript.clear()
    app.entry.delete("1.0", "end")
    app.tabs.select(app.state_panel)


def test_the_transcript_renders_every_kind_of_event(root):
    from polyrmc.agent.orchestrator import Event
    from polyrmc.agent.toolcall import ProviderFailure
    from polyrmc.gui import theme
    from polyrmc.gui.widgets import Transcript

    fonts = theme.apply(root)
    transcript = Transcript(root, fonts)

    transcript.you("what happened to run 12?")
    transcript.model("Let me see where this run stands before touching anything.")
    transcript.action("sweeping the smoothing windows", "window length 31")
    transcript.block(
        "refused",
        "REJECTED. window_length 7 is below the safe band; 21 to 61 would be valid.",
        "refused",
    )
    transcript.block("you declined this", "too narrow", "declined")
    failure = ProviderFailure(
        provider="gemini", model="x", headline="out of quota",
        meaning="That key is out of quota.", remedy="Top it up.", raw="{}",
    )
    transcript.banner(failure.headline, failure.meaning, failure.remedy, "data/logs/x.log")
    root.update_idletasks()

    shown = transcript.plain()
    # A refusal is never abbreviated: the band it names is the useful part.
    assert "21 to 61 would be valid" in shown
    assert "OUT OF QUOTA" in shown
    assert "Top it up." in shown
    assert Event  # the kinds above are the ones Event emits


def test_the_state_panel_shows_what_the_model_is_given(root, run_config):
    from polyrmc.agent.toolspec import render_state
    from polyrmc.gui import theme
    from polyrmc.gui.widgets import StatePanel

    fonts = theme.apply(root)
    surface = ToolSurface(state=RunState(run_id="r", source_file="none"), config=run_config)
    panel = StatePanel(root, fonts)
    rendered = render_state(surface)
    panel.show(rendered)
    root.update_idletasks()

    shown = panel.text.get("1.0", "end-1c")
    for line in rendered.splitlines():
        key, _, value = line.partition(": ")
        assert key in shown and (not value or value in shown)


def test_dismissing_the_approval_dialog_is_a_refusal(root):
    from polyrmc.gui import theme
    from polyrmc.gui.widgets import ApprovalDialog

    fonts = theme.apply(root)
    tool = build_registry()["pick_window"]
    call_ = ToolCall("c1", "pick_window", {"window_length": 31})
    dialog = ApprovalDialog(root, fonts, tool, call_, "choosing the smoothing window")
    try:
        # Whatever a dialog is doing before anyone touches it, the answer it is
        # holding is a refusal. There is no path from "closed" to "approved".
        assert dialog.answer.allowed is False
        dialog._decline()
    finally:
        if dialog.winfo_exists():
            dialog.destroy()


def test_the_approval_dialog_shows_every_argument(root):
    from polyrmc.gui import theme
    from polyrmc.gui.widgets import ApprovalDialog

    fonts = theme.apply(root)
    tool = build_registry()["set_declaration"]
    arguments = {
        "key": "c0_g_per_cm3",
        "value": 0.004,
        "note": "a" * 300,  # long enough that a summarising prompt would cut it
    }
    dialog = ApprovalDialog(
        root, fonts, tool, ToolCall("c1", "set_declaration", arguments), "recording a declaration"
    )
    try:
        root.update_idletasks()
        shown = _labels(dialog)
        for key, value in arguments.items():
            assert key.replace("_", " ") in shown
            assert str(value) in shown, f"{key} was abbreviated in an approval prompt"
    finally:
        dialog.destroy()


def test_a_commit_in_the_window_is_put_to_the_window(window):
    """The gate has to be pointed at the dialog, or every commit is refused.

    :class:`ApprovalPolicy` defaults to ``deny_all`` -- correct for a script with
    nobody at the terminal, and wrong for a window with somebody sitting at it.
    Left unwired, every commit came back "no one is at the terminal to approve".
    """
    assert window.policy.ask == window.bridge.ask


def test_the_window_opens_without_a_key(window):
    """A missing key disables asking, not the window."""
    assert window.agent is None
    assert "no model keyed" in window.model_label.cget("text")
    # The panels read past runs, notes and figures with no model at all.
    assert "run_id" in window.state_panel.text.get("1.0", "end-1c")
    assert window.registry, "the tool surface is there whether or not a brain is"


def test_a_dead_provider_stops_the_window_asking_again(window):
    """Enter in the box is not the button, and must not spend on a spent key."""
    window.dead = True
    window.entry.insert("1.0", "try again")
    window.submit()
    assert not window.bridge.busy
    assert "stopped answering" in window.transcript.plain()


def test_slash_commands_reach_the_panels(window):
    window._slash("/tools")
    assert window.tabs.select() == str(window.tools_panel)
    window._slash("/runs lysozyme")
    assert window.tabs.select() == str(window.runs_panel)
    assert window.runs_panel.query.get() == "lysozyme"
    window._slash("/nonsense")
    assert "unknown command" in window.transcript.plain()


def test_standing_approval_only_covers_commit_tools(window):
    window.always_allow("get_state")
    assert "get_state" not in window.policy.always
    window.always_allow("pick_window")
    assert "pick_window" in window.policy.always


# --- stopping ----------------------------------------------------------------


def test_stopping_an_idle_bridge_does_nothing(context):
    """Escape is a reflex, and it must be free to press when nothing is running."""
    bridge = Bridge(agent=None, surface=context.surface)
    bridge.request_stop()
    assert bridge.stopping is False
    assert bridge.should_stop() is False


def test_a_stop_ends_the_turn_and_the_next_question_starts_clean(context):
    gate = threading.Event()
    agent = Orchestrator(
        context=context, provider=Stalling(gate),
        approvals=ApprovalPolicy(ask=allow_all), registry=build_registry(),
    )
    bridge = Bridge(agent=agent, surface=context.surface)
    agent.should_stop = bridge.should_stop

    bridge.submit("first")
    bridge.request_stop()
    assert bridge.stopping is True
    gate.set()  # the model call already in flight still has to return
    collect(bridge)

    # The flag belongs to the question, not the session: a stop ends one turn.
    gate.clear()
    assert bridge.submit("second") is True
    assert bridge.stopping is False
    gate.set()
    collect(bridge)


def test_stopping_releases_a_commit_that_was_waiting_on_an_answer(context):
    """Someone who hit stop is not answering yes to the dialog behind it."""
    bridge = gated(context, "understood.")

    bridge.submit("record c0")
    request = wait_for(bridge, "ask").payload
    bridge.request_stop()

    assert request.answered
    verdict = request.await_reply()
    assert verdict.allowed is False
    assert "stopped" in verdict.reason
    collect(bridge)
    assert "c0_g_per_cm3" not in context.surface.state.declarations


def test_the_orchestrator_is_told_where_to_ask(window):
    """Wired, or the button sets a flag nothing ever reads."""
    assert window.bridge.should_stop() is False
    if window.agent is not None:
        assert window.agent.should_stop == window.bridge.should_stop


def test_the_button_is_always_the_thing_you_can_do_now(window):
    window._busy(True)
    assert window.ask_button.cget("text") == "Stop"
    assert str(window.ask_button.cget("state")) == "normal", (
        "the stop button is the one control that must stay live while busy"
    )
    assert str(window.entry.cget("state")) == "disabled"

    window._busy(False)
    assert window.ask_button.cget("text") == "Ask"
    assert str(window.entry.cget("state")) == "normal"


def test_the_window_says_stopping_not_stopped(window, context):
    """It has not stopped yet, and claiming otherwise makes people pull the plug.

    Driven against a provider that will not answer until the test lets it, which
    is the situation the button exists for: a turn is in flight, the researcher
    has seen enough, and the model call still has to come back before anything
    can end.
    """
    gate = threading.Event()
    agent = Orchestrator(
        context=context, provider=Stalling(gate, call("get_state")),
        approvals=ApprovalPolicy(ask=allow_all), registry=build_registry(),
    )
    original = window.bridge
    window.bridge = Bridge(agent=agent, surface=context.surface)
    agent.should_stop = window.bridge.should_stop
    try:
        window.bridge.submit("something slow")
        window._busy(True)
        window.stop()

        assert window.bridge.stopping is True
        assert window.ask_button.cget("text") == "stopping...", (
            "the label must not claim the turn has already ended"
        )
        shown = window.transcript.plain()
        assert "stopping" in shown and "already running to return" in shown

        gate.set()  # the in-flight call returns, and only then does the turn end
        signals = collect(window.bridge)
        assert any(
            s.kind == "event" and s.payload.kind == "stopped" for s in signals
        ), "the turn has to report that it stopped"
    finally:
        gate.set()
        window.bridge = original
        window._busy(False)


GUI_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "polyrmc" / "gui"


def test_the_window_offers_no_route_of_its_own():
    """The model decides what happens next; the window must not decide for it.

    Giving up the fixed route is the architecture, and a window is exactly where
    it would creep back in -- as a row of buttons that run the stages in order,
    which is the old graph with a mouse on it. So the only tools this package
    invokes on its own are ``read`` tools: they answer a question and cannot
    change a run. Everything that computes or commits reaches the run through
    the model's own tool calls, past the approval gate.
    """
    from polyrmc.agent.toolspec import build_registry

    registry = build_registry()
    sources = {p.name: p.read_text(encoding="utf-8") for p in GUI_PACKAGE.glob("*.py")}
    invoked = sorted(
        {
            name
            for text in sources.values()
            for name in re.findall(r"""_call_tool\(\s*["']([a-z_]+)["']""", text)
        }
    )
    assert invoked, "the panels run something, or this test is asserting nothing"
    for name in invoked:
        assert registry[name].tier == "read", (
            f"the window runs {name!r} ({registry[name].tier}) itself, without the "
            "model choosing to and without the approval gate"
        )

    # One door, so the check above cannot be walked around. A handler called
    # from anywhere else in the package would not appear in `invoked`.
    doors = sum(text.count(".handler(") for text in sources.values())
    assert doors == 1, "a tool is being run from somewhere other than _call_tool"


def test_the_models_markdown_is_rendered_not_displayed(root):
    """``###`` is a formatting instruction, and showing it is the same failure
    as printing call syntax at somebody."""
    from polyrmc.gui import theme
    from polyrmc.gui.widgets import Transcript

    transcript = Transcript(root, theme.apply(root))
    transcript.model(
        "### Current Run Status\n\n"
        "The run is **unloaded**:\n"
        "* **`c0_g_per_cm3`**: initial concentration -- *blocks reconstruction*\n"
        "* `ls_units` is undeclared, so ls_units blocks I_rayleigh\n"
    )
    root.update_idletasks()

    shown = transcript.plain()
    for punctuation in ("###", "**", "`"):
        assert punctuation not in shown, f"{punctuation} was displayed, not obeyed"
    assert "Current Run Status" in shown
    # Underscores are never emphasis: this lab writes declaration keys in prose,
    # and a key italicised through the middle is a key nobody can search for.
    assert "c0_g_per_cm3" in shown and "ls_units" in shown

    def spans(tag):
        return len(transcript.text.tag_ranges(tag)) // 2

    assert spans("md_head") == 1
    assert spans("md_strong") >= 2
    assert spans("md_code") >= 2, "code nested inside bold must still be code"
    assert spans("md_em") == 1


def test_prose_without_markdown_is_left_alone(root):
    from polyrmc.gui import theme
    from polyrmc.gui.widgets import Transcript

    transcript = Transcript(root, theme.apply(root))
    plain = "The sweep put nine odd windows inside the safe band; 31 is narrowest."
    transcript.model(plain)
    root.update_idletasks()
    assert transcript.plain().strip() == plain


def _labels(widget) -> str:
    """Every piece of text in a widget tree, for asserting nothing was cut."""
    import tkinter as tk

    found = []
    for child in widget.winfo_children():
        if isinstance(child, tk.Label):
            found.append(str(child.cget("text")))
        found.append(_labels(child))
    return "\n".join(found)
