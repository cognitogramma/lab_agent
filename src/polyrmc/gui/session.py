"""Getting one agent turn from a worker thread onto the screen.

The orchestrator is synchronous and it blocks. It calls a model, waits, runs
tools, waits, and -- when a tool commits -- stops dead in the middle of all that
to ask a person. Tk is the opposite shape: one thread, one event loop, and a
callback that takes four seconds is a window that has stopped repainting.

So the turn runs on a worker thread and reports through a queue. Three things in
this module are less obvious than that:

**The approval gate runs backwards.** Everywhere else the window asks and the
worker answers. Here the worker asks -- from inside
:meth:`~polyrmc.agent.orchestrator.Orchestrator.stream`, having already decided
it wants to commit -- and must not proceed until a person has answered. It puts
an :class:`ApprovalRequest` on the queue and blocks on it; the window shows the
dialog and calls :meth:`ApprovalRequest.reply`, which wakes the worker back up.
The gate is the same gate the terminal has, and it holds for the same reason:
the tool does not run until the answer comes back.

**A blocked worker must always be released.** If the window closes while a
commit is waiting, the answer never arrives and a daemon thread sits on an event
forever holding a half-finished run. :meth:`Bridge.close` answers any
outstanding request with a refusal, because "nobody answered" and "somebody
approved" have to stay distinguishable no matter how the session ended.

**The run is read on the thread that writes it.** The state panel is rendered
here, on the worker, between orchestrator events -- not on the window thread
while tools are mutating the run underneath it. Rendering state iterates the
declarations dict, and iterating a dict another thread is writing raises. The
window is handed finished text.

Nothing in this module imports Tk, so all of it can be tested without a display.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator

# Tools whose result means there is a new PNG on disk to offer.
FIGURE_TOOLS = frozenset({"plot_figure", "plot_columns"})


@dataclass
class ApprovalRequest:
    """One commit, put to the person at the window.

    Carries the tool and the call *unsummarised*. What gets shown is the
    window's business, but what it has to choose from must be everything, for
    the reason the terminal prints its arguments in full: a prompt that hides
    what is about to change teaches the reader to approve without looking.
    """

    tool: Any
    call: Any
    _answered: threading.Event = field(default_factory=threading.Event, repr=False)
    _answer: Any = field(default=None, repr=False)

    def reply(self, approval: Any) -> None:
        """Answer from the window thread. Idempotent -- first answer wins.

        Idempotent because two paths can answer: the dialog, and
        :meth:`Bridge.close` if the window goes away first. A second answer
        overwriting the first would let a refusal-on-shutdown turn into an
        approval that no one gave.
        """
        if self._answered.is_set():
            return
        self._answer = approval
        self._answered.set()

    def await_reply(self) -> Any:
        """Block the worker until the window answers."""
        self._answered.wait()
        return self._answer

    @property
    def answered(self) -> bool:
        return self._answered.is_set()


@dataclass(frozen=True)
class Signal:
    """One thing for the window to do, in the order it happened.

    ``kind`` is one of:

    ``event``    an orchestrator :class:`~polyrmc.agent.orchestrator.Event`
    ``state``    freshly rendered run state, as text
    ``figures``  a figure was written; rescan the directory
    ``ask``      an :class:`ApprovalRequest` waiting on a person
    ``crash``    the turn raised; the session survives, the turn does not
    ``done``     the turn ended -- ``ok``, ``capped`` or ``dead``
    """

    kind: str
    payload: Any = None


class Bridge:
    """One conversation, driven from a window.

    Holds the orchestrator and runs its turns one at a time. Turns are
    deliberately not concurrent: two questions in flight would interleave into
    one transcript and both would be answered against a run the other was
    changing.
    """

    def __init__(self, agent: Any, surface: Any) -> None:
        self.agent = agent
        self.surface = surface
        self.signals: "queue.Queue[Signal]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pending: ApprovalRequest | None = None
        self._closing = threading.Event()
        self._stopping = threading.Event()

    # --- the window's side ----------------------------------------------

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request_stop(self) -> None:
        """Ask the running turn to end at its next boundary.

        A poll, not a kill. The turn is somewhere inside a blocking HTTP request
        or a numpy pass over eighty thousand points, and neither can be
        interrupted without leaving the run in a state nobody can describe. So
        this asks, and the orchestrator answers at the next model call or tool
        -- which is why the window says "stopping" rather than "stopped".

        A commit waiting on approval is released as a refusal. Someone who hit
        stop is not answering yes to the dialog behind it.
        """
        if not self.busy:
            return
        self._stopping.set()
        pending = self._pending
        if pending is not None:
            from polyrmc.agent.orchestrator import Approval

            pending.reply(Approval(False, "you stopped the session"))

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def should_stop(self) -> bool:
        """The poll the orchestrator holds. Read from the turn thread."""
        return self._stopping.is_set() or self._closing.is_set()

    def submit(self, text: str) -> bool:
        """Start a turn. False if one is already running, or there is no model."""
        if self.agent is None or self.busy or self._closing.is_set():
            return False
        # Cleared per turn, not per session: a stop ends one question, and the
        # next one starts with nobody asking for it to end.
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run, args=(text,), name="polyrmc-turn", daemon=True
        )
        self._thread.start()
        return True

    def drain(self) -> Iterator[Signal]:
        """Everything reported since the last look. Never blocks."""
        while True:
            try:
                yield self.signals.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        """Stop accepting turns, and release anything waiting on an answer.

        The worker is a daemon thread, so a model call still in flight does not
        hold the process open. A commit waiting on approval is a different
        matter: it is blocked on an event nobody is left to set, and it is
        holding a tool it has not yet run. It is refused, and the refusal says
        why, because an audit trail that cannot tell a closed window from a
        researcher's yes is not an audit trail.
        """
        self._closing.set()
        pending = self._pending
        if pending is not None:
            from polyrmc.agent.orchestrator import Approval

            pending.reply(
                Approval(False, "the window was closed before this was answered")
            )

    # --- the worker's side ----------------------------------------------

    def ask(self, tool: Any, call: Any) -> Any:
        """The approval callback the orchestrator holds. Runs on the worker."""
        from polyrmc.agent.orchestrator import Approval

        if self._closing.is_set():
            return Approval(False, "the session is closing")
        request = ApprovalRequest(tool=tool, call=call)
        self._pending = request
        self.signals.put(Signal("ask", request))
        try:
            return request.await_reply() or Approval(False, "no answer")
        finally:
            self._pending = None

    def _run(self, text: str) -> None:
        outcome = "ok"
        try:
            for event in self.agent.stream(text):
                self.signals.put(Signal("event", event))
                if event.kind == "failed":
                    outcome = "dead"
                elif event.kind == "capped" and outcome == "ok":
                    outcome = "capped"
                if event.kind in {"result", "refused"}:
                    self._report_state()
                    if event.tool in FIGURE_TOOLS:
                        self.signals.put(Signal("figures"))
        except BaseException as error:  # noqa: BLE001 - a bug must not cost the record
            self.signals.put(Signal("crash", f"{type(error).__name__}: {error}"))
        finally:
            self._report_state()
            self.signals.put(Signal("done", outcome))

    def _report_state(self) -> None:
        """Render the run here, where nothing else is writing it."""
        from polyrmc.agent.toolspec import render_state

        try:
            self.signals.put(Signal("state", render_state(self.surface)))
        except Exception:  # noqa: BLE001 - a panel is not worth ending a turn over
            pass


__all__ = ["FIGURE_TOOLS", "ApprovalRequest", "Bridge", "Signal"]
