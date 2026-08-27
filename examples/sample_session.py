"""A full agent session, with the model's choices played from a plan.

Run this to see -- and to get a log of -- what a session looks like end to end,
on a machine with no API key and no quota left:

    python examples/sample_session.py

**What is real here and what is not.** Everything below the model is real. The
ARGEN file is read, the detectors run, the sweep computes an actual safe band,
the window is chosen *from that band* at the moment it is known, the CSV is
written and then verified by the same checks a live session uses. A tool that
refuses here refuses for the reason it would refuse for a researcher. What is
played is only the sequence of choices a model would have made -- which tool
next, and with what -- because that is the one part that costs tokens.

So the session log this writes is a real log with a rehearsed route through it.
That makes it the right thing to develop the *presentation* against, and the
wrong thing to quote as evidence of what a model decided.

The plan deliberately includes one call that gets refused: a smoothing window
outside the safe band. A transcript where nothing ever goes wrong is not much of
a sample, and the refusal path is the one worth looking at, since it is the only
place a tool's own words reach the terminal unshortened.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Runnable from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyrmc import cli  # noqa: E402
from polyrmc.agent.budget import Usage  # noqa: E402
from polyrmc.agent.toolcall import AssistantTurn, ToolCall  # noqa: E402

RAW = "data/raw/example_argen.txt"
DECLARATIONS = "examples/example_argen.decl.json"

QUESTION = (
    "Process this run: load it, check the declarations against the trace, see "
    "what we did with this sample before, pick a smoothing window, and emit a "
    "verified CSV. Stop and ask me before anything commits."
)


class PlannedProvider:
    """A brain whose next move is computed from live state, not recorded.

    A flat script cannot express "pick a window from the band", because the band
    does not exist until the sweep has run. Each step here is a callable
    resolved at the moment it is reached, so the arguments are drawn from what
    the tools actually returned -- the same thing a model does, and the reason
    this session can be refused by the real validator rather than by a rehearsal
    of one.

    The surface arrives through :meth:`bind` rather than the constructor. It has
    to be *the* surface the session mutates: building a second one from the same
    inputs looks equivalent and is not, because the sweep runs on one object and
    the plan then reads the band off the other, which is empty. That mistake is
    silent -- an empty band still produces a number to ask for -- so the plan
    refuses to guess when it finds one.
    """

    name = "planned"
    model = "offline-plan"

    def __init__(self) -> None:
        self.surface = None
        self.steps: list = []

    def bind(self, surface) -> None:
        self.surface = surface
        self.steps = list(build_plan(surface))

    def converse(self, system, messages, tools) -> AssistantTurn:
        if not self.steps:
            return AssistantTurn(
                text="Nothing further to do.", usage=_usage(len(messages))
            )
        turn = self.steps.pop(0)()
        turn.usage = _usage(len(messages))
        return turn


def _usage(rounds: int) -> Usage:
    """Plausible token counts, so the log's spend line is shaped like a real one.

    Input grows with the conversation because the transcript is re-sent every
    round; that is the whole reason a long session is expensive, and a sample
    that showed a flat figure would misrepresent the one cost that matters.
    """
    return Usage(requests=1, input_tokens=3800 + 1450 * rounds, output_tokens=120)


def say(text: str, **arguments) -> AssistantTurn:
    return AssistantTurn(text=text)


def call(name: str, text: str = "", **arguments) -> AssistantTurn:
    return AssistantTurn(
        text=text, tool_calls=[ToolCall(f"c_{name}", name, arguments)]
    )


def build_plan(surface):
    """The route, as a list of steps resolved when each is reached."""

    def band() -> list[int]:
        found = [int(candidate.value) for candidate in surface.get_shortlist()]
        if not found:
            raise SystemExit(
                "the sweep produced no safe band, so there is no window to "
                "choose and this sample cannot continue. Run "
                "`polyrmc process` on the same file to see why."
            )
        return found

    def choose_out_of_band():
        """Ask for a window the sweep did not offer, to show a refusal."""
        return call(
            "pick_window",
            "The band looks conservative to me; let me try a wider window.",
            window_length=max(band()) + 2,
        )

    def choose_from_band():
        return call(
            "pick_window",
            "Understood -- taking the widest window the band actually allows.",
            window_length=max(band()),
        )

    def decide_pending():
        """Answer the anomaly checkpoint the way a researcher would here.

        Keeping a flagged region is the conservative answer on a dilution run:
        the decay *is* the measurement, so a detector that fired on it has found
        the signal rather than a fault.
        """
        pending = [a for a in surface.state.anomalies if a.decision == "pending"]
        if not pending:
            return call("get_state")
        first = pending[0]
        return call(
            "set_anomaly_decision",
            start_index=first.start_index,
            end_index=first.end_index,
            decision="keep",
        )

    plan = [
        lambda: call(
            "load_run",
            "Loading the file first -- nothing can be computed before the "
            "time base is reconstructed and the channel selected.",
        ),
        lambda: call(
            "check_declarations",
            "Now testing what was declared against what the trace shows.",
        ),
        lambda: call(
            "search_past_runs",
            "Before choosing anything, let me see what this lab did with this "
            "sample before.",
            query="acd_dilution",
        ),
        lambda: call("sweep_windows", "Computing the safe band of windows."),
        choose_out_of_band,
        choose_from_band,
    ]
    # One decision per region the detectors could flag. The plan does not know
    # in advance how many there will be, so it offers enough steps and the
    # no-pending branch falls through to a harmless read.
    plan.extend([decide_pending] * 8)
    plan.extend(
        [
            lambda: call("emit_csv", "Writing the boundary CSV."),
            lambda: call("verify_output", "Running the pipeline's own checks."),
            lambda: call(
                "write_note",
                subject="example_argen: window and anomaly decisions",
                body=(
                    "Took the widest window in the safe band. The flagged "
                    "regions are the dilution decay itself and were kept, not "
                    "excised."
                ),
                tags=["example", "acd_dilution"],
            ),
            lambda: say(
                "Done. The run is loaded, the declarations agree with the "
                "trace, the window came from the computed safe band, and the "
                "emitted CSV passed verification. The flagged regions were kept "
                "rather than excised, which is the conservative reading on a "
                "dilution run -- the decay is the measurement."
            ),
        ]
    )
    return plan


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    # Its own runs directory and its own notebook, never the lab's.
    #
    # This session commits: it emits a CSV with a provenance sidecar, and it
    # files a note. Left on the defaults, both land in the directories the agent
    # searches as memory -- so "what did we do with this sample before" starts
    # answering partly out of a demonstration, with nothing marking which rows
    # were real. A sample that quietly seeds the run history is worse than no
    # sample at all.
    sandbox = root / "data" / "samples"
    sandbox.mkdir(parents=True, exist_ok=True)

    args = types.SimpleNamespace(
        stage="agent",
        raw=str(root / RAW),
        declarations=str(root / DECLARATIONS),
        config=None,
        answers=None,
        author="example",
        no_prompt=True,          # nobody is at this terminal
        auto_approve=True,       # so commits run and the sample reaches an emit
        provider="planned",
        model=None,
        max_steps=24,
        notes_dir=str(sandbox / "memory"),
        prompt=QUESTION,
        budget_requests=0,
        budget_tokens=0,
        out=str(sandbox / "sample-session_ls8.csv"),
        run_id="sample-session",
        runs_dir=str(sandbox / "runs"),
        verbose="--verbose" in sys.argv,
        channel=None,
        force=False,
    )

    provider = PlannedProvider()
    cli._resolve_provider = lambda resolved_args, interactive: provider

    # The surface the session mutates is built inside run_agent and handed to
    # the orchestrator. Intercepting that construction is how the plan gets the
    # real one; run_agent stays unaware that a planned provider exists.
    from polyrmc.agent import orchestrator as orchestrator_module

    real = orchestrator_module.Orchestrator

    def bind_then_build(**kwargs):
        kwargs["provider"].bind(kwargs["context"].surface)
        return real(**kwargs)

    orchestrator_module.Orchestrator = bind_then_build
    try:
        return cli.run_agent(args)
    finally:
        orchestrator_module.Orchestrator = real


if __name__ == "__main__":
    raise SystemExit(main())
