"""The two-lane judge, and what happens when the lanes disagree.

Design section 6.7, stages 2 and 3. A deterministic sweep has already produced
the **safe band**: every window in it whitens the residual and preserves the
kinetic landmarks. Stage 1 guarantees that no pathological window is reachable
by anything downstream, which is what makes the rest of this safe to hand to a
model at all.

Two lanes then read the same run independently:

*The LLM lane reads the numbers* -- a shortlist table of about sixteen rows,
plus the noise floor, the anomaly inventory, and the declared sample class. This
is reasoning over computed features, not arithmetic over eighty thousand points.

*The VLM lane reads the picture* -- the scatter with the shortlisted overlays
drawn on it, judged against a written rubric. It catches over-smoothing that the
residual statistics tolerate but the eye rejects.

Neither lane computes anything, and neither can leave the band: a choice outside
it is rejected by code before it reaches the run.

**Disagreement is information, not a fault.** Two independent readings of the
same trace reaching different conclusions is the cheapest available signal that
a run is unusual. It is surfaced to the researcher with both rationales side by
side. It is never averaged, and neither lane is preferred by default -- averaging
two defensible answers produces one that nobody argued for and that no rationale
explains.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from polyrmc.agent.provider import Resilient
from polyrmc.state import JudgeVerdict

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a rubric from disk.

    Rubrics live in files rather than in code on purpose. The subjectivity of
    this step is meant to be captured as an auditable, editable rubric -- a
    researcher who disagrees with the taste encoded here should be able to
    change it and see the effect, without touching Python.
    """
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


@dataclass
class Shortlist:
    """The safe band and the diagnostics that describe it."""

    candidates: list[Any]
    noise_floor: float
    context: dict[str, Any]

    @property
    def windows(self) -> list[int]:
        return [int(c.value) for c in self.candidates]

    @property
    def smallest(self) -> int:
        """The deterministic fallback: least smoothing, most data-faithful."""
        return min(self.windows)

    def render_table(self) -> str:
        """The shortlist as a table a model can read.

        Sixteen rows of computed diagnostics. Rendering it here rather than
        letting a model near the series is the concrete form of "models judge,
        they never compute".
        """
        header = (
            "window | residual_acf1 | slope_retention | amplitude_retention | "
            "residual_sigma"
        )
        lines = [header, "-" * len(header)]
        for candidate in self.candidates:
            f = candidate.features
            lines.append(
                f"{int(candidate.value):>6} | {f.get('residual_acf1', float('nan')):>13.4f}"
                f" | {f.get('slope_retention', float('nan')):>15.4f}"
                f" | {f.get('amplitude_retention', float('nan')):>19.4f}"
                f" | {f.get('residual_sigma', float('nan')):>14.6g}"
            )
        return "\n".join(lines)

    def render_context(self) -> str:
        lines = [f"noise_floor: {self.noise_floor:.6g}"]
        for key, value in sorted(self.context.items()):
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


def parse_verdict(raw: str, band: Sequence[int], lane: str) -> JudgeVerdict:
    """Read a lane's reply, and refuse anything outside the band.

    Accepts JSON, and falls back to finding a band member in free text -- a
    model that answers well but formats badly should not cost the run its
    judgment. What is never accepted is a window outside the band, however
    confidently it is argued for.
    """
    window: int | None = None
    rationale = raw.strip()
    cited: list[str] = []

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            window = int(payload.get("window_length", payload.get("window")))
            rationale = str(payload.get("rationale", "")).strip()
            cited = [str(c) for c in payload.get("cited_diagnostics", [])]
    except (ValueError, TypeError):
        pass

    if window is None:
        # Free text. Take the first number that is actually in the band, rather
        # than the first number of any kind -- rationales are full of numbers.
        for token in re.findall(r"\d+", raw):
            if int(token) in band:
                window = int(token)
                break

    if window is None:
        raise ValueError(
            f"the {lane} lane named no window from the safe band {list(band)}. "
            f"Reply was: {raw[:200]!r}"
        )
    if window not in band:
        raise ValueError(
            f"REJECTED. The {lane} lane chose window {window}, which is not in the "
            f"safe band {list(band)}. The band is what deterministic checks have "
            "certified; a choice outside it is not available."
        )
    if not cited:
        cited = [
            name
            for name in (
                "residual_acf1",
                "slope_retention",
                "amplitude_retention",
                "residual_sigma",
                "noise_floor",
            )
            if name in raw
        ]
    return JudgeVerdict(
        lane=lane, window_length=window, rationale=rationale, cited_diagnostics=cited
    )


def run_llm_lane(
    shortlist: Shortlist, guard: Resilient, system: str | None = None
) -> tuple[JudgeVerdict | None, list[str]]:
    """Ask the numeric lane. Returns the verdict and any errors met on the way."""
    prompt = (
        f"{shortlist.render_context()}\n\n"
        f"Shortlist (all of these passed the safety gates):\n"
        f"{shortlist.render_table()}\n\n"
        f"Choose one window from {shortlist.windows}. Reply as JSON with keys "
        f"window_length, rationale, cited_diagnostics."
    )
    outcome = guard.call(
        system or load_prompt("llm_lane.md"),
        prompt,
        fallback=lambda: None,
        parse=lambda raw: parse_verdict(raw, shortlist.windows, "llm"),
    )
    return outcome.value, outcome.errors


def run_vlm_lane(
    shortlist: Shortlist,
    guard: Resilient,
    image: bytes | None,
    system: str | None = None,
) -> tuple[JudgeVerdict | None, list[str]]:
    """Ask the visual lane. Without an image there is nothing to look at."""
    if image is None:
        return None, ["vlm lane skipped: no rendered plot was supplied"]
    prompt = (
        f"The plot shows the raw scatter with these windows overlaid: "
        f"{shortlist.windows}.\n\n{shortlist.render_context()}\n\n"
        f"Choose one. Reply as JSON with keys window_length, rationale."
    )
    outcome = guard.call(
        system or load_prompt("vlm_lane.md"),
        prompt,
        fallback=lambda: None,
        parse=lambda raw: parse_verdict(raw, shortlist.windows, "vlm"),
        image=image,
    )
    return outcome.value, outcome.errors


@dataclass
class Reconciliation:
    """The outcome of stage 3."""

    window_length: int | None
    status: str
    """One of: both_lanes, llm_only, vlm_only, escalated, fallback."""

    disagreed: bool
    fallback_taken: bool
    needs_researcher: bool
    message: str


def reconcile(
    llm: JudgeVerdict | None, vlm: JudgeVerdict | None, shortlist: Shortlist
) -> Reconciliation:
    """Combine the two lanes, or refuse to.

    The one thing this must not do is invent a compromise. Averaging two
    defensible windows produces a third that neither lane argued for and no
    rationale explains, and it hides exactly the signal that made the
    disagreement worth having.
    """
    if llm and vlm:
        if llm.window_length == vlm.window_length:
            return Reconciliation(
                llm.window_length,
                "both_lanes",
                disagreed=False,
                fallback_taken=False,
                needs_researcher=False,
                message=(
                    f"Both lanes chose window {llm.window_length}.\n"
                    f"  numbers: {llm.rationale}\n"
                    f"  picture: {vlm.rationale}"
                ),
            )
        return Reconciliation(
            None,
            "escalated",
            disagreed=True,
            fallback_taken=False,
            needs_researcher=True,
            message=(
                "The two lanes disagree, which is worth a look -- it usually means "
                "the run is unusual rather than that one lane is wrong.\n"
                f"  numbers chose {llm.window_length}: {llm.rationale}\n"
                f"  picture chose {vlm.window_length}: {vlm.rationale}\n"
                f"Safe band is {shortlist.windows}. Both choices are safe; pick one."
            ),
        )

    survivor = llm or vlm
    if survivor:
        lane = "llm_only" if llm else "vlm_only"
        missing = "vlm" if llm else "llm"
        return Reconciliation(
            survivor.window_length,
            lane,
            disagreed=False,
            fallback_taken=False,
            needs_researcher=False,
            message=(
                f"The {missing} lane was unavailable, so window "
                f"{survivor.window_length} rests on one reading: {survivor.rationale}"
            ),
        )

    return Reconciliation(
        shortlist.smallest,
        "fallback",
        disagreed=False,
        fallback_taken=True,
        needs_researcher=False,
        message=(
            f"Neither lane was available. Took the smallest window in the safe band "
            f"({shortlist.smallest}) -- least smoothing, most faithful to the data. "
            "The run is scientifically valid; what was lost is judgment, not "
            "correctness."
        ),
    )


__all__ = [
    "PROMPT_DIR",
    "Reconciliation",
    "Shortlist",
    "load_prompt",
    "parse_verdict",
    "reconcile",
    "run_llm_lane",
    "run_vlm_lane",
]
