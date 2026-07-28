"""Tier 1 -- bounded evaluator loops.

Each loop proposes deterministically and judges subjectively:

    propose -> judge -> re-propose, at most N iterations

The proposer generates a candidate set that is safe by construction. The judge
selects among candidates, or asks for a re-proposal with a direction hint. **It
cannot construct a candidate.** Loops are capped, and on exhaustion fall back to
the most conservative safe option.

Rejected candidates are logged alongside accepted ones. For fit-range selection
that is the entire point: which points were excluded from the linear region is
the scientific claim, and a reviewer must be able to see it.
"""

from polyrmc.tier1.loop import run_bounded_loop
from polyrmc.tier1.judge import ModelJudge, RecordedJudge, StaticJudge

__all__ = ["run_bounded_loop", "ModelJudge", "RecordedJudge", "StaticJudge"]
