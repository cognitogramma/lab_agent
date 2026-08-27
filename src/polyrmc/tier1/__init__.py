"""Tier 1 -- selection among candidates that tier 0 has already certified.

The proposer generates a candidate set that is safe by construction, and the
conservative option is taken from it. **Nothing here constructs a candidate**,
and nothing here consults a model.

There used to be a bounded propose-judge-re-propose harness in this package, so
that a judge could pick among the certified options and ask for a re-proposal.
The agent picks now, and it picks through the tool surface instead --
``sweep_windows`` offers the band as data, ``pick_window`` validates the choice
against it, and an invalid choice comes back as a refusal the model can read and
correct. That is the same guarantee with a better boundary, and keeping a second
mechanism for it meant two places to check whether a model could reach past the
gates.

Rejected candidates are still logged alongside accepted ones. For fit-range
selection that is the entire point: which points were excluded from the linear
region is the scientific claim, and a reviewer must be able to see it.
"""

from polyrmc.tier1.fit_range_selector import select_fit_range
from polyrmc.tier1.smoothing_selector import (
    conservative_window_candidate,
    select_smoothing_window,
)

__all__ = [
    "conservative_window_candidate",
    "select_fit_range",
    "select_smoothing_window",
]
