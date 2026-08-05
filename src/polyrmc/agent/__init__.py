"""The agent layer: the only place in the pipeline where a model is consulted.

Everything below this package is deterministic. The agent selects tools and
reasons about computed features; it never computes, and it cannot change a run
except through the tool surface, which rejects invalid actions in code.
"""

from polyrmc.agent.graph import GREETING, AgentGraph, Interrupt, Node, Step
from polyrmc.agent.lanes import Shortlist, reconcile
from polyrmc.agent.provider import Resilient, StubProvider

__all__ = [
    "GREETING",
    "AgentGraph",
    "Interrupt",
    "Node",
    "Resilient",
    "Shortlist",
    "Step",
    "StubProvider",
    "reconcile",
]
