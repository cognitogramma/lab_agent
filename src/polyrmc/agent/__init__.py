"""The agent layer: the only place in the pipeline where a model is consulted.

Everything below this package is deterministic. The agent chooses which tools to
call and reasons about computed features; it never computes, and it cannot
change a run except through the tool surface, which rejects invalid actions in
code.

The model drives. :class:`~polyrmc.agent.orchestrator.Orchestrator` runs the
loop -- model, tools, results, repeat -- with commits gated on a person. The
two-lane judge in :mod:`polyrmc.agent.lanes` remains for the one decision that
genuinely wants an independent numeric and visual reading of the same trace.
"""

from polyrmc.agent.lanes import Shortlist, reconcile
from polyrmc.agent.orchestrator import (
    GREETING,
    Approval,
    ApprovalPolicy,
    Event,
    Orchestrator,
    allow_all,
    default_system_prompt,
    deny_all,
)
from polyrmc.agent.provider import Resilient, StubProvider
from polyrmc.agent.toolcall import (
    AssistantTurn,
    Message,
    ProviderUnavailable,
    ScriptedProvider,
    ToolCall,
    ToolResult,
    build_provider,
)
from polyrmc.agent.toolspec import Tool, ToolContext, build_registry

__all__ = [
    "GREETING",
    "Approval",
    "ApprovalPolicy",
    "AssistantTurn",
    "Event",
    "Message",
    "Orchestrator",
    "ProviderUnavailable",
    "Resilient",
    "ScriptedProvider",
    "Shortlist",
    "StubProvider",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolResult",
    "allow_all",
    "build_provider",
    "build_registry",
    "default_system_prompt",
    "deny_all",
    "reconcile",
]
