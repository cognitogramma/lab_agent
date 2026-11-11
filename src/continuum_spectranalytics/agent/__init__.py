"""The agent layer: the only place in the pipeline where a model is consulted.

Everything below this package is deterministic. The agent chooses which tools to
call and reasons about computed features; it never computes, and it cannot
change a run except through the tool surface, which rejects invalid actions in
code.

The model drives. :class:`~continuum_spectranalytics.agent.orchestrator.Orchestrator` runs the
loop -- model, tools, results, repeat -- with commits gated on a person. There
is exactly one way for a model to reach a run, and it is that loop through those
tools.

There used to be a second way. A bounded propose-judge-re-propose harness sat in
tier 1 and called a model to pick among certified candidates, over its own
text-in/text-out transport, with a two-lane numeric-and-visual reading for the
smoothing decision. It worked, and it was the architecture this one replaced:
the route was fixed in Python and a model was consulted at one node of it. Two
mechanisms for "a model may select but never construct" meant two places to
check that the guarantee held, and only one of them is the way anybody drives
this system now.
"""

from continuum_spectranalytics.agent.orchestrator import (
    GREETING,
    Approval,
    ApprovalPolicy,
    Event,
    Orchestrator,
    allow_all,
    default_system_prompt,
    deny_all,
)
from continuum_spectranalytics.agent.toolcall import (
    AssistantTurn,
    Message,
    ProviderCallFailed,
    ProviderUnavailable,
    ScriptedProvider,
    ToolCall,
    ToolResult,
    build_provider,
)
from continuum_spectranalytics.agent.toolspec import Tool, ToolContext, build_registry
from continuum_spectranalytics.agent.transcript import Narrator, SessionLog

__all__ = [
    "GREETING",
    "Approval",
    "ApprovalPolicy",
    "AssistantTurn",
    "Event",
    "Message",
    "Narrator",
    "Orchestrator",
    "ProviderCallFailed",
    "ProviderUnavailable",
    "ScriptedProvider",
    "SessionLog",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolResult",
    "allow_all",
    "build_provider",
    "build_registry",
    "default_system_prompt",
    "deny_all",
]
