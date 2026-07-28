"""Builds the research assistant agent (LangGraph via langchain's create_agent)."""

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

from src.config import MAX_TOKENS, MODEL, SYSTEM_PROMPT
from src.tools import ALL_TOOLS


def build_agent(checkpointer=None):
    """Create the agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for conversation memory.
            Defaults to an in-memory saver (memory lasts for the process).
    """
    model = ChatAnthropic(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
    )
    return create_agent(
        model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )
