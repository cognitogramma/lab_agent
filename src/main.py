"""Interactive CLI for the lab research assistant.

Run from the project root:
    python -m src.main
"""

import os
import sys
import uuid

from src.agent import build_agent


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Create a .env file (see .env.example)\n"
            "or set the environment variable, then run again."
        )
        sys.exit(1)

    agent = build_agent()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Lab research assistant (model: claude-opus-4-8). Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        try:
            for step in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="values",
            ):
                message = step["messages"][-1]
                # Announce tool calls so the user can see what the agent is doing.
                if getattr(message, "tool_calls", None):
                    for call in message.tool_calls:
                        print(f"  [using tool: {call['name']}]")
            print(f"\nassistant> {message.text()}\n")
        except Exception as exc:  # surface API/network errors without crashing the loop
            print(f"\n[error] {exc}\n")


if __name__ == "__main__":
    main()
