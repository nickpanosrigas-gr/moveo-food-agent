# main.py
from __future__ import annotations

import json
import signal
import sys

from agent import FoodOrderAgent
from config.settings import settings
from src.utils.logger import setup_logging

# Initialize unified logging
setup_logging()


def main() -> None:
    agent = FoodOrderAgent(warmup_on_init=True)
    unloaded = False

    def _cleanup_resources(*_args) -> None:
        nonlocal unloaded
        if not unloaded:
            unloaded = True
            agent.unload_sync()

    signal.signal(signal.SIGTERM, lambda *_a: (_cleanup_resources(), sys.exit(0)))

    print("====================================================")
    print("  Moveo AI Food Ordering Agent (Interactive Mode)")
    print(f"  Backend: LLM={settings.LLM_PROVIDER} ({agent.llm.model})")
    print(f"           Speech={settings.SPEECH_PROVIDER}")
    print("  Type 'quit' or 'exit' to stop.")
    print("====================================================\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            print()

            try:
                response = agent.send(user_input)
                print()
                print(f"Agent: {response['message']}")

                if "tool_calls" in response:
                    print(f"  [MCP Tool Call]: {json.dumps(response['tool_calls'], indent=2)}")
                print()
            except KeyboardInterrupt:
                print("\n[Cancelled current turn]\n")
            except Exception as exc:
                print(f"\n[Error]: {exc}\n")
    finally:
        _cleanup_resources()
        print("Goodbye!")


if __name__ == "__main__":
    main()