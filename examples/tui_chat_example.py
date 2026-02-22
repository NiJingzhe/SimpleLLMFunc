"""SimpleLLMFunc Textual TUI example.

Run:
    poetry run python examples/tui_chat_example.py
"""

from __future__ import annotations

import os

from SimpleLLMFunc import OpenAICompatible, llm_chat, tui
from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.type import HistoryList


def load_llm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    provider_json_path = os.path.join(current_dir, "provider.json")
    models = OpenAICompatible.load_from_json_file(provider_json_path)
    return models["openrouter"]["qwen/qwen3.5-397b-a17b"]

llm = load_llm()
repl = PyRepl()


@tui()  # type: ignore
@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset],
    stream=True,
    enable_event=True,
)
async def agent(message: str, history: HistoryList):
    """You are a task assistant.

    - Keep answers concise and practical.
    - Use execute_code for math/programming verification when needed.
    - Use batch_process tool when user asks for step-by-step task execution.
    """


if __name__ == "__main__":
    agent()
