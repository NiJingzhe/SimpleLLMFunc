"""TUI demo for explicit SelfReference memory operations.

Run:
    poetry run python examples/tui_self_reference_example.py

What this example shows:
    1. Shared ``SelfReference`` injected into ``llm_chat``.
    2. Manual REPL binding via ``repl.attach_self_reference(self_reference)``.
    3. Agent-side memory operations with
       ``self_reference.memory["agent_main"].<method>()``.
    4. ``llm_chat`` auto-appends a SelfReference memory contract to system prompt.

Try these prompts in the chat UI:
    - "Use execute_code to print self_reference.memory.keys()"
    - "Set your system memory to always answer in one short paragraph"
    - "Append a note into self memory and print memory count"
"""

from __future__ import annotations

import os

from SimpleLLMFunc import OpenAICompatible, SelfReference, llm_chat, tui
from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.type import HistoryList


MEMORY_KEY = "agent_main"


def load_llm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    provider_json_path = os.path.join(current_dir, "provider.json")
    models = OpenAICompatible.load_from_json_file(provider_json_path)
    return models["openrouter"]["qwen/qwen3.5-397b-a17b"]


llm = load_llm()
self_reference = SelfReference()
repl = PyRepl()
repl.attach_self_reference(self_reference)


@tui(title="Self Reference Demo")
@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset],
    stream=True,
    enable_event=True,
    self_reference=self_reference,
    self_reference_key=MEMORY_KEY,
)
async def agent(message: str, history: HistoryList):
    """You are a practical coding assistant with a persistent Python REPL.

    Self-reference memory contract (strict):
    - Your memory handle is self_reference.memory["agent_main"].
    - Do not read or write any other memory key.
    - Never reassign the ``self_reference`` variable.
    - Modify memory only through memory-handle methods.

    Memory methods available:
    - count(): return number of messages in this memory key.
    - all(): return a deep-copied snapshot of all messages.
    - get(index): read one message by index.
    - append(message): append one message at tail.
    - insert(index, message): insert one message at index.
    - update(index, message): replace one message at index.
    - delete(index): remove one message by index.
    - clear(): clear all messages for this key.
    - replace(messages): replace full history with validated messages.
    - get_system_prompt(): read current system prompt memory.
    - set_system_prompt(text): overwrite system prompt memory.
    - append_system_prompt(text): append text to system prompt memory.

    Message validity requirements:
    - Keep OpenAI message format valid.
    - Keep assistant tool_calls matched with tool messages by tool_call_id.
    - Handle assistant messages with content=None safely.

    Execution notes:
    - Use execute_code for memory operations.
    - REPL state is persistent across calls.
    - Write direct snippets; never use ``if __name__ == "__main__":`` in REPL code.
    - ``input()`` is supported when interactive values are needed.
    - Forgetting memory is not ``reset_repl``.
    - ``reset_repl`` only clears REPL variables; forget memory via delete/replace/clear.

    Response style:
    - Be concise and action-oriented.
    """


if __name__ == "__main__":
    agent()
