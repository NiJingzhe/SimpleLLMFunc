"""TUI demo for runtime memory primitives.

Run:
    poetry run python examples/tui_runtime_memory_example.py

What this example shows:
    1. Shared ``SelfReference`` injected into ``llm_chat``.
    2. Runtime bridge exposed inside ``execute_code`` as ``runtime``.
    3. Agent-side memory operations with ``runtime.memory.<method>(...)``.
    4. ``llm_chat`` auto-appends memory contract guidance into system prompt.

Try these prompts in the chat UI:
    - "Use execute_code to print runtime.memory.keys()"
    - "Set your system memory to always answer in one short paragraph"
    - "Append a note into runtime memory and print memory count"
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
repl.install_primitive_pack("self_reference", backend=self_reference)


@tui(title="Runtime Memory Demo")
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

    Runtime memory contract (strict):
    - Your memory key is "agent_main".
    - Use runtime.memory.* primitives for memory operations.
    - Do not read or write any other memory key.
    - Never reassign the ``runtime`` variable.

    Memory primitives available:
    - runtime.memory.keys()
    - runtime.memory.count("agent_main")
    - runtime.memory.all("agent_main")
    - runtime.memory.get("agent_main", index)
    - runtime.memory.append("agent_main", message)
    - runtime.memory.insert("agent_main", index, message)
    - runtime.memory.update("agent_main", index, message)
    - runtime.memory.delete("agent_main", index)
    - runtime.memory.clear("agent_main")
    - runtime.memory.replace("agent_main", messages)
    - runtime.memory.get_system_prompt("agent_main")
    - runtime.memory.set_system_prompt("agent_main", text)
    - runtime.memory.append_system_prompt("agent_main", text)

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
