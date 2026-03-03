"""Adaptive runtime-fork demo (subagent-like pattern).

Run:
    poetry run python examples/tui_runtime_fork_subagent_example.py

What this example demonstrates:
1. One agent can decide when to act directly or fork itself.
2. Fork is triggered from REPL via ``runtime.fork.run(...)`` or ``runtime.fork.spawn(...)``.
3. Forked context inherits current memory snapshot.
4. Forked context can continue forking when deeper decomposition is useful.

Try prompts:
- "Write a short release note for a memory-fork feature."
- "If this task is complex, plan first and fork where useful."
- "This one is simple; if no fork is needed, just do it directly."
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


@tui(title="Adaptive Runtime-Fork Agent")  # type: ignore[misc]
@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset],
    stream=True,
    enable_event=True,
    self_reference=self_reference,
    self_reference_key=MEMORY_KEY,
)
async def agent(message: str, history: HistoryList):
    """You are one adaptive agent that can runtime-fork when needed.

    Decision policy:
    - You decide whether to answer directly, plan first, or fork.
    - Keep simple tasks direct; avoid unnecessary forks.
    - For complex tasks, you may plan briefly before delegating.
    - Forked contexts may continue planning and forking recursively.

    How to fork from REPL:
    1. Call ``execute_code``.
    2. In code, invoke:

       # Blocking fork:
       fork_result = runtime.fork.run("<one concrete executable task>")

       # Or concurrent pattern:
       handle = runtime.fork.spawn("<task A>")
       fork_result = runtime.fork.wait(handle["fork_id"])

       # Multiple spawned forks:
       # handles = [runtime.fork.spawn("<task A>"), ...]
       # fork_results = runtime.fork.wait_all([item["fork_id"] for item in handles])

       print("FORK_MEMORY_KEY:", fork_result["memory_key"])
       print("FORK_RESPONSE:", fork_result["response"])

       Replace ``<...>`` placeholders with concrete delegated work.
       In forked context, calling ``runtime.fork.run(...)`` without
       ``source_memory_key`` continues from current fork memory key.

    Execution guidance:
    - Use inherited memory as context, then decide the next best action.
    - If you fork, integrate child output into your final user response.
    - If no fork is needed, solve the request directly.

    Output style:
    - Clear, concise, action-oriented.
    """


if __name__ == "__main__":
    agent()
