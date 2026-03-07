"""Unified TUI demo for runtime selfref primitives (memory + fork).

Run:
    poetry run python examples/tui_runtime_selfref_example.py

What this example demonstrates:
1. ``SelfReference`` mounted as a ``PyRepl`` runtime backend via ``selfref`` pack.
2. One agent can use both ``runtime.selfref.history.*`` and ``runtime.selfref.fork.*``.
3. ``llm_chat`` auto-appends runtime primitive guidance into system prompt.
4. Forked context inherits memory snapshot from current selfref key.

Try prompts:
- "Use execute_code to print runtime.list_primitive_specs()"
- "Append a durable preference with runtime.selfref.history.append_system_prompt"
- "Remember a note in memory, then read it back"
- "Split this task into two forks and merge their results"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from SimpleLLMFunc import OpenAICompatible, llm_chat, tui
from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.builtin.primitive import SelfReference
from SimpleLLMFunc.hooks.events import CustomEvent
from SimpleLLMFunc.type import HistoryList
from SimpleLLMFunc.utils.tui import ToolRenderSnapshot


MEMORY_KEY = "agent_main"
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEBUG_LOG_PATH = PROJECT_ROOT / "logs" / "tui_runtime_selfref_debug.log"


def _build_local_debug_logger() -> logging.Logger:
    logger = logging.getLogger("simplellmfunc.examples.tui_runtime_selfref")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(DEBUG_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info("=== unified selfref TUI session started ===")
    return logger


LOCAL_DEBUG_LOGGER = _build_local_debug_logger()


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return repr(value)


def local_debug_event_hook(
    event: CustomEvent,
    snapshot: ToolRenderSnapshot,
):
    """Write local debug logs for runtime selfref fork events."""
    if snapshot.tool_name != "execute_code":
        return None

    if not (
        event.event_name.startswith("selfref_fork")
        or event.event_name.startswith("kernel_")
    ):
        return None

    LOCAL_DEBUG_LOGGER.info(
        "tool_call_id=%s event=%s snapshot_status=%s data=%s",
        event.tool_call_id or "",
        event.event_name,
        snapshot.status,
        _safe_json(event.data),
    )
    return None


def load_llm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    provider_json_path = os.path.join(current_dir, "provider.json")
    models = OpenAICompatible.load_from_json_file(provider_json_path)
    return models["openrouter"]["qwen/qwen3.5-397b-a17b"]


llm = load_llm()
self_reference = SelfReference()
repl = PyRepl()
repl.install_primitive_pack("selfref", backend=self_reference)


@tui(custom_event_hook=[local_debug_event_hook], title="Unified Selfref Agent")  # type: ignore
@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset],
    stream=True,
    enable_event=True,
    self_reference_key=MEMORY_KEY,
)
async def agent(message: str, history: HistoryList):
    """You are a practical coding assistant with one persistent Python REPL.

    Runtime selfref guidance:
    - Use runtime.list_primitives() and runtime.list_primitive_specs() for discovery.
    - Use runtime.selfref.history.* for durable memory operations.
    - Use runtime.selfref.fork.* for self-fork delegation when useful.
    - Your memory key is "agent_main".
    - Do not read or write any other memory key.
    - Never reassign the ``runtime`` variable.

    Fork usage patterns from execute_code:
    - Blocking fork:
      fork_result = runtime.selfref.fork.run("one concrete executable task")
    - Concurrent fork:
      handle = runtime.selfref.fork.spawn("task A")
      fork_result = runtime.selfref.fork.wait(handle["fork_id"])
    - Multi-fork wait:
      handles = [runtime.selfref.fork.spawn("task A"), runtime.selfref.fork.spawn("task B")]
      fork_results = runtime.selfref.fork.wait_all([item["fork_id"] for item in handles])

    Execution notes:
    - Use execute_code for memory/fork operations.
    - REPL state is persistent across calls.
    - Write direct snippets; never use ``if __name__ == "__main__":`` in REPL code.
    - ``input()`` is supported when interactive values are needed.
    - Forgetting memory is not ``reset_repl``.
    - ``reset_repl`` only clears REPL variables; forget memory with
      ``runtime.selfref.history.delete`` / ``runtime.selfref.history.replace`` /
      ``runtime.selfref.history.clear``.

    Response style:
    - Clear, concise, action-oriented.
    """


if __name__ == "__main__":
    print(f"[selfref-debug] local event log: {DEBUG_LOG_PATH}")
    agent()
