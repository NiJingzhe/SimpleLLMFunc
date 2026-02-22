"""SimpleLLMFunc Textual TUI example.

Run:
    poetry run python examples/tui_chat_example.py
"""

from __future__ import annotations

import asyncio
import os

from SimpleLLMFunc import OpenAICompatible, llm_chat, tool, tui
from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.hooks import ToolEventEmitter
from SimpleLLMFunc.hooks.events import CustomEvent
from SimpleLLMFunc.utils.tui import ToolEventRenderUpdate, ToolRenderSnapshot


def load_llm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    provider_json_path = os.path.join(current_dir, "provider.json")
    models = OpenAICompatible.load_from_json_file(provider_json_path)
    return models["openrouter"]["qwen/qwen3.5-397b-a17b"]


@tool(name="batch_process", description="Process items with progress events")
async def batch_process(
    items: list[str],
    event_emitter: ToolEventEmitter | None = None,
) -> str:
    """Process items and report progress via custom events."""

    total = len(items)
    for index, item in enumerate(items, start=1):
        await asyncio.sleep(0.1)
        if event_emitter:
            await event_emitter.emit(
                "batch_progress",
                {
                    "current": index,
                    "total": total,
                    "item": item,
                    "percent": int(index * 100 / total),
                },
            )
    return f"done: {total} items"


def batch_progress_hook(
    event: CustomEvent,
    _snapshot: ToolRenderSnapshot,
) -> ToolEventRenderUpdate | None:
    """Render batch progress events as concise status lines."""

    if event.event_name != "batch_progress" or not isinstance(event.data, dict):
        return None

    percent = event.data.get("percent")
    current = event.data.get("current")
    total = event.data.get("total")
    item = event.data.get("item")

    return ToolEventRenderUpdate(
        append_output=(f"[{percent}%] processing {item} ({current}/{total})\n")
    )


llm = load_llm()
repl = PyRepl()


@tui(custom_event_hook=[batch_progress_hook])
@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset, batch_process],
    stream=True,
    enable_event=True,
)
async def agent(message: str, history=None):
    """You are a task assistant.

    - Keep answers concise and practical.
    - Use execute_code for math/programming verification when needed.
    - Use batch_process tool when user asks for step-by-step task execution.
    """


if __name__ == "__main__":
    agent()
