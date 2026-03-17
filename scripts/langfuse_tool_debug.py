"""Run a deterministic tool-call trace for Langfuse debugging.

Usage:
    poetry run python scripts/langfuse_tool_debug.py

Notes:
    - Requires examples/provider.json with valid credentials.
    - The FileToolset workspace is scoped to ./sandbox.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from SimpleLLMFunc import OpenAICompatible, llm_chat
from SimpleLLMFunc.builtin import FileToolset
from SimpleLLMFunc.logger import async_log_context


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_DIR = PROJECT_ROOT / "sandbox"
PROVIDER_JSON = PROJECT_ROOT / "examples" / "provider.json"


def _load_llm() -> Any:
    if not PROVIDER_JSON.exists():
        raise FileNotFoundError(
            "examples/provider.json is required to run this script."
        )

    models = OpenAICompatible.load_from_json_file(str(PROVIDER_JSON))
    for provider_models in models.values():
        for model in provider_models.values():
            return model

    raise RuntimeError("No models found in examples/provider.json")


def _prepare_workspace() -> Path:
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    target = SANDBOX_DIR / "langfuse_debug.txt"
    target.write_text("hello from sandbox\n", encoding="utf-8")
    return target


async def _run() -> None:
    target = _prepare_workspace()
    llm = _load_llm()
    file_tools = FileToolset(SANDBOX_DIR).toolset

    @llm_chat(
        llm_interface=llm,
        toolkit=file_tools,
        stream=False,
        enable_event=False,
        max_tool_calls=1,
        tool_choice={"type": "function", "function": {"name": "read_file"}},
    )
    async def agent(message: str, history=None):
        """Always call read_file for the provided path, then reply DONE."""

    prompt = (
        "Call read_file for the file path 'langfuse_debug.txt' in the workspace. "
        "After the tool returns, reply with DONE."
    )

    trace_tag = f"langfuse_tool_debug_{int(time.time())}"
    async with async_log_context(trace_id=trace_tag):
        async for response, _history in agent(prompt, history=[]):
            if isinstance(response, str) and response.strip():
                print("response:", response.strip())

    print("trace_tag:", trace_tag)
    print("workspace:", SANDBOX_DIR.as_posix())
    print("target_file:", target.as_posix())


if __name__ == "__main__":
    asyncio.run(_run())
