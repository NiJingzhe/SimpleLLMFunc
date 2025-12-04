"""Assistant message construction helpers."""

from __future__ import annotations

from typing import Any, Dict, List


def build_assistant_tool_message(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construct the assistant message containing tool call descriptors."""

    if tool_calls:
        return {"role": "assistant", "content": None, "tool_calls": tool_calls}
    return {}


def build_assistant_response_message(content: str) -> Dict[str, Any]:
    """Construct a plain assistant response message."""

    return {
        "role": "assistant",
        "content": content,
    }

