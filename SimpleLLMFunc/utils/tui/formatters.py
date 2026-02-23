"""Formatting helpers for TUI rendering."""

from __future__ import annotations

import json
from typing import Any, Optional

from SimpleLLMFunc.type.llm import LLMUsage


def extract_stream_text(chunk: Any) -> str:
    """Extract normal content delta from a streaming chunk."""

    try:
        if not hasattr(chunk, "choices") or not chunk.choices:
            return ""
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if not delta:
            return ""
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            return content
    except Exception:
        return ""
    return ""


def extract_reasoning_delta(chunk: Any) -> str:
    """Extract reasoning delta when the provider exposes it."""

    try:
        if not hasattr(chunk, "choices") or not chunk.choices:
            return ""
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if not delta:
            return ""

        for attr in ("reasoning", "reasoning_content", "reasoning_text"):
            value = getattr(delta, attr, None)
            if isinstance(value, str) and value:
                return value

        details = getattr(delta, "reasoning_details", None)
        if isinstance(details, list):
            collected: list[str] = []
            for detail in details:
                if isinstance(detail, dict):
                    detail_type = str(detail.get("type", ""))
                    if detail_type.endswith("encrypted"):
                        continue
                    text = detail.get("text") or detail.get("data")
                    if isinstance(text, str) and text:
                        collected.append(text)
                else:
                    detail_type = str(getattr(detail, "type", ""))
                    if detail_type.endswith("encrypted"):
                        continue
                    text = getattr(detail, "text", None) or getattr(
                        detail, "data", None
                    )
                    if isinstance(text, str) and text:
                        collected.append(text)
            return "".join(collected)
    except Exception:
        return ""
    return ""


def extract_text_from_response(response: Any) -> str:
    """Extract message content from a non-streaming response."""

    try:
        if not hasattr(response, "choices") or not response.choices:
            return ""
        message = getattr(response.choices[0], "message", None)
        if not message:
            return ""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    except Exception:
        return ""
    return ""


def format_model_stats(
    execution_time: float,
    usage: Optional[LLMUsage],
    model_name: Optional[str] = None,
) -> str:
    """Format model call stats line."""

    model_label = model_name or "LLM"

    if usage is None:
        return f"{model_label} | {execution_time:.2f}s"

    return (
        f"{model_label} | {execution_time:.2f}s | "
        f"tokens {usage.prompt_tokens}/{usage.completion_tokens}/{usage.total_tokens} "
        "(in/out/total)"
    )


def format_tool_stats(execution_time: float, success: bool) -> str:
    """Format tool call stats line."""

    status = "success" if success else "error"
    return f"Tool | {execution_time:.2f}s | {status}"


def _format_primitive(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _render_fenced_code_block(code: str, language: str = "") -> list[str]:
    normalized = code.rstrip("\n")
    fence = "```"
    if "```" in normalized:
        fence = "````"

    fence_header = f"{fence}{language}" if language else fence
    return [fence_header, normalized, fence]


def format_tool_arguments_markdown(
    arguments: dict[str, Any],
    tool_name: Optional[str] = None,
) -> str:
    """Format tool arguments to readable markdown."""

    if not arguments:
        return "_No arguments_"

    lines: list[str] = []
    for key, value in arguments.items():
        if tool_name == "execute_code" and key == "code" and isinstance(value, str):
            lines.append(f"- `{key}`:")
            lines.extend(_render_fenced_code_block(value, language="python"))
            continue

        if isinstance(value, (dict, list)):
            pretty = json.dumps(value, ensure_ascii=False, indent=2)
            lines.append(f"- `{key}`:")
            lines.append("```json")
            lines.append(pretty)
            lines.append("```")
        else:
            lines.append(f"- `{key}`: `{_format_primitive(value)}`")
    return "\n".join(lines)


def format_tool_result_markdown(result: Any) -> str:
    """Format tool result to markdown-friendly text."""

    if isinstance(result, (dict, list)):
        pretty = json.dumps(result, ensure_ascii=False, indent=2)
        return f"```json\n{pretty}\n```"

    if isinstance(result, str):
        return result

    return _format_primitive(result)


def format_custom_event_fallback(event_name: str, data: Any) -> str:
    """Fallback rendering when no event hook handles the custom event."""

    if isinstance(data, dict):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = str(data)
    return f"[{event_name}] {payload}\n"


__all__ = [
    "extract_stream_text",
    "extract_reasoning_delta",
    "extract_text_from_response",
    "format_model_stats",
    "format_tool_stats",
    "format_tool_arguments_markdown",
    "format_tool_result_markdown",
    "format_custom_event_fallback",
]
