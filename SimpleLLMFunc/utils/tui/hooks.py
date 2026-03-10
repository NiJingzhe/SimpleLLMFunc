"""Custom-event hooks used by the Textual TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence, TypeAlias

from SimpleLLMFunc.hooks.events import CustomEvent


@dataclass
class ToolRenderSnapshot:
    """Current render snapshot for a running tool call."""

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    output: str = ""
    status: str = "running"
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEventRenderUpdate:
    """UI update returned by a custom-event hook."""

    append_output: str = ""
    replace_output: Optional[str] = None
    status: Optional[str] = None


ToolCustomEventHook: TypeAlias = Callable[
    [CustomEvent, ToolRenderSnapshot],
    Optional[ToolEventRenderUpdate],
]


def _extract_text_payload(data: Any) -> str:
    if isinstance(data, dict):
        text = data.get("text")
        if isinstance(text, str):
            return text
    if isinstance(data, str):
        return data
    return ""


def pyrepl_tool_event_hook(
    event: CustomEvent,
    snapshot: ToolRenderSnapshot,
) -> Optional[ToolEventRenderUpdate]:
    """Builtin hook for PyRepl custom events."""

    if snapshot.tool_name != "execute_code":
        return None

    if event.event_name == "kernel_stdout":
        text = _extract_text_payload(event.data)
        if text:
            return ToolEventRenderUpdate(append_output=text)
        return None

    if event.event_name == "kernel_stderr":
        text = _extract_text_payload(event.data)
        if text:
            return ToolEventRenderUpdate(append_output=text, status="stderr")
        return None

    if event.event_name == "kernel_input_request":
        prompt = ""
        if isinstance(event.data, dict):
            raw_prompt = event.data.get("prompt")
            if isinstance(raw_prompt, str):
                prompt = raw_prompt

        if prompt:
            if not prompt.endswith("\n"):
                prompt += "\n"
            return ToolEventRenderUpdate(append_output=prompt, status="waiting-input")

        return ToolEventRenderUpdate(status="waiting-input")

    if event.event_name in {
        "selfref_fork_start",
        "selfref_fork_spawned",
        "selfref_fork_end",
        "selfref_fork_error",
    }:
        if not isinstance(event.data, dict):
            return None

        fork_id = str(event.data.get("fork_id", ""))
        depth = event.data.get("depth", "")
        memory_key = str(event.data.get("memory_key", ""))

        if event.event_name == "selfref_fork_start":
            return ToolEventRenderUpdate(
                append_output=(
                    f"[fork start] id={fork_id} depth={depth} memory={memory_key}\n"
                ),
                status="running",
            )

        if event.event_name == "selfref_fork_spawned":
            return ToolEventRenderUpdate(
                append_output=(
                    f"[fork spawned] id={fork_id} depth={depth} memory={memory_key}\n"
                ),
                status="running",
            )

        if event.event_name == "selfref_fork_end":
            return ToolEventRenderUpdate(
                append_output=(
                    f"[fork done] id={fork_id} depth={depth} memory={memory_key}\n"
                ),
                status="success",
            )

        error_type = str(event.data.get("error_type", "RuntimeError"))
        error_message = str(event.data.get("error_message", ""))
        return ToolEventRenderUpdate(
            append_output=(
                f"[fork error] id={fork_id} depth={depth} {error_type}: {error_message}\n"
            ),
            status="error",
        )

    return None


def apply_tool_event_hooks(
    event: CustomEvent,
    snapshot: ToolRenderSnapshot,
    custom_hooks: Optional[Sequence[ToolCustomEventHook]] = None,
    builtin_hooks: Optional[Iterable[ToolCustomEventHook]] = None,
) -> Optional[ToolEventRenderUpdate]:
    """Apply custom and builtin hooks; return the first non-empty update."""

    all_hooks: list[ToolCustomEventHook] = []
    if custom_hooks:
        all_hooks.extend(custom_hooks)

    if builtin_hooks is None:
        all_hooks.append(pyrepl_tool_event_hook)
    else:
        all_hooks.extend(builtin_hooks)

    for hook in all_hooks:
        try:
            update = hook(event, snapshot)
        except Exception:
            continue
        if update is not None:
            return update

    return None


__all__ = [
    "ToolRenderSnapshot",
    "ToolEventRenderUpdate",
    "ToolCustomEventHook",
    "pyrepl_tool_event_hook",
    "apply_tool_event_hooks",
]
