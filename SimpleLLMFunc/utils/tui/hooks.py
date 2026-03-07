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


def _get_fork_stream_line_state(snapshot: ToolRenderSnapshot) -> dict[str, bool]:
    raw_state = snapshot.state.get("fork_stream_line_open")
    if not isinstance(raw_state, dict):
        raw_state = {}
        snapshot.state["fork_stream_line_open"] = raw_state

    line_state: dict[str, bool] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str):
            continue
        line_state[key] = bool(value)

    snapshot.state["fork_stream_line_open"] = line_state
    return line_state


def _format_fork_stream_delta(
    snapshot: ToolRenderSnapshot,
    *,
    fork_id: str,
    depth: Any,
    text: str,
) -> str:
    if not text:
        return ""

    line_state = _get_fork_stream_line_state(snapshot)
    rendered_parts: list[str] = []
    is_line_open = bool(line_state.get(fork_id, False))

    last_fork_id = snapshot.state.get("fork_stream_last_fork_id")
    if isinstance(last_fork_id, str) and last_fork_id and last_fork_id != fork_id:
        if bool(line_state.get(last_fork_id, False)):
            rendered_parts.append("\n")
            line_state[last_fork_id] = False
        is_line_open = False

    prefix = f"[fork stream] id={fork_id} depth={depth} | "

    for chunk in text.splitlines(keepends=True):
        if not is_line_open:
            rendered_parts.append(prefix)
            is_line_open = True

        rendered_parts.append(chunk)
        if chunk.endswith("\n"):
            is_line_open = False

    line_state[fork_id] = is_line_open
    snapshot.state["fork_stream_last_fork_id"] = fork_id
    return "".join(rendered_parts)


def _close_fork_stream(snapshot: ToolRenderSnapshot, fork_id: str) -> str:
    line_state = _get_fork_stream_line_state(snapshot)
    was_line_open = bool(line_state.pop(fork_id, False))
    if snapshot.state.get("fork_stream_last_fork_id") == fork_id:
        snapshot.state["fork_stream_last_fork_id"] = None
    if was_line_open:
        return "\n"
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
        "selfref_fork_stream_open",
        "selfref_fork_stream_delta",
        "selfref_fork_stream_close",
    }:
        if not isinstance(event.data, dict):
            return None

        fork_id = str(event.data.get("fork_id", ""))
        depth = event.data.get("depth", "")
        memory_key = str(event.data.get("memory_key", ""))

        if event.event_name == "selfref_fork_stream_open":
            _close_fork_stream(snapshot, fork_id)
            return ToolEventRenderUpdate(
                append_output=(
                    f"[fork stream open] id={fork_id} depth={depth} "
                    f"memory={memory_key}\n"
                ),
                status="running",
            )

        if event.event_name == "selfref_fork_stream_delta":
            text = _extract_text_payload(event.data)
            rendered_text = _format_fork_stream_delta(
                snapshot,
                fork_id=fork_id,
                depth=depth,
                text=text,
            )
            if not rendered_text:
                return None
            return ToolEventRenderUpdate(append_output=rendered_text, status="running")

        closing_newline = _close_fork_stream(snapshot, fork_id)
        status = str(event.data.get("status", "completed"))
        if status == "error":
            return ToolEventRenderUpdate(
                append_output=closing_newline,
                status="error",
            )

        return ToolEventRenderUpdate(
            append_output=(
                f"{closing_newline}[fork stream close] id={fork_id} "
                f"depth={depth} memory={memory_key} status={status}\n"
            ),
            status="running",
        )

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
