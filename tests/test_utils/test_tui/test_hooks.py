"""Tests for TUI custom event hooks."""

from __future__ import annotations

from datetime import datetime, timezone

from SimpleLLMFunc.hooks.events import CustomEvent, ReActEventType
from SimpleLLMFunc.utils.tui.hooks import (
    ToolEventRenderUpdate,
    ToolRenderSnapshot,
    apply_tool_event_hooks,
    pyrepl_tool_event_hook,
)


def _build_event(event_name: str, data: object) -> CustomEvent:
    return CustomEvent(
        event_type=ReActEventType.CUSTOM_EVENT,
        timestamp=datetime.now(timezone.utc),
        trace_id="trace-1",
        func_name="agent",
        iteration=1,
        event_name=event_name,
        data=data,
        tool_name="execute_code",
        tool_call_id="call-1",
    )


def test_pyrepl_tool_event_hook_parses_stdout() -> None:
    """PyRepl builtin hook should append stdout text."""
    snapshot = ToolRenderSnapshot(
        tool_name="execute_code",
        tool_call_id="call-1",
        arguments={"code": "print(1)"},
    )
    event = _build_event("kernel_stdout", {"text": "hello\n"})

    update = pyrepl_tool_event_hook(event, snapshot)

    assert update is not None
    assert update.append_output == "hello\n"
    assert update.status is None


def test_apply_tool_event_hooks_uses_custom_hook_first() -> None:
    """Custom hook should override builtin behavior when matched."""
    snapshot = ToolRenderSnapshot(
        tool_name="execute_code",
        tool_call_id="call-1",
        arguments={"code": "print(1)"},
    )
    event = _build_event("kernel_stdout", {"text": "hello\n"})

    def custom_hook(_event: CustomEvent, _snapshot: ToolRenderSnapshot):
        return ToolEventRenderUpdate(append_output="custom\n", status="working")

    update = apply_tool_event_hooks(
        event=event,
        snapshot=snapshot,
        custom_hooks=[custom_hook],
    )

    assert update is not None
    assert update.append_output == "custom\n"
    assert update.status == "working"
