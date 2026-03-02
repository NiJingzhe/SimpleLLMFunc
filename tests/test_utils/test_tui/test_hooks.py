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


def test_pyrepl_tool_event_hook_handles_input_request() -> None:
    """PyRepl hook should render input prompt and waiting status."""
    snapshot = ToolRenderSnapshot(
        tool_name="execute_code",
        tool_call_id="call-1",
        arguments={"code": "name = input('Name: ')"},
    )
    event = _build_event(
        "kernel_input_request",
        {"request_id": "req-1", "prompt": "Name: "},
    )

    update = pyrepl_tool_event_hook(event, snapshot)

    assert update is not None
    assert update.status == "waiting-input"
    assert update.append_output == "Name: \n"


def test_pyrepl_tool_event_hook_renders_fork_start() -> None:
    """PyRepl hook should render fork start lifecycle output."""
    snapshot = ToolRenderSnapshot(
        tool_name="execute_code",
        tool_call_id="call-1",
        arguments={"code": "self_reference.instance.fork_spawn('x')"},
    )
    event = _build_event(
        "selfref_fork_start",
        {
            "fork_id": "fork_1",
            "depth": 1,
            "memory_key": "agent_main::fork::1",
        },
    )

    update = pyrepl_tool_event_hook(event, snapshot)

    assert update is not None
    assert update.status == "running"
    assert "[fork start]" in update.append_output
    assert "fork_1" in update.append_output


def test_pyrepl_tool_event_hook_renders_fork_error() -> None:
    """PyRepl hook should render fork error lifecycle output."""
    snapshot = ToolRenderSnapshot(
        tool_name="execute_code",
        tool_call_id="call-1",
        arguments={"code": "self_reference.instance.fork_wait('fork_1')"},
    )
    event = _build_event(
        "selfref_fork_error",
        {
            "fork_id": "fork_1",
            "depth": 2,
            "error_type": "RuntimeError",
            "error_message": "boom",
        },
    )

    update = pyrepl_tool_event_hook(event, snapshot)

    assert update is not None
    assert update.status == "error"
    assert "[fork error]" in update.append_output
    assert "boom" in update.append_output
