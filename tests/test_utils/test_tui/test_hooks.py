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


def test_pyrepl_tool_event_hook_renders_fork_stream_progress() -> None:
    """PyRepl hook should render fork stream open/delta/close events."""
    snapshot = ToolRenderSnapshot(
        tool_name="execute_code",
        tool_call_id="call-1",
        arguments={"code": "runtime.selfref.fork.run('task')"},
    )

    open_event = _build_event(
        "selfref_fork_stream_open",
        {
            "fork_id": "fork_1",
            "depth": 1,
            "memory_key": "agent_main::fork::1",
        },
    )
    open_update = pyrepl_tool_event_hook(open_event, snapshot)

    assert open_update is not None
    assert open_update.status == "running"
    assert "[fork stream open]" in open_update.append_output

    delta_first = _build_event(
        "selfref_fork_stream_delta",
        {
            "fork_id": "fork_1",
            "depth": 1,
            "memory_key": "agent_main::fork::1",
            "text": "hello ",
        },
    )
    delta_first_update = pyrepl_tool_event_hook(delta_first, snapshot)

    assert delta_first_update is not None
    assert (
        delta_first_update.append_output == "[fork stream] id=fork_1 depth=1 | hello "
    )
    assert delta_first_update.status == "running"

    delta_second = _build_event(
        "selfref_fork_stream_delta",
        {
            "fork_id": "fork_1",
            "depth": 1,
            "memory_key": "agent_main::fork::1",
            "text": "world\nnext",
        },
    )
    delta_second_update = pyrepl_tool_event_hook(delta_second, snapshot)

    assert delta_second_update is not None
    assert (
        delta_second_update.append_output
        == "world\n[fork stream] id=fork_1 depth=1 | next"
    )
    assert delta_second_update.status == "running"

    close_event = _build_event(
        "selfref_fork_stream_close",
        {
            "fork_id": "fork_1",
            "depth": 1,
            "memory_key": "agent_main::fork::1",
            "status": "completed",
        },
    )
    close_update = pyrepl_tool_event_hook(close_event, snapshot)

    assert close_update is not None
    assert close_update.status == "running"
    assert close_update.append_output.startswith("\n[fork stream close]")


def test_pyrepl_tool_event_hook_marks_fork_stream_error_close() -> None:
    """Fork stream error close should update status to error."""
    snapshot = ToolRenderSnapshot(
        tool_name="execute_code",
        tool_call_id="call-1",
        arguments={"code": "runtime.selfref.fork.run('task')"},
    )

    delta_event = _build_event(
        "selfref_fork_stream_delta",
        {
            "fork_id": "fork_2",
            "depth": 2,
            "memory_key": "agent_main::fork::2",
            "text": "partial",
        },
    )
    _ = pyrepl_tool_event_hook(delta_event, snapshot)

    close_error_event = _build_event(
        "selfref_fork_stream_close",
        {
            "fork_id": "fork_2",
            "depth": 2,
            "memory_key": "agent_main::fork::2",
            "status": "error",
        },
    )
    close_error_update = pyrepl_tool_event_hook(close_error_event, snapshot)

    assert close_error_update is not None
    assert close_error_update.status == "error"
    assert close_error_update.append_output == "\n"


def test_pyrepl_tool_event_hook_preserves_interleaved_fork_readability() -> None:
    """Interleaved fork deltas should keep per-fork prefixes visible."""
    snapshot = ToolRenderSnapshot(
        tool_name="execute_code",
        tool_call_id="call-1",
        arguments={"code": "runtime.selfref.fork.spawn('task-a')"},
    )

    fork1_first = _build_event(
        "selfref_fork_stream_delta",
        {
            "fork_id": "fork_1",
            "depth": 1,
            "memory_key": "agent_main::fork::1",
            "text": "part-a",
        },
    )
    fork2 = _build_event(
        "selfref_fork_stream_delta",
        {
            "fork_id": "fork_2",
            "depth": 1,
            "memory_key": "agent_main::fork::2",
            "text": "other\n",
        },
    )
    fork1_second = _build_event(
        "selfref_fork_stream_delta",
        {
            "fork_id": "fork_1",
            "depth": 1,
            "memory_key": "agent_main::fork::1",
            "text": "part-b\n",
        },
    )

    first_update = pyrepl_tool_event_hook(fork1_first, snapshot)
    second_update = pyrepl_tool_event_hook(fork2, snapshot)
    third_update = pyrepl_tool_event_hook(fork1_second, snapshot)

    assert first_update is not None
    assert first_update.append_output == "[fork stream] id=fork_1 depth=1 | part-a"

    assert second_update is not None
    assert second_update.append_output == "\n[fork stream] id=fork_2 depth=1 | other\n"

    assert third_update is not None
    assert third_update.append_output == "[fork stream] id=fork_1 depth=1 | part-b\n"
