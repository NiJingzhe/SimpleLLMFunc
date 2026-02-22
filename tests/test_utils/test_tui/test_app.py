"""Regression tests for Textual TUI widget mounting."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import Mock

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from SimpleLLMFunc.hooks.input_stream import AgentInputRouter
from SimpleLLMFunc.hooks.stream import ReactOutput
from SimpleLLMFunc.utils.tui.app import AgentTUIApp


async def _unused_agent(_message: str) -> AsyncGenerator[ReactOutput, None]:
    if False:
        yield None


def _make_app(input_router: AgentInputRouter | None = None) -> AgentTUIApp:
    return AgentTUIApp(
        agent_func=_unused_agent,
        input_param="message",
        history_param=None,
        static_kwargs={},
        input_router=input_router,
    )


@pytest.mark.asyncio
async def test_append_user_message_mounts_children_after_parent() -> None:
    """User message mount order should not raise Textual MountError."""
    app = _make_app()

    async with app.run_test():
        await app._append_user_message("hello")

        bubble = app.query_one(".user-bubble")
        bubble.query_one(".role", Static)
        bubble.query_one(".body", Static)


@pytest.mark.asyncio
async def test_model_and_tool_blocks_mount_children_after_parent() -> None:
    """Model and tool sections should mount without detached-parent errors."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-1",
            tool_name="execute_code",
            arguments={"code": "print(1)"},
        )

        app.query_one(".model-bubble")
        tool_block = app.query_one(".tool-call")
        tool_block.query_one(".tool-status", Static)


@pytest.mark.asyncio
async def test_short_message_height_is_not_viewport_sized() -> None:
    """Short message bubbles should size to content instead of filling viewport."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app._append_user_message("hello")
        await pilot.pause(0.05)

        bubble = app.query_one(".user-bubble")
        chat_log = app.query_one("#chat-log", VerticalScroll)

        assert bubble.region.height < (chat_log.region.height // 2)


@pytest.mark.asyncio
async def test_ctrl_q_quits_app() -> None:
    """Ctrl+Q should provide a direct exit keybinding."""
    app = _make_app()

    async with app.run_test() as pilot:
        assert app.is_running
        await pilot.press("ctrl+q")
        await pilot.pause(0.05)
        assert not app.is_running


@pytest.mark.asyncio
async def test_streaming_model_content_auto_scrolls_to_bottom() -> None:
    """Model streaming updates should keep chat log pinned to latest content."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.append_model_content("llm_call_1", "word " * 2200)
        await pilot.pause(0.05)

        chat_log = app.query_one("#chat-log", VerticalScroll)
        assert chat_log.max_scroll_y > 0

        chat_log.scroll_home(animate=False, immediate=True)
        await pilot.pause(0.02)
        assert chat_log.scroll_y == 0

        await app.append_model_content("llm_call_1", "tail " * 200)
        await pilot.pause(0.05)

        assert chat_log.scroll_y >= chat_log.max_scroll_y - 0.5


@pytest.mark.asyncio
async def test_streaming_tool_output_auto_scrolls_to_bottom() -> None:
    """Tool streaming output should keep chat log pinned to latest content."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-1",
            tool_name="execute_code",
            arguments={"code": "print(1)"},
        )
        await app.append_tool_output(
            "call-1", "".join(f"line {i}\n" for i in range(160))
        )
        await pilot.pause(0.05)

        chat_log = app.query_one("#chat-log", VerticalScroll)
        assert chat_log.max_scroll_y > 0

        chat_log.scroll_home(animate=False, immediate=True)
        await pilot.pause(0.02)
        assert chat_log.scroll_y == 0

        await app.append_tool_output("call-1", "tail line\n")
        await pilot.pause(0.05)

        assert chat_log.scroll_y >= chat_log.max_scroll_y - 0.5


@pytest.mark.asyncio
async def test_pending_tool_input_submits_to_pyrepl() -> None:
    """When PyRepl requests input, Enter should submit to request id."""
    submit_mock = Mock(return_value=True)
    app = _make_app(input_router=AgentInputRouter(submit_tool_input=submit_mock))

    async with app.run_test() as pilot:
        app._busy = True
        await app.request_tool_input(
            tool_call_id="call-1",
            request_id="req-1",
            prompt="Name: ",
        )

        input_widget = app.query_one("#chat-input", Input)
        assert not input_widget.disabled
        assert "Tool input" in input_widget.placeholder

        await pilot.click("#chat-input")
        await pilot.press("A", "l", "i", "c", "e", "enter")
        await pilot.pause(0.05)

        submit_mock.assert_called_once_with("req-1", "Alice")
        assert not app._input_router.has_pending_tool_requests()
        assert input_widget.disabled


@pytest.mark.asyncio
async def test_chat_command_bypasses_pending_tool_input() -> None:
    """/chat should bypass pending tool-input routing."""
    submit_mock = Mock(return_value=True)
    app = _make_app(input_router=AgentInputRouter(submit_tool_input=submit_mock))

    async with app.run_test() as pilot:
        app._busy = True
        await app.request_tool_input(
            tool_call_id="call-2",
            request_id="req-2",
            prompt="Age: ",
        )

        await pilot.click("#chat-input")
        await pilot.press("/", "c", "h", "a", "t", "space", "h", "i", "enter")
        await pilot.pause(0.05)

        submit_mock.assert_not_called()
        assert app._input_router.has_pending_tool_requests()
