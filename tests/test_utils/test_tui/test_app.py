"""Regression tests for Textual TUI widget mounting."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Static

from SimpleLLMFunc.hooks.stream import ReactOutput
from SimpleLLMFunc.utils.tui.app import AgentTUIApp


async def _unused_agent(_message: str) -> AsyncGenerator[ReactOutput, None]:
    if False:
        yield None


def _make_app() -> AgentTUIApp:
    return AgentTUIApp(
        agent_func=_unused_agent,
        input_param="message",
        history_param=None,
        static_kwargs={},
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
