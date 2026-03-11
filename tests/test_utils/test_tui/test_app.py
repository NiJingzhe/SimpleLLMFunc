"""Regression tests for Textual TUI widget mounting."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import Mock

import pytest
from textual.css.query import NoMatches
from textual.containers import HorizontalScroll, VerticalScroll
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
    """Model streaming updates should keep model column pinned to latest content."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.append_model_content("llm_call_1", "word " * 2200)
        await pilot.pause(0.05)

        main_column = app.query_one("#agent-column-main", VerticalScroll)
        assert main_column.max_scroll_y > 0

        main_column.scroll_home(animate=False, immediate=True)
        await pilot.pause(0.02)
        assert main_column.scroll_y == 0

        await app.append_model_content("llm_call_1", "tail " * 200)
        await pilot.pause(0.05)

        assert main_column.scroll_y >= main_column.max_scroll_y - 0.5


@pytest.mark.asyncio
async def test_streaming_tool_output_auto_scrolls_to_bottom() -> None:
    """Tool streaming output should keep owning agent column pinned."""
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

        main_column = app.query_one("#agent-column-main", VerticalScroll)
        assert main_column.max_scroll_y > 0

        main_column.scroll_home(animate=False, immediate=True)
        await pilot.pause(0.02)
        assert main_column.scroll_y == 0

        await app.append_tool_output("call-1", "tail line\n")
        await pilot.pause(0.05)

        assert main_column.scroll_y >= main_column.max_scroll_y - 0.5


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


@pytest.mark.asyncio
async def test_main_stream_uses_single_agent_column() -> None:
    """Without forks, the message board should stay in single-column mode."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")

        board = app.query_one("#agent-board")
        columns = list(board.query(".agent-column"))

        assert len(columns) == 1
        app.query_one("#agent-column-main", VerticalScroll)


@pytest.mark.asyncio
async def test_fork_stream_adds_peer_column_and_unloads_on_finish() -> None:
    """Fork agent should get its own column and unload after completion."""
    app = _make_app()

    async with app.run_test() as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")

        board = app.query_one("#agent-board")
        columns = list(board.query(".agent-column"))
        assert len(columns) == 2
        app.query_one("#agent-column-main", VerticalScroll)
        app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        await app.finish_model_response("fork::fork_1", "fork | completed")
        await pilot.pause(0.05)

        with pytest.raises(NoMatches):
            app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        board = app.query_one("#agent-board")
        columns = list(board.query(".agent-column"))

        assert len(columns) == 1
        app.query_one("#agent-column-main", VerticalScroll)


@pytest.mark.asyncio
async def test_fork_error_column_stays_visible_for_inspection() -> None:
    """Errored fork column should remain visible instead of unloading immediately."""
    app = _make_app()

    async with app.run_test() as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.append_model_reasoning("fork::fork_1", "TypeError: missing argument")
        await app.finish_model_response("fork::fork_1", "fork | error")
        await pilot.pause(0.05)

        board = app.query_one("#agent-board")
        columns = list(board.query(".agent-column"))
        assert len(columns) == 2
        app.query_one("#agent-column-fork_fork_1", VerticalScroll)


@pytest.mark.asyncio
async def test_agent_board_scrolls_horizontally_when_more_than_three_columns() -> None:
    """Agent board should enable horizontal scrolling when columns exceed width."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.start_model_response("fork::fork_2")
        await app.start_model_response("fork::fork_3")
        await pilot.pause(0.05)

        board = app.query_one("#agent-board", HorizontalScroll)
        main_column = app.query_one("#agent-column-main", VerticalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        one_third = max(1, board.size.width // 3)
        assert main_column.region.width >= one_third - 1
        assert fork_column.region.width >= one_third - 1
        assert main_column.region.width <= one_third + 1
        assert fork_column.region.width <= one_third + 1
        assert main_column.region.width < board.size.width
        assert board.max_scroll_x > 0


@pytest.mark.asyncio
async def test_two_agent_columns_use_half_width_without_horizontal_scroll() -> None:
    """Two columns should split viewport in half with no horizontal overflow."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await pilot.pause(0.05)

        board = app.query_one("#agent-board", HorizontalScroll)
        main_column = app.query_one("#agent-column-main", VerticalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        one_half = max(1, board.size.width // 2)
        assert abs(main_column.region.width - one_half) <= 1
        assert abs(fork_column.region.width - one_half) <= 1
        assert board.max_scroll_x == 0


@pytest.mark.asyncio
async def test_three_agent_columns_fit_without_horizontal_scroll() -> None:
    """Exactly three columns should fit viewport without horizontal scroll."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.start_model_response("fork::fork_2")
        await pilot.pause(0.05)

        board = app.query_one("#agent-board", HorizontalScroll)
        main_column = app.query_one("#agent-column-main", VerticalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        one_third = max(1, board.size.width // 3)
        assert abs(main_column.region.width - one_third) <= 1
        assert abs(fork_column.region.width - one_third) <= 1
        assert board.max_scroll_x == 0


@pytest.mark.asyncio
async def test_fork_streaming_remains_visible_with_horizontal_scroll_layout() -> None:
    """Fork streaming should still render in >3-column horizontal layout."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.start_model_response("fork::fork_2")
        await app.start_model_response("fork::fork_3")
        await app.append_model_content("fork::fork_3", "streaming token")
        await pilot.pause(0.05)

        board = app.query_one("#agent-board", HorizontalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_3", VerticalScroll)

        one_third = max(1, board.size.width // 3)
        assert fork_column.region.width >= one_third - 1
        assert fork_column.region.width <= one_third + 1
        assert app._models["fork::fork_3"].content == "streaming token"


@pytest.mark.asyncio
async def test_same_fork_model_call_id_creates_new_assistant_bubble() -> None:
    """Repeated fork model start should append a new assistant bubble."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("fork::fork_1")
        await app.append_model_content("fork::fork_1", "first")

        await app.start_model_response("fork::fork_1")
        await app.append_model_content("fork::fork_1", "second")

        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)
        bubbles = list(fork_column.query(".model-bubble"))

        assert len(bubbles) == 2


@pytest.mark.asyncio
async def test_agent_columns_keep_independent_scroll_positions() -> None:
    """Appending to one column should not move sibling column scroll."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.append_model_content("llm_call_1", "main " * 2000)
        await app.append_model_content("fork::fork_1", "fork " * 2000)
        await pilot.pause(0.05)

        main_column = app.query_one("#agent-column-main", VerticalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)
        assert main_column.max_scroll_y > 0
        assert fork_column.max_scroll_y > 0

        main_column.scroll_home(animate=False, immediate=True)
        fork_column.scroll_home(animate=False, immediate=True)
        await pilot.pause(0.02)

        await app.append_model_content("llm_call_1", "tail " * 300)
        await pilot.pause(0.05)

        assert main_column.scroll_y >= main_column.max_scroll_y - 0.5
        assert fork_column.scroll_y == 0


@pytest.mark.asyncio
async def test_copy_all_text_contains_user_model_and_tool_content() -> None:
    """Copy-all action should include key rendered transcript sections."""
    app = _make_app()
    copied: list[str] = []

    def _copy_to_clipboard(text: str) -> None:
        copied.append(text)

    app.copy_to_clipboard = _copy_to_clipboard  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await app._append_user_message("hello from user")
        await app.start_model_response("llm_call_1")
        await app.append_model_content("llm_call_1", "assistant says hi")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-1",
            tool_name="execute_code",
            arguments={"code": "print('ok')"},
        )
        await app.append_tool_output("call-1", "ok\n")
        await app.finish_tool_call(
            tool_call_id="call-1",
            result_markdown="done",
            stats_line="Tool | 0.01s | success",
            success=True,
        )
        await app.finish_model_response("llm_call_1", "model | 0.02s")
        await pilot.pause(0.05)

        await app.action_copy_all_text()

    assert copied
    transcript = copied[-1]
    assert "hello from user" in transcript
    assert "assistant says hi" in transcript
    assert "Tool: execute_code" in transcript
    assert "ok" in transcript


@pytest.mark.asyncio
async def test_copy_command_routes_to_clipboard_action() -> None:
    """/copy command should trigger copy-all without sending chat input."""
    copy_mock = Mock()
    app = _make_app()
    app.copy_to_clipboard = copy_mock  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await app._append_user_message("copy me")
        await pilot.click("#chat-input")
        await pilot.press("/", "c", "o", "p", "y", "enter")
        await pilot.pause(0.05)

    copy_mock.assert_called_once()


@pytest.mark.asyncio
async def test_virtualization_archives_old_blocks_but_copy_keeps_full_text() -> None:
    """Old off-screen bubbles should archive while transcript copy remains complete."""
    app = _make_app()
    app._virtual_keep_live_blocks = 6
    copied: list[str] = []
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]

    async with app.run_test(size=(100, 28)) as pilot:
        for idx in range(18):
            await app._append_user_message(f"msg-{idx}")
        await pilot.pause(0.05)

        assert app._archive_count > 0
        assert len(list(app.query(".archived-block"))) > 0

        await app.action_copy_all_text()

    assert copied
    transcript = copied[-1]
    assert "msg-0" in transcript
    assert "msg-17" in transcript
