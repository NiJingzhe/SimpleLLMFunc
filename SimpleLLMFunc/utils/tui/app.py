"""Textual app implementation for SimpleLLMFunc chat TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional, Sequence

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import HorizontalScroll, VerticalGroup, VerticalScroll
from textual.widgets import Input, Static

from SimpleLLMFunc.hooks.input_stream import AgentInputRouter, UserInputEvent
from SimpleLLMFunc.hooks.stream import ReactOutput
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.utils.tui.core import consume_react_stream
from SimpleLLMFunc.utils.tui.formatters import format_tool_arguments_markdown
from SimpleLLMFunc.utils.tui.hooks import ToolCustomEventHook


@dataclass
class _ModelWidgets:
    root: VerticalGroup
    reasoning_widget: Static
    content_widget: Static
    tool_list_widget: VerticalGroup
    stats_widget: Static
    content: str = ""
    reasoning: str = ""


@dataclass
class _ToolWidgets:
    root: VerticalGroup
    model_call_id: str
    tool_name: str
    arguments_widget: Static
    status_widget: Static
    output_widget: Static
    result_widget: Static
    stats_widget: Static
    arguments: dict[str, Any] = field(default_factory=dict)
    output: str = ""


@dataclass
class _AgentColumnWidgets:
    root: VerticalScroll
    header_widget: Static
    message_list_widget: VerticalGroup


class AgentTUIApp(App[None]):
    """Terminal chat UI for llm_chat event streams."""

    DEFAULT_INPUT_PLACEHOLDER = "Type a message and press Enter"
    MAIN_AGENT_KEY = "main"

    CSS = """
    Screen {
        layout: vertical;
        background: #0f1115;
    }

    #chat-log {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
    }

    #chat-input {
        dock: bottom;
        margin: 0 1 1 1;
        border: tall #6f87a8;
    }

    .bubble {
        height: auto;
        margin: 0 0 1 0;
        padding: 1;
        border: round #3a4252;
        background: #151922;
    }

    .tool-list {
        height: auto;
    }

    .user-bubble {
        border: round #5f8d5a;
        background: #152019;
    }

    .model-bubble {
        border: round #5978a1;
        background: #151d2a;
    }

    .agent-board {
        layout: horizontal;
        width: 100%;
        height: 1fr;
        margin: 0 0 1 0;
        overflow-x: auto;
        overflow-y: hidden;
    }

    .agent-column {
        width: auto;
        min-width: 33%;
        height: 1fr;
        margin: 0;
        overflow-y: auto;
    }

    #agent-column-main {
        margin: 0;
    }

    .agent-column-header {
        margin: 0 0 1 0;
        padding: 0 1;
        border: round #3a4252;
        background: #121722;
        color: #a8b8d0;
        text-style: bold;
    }

    .agent-column-messages {
        height: auto;
    }

    .role {
        text-style: bold;
        color: #a8b8d0;
        margin: 0 0 1 0;
    }

    .reasoning {
        color: #7f8798;
        margin: 0 0 1 0;
    }

    .stats {
        color: #7f8798;
        margin: 1 0 0 0;
    }

    .tool-call {
        height: auto;
        margin: 1 0 0 0;
        padding: 1;
        border: round #8f7a4d;
        background: #221f16;
    }

    .tool-status {
        color: #d4c9a2;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        agent_func: Callable[..., AsyncGenerator[ReactOutput, None]],
        input_param: str,
        history_param: Optional[str],
        static_kwargs: dict[str, Any],
        custom_hooks: Optional[Sequence[ToolCustomEventHook]] = None,
        title_text: str = "SimpleLLMFunc TUI",
        initial_history: Optional[MessageList] = None,
        input_router: Optional[AgentInputRouter] = None,
    ):
        super().__init__()
        self.title = title_text
        self.agent_func = agent_func
        self.input_param = input_param
        self.history_param = history_param
        self.static_kwargs = static_kwargs
        self.custom_hooks = list(custom_hooks or [])

        self.history: MessageList = list(initial_history or [])
        self._busy = False

        if input_router is not None:
            self._input_router = input_router
        else:
            from SimpleLLMFunc.builtin import PyRepl

            self._input_router = AgentInputRouter(submit_tool_input=PyRepl.submit_input)

        self._models: dict[str, _ModelWidgets] = {}
        self._tools: dict[str, _ToolWidgets] = {}
        self._agent_board: Optional[HorizontalScroll] = None
        self._agent_columns: dict[str, _AgentColumnWidgets] = {}
        self._model_agent_key: dict[str, str] = {}
        self._scroll_after_refresh_pending = False
        self._agent_scroll_pending: set[str] = set()
        self._agent_layout_pending = False

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        yield Input(placeholder=self.DEFAULT_INPUT_PLACEHOLDER, id="chat-input")

    async def on_mount(self) -> None:
        await self._append_system_hint(
            "TUI ready. Type a message to start. Use /chat <message> to bypass pending tool input; use /exit, /quit, /q, or Ctrl+Q to quit."
        )
        self.query_one("#chat-input", Input).focus()

    def _scroll_chat_log_to_end(self) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.scroll_end(animate=False, force=True, immediate=True)

    def _auto_scroll_chat_log(self) -> None:
        self._scroll_chat_log_to_end()

        if self._scroll_after_refresh_pending:
            return

        self._scroll_after_refresh_pending = True

        def _after_refresh_scroll() -> None:
            self._scroll_after_refresh_pending = False
            self._scroll_chat_log_to_end()

        self.call_after_refresh(_after_refresh_scroll)

    def _scroll_agent_column_to_end(self, agent_key: str) -> None:
        column = self._agent_columns.get(agent_key)
        if column is None:
            return
        column.root.scroll_end(animate=False, force=True, immediate=True)

    def _auto_scroll_agent_column(self, agent_key: str) -> None:
        self._scroll_agent_column_to_end(agent_key)

        if agent_key in self._agent_scroll_pending:
            return

        self._agent_scroll_pending.add(agent_key)

        def _after_refresh_scroll() -> None:
            self._agent_scroll_pending.discard(agent_key)
            self._scroll_agent_column_to_end(agent_key)

        self.call_after_refresh(_after_refresh_scroll)

    def _auto_scroll_for_model(self, model_call_id: str) -> None:
        agent_key = self._model_agent_key.get(
            model_call_id,
            self._resolve_agent_key(model_call_id),
        )
        self._auto_scroll_agent_column(agent_key)

    def _auto_scroll_for_tool(self, tool_call_id: str) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return
        self._auto_scroll_for_model(tool.model_call_id)

    def _schedule_agent_column_layout(self) -> None:
        if self._agent_layout_pending:
            return

        self._agent_layout_pending = True

        def _after_refresh_layout() -> None:
            self._agent_layout_pending = False
            self._reflow_agent_column_widths()

        self.call_after_refresh(_after_refresh_layout)

    def _reflow_agent_column_widths(self) -> None:
        if not self._agent_columns:
            return

        column_count = len(self._agent_columns)

        if column_count <= 1:
            target_width = "100%"
        elif column_count == 2:
            target_width = "50%"
        else:
            target_width = "33%"

        for column in self._agent_columns.values():
            column.root.styles.min_width = "33%"
            column.root.styles.width = target_width

    def _set_chat_placeholder(self) -> None:
        input_widget = self.query_one("#chat-input", Input)
        input_widget.placeholder = self.DEFAULT_INPUT_PLACEHOLDER

    def _set_tool_input_placeholder(self, prompt: str) -> None:
        input_widget = self.query_one("#chat-input", Input)
        prompt_text = prompt.strip()
        if prompt_text:
            input_widget.placeholder = f"Tool input: {prompt_text}"
        else:
            input_widget.placeholder = (
                "Tool input required; type response and press Enter"
            )

    def _refresh_input_widget_state(self) -> None:
        input_widget = self.query_one("#chat-input", Input)
        pending_request = self._input_router.peek_pending_tool_request()

        if pending_request is not None:
            input_widget.disabled = False
            self._set_tool_input_placeholder(pending_request.prompt)
            input_widget.focus()
            return

        self._set_chat_placeholder()
        if self._busy:
            input_widget.disabled = True
        else:
            input_widget.disabled = False
            input_widget.focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        input_widget = self.query_one("#chat-input", Input)
        stripped_value = event.value.strip()

        if stripped_value.lower() in {"/exit", "/quit", "/q"}:
            self.exit()
            return

        force_chat = False
        input_payload = event.value
        if stripped_value.lower().startswith("/chat"):
            force_chat = True
            input_payload = stripped_value[5:].lstrip()

        route_result = self._input_router.route_input(
            UserInputEvent(
                text=input_payload,
                force_chat=force_chat,
            )
        )

        if route_result.route == "tool":
            if route_result.request is not None:
                await self.append_tool_output(
                    route_result.request.tool_call_id,
                    f"> {route_result.submitted_text}\n",
                )
                await self.set_tool_status(route_result.request.tool_call_id, "running")

            input_widget.value = ""
            self._refresh_input_widget_state()
            return

        if route_result.route == "rejected":
            await self._append_system_hint(
                route_result.reason or "Tool input request expired."
            )
            input_widget.value = ""
            self._refresh_input_widget_state()
            return

        if route_result.route == "noop":
            return

        user_text = route_result.chat_text

        if self._busy:
            await self._append_system_hint("Agent is still responding, please wait.")
            return

        input_widget.value = ""

        await self._append_user_message(user_text)
        self._busy = True
        input_widget.disabled = True

        self.run_worker(self._run_turn(user_text), thread=False)

    async def _run_turn(self, user_text: str) -> None:
        input_widget = self.query_one("#chat-input", Input)
        try:
            call_kwargs = dict(self.static_kwargs)
            call_kwargs[self.input_param] = user_text
            if self.history_param:
                call_kwargs[self.history_param] = self.history

            stream = self.agent_func(**call_kwargs)
            new_history = await consume_react_stream(
                stream,
                adapter=self,
                custom_hooks=self.custom_hooks,
            )
            if new_history:
                self.history = new_history
        except Exception as exc:
            await self._append_system_hint(f"Agent error: {exc}")
        finally:
            self._busy = False
            self._input_router.clear_all_requests()
            self._refresh_input_widget_state()

    async def _append_user_message(self, text: str) -> None:
        bubble = VerticalGroup(classes="bubble user-bubble")
        role = Static("User", classes="role")
        body = Static(classes="body")
        body.update(Markdown(text))

        await self._mount_into_agent_column(self.MAIN_AGENT_KEY, bubble)
        await bubble.mount(role)
        await bubble.mount(body)
        self._auto_scroll_agent_column(self.MAIN_AGENT_KEY)

    async def _append_system_hint(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        hint = Static(Text(text, style="#7f8798"))
        await chat_log.mount(hint)
        self._auto_scroll_chat_log()

    def _is_fork_model_call_id(self, model_call_id: str) -> bool:
        return model_call_id.startswith("fork::")

    def _extract_fork_id(self, model_call_id: str) -> str:
        if not self._is_fork_model_call_id(model_call_id):
            return model_call_id
        return model_call_id.split("::", 1)[1]

    def _resolve_agent_key(self, model_call_id: str) -> str:
        if self._is_fork_model_call_id(model_call_id):
            return self._extract_fork_id(model_call_id)
        return self.MAIN_AGENT_KEY

    def _normalize_dom_id_suffix(self, value: str) -> str:
        normalized_chars: list[str] = []
        for char in value:
            if char.isalnum() or char == "_":
                normalized_chars.append(char)
            else:
                normalized_chars.append("_")

        normalized = "".join(normalized_chars).strip("_")
        if normalized:
            return normalized
        return "agent"

    def _build_agent_column_dom_id(self, agent_key: str) -> str:
        if agent_key == self.MAIN_AGENT_KEY:
            return "agent-column-main"
        return f"agent-column-fork_{self._normalize_dom_id_suffix(agent_key)}"

    def _build_agent_column_label(self, agent_key: str) -> str:
        if agent_key == self.MAIN_AGENT_KEY:
            return "Main Agent"
        return f"Fork {agent_key}"

    async def _ensure_agent_board(self) -> HorizontalScroll:
        if self._agent_board is not None:
            return self._agent_board

        chat_log = self.query_one("#chat-log", VerticalScroll)
        board = HorizontalScroll(classes="agent-board", id="agent-board")
        await chat_log.mount(board)
        self._agent_board = board
        self._schedule_agent_column_layout()
        return board

    async def _ensure_agent_column(self, agent_key: str) -> _AgentColumnWidgets:
        existing = self._agent_columns.get(agent_key)
        if existing is not None:
            return existing

        if (
            agent_key != self.MAIN_AGENT_KEY
            and self.MAIN_AGENT_KEY not in self._agent_columns
        ):
            await self._ensure_agent_column(self.MAIN_AGENT_KEY)

        board = await self._ensure_agent_board()
        column = VerticalScroll(
            classes="agent-column",
            id=self._build_agent_column_dom_id(agent_key),
        )
        header = Static(
            self._build_agent_column_label(agent_key),
            classes="agent-column-header",
        )
        message_list = VerticalGroup(classes="agent-column-messages")

        await board.mount(column)
        await column.mount(header)
        await column.mount(message_list)

        created = _AgentColumnWidgets(
            root=column,
            header_widget=header,
            message_list_widget=message_list,
        )
        self._agent_columns[agent_key] = created
        self._schedule_agent_column_layout()
        return created

    async def _mount_into_agent_column(
        self, agent_key: str, widget: VerticalGroup
    ) -> None:
        column = await self._ensure_agent_column(agent_key)
        await column.message_list_widget.mount(widget)

    async def _teardown_agent_column(self, agent_key: str) -> None:
        if agent_key == self.MAIN_AGENT_KEY:
            return

        self._agent_scroll_pending.discard(agent_key)

        column = self._agent_columns.pop(agent_key, None)
        if column is None:
            return

        removed_model_call_ids = [
            model_call_id
            for model_call_id, mapped_agent_key in self._model_agent_key.items()
            if mapped_agent_key == agent_key
        ]
        for model_call_id in removed_model_call_ids:
            self._model_agent_key.pop(model_call_id, None)
            self._models.pop(model_call_id, None)

        removed_tool_call_ids = [
            tool_call_id
            for tool_call_id, tool in self._tools.items()
            if tool.model_call_id in removed_model_call_ids
        ]
        for tool_call_id in removed_tool_call_ids:
            self._tools.pop(tool_call_id, None)
            self._input_router.clear_tool_requests(tool_call_id)

        await column.root.remove()
        self._schedule_agent_column_layout()

    async def _mount_model_root(
        self,
        model_call_id: str,
        root: VerticalGroup,
    ) -> None:
        agent_key = self._resolve_agent_key(model_call_id)
        self._model_agent_key[model_call_id] = agent_key
        await self._mount_into_agent_column(agent_key, root)

    # ------------------------------------------------------------------
    # Adapter methods used by consume_react_stream
    # ------------------------------------------------------------------

    async def start_model_response(self, model_call_id: str) -> None:
        root = VerticalGroup(classes="bubble model-bubble")
        role = Static("Assistant", classes="role")
        reasoning_widget = Static("", classes="reasoning")
        content_widget = Static("", classes="body")
        tool_list_widget = VerticalGroup(classes="tool-list")
        stats_widget = Static("", classes="stats")

        await self._mount_model_root(model_call_id, root)
        await root.mount(role)
        await root.mount(reasoning_widget)
        await root.mount(content_widget)
        await root.mount(tool_list_widget)
        await root.mount(stats_widget)

        self._models[model_call_id] = _ModelWidgets(
            root=root,
            reasoning_widget=reasoning_widget,
            content_widget=content_widget,
            tool_list_widget=tool_list_widget,
            stats_widget=stats_widget,
        )
        self._auto_scroll_for_model(model_call_id)

    async def append_model_content(
        self, model_call_id: str, content_delta: str
    ) -> None:
        model = self._models.get(model_call_id)
        if model is None:
            return

        model.content += content_delta
        model.content_widget.update(Markdown(model.content or " "))
        self._auto_scroll_for_model(model_call_id)

    async def append_model_reasoning(
        self,
        model_call_id: str,
        reasoning_delta: str,
    ) -> None:
        model = self._models.get(model_call_id)
        if model is None:
            return

        model.reasoning += reasoning_delta
        model.reasoning_widget.update(model.reasoning)
        self._auto_scroll_for_model(model_call_id)

    async def finish_model_response(self, model_call_id: str, stats_line: str) -> None:
        model = self._models.get(model_call_id)
        if model is None:
            return

        model.stats_widget.update(stats_line)

        if self._is_fork_model_call_id(model_call_id) and stats_line.startswith(
            "fork |"
        ):
            agent_key = self._model_agent_key.get(model_call_id)
            if agent_key is not None:
                await self._teardown_agent_column(agent_key)

        self._auto_scroll_for_model(model_call_id)

    async def start_tool_call(
        self,
        model_call_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        existing_tool = self._tools.get(tool_call_id)
        if existing_tool is not None:
            existing_tool.tool_name = tool_name
            existing_tool.arguments = dict(arguments)
            existing_tool.arguments_widget.update(
                Markdown(
                    format_tool_arguments_markdown(
                        existing_tool.arguments,
                        tool_name=tool_name,
                    )
                )
            )
            existing_tool.status_widget.update("running")
            self._auto_scroll_for_tool(tool_call_id)
            return

        model = self._models.get(model_call_id)
        if model is None:
            return

        root = VerticalGroup(classes="tool-call")
        title = Static(f"Tool: {tool_name}", classes="role")
        args_widget = Static()
        status_widget = Static("running", classes="tool-status")
        output_widget = Static()
        result_widget = Static()
        stats_widget = Static("", classes="stats")

        args_widget.update(
            Markdown(
                format_tool_arguments_markdown(
                    arguments,
                    tool_name=tool_name,
                )
            )
        )

        await model.tool_list_widget.mount(root)
        await root.mount(title)
        await root.mount(args_widget)
        await root.mount(status_widget)
        await root.mount(output_widget)
        await root.mount(result_widget)
        await root.mount(stats_widget)

        self._tools[tool_call_id] = _ToolWidgets(
            root=root,
            model_call_id=model_call_id,
            tool_name=tool_name,
            arguments_widget=args_widget,
            status_widget=status_widget,
            output_widget=output_widget,
            result_widget=result_widget,
            stats_widget=stats_widget,
            arguments=dict(arguments),
        )
        self._auto_scroll_for_tool(tool_call_id)

    async def append_tool_argument(
        self,
        tool_call_id: str,
        argname: str,
        argcontent_delta: str,
    ) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return

        previous = tool.arguments.get(argname, "")
        if not isinstance(previous, str):
            previous = str(previous)
        tool.arguments[argname] = previous + argcontent_delta
        tool.arguments_widget.update(
            Markdown(
                format_tool_arguments_markdown(
                    tool.arguments,
                    tool_name=tool.tool_name,
                )
            )
        )
        self._auto_scroll_for_tool(tool_call_id)

    async def append_tool_output(self, tool_call_id: str, output_delta: str) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return

        tool.output += output_delta
        if tool.output:
            tool.output_widget.update(Markdown(f"```text\n{tool.output}\n```"))
            self._auto_scroll_for_tool(tool_call_id)

    async def set_tool_status(self, tool_call_id: str, status: str) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return

        tool.status_widget.update(status)
        self._auto_scroll_for_tool(tool_call_id)

    async def request_tool_input(
        self,
        tool_call_id: str,
        request_id: str,
        prompt: str,
    ) -> None:
        self._input_router.register_tool_request(
            tool_call_id=tool_call_id,
            request_id=request_id,
            prompt=prompt,
        )
        self._refresh_input_widget_state()

    async def clear_tool_input(self, tool_call_id: str) -> None:
        self._input_router.clear_tool_requests(tool_call_id)
        self._refresh_input_widget_state()

    async def finish_tool_call(
        self,
        tool_call_id: str,
        result_markdown: str,
        stats_line: str,
        success: bool,
    ) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return

        tool.status_widget.update("success" if success else "error")
        tool.result_widget.update(Markdown(result_markdown or ""))
        tool.stats_widget.update(stats_line)
        self._auto_scroll_for_tool(tool_call_id)


__all__ = ["AgentTUIApp"]
