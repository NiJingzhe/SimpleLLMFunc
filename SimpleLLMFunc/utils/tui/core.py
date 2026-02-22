"""Core stream-consumption logic for the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, Protocol, Sequence

from SimpleLLMFunc.hooks.events import (
    CustomEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    ReactEndEvent,
    ReActEvent,
    ToolCallEndEvent,
    ToolCallErrorEvent,
    ToolCallStartEvent,
)
from SimpleLLMFunc.hooks.stream import ReactOutput, is_event_yield
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.utils.tui.formatters import (
    extract_reasoning_delta,
    extract_stream_text,
    extract_text_from_response,
    format_custom_event_fallback,
    format_model_stats,
    format_tool_result_markdown,
    format_tool_stats,
)
from SimpleLLMFunc.utils.tui.hooks import (
    ToolCustomEventHook,
    ToolRenderSnapshot,
    apply_tool_event_hooks,
)


class TUIStreamAdapter(Protocol):
    """Adapter protocol for rendering stream events."""

    async def start_model_response(self, model_call_id: str) -> None: ...

    async def append_model_content(
        self, model_call_id: str, content_delta: str
    ) -> None: ...

    async def append_model_reasoning(
        self,
        model_call_id: str,
        reasoning_delta: str,
    ) -> None: ...

    async def finish_model_response(
        self, model_call_id: str, stats_line: str
    ) -> None: ...

    async def start_tool_call(
        self,
        model_call_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
    ) -> None: ...

    async def append_tool_output(
        self, tool_call_id: str, output_delta: str
    ) -> None: ...

    async def set_tool_status(self, tool_call_id: str, status: str) -> None: ...

    async def finish_tool_call(
        self,
        tool_call_id: str,
        result_markdown: str,
        stats_line: str,
        success: bool,
    ) -> None: ...


@dataclass
class _StreamConsumeState:
    """Mutable internal state while consuming one agent turn."""

    llm_call_count: int = 0
    active_model_call_id: Optional[str] = None
    model_has_chunk_text: dict[str, bool] = field(default_factory=dict)
    tool_snapshots: dict[str, ToolRenderSnapshot] = field(default_factory=dict)
    running_tool_call_ids: list[str] = field(default_factory=list)
    final_history: MessageList = field(default_factory=list)


async def _ensure_active_model_call(
    adapter: TUIStreamAdapter,
    state: _StreamConsumeState,
) -> str:
    if state.active_model_call_id:
        return state.active_model_call_id

    state.llm_call_count += 1
    model_call_id = f"llm_call_{state.llm_call_count}"
    state.active_model_call_id = model_call_id
    state.model_has_chunk_text[model_call_id] = False
    await adapter.start_model_response(model_call_id)
    return model_call_id


async def _handle_event(
    event: ReActEvent,
    adapter: TUIStreamAdapter,
    state: _StreamConsumeState,
    custom_hooks: Sequence[ToolCustomEventHook],
) -> None:
    if isinstance(event, LLMCallStartEvent):
        state.llm_call_count += 1
        state.active_model_call_id = f"llm_call_{state.llm_call_count}"
        state.model_has_chunk_text[state.active_model_call_id] = False
        await adapter.start_model_response(state.active_model_call_id)
        return

    if isinstance(event, LLMChunkArriveEvent):
        model_call_id = await _ensure_active_model_call(adapter, state)

        content_delta = extract_stream_text(event.chunk)
        if content_delta:
            state.model_has_chunk_text[model_call_id] = True
            await adapter.append_model_content(model_call_id, content_delta)

        reasoning_delta = extract_reasoning_delta(event.chunk)
        if reasoning_delta:
            await adapter.append_model_reasoning(model_call_id, reasoning_delta)
        return

    if isinstance(event, LLMCallEndEvent):
        model_call_id = await _ensure_active_model_call(adapter, state)

        if not state.model_has_chunk_text.get(model_call_id, False):
            text = extract_text_from_response(event.response)
            if text:
                await adapter.append_model_content(model_call_id, text)

        stats_line = format_model_stats(event.execution_time, event.usage)
        await adapter.finish_model_response(model_call_id, stats_line)
        return

    if isinstance(event, ToolCallStartEvent):
        model_call_id = await _ensure_active_model_call(adapter, state)
        await adapter.start_tool_call(
            model_call_id=model_call_id,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            arguments=event.arguments,
        )
        state.tool_snapshots[event.tool_call_id] = ToolRenderSnapshot(
            tool_name=event.tool_name,
            tool_call_id=event.tool_call_id,
            arguments=event.arguments,
        )
        state.running_tool_call_ids.append(event.tool_call_id)
        return

    if isinstance(event, CustomEvent):
        tool_call_id = event.tool_call_id
        if not tool_call_id and state.running_tool_call_ids:
            tool_call_id = state.running_tool_call_ids[-1]

        if not tool_call_id:
            return

        snapshot = state.tool_snapshots.get(tool_call_id)
        if snapshot is None:
            return

        update = apply_tool_event_hooks(
            event=event,
            snapshot=snapshot,
            custom_hooks=custom_hooks,
        )

        if update is None:
            update_text = format_custom_event_fallback(event.event_name, event.data)
            await adapter.append_tool_output(tool_call_id, update_text)
            snapshot.output += update_text
            return

        if update.replace_output is not None:
            snapshot.output = update.replace_output
            await adapter.append_tool_output(
                tool_call_id,
                update.replace_output,
            )
        elif update.append_output:
            snapshot.output += update.append_output
            await adapter.append_tool_output(tool_call_id, update.append_output)

        if update.status:
            snapshot.status = update.status
            await adapter.set_tool_status(tool_call_id, update.status)
        return

    if isinstance(event, ToolCallEndEvent):
        result_markdown = format_tool_result_markdown(event.result)
        stats_line = format_tool_stats(event.execution_time, event.success)
        await adapter.finish_tool_call(
            tool_call_id=event.tool_call_id,
            result_markdown=result_markdown,
            stats_line=stats_line,
            success=event.success,
        )
        if event.tool_call_id in state.running_tool_call_ids:
            state.running_tool_call_ids.remove(event.tool_call_id)
        snapshot = state.tool_snapshots.get(event.tool_call_id)
        if snapshot is not None:
            snapshot.status = "success" if event.success else "error"
        return

    if isinstance(event, ToolCallErrorEvent):
        result_markdown = format_tool_result_markdown(
            {
                "error": event.error_message,
                "error_type": event.error_type,
            }
        )
        stats_line = format_tool_stats(event.execution_time, False)
        await adapter.finish_tool_call(
            tool_call_id=event.tool_call_id,
            result_markdown=result_markdown,
            stats_line=stats_line,
            success=False,
        )
        if event.tool_call_id in state.running_tool_call_ids:
            state.running_tool_call_ids.remove(event.tool_call_id)
        snapshot = state.tool_snapshots.get(event.tool_call_id)
        if snapshot is not None:
            snapshot.status = "error"
        return

    if isinstance(event, ReactEndEvent):
        state.final_history = event.final_messages


async def consume_react_stream(
    stream: AsyncGenerator[ReactOutput, None],
    adapter: TUIStreamAdapter,
    custom_hooks: Optional[Sequence[ToolCustomEventHook]] = None,
) -> MessageList:
    """Consume one react event stream and update the adapter in real-time."""

    hooks = list(custom_hooks or [])
    state = _StreamConsumeState()
    saw_event = False

    async for output in stream:
        if not is_event_yield(output):
            continue

        saw_event = True

        await _handle_event(
            event=output.event,
            adapter=adapter,
            state=state,
            custom_hooks=hooks,
        )

    if not saw_event:
        raise ValueError(
            "TUI requires llm_chat(enable_event=True). No event stream was received."
        )

    return state.final_history


__all__ = [
    "TUIStreamAdapter",
    "consume_react_stream",
]
