"""Tests for TUI stream consumer core logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function as OpenAIFunction,
)
from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.hooks.events import (
    CustomEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    ReactEndEvent,
    ReActEventType,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from SimpleLLMFunc.hooks.stream import EventYield, ReactOutput
from SimpleLLMFunc.utils.tui.core import consume_react_stream
from SimpleLLMFunc.utils.tui.hooks import ToolEventRenderUpdate, ToolRenderSnapshot


def _make_chunk(content: str) -> ChatCompletionChunk:
    delta = ChoiceDelta(content=content, role="assistant")
    choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
    return ChatCompletionChunk(
        id="chunk-id",
        choices=[choice],
        created=123,
        model="test-model",
        object="chat.completion.chunk",
    )


def _make_response(content: str) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(finish_reason="stop", index=0, message=message)
    return ChatCompletion(
        id="resp-id",
        choices=[choice],
        created=123,
        model="test-model",
        object="chat.completion",
    )


@dataclass
class FakeAdapter:
    """Fake adapter used to observe core events."""

    model_content: dict[str, str] = field(default_factory=dict)
    model_stats: dict[str, str] = field(default_factory=dict)
    tool_output: dict[str, str] = field(default_factory=dict)
    tool_stats: dict[str, str] = field(default_factory=dict)
    tool_status: dict[str, str] = field(default_factory=dict)
    tool_results: dict[str, str] = field(default_factory=dict)
    model_start_order: list[str] = field(default_factory=list)

    async def start_model_response(self, model_call_id: str) -> None:
        self.model_start_order.append(model_call_id)
        self.model_content[model_call_id] = ""

    async def append_model_content(
        self, model_call_id: str, content_delta: str
    ) -> None:
        self.model_content[model_call_id] += content_delta

    async def append_model_reasoning(
        self, model_call_id: str, reasoning_delta: str
    ) -> None:
        # Not required for this test case.
        pass

    async def finish_model_response(self, model_call_id: str, stats_line: str) -> None:
        self.model_stats[model_call_id] = stats_line

    async def start_tool_call(
        self,
        model_call_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        self.tool_output[tool_call_id] = ""
        self.tool_status[tool_call_id] = "running"

    async def append_tool_output(self, tool_call_id: str, output_delta: str) -> None:
        self.tool_output[tool_call_id] += output_delta

    async def set_tool_status(self, tool_call_id: str, status: str) -> None:
        self.tool_status[tool_call_id] = status

    async def finish_tool_call(
        self,
        tool_call_id: str,
        result_markdown: str,
        stats_line: str,
        success: bool,
    ) -> None:
        self.tool_results[tool_call_id] = result_markdown
        self.tool_stats[tool_call_id] = stats_line
        self.tool_status[tool_call_id] = "success" if success else "error"


@pytest.mark.asyncio
async def test_consume_react_stream_updates_model_and_tool_blocks() -> None:
    """Consumer should stream model text and custom tool output."""
    adapter = FakeAdapter()
    ts = datetime.now(timezone.utc)

    tool_call = ChatCompletionMessageToolCall(
        id="call-1",
        type="function",
        function=OpenAIFunction(name="execute_code", arguments='{"code": "print(1)"}'),
    )
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    final_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hello world"},
    ]

    async def _stream() -> AsyncGenerator[ReactOutput, None]:
        yield EventYield(
            event=LLMCallStartEvent(
                event_type=ReActEventType.LLM_CALL_START,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=0,
                messages=[],
                tools=None,
                llm_kwargs={},
                stream=True,
            )
        )
        yield EventYield(
            event=LLMChunkArriveEvent(
                event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=0,
                chunk=_make_chunk("Hello "),
                accumulated_content="Hello ",
                chunk_index=0,
            )
        )
        yield EventYield(
            event=LLMChunkArriveEvent(
                event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=0,
                chunk=_make_chunk("world"),
                accumulated_content="Hello world",
                chunk_index=1,
            )
        )
        yield EventYield(
            event=LLMCallEndEvent(
                event_type=ReActEventType.LLM_CALL_END,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=0,
                response=_make_response("Hello world"),
                messages=[],
                tool_calls=[tool_call],
                execution_time=1.2,
                usage=usage,
            )
        )
        yield EventYield(
            event=ToolCallStartEvent(
                event_type=ReActEventType.TOOL_CALL_START,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=1,
                tool_name="execute_code",
                tool_call_id="call-1",
                arguments={"code": "print(1)"},
                tool_call=tool_call,
            )
        )
        yield EventYield(
            event=CustomEvent(
                event_type=ReActEventType.CUSTOM_EVENT,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=1,
                event_name="kernel_stdout",
                data={"text": "1\n"},
                tool_name="execute_code",
                tool_call_id="call-1",
            )
        )
        yield EventYield(
            event=ToolCallEndEvent(
                event_type=ReActEventType.TOOL_CALL_END,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=1,
                tool_name="execute_code",
                tool_call_id="call-1",
                arguments={"code": "print(1)"},
                result={"stdout": "1\n", "success": True},
                execution_time=0.3,
                success=True,
            )
        )
        yield EventYield(
            event=ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=1,
                final_response="Hello world",
                final_messages=final_messages,
                total_iterations=1,
                total_execution_time=2.0,
                total_tool_calls=1,
                total_llm_calls=1,
                total_token_usage=usage,
            )
        )

    history = await consume_react_stream(_stream(), adapter=adapter)

    assert adapter.model_start_order == ["llm_call_1"]
    assert adapter.model_content["llm_call_1"] == "Hello world"
    assert "1.20s" in adapter.model_stats["llm_call_1"]
    assert adapter.tool_output["call-1"] == "1\n"
    assert "0.30s" in adapter.tool_stats["call-1"]
    assert adapter.tool_status["call-1"] == "success"
    assert history == final_messages


@pytest.mark.asyncio
async def test_consume_react_stream_custom_hook_overrides_output() -> None:
    """Custom hook should control how custom event updates tool output."""
    adapter = FakeAdapter()
    ts = datetime.now(timezone.utc)
    tool_call = ChatCompletionMessageToolCall(
        id="call-2",
        type="function",
        function=OpenAIFunction(name="batch_process", arguments='{"items": ["a"]}'),
    )

    def custom_hook(event: CustomEvent, _snapshot: ToolRenderSnapshot):
        if event.event_name == "batch_progress":
            return ToolEventRenderUpdate(
                append_output=f"progress={event.data['percent']}%\n"
            )
        return None

    async def _stream() -> AsyncGenerator[ReactOutput, None]:
        yield EventYield(
            event=LLMCallStartEvent(
                event_type=ReActEventType.LLM_CALL_START,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=0,
                messages=[],
                tools=None,
                llm_kwargs={},
                stream=False,
            )
        )
        yield EventYield(
            event=ToolCallStartEvent(
                event_type=ReActEventType.TOOL_CALL_START,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=1,
                tool_name="batch_process",
                tool_call_id="call-2",
                arguments={"items": ["a"]},
                tool_call=tool_call,
            )
        )
        yield EventYield(
            event=CustomEvent(
                event_type=ReActEventType.CUSTOM_EVENT,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=1,
                event_name="batch_progress",
                data={"percent": 50},
                tool_name="batch_process",
                tool_call_id="call-2",
            )
        )
        yield EventYield(
            event=ToolCallEndEvent(
                event_type=ReActEventType.TOOL_CALL_END,
                timestamp=ts,
                trace_id="trace-1",
                func_name="agent",
                iteration=1,
                tool_name="batch_process",
                tool_call_id="call-2",
                arguments={"items": ["a"]},
                result="done",
                execution_time=0.1,
                success=True,
            )
        )

    await consume_react_stream(_stream(), adapter=adapter, custom_hooks=[custom_hook])

    assert adapter.tool_output["call-2"] == "progress=50%\n"


@pytest.mark.asyncio
async def test_consume_react_stream_requires_event_mode() -> None:
    """TUI consumer should fail fast without EventYield outputs."""
    adapter = FakeAdapter()

    async def _stream() -> AsyncGenerator[ReactOutput, None]:
        if False:
            yield None

    with pytest.raises(ValueError, match="enable_event=True"):
        await consume_react_stream(_stream(), adapter=adapter)
