"""Tests for ReAct event types."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from SimpleLLMFunc.type.message import MessageParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function as OpenAIFunction,
)
from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.hooks.events import (
    LLMCallEndEvent,
    LLMCallErrorEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    ReactEndEvent,
    ReactIterationEndEvent,
    ReactIterationStartEvent,
    ReactStartEvent,
    ReActEvent,
    ReActEventType,
    ToolCallEndEvent,
    ToolCallErrorEvent,
    ToolCallResult,
    ToolCallArgumentsDeltaEvent,
    ToolCallStartEvent,
    ToolCallsBatchEndEvent,
    ToolCallsBatchStartEvent,
)
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.type.tool_call import ToolCall


class TestReActEventType:
    """Test ReActEventType enum."""

    def test_event_type_values(self):
        """Test that all event types have correct values."""
        assert ReActEventType.REACT_START == "react_start"
        assert ReActEventType.REACT_ITERATION_START == "react_iteration_start"
        assert ReActEventType.LLM_CALL_START == "llm_call_start"
        assert ReActEventType.LLM_CHUNK_ARRIVE == "llm_chunk_arrive"
        assert ReActEventType.LLM_CALL_END == "llm_call_end"
        assert ReActEventType.LLM_CALL_ERROR == "llm_call_error"
        assert ReActEventType.TOOL_CALLS_BATCH_START == "tool_calls_batch_start"
        assert ReActEventType.TOOL_CALL_START == "tool_call_start"
        assert ReActEventType.TOOL_CALL_ARGUMENTS_DELTA == "tool_call_arguments_delta"
        assert ReActEventType.TOOL_CALL_END == "tool_call_end"
        assert ReActEventType.TOOL_CALL_ERROR == "tool_call_error"
        assert ReActEventType.TOOL_CALLS_BATCH_END == "tool_calls_batch_end"
        assert ReActEventType.REACT_ITERATION_END == "react_iteration_end"
        assert ReActEventType.REACT_END == "react_end"


class TestReActEvent:
    """Test base ReActEvent class."""

    def test_base_event_creation(self):
        """Test creating a base event."""
        event = ReActEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
        )
        assert event.event_type == ReActEventType.REACT_START
        assert event.trace_id == "test-trace-123"
        assert event.func_name == "test_func"
        assert event.iteration == 0
        assert isinstance(event.extra, dict)
        assert len(event.extra) == 0

    def test_base_event_with_extra(self):
        """Test creating a base event with extra fields."""
        # 由于 ReActEvent 是基类，不能直接添加 extra 字段（会导致子类无法添加非默认字段）
        # 所以我们测试子类 ReactStartEvent 的 extra 字段
        event = ReactStartEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            user_task_prompt="test",
            initial_messages=[],
            available_tools=None,
            extra={"custom_field": "custom_value"},
        )
        assert event.extra["custom_field"] == "custom_value"


class TestReactStartEvent:
    """Test ReactStartEvent."""

    def test_react_start_event_creation(self):
        """Test creating a ReactStartEvent."""
        messages: MessageList = [{"role": "user", "content": "Hello"}]
        event = ReactStartEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            user_task_prompt="Hello",
            initial_messages=messages,
            available_tools=None,
        )
        assert event.user_task_prompt == "Hello"
        assert len(event.initial_messages) == 1
        assert event.available_tools is None


class TestLLMCallStartEvent:
    """Test LLMCallStartEvent."""

    def test_llm_call_start_event_creation(self):
        """Test creating a LLMCallStartEvent."""
        messages: MessageList = [{"role": "user", "content": "Hello"}]
        event = LLMCallStartEvent(
            event_type=ReActEventType.LLM_CALL_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            messages=messages,
            tools=None,
            llm_kwargs={"temperature": 0.7},
            stream=False,
        )
        assert len(event.messages) == 1
        assert event.llm_kwargs["temperature"] == 0.7
        assert event.stream is False


class TestLLMChunkArriveEvent:
    """Test LLMChunkArriveEvent."""

    def test_llm_chunk_arrive_event_creation(self):
        """Test creating a LLMChunkArriveEvent."""
        chunk = ChatCompletionChunk(
            id="test-id",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        event = LLMChunkArriveEvent(
            event_type=ReActEventType.LLM_CHUNK_ARRIVE,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            chunk=chunk,
            accumulated_content="Hello",
            chunk_index=0,
        )
        assert event.chunk.id == "test-id"
        assert event.accumulated_content == "Hello"
        assert event.chunk_index == 0


class TestLLMCallEndEvent:
    """Test LLMCallEndEvent."""

    def test_llm_call_end_event_creation(self):
        """Test creating a LLMCallEndEvent."""
        response = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello"),
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )
        messages: MessageList = [
            ChatCompletionMessage(role="assistant", content="Hello")
        ]
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=OpenAIFunction(name="test_tool", arguments='{"key": "value"}'),
        )
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        event = LLMCallEndEvent(
            event_type=ReActEventType.LLM_CALL_END,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            response=response,
            messages=messages,
            tool_calls=[tool_call],
            usage=usage,
            execution_time=1.5,
        )
        assert event.response.id == "test-id"
        assert len(event.messages) == 1
        assert len(event.tool_calls) == 1
        assert event.tool_calls[0].id == "call_123"
        assert event.usage is not None
        assert event.usage.total_tokens == 30
        assert event.execution_time == 1.5


class TestLLMCallErrorEvent:
    """Test LLMCallErrorEvent."""

    def test_llm_call_error_event_creation(self):
        """Test creating a LLMCallErrorEvent."""
        error = ValueError("Test error")
        messages: MessageList = [{"role": "user", "content": "Hello"}]
        event = LLMCallErrorEvent(
            event_type=ReActEventType.LLM_CALL_ERROR,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            error=error,
            error_message="Test error",
            error_type="ValueError",
            messages=messages,
            llm_kwargs={"temperature": 0.7},
        )
        assert isinstance(event.error, ValueError)
        assert event.error_message == "Test error"
        assert event.error_type == "ValueError"
        assert len(event.messages) == 1


class TestToolCallsBatchStartEvent:
    """Test ToolCallsBatchStartEvent."""

    def test_tool_calls_batch_start_event_creation(self):
        """Test creating a ToolCallsBatchStartEvent."""
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=OpenAIFunction(name="test_tool", arguments='{"key": "value"}'),
        )
        event = ToolCallsBatchStartEvent(
            event_type=ReActEventType.TOOL_CALLS_BATCH_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            tool_calls=[tool_call],
            batch_size=1,
        )
        assert len(event.tool_calls) == 1
        assert event.tool_calls[0].id == "call_123"
        assert event.batch_size == 1


class TestToolCallStartEvent:
    """Test ToolCallStartEvent."""

    def test_tool_call_start_event_creation(self):
        """Test creating a ToolCallStartEvent."""
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=OpenAIFunction(name="test_tool", arguments='{"key": "value"}'),
        )
        event = ToolCallStartEvent(
            event_type=ReActEventType.TOOL_CALL_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            tool_name="test_tool",
            tool_call_id="call_123",
            arguments={"key": "value"},
            tool_call=tool_call,
        )
        assert event.tool_name == "test_tool"
        assert event.tool_call_id == "call_123"
        assert event.arguments == {"key": "value"}
        assert event.tool_call.id == "call_123"


class TestToolCallEndEvent:
    """Test ToolCallEndEvent."""

    def test_tool_call_end_event_creation(self):
        """Test creating a ToolCallEndEvent."""
        event = ToolCallEndEvent(
            event_type=ReActEventType.TOOL_CALL_END,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            tool_name="test_tool",
            tool_call_id="call_123",
            arguments={"key": "value"},
            result="success",
            execution_time=0.5,
            success=True,
        )
        assert event.tool_name == "test_tool"
        assert event.tool_call_id == "call_123"
        assert event.result == "success"
        assert event.execution_time == 0.5
        assert event.success is True


class TestToolCallArgumentsDeltaEvent:
    """Test ToolCallArgumentsDeltaEvent."""

    def test_tool_call_arguments_delta_event_creation(self):
        """Test creating a ToolCallArgumentsDeltaEvent."""
        event = ToolCallArgumentsDeltaEvent(
            event_type=ReActEventType.TOOL_CALL_ARGUMENTS_DELTA,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            tool_name="execute_code",
            tool_call_id="call_123",
            argname="code",
            argcontent_delta="print(1)",
        )

        assert event.tool_name == "execute_code"
        assert event.tool_call_id == "call_123"
        assert event.argname == "code"
        assert event.argcontent_delta == "print(1)"


class TestToolCallErrorEvent:
    """Test ToolCallErrorEvent."""

    def test_tool_call_error_event_creation(self):
        """Test creating a ToolCallErrorEvent."""
        error = ValueError("Tool error")
        event = ToolCallErrorEvent(
            event_type=ReActEventType.TOOL_CALL_ERROR,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            tool_name="test_tool",
            tool_call_id="call_123",
            arguments={"key": "value"},
            error=error,
            error_message="Tool error",
            error_type="ValueError",
            execution_time=0.5,
        )
        assert event.tool_name == "test_tool"
        assert isinstance(event.error, ValueError)
        assert event.error_message == "Tool error"
        assert event.error_type == "ValueError"
        assert event.execution_time == 0.5


class TestToolCallsBatchEndEvent:
    """Test ToolCallsBatchEndEvent."""

    def test_tool_calls_batch_end_event_creation(self):
        """Test creating a ToolCallsBatchEndEvent."""
        tool_result: ToolCallResult = {
            "tool_name": "test_tool",
            "tool_call_id": "call_123",
            "result": "success",
            "execution_time": 0.5,
            "success": True,
        }
        event = ToolCallsBatchEndEvent(
            event_type=ReActEventType.TOOL_CALLS_BATCH_END,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            tool_results=[tool_result],
            batch_size=1,
            total_execution_time=0.5,
            success_count=1,
            error_count=0,
        )
        assert len(event.tool_results) == 1
        assert event.tool_results[0]["tool_name"] == "test_tool"
        assert event.batch_size == 1
        assert event.total_execution_time == 0.5
        assert event.success_count == 1
        assert event.error_count == 0


class TestReactIterationStartEvent:
    """Test ReactIterationStartEvent."""

    def test_react_iteration_start_event_creation(self):
        """Test creating a ReactIterationStartEvent."""
        messages: MessageList = [{"role": "user", "content": "Hello"}]
        event = ReactIterationStartEvent(
            event_type=ReActEventType.REACT_ITERATION_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=1,
            current_messages=messages,
        )
        assert event.iteration == 1
        assert len(event.current_messages) == 1


class TestReactIterationEndEvent:
    """Test ReactIterationEndEvent."""

    def test_react_iteration_end_event_creation(self):
        """Test creating a ReactIterationEndEvent."""
        messages: MessageList = [{"role": "user", "content": "Hello"}]
        event = ReactIterationEndEvent(
            event_type=ReActEventType.REACT_ITERATION_END,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=1,
            messages=messages,
            iteration_time=2.0,
            tool_calls_count=2,
        )
        assert event.iteration == 1
        assert len(event.messages) == 1
        assert event.iteration_time == 2.0
        assert event.tool_calls_count == 2


class TestReactEndEvent:
    """Test ReactEndEvent."""

    def test_react_end_event_creation(self):
        """Test creating a ReactEndEvent."""
        messages: MessageList = [
            ChatCompletionMessage(role="assistant", content="Hello")
        ]
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        event = ReactEndEvent(
            event_type=ReActEventType.REACT_END,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            final_response="Hello",
            final_messages=messages,
            total_iterations=2,
            total_execution_time=5.0,
            total_tool_calls=3,
            total_llm_calls=2,
            total_token_usage=usage,
        )
        assert event.final_response == "Hello"
        assert len(event.final_messages) == 1
        assert event.total_iterations == 2
        assert event.total_execution_time == 5.0
        assert event.total_tool_calls == 3
        assert event.total_llm_calls == 2
        assert event.total_token_usage is not None
        assert event.total_token_usage.total_tokens == 30


class TestEventInheritance:
    """Test that all events inherit from ReActEvent."""

    def test_all_events_are_instances_of_react_event(self):
        """Test that all event types are instances of ReActEvent."""
        timestamp = datetime.now(timezone.utc)
        base_kwargs = {
            "timestamp": timestamp,
            "trace_id": "test-trace",
            "func_name": "test_func",
            "iteration": 0,
        }

        events = [
            ReactStartEvent(
                event_type=ReActEventType.REACT_START,
                user_task_prompt="test",
                initial_messages=[],
                available_tools=None,
                **base_kwargs,
            ),
            ReactIterationStartEvent(
                event_type=ReActEventType.REACT_ITERATION_START,
                current_messages=[],
                **base_kwargs,
            ),
            LLMCallStartEvent(
                event_type=ReActEventType.LLM_CALL_START,
                messages=[],
                tools=None,
                llm_kwargs={},
                stream=False,
                **base_kwargs,
            ),
        ]

        for event in events:
            assert isinstance(event, ReActEvent)
            assert hasattr(event, "event_type")
            assert hasattr(event, "timestamp")
            assert hasattr(event, "trace_id")
            assert hasattr(event, "func_name")
            assert hasattr(event, "iteration")
            assert hasattr(event, "extra")
