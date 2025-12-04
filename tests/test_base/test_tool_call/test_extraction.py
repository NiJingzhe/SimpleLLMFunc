"""Tests for base.tool_call.extraction module."""

from __future__ import annotations

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from SimpleLLMFunc.base.tool_call.extraction import (
    accumulate_tool_calls_from_chunks,
    extract_tool_calls,
    extract_tool_calls_from_stream_response,
)


class TestExtractToolCalls:
    """Tests for extract_tool_calls function."""

    def test_extract_single_tool_call(self, mock_chat_completion_with_tool_calls) -> None:
        """Test extracting single tool call."""
        result = extract_tool_calls(mock_chat_completion_with_tool_calls)
        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["function"]["name"] == "test_tool"

    def test_extract_multiple_tool_calls(self) -> None:
        """Test extracting multiple tool calls."""
        tool_call1 = ChatCompletionMessageToolCall(
            id="call_1",
            function=Function(name="tool1", arguments='{"arg": "value1"}'),
            type="function",
        )
        tool_call2 = ChatCompletionMessageToolCall(
            id="call_2",
            function=Function(name="tool2", arguments='{"arg": "value2"}'),
            type="function",
        )
        message = ChatCompletionMessage(
            role="assistant", content=None, tool_calls=[tool_call1, tool_call2]
        )
        choice = Choice(finish_reason="tool_calls", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = extract_tool_calls(response)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"

    def test_extract_no_tool_calls(self, mock_chat_completion) -> None:
        """Test extracting when no tool calls present."""
        result = extract_tool_calls(mock_chat_completion)
        assert result == []

    def test_extract_invalid_response(self) -> None:
        """Test extracting from invalid response."""
        result = extract_tool_calls({})
        assert result == []


class TestExtractToolCallsFromStreamResponse:
    """Tests for extract_tool_calls_from_stream_response function."""

    def test_extract_from_stream_chunk(self) -> None:
        """Test extracting tool calls from stream chunk."""
        from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
        from openai.types.chat.chat_completion_chunk import ChoiceDelta
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall as StreamToolCall,
        )

        tool_call = StreamToolCall(
            id="call_123",
            function=Function(name="test_tool", arguments='{"arg": "value"}'),
            type="function",
            index=0,
        )
        delta = ChoiceDelta(
            content=None,
            role="assistant",
            tool_calls=[tool_call],
        )
        choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
        chunk = ChatCompletionChunk(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion.chunk",
        )
        result = extract_tool_calls_from_stream_response(chunk)
        assert len(result) == 1
        assert result[0]["id"] == "call_123"

    def test_extract_no_tool_calls_in_chunk(self, mock_chat_completion_chunk) -> None:
        """Test extracting when no tool calls in chunk."""
        result = extract_tool_calls_from_stream_response(mock_chat_completion_chunk)
        assert result == []

    def test_extract_invalid_chunk(self) -> None:
        """Test extracting from invalid chunk."""
        result = extract_tool_calls_from_stream_response({})
        assert result == []


class TestAccumulateToolCallsFromChunks:
    """Tests for accumulate_tool_calls_from_chunks function."""

    def test_accumulate_single_chunk(self) -> None:
        """Test accumulating single tool call chunk."""
        chunks = [
            {
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        result = accumulate_tool_calls_from_chunks(chunks)
        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["function"]["name"] == "test_tool"

    def test_accumulate_multiple_chunks(self) -> None:
        """Test accumulating multiple tool call chunks."""
        chunks = [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "tool1", "arguments": '{"arg": "value1"}'},
            },
            {
                "index": 1,
                "id": "call_2",
                "type": "function",
                "function": {"name": "tool2", "arguments": '{"arg": "value2"}'},
            },
        ]
        result = accumulate_tool_calls_from_chunks(chunks)
        assert len(result) == 2

    def test_accumulate_incremental_arguments(self) -> None:
        """Test accumulating incremental arguments."""
        chunks = [
            {
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "val'},
            },
            {
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {"arguments": 'ue"}'},
            },
        ]
        result = accumulate_tool_calls_from_chunks(chunks)
        assert len(result) == 1
        assert result[0]["function"]["arguments"] == '{"arg": "value"}'

    def test_accumulate_chunk_without_index(self) -> None:
        """Test accumulating chunk without index."""
        chunks = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"},
            }
        ]
        result = accumulate_tool_calls_from_chunks(chunks)
        # Should skip chunks without index
        assert len(result) == 0

    def test_accumulate_incomplete_chunk(self) -> None:
        """Test accumulating incomplete chunk."""
        chunks = [
            {
                "index": 0,
                "id": None,
                "type": "function",
                "function": {"name": None, "arguments": ""},
            }
        ]
        result = accumulate_tool_calls_from_chunks(chunks)
        # Should skip incomplete chunks
        assert len(result) == 0

    def test_accumulate_empty_chunks(self) -> None:
        """Test accumulating empty chunks list."""
        result = accumulate_tool_calls_from_chunks([])
        assert result == []

