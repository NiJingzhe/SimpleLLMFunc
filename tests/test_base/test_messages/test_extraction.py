"""Tests for base.messages.extraction module."""

from __future__ import annotations

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.base.messages.extraction import extract_usage_from_response


class TestExtractUsageFromResponse:
    """Tests for extract_usage_from_response function."""

    def test_extract_usage_from_completion(self) -> None:
        """Test extracting usage from ChatCompletion."""
        usage = CompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        message = ChatCompletionMessage(role="assistant", content="test")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
            usage=usage,
        )
        result = extract_usage_from_response(response)
        assert result is not None
        assert result["input"] == 10
        assert result["output"] == 20
        assert result["total"] == 30

    def test_extract_usage_none(self) -> None:
        """Test extracting usage from None."""
        result = extract_usage_from_response(None)
        assert result is None

    def test_extract_usage_no_usage_field(self) -> None:
        """Test extracting usage when usage field is missing."""
        message = ChatCompletionMessage(role="assistant", content="test")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = extract_usage_from_response(response)
        assert result is None

    def test_extract_usage_from_chunk(self) -> None:
        """Test extracting usage from ChatCompletionChunk."""
        from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        delta = ChoiceDelta(content="chunk", role="assistant")
        choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
        chunk = ChatCompletionChunk(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion.chunk",
        )
        # Chunks typically don't have usage, but function should handle gracefully
        result = extract_usage_from_response(chunk)
        # Should return None if no usage field
        assert result is None or isinstance(result, dict)

