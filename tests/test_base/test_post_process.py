"""Tests for base.post_process module."""

from __future__ import annotations

import json
from typing import Optional
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel

from SimpleLLMFunc.base.post_process import (
    _convert_to_dict,
    _convert_to_pydantic_model,
    _convert_to_primitive_type,
    extract_content_from_response,
    extract_content_from_stream_response,
    process_response,
)


class TestExtractContentFromResponse:
    """Tests for extract_content_from_response function."""

    def test_extract_content_normal(self, mock_chat_completion: ChatCompletion) -> None:
        """Test extracting content from normal response."""
        result = extract_content_from_response(mock_chat_completion, "test_func")
        assert result == "Test response"

    def test_extract_content_empty(self) -> None:
        """Test extracting content from empty response."""
        message = ChatCompletionMessage(role="assistant", content="")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = extract_content_from_response(response, "test_func")
        assert result == ""

    def test_extract_content_none(self) -> None:
        """Test extracting content when content is None."""
        message = ChatCompletionMessage(role="assistant", content=None)
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = extract_content_from_response(response, "test_func")
        assert result == ""

    def test_extract_content_no_choices(self) -> None:
        """Test extracting content when response has no choices."""
        response = ChatCompletion(
            id="test",
            choices=[],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = extract_content_from_response(response, "test_func")
        assert result == ""

    def test_extract_content_invalid_response(self) -> None:
        """Test extracting content from invalid response."""
        result = extract_content_from_response({}, "test_func")
        assert result == ""


class TestExtractContentFromStreamResponse:
    """Tests for extract_content_from_stream_response function."""

    def test_extract_content_from_chunk(self, mock_chat_completion_chunk) -> None:
        """Test extracting content from stream chunk."""
        result = extract_content_from_stream_response(
            mock_chat_completion_chunk, "test_func"
        )
        assert result == "chunk"

    def test_extract_content_empty_chunk(self) -> None:
        """Test extracting content from empty chunk."""
        from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        delta = ChoiceDelta(content="", role="assistant")
        choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
        chunk = ChatCompletionChunk(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion.chunk",
        )
        result = extract_content_from_stream_response(chunk, "test_func")
        assert result == ""

    def test_extract_content_none_chunk(self) -> None:
        """Test extracting content when chunk content is None."""
        from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        delta = ChoiceDelta(content=None, role="assistant")
        choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
        chunk = ChatCompletionChunk(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion.chunk",
        )
        result = extract_content_from_stream_response(chunk, "test_func")
        assert result == ""

    def test_extract_content_no_chunk(self) -> None:
        """Test extracting content when chunk is None."""
        result = extract_content_from_stream_response(None, "test_func")
        assert result == ""


class TestConvertToPrimitiveType:
    """Tests for _convert_to_primitive_type function."""

    def test_convert_to_int(self) -> None:
        """Test converting to int."""
        result = _convert_to_primitive_type("123", int)
        assert result == 123
        assert isinstance(result, int)

    def test_convert_to_int_with_whitespace(self) -> None:
        """Test converting to int with whitespace."""
        result = _convert_to_primitive_type("  456  ", int)
        assert result == 456

    def test_convert_to_float(self) -> None:
        """Test converting to float."""
        result = _convert_to_primitive_type("123.45", float)
        assert result == 123.45
        assert isinstance(result, float)

    def test_convert_to_bool_true(self) -> None:
        """Test converting to bool (true cases)."""
        assert _convert_to_primitive_type("true", bool) is True
        assert _convert_to_primitive_type("True", bool) is True
        assert _convert_to_primitive_type("yes", bool) is True
        assert _convert_to_primitive_type("1", bool) is True

    def test_convert_to_bool_false(self) -> None:
        """Test converting to bool (false cases)."""
        assert _convert_to_primitive_type("false", bool) is False
        assert _convert_to_primitive_type("no", bool) is False
        assert _convert_to_primitive_type("0", bool) is False

    def test_convert_to_int_invalid(self) -> None:
        """Test converting invalid string to int."""
        with pytest.raises(ValueError):
            _convert_to_primitive_type("not_a_number", int)

    def test_convert_to_float_invalid(self) -> None:
        """Test converting invalid string to float."""
        with pytest.raises(ValueError):
            _convert_to_primitive_type("not_a_number", float)

    def test_convert_unsupported_type(self) -> None:
        """Test converting to unsupported type."""
        with pytest.raises(ValueError):
            _convert_to_primitive_type("test", str)  # str is not handled by this function


class TestConvertToDict:
    """Tests for _convert_to_dict function."""

    def test_convert_valid_json(self) -> None:
        """Test converting valid JSON string."""
        json_str = '{"key": "value", "number": 123}'
        result = _convert_to_dict(json_str, "test_func")
        assert result == {"key": "value", "number": 123}

    def test_convert_json_with_code_block(self) -> None:
        """Test converting JSON wrapped in code block."""
        json_str = '```json\n{"key": "value"}\n```'
        result = _convert_to_dict(json_str, "test_func")
        assert result == {"key": "value"}

    def test_convert_json_with_markdown_code_block(self) -> None:
        """Test converting JSON in markdown code block."""
        json_str = "```json\n{\"key\": \"value\"}\n```"
        result = _convert_to_dict(json_str, "test_func")
        assert result == {"key": "value"}

    def test_convert_invalid_json(self) -> None:
        """Test converting invalid JSON."""
        with pytest.raises(ValueError):
            _convert_to_dict("not a json", "test_func")

    def test_convert_empty_string(self) -> None:
        """Test converting empty string."""
        with pytest.raises(ValueError):
            _convert_to_dict("", "test_func")


class TestConvertToPydanticModel:
    """Tests for _convert_to_pydantic_model function."""

    def test_convert_valid_json_to_model(self, sample_pydantic_model) -> None:
        """Test converting valid JSON to Pydantic model."""
        json_str = '{"name": "John", "age": 30, "email": "john@example.com"}'
        result = _convert_to_pydantic_model(json_str, sample_pydantic_model, "test_func")
        assert isinstance(result, sample_pydantic_model)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_convert_json_with_code_block_to_model(self, sample_pydantic_model) -> None:
        """Test converting JSON in code block to model."""
        json_str = '```json\n{"name": "Jane", "age": 25}\n```'
        result = _convert_to_pydantic_model(json_str, sample_pydantic_model, "test_func")
        assert result.name == "Jane"
        assert result.age == 25

    def test_convert_empty_string_to_model(self, sample_pydantic_model) -> None:
        """Test converting empty string to model."""
        with pytest.raises(ValueError):
            _convert_to_pydantic_model("", sample_pydantic_model, "test_func")

    def test_convert_invalid_json_to_model(self, sample_pydantic_model) -> None:
        """Test converting invalid JSON to model."""
        with pytest.raises(ValueError):
            _convert_to_pydantic_model("not json", sample_pydantic_model, "test_func")


class TestProcessResponse:
    """Tests for process_response function."""

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_str(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as str."""
        mock_get_context.return_value = "test_func"
        result = process_response(mock_chat_completion, str)
        assert result == "Test response"
        assert isinstance(result, str)

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_int(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as int."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(role="assistant", content="123")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, int)
        assert result == 123
        assert isinstance(result, int)

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_float(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as float."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(role="assistant", content="123.45")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, float)
        assert result == 123.45
        assert isinstance(result, float)

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_bool(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as bool."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(role="assistant", content="true")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, bool)
        assert result is True

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_dict(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as dict."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(
            role="assistant", content='{"key": "value", "number": 123}'
        )
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, dict)
        assert result == {"key": "value", "number": 123}
        assert isinstance(result, dict)

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_pydantic_model(
        self,
        mock_get_context: Any,
        mock_chat_completion: ChatCompletion,
        sample_pydantic_model,
    ) -> None:
        """Test processing response as Pydantic model."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(
            role="assistant", content='{"name": "Alice", "age": 28}'
        )
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, sample_pydantic_model)
        assert isinstance(result, sample_pydantic_model)
        assert result.name == "Alice"
        assert result.age == 28

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_none_type(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response with None return type."""
        mock_get_context.return_value = "test_func"
        result = process_response(mock_chat_completion, None)
        assert result == "Test response"

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_empty_content(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response with empty content."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(role="assistant", content="")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, str)
        assert result == ""

