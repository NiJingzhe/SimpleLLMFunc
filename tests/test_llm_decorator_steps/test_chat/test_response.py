"""Tests for llm_decorator.steps.chat.response module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.chat.response import (
    extract_stream_response_content,
    process_chat_response_stream,
    process_single_chat_response,
)


class TestExtractStreamResponseContent:
    """Tests for extract_stream_response_content function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.chat.response.extract_content_from_stream_response")
    def test_extract_content(self, mock_extract: Any, mock_chat_completion_chunk: Any) -> None:
        """Test extracting stream response content."""
        mock_extract.return_value = "chunk content"
        result = extract_stream_response_content(mock_chat_completion_chunk, "test_func")
        assert result == "chunk content"
        mock_extract.assert_called_once_with(mock_chat_completion_chunk, "test_func")


class TestProcessSingleChatResponse:
    """Tests for process_single_chat_response function."""

    def test_process_raw_mode(self, mock_chat_completion: Any) -> None:
        """Test processing response in raw mode."""
        result = process_single_chat_response(
            mock_chat_completion, "raw", False, "test_func"
        )
        assert result == mock_chat_completion

    @patch("SimpleLLMFunc.llm_decorator.steps.chat.response.extract_content_from_response")
    def test_process_text_mode_non_stream(
        self, mock_extract: Any, mock_chat_completion: Any
    ) -> None:
        """Test processing response in text mode (non-stream)."""
        mock_extract.return_value = "content"
        result = process_single_chat_response(
            mock_chat_completion, "text", False, "test_func"
        )
        assert result == "content"

    @patch("SimpleLLMFunc.llm_decorator.steps.chat.response.extract_stream_response_content")
    def test_process_text_mode_stream(
        self, mock_extract: Any, mock_chat_completion_chunk: Any
    ) -> None:
        """Test processing response in text mode (stream)."""
        mock_extract.return_value = "chunk"
        result = process_single_chat_response(
            mock_chat_completion_chunk, "text", True, "test_func"
        )
        assert result == "chunk"


class TestProcessChatResponseStream:
    """Tests for process_chat_response_stream function."""

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.llm_decorator.steps.chat.response.process_single_chat_response")
    @patch("SimpleLLMFunc.llm_decorator.steps.chat.response.app_log")
    async def test_process_stream_text_mode(
        self, mock_app_log: Any, mock_process: Any, sample_messages: list
    ) -> None:
        """Test processing stream in text mode."""
        mock_process.return_value = "content"
        
        async def mock_stream():
            yield "response1"
            yield "response2"
        
        results = []
        async for content, history in process_chat_response_stream(
            mock_stream(), "text", sample_messages, "test_func", True
        ):
            results.append((content, history))
        
        assert len(results) >= 2  # Should have responses + end marker
        assert results[-1][0] == ""  # End marker should be empty string

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.llm_decorator.steps.chat.response.process_single_chat_response")
    async def test_process_stream_raw_mode(
        self, mock_process: Any, sample_messages: list, mock_chat_completion: Any
    ) -> None:
        """Test processing stream in raw mode."""
        mock_process.return_value = mock_chat_completion
        
        async def mock_stream():
            yield mock_chat_completion
        
        results = []
        async for content, history in process_chat_response_stream(
            mock_stream(), "raw", sample_messages, "test_func", False
        ):
            results.append((content, history))
        
        assert len(results) >= 1

