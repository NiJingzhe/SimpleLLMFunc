"""Tests for llm_decorator.steps.function.response module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.function.response import (
    extract_response_content,
    parse_and_validate_response,
    parse_response_to_type,
)


class TestExtractResponseContent:
    """Tests for extract_response_content function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.function.response.extract_content_from_response")
    def test_extract_content(self, mock_extract: Any, mock_chat_completion: Any) -> None:
        """Test extracting response content."""
        mock_extract.return_value = "test content"
        result = extract_response_content(mock_chat_completion, "test_func")
        assert result == "test content"
        mock_extract.assert_called_once_with(mock_chat_completion, "test_func")


class TestParseResponseToType:
    """Tests for parse_response_to_type function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.function.response.process_response")
    def test_parse_to_type(self, mock_process: Any, mock_chat_completion: Any) -> None:
        """Test parsing response to type."""
        mock_process.return_value = "parsed result"
        result = parse_response_to_type(mock_chat_completion, str)
        assert result == "parsed result"
        mock_process.assert_called_once_with(mock_chat_completion, str)


class TestParseAndValidateResponse:
    """Tests for parse_and_validate_response function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.function.response.parse_response_to_type")
    def test_parse_and_validate(
        self, mock_parse: Any, mock_chat_completion: Any
    ) -> None:
        """Test parsing and validating response."""
        mock_parse.return_value = "result"
        result = parse_and_validate_response(
            mock_chat_completion, str, "test_func"
        )
        assert result == "result"
        mock_parse.assert_called_once_with(mock_chat_completion, str)

