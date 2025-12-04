"""Tests for llm_decorator.steps.chat.message module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.chat.message import (
    build_chat_messages,
    build_chat_system_prompt,
    build_chat_user_message_content,
    extract_conversation_history,
    filter_history_messages,
)


class TestExtractConversationHistory:
    """Tests for extract_conversation_history function."""

    def test_extract_history_exists(self) -> None:
        """Test extracting history when it exists."""
        arguments = {
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "message": "test",
        }
        result = extract_conversation_history(arguments, "test_func")
        assert result is not None
        assert len(result) == 2

    def test_extract_history_not_exists(self) -> None:
        """Test extracting history when it doesn't exist."""
        arguments = {"message": "test"}
        result = extract_conversation_history(arguments, "test_func")
        assert result is None

    def test_extract_history_invalid_format(self) -> None:
        """Test extracting history with invalid format."""
        arguments = {"history": "not a list"}
        result = extract_conversation_history(arguments, "test_func")
        assert result is None


class TestBuildChatSystemPrompt:
    """Tests for build_chat_system_prompt function."""

    def test_build_with_docstring(self) -> None:
        """Test building system prompt with docstring."""
        result = build_chat_system_prompt("Test docstring", None)
        assert result == "Test docstring"

    def test_build_without_docstring(self) -> None:
        """Test building system prompt without docstring."""
        result = build_chat_system_prompt("", None)
        assert result is None

    def test_build_with_tools(self) -> None:
        """Test building system prompt with tools."""
        tools = [
            {
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                }
            }
        ]
        result = build_chat_system_prompt("Test docstring", tools)
        assert result is not None
        assert "test_tool" in result


class TestBuildChatUserMessageContent:
    """Tests for build_chat_user_message_content function."""

    def test_build_text_content(self) -> None:
        """Test building text user message content."""
        arguments = {"message": "Hello", "param": "value"}
        type_hints = {"message": str, "param": str}
        result = build_chat_user_message_content(
            arguments, type_hints, False, ["history"]
        )
        assert isinstance(result, str)
        assert "Hello" in result

    @patch("SimpleLLMFunc.llm_decorator.steps.chat.message.build_multimodal_content")
    def test_build_multimodal_content(
        self, mock_build_multimodal: Any
    ) -> None:
        """Test building multimodal user message content."""
        mock_build_multimodal.return_value = [
            {"type": "text", "text": "test"}
        ]
        arguments = {"image": "test"}
        type_hints = {"image": str}
        result = build_chat_user_message_content(
            arguments, type_hints, True, ["history"]
        )
        assert isinstance(result, list)
        mock_build_multimodal.assert_called_once()


class TestFilterHistoryMessages:
    """Tests for filter_history_messages function."""

    def test_filter_valid_messages(self) -> None:
        """Test filtering valid history messages."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = filter_history_messages(history, "test_func")
        assert len(result) == 2

    def test_filter_system_messages(self) -> None:
        """Test filtering out system messages."""
        history = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "Hello"},
        ]
        result = filter_history_messages(history, "test_func")
        assert len(result) == 1
        assert result[0]["role"] == "user"


class TestBuildChatMessages:
    """Tests for build_chat_messages function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.chat.message.process_tools")
    @patch("SimpleLLMFunc.llm_decorator.steps.chat.message.has_multimodal_content")
    def test_build_messages(
        self, mock_has_multimodal: Any, mock_process_tools: Any
    ) -> None:
        """Test building chat messages."""
        mock_process_tools.return_value = (None, {})
        mock_has_multimodal.return_value = False
        
        from SimpleLLMFunc.llm_decorator.steps.common.types import FunctionSignature
        import inspect

        def test_func(message: str) -> str:
            """Test function."""
            return "result"

        sig = inspect.signature(test_func)
        bound = sig.bind("Hello")
        bound.apply_defaults()
        
        signature = FunctionSignature(
            func_name="test_func",
            trace_id="trace_123",
            bound_args=bound,
            signature=sig,
            type_hints={"message": str, "return": str},
            return_type=str,
            docstring="Test function.",
        )

        result = build_chat_messages(signature, None, ["history"])
        assert len(result) >= 1

