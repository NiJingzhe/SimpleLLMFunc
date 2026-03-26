"""Tests for llm_decorator.steps.chat.message module."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.chat.message import (
    build_chat_messages,
    build_chat_system_prompt,
    build_chat_user_message_content,
    extract_history_system_prompt,
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
        result = build_chat_system_prompt("Test docstring")
        assert result == "Test docstring"

    def test_build_without_docstring(self) -> None:
        """Test building system prompt without docstring."""
        result = build_chat_system_prompt("")
        assert result is None

    def test_build_does_not_prepend_tool_list(self) -> None:
        """System prompt should stay focused on docstring/history text."""
        result = build_chat_system_prompt("Test docstring")
        assert result == "Test docstring"

    def test_build_prefers_history_system_prompt(self) -> None:
        """History-derived system prompt should override docstring text."""
        result = build_chat_system_prompt(
            "Docstring prompt",
            history_system_prompt="History prompt",
        )
        assert result == "History prompt"


class TestExtractHistorySystemPrompt:
    """Tests for extract_history_system_prompt helper."""

    def test_extract_latest_system_prompt(self) -> None:
        history = [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "new system"},
        ]
        assert extract_history_system_prompt(history) == "new system"

    def test_extract_system_prompt_returns_none_when_missing(self) -> None:
        history = [{"role": "user", "content": "hi"}]
        assert extract_history_system_prompt(history) is None


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
    def test_build_multimodal_content(self, mock_build_multimodal: Any) -> None:
        """Test building multimodal user message content."""
        mock_build_multimodal.return_value = [{"type": "text", "text": "test"}]
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
        assert cast(dict[str, Any], result[0])["role"] == "user"


class TestBuildChatMessages:
    """Tests for build_chat_messages function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.chat.message.has_multimodal_content")
    def test_build_messages(self, mock_has_multimodal: Any) -> None:
        """Test building chat messages."""
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

    @patch("SimpleLLMFunc.llm_decorator.steps.chat.message.has_multimodal_content")
    def test_build_messages_uses_history_system_prompt(
        self, mock_has_multimodal: Any
    ) -> None:
        """History system prompt should override docstring in built messages."""
        mock_has_multimodal.return_value = False

        from SimpleLLMFunc.llm_decorator.steps.common.types import FunctionSignature
        import inspect

        def test_func(message: str, history=None) -> str:
            """Docstring system prompt."""
            return "result"

        sig = inspect.signature(test_func)
        history = [
            {"role": "system", "content": "Runtime system prompt"},
            {"role": "user", "content": "hello"},
        ]
        bound = sig.bind("Hello", history)
        bound.apply_defaults()

        signature = FunctionSignature(
            func_name="test_func",
            trace_id="trace_123",
            bound_args=bound,
            signature=sig,
            type_hints={"message": str, "history": list, "return": str},
            return_type=str,
            docstring="Docstring system prompt.",
        )

        result = build_chat_messages(signature, None, ["history"])
        first_message = cast(dict[str, Any], result[0])
        assert first_message["role"] == "system"
        assert first_message["content"] == "Runtime system prompt"
