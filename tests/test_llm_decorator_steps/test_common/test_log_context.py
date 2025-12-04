"""Tests for llm_decorator.steps.common.log_context module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.common.log_context import (
    create_log_context_manager,
    log_function_call,
    setup_log_context,
)


class TestLogFunctionCall:
    """Tests for log_function_call function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.common.log_context.app_log")
    @patch("SimpleLLMFunc.llm_decorator.steps.common.log_context.get_location")
    def test_log_function_call(
        self, mock_get_location: MagicMock, mock_app_log: MagicMock
    ) -> None:
        """Test logging function call."""
        mock_get_location.return_value = "test_location"
        log_function_call("test_func", {"param1": "value1", "param2": 123})
        mock_app_log.assert_called_once()
        call_args = mock_app_log.call_args[0][0]
        assert "test_func" in call_args
        assert "param1" in call_args or "value1" in call_args


class TestCreateLogContextManager:
    """Tests for create_log_context_manager function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.common.log_context.async_log_context")
    def test_create_log_context_manager(self, mock_async_log_context: MagicMock) -> None:
        """Test creating log context manager."""
        mock_context = AsyncMock()
        mock_async_log_context.return_value = mock_context
        
        result = create_log_context_manager("test_func", "trace_123")
        
        assert result == mock_context
        mock_async_log_context.assert_called_once_with(
            trace_id="trace_123",
            function_name="test_func",
            input_tokens=0,
            output_tokens=0,
        )


class TestSetupLogContext:
    """Tests for setup_log_context function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.common.log_context.log_function_call")
    @patch("SimpleLLMFunc.llm_decorator.steps.common.log_context.create_log_context_manager")
    def test_setup_log_context(
        self,
        mock_create_manager: MagicMock,
        mock_log_call: MagicMock,
    ) -> None:
        """Test setting up log context."""
        mock_context = AsyncMock()
        mock_create_manager.return_value = mock_context
        
        result = setup_log_context(
            func_name="test_func",
            trace_id="trace_123",
            arguments={"param1": "value1"},
        )
        
        mock_log_call.assert_called_once_with("test_func", {"param1": "value1"})
        mock_create_manager.assert_called_once_with("test_func", "trace_123")
        assert result == mock_context

