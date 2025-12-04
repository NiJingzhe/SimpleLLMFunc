"""Tests for llm_decorator.steps.common.signature module."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.common.signature import (
    bind_function_arguments,
    build_function_signature,
    extract_function_metadata,
    extract_template_params,
    generate_trace_id,
    parse_function_signature,
)
from SimpleLLMFunc.llm_decorator.steps.common.types import FunctionSignature


class TestExtractTemplateParams:
    """Tests for extract_template_params function."""

    def test_extract_template_params_exists(self) -> None:
        """Test extracting template params when they exist."""
        kwargs: Dict[str, Any] = {"_template_params": {"key": "value"}, "other": "param"}
        result = extract_template_params(kwargs)
        assert result == {"key": "value"}
        assert "_template_params" not in kwargs

    def test_extract_template_params_not_exists(self) -> None:
        """Test extracting template params when they don't exist."""
        kwargs: Dict[str, Any] = {"other": "param"}
        result = extract_template_params(kwargs)
        assert result is None
        assert "other" in kwargs


class TestExtractFunctionMetadata:
    """Tests for extract_function_metadata function."""

    def test_extract_metadata_simple_function(self) -> None:
        """Test extracting metadata from simple function."""

        def test_func(param1: str, param2: int) -> str:
            """Test function docstring."""
            return "result"

        signature, type_hints, return_type, docstring, func_name = (
            extract_function_metadata(test_func)
        )

        assert func_name == "test_func"
        assert docstring == "Test function docstring."
        assert return_type == str
        assert "param1" in type_hints
        assert "param2" in type_hints

    def test_extract_metadata_no_docstring(self) -> None:
        """Test extracting metadata from function without docstring."""

        def test_func(param: str) -> str:
            return "result"

        signature, type_hints, return_type, docstring, func_name = (
            extract_function_metadata(test_func)
        )

        assert func_name == "test_func"
        assert docstring == ""


class TestGenerateTraceId:
    """Tests for generate_trace_id function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.common.signature.get_current_trace_id")
    def test_generate_trace_id_no_context(self, mock_get_trace: Any) -> None:
        """Test generating trace ID without context."""
        mock_get_trace.return_value = None
        trace_id = generate_trace_id("test_func")
        assert trace_id.startswith("test_func_")
        assert len(trace_id) > len("test_func_")

    @patch("SimpleLLMFunc.llm_decorator.steps.common.signature.get_current_trace_id")
    def test_generate_trace_id_with_context(self, mock_get_trace: Any) -> None:
        """Test generating trace ID with context."""
        mock_get_trace.return_value = "parent_trace_123"
        trace_id = generate_trace_id("test_func")
        assert trace_id.startswith("test_func_")
        assert "parent_trace_123" in trace_id


class TestBindFunctionArguments:
    """Tests for bind_function_arguments function."""

    def test_bind_positional_args(self) -> None:
        """Test binding positional arguments."""

        def test_func(param1: str, param2: int) -> str:
            return "result"

        sig = inspect.signature(test_func)
        bound = bind_function_arguments(sig, ("test", 123), {})
        assert bound.arguments["param1"] == "test"
        assert bound.arguments["param2"] == 123

    def test_bind_keyword_args(self) -> None:
        """Test binding keyword arguments."""

        def test_func(param1: str, param2: int) -> str:
            return "result"

        sig = inspect.signature(test_func)
        bound = bind_function_arguments(sig, (), {"param1": "test", "param2": 123})
        assert bound.arguments["param1"] == "test"
        assert bound.arguments["param2"] == 123

    def test_bind_with_defaults(self) -> None:
        """Test binding with default values."""

        def test_func(param1: str, param2: int = 10) -> str:
            return "result"

        sig = inspect.signature(test_func)
        bound = bind_function_arguments(sig, ("test",), {})
        assert bound.arguments["param1"] == "test"
        assert bound.arguments["param2"] == 10


class TestBuildFunctionSignature:
    """Tests for build_function_signature function."""

    def test_build_signature(self, sample_bound_args: inspect.BoundArguments) -> None:
        """Test building function signature."""

        def test_func(param1: str, param2: int = 10) -> str:
            """Test function."""
            return "result"

        sig = inspect.signature(test_func)
        type_hints = {"param1": str, "param2": int, "return": str}

        result = build_function_signature(
            func_name="test_func",
            trace_id="trace_123",
            bound_args=sample_bound_args,
            signature=sig,
            type_hints=type_hints,
            return_type=str,
            docstring="Test function.",
        )

        assert isinstance(result, FunctionSignature)
        assert result.func_name == "test_func"
        assert result.trace_id == "trace_123"
        assert result.return_type == str
        assert result.docstring == "Test function."


class TestParseFunctionSignature:
    """Tests for parse_function_signature function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.common.signature.generate_trace_id")
    def test_parse_signature_simple(
        self, mock_generate_trace: Any
    ) -> None:
        """Test parsing simple function signature."""
        mock_generate_trace.return_value = "trace_123"

        def test_func(param1: str) -> str:
            """Test function."""
            return "result"

        signature, template_params = parse_function_signature(
            test_func, ("test",), {}
        )

        assert isinstance(signature, FunctionSignature)
        assert signature.func_name == "test_func"
        assert signature.trace_id == "trace_123"
        assert template_params is None

    @patch("SimpleLLMFunc.llm_decorator.steps.common.signature.generate_trace_id")
    def test_parse_signature_with_template_params(
        self, mock_generate_trace: Any
    ) -> None:
        """Test parsing function signature with template params."""
        mock_generate_trace.return_value = "trace_123"

        def test_func(param1: str) -> str:
            """Test function."""
            return "result"

        kwargs = {"_template_params": {"key": "value"}}
        signature, template_params = parse_function_signature(
            test_func, ("test",), kwargs
        )

        assert template_params == {"key": "value"}
        assert "_template_params" not in kwargs

