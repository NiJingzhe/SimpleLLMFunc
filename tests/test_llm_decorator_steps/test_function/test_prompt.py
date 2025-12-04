"""Tests for llm_decorator.steps.function.prompt module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.function.prompt import (
    build_initial_prompts,
    build_parameter_type_descriptions,
    build_return_type_description,
    build_text_messages,
)


class TestBuildParameterTypeDescriptions:
    """Tests for build_parameter_type_descriptions function."""

    def test_build_descriptions(self) -> None:
        """Test building parameter type descriptions."""
        param_type_hints = {"param1": str, "param2": int}
        result = build_parameter_type_descriptions(param_type_hints)
        assert len(result) == 2
        assert any("param1" in desc for desc in result)
        assert any("param2" in desc for desc in result)

    def test_build_empty(self) -> None:
        """Test building descriptions from empty hints."""
        result = build_parameter_type_descriptions({})
        assert result == []


class TestBuildReturnTypeDescription:
    """Tests for build_return_type_description function."""

    def test_build_str_type(self) -> None:
        """Test building description for str type."""
        result = build_return_type_description(str)
        assert "str" in result.lower() or "string" in result.lower()

    def test_build_none_type(self) -> None:
        """Test building description for None type."""
        result = build_return_type_description(None)
        assert "未知" in result or "unknown" in result.lower()


class TestBuildTextMessages:
    """Tests for build_text_messages function."""

    def test_build_messages(self) -> None:
        """Test building text messages."""
        result = build_text_messages(
            processed_docstring="Test function",
            param_type_descriptions=["  - param1: str"],
            return_type_description="str",
            arguments={"param1": "value1"},
            system_template="Function: {function_description}",
            user_template="Params: {parameters}",
        )
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"


class TestBuildInitialPrompts:
    """Tests for build_initial_prompts function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.function.prompt.has_multimodal_content")
    def test_build_text_prompts(
        self, mock_has_multimodal: Any
    ) -> None:
        """Test building text prompts."""
        mock_has_multimodal.return_value = False
        from SimpleLLMFunc.llm_decorator.steps.common.types import FunctionSignature
        import inspect

        def test_func(param1: str) -> str:
            """Test function."""
            return "result"

        sig = inspect.signature(test_func)
        bound = sig.bind("test")
        bound.apply_defaults()
        
        signature = FunctionSignature(
            func_name="test_func",
            trace_id="trace_123",
            bound_args=bound,
            signature=sig,
            type_hints={"param1": str, "return": str},
            return_type=str,
            docstring="Test function.",
        )

        result = build_initial_prompts(signature)
        assert len(result) >= 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    @patch("SimpleLLMFunc.llm_decorator.steps.function.prompt.has_multimodal_content")
    @patch("SimpleLLMFunc.llm_decorator.steps.function.prompt.build_multimodal_content")
    def test_build_multimodal_prompts(
        self, mock_build_multimodal: Any, mock_has_multimodal: Any
    ) -> None:
        """Test building multimodal prompts."""
        mock_has_multimodal.return_value = True
        mock_build_multimodal.return_value = [
            {"type": "text", "text": "test"}
        ]
        
        from SimpleLLMFunc.llm_decorator.steps.common.types import FunctionSignature
        import inspect

        def test_func(param1: str) -> str:
            """Test function."""
            return "result"

        sig = inspect.signature(test_func)
        bound = sig.bind("test")
        bound.apply_defaults()
        
        signature = FunctionSignature(
            func_name="test_func",
            trace_id="trace_123",
            bound_args=bound,
            signature=sig,
            type_hints={"param1": str, "return": str},
            return_type=str,
            docstring="Test function.",
        )

        result = build_initial_prompts(signature)
        assert len(result) >= 2
        mock_build_multimodal.assert_called()

