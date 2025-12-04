"""Tests for llm_decorator.steps.common.prompt module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.common.prompt import (
    extract_parameter_type_hints,
    process_docstring_template,
)


class TestProcessDocstringTemplate:
    """Tests for process_docstring_template function."""

    def test_process_without_template_params(self) -> None:
        """Test processing docstring without template params."""
        docstring = "Simple docstring"
        result = process_docstring_template(docstring, None)
        assert result == docstring

    def test_process_with_template_params(self) -> None:
        """Test processing docstring with template params."""
        docstring = "Function with {param1} and {param2}"
        template_params = {"param1": "value1", "param2": "value2"}
        result = process_docstring_template(docstring, template_params)
        assert result == "Function with value1 and value2"

    @patch("SimpleLLMFunc.llm_decorator.steps.common.prompt.push_warning")
    @patch("SimpleLLMFunc.llm_decorator.steps.common.prompt.get_location")
    def test_process_missing_template_param(
        self, mock_get_location: Any, mock_push_warning: Any
    ) -> None:
        """Test processing docstring with missing template param."""
        mock_get_location.return_value = "test_location"
        docstring = "Function with {param1} and {param2}"
        template_params = {"param1": "value1"}  # Missing param2
        result = process_docstring_template(docstring, template_params)
        assert result == docstring  # Should return original
        mock_push_warning.assert_called()

    @patch("SimpleLLMFunc.llm_decorator.steps.common.prompt.push_warning")
    @patch("SimpleLLMFunc.llm_decorator.steps.common.prompt.get_location")
    def test_process_invalid_template(
        self, mock_get_location: Any, mock_push_warning: Any
    ) -> None:
        """Test processing docstring with invalid template."""
        mock_get_location.return_value = "test_location"
        docstring = "Function with {invalid"
        template_params = {"param1": "value1"}
        result = process_docstring_template(docstring, template_params)
        assert result == docstring  # Should return original
        mock_push_warning.assert_called()


class TestExtractParameterTypeHints:
    """Tests for extract_parameter_type_hints function."""

    def test_extract_excluding_return(self) -> None:
        """Test extracting type hints excluding return."""
        type_hints = {
            "param1": str,
            "param2": int,
            "return": str,
        }
        result = extract_parameter_type_hints(type_hints)
        assert "param1" in result
        assert "param2" in result
        assert "return" not in result

    def test_extract_no_return(self) -> None:
        """Test extracting type hints when no return type."""
        type_hints = {
            "param1": str,
            "param2": int,
        }
        result = extract_parameter_type_hints(type_hints)
        assert len(result) == 2
        assert "param1" in result
        assert "param2" in result

    def test_extract_empty(self) -> None:
        """Test extracting from empty type hints."""
        result = extract_parameter_type_hints({})
        assert result == {}

