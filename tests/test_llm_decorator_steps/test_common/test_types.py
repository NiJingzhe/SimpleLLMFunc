"""Tests for llm_decorator.steps.common.types module."""

from __future__ import annotations

import inspect

import pytest

from SimpleLLMFunc.llm_decorator.steps.common.types import FunctionSignature


class TestFunctionSignature:
    """Tests for FunctionSignature NamedTuple."""

    def test_create_signature(self, sample_bound_args: inspect.BoundArguments) -> None:
        """Test creating FunctionSignature."""
        sig = inspect.signature(lambda x: x)
        type_hints = {"x": str, "return": str}
        
        signature = FunctionSignature(
            func_name="test_func",
            trace_id="trace_123",
            bound_args=sample_bound_args,
            signature=sig,
            type_hints=type_hints,
            return_type=str,
            docstring="Test function.",
        )
        
        assert signature.func_name == "test_func"
        assert signature.trace_id == "trace_123"
        assert signature.return_type == str
        assert signature.docstring == "Test function."
        assert signature.type_hints == type_hints

    def test_signature_immutability(self, sample_bound_args: inspect.BoundArguments) -> None:
        """Test that FunctionSignature is immutable."""
        sig = inspect.signature(lambda x: x)
        type_hints = {"x": str, "return": str}
        
        signature = FunctionSignature(
            func_name="test_func",
            trace_id="trace_123",
            bound_args=sample_bound_args,
            signature=sig,
            type_hints=type_hints,
            return_type=str,
            docstring="Test function.",
        )
        
        # NamedTuple is immutable, should raise AttributeError
        with pytest.raises(AttributeError):
            signature.func_name = "new_name"

