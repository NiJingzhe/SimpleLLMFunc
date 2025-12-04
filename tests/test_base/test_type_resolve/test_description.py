"""Tests for base.type_resolve.description module."""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel

from SimpleLLMFunc.base.type_resolve.description import (
    describe_pydantic_model,
    get_detailed_type_description,
)


class TestGetDetailedTypeDescription:
    """Tests for get_detailed_type_description function."""

    def test_primitive_types(self) -> None:
        """Test primitive type descriptions."""
        assert get_detailed_type_description(str) == "<class 'str'>"
        assert get_detailed_type_description(int) == "<class 'int'>"
        assert get_detailed_type_description(float) == "<class 'float'>"
        assert get_detailed_type_description(bool) == "<class 'bool'>"

    def test_none_type(self) -> None:
        """Test None type description."""
        assert get_detailed_type_description(None) == "未知类型"

    def test_list_type(self) -> None:
        """Test List type description."""
        result = get_detailed_type_description(List[str])
        assert "List" in result
        assert "str" in result

    def test_dict_type(self) -> None:
        """Test Dict type description."""
        result = get_detailed_type_description(Dict[str, int])
        assert "Dict" in result
        assert "str" in result
        assert "int" in result

    def test_nested_types(self) -> None:
        """Test nested type descriptions."""
        result = get_detailed_type_description(List[Dict[str, int]])
        assert "List" in result
        assert "Dict" in result

    def test_pydantic_model(self, sample_pydantic_model) -> None:
        """Test Pydantic model type description."""
        result = get_detailed_type_description(sample_pydantic_model)
        assert "SampleModel" in result or "Pydantic" in result.lower()


class TestDescribePydanticModel:
    """Tests for describe_pydantic_model function."""

    def test_simple_model(self, sample_pydantic_model) -> None:
        """Test describing a simple Pydantic model."""
        result = describe_pydantic_model(sample_pydantic_model)
        assert "SampleModel" in result
        assert "name" in result
        assert "age" in result
        assert "email" in result

    def test_model_with_required_fields(self) -> None:
        """Test describing model with required fields."""

        class RequiredModel(BaseModel):
            name: str
            age: int

        result = describe_pydantic_model(RequiredModel)
        assert "RequiredModel" in result
        assert "name" in result
        assert "age" in result
        assert "必填" in result

    def test_model_with_optional_fields(self) -> None:
        """Test describing model with optional fields."""

        class OptionalModel(BaseModel):
            name: str
            email: Optional[str] = None

        result = describe_pydantic_model(OptionalModel)
        assert "OptionalModel" in result
        assert "可选" in result

    def test_model_with_constraints(self) -> None:
        """Test describing model with field constraints."""

        class ConstrainedModel(BaseModel):
            age: int
            score: float

            class Config:
                json_schema_extra = {
                    "properties": {
                        "age": {"minimum": 0, "maximum": 150},
                        "score": {"minimum": 0.0, "maximum": 100.0},
                    }
                }

        result = describe_pydantic_model(ConstrainedModel)
        assert "ConstrainedModel" in result

