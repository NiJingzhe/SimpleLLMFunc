"""Tests for base.type_resolve.description module."""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel, ConfigDict

from SimpleLLMFunc.base.type_resolve.description import (
    build_type_description_xml,
    describe_pydantic_model,
    generate_example_xml,
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
            model_config = ConfigDict(
                json_schema_extra={
                    "properties": {
                        "age": {"minimum": 0, "maximum": 150},
                        "score": {"minimum": 0.0, "maximum": 100.0},
                    }
                }
            )
            
            age: int
            score: float

        result = describe_pydantic_model(ConstrainedModel)
        assert "ConstrainedModel" in result


class TestBuildTypeDescriptionXml:
    """Tests for build_type_description_xml function."""

    def test_simple_model(self, sample_pydantic_model) -> None:
        """Test building XML schema description for simple model."""
        result = build_type_description_xml(sample_pydantic_model)
        assert "SampleModel" in result
        assert "name" in result or "<name>" in result

    def test_model_with_list(self) -> None:
        """Test building XML schema for model with list."""

        class ListModel(BaseModel):
            items: List[str]

        result = build_type_description_xml(ListModel)
        assert "ListModel" in result
        assert "items" in result or "<items>" in result

    def test_nested_model(self) -> None:
        """Test building XML schema for nested model."""

        class InnerModel(BaseModel):
            value: str

        class OuterModel(BaseModel):
            inner: InnerModel

        result = build_type_description_xml(OuterModel)
        assert "OuterModel" in result
        assert "InnerModel" in result or "inner" in result

    def test_primitive_type(self) -> None:
        """Test building XML schema for primitive type."""
        result = build_type_description_xml(str)
        assert "string" in result.lower() or "str" in result.lower()

    def test_dict_type(self) -> None:
        """Test building XML schema for Dict type."""
        result = build_type_description_xml(Dict[str, int])
        assert "dict" in result.lower() or "Dict" in result

    def test_list_type(self) -> None:
        """Test building XML schema for List type."""
        result = build_type_description_xml(List[str])
        assert "list" in result.lower() or "List" in result


class TestGenerateExampleXml:
    """Tests for generate_example_xml function."""

    def test_simple_model(self, sample_pydantic_model) -> None:
        """Test generating XML example for simple model."""
        result = generate_example_xml(sample_pydantic_model)
        assert result.startswith("<")
        assert "SampleModel" in result
        assert "</SampleModel>" in result

    def test_model_with_list(self) -> None:
        """Test generating XML example for model with list."""

        class ListModel(BaseModel):
            items: List[str]

        result = generate_example_xml(ListModel)
        assert result.startswith("<")
        assert "ListModel" in result
        assert "<item>" in result or "items" in result

    def test_nested_model(self) -> None:
        """Test generating XML example for nested model."""

        class InnerModel(BaseModel):
            value: str

        class OuterModel(BaseModel):
            inner: InnerModel

        result = generate_example_xml(OuterModel)
        assert result.startswith("<")
        assert "OuterModel" in result
        assert "inner" in result

    def test_primitive_type(self) -> None:
        """Test generating XML example for primitive type."""
        result = generate_example_xml(str)
        assert result.startswith("<")
        assert result.endswith(">")
        assert "example" in result.lower()

    def test_dict_type(self) -> None:
        """Test generating XML example for Dict type."""
        result = generate_example_xml(Dict[str, int])
        assert result.startswith("<")
        assert "key" in result.lower() or "value" in result.lower()

    def test_list_type(self) -> None:
        """Test generating XML example for List type."""
        result = generate_example_xml(List[str])
        assert result.startswith("<")
        assert "<item>" in result.lower() or "item" in result.lower()

    def test_optional_type(self) -> None:
        """Test generating XML example for Optional type."""

        class OptionalModel(BaseModel):
            name: Optional[str] = None

        result = generate_example_xml(OptionalModel)
        assert result.startswith("<")
        assert "OptionalModel" in result

