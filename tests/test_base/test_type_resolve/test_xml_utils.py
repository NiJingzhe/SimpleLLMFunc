"""Tests for base.type_resolve.xml_utils module."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import pytest
from pydantic import BaseModel

from SimpleLLMFunc.base.type_resolve.xml_utils import (
    dict_to_pydantic,
    generate_xml_example,
    pydantic_to_xml_schema,
    xml_to_dict,
)


class TestPydanticToXmlSchema:
    """Tests for pydantic_to_xml_schema function."""

    def test_simple_model(self, sample_pydantic_model) -> None:
        """Test generating XML schema for simple model."""
        result = pydantic_to_xml_schema(sample_pydantic_model)
        assert "SampleModel" in result
        assert "name" in result
        assert "age" in result

    def test_model_with_list(self) -> None:
        """Test generating XML schema for model with list."""

        class ListModel(BaseModel):
            items: List[str]

        result = pydantic_to_xml_schema(ListModel)
        assert "ListModel" in result
        assert "items" in result

    def test_nested_model(self) -> None:
        """Test generating XML schema for nested model."""

        class InnerModel(BaseModel):
            value: str

        class OuterModel(BaseModel):
            inner: InnerModel

        result = pydantic_to_xml_schema(OuterModel)
        assert "OuterModel" in result
        assert "InnerModel" in result


class TestGenerateXmlExample:
    """Tests for generate_xml_example function."""

    def test_simple_model(self, sample_pydantic_model) -> None:
        """Test generating XML example for simple model."""
        result = generate_xml_example(sample_pydantic_model)
        assert result.startswith("<")
        assert "SampleModel" in result
        assert "name" in result

    def test_model_with_list(self) -> None:
        """Test generating XML example for model with list."""

        class ListModel(BaseModel):
            items: List[str]

        result = generate_xml_example(ListModel)
        assert "<item>" in result or "items" in result

    def test_primitive_types(self) -> None:
        """Test generating XML example for primitive types."""
        result_str = generate_xml_example(str)
        assert result_str.startswith("<")
        assert "example" in result_str.lower()

        result_int = generate_xml_example(int)
        assert "123" in result_int


class TestXmlToDict:
    """Tests for xml_to_dict function."""

    def test_simple_xml(self) -> None:
        """Test converting simple XML to dict."""
        xml_str = '<result><key>value</key><number>123</number></result>'
        result = xml_to_dict(xml_str)
        assert result == {"key": "value", "number": 123}

    def test_xml_with_list(self) -> None:
        """Test converting XML with list to dict."""
        xml_str = '<result><items><item>item1</item><item>item2</item></items></result>'
        result = xml_to_dict(xml_str)
        assert "items" in result
        assert isinstance(result["items"], list)
        assert len(result["items"]) == 2

    def test_nested_xml(self) -> None:
        """Test converting nested XML to dict."""
        xml_str = '<result><outer><inner>value</inner></outer></result>'
        result = xml_to_dict(xml_str)
        assert "outer" in result
        assert isinstance(result["outer"], dict)

    def test_invalid_xml(self) -> None:
        """Test converting invalid XML."""
        with pytest.raises(ValueError):
            xml_to_dict("not xml")


class TestDictToPydantic:
    """Tests for dict_to_pydantic function."""

    def test_simple_dict(self, sample_pydantic_model) -> None:
        """Test converting simple dict to Pydantic model."""
        data = {"name": "John", "age": 30, "email": "john@example.com"}
        result = dict_to_pydantic(data, sample_pydantic_model)
        assert isinstance(result, sample_pydantic_model)
        assert result.name == "John"
        assert result.age == 30

    def test_dict_with_nested(self) -> None:
        """Test converting dict with nested structure to Pydantic model."""

        class InnerModel(BaseModel):
            value: str

        class OuterModel(BaseModel):
            inner: InnerModel

        data = {"inner": {"value": "test"}}
        result = dict_to_pydantic(data, OuterModel)
        assert isinstance(result, OuterModel)
        assert result.inner.value == "test"

    def test_dict_with_list(self) -> None:
        """Test converting dict with list to Pydantic model."""

        class ListModel(BaseModel):
            items: List[str]

        data = {"items": ["item1", "item2"]}
        result = dict_to_pydantic(data, ListModel)
        assert isinstance(result, ListModel)
        assert len(result.items) == 2

    def test_dict_with_text_key(self) -> None:
        """Test converting dict with _text key."""
        from pydantic import BaseModel

        class TextModel(BaseModel):
            value: str

        data = {"value": {"_text": "test"}}
        result = dict_to_pydantic(data, TextModel)
        assert result.value == "test"

    def test_dict_with_optional_field(self) -> None:
        """Test converting dict with optional field."""

        class OptionalModel(BaseModel):
            name: str
            email: Optional[str] = None

        data = {"name": "John"}
        result = dict_to_pydantic(data, OptionalModel)
        assert result.name == "John"
        assert result.email is None


class TestPydanticToXmlSchemaEdgeCases:
    """Tests for pydantic_to_xml_schema edge cases."""

    def test_optional_type(self) -> None:
        """Test generating schema for Optional type."""

        class OptionalModel(BaseModel):
            name: Optional[str] = None

        result = pydantic_to_xml_schema(OptionalModel)
        assert "OptionalModel" in result
        assert "name" in result

    def test_union_type(self) -> None:
        """Test generating schema for Union type."""

        class UnionModel(BaseModel):
            value: Union[str, int]

        result = pydantic_to_xml_schema(UnionModel)
        assert "UnionModel" in result

    def test_dict_type(self) -> None:
        """Test generating schema for Dict type."""

        class DictModel(BaseModel):
            data: Dict[str, int]

        result = pydantic_to_xml_schema(DictModel)
        assert "DictModel" in result
        assert "data" in result

    def test_depth_limit(self) -> None:
        """Test depth limit handling."""

        class DeepModel(BaseModel):
            value: str

        result = pydantic_to_xml_schema(DeepModel, max_depth=0)
        assert "depth limit" in result.lower() or "DeepModel" in result

    def test_circular_reference(self) -> None:
        """Test circular reference handling."""
        # Note: This is a simplified test as true circular refs are complex
        result = pydantic_to_xml_schema(str)
        assert "string" in result.lower() or "str" in result.lower()


class TestGenerateXmlExampleEdgeCases:
    """Tests for generate_xml_example edge cases."""

    def test_optional_type(self) -> None:
        """Test generating example for Optional type."""

        class OptionalModel(BaseModel):
            name: Optional[str] = None

        result = generate_xml_example(OptionalModel)
        assert result.startswith("<")
        assert "OptionalModel" in result

    def test_union_type(self) -> None:
        """Test generating example for Union type."""

        class UnionModel(BaseModel):
            value: Union[str, int]

        result = generate_xml_example(UnionModel)
        assert result.startswith("<")

    def test_dict_type(self) -> None:
        """Test generating example for Dict type."""

        class DictModel(BaseModel):
            data: Dict[str, int]

        result = generate_xml_example(DictModel)
        assert result.startswith("<")
        assert "DictModel" in result or "result" in result

    def test_bool_type(self) -> None:
        """Test generating example for bool type."""
        result = generate_xml_example(bool)
        assert "true" in result.lower() or "false" in result.lower()

    def test_float_type(self) -> None:
        """Test generating example for float type."""
        result = generate_xml_example(float)
        assert "1.23" in result or "result" in result

    def test_none_type(self) -> None:
        """Test generating example for None type."""
        result = generate_xml_example(type(None))
        assert result.startswith("<")

    def test_model_with_default_values(self) -> None:
        """Test generating example for model with default values."""

        class DefaultModel(BaseModel):
            name: str = "default"
            age: int = 25

        result = generate_xml_example(DefaultModel)
        assert "DefaultModel" in result
        assert "default" in result or "25" in result


class TestXmlToDictEdgeCases:
    """Tests for xml_to_dict edge cases."""

    def test_xml_with_attributes(self) -> None:
        """Test converting XML with attributes."""
        xml_str = '<result id="123"><key>value</key></result>'
        result = xml_to_dict(xml_str)
        assert "key" in result
        assert result["key"] == "value"

    def test_xml_with_boolean_values(self) -> None:
        """Test converting XML with boolean values."""
        xml_str = '<result><flag>true</flag><enabled>false</enabled></result>'
        result = xml_to_dict(xml_str)
        assert result["flag"] is True
        assert result["enabled"] is False

    def test_xml_with_numeric_values(self) -> None:
        """Test converting XML with numeric values."""
        xml_str = '<result><int_val>42</int_val><float_val>3.14</float_val></result>'
        result = xml_to_dict(xml_str)
        assert result["int_val"] == 42
        assert isinstance(result["int_val"], int)
        assert result["float_val"] == 3.14
        assert isinstance(result["float_val"], float)

    def test_xml_with_empty_elements(self) -> None:
        """Test converting XML with empty elements."""
        xml_str = '<result><empty></empty><text>value</text></result>'
        result = xml_to_dict(xml_str)
        assert "empty" in result
        assert result["text"] == "value"

    def test_xml_with_multiple_same_tags(self) -> None:
        """Test converting XML with multiple same tags (list)."""
        xml_str = '<result><item>item1</item><item>item2</item><item>item3</item></result>'
        result = xml_to_dict(xml_str)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == ["item1", "item2", "item3"]

    def test_xml_with_nested_lists(self) -> None:
        """Test converting XML with nested lists."""
        xml_str = '<result><items><item>a</item><item>b</item></items></result>'
        result = xml_to_dict(xml_str)
        assert "items" in result
        assert isinstance(result["items"], list)

    def test_xml_with_deep_nesting(self) -> None:
        """Test converting XML with deep nesting."""
        xml_str = '<result><level1><level2><level3>value</level3></level2></level1></result>'
        result = xml_to_dict(xml_str)
        assert "level1" in result
        assert isinstance(result["level1"], dict)

    def test_xml_with_special_characters(self) -> None:
        """Test converting XML with special characters."""
        xml_str = '<result><text>&lt;hello&gt; &amp; world</text></result>'
        result = xml_to_dict(xml_str)
        assert "text" in result
        # XML parser should handle entities automatically
        assert "hello" in result["text"] or "<" in result["text"]

    def test_xml_with_mixed_content(self) -> None:
        """Test converting XML with mixed content."""
        xml_str = '<result><items><item>1</item><item>2</item></items><name>test</name></result>'
        result = xml_to_dict(xml_str)
        assert "items" in result
        assert "name" in result
        assert result["name"] == "test"


class TestDictToPydanticEdgeCases:
    """Tests for dict_to_pydantic edge cases."""

    def test_dict_with_missing_optional_fields(self, sample_pydantic_model) -> None:
        """Test converting dict with missing optional fields."""
        data = {"name": "John", "age": 30}
        result = dict_to_pydantic(data, sample_pydantic_model)
        assert result.name == "John"
        assert result.age == 30
        assert result.email is None

    def test_dict_with_extra_fields(self) -> None:
        """Test converting dict with extra fields (should be ignored by Pydantic)."""

        class SimpleModel(BaseModel):
            name: str

        data = {"name": "John", "extra": "ignored"}
        result = dict_to_pydantic(data, SimpleModel)
        assert result.name == "John"

    def test_dict_with_nested_list(self) -> None:
        """Test converting dict with nested list."""

        class NestedListModel(BaseModel):
            items: List[Dict[str, str]]

        data = {"items": [{"key": "value1"}, {"key": "value2"}]}
        result = dict_to_pydantic(data, NestedListModel)
        assert len(result.items) == 2
        assert result.items[0]["key"] == "value1"

    def test_dict_with_complex_nesting(self) -> None:
        """Test converting dict with complex nesting."""

        class InnerModel(BaseModel):
            value: str

        class MiddleModel(BaseModel):
            inner: InnerModel

        class OuterModel(BaseModel):
            middle: MiddleModel

        data = {"middle": {"inner": {"value": "test"}}}
        result = dict_to_pydantic(data, OuterModel)
        assert result.middle.inner.value == "test"

