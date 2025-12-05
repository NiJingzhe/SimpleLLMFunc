"""Tests for base.post_process module."""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel

from SimpleLLMFunc.base.post_process import (
    _convert_from_xml,
    _convert_xml_to_pydantic,
    _convert_to_primitive_type,
    extract_content_from_response,
    extract_content_from_stream_response,
    process_response,
)
# Import internal function for testing
from SimpleLLMFunc.base import post_process


class TestExtractContentFromResponse:
    """Tests for extract_content_from_response function."""

    def test_extract_content_normal(self, mock_chat_completion: ChatCompletion) -> None:
        """Test extracting content from normal response."""
        result = extract_content_from_response(mock_chat_completion, "test_func")
        assert result == "Test response"

    def test_extract_content_empty(self) -> None:
        """Test extracting content from empty response."""
        message = ChatCompletionMessage(role="assistant", content="")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = extract_content_from_response(response, "test_func")
        assert result == ""

    def test_extract_content_none(self) -> None:
        """Test extracting content when content is None."""
        message = ChatCompletionMessage(role="assistant", content=None)
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = extract_content_from_response(response, "test_func")
        assert result == ""

    def test_extract_content_no_choices(self) -> None:
        """Test extracting content when response has no choices."""
        response = ChatCompletion(
            id="test",
            choices=[],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = extract_content_from_response(response, "test_func")
        assert result == ""

    def test_extract_content_invalid_response(self) -> None:
        """Test extracting content from invalid response."""
        result = extract_content_from_response({}, "test_func")
        assert result == ""


class TestExtractContentFromStreamResponse:
    """Tests for extract_content_from_stream_response function."""

    def test_extract_content_from_chunk(self, mock_chat_completion_chunk) -> None:
        """Test extracting content from stream chunk."""
        result = extract_content_from_stream_response(
            mock_chat_completion_chunk, "test_func"
        )
        assert result == "chunk"

    def test_extract_content_empty_chunk(self) -> None:
        """Test extracting content from empty chunk."""
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            Choice as ChunkChoice,
            ChoiceDelta,
        )

        delta = ChoiceDelta(content="", role="assistant")
        choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
        chunk = ChatCompletionChunk(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion.chunk",
        )
        result = extract_content_from_stream_response(chunk, "test_func")
        assert result == ""

    def test_extract_content_none_chunk(self) -> None:
        """Test extracting content when chunk content is None."""
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            Choice as ChunkChoice,
            ChoiceDelta,
        )

        delta = ChoiceDelta(content=None, role="assistant")
        choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
        chunk = ChatCompletionChunk(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion.chunk",
        )
        result = extract_content_from_stream_response(chunk, "test_func")
        assert result == ""

    def test_extract_content_no_chunk(self) -> None:
        """Test extracting content when chunk is None."""
        result = extract_content_from_stream_response(None, "test_func")
        assert result == ""


class TestConvertToPrimitiveType:
    """Tests for _convert_to_primitive_type function."""

    def test_convert_to_int(self) -> None:
        """Test converting to int."""
        result = _convert_to_primitive_type("123", int)
        assert result == 123
        assert isinstance(result, int)

    def test_convert_to_int_with_whitespace(self) -> None:
        """Test converting to int with whitespace."""
        result = _convert_to_primitive_type("  456  ", int)
        assert result == 456

    def test_convert_to_float(self) -> None:
        """Test converting to float."""
        result = _convert_to_primitive_type("123.45", float)
        assert result == 123.45
        assert isinstance(result, float)

    def test_convert_to_bool_true(self) -> None:
        """Test converting to bool (true cases)."""
        assert _convert_to_primitive_type("true", bool) is True
        assert _convert_to_primitive_type("True", bool) is True
        assert _convert_to_primitive_type("yes", bool) is True
        assert _convert_to_primitive_type("1", bool) is True

    def test_convert_to_bool_false(self) -> None:
        """Test converting to bool (false cases)."""
        assert _convert_to_primitive_type("false", bool) is False
        assert _convert_to_primitive_type("no", bool) is False
        assert _convert_to_primitive_type("0", bool) is False

    def test_convert_to_int_invalid(self) -> None:
        """Test converting invalid string to int."""
        with pytest.raises(ValueError):
            _convert_to_primitive_type("not_a_number", int)

    def test_convert_to_float_invalid(self) -> None:
        """Test converting invalid string to float."""
        with pytest.raises(ValueError):
            _convert_to_primitive_type("not_a_number", float)

    def test_convert_unsupported_type(self) -> None:
        """Test converting to unsupported type."""
        with pytest.raises(ValueError):
            _convert_to_primitive_type("test", str)  # str is not handled by this function


class TestExtractXmlContent:
    """Tests for _extract_xml_content function."""

    def test_extract_pure_xml(self) -> None:
        """Test extracting pure XML."""
        content = '<result><key>value</key></result>'
        result = post_process._extract_xml_content(content)
        assert result == '<result><key>value</key></result>'

    def test_extract_xml_from_code_block(self) -> None:
        """Test extracting XML from code block."""
        content = '```xml\n<result><key>value</key></result>\n```'
        result = post_process._extract_xml_content(content)
        assert result == '<result><key>value</key></result>'

    def test_extract_xml_with_whitespace(self) -> None:
        """Test extracting XML with whitespace."""
        content = '   <result><key>value</key></result>   '
        result = post_process._extract_xml_content(content)
        assert result == '<result><key>value</key></result>'

    def test_extract_xml_with_code_block_whitespace(self) -> None:
        """Test extracting XML from code block with whitespace."""
        content = '```xml\n  <result><key>value</key></result>  \n```'
        result = post_process._extract_xml_content(content)
        assert result == '<result><key>value</key></result>'

    def test_extract_xml_with_multiline_code_block(self) -> None:
        """Test extracting XML from multiline code block."""
        content = '''```xml
<result>
  <key>value</key>
</result>
```'''
        result = post_process._extract_xml_content(content)
        assert '<result>' in result
        assert '<key>value</key>' in result


class TestConvertFromXml:
    """Tests for _convert_from_xml function."""

    def test_convert_valid_xml(self) -> None:
        """Test converting valid XML string."""
        xml_str = '<result><key>value</key><number>123</number></result>'
        result = _convert_from_xml(xml_str, "test_func")
        assert result == {"key": "value", "number": 123}
        assert isinstance(result["number"], int)

    def test_convert_xml_with_code_block(self) -> None:
        """Test converting XML wrapped in code block."""
        xml_str = '```xml\n<result><key>value</key></result>\n```'
        result = _convert_from_xml(xml_str, "test_func")
        assert result == {"key": "value"}

    def test_convert_xml_with_list(self) -> None:
        """Test converting XML with list items."""
        xml_str = '<result><items><item>item1</item><item>item2</item></items></result>'
        result = _convert_from_xml(xml_str, "test_func")
        assert "items" in result
        assert isinstance(result.get("items"), list)

    def test_convert_invalid_xml(self) -> None:
        """Test converting invalid XML."""
        with pytest.raises(ValueError):
            _convert_from_xml("not xml", "test_func")

    def test_convert_empty_string(self) -> None:
        """Test converting empty string."""
        with pytest.raises(ValueError):
            _convert_from_xml("", "test_func")

    def test_convert_xml_with_nested_structure(self) -> None:
        """Test converting XML with nested structure."""
        xml_str = '<result><outer><inner>value</inner></outer></result>'
        result = _convert_from_xml(xml_str, "test_func")
        assert "outer" in result
        assert isinstance(result["outer"], dict)
        assert result["outer"]["inner"] == "value"

    def test_convert_xml_with_numeric_types(self) -> None:
        """Test converting XML with numeric types."""
        xml_str = '<result><int_val>42</int_val><float_val>3.14</float_val><bool_val>true</bool_val></result>'
        result = _convert_from_xml(xml_str, "test_func")
        assert result["int_val"] == 42
        assert isinstance(result["int_val"], int)
        assert result["float_val"] == 3.14
        assert isinstance(result["float_val"], float)
        assert result["bool_val"] is True

    def test_convert_xml_with_pure_xml_start(self) -> None:
        """Test converting XML that starts with <."""
        xml_str = '<result><key>value</key></result>'
        result = _convert_from_xml(xml_str, "test_func")
        assert result == {"key": "value"}


class TestConvertXmlToPydantic:
    """Tests for _convert_xml_to_pydantic function."""

    def test_convert_valid_xml_to_model(self, sample_pydantic_model) -> None:
        """Test converting valid XML to Pydantic model."""
        xml_str = '<SampleModel><name>John</name><age>30</age><email>john@example.com</email></SampleModel>'
        result = _convert_xml_to_pydantic(xml_str, sample_pydantic_model, "test_func")
        assert isinstance(result, sample_pydantic_model)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_convert_xml_with_code_block_to_model(self, sample_pydantic_model) -> None:
        """Test converting XML in code block to model."""
        xml_str = '```xml\n<SampleModel><name>Jane</name><age>25</age></SampleModel>\n```'
        result = _convert_xml_to_pydantic(xml_str, sample_pydantic_model, "test_func")
        assert result.name == "Jane"
        assert result.age == 25

    def test_convert_xml_with_result_wrapper(self, sample_pydantic_model) -> None:
        """Test converting XML with result wrapper."""
        xml_str = '<result><SampleModel><name>Bob</name><age>35</age></SampleModel></result>'
        result = _convert_xml_to_pydantic(xml_str, sample_pydantic_model, "test_func")
        assert result.name == "Bob"
        assert result.age == 35

    def test_convert_empty_string_to_model(self, sample_pydantic_model) -> None:
        """Test converting empty string to model."""
        with pytest.raises(ValueError):
            _convert_xml_to_pydantic("", sample_pydantic_model, "test_func")

    def test_convert_invalid_xml_to_model(self, sample_pydantic_model) -> None:
        """Test converting invalid XML to model."""
        with pytest.raises(ValueError):
            _convert_xml_to_pydantic("not xml", sample_pydantic_model, "test_func")

    def test_convert_xml_with_nested_model(self) -> None:
        """Test converting XML with nested Pydantic model."""

        class InnerModel(BaseModel):
            value: str

        class OuterModel(BaseModel):
            inner: InnerModel

        xml_str = '<OuterModel><inner><value>test</value></inner></OuterModel>'
        result = _convert_xml_to_pydantic(xml_str, OuterModel, "test_func")
        assert isinstance(result, OuterModel)
        assert result.inner.value == "test"

    def test_convert_xml_with_list_in_model(self) -> None:
        """Test converting XML with list in Pydantic model."""
        from typing import List

        class ListModel(BaseModel):
            items: List[str]

        xml_str = '<ListModel><items><item>item1</item><item>item2</item></items></ListModel>'
        result = _convert_xml_to_pydantic(xml_str, ListModel, "test_func")
        assert isinstance(result, ListModel)
        assert len(result.items) == 2
        assert "item1" in result.items

    def test_convert_xml_with_missing_optional_fields(self, sample_pydantic_model) -> None:
        """Test converting XML with missing optional fields."""
        xml_str = '<SampleModel><name>John</name><age>30</age></SampleModel>'
        result = _convert_xml_to_pydantic(xml_str, sample_pydantic_model, "test_func")
        assert result.name == "John"
        assert result.age == 30
        assert result.email is None


class TestProcessResponse:
    """Tests for process_response function."""

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_str(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as str."""
        mock_get_context.return_value = "test_func"
        result = process_response(mock_chat_completion, str)
        assert result == "Test response"
        assert isinstance(result, str)

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_int(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as int."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(role="assistant", content="123")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, int)
        assert result == 123
        assert isinstance(result, int)

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_float(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as float."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(role="assistant", content="123.45")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, float)
        assert result == 123.45
        assert isinstance(result, float)

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_bool(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as bool."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(role="assistant", content="true")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, bool)
        assert result is True

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_dict(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as dict."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(
            role="assistant", content='<result><key>value</key><number>123</number></result>'
        )
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, dict)
        assert result == {"key": "value", "number": 123}
        assert isinstance(result, dict)

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_pydantic_model(
        self,
        mock_get_context: Any,
        mock_chat_completion: ChatCompletion,
        sample_pydantic_model,
    ) -> None:
        """Test processing response as Pydantic model."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(
            role="assistant", content='<SampleModel><name>Alice</name><age>28</age></SampleModel>'
        )
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, sample_pydantic_model)
        assert isinstance(result, sample_pydantic_model)
        assert result.name == "Alice"
        assert result.age == 28

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_none_type(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response with None return type."""
        mock_get_context.return_value = "test_func"
        result = process_response(mock_chat_completion, None)
        assert result == "Test response"

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_empty_content(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response with empty content."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(role="assistant", content="")
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, str)
        assert result == ""

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_list_str(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as List[str]."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(
            role="assistant", content='<result><item>item1</item><item>item2</item><item>item3</item></result>'
        )
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, List[str])
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == ["item1", "item2", "item3"]

    @patch("SimpleLLMFunc.base.post_process.get_current_context_attribute")
    def test_process_response_list_int(
        self, mock_get_context: Any, mock_chat_completion: ChatCompletion
    ) -> None:
        """Test processing response as List[int]."""
        mock_get_context.return_value = "test_func"
        message = ChatCompletionMessage(
            role="assistant", content='<result><item>1</item><item>2</item><item>3</item></result>'
        )
        choice = Choice(finish_reason="stop", index=0, message=message)
        response = ChatCompletion(
            id="test",
            choices=[choice],
            created=0,
            model="test",
            object="chat.completion",
        )
        result = process_response(response, List[int])
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [1, 2, 3]
        assert all(isinstance(x, int) for x in result)

