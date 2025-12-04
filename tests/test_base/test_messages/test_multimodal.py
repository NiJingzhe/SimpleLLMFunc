"""Tests for base.messages.multimodal module."""

from __future__ import annotations

from typing import List, Optional, Union

from SimpleLLMFunc.base.messages.multimodal import (
    build_multimodal_content,
    create_image_path_content,
    create_image_url_content,
    create_text_content,
    handle_union_type,
    parse_multimodal_parameter,
)
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text


class TestCreateTextContent:
    """Tests for create_text_content function."""

    def test_create_with_string(self) -> None:
        """Test creating text content from string."""
        result = create_text_content("Hello", "param1")
        assert result == {"type": "text", "text": "param1: Hello"}

    def test_create_with_text_object(self, text_content: Text) -> None:
        """Test creating text content from Text object."""
        result = create_text_content(text_content, "param1")
        assert result["type"] == "text"
        assert "param1" in result["text"]

    def test_create_with_number(self) -> None:
        """Test creating text content from number."""
        result = create_text_content(123, "param1")
        assert result == {"type": "text", "text": "param1: 123"}


class TestCreateImageUrlContent:
    """Tests for create_image_url_content function."""

    def test_create_with_img_url(self, img_url: ImgUrl) -> None:
        """Test creating image URL content from ImgUrl."""
        result = create_image_url_content(img_url, "param1")
        assert result["type"] == "image_url"
        assert "image_url" in result
        assert result["image_url"]["url"] == img_url.url

    def test_create_with_string_url(self) -> None:
        """Test creating image URL content from string."""
        result = create_image_url_content("https://example.com/image.png", "param1")
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/image.png"

    def test_create_with_none(self) -> None:
        """Test creating image URL content from None."""
        result = create_image_url_content(None, "param1")
        assert result["type"] == "text"
        assert "None" in result["text"]


class TestCreateImagePathContent:
    """Tests for create_image_path_content function."""

    def test_create_with_img_path(self, img_path: ImgPath) -> None:
        """Test creating image path content from ImgPath."""
        result = create_image_path_content(img_path, "param1")
        assert result["type"] == "image_url"
        assert "image_url" in result
        assert "data:" in result["image_url"]["url"]
        assert "base64" in result["image_url"]["url"]

    def test_create_with_none(self) -> None:
        """Test creating image path content from None."""
        result = create_image_path_content(None, "param1")
        assert result["type"] == "text"
        assert "None" in result["text"]


class TestParseMultimodalParameter:
    """Tests for parse_multimodal_parameter function."""

    def test_parse_text(self, text_content: Text) -> None:
        """Test parsing Text parameter."""
        result = parse_multimodal_parameter(text_content, Text, "param1")
        assert len(result) == 1
        assert result[0]["type"] == "text"

    def test_parse_img_url(self, img_url: ImgUrl) -> None:
        """Test parsing ImgUrl parameter."""
        result = parse_multimodal_parameter(img_url, ImgUrl, "param1")
        assert len(result) == 1
        assert result[0]["type"] == "image_url"

    def test_parse_img_path(self, img_path: ImgPath) -> None:
        """Test parsing ImgPath parameter."""
        result = parse_multimodal_parameter(img_path, ImgPath, "param1")
        assert len(result) == 1
        assert result[0]["type"] == "image_url"

    def test_parse_string(self) -> None:
        """Test parsing string parameter."""
        result = parse_multimodal_parameter("test", str, "param1")
        assert len(result) == 1
        assert result[0]["type"] == "text"

    def test_parse_list_of_text(self, text_content: Text) -> None:
        """Test parsing List[Text] parameter."""
        result = parse_multimodal_parameter([text_content], List[Text], "param1")
        assert len(result) >= 1

    def test_parse_union_type(self, text_content: Text) -> None:
        """Test parsing Union type parameter."""
        result = parse_multimodal_parameter(text_content, Union[str, Text], "param1")
        assert len(result) >= 1

    def test_parse_none(self) -> None:
        """Test parsing None parameter."""
        result = parse_multimodal_parameter(None, Optional[str], "param1")
        assert result == []


class TestHandleUnionType:
    """Tests for handle_union_type function."""

    def test_handle_text_in_union(self, text_content: Text) -> None:
        """Test handling Text in Union type."""
        result = handle_union_type(text_content, (str, Text), "param1")
        assert len(result) >= 1

    def test_handle_img_url_in_union(self, img_url: ImgUrl) -> None:
        """Test handling ImgUrl in Union type."""
        result = handle_union_type(img_url, (str, ImgUrl), "param1")
        assert len(result) >= 1

    def test_handle_list_in_union(self, text_content: Text) -> None:
        """Test handling list in Union type."""
        result = handle_union_type([text_content], (str, Text), "param1")
        assert len(result) >= 1


class TestBuildMultimodalContent:
    """Tests for build_multimodal_content function."""

    def test_build_with_text(self, text_content: Text) -> None:
        """Test building multimodal content with Text."""
        arguments = {"text": text_content}
        type_hints = {"text": Text}
        result = build_multimodal_content(arguments, type_hints)
        assert len(result) >= 1

    def test_build_with_multiple_params(self, text_content: Text, img_url: ImgUrl) -> None:
        """Test building multimodal content with multiple parameters."""
        arguments = {"text": text_content, "image": img_url}
        type_hints = {"text": Text, "image": ImgUrl}
        result = build_multimodal_content(arguments, type_hints)
        assert len(result) >= 2

    def test_build_with_exclude_params(self, text_content: Text) -> None:
        """Test building multimodal content with excluded parameters."""
        arguments = {"text": text_content, "history": []}
        type_hints = {"text": Text, "history": List}
        result = build_multimodal_content(
            arguments, type_hints, exclude_params=["history"]
        )
        # Should only include text, not history
        assert len(result) >= 1

    def test_build_without_type_hints(self) -> None:
        """Test building multimodal content without type hints."""
        arguments = {"text": "plain text"}
        type_hints = {}
        result = build_multimodal_content(arguments, type_hints)
        assert len(result) >= 1
        # Should default to text content
        assert result[0]["type"] == "text"

