"""Tests for base.type_resolve.multimodal module."""

from __future__ import annotations

from typing import List, Optional, Union

import pytest

from SimpleLLMFunc.base.type_resolve.multimodal import (
    has_multimodal_content,
    is_multimodal_type,
)
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text


class TestIsMultimodalType:
    """Tests for is_multimodal_type function."""

    def test_text_instance(self, text_content: Text) -> None:
        """Test Text instance detection."""
        assert is_multimodal_type(text_content, Text) is True

    def test_img_url_instance(self, img_url: ImgUrl) -> None:
        """Test ImgUrl instance detection."""
        assert is_multimodal_type(img_url, ImgUrl) is True

    def test_img_path_instance(self, img_path: ImgPath) -> None:
        """Test ImgPath instance detection."""
        assert is_multimodal_type(img_path, ImgPath) is True

    def test_text_annotation(self, text_content: Text) -> None:
        """Test Text annotation detection."""
        assert is_multimodal_type(text_content, Text) is True

    def test_img_url_annotation(self, img_url: ImgUrl) -> None:
        """Test ImgUrl annotation detection."""
        assert is_multimodal_type(img_url, ImgUrl) is True

    def test_img_path_annotation(self, img_path: ImgPath) -> None:
        """Test ImgPath annotation detection."""
        assert is_multimodal_type(img_path, ImgPath) is True

    def test_union_type_with_multimodal(self, text_content: Text) -> None:
        """Test Union type containing multimodal."""
        assert is_multimodal_type(text_content, Union[str, Text]) is True

    def test_union_type_without_multimodal(self) -> None:
        """Test Union type without multimodal."""
        assert is_multimodal_type("test", Union[str, int]) is False

    def test_list_of_multimodal_types(self) -> None:
        """Test List of multimodal types."""
        assert is_multimodal_type([], List[Text]) is True
        assert is_multimodal_type([], List[ImgUrl]) is True
        assert is_multimodal_type([], List[ImgPath]) is True

    def test_list_with_multimodal_items(self, text_content: Text) -> None:
        """Test list containing multimodal items."""
        assert is_multimodal_type([text_content], List[str]) is True

    def test_non_multimodal_type(self) -> None:
        """Test non-multimodal type."""
        assert is_multimodal_type("test", str) is False
        assert is_multimodal_type(123, int) is False

    def test_optional_multimodal(self, text_content: Text) -> None:
        """Test Optional multimodal type."""
        assert is_multimodal_type(text_content, Optional[Text]) is True


class TestHasMultimodalContent:
    """Tests for has_multimodal_content function."""

    def test_has_text_content(self, text_content: Text) -> None:
        """Test detecting Text content."""
        arguments = {"text": text_content}
        type_hints = {"text": Text}
        assert has_multimodal_content(arguments, type_hints) is True

    def test_has_img_url_content(self, img_url: ImgUrl) -> None:
        """Test detecting ImgUrl content."""
        arguments = {"image": img_url}
        type_hints = {"image": ImgUrl}
        assert has_multimodal_content(arguments, type_hints) is True

    def test_has_img_path_content(self, img_path: ImgPath) -> None:
        """Test detecting ImgPath content."""
        arguments = {"image": img_path}
        type_hints = {"image": ImgPath}
        assert has_multimodal_content(arguments, type_hints) is True

    def test_no_multimodal_content(self) -> None:
        """Test detecting no multimodal content."""
        arguments = {"text": "plain text", "number": 123}
        type_hints = {"text": str, "number": int}
        assert has_multimodal_content(arguments, type_hints) is False

    def test_exclude_params(self, text_content: Text) -> None:
        """Test excluding parameters from check."""
        arguments = {"text": text_content, "history": []}
        type_hints = {"text": Text, "history": List}
        assert (
            has_multimodal_content(arguments, type_hints, exclude_params=["history"])
            is True
        )
        assert (
            has_multimodal_content(arguments, type_hints, exclude_params=["text"])
            is False
        )

    def test_union_type_in_arguments(self, text_content: Text) -> None:
        """Test Union type in arguments."""
        arguments = {"content": text_content}
        type_hints = {"content": Union[str, Text]}
        assert has_multimodal_content(arguments, type_hints) is True

    def test_list_multimodal_in_arguments(self, text_content: Text) -> None:
        """Test List of multimodal in arguments."""
        arguments = {"contents": [text_content]}
        type_hints = {"contents": List[Text]}
        assert has_multimodal_content(arguments, type_hints) is True

    def test_empty_arguments(self) -> None:
        """Test empty arguments."""
        assert has_multimodal_content({}, {}) is False

    def test_missing_type_hint(self, text_content: Text) -> None:
        """Test argument without type hint."""
        arguments = {"text": text_content}
        type_hints = {}
        # Should not raise error, just return False
        assert has_multimodal_content(arguments, type_hints) is False

