"""Type resolution helpers for LLM decorators."""

from SimpleLLMFunc.base.type_resolve.description import (
    build_type_description_xml,
    describe_pydantic_model,
    generate_example_xml,
    get_detailed_type_description,
)
from SimpleLLMFunc.base.type_resolve.multimodal import (
    has_multimodal_content,
    is_multimodal_type,
)

__all__ = [
    "get_detailed_type_description",
    "has_multimodal_content",
    "is_multimodal_type",
    "describe_pydantic_model",
    "build_type_description_xml",
    "generate_example_xml",
]

