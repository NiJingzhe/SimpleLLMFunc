"""Type resolution helpers for LLM decorators."""

from SimpleLLMFunc.base.type_resolve.description import (
    describe_pydantic_model,
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
]

