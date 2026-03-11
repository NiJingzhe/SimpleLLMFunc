"""Backward-compatible re-export for self-reference primitives.

Prefer importing from ``SimpleLLMFunc.builtin.self_reference``.
"""

from SimpleLLMFunc.builtin.self_reference import (
    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM,
    SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM,
    SelfReference,
    SelfReferenceInstanceHandle,
    SelfReferenceMemoryHandle,
    SelfReferenceMemoryProxy,
)

__all__ = [
    "SelfReference",
    "SelfReferenceMemoryHandle",
    "SelfReferenceMemoryProxy",
    "SelfReferenceInstanceHandle",
    "SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM",
    "SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM",
]
