"""Runtime primitive system exports."""

from .primitives import (
    PrimitiveCallContext,
    PrimitiveHandler,
    PrimitiveRegistry,
    PrimitiveSpec,
)
from .builtin_self_reference import register_self_reference_primitives

__all__ = [
    "PrimitiveCallContext",
    "PrimitiveHandler",
    "PrimitiveRegistry",
    "PrimitiveSpec",
    "register_self_reference_primitives",
]
