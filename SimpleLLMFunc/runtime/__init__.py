"""Runtime primitive system exports."""

from .primitives import (
    PrimitiveCallContext,
    PrimitiveHandler,
    PrimitiveParameterSpec,
    PrimitiveRegistry,
    PrimitiveSpec,
    primitive_spec,
)
from .builtin_self_reference import register_self_reference_primitives

__all__ = [
    "PrimitiveCallContext",
    "PrimitiveHandler",
    "PrimitiveParameterSpec",
    "PrimitiveRegistry",
    "PrimitiveSpec",
    "primitive_spec",
    "register_self_reference_primitives",
]
