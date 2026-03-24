"""Runtime primitive system exports."""

from .primitives import (
    ForkContext,
    PrimitiveCallContext,
    PrimitiveContract,
    PrimitiveHandler,
    PrimitivePack,
    PrimitiveParameterSpec,
    PrimitiveRegistry,
    PrimitiveSpec,
    RuntimePrimitiveBackend,
    primitive,
    primitive_spec,
)
from typing import Any

__all__ = [
    "PrimitiveCallContext",
    "PrimitiveContract",
    "ForkContext",
    "PrimitiveHandler",
    "PrimitivePack",
    "PrimitiveParameterSpec",
    "PrimitiveRegistry",
    "PrimitiveSpec",
    "RuntimePrimitiveBackend",
    "primitive",
    "primitive_spec",
    "register_self_reference_primitives",
    "build_self_reference_pack",
    "DEFAULT_SELF_REFERENCE_BACKEND_NAME",
]


def register_self_reference_primitives(*args: Any, **kwargs: Any) -> Any:
    from .builtin_self_reference import register_self_reference_primitives as _impl

    return _impl(*args, **kwargs)


def build_self_reference_pack(*args: Any, **kwargs: Any) -> Any:
    from .builtin_self_reference import build_self_reference_pack as _impl

    return _impl(*args, **kwargs)


DEFAULT_SELF_REFERENCE_BACKEND_NAME = "selfref"
