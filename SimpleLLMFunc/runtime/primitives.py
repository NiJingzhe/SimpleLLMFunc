"""Primitive registry and runtime call context.

This module defines a small runtime primitive system used by ``PyRepl``.
Primitives are host-registered callables that can be invoked from the worker
process through a structured RPC bridge.
"""

from __future__ import annotations

import inspect
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast


PrimitiveHandler = Callable[..., Any]

_PRIMITIVE_NAME_PATTERN = re.compile(
    r"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)*$"
)


def _normalize_primitive_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("primitive name must be a non-empty string")

    normalized = name.strip()
    if not normalized:
        raise ValueError("primitive name must be a non-empty string")

    if _PRIMITIVE_NAME_PATTERN.match(normalized) is None:
        raise ValueError(
            "primitive name must match '<segment>[.<segment>...]' with "
            "alnum/underscore segments"
        )

    return normalized


@dataclass
class PrimitiveCallContext:
    """Execution context passed to every primitive handler."""

    primitive_name: str
    call_id: str
    execution_id: str
    event_emitter: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def emit(self, event_name: str, data: Dict[str, Any]) -> None:
        """Emit a custom event through the bound emitter if available."""

        emitter = self.event_emitter
        if emitter is None:
            return

        emit = getattr(emitter, "emit", None)
        if not callable(emit):
            return

        maybe_awaitable = emit(event_name, data)
        if inspect.isawaitable(maybe_awaitable):
            await cast(Awaitable[Any], maybe_awaitable)


@dataclass(frozen=True)
class PrimitiveSpec:
    """Registered primitive metadata."""

    name: str
    handler: PrimitiveHandler
    description: str = ""


class PrimitiveRegistry:
    """Host-side primitive registry.

    The registry is intentionally simple:
    - Host code registers primitive handlers.
    - Worker code invokes primitives by name via RPC.
    - Registry executes handlers with a typed context object.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._specs: Dict[str, PrimitiveSpec] = {}

    def register(
        self,
        name: str,
        handler: PrimitiveHandler,
        *,
        description: str = "",
        replace: bool = False,
    ) -> None:
        """Register one primitive handler.

        Args:
            name: Dotted primitive name (for example ``fork.spawn``).
            handler: Callable invoked as ``handler(context, *args, **kwargs)``.
            description: Optional human-readable description.
            replace: Whether to replace existing registrations.
        """

        normalized = _normalize_primitive_name(name)
        if not callable(handler):
            raise ValueError("primitive handler must be callable")

        spec = PrimitiveSpec(
            name=normalized,
            handler=handler,
            description=description.strip(),
        )

        with self._lock:
            if normalized in self._specs and not replace:
                raise ValueError(f"primitive '{normalized}' is already registered")
            self._specs[normalized] = spec

    def unregister(self, name: str) -> None:
        """Remove one primitive registration if it exists."""

        normalized = _normalize_primitive_name(name)
        with self._lock:
            self._specs.pop(normalized, None)

    def is_registered(self, name: str) -> bool:
        normalized = _normalize_primitive_name(name)
        with self._lock:
            return normalized in self._specs

    def list_names(self) -> List[str]:
        with self._lock:
            names = list(self._specs.keys())
        names.sort()
        return names

    def _get_spec(self, name: str) -> PrimitiveSpec:
        normalized = _normalize_primitive_name(name)
        with self._lock:
            spec = self._specs.get(normalized)
        if spec is None:
            raise KeyError(f"primitive '{normalized}' is not registered")
        return spec

    async def call(
        self,
        name: str,
        *,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        context: PrimitiveCallContext,
    ) -> Any:
        """Invoke a registered primitive handler."""

        spec = self._get_spec(name)
        payload_args = list(args) if isinstance(args, list) else []
        payload_kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}

        await context.emit(
            "runtime_primitive_start",
            {
                "call_id": context.call_id,
                "execution_id": context.execution_id,
                "primitive": spec.name,
            },
        )

        try:
            maybe_result = spec.handler(context, *payload_args, **payload_kwargs)
            if inspect.isawaitable(maybe_result):
                result = await cast(Awaitable[Any], maybe_result)
            else:
                result = maybe_result
        except Exception as exc:
            await context.emit(
                "runtime_primitive_error",
                {
                    "call_id": context.call_id,
                    "execution_id": context.execution_id,
                    "primitive": spec.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise

        await context.emit(
            "runtime_primitive_end",
            {
                "call_id": context.call_id,
                "execution_id": context.execution_id,
                "primitive": spec.name,
                "status": "completed",
            },
        )

        return result


__all__ = [
    "PrimitiveCallContext",
    "PrimitiveHandler",
    "PrimitiveRegistry",
    "PrimitiveSpec",
]
