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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, cast


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


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _annotation_to_text(annotation: Any) -> str:
    if annotation is inspect.Signature.empty:
        return "Any"

    if annotation is None:
        return "None"

    if isinstance(annotation, str):
        normalized = annotation.strip()
        return normalized or "Any"

    if isinstance(annotation, type):
        return annotation.__name__

    text = str(annotation)
    if text.startswith("typing."):
        text = text[len("typing.") :]
    return text


def _parameter_kind_to_text(kind: Any) -> str:
    mapping = {
        inspect.Parameter.POSITIONAL_ONLY: "positional_only",
        inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional_or_keyword",
        inspect.Parameter.KEYWORD_ONLY: "keyword_only",
        inspect.Parameter.VAR_POSITIONAL: "var_positional",
        inspect.Parameter.VAR_KEYWORD: "var_keyword",
    }
    return mapping.get(kind, "positional_or_keyword")


@dataclass(frozen=True)
class PrimitiveParameterSpec:
    """Structured parameter metadata for runtime primitive specs."""

    name: str
    type: str = "Any"
    required: bool = False
    description: str = ""
    kind: str = "positional_or_keyword"
    default: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
            "kind": self.kind,
        }
        if self.default is not None:
            payload["default"] = self.default
        return payload


def _coerce_parameter_spec(raw: Any) -> Optional[PrimitiveParameterSpec]:
    if isinstance(raw, PrimitiveParameterSpec):
        return raw

    if not isinstance(raw, dict):
        return None

    raw_name = raw.get("name")
    if not isinstance(raw_name, str):
        return None
    name = raw_name.strip()
    if not name:
        return None

    raw_required = raw.get("required")
    required = raw_required if isinstance(raw_required, bool) else False

    default_value = raw.get("default")
    default_text: Optional[str]
    if default_value is None:
        default_text = None
    else:
        default_text = str(default_value)

    return PrimitiveParameterSpec(
        name=name,
        type=_normalize_text(raw.get("type")) or "Any",
        required=required,
        description=_normalize_text(raw.get("description")),
        kind=_normalize_text(raw.get("kind")) or "positional_or_keyword",
        default=default_text,
    )


def _coerce_parameter_specs(
    raw_parameters: Optional[Sequence[Any]],
) -> tuple[PrimitiveParameterSpec, ...]:
    if not raw_parameters:
        return ()

    normalized: List[PrimitiveParameterSpec] = []
    for item in raw_parameters:
        parameter = _coerce_parameter_spec(item)
        if parameter is not None:
            normalized.append(parameter)
    return tuple(normalized)


def _infer_parameter_specs(
    handler: PrimitiveHandler,
) -> tuple[PrimitiveParameterSpec, ...]:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return ()

    params = list(signature.parameters.values())
    if params:
        params = params[1:]

    inferred: List[PrimitiveParameterSpec] = []
    for param in params:
        kind = _parameter_kind_to_text(param.kind)
        name = param.name
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            name = f"*{name}"
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            name = f"**{name}"

        required = param.default is inspect.Signature.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        default_text: Optional[str] = None
        if param.default is not inspect.Signature.empty:
            default_text = repr(param.default)

        inferred.append(
            PrimitiveParameterSpec(
                name=name,
                type=_annotation_to_text(param.annotation),
                required=required,
                description=f"Parameter `{param.name}`.",
                kind=kind,
                default=default_text,
            )
        )

    return tuple(inferred)


def _infer_output_type(handler: PrimitiveHandler) -> str:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return "Any"
    return _annotation_to_text(signature.return_annotation)


def _coerce_best_practices(
    raw_best_practices: Optional[Sequence[Any]],
) -> tuple[str, ...]:
    if not raw_best_practices:
        return ()

    normalized: List[str] = []
    seen: set[str] = set()
    for item in raw_best_practices:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)

    return tuple(normalized)


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
    input_type: str = ""
    output_type: str = ""
    parameters: tuple[PrimitiveParameterSpec, ...] = ()
    best_practices: tuple[str, ...] = ()

    def to_public_dict(self) -> Dict[str, Any]:
        resolved_parameters = self.parameters or _infer_parameter_specs(self.handler)
        resolved_input_type = self.input_type or (
            "PrimitiveCallContext + declared parameters"
            if resolved_parameters
            else "PrimitiveCallContext only"
        )
        resolved_output_type = self.output_type or _infer_output_type(self.handler)

        return {
            "name": self.name,
            "description": self.description,
            "input_type": resolved_input_type,
            "output_type": resolved_output_type,
            "parameters": [
                parameter.to_public_dict() for parameter in resolved_parameters
            ],
            "best_practices": list(self.best_practices),
        }


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
        input_type: str = "",
        output_type: str = "",
        parameters: Optional[Sequence[Any]] = None,
        best_practices: Optional[Sequence[Any]] = None,
        replace: bool = False,
    ) -> None:
        """Register one primitive handler.

        Args:
            name: Dotted primitive name (for example ``selfref.fork.spawn``).
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
            input_type=input_type.strip(),
            output_type=output_type.strip(),
            parameters=_coerce_parameter_specs(parameters),
            best_practices=_coerce_best_practices(best_practices),
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

    def list_specs(self) -> List[PrimitiveSpec]:
        """List registered primitive specs sorted by primitive name."""

        with self._lock:
            specs = list(self._specs.values())
        specs.sort(key=lambda item: item.name)
        return specs

    def list_spec_payloads(self) -> List[Dict[str, Any]]:
        """List structured primitive payloads sorted by primitive name."""

        return [spec.to_public_dict() for spec in self.list_specs()]

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
    "PrimitiveParameterSpec",
    "PrimitiveRegistry",
    "PrimitiveSpec",
]
