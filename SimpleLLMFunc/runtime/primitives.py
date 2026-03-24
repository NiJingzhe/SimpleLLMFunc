"""Primitive registry and runtime call context.

Primitive = host-registered callable, callable in REPL without import as
runtime.namespace.name(...). Used by PyRepl; invoked from worker via RPC.
"""

from __future__ import annotations

import inspect
import re
import threading
import textwrap
from abc import ABC
from xml.sax.saxutils import escape as _xml_escape
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, cast


PrimitiveHandler = Callable[..., Any]

_UNSET = object()

RUNTIME_RESULT_SENTINEL = "__simplellmfunc_runtime_result__"
RUNTIME_RESULT_VALUE_KEY = "value"
RUNTIME_RESULT_NEXT_STEPS_KEY = "next_steps"
RUNTIME_RESULT_PRIMITIVE_KEY = "primitive"

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


def _xml_escape_text(value: Any) -> str:
    return _xml_escape(str(value))


def _xml_escape_attribute(value: Any) -> str:
    return _xml_escape(
        str(value),
        {
            '"': "&quot;",
            "'": "&apos;",
        },
    )


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


def _normalize_pack_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("primitive pack name must be a non-empty string")

    normalized = name.strip()
    if not normalized:
        raise ValueError("primitive pack name must be a non-empty string")

    if "." in normalized:
        raise ValueError("primitive pack name must be a single segment")

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", normalized):
        raise ValueError(
            "primitive pack name must start with a letter and contain only "
            "letters, digits, or underscore"
        )

    return normalized


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

    def to_xml_element(self) -> str:
        attributes = [
            f'name="{_xml_escape_attribute(self.name)}"',
            f'type="{_xml_escape_attribute(self.type)}"',
            f'required="{str(bool(self.required)).lower()}"',
            f'kind="{_xml_escape_attribute(self.kind)}"',
        ]
        if self.default is not None:
            attributes.append(f'default="{_xml_escape_attribute(self.default)}"')

        description_text = _xml_escape_text(self.description)
        joined_attributes = " ".join(attributes)
        return f"<parameter {joined_attributes}>{description_text}</parameter>"


@dataclass(frozen=True)
class PrimitiveContract:
    """Typed primitive metadata contract shared across registry and packs."""

    description: str = ""
    input_type: str = ""
    output_type: str = ""
    output_parsing: str = ""
    parameters: tuple[PrimitiveParameterSpec, ...] = ()
    best_practices: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrimitivePackEntry:
    """One primitive definition within a pack."""

    name: str
    handler: PrimitiveHandler
    contract: PrimitiveContract = field(default_factory=PrimitiveContract)


@dataclass(frozen=True)
class ForkContext:
    """Context passed to backend clone hooks during fork."""

    parent_pack_name: str
    backend_name: str
    child_fork_id: Optional[str] = None
    source_memory_key: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RuntimePrimitiveBackend(ABC):
    """Optional base class for backends that need fork lifecycle control."""

    def clone_for_fork(self, *, context: ForkContext) -> "RuntimePrimitiveBackend":
        """Return a backend instance for the child agent.

        Default behavior shares the same backend instance.
        """

        _ = context
        return self

    def on_install(self, repl: Any) -> None:
        """Called once when a backend is installed into a REPL."""

        _ = repl

    def on_close(self, repl: Any) -> None:
        """Called when the hosting REPL is closed."""

        _ = repl


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


def _coerce_primitive_contract(raw: Optional[Any]) -> PrimitiveContract:
    if raw is None:
        return PrimitiveContract()

    if isinstance(raw, PrimitiveContract):
        return raw

    if not isinstance(raw, dict):
        raise ValueError("primitive contract must be a PrimitiveContract or dict")

    return PrimitiveContract(
        description=_normalize_text(raw.get("description")),
        input_type=_normalize_text(raw.get("input_type")),
        output_type=_normalize_text(raw.get("output_type")),
        output_parsing=_normalize_text(raw.get("output_parsing")),
        parameters=_coerce_parameter_specs(raw.get("parameters")),
        next_steps=_coerce_next_steps(raw.get("next_steps")),
    )


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


def _coerce_next_steps(raw_next_steps: Optional[Any]) -> tuple[str, ...]:
    if raw_next_steps is None:
        return ()

    if isinstance(raw_next_steps, str):
        text = raw_next_steps.strip()
        return (text,) if text else ()

    if isinstance(raw_next_steps, Sequence):
        normalized: List[str] = []
        seen: set[str] = set()
        for item in raw_next_steps:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
        return tuple(normalized)

    return ()


def _normalize_parameter_lookup_name(name: str) -> str:
    return name.lstrip("*").strip()


def _parse_primitive_docstring(docstring: Optional[str]) -> Dict[str, Any]:
    text = textwrap.dedent(docstring or "").strip()
    if not text:
        return {}

    recognized = {
        "use": "description",
        "input": "input_type",
        "output": "output_type",
        "parse": "output_parsing",
        "parameters": "parameters",
        "best practices": "best_practices",
        "best_practices": "best_practices",
    }
    sections: Dict[str, List[str]] = {}
    current_key: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if current_key is not None:
                sections.setdefault(current_key, []).append("")
            continue

        matched_heading = False
        for candidate, target_key in recognized.items():
            prefix = f"{candidate}:"
            if stripped.lower().startswith(prefix):
                current_key = target_key
                remainder = stripped[len(prefix) :].strip()
                sections.setdefault(current_key, [])
                if remainder:
                    sections[current_key].append(remainder)
                matched_heading = True
                break
        if matched_heading:
            continue

        if current_key is None:
            current_key = "description"
            sections.setdefault(current_key, [])
        sections[current_key].append(stripped)

    payload: Dict[str, Any] = {}
    parameter_descriptions: Dict[str, str] = {}
    best_practices: List[str] = []

    for key, values in sections.items():
        non_empty = [value.strip() for value in values if value.strip()]
        if not non_empty:
            continue

        if key == "parameters":
            for item in non_empty:
                normalized = item
                if normalized.startswith("-"):
                    normalized = normalized[1:].strip()
                name, sep, description = normalized.partition(":")
                if not sep:
                    continue
                lookup_name = _normalize_parameter_lookup_name(name)
                text_value = description.strip()
                if lookup_name and text_value:
                    parameter_descriptions[lookup_name] = text_value
            continue

        if key == "best_practices":
            for item in non_empty:
                normalized = item[1:].strip() if item.startswith("-") else item
                if normalized and normalized not in best_practices:
                    best_practices.append(normalized)
            continue

        payload[key] = " ".join(non_empty)

    if parameter_descriptions:
        payload["parameter_descriptions"] = parameter_descriptions
    if best_practices:
        payload["best_practices"] = tuple(best_practices)
    return payload


def primitive(
    *,
    description: str = "",
    input_type: str = "",
    output_type: str = "",
    output_parsing: str = "",
    parameters: Optional[Sequence[Any]] = None,
    next_steps: Optional[Any] = None,
) -> Callable[[PrimitiveHandler], PrimitiveHandler]:
    """Attach primitive spec metadata parsed from the handler docstring.

    This decorator is the canonical way to declare runtime primitive contracts.
    """

    def decorator(handler: PrimitiveHandler) -> PrimitiveHandler:
        metadata = _parse_primitive_docstring(getattr(handler, "__doc__", None))
        if description.strip():
            metadata["description"] = description.strip()
        if input_type.strip():
            metadata["input_type"] = input_type.strip()
        if output_type.strip():
            metadata["output_type"] = output_type.strip()
        if output_parsing.strip():
            metadata["output_parsing"] = output_parsing.strip()
        if parameters is not None:
            metadata["parameters"] = _coerce_parameter_specs(parameters)
        if next_steps is not None:
            metadata["next_steps"] = _coerce_next_steps(next_steps)
        setattr(handler, "__simplellmfunc_primitive_spec__", metadata)
        return handler

    return decorator


def primitive_spec(
    *,
    description: str = "",
    input_type: str = "",
    output_type: str = "",
    output_parsing: str = "",
    parameters: Optional[Sequence[Any]] = None,
    next_steps: Optional[Any] = None,
) -> Callable[[PrimitiveHandler], PrimitiveHandler]:
    """Backward-compatible alias for :func:`primitive`."""

    return primitive(
        description=description,
        input_type=input_type,
        output_type=output_type,
        output_parsing=output_parsing,
        parameters=parameters,
        next_steps=next_steps,
    )


def _extract_handler_primitive_spec(handler: PrimitiveHandler) -> Dict[str, Any]:
    raw = getattr(handler, "__simplellmfunc_primitive_spec__", None)
    if isinstance(raw, dict):
        return dict(raw)
    return _parse_primitive_docstring(getattr(handler, "__doc__", None))


def _apply_parameter_descriptions(
    parameters: Sequence[PrimitiveParameterSpec],
    descriptions: Dict[str, str],
) -> tuple[PrimitiveParameterSpec, ...]:
    if not descriptions:
        return tuple(parameters)

    normalized: List[PrimitiveParameterSpec] = []
    for parameter in parameters:
        description = descriptions.get(_normalize_parameter_lookup_name(parameter.name))
        if description:
            normalized.append(
                PrimitiveParameterSpec(
                    name=parameter.name,
                    type=parameter.type,
                    required=parameter.required,
                    description=description,
                    kind=parameter.kind,
                    default=parameter.default,
                )
            )
            continue
        normalized.append(parameter)
    return tuple(normalized)


def _resolve_primitive_contract(
    handler: PrimitiveHandler,
    *,
    contract: Optional[Any] = None,
    description: str = "",
    input_type: str = "",
    output_type: str = "",
    output_parsing: str = "",
    parameters: Optional[Sequence[Any]] = None,
    next_steps: Optional[Any] = None,
) -> PrimitiveContract:
    handler_spec = _extract_handler_primitive_spec(handler)
    contract_value = _coerce_primitive_contract(contract)

    parameter_specs = _coerce_parameter_specs(parameters)
    if not parameter_specs and contract_value.parameters:
        parameter_specs = contract_value.parameters
    if not parameter_specs:
        doc_parameters = handler_spec.get("parameters")
        if isinstance(doc_parameters, tuple):
            parameter_specs = doc_parameters
    if not parameter_specs:
        parameter_specs = _infer_parameter_specs(handler)
    parameter_specs = _apply_parameter_descriptions(
        parameter_specs,
        handler_spec.get("parameter_descriptions", {}),
    )

    best_practices_value = cast(
        tuple[str, ...],
        handler_spec.get("best_practices", ()),
    )
    if not best_practices_value:
        raise ValueError("primitive docstring must include a 'Best Practices' section")

    next_steps_value = _coerce_next_steps(next_steps)
    if not next_steps_value and contract_value.next_steps:
        next_steps_value = contract_value.next_steps
    if not next_steps_value:
        next_steps_value = cast(tuple[str, ...], handler_spec.get("next_steps", ()))

    resolved_description = _normalize_text(description)
    if not resolved_description:
        resolved_description = (
            contract_value.description
            or str(handler_spec.get("description", "")).strip()
        )

    resolved_input_type = _normalize_text(input_type)
    if not resolved_input_type:
        resolved_input_type = (
            contract_value.input_type or str(handler_spec.get("input_type", "")).strip()
        )

    resolved_output_type = _normalize_text(output_type)
    if not resolved_output_type:
        resolved_output_type = (
            contract_value.output_type
            or str(handler_spec.get("output_type", "")).strip()
        )

    resolved_output_parsing = _normalize_text(output_parsing)
    if not resolved_output_parsing:
        resolved_output_parsing = (
            contract_value.output_parsing
            or str(handler_spec.get("output_parsing", "")).strip()
        )

    return PrimitiveContract(
        description=resolved_description,
        input_type=resolved_input_type,
        output_type=resolved_output_type,
        output_parsing=resolved_output_parsing,
        parameters=parameter_specs,
        best_practices=best_practices_value,
        next_steps=next_steps_value,
    )


class PrimitivePack:
    """Declarative runtime extension pack with backend and primitive entries."""

    def __init__(
        self,
        name: str,
        *,
        backend: Any,
        backend_name: Optional[str] = None,
    ) -> None:
        self.name = _normalize_pack_name(name)
        self.backend_name = _normalize_pack_name(backend_name or self.name)
        self.backend = backend
        self._entries: Dict[str, PrimitivePackEntry] = {}

    @property
    def primitives(self) -> tuple[PrimitivePackEntry, ...]:
        return tuple(self._entries[name] for name in sorted(self._entries))

    def _compose_primitive_name(self, name: str) -> str:
        normalized = _normalize_primitive_name(name)
        if normalized.startswith(f"{self.name}."):
            return normalized
        return f"{self.name}.{normalized}"

    def add_primitive(
        self,
        name: str,
        handler: PrimitiveHandler,
        *,
        contract: Optional[Any] = None,
        description: str = "",
        input_type: str = "",
        output_type: str = "",
        output_parsing: str = "",
        parameters: Optional[Sequence[Any]] = None,
        next_steps: Optional[Any] = None,
        replace: bool = False,
    ) -> PrimitiveHandler:
        full_name = self._compose_primitive_name(name)
        resolved_contract = _resolve_primitive_contract(
            handler,
            contract=contract,
            description=description,
            input_type=input_type,
            output_type=output_type,
            output_parsing=output_parsing,
            parameters=parameters,
            next_steps=next_steps,
        )

        if full_name in self._entries and not replace:
            raise ValueError(f"primitive '{full_name}' is already declared in pack")

        self._entries[full_name] = PrimitivePackEntry(
            name=full_name,
            handler=handler,
            contract=resolved_contract,
        )
        return handler

    def primitive(
        self,
        name: str,
        *,
        contract: Optional[Any] = None,
        description: str = "",
        input_type: str = "",
        output_type: str = "",
        output_parsing: str = "",
        parameters: Optional[Sequence[Any]] = None,
        next_steps: Optional[Any] = None,
        replace: bool = False,
    ) -> Callable[[PrimitiveHandler], PrimitiveHandler]:
        """Declare one primitive inside the pack."""

        def decorator(handler: PrimitiveHandler) -> PrimitiveHandler:
            return self.add_primitive(
                name,
                handler,
                contract=contract,
                description=description,
                input_type=input_type,
                output_type=output_type,
                output_parsing=output_parsing,
                parameters=parameters,
                next_steps=next_steps,
                replace=replace,
            )

        return decorator

    def clone(
        self,
        *,
        backend: Any = _UNSET,
        backend_name: Optional[str] = None,
        fork_context: Optional[ForkContext] = None,
    ) -> "PrimitivePack":
        """Clone the pack definition with an optional backend override."""

        resolved_backend = self.backend if backend is _UNSET else backend
        if (
            backend is _UNSET
            and fork_context is not None
            and isinstance(resolved_backend, RuntimePrimitiveBackend)
        ):
            resolved_backend = resolved_backend.clone_for_fork(context=fork_context)
        cloned = PrimitivePack(
            self.name,
            backend=resolved_backend,
            backend_name=backend_name or self.backend_name,
        )
        cloned._entries = dict(self._entries)
        return cloned

    def install(self, repl: Any, *, replace: bool = False) -> None:
        """Install this pack into a compatible REPL host."""

        install = getattr(repl, "install_pack", None)
        if not callable(install):
            raise ValueError("repl must provide install_pack(pack, replace=False)")
        install(self, replace=replace)


@dataclass
class PrimitiveCallContext:
    """Execution context passed to every primitive handler."""

    primitive_name: str
    call_id: str
    execution_id: str
    event_emitter: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    repl: Any = None
    registry: Optional["PrimitiveRegistry"] = None
    backend_name: Optional[str] = None
    backend: Any = None

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

    def get_backend(self, name: str) -> Any:
        """Resolve one runtime backend by name through the bound REPL host."""

        repl = self.repl
        getter = getattr(repl, "get_runtime_backend", None)
        if not callable(getter):
            raise RuntimeError(
                "primitive call context is not bound to a runtime host backend lookup"
            )
        return getter(name)


@dataclass(frozen=True)
class PrimitiveSpec:
    """Registered primitive metadata."""

    name: str
    handler: PrimitiveHandler
    description: str = ""
    input_type: str = ""
    output_type: str = ""
    output_parsing: str = ""
    parameters: tuple[PrimitiveParameterSpec, ...] = ()
    best_practices: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    backend_name: Optional[str] = None

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
            "output_parsing": self.output_parsing,
            "parameters": [
                parameter.to_public_dict() for parameter in resolved_parameters
            ],
            "best_practices": list(self.best_practices),
        }

    def to_xml_element(self, *, root_tag: str = "primitive") -> str:
        resolved_parameters = self.parameters or _infer_parameter_specs(self.handler)
        resolved_input_type = self.input_type or (
            "PrimitiveCallContext + declared parameters"
            if resolved_parameters
            else "PrimitiveCallContext only"
        )
        resolved_output_type = self.output_type or _infer_output_type(self.handler)

        lines: List[str] = [
            f"<{root_tag}>",
            f"<name>{_xml_escape_text(self.name)}</name>",
            f"<description>{_xml_escape_text(self.description)}</description>",
            f"<input_type>{_xml_escape_text(resolved_input_type)}</input_type>",
            f"<output_type>{_xml_escape_text(resolved_output_type)}</output_type>",
            f"<output_parsing>{_xml_escape_text(self.output_parsing)}</output_parsing>",
            "<parameters>",
        ]

        for parameter in resolved_parameters:
            lines.append(parameter.to_xml_element())

        lines.append("</parameters>")
        lines.append("<best_practices>")
        for index, practice in enumerate(self.best_practices, start=1):
            lines.append(
                f'<best_practice index="{index}">{_xml_escape_text(practice)}</best_practice>'
            )
        lines.append("</best_practices>")
        lines.append(f"</{root_tag}>")
        return "\n".join(lines)


def _format_parameter_hints(spec: PrimitiveSpec) -> str:
    parameters = spec.parameters or _infer_parameter_specs(spec.handler)
    if not parameters:
        return "Parameter requirements: none"

    chunks: List[str] = []
    for parameter in parameters:
        required_text = "required" if parameter.required else "optional"
        default_text = (
            f", default={parameter.default}" if parameter.default is not None else ""
        )
        chunks.append(
            f"{parameter.name}({parameter.type}, {required_text}{default_text})"
        )
    return "Parameter requirements: " + "; ".join(chunks)


def _wrap_runtime_result(
    value: Any,
    next_steps: tuple[str, ...],
    primitive_name: str,
) -> Any:
    if not next_steps:
        return value

    return {
        RUNTIME_RESULT_SENTINEL: True,
        RUNTIME_RESULT_VALUE_KEY: value,
        RUNTIME_RESULT_NEXT_STEPS_KEY: list(next_steps),
        RUNTIME_RESULT_PRIMITIVE_KEY: primitive_name,
    }


def _append_hint_message(message: str, hint: str) -> str:
    if not hint:
        return message
    if hint in message:
        return message
    if not message:
        return hint
    return f"{message}\n{hint}"


def _is_argument_error(exc: Exception) -> bool:
    if not isinstance(exc, TypeError):
        return False
    text = str(exc)
    tokens = (
        "missing",
        "positional argument",
        "unexpected keyword",
        "got an unexpected",
        "keyword-only",
    )
    return any(token in text for token in tokens)


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
        contract: Optional[Any] = None,
        description: str = "",
        input_type: str = "",
        output_type: str = "",
        output_parsing: str = "",
        parameters: Optional[Sequence[Any]] = None,
        next_steps: Optional[Any] = None,
        backend_name: Optional[str] = None,
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

        normalized_backend_name: Optional[str] = None
        if backend_name is not None:
            normalized_backend_name = _normalize_pack_name(backend_name)

        resolved_contract = _resolve_primitive_contract(
            handler,
            contract=contract,
            description=description,
            input_type=input_type,
            output_type=output_type,
            output_parsing=output_parsing,
            parameters=parameters,
            next_steps=next_steps,
        )

        spec = PrimitiveSpec(
            name=normalized,
            handler=handler,
            description=resolved_contract.description,
            input_type=resolved_contract.input_type,
            output_type=resolved_contract.output_type,
            output_parsing=resolved_contract.output_parsing,
            parameters=resolved_contract.parameters,
            best_practices=resolved_contract.best_practices,
            next_steps=resolved_contract.next_steps,
            backend_name=normalized_backend_name,
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

    def list_names(
        self,
        *,
        prefix: Optional[str] = None,
        contains: Optional[str] = None,
    ) -> List[str]:
        with self._lock:
            names = list(self._specs.keys())

        normalized_prefix: Optional[str] = None
        if isinstance(prefix, str):
            stripped_prefix = prefix.strip()
            if stripped_prefix:
                normalized_prefix = stripped_prefix

        normalized_contains: Optional[str] = None
        if isinstance(contains, str):
            stripped_contains = contains.strip()
            if stripped_contains:
                normalized_contains = stripped_contains

        if normalized_prefix is not None:
            names = [item for item in names if item.startswith(normalized_prefix)]

        if normalized_contains is not None:
            names = [item for item in names if normalized_contains in item]

        names.sort()
        return names

    def list_specs(
        self,
        *,
        names: Optional[Sequence[str]] = None,
        prefix: Optional[str] = None,
        contains: Optional[str] = None,
    ) -> List[PrimitiveSpec]:
        """List registered primitive specs with optional name/prefix/contains filters."""

        selected_names: Optional[set[str]] = None
        if names is not None:
            normalized_names: set[str] = set()
            for item in names:
                if not isinstance(item, str):
                    continue
                candidate = item.strip()
                if not candidate:
                    continue
                normalized_names.add(_normalize_primitive_name(candidate))
            selected_names = normalized_names

        normalized_prefix: Optional[str] = None
        if isinstance(prefix, str):
            stripped_prefix = prefix.strip()
            if stripped_prefix:
                normalized_prefix = stripped_prefix

        normalized_contains: Optional[str] = None
        if isinstance(contains, str):
            stripped_contains = contains.strip()
            if stripped_contains:
                normalized_contains = stripped_contains

        with self._lock:
            specs = list(self._specs.values())

        if selected_names is not None:
            specs = [item for item in specs if item.name in selected_names]

        if normalized_prefix is not None:
            specs = [item for item in specs if item.name.startswith(normalized_prefix)]

        if normalized_contains is not None:
            specs = [item for item in specs if normalized_contains in item.name]

        specs.sort(key=lambda item: item.name)
        return specs

    def list_spec_payloads(
        self,
        *,
        names: Optional[Sequence[str]] = None,
        prefix: Optional[str] = None,
        contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List structured primitive payloads with optional filters."""

        return [
            spec.to_public_dict()
            for spec in self.list_specs(names=names, prefix=prefix, contains=contains)
        ]

    def list_spec_xml(
        self,
        *,
        names: Optional[Sequence[str]] = None,
        prefix: Optional[str] = None,
        contains: Optional[str] = None,
    ) -> str:
        """List primitive specs as XML string payload."""

        lines = ["<primitive_specs>"]
        for spec in self.list_specs(names=names, prefix=prefix, contains=contains):
            lines.append(spec.to_xml_element(root_tag="primitive"))
        lines.append("</primitive_specs>")
        return "\n".join(lines)

    def get_spec_payload(self, name: str) -> Dict[str, Any]:
        """Return one structured primitive payload by exact primitive name."""

        return self._get_spec(name).to_public_dict()

    def get_spec_xml(self, name: str) -> str:
        """Return one primitive spec as XML string payload."""

        return self._get_spec(name).to_xml_element(root_tag="primitive_spec")

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
        try:
            spec = self._get_spec(name)
        except KeyError as exc:
            base_message = str(exc.args[0]) if exc.args else str(exc)
            hint = (
                "Hint: primitive is not registered. Use "
                "runtime.list_primitives(contains='...') to discover names."
            )
            raise KeyError(_append_hint_message(base_message, hint)) from exc
        payload_args = list(args) if isinstance(args, list) else []
        payload_kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}

        context.registry = self
        context.primitive_name = spec.name
        context.backend_name = spec.backend_name
        context.metadata.setdefault("primitive_name", spec.name)
        if spec.backend_name is not None:
            context.metadata.setdefault("backend_name", spec.backend_name)
            try:
                context.backend = context.get_backend(spec.backend_name)
            except Exception:
                context.backend = None
        else:
            context.backend = None

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
            enhanced_exc = exc
            if _is_argument_error(exc) or isinstance(exc, ValueError):
                hint = _format_parameter_hints(spec)
                enhanced_message = _append_hint_message(str(exc), hint)
                enhanced_exc = type(exc)(enhanced_message)
            await context.emit(
                "runtime_primitive_error",
                {
                    "call_id": context.call_id,
                    "execution_id": context.execution_id,
                    "primitive": spec.name,
                    "error_type": type(enhanced_exc).__name__,
                    "error_message": str(enhanced_exc),
                },
            )
            raise enhanced_exc from exc

        await context.emit(
            "runtime_primitive_end",
            {
                "call_id": context.call_id,
                "execution_id": context.execution_id,
                "primitive": spec.name,
                "status": "completed",
            },
        )

        return _wrap_runtime_result(result, spec.next_steps, spec.name)


__all__ = [
    "RUNTIME_RESULT_NEXT_STEPS_KEY",
    "RUNTIME_RESULT_PRIMITIVE_KEY",
    "RUNTIME_RESULT_SENTINEL",
    "RUNTIME_RESULT_VALUE_KEY",
    "PrimitiveCallContext",
    "PrimitiveContract",
    "ForkContext",
    "PrimitiveHandler",
    "PrimitivePack",
    "PrimitivePackEntry",
    "RuntimePrimitiveBackend",
    "primitive",
    "primitive_spec",
    "PrimitiveParameterSpec",
    "PrimitiveRegistry",
    "PrimitiveSpec",
]
