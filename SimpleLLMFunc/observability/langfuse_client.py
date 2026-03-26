from functools import lru_cache
from typing import Any, Callable, Optional
import contextvars

from langfuse import Langfuse
from langfuse.types import TraceContext
from opentelemetry import trace as otel_trace_api

from SimpleLLMFunc.observability.langfuse_config import langfuse_config


def _export_all_spans(_: Any) -> bool:
    return True


def coerce_langfuse_metadata(
    metadata: Optional[dict[str, Any]],
) -> dict[str, str]:
    if not metadata:
        return {}

    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str):
            text = value
        else:
            text = str(value)
        if not text:
            continue
        normalized[str(key)] = text
    return normalized


def _resolve_should_export_span() -> Optional[Callable[[Any], bool]]:
    if langfuse_config.LANGFUSE_EXPORT_ALL_SPANS:
        return _export_all_spans
    return None


@lru_cache
def get_langfuse_client() -> Langfuse:
    client_kwargs: dict[str, Any] = {
        "public_key": langfuse_config.LANGFUSE_PUBLIC_KEY,
        "secret_key": langfuse_config.LANGFUSE_SECRET_KEY,
        "host": langfuse_config.LANGFUSE_BASE_URL,
    }

    should_export_span = _resolve_should_export_span()
    if should_export_span is not None:
        client_kwargs["should_export_span"] = should_export_span

    return Langfuse(**client_kwargs)


# 全局配置实例
langfuse_client = get_langfuse_client()


def flush_all_observations() -> None:
    langfuse_client.flush()


_langfuse_trace_context_var: contextvars.ContextVar[Optional[TraceContext]] = (
    contextvars.ContextVar("langfuse_trace_context", default=None)
)


def set_langfuse_trace_context(
    trace_context: TraceContext,
) -> contextvars.Token[Optional[TraceContext]]:
    return _langfuse_trace_context_var.set(trace_context)


def reset_langfuse_trace_context(
    token: contextvars.Token[Optional[TraceContext]],
) -> None:
    try:
        _langfuse_trace_context_var.reset(token)
    except (ValueError, RuntimeError):
        # Token may be from a different context or already reset.
        return


def update_langfuse_parent_span(parent_span_id: Optional[str]) -> None:
    if not parent_span_id:
        return
    current = _langfuse_trace_context_var.get()
    if not current:
        return
    updated: TraceContext = {"trace_id": current.get("trace_id", "")}
    if updated["trace_id"]:
        updated["parent_span_id"] = parent_span_id
        _langfuse_trace_context_var.set(updated)


def _format_trace_id(trace_id: Any) -> Optional[str]:
    if isinstance(trace_id, str):
        return trace_id or None
    if not trace_id:
        return None
    return f"{int(trace_id):032x}"


def _format_span_id(span_id: Any) -> Optional[str]:
    if isinstance(span_id, str):
        return span_id or None
    if not span_id:
        return None
    return f"{int(span_id):016x}"


def _get_active_otel_trace_context() -> Optional[TraceContext]:
    current_span = otel_trace_api.get_current_span()
    if current_span is otel_trace_api.INVALID_SPAN:
        return None

    get_span_context = getattr(current_span, "get_span_context", None)
    if not callable(get_span_context):
        return None

    span_context = get_span_context()
    is_valid = getattr(span_context, "is_valid", False)
    if not is_valid:
        return None

    trace_id = _format_trace_id(getattr(span_context, "trace_id", 0))
    if not trace_id:
        return None

    context: TraceContext = {"trace_id": trace_id}
    parent_span_id = _format_span_id(getattr(span_context, "span_id", 0))
    if parent_span_id:
        context["parent_span_id"] = parent_span_id
    return context


def get_langfuse_trace_context() -> Optional[TraceContext]:
    active_context = _get_active_otel_trace_context()
    if active_context:
        return active_context

    current = _langfuse_trace_context_var.get()
    if current:
        return dict(current)

    return None


__all__ = [
    "langfuse_client",
    "coerce_langfuse_metadata",
    "get_langfuse_trace_context",
    "set_langfuse_trace_context",
    "reset_langfuse_trace_context",
    "update_langfuse_parent_span",
    "flush_all_observations",
]
