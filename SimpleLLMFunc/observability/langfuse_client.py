from functools import lru_cache
from typing import Any, Callable, Optional

from langfuse import Langfuse
from langfuse.types import TraceContext

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


def get_langfuse_trace_context() -> Optional[TraceContext]:
    trace_id = langfuse_client.get_current_trace_id()
    if not trace_id:
        return None

    context: TraceContext = {"trace_id": trace_id}
    parent_span_id = langfuse_client.get_current_observation_id()
    if parent_span_id:
        context["parent_span_id"] = parent_span_id
    return context


__all__ = [
    "langfuse_client",
    "coerce_langfuse_metadata",
    "get_langfuse_trace_context",
    "flush_all_observations",
]
