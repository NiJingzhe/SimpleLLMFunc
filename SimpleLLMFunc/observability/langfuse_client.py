from functools import lru_cache
from typing import Any, Callable, Optional

from langfuse import Langfuse

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


__all__ = [
    "langfuse_client",
    "coerce_langfuse_metadata",
    "flush_all_observations",
]
