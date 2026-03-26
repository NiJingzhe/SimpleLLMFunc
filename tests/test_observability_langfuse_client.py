"""Tests for Langfuse trace-context helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

langfuse_client_module = importlib.import_module(
    "SimpleLLMFunc.observability.langfuse_client"
)

from SimpleLLMFunc.observability.langfuse_client import (
    get_langfuse_trace_context,
    reset_langfuse_trace_context,
    set_langfuse_trace_context,
)


@dataclass
class _FakeSpanContext:
    trace_id: int
    span_id: int
    is_valid: bool = True


class _FakeSpan:
    def __init__(self, *, trace_id: int, span_id: int, is_valid: bool = True) -> None:
        self._span_context = _FakeSpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_valid=is_valid,
        )

    def get_span_context(self) -> _FakeSpanContext:
        return self._span_context


def test_get_langfuse_trace_context_prefers_contextvar(monkeypatch) -> None:
    token = set_langfuse_trace_context(
        {
            "trace_id": "trace-root",
            "parent_span_id": "span-root",
        }
    )

    try:
        monkeypatch.setattr(
            langfuse_client_module.otel_trace_api,
            "get_current_span",
            lambda: langfuse_client_module.otel_trace_api.INVALID_SPAN,
        )

        assert get_langfuse_trace_context() == {
            "trace_id": "trace-root",
            "parent_span_id": "span-root",
        }
    finally:
        reset_langfuse_trace_context(token)


def test_get_langfuse_trace_context_returns_none_without_context(monkeypatch) -> None:
    monkeypatch.setattr(
        langfuse_client_module.otel_trace_api,
        "get_current_span",
        lambda: langfuse_client_module.otel_trace_api.INVALID_SPAN,
    )

    assert get_langfuse_trace_context() is None


def test_get_langfuse_trace_context_reads_active_otel_span(monkeypatch) -> None:
    monkeypatch.setattr(
        langfuse_client_module.otel_trace_api,
        "get_current_span",
        lambda: _FakeSpan(trace_id=0x1234, span_id=0x5678),
    )

    assert get_langfuse_trace_context() == {
        "trace_id": "00000000000000000000000000001234",
        "parent_span_id": "0000000000005678",
    }
