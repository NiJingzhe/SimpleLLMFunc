"""Tests for primitive docstring parsing and best practice precedence."""

from __future__ import annotations

from SimpleLLMFunc.runtime.primitives import PrimitiveRegistry, primitive


def test_docstring_best_practices_are_primary_and_merged() -> None:
    registry = PrimitiveRegistry()

    @primitive()
    def handler(ctx):
        """
        Use: Demo primitive.
        Best Practices:
        - doc-1
        - doc-2
        """
        _ = ctx

    registry.register("demo.best_practices", handler)

    payload = registry.get_spec_payload("demo.best_practices")

    assert payload["best_practices"] == ["doc-1", "doc-2"]


def test_docstring_best_practices_used_when_explicit_absent() -> None:
    registry = PrimitiveRegistry()

    @primitive()
    def handler(ctx):
        """
        Use: Demo primitive.
        Best Practices:
        - doc-only
        """
        _ = ctx

    registry.register("demo.doc_only", handler)

    payload = registry.get_spec_payload("demo.doc_only")

    assert payload["best_practices"] == ["doc-only"]


def test_missing_best_practices_raises_error() -> None:
    registry = PrimitiveRegistry()

    @primitive()
    def handler(ctx):
        """Use: Demo primitive without best practices."""
        _ = ctx

    try:
        registry.register("demo.missing_best_practices", handler)
    except ValueError as exc:
        assert "Best Practices" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing Best Practices")
