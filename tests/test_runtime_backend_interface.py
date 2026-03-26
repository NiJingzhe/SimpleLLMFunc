"""Tests for RuntimePrimitiveBackend interface behavior."""

from __future__ import annotations

from typing import Any

import pytest

from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.runtime import (
    PrimitiveCallContext,
    PrimitivePack,
    PrimitiveRegistry,
    RuntimePrimitiveBackend,
    primitive,
)


class _TrackingBackend(RuntimePrimitiveBackend):
    def __init__(self) -> None:
        self.install_calls = 0
        self.close_calls = 0
        self.clone_calls = 0
        self.clone_contexts: list[Any] = []

    def clone_for_fork(self, *, context):
        self.clone_calls += 1
        self.clone_contexts.append(context)
        return self

    def on_install(self, repl):
        _ = repl
        self.install_calls += 1

    def on_close(self, repl):
        _ = repl
        self.close_calls += 1


def _build_pack(
    backend: RuntimePrimitiveBackend,
    *,
    backend_name: str | None = None,
) -> PrimitivePack:
    pack = PrimitivePack("demo", backend=backend, backend_name=backend_name)

    @pack.primitive("ping")
    def demo_ping(ctx):
        """
        Use: Health check for the demo backend.
        Output: `str`.
        Best Practices:
        - Use for smoke tests only.
        """
        _ = ctx
        return "pong"

    return pack


def test_runtime_backend_hooks_and_clone_for_fork() -> None:
    backend = _TrackingBackend()
    repl = PyRepl()

    repl.install_pack(_build_pack(backend))

    assert backend.install_calls == 1
    assert backend.clone_calls == 0

    forked = repl._clone_for_fork()

    assert backend.clone_calls == 1
    assert backend.clone_contexts
    context = backend.clone_contexts[0]
    assert context.parent_pack_name == "demo"
    assert context.backend_name == "demo"
    assert forked.get_runtime_backend("demo") is backend

    repl.close()
    forked.close()

    assert backend.close_calls == 2


def test_install_pack_replace_closes_previous_backend() -> None:
    first_backend = _TrackingBackend()
    second_backend = _TrackingBackend()
    repl = PyRepl()

    repl.install_pack(_build_pack(first_backend))
    repl.install_pack(_build_pack(second_backend), replace=True)

    assert first_backend.install_calls == 1
    assert first_backend.close_calls == 1
    assert second_backend.install_calls == 1
    assert second_backend.close_calls == 0

    repl.close()

    assert first_backend.close_calls == 1
    assert second_backend.close_calls == 1


def test_install_pack_replace_same_backend_skips_duplicate_lifecycle_calls() -> None:
    backend = _TrackingBackend()
    repl = PyRepl()

    repl.install_pack(_build_pack(backend))
    repl.install_pack(_build_pack(backend), replace=True)

    assert backend.install_calls == 1
    assert backend.close_calls == 0

    repl.close()

    assert backend.close_calls == 1


@pytest.mark.asyncio
async def test_pack_backend_name_supports_single_segment_non_identifier() -> None:
    repl = PyRepl()
    pack = PrimitivePack(
        "demo",
        backend={"value": "ok"},
        backend_name="demo-backend",
    )

    @pack.primitive("read")
    def demo_read(ctx):
        """
        Use: Read one value from the demo backend.
        Output: `str`.
        Best Practices:
        - Use for backend-binding verification only.
        """
        backend = ctx.backend
        if not isinstance(backend, dict):
            raise RuntimeError("demo backend must be a dict")
        return backend["value"]

    repl.install_pack(pack)

    assert repl.get_runtime_backend("demo-backend") == {"value": "ok"}

    result = await repl.execute("print(runtime.demo.read())")

    assert result["success"] is True
    assert result["stdout"].splitlines() == ["ok"]


@pytest.mark.asyncio
async def test_backend_bound_primitive_fails_fast_when_backend_missing() -> None:
    registry = PrimitiveRegistry()

    @primitive()
    def needs_backend(ctx):
        """
        Use: Resolve a backend-bound primitive.
        Output: `str`.
        Best Practices:
        - Use for backend resolution tests only.
        """
        _ = ctx
        return "ok"

    registry.register(
        "demo.needs_backend",
        needs_backend,
        backend_name="missing-backend",
    )

    context = PrimitiveCallContext(
        primitive_name="demo.needs_backend",
        call_id="call-1",
        execution_id="exec-1",
        repl=PyRepl(),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to resolve backend 'missing-backend' for primitive "
            "'demo.needs_backend'"
        ),
    ):
        await registry.call("demo.needs_backend", context=context)

    assert context.backend is None
    assert context.metadata["backend_name"] == "missing-backend"
    assert context.metadata["backend_resolution_error"] == {
        "backend_name": "missing-backend",
        "error_type": "MissingBackend",
        "error_message": "runtime backend 'missing-backend' is not registered",
    }
