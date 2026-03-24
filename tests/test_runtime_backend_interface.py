"""Tests for RuntimePrimitiveBackend interface behavior."""

from __future__ import annotations

from typing import Any

from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.runtime import PrimitivePack, RuntimePrimitiveBackend


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


def _build_pack(backend: RuntimePrimitiveBackend) -> PrimitivePack:
    pack = PrimitivePack("demo", backend=backend)

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
