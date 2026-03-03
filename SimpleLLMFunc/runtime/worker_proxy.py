"""Worker-side proxy objects for runtime primitive RPC calls."""

from __future__ import annotations

from typing import Any, Optional, Protocol


class PrimitiveTransport(Protocol):
    """Transport contract implemented by the worker host."""

    def call_primitive(
        self,
        name: str,
        args: list[Any],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any: ...


class WorkerRuntimeNamespace:
    """Dynamic dotted primitive namespace.

    Example:
        runtime.fork.spawn("task") -> primitive name ``fork.spawn``
    """

    def __init__(self, transport: PrimitiveTransport, path: str):
        self._transport = transport
        self._path = path

    def __getattr__(self, name: str) -> "WorkerRuntimeNamespace":
        if name.startswith("_"):
            raise AttributeError(name)
        if not self._path:
            return WorkerRuntimeNamespace(self._transport, name)
        return WorkerRuntimeNamespace(self._transport, f"{self._path}.{name}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._path:
            raise TypeError("runtime root is not directly callable")
        return self._transport.call_primitive(
            name=self._path,
            args=list(args),
            kwargs=kwargs,
        )


class WorkerRuntimeProxy:
    """Worker global ``runtime`` object."""

    def __init__(self, transport: PrimitiveTransport):
        self._transport = transport

    def __getattr__(self, name: str) -> WorkerRuntimeNamespace:
        if name.startswith("_"):
            raise AttributeError(name)
        return WorkerRuntimeNamespace(self._transport, name)

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call one primitive by explicit dotted name."""

        return self._transport.call_primitive(
            name=name,
            args=list(args),
            kwargs=kwargs,
        )

    def list_primitives(self) -> list[str]:
        """List primitive names from host registry."""

        result = self.call("runtime.list_primitives")
        if isinstance(result, list):
            return [str(item) for item in result]
        return []

    def list_backends(self) -> list[str]:
        """List registered runtime backend names."""

        result = self.call("runtime.list_backends")
        if isinstance(result, list):
            return [str(item) for item in result]
        return []


__all__ = [
    "PrimitiveTransport",
    "WorkerRuntimeNamespace",
    "WorkerRuntimeProxy",
]
