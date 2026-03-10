"""Worker-side proxy objects for runtime primitive RPC calls."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Union


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
        runtime.selfref.fork.spawn("task") -> primitive name ``selfref.fork.spawn``
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

    def list_primitives(self, prefix: Optional[str] = None) -> list[str]:
        """List primitive names from host registry."""

        call_kwargs: dict[str, Any] = {}
        if prefix is not None:
            call_kwargs["prefix"] = prefix

        result = self.call("runtime.list_primitives", **call_kwargs)
        if isinstance(result, list):
            return [str(item) for item in result]
        return []

    def list_primitive_specs(
        self,
        names: Optional[list[str]] = None,
        prefix: Optional[str] = None,
        format: str = "xml",
    ) -> Union[list[dict[str, Any]], str]:
        """List primitive specs from host registry as dict payloads or XML."""

        call_kwargs: dict[str, Any] = {}
        if names is not None:
            call_kwargs["names"] = names
        if prefix is not None:
            call_kwargs["prefix"] = prefix
        if isinstance(format, str):
            call_kwargs["format"] = format

        result = self.call("runtime.list_primitive_specs", **call_kwargs)
        if isinstance(result, str):
            return result
        if not isinstance(result, list):
            return []

        specs: list[dict[str, Any]] = []
        for item in result:
            if not isinstance(item, dict):
                continue

            name = item.get("name")
            if not isinstance(name, str):
                continue

            normalized_item: dict[str, Any] = {"name": name}
            for key, value in item.items():
                if key == "name":
                    continue
                if isinstance(key, str):
                    normalized_item[key] = value

            if not isinstance(normalized_item.get("description"), str):
                normalized_item["description"] = ""

            specs.append(normalized_item)

        return specs

    def get_primitive_spec(
        self,
        name: str,
        *,
        format: str = "xml",
    ) -> Union[dict[str, Any], str]:
        """Get one primitive spec by exact primitive name as dict or XML."""

        call_kwargs: dict[str, Any] = {}
        if isinstance(format, str):
            call_kwargs["format"] = format

        result = self.call("runtime.get_primitive_spec", name, **call_kwargs)
        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            return {}

        raw_name = result.get("name")
        if not isinstance(raw_name, str):
            return {}

        normalized_item: dict[str, Any] = {"name": raw_name}
        for key, value in result.items():
            if key == "name":
                continue
            if isinstance(key, str):
                normalized_item[key] = value

        if not isinstance(normalized_item.get("description"), str):
            normalized_item["description"] = ""

        return normalized_item

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
