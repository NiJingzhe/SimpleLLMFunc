"""Self-reference runtime primitives for agent memory management.

This module provides a framework-level ``SelfReference`` object that can be
shared between ``llm_chat`` and tool runtimes (for example ``PyRepl``).
"""

from __future__ import annotations

import asyncio
import copy
import contextvars
import inspect
import threading
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

_ALLOWED_ROLES = {"system", "user", "assistant", "tool", "function"}
_TRANSIENT_MESSAGE_FIELDS = {"reasoning_details"}
MemoryHistory = List[Dict[str, Any]]
HISTORY_PARAM_NAMES = ("history", "chat_history")
SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM = "__self_reference_key_override"
SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM = "__self_reference_toolkit_override"
_AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR = "__simplellmfunc_accepts_template_params__"
_AGENT_FORK_TOOLKIT_FACTORY_ATTR = "__simplellmfunc_fork_toolkit_factory__"


def _normalize_key(key: str) -> str:
    if not isinstance(key, str):
        raise ValueError("key must be a non-empty string")
    normalized = key.strip()
    if not normalized:
        raise ValueError("key must be a non-empty string")
    return normalized


def _clone_messages(messages: MemoryHistory) -> MemoryHistory:
    cloned: MemoryHistory = []
    for message in messages:
        sanitized = copy.deepcopy(message)
        for field in _TRANSIENT_MESSAGE_FIELDS:
            sanitized.pop(field, None)
        cloned.append(sanitized)
    return cloned


def _is_valid_content_for_role(role: str, content: Any) -> bool:
    if role == "system":
        return isinstance(content, str)
    if role == "user":
        return isinstance(content, (str, list))
    if role == "assistant":
        return content is None or isinstance(content, (str, list))
    if role == "tool":
        return isinstance(content, str)
    if role == "function":
        return isinstance(content, str)
    return False


def _validate_message_shape(message: Dict[str, Any], index: int) -> None:
    role = message.get("role")
    if not isinstance(role, str) or role not in _ALLOWED_ROLES:
        raise ValueError(f"Invalid message role at index {index}: {role!r}")

    if "content" not in message:
        raise ValueError(
            f"Message at index {index} is missing required field 'content'"
        )

    if not _is_valid_content_for_role(role, message.get("content")):
        raise ValueError(
            f"Invalid content for role '{role}' at index {index}: "
            f"{type(message.get('content')).__name__}"
        )

    if role == "assistant" and "tool_calls" in message:
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise ValueError("assistant.tool_calls must be a list when present")
        if isinstance(tool_calls, list):
            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    raise ValueError(
                        "assistant.tool_calls entries must be dict objects "
                        f"(index {index}, tool_call {call_index})"
                    )
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(
                        "assistant.tool_calls entries must contain non-empty id "
                        f"(index {index}, tool_call {call_index})"
                    )

    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ValueError("tool messages must contain non-empty tool_call_id")


def _validate_tool_linkage(messages: MemoryHistory) -> None:
    pending_tool_call_ids: List[str] = []

    for index, message in enumerate(messages):
        role = message.get("role")

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if pending_tool_call_ids and not tool_calls:
                raise ValueError(
                    "Missing tool results before next assistant message "
                    f"(index {index})"
                )
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    call_id = tool_call.get("id")
                    if isinstance(call_id, str) and call_id:
                        pending_tool_call_ids.append(call_id)
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not pending_tool_call_ids:
                raise ValueError(
                    "Tool message appears without preceding assistant tool_calls "
                    f"(index {index})"
                )
            if tool_call_id not in pending_tool_call_ids:
                raise ValueError(
                    "tool_call_id does not match pending assistant tool_calls "
                    f"(index {index})"
                )
            pending_tool_call_ids.remove(tool_call_id)
            continue

        if pending_tool_call_ids:
            raise ValueError(
                "Pending assistant tool_calls must be followed by matching tool "
                f"messages before role '{role}' (index {index})"
            )

    if pending_tool_call_ids:
        raise ValueError("Unmatched assistant tool_calls without tool results")


def _validate_history_for_memory_methods(messages: MemoryHistory) -> None:
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"history item at index {index} must be a dict")
        _validate_message_shape(message, index)

    _validate_tool_linkage(messages)


def _coerce_history_list(history: List[Any]) -> MemoryHistory:
    if not isinstance(history, list):
        raise ValueError("history must be List[Dict[str, Any]]")
    if not all(isinstance(item, dict) for item in history):
        raise ValueError("history must be List[Dict[str, Any]]")
    return _clone_messages(history)


def _filter_non_system_messages(messages: MemoryHistory) -> MemoryHistory:
    filtered: MemoryHistory = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if "role" in message and "content" in message:
            filtered.append(copy.deepcopy(message))
    return filtered


def _extract_latest_system_prompt(messages: MemoryHistory) -> Optional[str]:
    for message in reversed(messages):
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    return None


def _extract_history_from_any(value: Any) -> Optional[MemoryHistory]:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, dict) for item in value):
        return None
    return _coerce_history_list(value)


def _extract_response_and_history_from_output(
    output: Any,
) -> tuple[Any, Optional[MemoryHistory]]:
    if isinstance(output, tuple) and len(output) == 2:
        maybe_history = _extract_history_from_any(output[1])
        if maybe_history is not None:
            return output[0], maybe_history

    event = getattr(output, "event", None)
    if event is not None:
        maybe_history = _extract_history_from_any(
            getattr(event, "final_messages", None)
        )
        if maybe_history is not None:
            return getattr(event, "final_response", None), maybe_history

    return output, None


def _normalize_fork_ids(value: Optional[List[Any]]) -> Optional[List[str]]:
    if value is None:
        return None

    if not isinstance(value, list):
        raise ValueError("fork_ids must be a list of fork_id strings")

    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized.append(_normalize_key(item))
            continue

        if isinstance(item, dict):
            fork_id = item.get("fork_id")
            if isinstance(fork_id, str) and fork_id.strip():
                normalized.append(_normalize_key(fork_id))
                continue

        raise ValueError("fork_ids must contain fork_id strings or dicts with fork_id")

    return normalized


def _extract_text_from_model_like_response(response: Any) -> str:
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str):
            return content

    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]

    delta = getattr(first_choice, "delta", None)
    if delta is not None:
        delta_content = getattr(delta, "content", None)
        if isinstance(delta_content, str):
            return delta_content

    message = getattr(first_choice, "message", None)
    if message is not None:
        message_content = getattr(message, "content", None)
        if isinstance(message_content, str):
            return message_content

    return ""


def _extract_stream_text_from_agent_output(output: Any) -> str:
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], str):
        return output[0]

    if getattr(output, "type", None) == "response":
        return _extract_text_from_model_like_response(getattr(output, "response", None))

    return ""


def _extract_event_and_origin_from_agent_output(output: Any) -> tuple[Any, Any]:
    if getattr(output, "type", None) != "event":
        return None, None
    return getattr(output, "event", None), getattr(output, "origin", None)


async def _consume_agent_call_output(
    call_output: Any,
    *,
    event_emitter: Any = None,
    fork_id: Optional[str] = None,
    parent_fork_id: Optional[str] = None,
    depth: Optional[int] = None,
    source_memory_key: Optional[str] = None,
    memory_key: Optional[str] = None,
) -> tuple[Any, Optional[MemoryHistory]]:
    if inspect.isawaitable(call_output):
        awaited_output = await cast(Awaitable[Any], call_output)
        return _extract_response_and_history_from_output(awaited_output)

    if hasattr(call_output, "__aiter__"):
        last_response: Any = None
        last_history: Optional[MemoryHistory] = None
        stream_forwarding_enabled = (
            fork_id is not None
            and depth is not None
            and source_memory_key is not None
            and memory_key is not None
        )

        async for output in call_output:
            if stream_forwarding_enabled:
                forwarded_event, forwarded_origin = (
                    _extract_event_and_origin_from_agent_output(output)
                )
                if forwarded_event is not None:
                    await _emit_fork_agent_event(
                        event_emitter,
                        forwarded_event,
                        fork_id=cast(str, fork_id),
                        depth=cast(int, depth),
                        source_memory_key=cast(str, source_memory_key),
                        memory_key=cast(str, memory_key),
                        forwarded_origin=forwarded_origin,
                    )

            response, history = _extract_response_and_history_from_output(output)
            last_response = response
            if history is not None:
                last_history = history

        return last_response, last_history

    return _extract_response_and_history_from_output(call_output)


def _extract_history_param_name(agent_instance: Any) -> Optional[str]:
    try:
        signature = inspect.signature(agent_instance)
    except (TypeError, ValueError):
        return None

    for candidate in HISTORY_PARAM_NAMES:
        if candidate in signature.parameters:
            return candidate

    return None


def _agent_supports_template_params(agent_instance: Any) -> bool:
    if bool(getattr(agent_instance, _AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR, False)):
        return True

    try:
        signature = inspect.signature(agent_instance)
    except (TypeError, ValueError):
        return False

    if "_template_params" in signature.parameters:
        return True

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return False


def _get_agent_fork_toolkit_factory(
    agent_instance: Any,
) -> Optional[Callable[[Any], Any]]:
    maybe_factory = getattr(agent_instance, _AGENT_FORK_TOOLKIT_FACTORY_ATTR, None)
    if callable(maybe_factory):
        return cast(Optional[Callable[[Any], Any]], maybe_factory)
    return None


def _build_fork_error_result(
    *,
    fork_id: str,
    source_memory_key: str,
    memory_key: str,
    parent_fork_id: Optional[str],
    depth: int,
    error: BaseException,
) -> Dict[str, Any]:
    return {
        "fork_id": fork_id,
        "parent_fork_id": parent_fork_id,
        "depth": depth,
        "source_memory_key": source_memory_key,
        "memory_key": memory_key,
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "response": None,
        "history": [],
        "history_count": 0,
        "history_included": True,
    }


async def _emit_fork_custom_event(
    event_emitter: Any,
    event_name: str,
    payload: Dict[str, Any],
) -> None:
    if event_emitter is None:
        return

    emit = getattr(event_emitter, "emit", None)
    if not callable(emit):
        return

    maybe_awaitable = emit(event_name, payload)
    if inspect.isawaitable(maybe_awaitable):
        await cast(Awaitable[Any], maybe_awaitable)


async def _emit_fork_agent_event(
    event_emitter: Any,
    event: Any,
    *,
    fork_id: str,
    depth: int,
    source_memory_key: str,
    memory_key: str,
    forwarded_origin: Any = None,
) -> None:
    if event_emitter is None:
        return

    emit_event = getattr(event_emitter, "emit_event", None)
    if not callable(emit_event):
        return

    resolved_origin_overrides: Dict[str, Any]
    forwarded_fork_id = getattr(forwarded_origin, "fork_id", None)
    if isinstance(forwarded_fork_id, str) and forwarded_fork_id:
        resolved_origin_overrides = {
            "fork_id": forwarded_fork_id,
        }

        forwarded_depth = getattr(forwarded_origin, "fork_depth", None)
        if isinstance(forwarded_depth, int):
            resolved_origin_overrides["fork_depth"] = forwarded_depth
        else:
            resolved_origin_overrides["fork_depth"] = depth

        forwarded_source_memory_key = getattr(
            forwarded_origin,
            "source_memory_key",
            None,
        )
        if isinstance(forwarded_source_memory_key, str) and forwarded_source_memory_key:
            resolved_origin_overrides["source_memory_key"] = forwarded_source_memory_key
        else:
            resolved_origin_overrides["source_memory_key"] = source_memory_key

        forwarded_memory_key = getattr(forwarded_origin, "memory_key", None)
        if isinstance(forwarded_memory_key, str) and forwarded_memory_key:
            resolved_origin_overrides["memory_key"] = forwarded_memory_key
        else:
            resolved_origin_overrides["memory_key"] = memory_key

        forwarded_tool_name = getattr(forwarded_origin, "tool_name", None)
        if isinstance(forwarded_tool_name, str) and forwarded_tool_name:
            resolved_origin_overrides["tool_name"] = forwarded_tool_name

        forwarded_tool_call_id = getattr(forwarded_origin, "tool_call_id", None)
        if isinstance(forwarded_tool_call_id, str) and forwarded_tool_call_id:
            resolved_origin_overrides["tool_call_id"] = forwarded_tool_call_id
    else:
        resolved_origin_overrides = {
            "fork_id": fork_id,
            "fork_depth": depth,
            "source_memory_key": source_memory_key,
            "memory_key": memory_key,
        }

    maybe_awaitable = emit_event(
        event,
        origin_overrides=resolved_origin_overrides,
    )
    if inspect.isawaitable(maybe_awaitable):
        await cast(Awaitable[Any], maybe_awaitable)


class SelfReferenceMemoryHandle:
    """Keyed memory view used by ``self_reference.memory[<key>]``."""

    def __init__(self, owner: "SelfReference", key: str):
        self._owner = owner
        self._key = key

    def count(self) -> int:
        return len(self._owner.snapshot_history(self._key))

    def all(self) -> MemoryHistory:
        return self._owner.snapshot_history(self._key)

    def get(self, index: int) -> Dict[str, Any]:
        messages = self._owner.snapshot_history(self._key)
        return copy.deepcopy(messages[index])

    def append(self, message: Dict[str, Any]) -> None:
        self._owner.append_message(self._key, message)

    def insert(self, index: int, message: Dict[str, Any]) -> None:
        self._owner.insert_message(self._key, index, message)

    def update(self, index: int, message: Dict[str, Any]) -> None:
        self._owner.update_message(self._key, index, message)

    def delete(self, index: int) -> None:
        self._owner.delete_message(self._key, index)

    def replace(self, messages: List[Dict[str, Any]]) -> None:
        self._owner.replace_history(self._key, messages, strict=True)

    def clear(self) -> None:
        system_prompt = self._owner.get_system_prompt(self._key)
        replacement: List[Dict[str, Any]] = []
        if system_prompt is not None:
            replacement.append({"role": "system", "content": system_prompt})
        self._owner.replace_history(self._key, replacement, strict=True)

    def get_system_prompt(self) -> Optional[str]:
        return self._owner.get_system_prompt(self._key)

    def set_system_prompt(self, text: str) -> None:
        self._owner.set_system_prompt(self._key, text)

    def append_system_prompt(self, text: str) -> None:
        self._owner.append_system_prompt(self._key, text)


class SelfReferenceMemoryProxy:
    """Container object exposed as ``self_reference.memory``."""

    def __init__(self, owner: "SelfReference"):
        self._owner = owner

    def __getitem__(self, key: str) -> SelfReferenceMemoryHandle:
        normalized_key = _normalize_key(key)
        if not self._owner.has_history(normalized_key):
            raise KeyError(
                f"Memory key '{normalized_key}' is not bound. "
                "Bind it before using self_reference.memory[key]."
            )
        return SelfReferenceMemoryHandle(self._owner, normalized_key)

    def keys(self) -> List[str]:
        return self._owner.list_history_keys()

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._owner.has_history(key)


class SelfReferenceInstanceHandle:
    """Container object exposed as ``self_reference.instance``."""

    def __init__(self, owner: "SelfReference"):
        self._owner = owner

    def is_bound(self) -> bool:
        return self._owner.get_agent_instance() is not None

    def get(self) -> Optional[Any]:
        return self._owner.get_agent_instance()

    async def fork(
        self,
        *agent_args: Any,
        source_memory_key: Optional[str] = None,
        fork_memory_key: Optional[str] = None,
        _event_emitter: Any = None,
        include_history: bool = False,
        **agent_kwargs: Any,
    ) -> Dict[str, Any]:
        return await self._owner.fork_agent_instance(
            *agent_args,
            source_memory_key=source_memory_key,
            fork_memory_key=fork_memory_key,
            _event_emitter=_event_emitter,
            include_history=include_history,
            **agent_kwargs,
        )

    async def fork_spawn(
        self,
        *agent_args: Any,
        source_memory_key: Optional[str] = None,
        fork_memory_key: Optional[str] = None,
        _event_emitter: Any = None,
        **agent_kwargs: Any,
    ) -> Dict[str, Any]:
        return await self._owner.spawn_agent_instance(
            *agent_args,
            source_memory_key=source_memory_key,
            fork_memory_key=fork_memory_key,
            _event_emitter=_event_emitter,
            **agent_kwargs,
        )

    async def fork_wait(
        self,
        fork_id: str,
        include_history: bool = False,
    ) -> Dict[str, Any]:
        return await self._owner.wait_fork_result(
            fork_id,
            include_history=include_history,
        )

    async def fork_wait_all(
        self,
        fork_ids: Optional[List[str]] = None,
        include_history: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        return await self._owner.wait_all_fork_results(
            fork_ids,
            include_history=include_history,
        )


class SelfReference:
    """Shared self-reference state object for agent memory operations."""

    def __init__(self):
        self._lock = threading.RLock()
        self._history_store: Dict[str, MemoryHistory] = {}
        self._memory_proxy = SelfReferenceMemoryProxy(self)
        self._instance_proxy = SelfReferenceInstanceHandle(self)
        self._agent_instance: Optional[Any] = None
        self._agent_default_memory_key: Optional[str] = None
        self._fork_counter = 0
        self._fork_id_counter = 0
        self._fork_tasks: Dict[str, asyncio.Task[Dict[str, Any]]] = {}
        self._fork_results: Dict[str, Dict[str, Any]] = {}
        self._active_memory_key_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_key_{id(self)}",
                default=None,
            )
        )
        self._active_fork_id_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_fork_id_{id(self)}",
                default=None,
            )
        )
        self._active_fork_depth_var: contextvars.ContextVar[int] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_fork_depth_{id(self)}",
                default=0,
            )
        )
        self._active_runtime_toolkit_var: contextvars.ContextVar[Any] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_toolkit_{id(self)}",
                default=None,
            )
        )

    @property
    def memory(self) -> SelfReferenceMemoryProxy:
        return self._memory_proxy

    @property
    def instance(self) -> SelfReferenceInstanceHandle:
        return self._instance_proxy

    def bind_agent_instance(
        self,
        agent_instance: Any,
        default_memory_key: Optional[str] = None,
    ) -> None:
        """Bind top-level agent callable for recursive self-fork use-cases."""

        if not callable(agent_instance):
            raise ValueError("agent_instance must be callable")

        normalized_default_key: Optional[str] = None
        if default_memory_key is not None:
            normalized_default_key = _normalize_key(default_memory_key)

        with self._lock:
            self._agent_instance = agent_instance
            self._agent_default_memory_key = normalized_default_key

    def get_agent_instance(self) -> Optional[Any]:
        with self._lock:
            return self._agent_instance

    def get_agent_default_memory_key(self) -> Optional[str]:
        with self._lock:
            return self._agent_default_memory_key

    def _set_active_memory_key(self, key: str) -> contextvars.Token[Optional[str]]:
        return self._active_memory_key_var.set(_normalize_key(key))

    def _reset_active_memory_key(self, token: contextvars.Token[Optional[str]]) -> None:
        self._active_memory_key_var.reset(token)

    def _get_active_memory_key(self) -> Optional[str]:
        return self._active_memory_key_var.get()

    def _set_active_fork_context(
        self,
        fork_id: str,
        depth: int,
    ) -> tuple[contextvars.Token[Optional[str]], contextvars.Token[int]]:
        fork_token = self._active_fork_id_var.set(fork_id)
        depth_token = self._active_fork_depth_var.set(depth)
        return fork_token, depth_token

    def _reset_active_fork_context(
        self,
        tokens: tuple[contextvars.Token[Optional[str]], contextvars.Token[int]],
    ) -> None:
        fork_token, depth_token = tokens
        self._active_fork_id_var.reset(fork_token)
        self._active_fork_depth_var.reset(depth_token)

    def _get_active_fork_id(self) -> Optional[str]:
        return self._active_fork_id_var.get()

    def _get_active_fork_depth(self) -> int:
        return self._active_fork_depth_var.get()

    def _set_active_runtime_toolkit(self, toolkit: Any) -> contextvars.Token[Any]:
        return self._active_runtime_toolkit_var.set(toolkit)

    def _reset_active_runtime_toolkit(self, token: contextvars.Token[Any]) -> None:
        self._active_runtime_toolkit_var.reset(token)

    def _get_active_runtime_toolkit(self) -> Any:
        return self._active_runtime_toolkit_var.get()

    def bind_history(self, key: str, history: List[Dict[str, Any]]) -> None:
        normalized_key = _normalize_key(key)
        normalized_history = _coerce_history_list(history)
        with self._lock:
            self._history_store[normalized_key] = normalized_history

    def unbind_history(self, key: str) -> None:
        normalized_key = _normalize_key(key)
        with self._lock:
            self._history_store.pop(normalized_key, None)

    def list_history_keys(self) -> List[str]:
        with self._lock:
            keys = list(self._history_store.keys())
        keys.sort()
        return keys

    def has_history(self, key: str) -> bool:
        normalized_key = _normalize_key(key)
        with self._lock:
            return normalized_key in self._history_store

    def snapshot_history(self, key: str) -> MemoryHistory:
        normalized_key = _normalize_key(key)
        with self._lock:
            if normalized_key not in self._history_store:
                raise KeyError(f"Memory key '{normalized_key}' is not bound")
            return _clone_messages(self._history_store[normalized_key])

    def filtered_history_count(self, key: str) -> int:
        messages = self.snapshot_history(key)
        return len(_filter_non_system_messages(messages))

    def merge_turn_history(
        self,
        key: str,
        baseline_history_count: int,
        updated_history: List[Dict[str, Any]],
        commit: bool = False,
    ) -> MemoryHistory:
        normalized_key = _normalize_key(key)
        updated_history_dicts = _coerce_history_list(updated_history)

        with self._lock:
            runtime_messages = _clone_messages(
                self._history_store.get(normalized_key, [])
            )

        runtime_non_system = _filter_non_system_messages(runtime_messages)
        runtime_system_prompt = _extract_latest_system_prompt(runtime_messages)

        if baseline_history_count < 0:
            baseline_history_count = 0

        merged: MemoryHistory = []

        updated_system_prompt: Optional[str] = None
        if updated_history_dicts:
            first = updated_history_dicts[0]
            first_content = first.get("content")
            if first.get("role") == "system" and isinstance(first_content, str):
                updated_system_prompt = first_content

        effective_system_prompt = runtime_system_prompt or updated_system_prompt
        if effective_system_prompt is not None:
            merged.append({"role": "system", "content": effective_system_prompt})

        tail_start = baseline_history_count
        if updated_system_prompt is not None:
            tail_start += 1

        tail_start = min(tail_start, len(updated_history_dicts))
        merged.extend(runtime_non_system)
        merged.extend(_clone_messages(updated_history_dicts[tail_start:]))

        if commit:
            with self._lock:
                self._history_store[normalized_key] = _clone_messages(merged)

        return merged

    def replace_history(
        self,
        key: str,
        messages: List[Dict[str, Any]],
        strict: bool = False,
    ) -> None:
        normalized_key = _normalize_key(key)
        normalized_messages = _coerce_history_list(messages)
        if strict:
            _validate_history_for_memory_methods(normalized_messages)

        with self._lock:
            if normalized_key not in self._history_store:
                raise KeyError(f"Memory key '{normalized_key}' is not bound")
            self._history_store[normalized_key] = normalized_messages

    def append_message(self, key: str, message: Dict[str, Any]) -> None:
        self._mutate_messages(key, lambda msgs: msgs.append(copy.deepcopy(message)))

    def insert_message(self, key: str, index: int, message: Dict[str, Any]) -> None:
        self._mutate_messages(
            key,
            lambda msgs: msgs.insert(index, copy.deepcopy(message)),
        )

    def update_message(self, key: str, index: int, message: Dict[str, Any]) -> None:
        def mutate(messages: MemoryHistory) -> None:
            messages[index] = copy.deepcopy(message)

        self._mutate_messages(key, mutate)

    def delete_message(self, key: str, index: int) -> None:
        def mutate(messages: MemoryHistory) -> None:
            messages.pop(index)

        self._mutate_messages(key, mutate)

    def get_system_prompt(self, key: str) -> Optional[str]:
        messages = self.snapshot_history(key)
        return _extract_latest_system_prompt(messages)

    def set_system_prompt(self, key: str, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError("system prompt text must be a string")

        def mutate(messages: MemoryHistory) -> None:
            non_system = [msg for msg in messages if msg.get("role") != "system"]
            messages.clear()
            messages.append({"role": "system", "content": text})
            messages.extend(non_system)

        self._mutate_messages(key, mutate)

    def append_system_prompt(self, key: str, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError("system prompt text must be a string")

        current = self.get_system_prompt(key)
        if current:
            updated = f"{current}\n{text}"
        else:
            updated = text
        self.set_system_prompt(key, updated)

    def resolve_history_key(self, key: Optional[str] = None) -> str:
        """Resolve one usable history key for runtime self-reference operations.

        Resolution order when ``key`` is omitted:
        1. Active key in current execution context.
        2. Bound default key from ``bind_agent_instance``.
        3. The only bound key when exactly one exists.
        """

        if key is not None:
            normalized = _normalize_key(key)
            if not self.has_history(normalized):
                raise KeyError(f"Memory key '{normalized}' is not bound")
            return normalized

        active_key = self._get_active_memory_key()
        if active_key is not None and self.has_history(active_key):
            return active_key

        default_key = self.get_agent_default_memory_key()
        if default_key is not None:
            if not self.has_history(default_key):
                self.bind_history(default_key, [])
            return default_key

        keys = self.list_history_keys()
        if len(keys) == 1:
            return keys[0]

        if not keys:
            raise ValueError(
                "history key is required because no memory key is available"
            )

        raise ValueError("history key is required when multiple memory keys are bound")

    def _resolve_source_memory_key_for_fork(
        self,
        source_memory_key: Optional[str],
    ) -> str:
        try:
            return self.resolve_history_key(source_memory_key)
        except ValueError as exc:
            message = str(exc)
            if message == "history key is required because no memory key is available":
                raise ValueError(
                    "source_memory_key is required because no memory key is available"
                ) from exc
            if message == "history key is required when multiple memory keys are bound":
                raise ValueError(
                    "source_memory_key is required when multiple memory keys are bound"
                ) from exc
            raise

    def _build_fork_memory_key(self, source_memory_key: str) -> str:
        with self._lock:
            while True:
                self._fork_counter += 1
                candidate = f"{source_memory_key}::fork::{self._fork_counter}"
                if candidate not in self._history_store:
                    return candidate

    def _build_fork_id(self) -> str:
        with self._lock:
            self._fork_id_counter += 1
            return f"fork_{self._fork_id_counter}"

    def _resolve_child_toolkit_override(self, agent_instance: Any) -> Any:
        parent_runtime_toolkit = self._get_active_runtime_toolkit()
        toolkit_factory = _get_agent_fork_toolkit_factory(agent_instance)
        if toolkit_factory is None:
            return parent_runtime_toolkit

        try:
            return toolkit_factory(parent_runtime_toolkit)
        except Exception:
            return parent_runtime_toolkit

    def _build_fork_template_params(
        self,
        existing_template_params: Any,
        fork_memory_key: str,
        toolkit_override: Any,
    ) -> Dict[str, Any]:
        if existing_template_params is None:
            merged_template_params: Dict[str, Any] = {}
        elif isinstance(existing_template_params, dict):
            merged_template_params = copy.deepcopy(existing_template_params)
        else:
            raise ValueError("_template_params must be a dict when provided")

        merged_template_params[SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM] = (
            fork_memory_key
        )
        if toolkit_override is not None:
            merged_template_params[SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM] = (
                toolkit_override
            )
        return merged_template_params

    def _history_length(self, key: str) -> int:
        normalized_key = _normalize_key(key)
        with self._lock:
            if normalized_key not in self._history_store:
                raise KeyError(f"Memory key '{normalized_key}' is not bound")
            return len(self._history_store[normalized_key])

    def _compact_fork_result_payload(self, result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            compact_result = copy.deepcopy(result)
        except Exception:
            compact_result = dict(result)
        compact_result.pop("history", None)

        history_count = compact_result.get("history_count")
        if not isinstance(history_count, int):
            memory_key = compact_result.get("memory_key")
            if isinstance(memory_key, str) and memory_key:
                try:
                    history_count = self._history_length(memory_key)
                except Exception:
                    history_count = 0
            else:
                history_count = 0
            compact_result["history_count"] = history_count

        compact_result["history_included"] = False
        return compact_result

    def _materialize_fork_result_payload(
        self,
        result: Dict[str, Any],
        *,
        include_history: bool,
    ) -> Dict[str, Any]:
        materialized = self._compact_fork_result_payload(result)
        if not include_history:
            return materialized

        memory_key = materialized.get("memory_key")
        history_snapshot: MemoryHistory = []
        if isinstance(memory_key, str) and memory_key:
            try:
                history_snapshot = self.snapshot_history(memory_key)
            except Exception:
                history_snapshot = []

        materialized["history"] = history_snapshot
        materialized["history_count"] = len(history_snapshot)
        materialized["history_included"] = True
        return materialized

    async def fork_agent_instance(
        self,
        *agent_args: Any,
        source_memory_key: Optional[str] = None,
        fork_memory_key: Optional[str] = None,
        _event_emitter: Any = None,
        include_history: bool = False,
        _fork_id: Optional[str] = None,
        _parent_fork_id: Optional[str] = None,
        _fork_depth: Optional[int] = None,
        **agent_kwargs: Any,
    ) -> Dict[str, Any]:
        with self._lock:
            agent_instance = self._agent_instance

        if agent_instance is None:
            raise RuntimeError("No agent instance is bound to self_reference")

        source_key = self._resolve_source_memory_key_for_fork(source_memory_key)
        inherited_history = self.snapshot_history(source_key)

        if fork_memory_key is None:
            target_key = self._build_fork_memory_key(source_key)
        else:
            target_key = _normalize_key(fork_memory_key)

        fork_id = _fork_id if _fork_id is not None else self._build_fork_id()
        parent_fork_id = (
            _parent_fork_id
            if _parent_fork_id is not None
            else self._get_active_fork_id()
        )
        fork_depth = (
            _fork_depth
            if _fork_depth is not None
            else self._get_active_fork_depth() + 1
        )

        await _emit_fork_custom_event(
            _event_emitter,
            "selfref_fork_start",
            {
                "fork_id": fork_id,
                "parent_fork_id": parent_fork_id,
                "depth": fork_depth,
                "source_memory_key": source_key,
                "memory_key": target_key,
                "status": "running",
            },
        )

        self.bind_history(target_key, inherited_history)

        call_kwargs = dict(agent_kwargs)
        history_param_name = _extract_history_param_name(agent_instance)
        if history_param_name is not None and history_param_name not in call_kwargs:
            call_kwargs[history_param_name] = self.snapshot_history(target_key)

        child_toolkit_override = self._resolve_child_toolkit_override(agent_instance)

        if _agent_supports_template_params(agent_instance):
            call_kwargs["_template_params"] = self._build_fork_template_params(
                call_kwargs.get("_template_params"),
                target_key,
                child_toolkit_override,
            )

        active_key_token = self._set_active_memory_key(target_key)
        active_fork_tokens = self._set_active_fork_context(
            fork_id=fork_id,
            depth=fork_depth,
        )
        active_toolkit_token = self._set_active_runtime_toolkit(child_toolkit_override)
        try:
            call_output = agent_instance(*agent_args, **call_kwargs)
            response, final_history = await _consume_agent_call_output(
                call_output,
                event_emitter=_event_emitter,
                fork_id=fork_id,
                parent_fork_id=parent_fork_id,
                depth=fork_depth,
                source_memory_key=source_key,
                memory_key=target_key,
            )
        except Exception as exc:
            await _emit_fork_custom_event(
                _event_emitter,
                "selfref_fork_error",
                {
                    "fork_id": fork_id,
                    "parent_fork_id": parent_fork_id,
                    "depth": fork_depth,
                    "source_memory_key": source_key,
                    "memory_key": target_key,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise
        finally:
            self._reset_active_memory_key(active_key_token)
            self._reset_active_fork_context(active_fork_tokens)
            self._reset_active_runtime_toolkit(active_toolkit_token)

        if final_history is not None:
            self.bind_history(target_key, final_history)

        completed_result = self._materialize_fork_result_payload(
            {
                "fork_id": fork_id,
                "parent_fork_id": parent_fork_id,
                "depth": fork_depth,
                "source_memory_key": source_key,
                "memory_key": target_key,
                "status": "completed",
                "response": response,
            },
            include_history=include_history,
        )

        await _emit_fork_custom_event(
            _event_emitter,
            "selfref_fork_end",
            {
                "fork_id": fork_id,
                "parent_fork_id": parent_fork_id,
                "depth": fork_depth,
                "source_memory_key": source_key,
                "memory_key": target_key,
                "status": "completed",
            },
        )

        return completed_result

    async def spawn_agent_instance(
        self,
        *agent_args: Any,
        source_memory_key: Optional[str] = None,
        fork_memory_key: Optional[str] = None,
        _event_emitter: Any = None,
        **agent_kwargs: Any,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._agent_instance is None:
                raise RuntimeError("No agent instance is bound to self_reference")

        source_key = self._resolve_source_memory_key_for_fork(source_memory_key)
        if fork_memory_key is None:
            target_key = self._build_fork_memory_key(source_key)
        else:
            target_key = _normalize_key(fork_memory_key)

        fork_id = self._build_fork_id()
        parent_fork_id = self._get_active_fork_id()
        fork_depth = self._get_active_fork_depth() + 1

        fork_task = asyncio.create_task(
            self.fork_agent_instance(
                *agent_args,
                source_memory_key=source_key,
                fork_memory_key=target_key,
                _event_emitter=_event_emitter,
                _fork_id=fork_id,
                _parent_fork_id=parent_fork_id,
                _fork_depth=fork_depth,
                **agent_kwargs,
            )
        )

        def on_fork_task_done(done_task: asyncio.Task[Dict[str, Any]]) -> None:
            try:
                done_result = done_task.result()
            except BaseException as exc:
                done_result = _build_fork_error_result(
                    fork_id=fork_id,
                    source_memory_key=source_key,
                    memory_key=target_key,
                    parent_fork_id=parent_fork_id,
                    depth=fork_depth,
                    error=exc,
                )

            compact_result = self._compact_fork_result_payload(done_result)

            with self._lock:
                self._fork_results[fork_id] = compact_result
                self._fork_tasks.pop(fork_id, None)

        fork_task.add_done_callback(on_fork_task_done)

        with self._lock:
            self._fork_tasks[fork_id] = fork_task
            self._fork_results.pop(fork_id, None)

        await _emit_fork_custom_event(
            _event_emitter,
            "selfref_fork_spawned",
            {
                "fork_id": fork_id,
                "parent_fork_id": parent_fork_id,
                "depth": fork_depth,
                "source_memory_key": source_key,
                "memory_key": target_key,
                "status": "running",
            },
        )

        return {
            "fork_id": fork_id,
            "parent_fork_id": parent_fork_id,
            "depth": fork_depth,
            "source_memory_key": source_key,
            "memory_key": target_key,
            "status": "running",
        }

    async def wait_fork_result(
        self,
        fork_id: str,
        include_history: bool = False,
    ) -> Dict[str, Any]:
        normalized_fork_id = _normalize_key(fork_id)

        with self._lock:
            completed_result = self._fork_results.get(normalized_fork_id)
            running_task = self._fork_tasks.get(normalized_fork_id)

        if completed_result is not None:
            return self._materialize_fork_result_payload(
                completed_result,
                include_history=include_history,
            )

        if running_task is None:
            raise KeyError(f"fork_id '{normalized_fork_id}' is not found")

        try:
            await running_task
        except Exception:
            pass

        await asyncio.sleep(0)

        with self._lock:
            result_after_wait = self._fork_results.get(normalized_fork_id)

        if result_after_wait is None and running_task.done():
            try:
                direct_result = running_task.result()
            except Exception as exc:
                result_after_wait = {
                    "fork_id": normalized_fork_id,
                    "parent_fork_id": None,
                    "depth": 0,
                    "source_memory_key": "",
                    "memory_key": "",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "response": None,
                    "history": [],
                    "history_count": 0,
                    "history_included": True,
                }
            else:
                result_after_wait = direct_result

            compact_result = self._compact_fork_result_payload(result_after_wait)

            with self._lock:
                self._fork_results[normalized_fork_id] = compact_result

            result_after_wait = compact_result

        if result_after_wait is None:
            raise KeyError(f"fork_id '{normalized_fork_id}' has no result")

        return self._materialize_fork_result_payload(
            result_after_wait,
            include_history=include_history,
        )

    async def wait_all_fork_results(
        self,
        fork_ids: Optional[List[Any]] = None,
        include_history: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Wait for multiple fork results; accepts fork_id strings or fork-handle dicts."""
        normalized_ids = _normalize_fork_ids(cast(Optional[List[Any]], fork_ids))

        if normalized_ids is None:
            with self._lock:
                target_ids = sorted(
                    set(self._fork_tasks.keys()) | set(self._fork_results.keys())
                )
        else:
            target_ids = normalized_ids

        collected: Dict[str, Dict[str, Any]] = {}
        for target_id in target_ids:
            collected[target_id] = await self.wait_fork_result(
                target_id,
                include_history=include_history,
            )

        return collected

    def _mutate_messages(
        self,
        key: str,
        mutator: Callable[[MemoryHistory], None],
    ) -> None:
        normalized_key = _normalize_key(key)

        with self._lock:
            if normalized_key not in self._history_store:
                raise KeyError(f"Memory key '{normalized_key}' is not bound")
            messages = _clone_messages(self._history_store[normalized_key])

        mutator(messages)
        messages = _clone_messages(messages)
        _validate_history_for_memory_methods(messages)

        with self._lock:
            self._history_store[normalized_key] = messages


__all__ = [
    "SelfReference",
    "SelfReferenceMemoryHandle",
    "SelfReferenceMemoryProxy",
    "SelfReferenceInstanceHandle",
    "SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM",
    "SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM",
]
