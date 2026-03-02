"""Self-reference runtime primitives for agent memory management.

This module provides a framework-level ``SelfReference`` object that can be
shared between ``llm_chat`` and tool runtimes (for example ``PyRepl``).
"""

from __future__ import annotations

import copy
import contextvars
import inspect
import threading
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

_ALLOWED_ROLES = {"system", "user", "assistant", "tool", "function"}
MemoryHistory = List[Dict[str, Any]]
HISTORY_PARAM_NAMES = ("history", "chat_history")
SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM = "__self_reference_key_override"
_AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR = "__simplellmfunc_accepts_template_params__"


def _normalize_key(key: str) -> str:
    if not isinstance(key, str):
        raise ValueError("key must be a non-empty string")
    normalized = key.strip()
    if not normalized:
        raise ValueError("key must be a non-empty string")
    return normalized


def _clone_messages(messages: MemoryHistory) -> MemoryHistory:
    return copy.deepcopy(messages)


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


async def _consume_agent_call_output(
    call_output: Any,
) -> tuple[Any, Optional[MemoryHistory]]:
    if inspect.isawaitable(call_output):
        awaited_output = await cast(Awaitable[Any], call_output)
        return _extract_response_and_history_from_output(awaited_output)

    if hasattr(call_output, "__aiter__"):
        last_response: Any = None
        last_history: Optional[MemoryHistory] = None

        async for output in call_output:
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
        self._owner.replace_history(self._key, [], strict=True)

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
        **agent_kwargs: Any,
    ) -> Dict[str, Any]:
        return await self._owner.fork_agent_instance(
            *agent_args,
            source_memory_key=source_memory_key,
            fork_memory_key=fork_memory_key,
            **agent_kwargs,
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
        self._active_memory_key_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_key_{id(self)}",
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

    def _resolve_source_memory_key_for_fork(
        self,
        source_memory_key: Optional[str],
    ) -> str:
        if source_memory_key is not None:
            normalized = _normalize_key(source_memory_key)
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
                "source_memory_key is required because no memory key is available"
            )

        raise ValueError(
            "source_memory_key is required when multiple memory keys are bound"
        )

    def _build_fork_memory_key(self, source_memory_key: str) -> str:
        with self._lock:
            while True:
                self._fork_counter += 1
                candidate = f"{source_memory_key}::fork::{self._fork_counter}"
                if candidate not in self._history_store:
                    return candidate

    def _build_fork_template_params(
        self,
        existing_template_params: Any,
        fork_memory_key: str,
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
        return merged_template_params

    async def fork_agent_instance(
        self,
        *agent_args: Any,
        source_memory_key: Optional[str] = None,
        fork_memory_key: Optional[str] = None,
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

        self.bind_history(target_key, inherited_history)

        call_kwargs = dict(agent_kwargs)
        history_param_name = _extract_history_param_name(agent_instance)
        if history_param_name is not None and history_param_name not in call_kwargs:
            call_kwargs[history_param_name] = self.snapshot_history(target_key)

        if _agent_supports_template_params(agent_instance):
            call_kwargs["_template_params"] = self._build_fork_template_params(
                call_kwargs.get("_template_params"),
                target_key,
            )

        active_key_token = self._set_active_memory_key(target_key)
        try:
            call_output = agent_instance(*agent_args, **call_kwargs)
            response, final_history = await _consume_agent_call_output(call_output)
        finally:
            self._reset_active_memory_key(active_key_token)

        if final_history is not None:
            self.bind_history(target_key, final_history)

        return {
            "source_memory_key": source_key,
            "memory_key": target_key,
            "response": response,
            "history": self.snapshot_history(target_key),
        }

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
        _validate_history_for_memory_methods(messages)

        with self._lock:
            self._history_store[normalized_key] = messages


__all__ = [
    "SelfReference",
    "SelfReferenceMemoryHandle",
    "SelfReferenceMemoryProxy",
    "SelfReferenceInstanceHandle",
    "SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM",
]
