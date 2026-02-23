"""Self-reference runtime primitives for agent memory management.

This module provides a framework-level ``SelfReference`` object that can be
shared between ``llm_chat`` and tool runtimes (for example ``PyRepl``).
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Callable, Dict, List, Optional

_ALLOWED_ROLES = {"system", "user", "assistant", "tool", "function"}
MemoryHistory = List[Dict[str, Any]]


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


class SelfReference:
    """Shared self-reference state object for agent memory operations."""

    def __init__(self):
        self._lock = threading.RLock()
        self._history_store: Dict[str, MemoryHistory] = {}
        self._memory_proxy = SelfReferenceMemoryProxy(self)
        self._agent_instance: Optional[Any] = None

    @property
    def memory(self) -> SelfReferenceMemoryProxy:
        return self._memory_proxy

    def bind_agent_instance(self, agent_instance: Any) -> None:
        """Bind top-level agent instance for future recursive use-cases."""
        with self._lock:
            self._agent_instance = agent_instance

    def get_agent_instance(self) -> Optional[Any]:
        with self._lock:
            return self._agent_instance

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
        self._mutate_messages(key, lambda msgs: msgs.pop(index))

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


__all__ = ["SelfReference", "SelfReferenceMemoryHandle", "SelfReferenceMemoryProxy"]
