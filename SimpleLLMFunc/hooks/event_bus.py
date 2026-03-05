"""Unified event bus for ReAct and tool events."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from typing import Any, Dict, Optional

from SimpleLLMFunc.hooks.events import CustomEvent, ReActEvent
from SimpleLLMFunc.hooks.stream import EventOrigin, EventYield


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class EventBus:
    """Single event ingress for one agent call tree."""

    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        agent_call_id: Optional[str] = None,
        parent_agent_call_id: Optional[str] = None,
        fork_id: Optional[str] = None,
        fork_depth: int = 0,
        fork_seq: Optional[int] = None,
        selfref_instance_id: Optional[str] = None,
        source_memory_key: Optional[str] = None,
        memory_key: Optional[str] = None,
    ):
        self._queue: asyncio.Queue[EventYield] = asyncio.Queue()
        self._event_seq = 0
        self._origin_template = EventOrigin(
            session_id=session_id or _new_id("session"),
            agent_call_id=agent_call_id or _new_id("agent"),
            event_seq=0,
            parent_agent_call_id=parent_agent_call_id,
            fork_id=fork_id,
            fork_depth=int(fork_depth),
            fork_seq=fork_seq,
            selfref_instance_id=selfref_instance_id,
            source_memory_key=source_memory_key,
            memory_key=memory_key,
        )

    def next_origin(self, **overrides: Any) -> EventOrigin:
        """Build next origin payload with monotonic sequence."""
        self._event_seq += 1
        origin = replace(self._origin_template, event_seq=self._event_seq)
        if overrides:
            origin = replace(origin, **overrides)
        return origin

    async def emit_event(
        self,
        event: ReActEvent,
        *,
        origin: Optional[EventOrigin] = None,
        origin_overrides: Optional[Dict[str, Any]] = None,
    ) -> EventYield:
        """Publish one event to the bus and enrich metadata."""
        resolved_origin = origin
        if resolved_origin is None:
            resolved_origin = self.next_origin(**(origin_overrides or {}))

        if not hasattr(event, "extra") or event.extra is None:
            event.extra = {}
        event.extra["origin"] = resolved_origin.as_dict()

        if isinstance(event, CustomEvent):
            if event.tool_name is None and resolved_origin.tool_name:
                event.tool_name = resolved_origin.tool_name
            if event.tool_call_id is None and resolved_origin.tool_call_id:
                event.tool_call_id = resolved_origin.tool_call_id

        output = EventYield(event=event, origin=resolved_origin)
        await self._queue.put(output)
        return output

    async def emit_and_get(
        self,
        event: ReActEvent,
        *,
        origin: Optional[EventOrigin] = None,
        origin_overrides: Optional[Dict[str, Any]] = None,
    ) -> EventYield:
        """Publish one event and return that queued output."""
        await self.emit_event(
            event,
            origin=origin,
            origin_overrides=origin_overrides,
        )
        return self._queue.get_nowait()

    async def get(self) -> EventYield:
        return await self._queue.get()

    def get_nowait(self) -> EventYield:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()


__all__ = ["EventBus"]
