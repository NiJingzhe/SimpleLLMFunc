"""Tests for unified event bus."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from SimpleLLMFunc.hooks.event_bus import EventBus
from SimpleLLMFunc.hooks.events import ReActEventType, ReactStartEvent


@pytest.mark.asyncio
async def test_event_bus_emit_event_sets_origin_and_extra() -> None:
    """EventBus should attach origin metadata to EventYield and event.extra."""
    bus = EventBus(
        session_id="session-1",
        agent_call_id="agent-root",
    )

    event = ReactStartEvent(
        event_type=ReActEventType.REACT_START,
        timestamp=datetime.now(timezone.utc),
        trace_id="trace-1",
        func_name="agent",
        iteration=0,
        user_task_prompt="task",
        initial_messages=[],
        available_tools=None,
    )

    published = await bus.emit_and_get(event)

    assert published.origin.session_id == "session-1"
    assert published.origin.agent_call_id == "agent-root"
    assert published.origin.event_seq == 1
    assert isinstance(event.extra, dict)
    assert event.extra["origin"]["session_id"] == "session-1"
    assert event.extra["origin"]["agent_call_id"] == "agent-root"
    assert event.extra["origin"]["event_seq"] == 1


@pytest.mark.asyncio
async def test_event_bus_event_seq_increases() -> None:
    """Event sequence should be monotonic for one bus session."""
    bus = EventBus(session_id="session-1", agent_call_id="agent-root")

    first = ReactStartEvent(
        event_type=ReActEventType.REACT_START,
        timestamp=datetime.now(timezone.utc),
        trace_id="trace-1",
        func_name="agent",
        iteration=0,
        user_task_prompt="task-1",
        initial_messages=[],
        available_tools=None,
    )
    second = ReactStartEvent(
        event_type=ReActEventType.REACT_START,
        timestamp=datetime.now(timezone.utc),
        trace_id="trace-1",
        func_name="agent",
        iteration=0,
        user_task_prompt="task-2",
        initial_messages=[],
        available_tools=None,
    )

    first_output = await bus.emit_and_get(first)
    second_output = await bus.emit_and_get(second)

    assert first_output.origin.event_seq == 1
    assert second_output.origin.event_seq == 2
