"""Tests for custom tool event emitter feature."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import pytest

from SimpleLLMFunc.hooks.events import (
    CustomEvent,
    ReactStartEvent,
    ReActEventType,
)
from SimpleLLMFunc.hooks.stream import EventOrigin
from SimpleLLMFunc.hooks.event_emitter import (
    ToolEventEmitter,
    NoOpEventEmitter,
)


class TestToolEventEmitter:
    """Test ToolEventEmitter class."""

    @pytest.mark.asyncio
    async def test_emit_custom_event(self):
        """Test emitting a custom event."""
        emitter = ToolEventEmitter(
            _trace_id="test-trace-123",
            _func_name="test_func",
            _iteration=1,
        )

        await emitter.emit("progress", {"percent": 50})

        assert emitter.has_events()
        events = await emitter.get_events()
        assert len(events) == 1

        event = events[0].event
        assert isinstance(event, CustomEvent)
        assert event.event_name == "progress"
        assert event.data == {"percent": 50}
        assert event.trace_id == "test-trace-123"
        assert event.func_name == "test_func"
        assert event.iteration == 1

    @pytest.mark.asyncio
    async def test_emit_custom_event_with_tool_context(self):
        """Test emitting custom event carries tool metadata."""
        emitter = ToolEventEmitter(
            _trace_id="test-trace-123",
            _func_name="test_func",
            _iteration=1,
            _tool_name="execute_code",
            _tool_call_id="call_abc",
        )

        await emitter.emit("kernel_stdout", {"text": "hello\n"})

        events = await emitter.get_events()
        assert len(events) == 1

        event = events[0].event
        assert isinstance(event, CustomEvent)
        assert event.tool_name == "execute_code"
        assert event.tool_call_id == "call_abc"

    @pytest.mark.asyncio
    async def test_emit_multiple_events(self):
        """Test emitting multiple custom events."""
        emitter = ToolEventEmitter()

        await emitter.emit("step1", {"value": 1})
        await emitter.emit("step2", {"value": 2})
        await emitter.emit("step3", {"value": 3})

        events = await emitter.get_events()
        assert len(events) == 3

        assert events[0].event.event_name == "step1"
        assert events[1].event.event_name == "step2"
        assert events[2].event.event_name == "step3"

    @pytest.mark.asyncio
    async def test_emit_batch(self):
        """Test batch emitting events."""
        emitter = ToolEventEmitter()

        await emitter.emit_batch(
            [
                ("step1", {"value": 1}),
                ("step2", {"value": 2}),
            ]
        )

        events = await emitter.get_events()
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_events_clears_queue(self):
        """Test that get_events clears the queue."""
        emitter = ToolEventEmitter()

        await emitter.emit("test", {})
        await emitter.get_events()

        assert not emitter.has_events()

    @pytest.mark.asyncio
    async def test_has_events_false_when_empty(self):
        """Test has_events returns False when no events."""
        emitter = ToolEventEmitter()
        assert not emitter.has_events()

    @pytest.mark.asyncio
    async def test_emit_event_keeps_event_and_origin(self):
        """emit_event should push event with explicit origin metadata."""
        emitter = ToolEventEmitter(
            _trace_id="test-trace-123",
            _func_name="test_func",
            _iteration=1,
        )
        event = ReactStartEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace-123",
            func_name="test_func",
            iteration=0,
            user_task_prompt="hello",
            initial_messages=[],
            available_tools=None,
        )
        origin = EventOrigin(
            session_id="session-1",
            agent_call_id="agent-root",
            event_seq=2,
            fork_depth=0,
        )

        await emitter.emit_event(event, origin=origin)

        events = await emitter.get_events()
        assert len(events) == 1
        assert events[0].event is event
        assert events[0].origin.agent_call_id == "agent-root"
        assert events[0].origin.event_seq == 2


class TestNoOpEventEmitter:
    """Test NoOpEventEmitter class."""

    @pytest.mark.asyncio
    async def test_emit_does_nothing(self):
        """Test that emit does nothing."""
        emitter = NoOpEventEmitter()
        await emitter.emit("test", {})
        # Should not raise any error

    @pytest.mark.asyncio
    async def test_emit_batch_does_nothing(self):
        """Test that emit_batch does nothing."""
        emitter = NoOpEventEmitter()
        await emitter.emit_batch([("test", {})])
        # Should not raise any error

    @pytest.mark.asyncio
    async def test_emit_event_does_nothing(self):
        """Test that emit_event does nothing."""
        emitter = NoOpEventEmitter()
        event = ReactStartEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace",
            func_name="test_func",
            iteration=0,
            user_task_prompt="hello",
            initial_messages=[],
            available_tools=None,
        )
        await emitter.emit_event(event)

    @pytest.mark.asyncio
    async def test_get_events_returns_empty(self):
        """Test that get_events returns empty list."""
        emitter = NoOpEventEmitter()
        events = await emitter.get_events()
        assert events == []

    @pytest.mark.asyncio
    async def test_has_events_returns_false(self):
        """Test that has_events returns False."""
        emitter = NoOpEventEmitter()
        assert not emitter.has_events()


class TestCustomEvent:
    """Test CustomEvent class."""

    def test_custom_event_creation(self):
        """Test creating a custom event."""
        from datetime import datetime, timezone

        event = CustomEvent(
            event_type=ReActEventType.CUSTOM_EVENT,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace",
            func_name="test_func",
            iteration=1,
            event_name="my_event",
            data={"key": "value"},
        )

        assert event.event_name == "my_event"
        assert event.data == {"key": "value"}
        assert event.event_type == ReActEventType.CUSTOM_EVENT

    def test_custom_event_with_none_data(self):
        """Test custom event with None data."""
        from datetime import datetime, timezone

        event = CustomEvent(
            event_type=ReActEventType.CUSTOM_EVENT,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace",
            func_name="test_func",
            iteration=1,
            event_name="progress",
        )

        assert event.event_name == "progress"
        assert event.data is None

    def test_custom_event_tool_context_defaults_to_none(self):
        """Test custom event tool context defaults."""
        from datetime import datetime, timezone

        event = CustomEvent(
            event_type=ReActEventType.CUSTOM_EVENT,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace",
            func_name="test_func",
            iteration=1,
            event_name="progress",
            data={"step": 1},
        )

        assert event.tool_name is None
        assert event.tool_call_id is None
