"""Tests for SelfReference memory proxy behavior."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from SimpleLLMFunc.hooks.events import CustomEvent, ReActEventType
from SimpleLLMFunc.hooks.stream import EventOrigin, EventYield
from SimpleLLMFunc.self_reference import SelfReference


class _CollectingEmitter:
    """Test helper that records emitted custom events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.react_events: list[EventYield] = []

    async def emit(self, event_name: str, payload: dict[str, Any]) -> None:
        self.events.append((event_name, payload))

    async def emit_event(
        self,
        event: Any,
        *,
        origin: EventOrigin | None = None,
        origin_overrides: dict[str, Any] | None = None,
    ) -> None:
        resolved_origin = origin or EventOrigin(
            session_id="test-session",
            agent_call_id="test-agent",
            event_seq=len(self.react_events) + 1,
            fork_depth=0,
        )
        if origin_overrides:
            resolved_origin = replace(resolved_origin, **origin_overrides)
        self.react_events.append(EventYield(event=event, origin=resolved_origin))


class TestSelfReferenceMemoryProxy:
    """Validate memory-handle CRUD and helper operations."""

    def test_memory_handle_requires_bound_key(self) -> None:
        self_reference = SelfReference()

        with pytest.raises(KeyError, match="is not bound"):
            _ = self_reference.memory["agent_main"]

    def test_memory_handle_crud(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history(
            "agent_main", [{"role": "user", "content": "hello"}]
        )

        handle = self_reference.memory["agent_main"]
        handle.append({"role": "assistant", "content": "hi"})
        handle.insert(1, {"role": "user", "content": "follow up"})
        handle.update(0, {"role": "user", "content": "hello updated"})
        handle.delete(1)

        assert handle.count() == 2
        assert handle.get(0)["content"] == "hello updated"
        assert handle.get(1)["role"] == "assistant"

    def test_system_prompt_helpers(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history(
            "agent_main", [{"role": "user", "content": "hello"}]
        )

        handle = self_reference.memory["agent_main"]
        handle.set_system_prompt("Rule A")
        handle.append_system_prompt("Rule B")

        assert handle.get_system_prompt() == "Rule A\nRule B"
        assert handle.all()[0] == {"role": "system", "content": "Rule A\nRule B"}

    def test_replace_validates_tool_linkage(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [])

        handle = self_reference.memory["agent_main"]
        with pytest.raises(ValueError, match="without preceding assistant tool_calls"):
            handle.replace(
                [
                    {"role": "tool", "tool_call_id": "tc_bad", "content": "{}"},
                ]
            )

    def test_resolve_history_key_with_single_bound_key(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [])

        assert self_reference.resolve_history_key() == "agent_main"

    def test_resolve_history_key_requires_explicit_key_when_ambiguous(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [])
        self_reference.bind_history("agent_other", [])

        with pytest.raises(ValueError, match="history key is required"):
            self_reference.resolve_history_key()


class TestSelfReferenceTurnMerge:
    """Validate turn-merge behavior used by llm_chat integration."""

    def test_merge_turn_history_preserves_memory_edits(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        self_reference.memory["agent_main"].append(
            {"role": "user", "content": "[plan] keep me"}
        )

        merged = self_reference.merge_turn_history(
            key="agent_main",
            baseline_history_count=1,
            updated_history=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "seed"},
                {"role": "user", "content": "message: hello"},
                {"role": "assistant", "content": "done"},
            ],
            commit=True,
        )

        assert merged == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "seed"},
            {"role": "user", "content": "[plan] keep me"},
            {"role": "user", "content": "message: hello"},
            {"role": "assistant", "content": "done"},
        ]
        assert self_reference.snapshot_history("agent_main") == merged

    def test_merge_turn_history_prefers_runtime_system_prompt(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])
        self_reference.memory["agent_main"].set_system_prompt("runtime system")

        merged = self_reference.merge_turn_history(
            key="agent_main",
            baseline_history_count=1,
            updated_history=[
                {"role": "system", "content": "docstring system"},
                {"role": "user", "content": "seed"},
                {"role": "assistant", "content": "done"},
            ],
            commit=False,
        )

        assert merged[0] == {"role": "system", "content": "runtime system"}


class TestSelfReferenceInstanceProxy:
    """Validate self instance binding and fork behavior."""

    @pytest.mark.asyncio
    async def test_instance_fork_requires_bound_agent_instance(self) -> None:
        self_reference = SelfReference()

        with pytest.raises(RuntimeError, match="No agent instance"):
            await self_reference.instance.fork("sub-task")

    @pytest.mark.asyncio
    async def test_instance_fork_inherits_memory_snapshot_to_child_key(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        observed_calls: list[dict[str, Any]] = []

        async def fake_agent(message: str, history=None):
            observed_calls.append(
                {
                    "message": message,
                    "history": list(history or []),
                }
            )
            yield (
                f"forked:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        fork_result = await self_reference.instance.fork("sub-task")

        assert fork_result["source_memory_key"] == "agent_main"
        assert fork_result["response"] == "forked:sub-task"
        assert observed_calls[0]["history"] == [{"role": "user", "content": "seed"}]

        fork_memory_key = fork_result["memory_key"]
        assert isinstance(fork_memory_key, str)
        assert fork_memory_key.startswith("agent_main::fork::")

        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "seed"}
        ]
        assert self_reference.snapshot_history(fork_memory_key) == [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "child done"},
        ]

    @pytest.mark.asyncio
    async def test_nested_fork_defaults_to_current_fork_memory_key(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history(
            "agent_main",
            [{"role": "user", "content": "root seed"}],
        )

        async def fake_agent(message: str, history=None):
            if message == "spawn-child":
                nested = await self_reference.instance.fork("leaf-task")
                yield (
                    nested,
                    list(history or []),
                )
                return

            yield (
                "leaf-done",
                [
                    *(history or []),
                    {"role": "assistant", "content": "leaf done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        parent_fork = await self_reference.instance.fork("spawn-child")
        child_fork = parent_fork["response"]

        assert parent_fork["source_memory_key"] == "agent_main"
        assert parent_fork["memory_key"].startswith("agent_main::fork::")

        assert isinstance(child_fork, dict)
        assert child_fork["source_memory_key"] == parent_fork["memory_key"]
        assert child_fork["memory_key"].startswith(
            f"{parent_fork['memory_key']}::fork::"
        )
        assert self_reference.snapshot_history(child_fork["memory_key"])[-1] == {
            "role": "assistant",
            "content": "leaf done",
        }

    @pytest.mark.asyncio
    async def test_instance_fork_spawn_and_wait(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            await asyncio.sleep(0.01)
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": f"child:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        spawned = await self_reference.instance.fork_spawn("task-a")
        assert spawned["status"] == "running"
        assert spawned["fork_id"].startswith("fork_")

        completed = await self_reference.instance.fork_wait(spawned["fork_id"])
        assert completed["status"] == "completed"
        assert completed["response"] == "done:task-a"
        assert completed["memory_key"] == spawned["memory_key"]
        assert self_reference.snapshot_history(completed["memory_key"])[-1] == {
            "role": "assistant",
            "content": "child:task-a",
        }

    @pytest.mark.asyncio
    async def test_instance_fork_wait_all_collects_spawned_children(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            await asyncio.sleep(0.01)
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": f"child:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        first = await self_reference.instance.fork_spawn("task-a")
        second = await self_reference.instance.fork_spawn("task-b")

        all_results = await self_reference.instance.fork_wait_all(
            [first["fork_id"], second["fork_id"]]
        )
        assert set(all_results.keys()) == {first["fork_id"], second["fork_id"]}
        assert all(result["status"] == "completed" for result in all_results.values())

    @pytest.mark.asyncio
    async def test_instance_fork_emits_child_stream_events(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            yield (
                "alpha ",
                list(history or []),
            )
            yield (
                "beta\n",
                list(history or []),
            )
            yield (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        emitter = _CollectingEmitter()

        result = await self_reference.instance.fork(
            "task-a",
            _event_emitter=emitter,
        )

        event_names = [name for name, _ in emitter.events]
        assert "selfref_fork_stream_open" in event_names
        assert "selfref_fork_stream_delta" in event_names
        assert "selfref_fork_stream_close" in event_names

        delta_texts = [
            payload.get("text")
            for name, payload in emitter.events
            if name == "selfref_fork_stream_delta"
        ]
        assert delta_texts == ["alpha ", "beta\n", "done:task-a"]

        for name, payload in emitter.events:
            if not name.startswith("selfref_fork_stream"):
                continue
            assert payload.get("fork_id") == result["fork_id"]
            assert payload.get("memory_key") == result["memory_key"]

    @pytest.mark.asyncio
    async def test_instance_fork_stream_close_marks_error_when_child_fails(
        self,
    ) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(_message: str, history=None):
            yield (
                "partial",
                list(history or []),
            )
            raise RuntimeError("boom")

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        emitter = _CollectingEmitter()

        with pytest.raises(RuntimeError, match="boom"):
            await self_reference.instance.fork(
                "task-a",
                _event_emitter=emitter,
            )

        delta_texts = [
            payload.get("text")
            for name, payload in emitter.events
            if name == "selfref_fork_stream_delta"
        ]
        assert delta_texts == ["partial"]

        close_events = [
            payload
            for name, payload in emitter.events
            if name == "selfref_fork_stream_close"
        ]
        assert len(close_events) == 1
        assert close_events[0].get("status") == "error"

        error_events = [
            payload for name, payload in emitter.events if name == "selfref_fork_error"
        ]
        assert len(error_events) == 1
        assert error_events[0].get("error_type") == "RuntimeError"
        assert error_events[0].get("error_message") == "boom"

    @pytest.mark.asyncio
    async def test_instance_fork_forwards_child_events(self) -> None:
        """Fork should forward child EventYield payloads as direct events."""
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            yield EventYield(
                event=CustomEvent(
                    event_type=ReActEventType.CUSTOM_EVENT,
                    timestamp=datetime.now(timezone.utc),
                    trace_id="trace-child",
                    func_name="agent",
                    iteration=1,
                    event_name="child_progress",
                    data={"message": message},
                )
            )
            yield (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        emitter = _CollectingEmitter()

        result = await self_reference.instance.fork(
            "task-a",
            _event_emitter=emitter,
        )

        forwarded_events = [
            event_yield.event
            for event_yield in emitter.react_events
            if isinstance(event_yield.event, CustomEvent)
        ]
        assert len(forwarded_events) == 1
        assert forwarded_events[0].event_name == "child_progress"

        forwarded_origin = emitter.react_events[0].origin
        assert forwarded_origin.fork_id == result["fork_id"]
        assert forwarded_origin.memory_key == result["memory_key"]
        assert forwarded_origin.fork_depth == result["depth"]
