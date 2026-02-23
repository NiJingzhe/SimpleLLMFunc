"""Tests for SelfReference memory proxy behavior."""

from __future__ import annotations

import pytest

from SimpleLLMFunc.self_reference import SelfReference


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
