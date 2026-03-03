"""Tests for llm_chat decorator self-reference integration behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, AsyncGenerator, cast
from unittest.mock import MagicMock, patch

import pytest

from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.hooks.events import ReActEventType, ReactEndEvent
from SimpleLLMFunc.hooks.stream import EventYield, ReactOutput
from SimpleLLMFunc.llm_decorator.llm_chat_decorator import llm_chat
from SimpleLLMFunc.self_reference import SelfReference
from SimpleLLMFunc.tool import Tool


class _DummyObservation:
    """Simple context manager used to stub Langfuse observations."""

    def __enter__(self) -> "_DummyObservation":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        return None


def test_llm_chat_binds_wrapped_agent_instance_to_self_reference() -> None:
    """SelfReference should mount the decorated callable for recursive fork use."""

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    self_reference = SelfReference()

    @llm_chat(
        llm_interface=mock_llm,
        self_reference=self_reference,
        self_reference_key="agent_main",
    )
    async def agent(message: str, history=None):
        """test agent"""

    assert self_reference.get_agent_instance() is agent


@pytest.mark.asyncio
async def test_llm_chat_does_not_auto_attach_self_reference_to_pyrepl() -> None:
    """Decorator stays decoupled and does not inject self_reference by itself."""

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = args
        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    repl = PyRepl()
    history = [{"role": "user", "content": "seed"}]

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(llm_interface=mock_llm, toolkit=cast(Any, repl.toolset))
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[Any, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _history in stream:
            pass

    assert repl.namespace.get("self_reference") is None


@pytest.mark.asyncio
async def test_llm_chat_fork_uses_isolated_pyrepl_session_toolkit() -> None:
    """Forked child should run with a cloned PyRepl toolkit session."""

    observed_toolkits: list[Any] = []

    async def fake_execute_react_loop_streaming(*args: Any, **kwargs: Any):
        _ = args
        observed_toolkits.append(kwargs.get("toolkit"))
        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    self_reference = SelfReference()
    root_repl = PyRepl()
    root_repl.install_primitive_pack("self_reference", backend=self_reference)

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, root_repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        self_reference.bind_history("agent_main", [])
        await self_reference.instance.fork("hello")

    assert observed_toolkits
    toolkit_used = observed_toolkits[-1]
    assert isinstance(toolkit_used, list)

    execute_tool = next(
        tool
        for tool in toolkit_used
        if isinstance(tool, Tool) and tool.name == "execute_code"
    )
    child_repl = getattr(execute_tool.func, "__self__", None)

    assert isinstance(child_repl, PyRepl)
    assert child_repl is not root_repl
    assert child_repl.get_runtime_backend("self_reference") is self_reference


@pytest.mark.asyncio
async def test_llm_chat_event_mode_merges_self_reference_memory_mutations() -> None:
    """Event mode should merge memory-handle edits into final history."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[ReactOutput, None]:
        _ = (args, kwargs)

        self_reference.memory["agent_main"].append(
            {"role": "user", "content": "[plan] keep me"}
        )

        yield EventYield(
            event=ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=datetime.now(timezone.utc),
                trace_id="trace-test",
                func_name="agent",
                iteration=1,
                final_response="done",
                final_messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "seed"},
                    {"role": "user", "content": "message: hello"},
                    {"role": "assistant", "content": "done"},
                ],
                total_iterations=1,
                total_execution_time=0.01,
                total_tool_calls=1,
                total_llm_calls=1,
            )
        )

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            enable_event=True,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[ReactOutput, None],
            agent("hello", history=history),
        )

        outputs: list[ReactOutput] = []
        async for output in stream:
            outputs.append(output)

    assert len(outputs) == 1
    final_output = outputs[0]
    assert isinstance(final_output, EventYield)
    assert isinstance(final_output.event, ReactEndEvent)

    final_history = final_output.event.final_messages
    assert final_history[0].get("role") == "system"
    assert final_history[0].get("content") == "test agent"
    assert final_history[1:] == [
        {"role": "user", "content": "seed"},
        {"role": "user", "content": "[plan] keep me"},
        {"role": "user", "content": "message: hello"},
        {"role": "assistant", "content": "done"},
    ]
    assert history == final_history
    assert self_reference.snapshot_history("agent_main") == final_history


@pytest.mark.asyncio
async def test_llm_chat_non_event_mode_merges_self_reference_memory_mutations() -> None:
    """Tuple mode should also merge memory-handle edits into updated history."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (args, kwargs)

        self_reference.memory["agent_main"].append(
            {"role": "user", "content": "[plan] keep me"}
        )

        yield (
            "done",
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "seed"},
                {"role": "user", "content": "message: hello"},
                {"role": "assistant", "content": "done"},
            ],
        )

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        outputs: list[tuple[str, list[dict[str, Any]]]] = []
        async for output in stream:
            outputs.append(output)

    assert len(outputs) == 1
    output_content, output_history = outputs[0]
    assert output_content == "done"
    assert output_history[0].get("role") == "system"
    assert output_history[0].get("content") == "test agent"
    assert output_history[1:] == [
        {"role": "user", "content": "seed"},
        {"role": "user", "content": "[plan] keep me"},
        {"role": "user", "content": "message: hello"},
        {"role": "assistant", "content": "done"},
    ]
    assert history == output_history
    assert self_reference.snapshot_history("agent_main") == output_history


@pytest.mark.asyncio
async def test_llm_chat_uses_function_name_as_default_self_reference_key() -> None:
    """When no key is provided, llm_chat should use function name."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()
    captured_system_prompt: str | None = None

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        nonlocal captured_system_prompt
        _ = args

        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                captured_system_prompt = maybe_prompt

        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(llm_interface=mock_llm, self_reference=self_reference)
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _history in stream:
            pass

    assert self_reference.list_history_keys() == ["agent"]
    assert self_reference.snapshot_history("agent") == history
    assert captured_system_prompt is not None
    assert "[SelfReference Memory Contract]" in captured_system_prompt
    assert 'runtime.memory.count("agent")' in captured_system_prompt
    assert "runtime.fork.run(message" in captured_system_prompt

    expected_method_lines = [
        'runtime.memory.count("agent")',
        'runtime.memory.all("agent")',
        'runtime.memory.get("agent", index)',
        'runtime.memory.append("agent", message)',
        'runtime.memory.append_system_prompt("agent", text)',
    ]
    for line in expected_method_lines:
        assert line in captured_system_prompt

    expected_forgetting_lines = [
        "- Forgetting memory:",
        "reset_repl only clears Python variables in REPL.",
        "It does NOT delete conversation memory in runtime backends.",
        'runtime.memory.delete("agent", index)',
        'runtime.memory.replace("agent", messages)',
        'runtime.memory.clear("agent")',
    ]
    for line in expected_forgetting_lines:
        assert line in captured_system_prompt


@pytest.mark.asyncio
async def test_llm_chat_persists_runtime_system_prompt_across_turns() -> None:
    """Runtime system prompt edits should be reused on subsequent turns."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()
    observed_system_prompts: list[str] = []
    call_count = 0

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        nonlocal call_count
        _ = args

        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                observed_system_prompts.append(maybe_prompt)

        current_call = call_count
        call_count += 1

        if current_call == 0:
            self_reference.memory["agent_main"].set_system_prompt("runtime system")

        yield (
            f"done-{current_call + 1}",
            [
                *messages,
                {"role": "assistant", "content": f"done-{current_call + 1}"},
            ],
        )

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """docstring system"""

        first_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 1", history=history),
        )
        async for _content, _history in first_stream:
            pass

        second_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 2", history=history),
        )
        async for _content, _history in second_stream:
            pass

    assert call_count == 2
    assert len(observed_system_prompts) == 2
    assert "docstring system" in observed_system_prompts[0]
    assert "runtime system" in observed_system_prompts[1]
    assert observed_system_prompts[0].count("[SelfReference Memory Contract]") == 1
    assert observed_system_prompts[1].count("[SelfReference Memory Contract]") == 1
    assert 'runtime.memory.count("agent_main")' in observed_system_prompts[1]
    assert history[0] == {"role": "system", "content": "runtime system"}
    assert self_reference.snapshot_history("agent_main")[0] == {
        "role": "system",
        "content": "runtime system",
    }


@pytest.mark.asyncio
async def test_append_system_prompt_persists_without_contract_pollution() -> None:
    """append_system_prompt should persist durable text, not contract block."""

    history: list[dict[str, Any]] = []
    self_reference = SelfReference()
    observed_system_prompts: list[str] = []
    call_count = 0

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        nonlocal call_count
        _ = args

        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                observed_system_prompts.append(maybe_prompt)

        if call_count == 0:
            self_reference.memory["agent_main"].append_system_prompt("Preference A")

        call_count += 1
        yield (
            f"done-{call_count}",
            [
                *messages,
                {"role": "assistant", "content": f"done-{call_count}"},
            ],
        )

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """docstring system"""

        first_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 1", history=history),
        )
        async for _content, _history in first_stream:
            pass

        second_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 2", history=history),
        )
        async for _content, _history in second_stream:
            pass

    assert call_count == 2
    assert len(observed_system_prompts) == 2
    assert "Preference A" in observed_system_prompts[1]
    assert observed_system_prompts[1].count("[SelfReference Memory Contract]") == 1

    persisted_system_prompt = self_reference.memory["agent_main"].get_system_prompt()
    assert persisted_system_prompt == "docstring system\nPreference A"


@pytest.mark.asyncio
async def test_llm_chat_deduplicates_self_reference_contract_prompt() -> None:
    """Auto-added SelfReference contract should not duplicate across turns."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()
    observed_system_prompts: list[str] = []

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = args
        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                observed_system_prompts.append(maybe_prompt)
        yield "ok", messages

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """docstring system"""

        first_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 1", history=history),
        )
        async for _content, _history in first_stream:
            pass

        second_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 2", history=history),
        )
        async for _content, _history in second_stream:
            pass

    assert len(observed_system_prompts) == 2
    assert observed_system_prompts[0].count("[SelfReference Memory Contract]") == 1
    assert observed_system_prompts[1].count("[SelfReference Memory Contract]") == 1


@pytest.mark.asyncio
async def test_llm_chat_seeds_system_prompt_into_empty_self_reference_memory() -> None:
    """Empty memory should contain seeded system prompt before first tool run."""

    history: list[dict[str, Any]] = []
    self_reference = SelfReference()
    observed_memory_lengths: list[int] = []
    observed_first_roles: list[str | None] = []

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = args
        memory_snapshot = self_reference.memory["agent_main"].all()
        observed_memory_lengths.append(len(memory_snapshot))
        if memory_snapshot:
            observed_first_roles.append(memory_snapshot[0].get("role"))
        else:
            observed_first_roles.append(None)
        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """docstring system"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _history in stream:
            pass

    assert observed_memory_lengths == [1]
    assert observed_first_roles == ["system"]
    seeded_system_prompt = self_reference.memory["agent_main"].get_system_prompt()
    assert isinstance(seeded_system_prompt, str)
    assert "docstring system" in seeded_system_prompt
    assert "[SelfReference Memory Contract]" not in seeded_system_prompt
