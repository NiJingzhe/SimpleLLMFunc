"""Tests for llm_chat decorator self-reference integration behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, AsyncGenerator, cast
from unittest.mock import MagicMock, patch

import pytest

from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.hooks.events import ReActEventType, ReactEndEvent
from SimpleLLMFunc.hooks.stream import EventOrigin, EventYield, ReactOutput
from SimpleLLMFunc.llm_decorator.llm_chat_decorator import llm_chat
from SimpleLLMFunc.self_reference import SelfReference
from SimpleLLMFunc.tool import Tool


_MUST_PROMPT_BLOCK = "<must_principles>"
_MUST_PROMPT_RULE = (
    "Never use chat-style XML text in assistant messages to invoke tools"
)


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


def test_llm_chat_strict_signature_enforces_history_message_shape() -> None:
    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    @llm_chat(llm_interface=mock_llm, strict_signature=True)
    async def agent(history, message: str, _template_params=None):
        """test agent"""

    assert callable(agent)


def test_llm_chat_strict_signature_rejects_non_canonical_shapes() -> None:
    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with pytest.raises(TypeError, match="first parameter"):

        @llm_chat(llm_interface=mock_llm, strict_signature=True)
        async def bad_agent(message: str, history=None):
            """bad agent"""

    with pytest.raises(TypeError, match="second parameter"):

        @llm_chat(llm_interface=mock_llm, strict_signature=True)
        async def bad_agent2(history, message):
            """bad agent"""

    with pytest.raises(TypeError, match="second parameter name"):

        @llm_chat(llm_interface=mock_llm, strict_signature=True)
        async def bad_agent3(history, user_message: str):
            """bad agent"""

    with pytest.raises(TypeError, match="only allows"):

        @llm_chat(llm_interface=mock_llm, strict_signature=True)
        async def bad_agent4(history, message: str, extra: int):
            """bad agent"""


@pytest.mark.asyncio
async def test_llm_chat_does_not_auto_attach_self_reference_to_pyrepl() -> None:
    """Decorator stays decoupled and does not inject self_reference by itself."""

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
    assert captured_system_prompt is not None
    assert "[Runtime Primitive Contract]" not in captured_system_prompt
    assert "<tool_best_practices>" in captured_system_prompt
    assert "execute_code" in captured_system_prompt
    assert "<runtime_primitive_contract>" in captured_system_prompt
    assert "runtime.list_primitives()" in captured_system_prompt
    assert "runtime.list_primitives(prefix='...')" in captured_system_prompt
    assert "runtime.get_primitive_spec(name)" in captured_system_prompt
    assert "runtime.list_primitive_specs(names=[...])" in captured_system_prompt
    assert _MUST_PROMPT_BLOCK in captured_system_prompt
    assert _MUST_PROMPT_RULE in captured_system_prompt
    assert "永远不要" not in captured_system_prompt
    assert "必须通过模型原生" not in captured_system_prompt
    assert captured_system_prompt.index(
        "<tool_best_practices>"
    ) < captured_system_prompt.index("test agent")
    assert captured_system_prompt.rfind(
        _MUST_PROMPT_BLOCK
    ) > captured_system_prompt.index("test agent")
    assert "Mounted primitive summary:" not in captured_system_prompt
    assert "Use memory key" not in captured_system_prompt


@pytest.mark.asyncio
async def test_llm_chat_auto_resolves_self_reference_from_pyrepl_backend() -> None:
    """Decorator should auto-resolve SelfReference from mounted PyRepl runtime backend."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
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

    self_reference = SelfReference()
    repl = PyRepl()
    repl.install_primitive_pack("selfref", backend=self_reference)

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
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _history in stream:
            pass

    assert self_reference.get_agent_instance() is agent
    assert self_reference.list_history_keys() == ["agent"]
    assert captured_system_prompt is not None
    assert "[Runtime Primitive Contract]" not in captured_system_prompt
    assert "<tool_best_practices>" in captured_system_prompt
    assert "<runtime_primitive_contract>" in captured_system_prompt
    assert "<progressive_disclosure>" in captured_system_prompt
    assert "runtime.get_primitive_spec(name)" in captured_system_prompt
    assert "do not dump full primitive spec lists" in captured_system_prompt
    assert "<spec_rule>For each primitive" in captured_system_prompt
    assert "<fork_result_safety>" in captured_system_prompt
    assert "NEVER print raw fork result dicts" in captured_system_prompt
    assert "Do not treat gather_all result as a list" in captured_system_prompt
    assert _MUST_PROMPT_BLOCK in captured_system_prompt
    assert _MUST_PROMPT_RULE in captured_system_prompt
    assert "<active_selfref_key>agent</active_selfref_key>" in captured_system_prompt
    assert "Mounted primitive summary:" not in captured_system_prompt


@pytest.mark.asyncio
async def test_llm_chat_injects_active_selfref_key_for_runtime_history_ops() -> None:
    """Runtime selfref history calls without key should use decorator memory key."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = args

        runtime_toolkit = kwargs["toolkit"]
        execute_tool = next(
            tool
            for tool in runtime_toolkit
            if isinstance(tool, Tool) and tool.name == "execute_code"
        )
        execute_func = cast(Any, execute_tool.func)

        execute_result = await execute_func(
            "runtime.selfref.history.append({'role': 'assistant', 'content': 'from-selfref'})"
        )
        assert execute_result["success"] is True

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
    self_reference.bind_history(
        "agent_main", [{"role": "user", "content": "seed-main"}]
    )
    self_reference.bind_history("other", [{"role": "user", "content": "seed-other"}])

    repl = PyRepl()
    repl.install_primitive_pack("selfref", backend=self_reference)

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
            toolkit=cast(Any, repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _updated_history in stream:
            pass

    main_history = self_reference.snapshot_history("agent_main")
    other_history = self_reference.snapshot_history("other")
    assert any(
        item.get("role") == "assistant" and item.get("content") == "from-selfref"
        for item in main_history
    )
    assert other_history == [{"role": "user", "content": "seed-other"}]


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
    root_repl.install_primitive_pack("selfref", backend=self_reference)

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
    assert child_repl.get_runtime_backend("selfref") is self_reference
    assert child_repl._closed is True
    assert root_repl._closed is False


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
async def test_llm_chat_event_mode_ignores_fork_react_end_for_history_merge() -> None:
    """Fork-scoped ReactEndEvent should not be merged into main memory."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()

    child_final_messages = [
        {"role": "system", "content": "child system"},
        {"role": "user", "content": "child prompt"},
        {"role": "assistant", "content": "child done"},
    ]

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
                trace_id="trace-child",
                func_name="agent",
                iteration=1,
                final_response="child done",
                final_messages=child_final_messages,
                total_iterations=1,
                total_execution_time=0.01,
                total_tool_calls=0,
                total_llm_calls=1,
            ),
            origin=EventOrigin(
                session_id="trace-main",
                agent_call_id="agent-main",
                event_seq=1,
                fork_id="fork_1",
                fork_depth=1,
                source_memory_key="agent_main",
                memory_key="agent_main::fork::1",
            ),
        )

        yield EventYield(
            event=ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=datetime.now(timezone.utc),
                trace_id="trace-main",
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

    assert len(outputs) == 2

    child_output = outputs[0]
    assert isinstance(child_output, EventYield)
    assert isinstance(child_output.event, ReactEndEvent)
    assert child_output.event.final_messages == child_final_messages

    final_output = outputs[-1]
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
    assert not any(item.get("content") == "child prompt" for item in final_history)
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
    assert "test agent" in captured_system_prompt
    assert _MUST_PROMPT_BLOCK in captured_system_prompt
    assert _MUST_PROMPT_RULE in captured_system_prompt
    assert captured_system_prompt.strip().endswith("</must_principles>")
    assert "[Runtime Primitive Contract]" not in captured_system_prompt


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
    assert _MUST_PROMPT_BLOCK in observed_system_prompts[0]
    assert _MUST_PROMPT_BLOCK in observed_system_prompts[1]
    assert "[Runtime Primitive Contract]" not in observed_system_prompts[0]
    assert "[Runtime Primitive Contract]" not in observed_system_prompts[1]
    assert history[0] == {"role": "system", "content": "runtime system"}
    assert self_reference.snapshot_history("agent_main")[0] == {
        "role": "system",
        "content": "runtime system",
    }


@pytest.mark.asyncio
async def test_append_system_prompt_persists_without_contract_pollution() -> None:
    """append_system_prompt should persist durable text, not runtime contract block."""

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
    assert _MUST_PROMPT_BLOCK in observed_system_prompts[0]
    assert _MUST_PROMPT_BLOCK in observed_system_prompts[1]
    assert "[Runtime Primitive Contract]" not in observed_system_prompts[1]

    persisted_system_prompt = self_reference.memory["agent_main"].get_system_prompt()
    assert persisted_system_prompt == "docstring system\nPreference A"


@pytest.mark.asyncio
async def test_llm_chat_deduplicates_runtime_primitive_contract_prompt() -> None:
    """Runtime guidance should stay deduplicated in Tool Best Practices."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()
    repl = PyRepl()
    repl.install_primitive_pack("selfref", backend=self_reference)
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
            toolkit=cast(Any, repl.toolset),
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
    assert observed_system_prompts[0].count("[Runtime Primitive Contract]") == 0
    assert observed_system_prompts[1].count("[Runtime Primitive Contract]") == 0
    assert observed_system_prompts[0].count("<tool_best_practices>") == 1
    assert observed_system_prompts[1].count("<tool_best_practices>") == 1
    assert observed_system_prompts[0].count("<runtime_primitive_contract>") == 1
    assert observed_system_prompts[1].count("<runtime_primitive_contract>") == 1
    assert observed_system_prompts[0].count(_MUST_PROMPT_BLOCK) == 1
    assert observed_system_prompts[1].count(_MUST_PROMPT_BLOCK) == 1


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
    assert "[Runtime Primitive Contract]" not in seeded_system_prompt
    assert "[SelfReference Memory Contract]" not in seeded_system_prompt
    assert _MUST_PROMPT_BLOCK not in seeded_system_prompt
