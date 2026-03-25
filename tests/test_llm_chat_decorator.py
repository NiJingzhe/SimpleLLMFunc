"""Tests for llm_chat decorator self-reference integration behavior."""

from __future__ import annotations

import contextvars
import inspect
import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional, cast
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as ChatCompletionMessageToolCall,
    Function,
)

from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.hooks.events import ReActEventType, ReactEndEvent
from SimpleLLMFunc.hooks.stream import EventOrigin, EventYield, ReactOutput
from SimpleLLMFunc.llm_decorator.llm_chat_decorator import (
    DEFAULT_MAX_TOOL_CALLS,
    llm_chat,
)
from SimpleLLMFunc.observability.langfuse_client import (
    langfuse_client as shared_langfuse_client,
    reset_langfuse_trace_context,
    set_langfuse_trace_context,
)
from SimpleLLMFunc.self_reference import SelfReference
from SimpleLLMFunc.tool import Tool


_MUST_PROMPT_BLOCK = "<must_principles>"
_MUST_PROMPT_RULE = (
    "Invoke tools through native structured tool_calls / function-calling fields"
)


class _DummyObservation:
    """Simple context manager used to stub Langfuse observations."""

    def __enter__(self) -> "_DummyObservation":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        return None


class _TrackingObservation:
    """Context manager that tracks nested Langfuse observations."""

    def __init__(
        self,
        tracker: "_TrackingLangfuseClient",
        record: dict[str, Any],
    ) -> None:
        self._tracker = tracker
        self._record = record
        self._trace_token: Optional[contextvars.Token[Optional[str]]] = None
        self._observation_token: Optional[contextvars.Token[Optional[str]]] = None

    def __enter__(self) -> "_TrackingObservation":
        self._trace_token = self._tracker._trace_id_var.set(self._record["trace_id"])
        self._observation_token = self._tracker._observation_id_var.set(
            self._record["span_id"]
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._observation_token is not None:
            self._tracker._observation_id_var.reset(self._observation_token)
        if self._trace_token is not None:
            self._tracker._trace_id_var.reset(self._trace_token)
        return None

    def update(self, **kwargs: Any) -> None:
        self._record.setdefault("updates", []).append(kwargs)


class _TrackingLangfuseClient:
    """Tiny contextvar-backed Langfuse stub for trace propagation tests."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._counter = 0
        self._trace_id_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar("test_langfuse_trace_id", default=None)
        )
        self._observation_id_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar("test_langfuse_observation_id", default=None)
        )

    def start_as_current_observation(self, **kwargs: Any) -> _TrackingObservation:
        self._counter += 1
        trace_context = kwargs.get("trace_context")
        current_trace_id = self._trace_id_var.get()
        trace_id = ""
        if isinstance(trace_context, dict):
            raw_trace_id = trace_context.get("trace_id")
            if isinstance(raw_trace_id, str):
                trace_id = raw_trace_id
        if not trace_id:
            trace_id = current_trace_id or f"trace_{self._counter}"

        parent_span_id: Optional[str] = None
        if isinstance(trace_context, dict):
            raw_parent_span_id = trace_context.get("parent_span_id")
            if isinstance(raw_parent_span_id, str) and raw_parent_span_id:
                parent_span_id = raw_parent_span_id
        if parent_span_id is None:
            parent_span_id = self._observation_id_var.get()

        record = {
            "span_id": f"obs_{self._counter}",
            "as_type": kwargs.get("as_type"),
            "name": kwargs.get("name"),
            "input": kwargs.get("input"),
            "metadata": kwargs.get("metadata"),
            "trace_id": trace_id,
            "parent_span_id": parent_span_id,
            "trace_context": trace_context,
        }
        self.records.append(record)
        return _TrackingObservation(self, record)

    def get_current_trace_id(self) -> str:
        return self._trace_id_var.get() or ""

    def get_current_observation_id(self) -> str:
        return self._observation_id_var.get() or ""

    def create_trace_id(self) -> str:
        self._counter += 1
        return f"trace_{self._counter}"


def _make_chat_completion(content: Optional[str]) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(finish_reason="stop", index=0, message=message)
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )


def _make_tool_call_completion(
    tool_name: str, arguments: Dict[str, Any]
) -> ChatCompletion:
    tool_call = ChatCompletionMessageToolCall(
        id="call_execute_code",
        function=Function(
            name=tool_name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
        type="function",
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )
    choice = Choice(
        finish_reason="tool_calls",
        index=0,
        message=message,
    )
    return ChatCompletion(
        id="test-tool-call",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )


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


def test_llm_chat_default_max_tool_calls_is_none() -> None:
    """llm_chat should not impose a default tool-call limit."""

    signature = inspect.signature(llm_chat)

    assert DEFAULT_MAX_TOOL_CALLS is None
    assert signature.parameters["max_tool_calls"].default is None


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
async def test_llm_chat_auto_resolves_builtin_self_reference_from_pyrepl() -> None:
    """Decorator should pick up the builtin selfref pack from a default PyRepl."""

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
    self_reference = repl.get_runtime_backend("selfref")
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

    assert isinstance(self_reference, SelfReference)
    assert self_reference.get_agent_instance() is agent
    assert self_reference.list_history_keys() == ["agent"]
    assert repl.namespace.get("self_reference") is None
    assert captured_system_prompt is not None
    assert "[Runtime Primitive Contract]" not in captured_system_prompt
    assert "<tool_best_practices>" in captured_system_prompt
    assert "execute_code" in captured_system_prompt
    assert "<runtime_primitive_contract>" in captured_system_prompt
    assert "Installed primitive packs:" in captured_system_prompt
    assert "- selfref:" in captured_system_prompt
    assert captured_system_prompt.count("runtime.list_primitives()") == 1
    assert (
        captured_system_prompt.count("runtime.list_primitives(contains='<namespace>.')")
        == 1
    )
    assert captured_system_prompt.count("runtime.get_primitive_spec(name)") == 1
    assert (
        captured_system_prompt.count("runtime.list_primitive_specs(contains='...')")
        == 1
    )
    assert _MUST_PROMPT_BLOCK in captured_system_prompt
    assert _MUST_PROMPT_RULE in captured_system_prompt
    assert "Active selfref key: agent" in captured_system_prompt
    assert (
        "Use assistant content for natural-language reasoning and final responses."
        in captured_system_prompt
    )
    assert (
        "Keep tool invocation payloads in the native tool channel."
        in captured_system_prompt
    )
    assert captured_system_prompt.index(
        "<tool_best_practices>"
    ) < captured_system_prompt.index("test agent")
    assert captured_system_prompt.rfind(
        _MUST_PROMPT_BLOCK
    ) > captured_system_prompt.index("test agent")
    assert (
        "You can use the following tools flexibly according to the real case and tool description:"
        not in captured_system_prompt
    )
    assert (
        "For fork results, read status/response/memory_key/history_count; if status is error, inspect error_type/error_message before retrying."
        not in captured_system_prompt
    )
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
    repl = PyRepl(self_reference=self_reference)

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
    assert captured_system_prompt.count("runtime.get_primitive_spec(name)") == 1
    assert (
        "keep prompt context focused on the selected primitives"
        in captured_system_prompt
    )
    assert "Installed primitive packs:" in captured_system_prompt
    assert "- selfref:" in captured_system_prompt
    assert captured_system_prompt.count("selfref = your agent state") == 1
    assert (
        "Summarize the selected result fields in chat responses."
        in captured_system_prompt
    )
    assert (
        "Treat runtime.selfref.fork.gather_all results as dict[fork_id -> ForkResult] and iterate with .items() or .values()."
        in captured_system_prompt
    )
    assert _MUST_PROMPT_BLOCK in captured_system_prompt
    assert _MUST_PROMPT_RULE in captured_system_prompt
    assert "Active selfref key: agent" in captured_system_prompt
    assert (
        "You can use the following tools flexibly according to the real case and tool description:"
        not in captured_system_prompt
    )
    assert (
        "For fork results, read status/response/memory_key/history_count; if status is error, inspect error_type/error_message before retrying."
        not in captured_system_prompt
    )
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
        execute_result = await execute_tool.run(
            "runtime.selfref.history.append({'role': 'assistant', 'content': 'from-selfref'})"
        )
        assert "Execution succeeded" in execute_result

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

    repl = PyRepl(self_reference=self_reference)

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
    root_repl = PyRepl(self_reference=self_reference)

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
async def test_llm_chat_fork_clones_custom_pyrepl_pack_primitives() -> None:
    """Forked child should preserve first-class custom PrimitivePack installs."""

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
    root_repl = PyRepl(self_reference=self_reference)

    constants_pack = root_repl.pack(
        "constants",
        backend={"project": "SimpleLLMFunc", "branch": "feat/better-primitive-dev"},
    )

    @constants_pack.primitive("get")
    def constants_get(ctx, key: str):
        """
        Use: Read one value from constants backend.
        Input: `key: str`.
        Output: `str | None`.
        Best Practices:
        - Prefer reading a single key per call.
        """
        backend = ctx.backend
        if not isinstance(backend, dict):
            raise RuntimeError("constants backend must be a dict")
        return backend.get(key)

    root_repl.install_pack(constants_pack)

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
    execute_tool = next(
        tool
        for tool in toolkit_used
        if isinstance(tool, Tool) and tool.name == "execute_code"
    )
    child_repl = getattr(execute_tool.func, "__self__", None)

    assert isinstance(child_repl, PyRepl)
    assert child_repl is not root_repl
    assert child_repl.get_runtime_backend("constants") == {
        "project": "SimpleLLMFunc",
        "branch": "feat/better-primitive-dev",
    }
    assert "constants.get" in child_repl.list_primitives()
    assert child_repl.list_installed_packs() == ["constants", "selfref"]


@pytest.mark.asyncio
async def test_llm_chat_selfref_fork_spawn_preserves_langfuse_trace_context() -> None:
    """Child agent spans should stay on the same trace and nest under execute_code."""

    fork_code = (
        "handle = runtime.selfref.fork.spawn('child task')\n"
        "results = runtime.selfref.fork.gather_all(handle)\n"
        "print(results[handle['fork_id']]['response'])"
    )

    self_reference = SelfReference()
    repl = PyRepl(self_reference=self_reference)
    tracker = _TrackingLangfuseClient()

    async def fake_chat(messages: list[dict[str, Any]], tools=None, **kwargs: Any):
        _ = kwargs

        if self_reference._get_active_fork_id() is not None:
            return _make_chat_completion("child-ok")

        has_tool_result = any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in messages
        )
        if tools and not has_tool_result:
            return _make_tool_call_completion(
                "execute_code",
                {"code": fork_code},
            )

        return _make_chat_completion("parent-ok")

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    mock_llm.chat.side_effect = fake_chat

    history = [{"role": "user", "content": "seed"}]

    with (
        patch.object(
            shared_langfuse_client,
            "start_as_current_observation",
            side_effect=tracker.start_as_current_observation,
        ),
        patch.object(
            shared_langfuse_client,
            "get_current_trace_id",
            side_effect=tracker.get_current_trace_id,
        ),
        patch.object(
            shared_langfuse_client,
            "get_current_observation_id",
            side_effect=tracker.get_current_observation_id,
        ),
        patch.object(
            shared_langfuse_client,
            "create_trace_id",
            side_effect=tracker.create_trace_id,
        ),
    ):
        trace_token = set_langfuse_trace_context({"trace_id": "trace-root"})
        try:

            @llm_chat(
                llm_interface=mock_llm,
                toolkit=cast(Any, repl.toolset),
                self_reference=self_reference,
                self_reference_key="agent_main",
                max_tool_calls=None,
            )
            async def agent(message: str, history=None):
                """test agent"""

            stream = cast(
                AsyncGenerator[tuple[Any, list[dict[str, Any]]], None],
                agent("root task", history=history),
            )

            async for _content, _updated_history in stream:
                pass
        finally:
            reset_langfuse_trace_context(trace_token)

    parent_chat_span = next(
        record
        for record in tracker.records
        if record.get("name") == "agent_chat_call"
        and isinstance(record.get("input"), dict)
        and record["input"].get("message") == "root task"
    )
    child_chat_span = next(
        record
        for record in tracker.records
        if record.get("name") == "agent_chat_call"
        and isinstance(record.get("input"), dict)
        and record["input"].get("message") == "child task"
    )
    execute_code_span = next(
        record
        for record in tracker.records
        if record.get("as_type") == "tool" and record.get("name") == "execute_code"
    )

    assert parent_chat_span["trace_id"] == "trace-root"
    assert execute_code_span["trace_id"] == "trace-root"
    assert child_chat_span["trace_id"] == "trace-root"
    assert execute_code_span["parent_span_id"] == parent_chat_span["span_id"]
    assert child_chat_span["parent_span_id"] == execute_code_span["span_id"]


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
    repl = PyRepl(self_reference=self_reference)
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
