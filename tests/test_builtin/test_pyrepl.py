"""Tests for PyRepl builtin tool."""

from __future__ import annotations

import pytest
import asyncio

from SimpleLLMFunc.hooks.events import CustomEvent


async def _wait_for_input_request(
    emitter,
    seen_request_ids: set[str] | None = None,
    timeout: float = 2.0,
) -> tuple[str, str]:
    """Wait until one unseen kernel_input_request event is emitted."""

    seen = seen_request_ids if seen_request_ids is not None else set()
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        events = await emitter.get_events()
        for event_yield in events:
            event = event_yield.event
            if not isinstance(event, CustomEvent):
                continue
            if event.event_name != "kernel_input_request":
                continue

            data = getattr(event, "data", None)
            if not isinstance(data, dict):
                continue

            request_id = data.get("request_id")
            prompt = data.get("prompt", "")
            if not isinstance(request_id, str) or not request_id:
                continue
            if request_id in seen:
                continue
            if not isinstance(prompt, str):
                prompt = ""

            return request_id, prompt

        await asyncio.sleep(0.01)

    raise AssertionError("Timed out waiting for kernel_input_request event")


class TestPyReplCreation:
    """Test PyRepl class creation."""

    def test_repl_creation(self):
        """Test creating a PyRepl instance."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        assert repl is not None
        assert repl.namespace == {}

    def test_repl_has_lock(self):
        """Test that PyRepl has a lock for thread safety."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        assert repl._lock is not None

    def test_repl_timeout_defaults(self):
        """PyRepl should expose documented timeout defaults."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        assert (
            repl.execution_timeout_seconds == PyRepl.DEFAULT_EXECUTION_TIMEOUT_SECONDS
        )
        assert (
            repl.input_idle_timeout_seconds == PyRepl.DEFAULT_INPUT_IDLE_TIMEOUT_SECONDS
        )

    def test_repl_rejects_non_positive_timeouts(self):
        """Timeout values should be validated at construction time."""
        from SimpleLLMFunc.builtin import PyRepl

        with pytest.raises(ValueError, match="execution_timeout_seconds"):
            PyRepl(execution_timeout_seconds=0)

        with pytest.raises(ValueError, match="input_idle_timeout_seconds"):
            PyRepl(input_idle_timeout_seconds=0)


class TestPyReplToolset:
    """Test PyRepl.toolset property."""

    def test_toolset_returns_list(self):
        """Test that toolset returns a list."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        toolset = repl.toolset
        assert isinstance(toolset, list)

    def test_toolset_contains_expected_tools(self):
        """Test that toolset contains expected tool names."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        toolset = repl.toolset

        tool_names = [tool.name for tool in toolset]
        assert "execute_code" in tool_names
        assert "reset_repl" in tool_names
        assert "list_variables" in tool_names

    def test_execute_tool_description_has_repl_guidance(self):
        """execute_code description should guide LLM usage clearly."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        description = execute_tool.description
        assert "persistent REPL session" in description
        assert 'if __name__ == "__main__"' in description
        assert "input()" in description
        assert "does not delete self-reference conversation memory" in description

    def test_all_tool_descriptions_are_english_guidance(self):
        """Builtin tool descriptions should be explicit English guidance."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        descriptions = {tool.name: tool.description for tool in repl.toolset}

        assert "Reset REPL runtime variables" in descriptions["reset_repl"]
        assert "preserves attached self_reference object" in descriptions["reset_repl"]
        assert "List user-defined variables" in descriptions["list_variables"]
        assert (
            "excluding private names and self_reference"
            in descriptions["list_variables"]
        )


class TestPyReplExecute:
    """Test PyRepl execute functionality."""

    @pytest.mark.asyncio
    async def test_execute_simple_print(self):
        """Test basic print execution."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("print('hello')")

        assert result["success"] is True
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_variable_assignment(self):
        """Test variable assignment."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("x = 100")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_variable_persistence(self):
        """Test that variables persist across execute calls."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        await repl.execute("x = 100")
        result = await repl.execute("print(x * 2)")

        assert result["success"] is True
        assert "200" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test error handling."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("1/0")

        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_execute_expression_result(self):
        """Test expression evaluation returns result."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("1 + 1")

        assert result["success"] is True
        assert result["return_value"] == "2"


class TestPyReplReset:
    """Test PyRepl reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_variables(self):
        """Test that reset clears all variables."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        await repl.execute("x = 100")
        await repl.execute("y = 200")

        result = await repl.reset()
        assert "已重置" in result

        # Verify variables are cleared
        vars = await repl.list_variables()
        assert len(vars) == 0


class TestPyReplListVariables:
    """Test PyRepl list_variables functionality."""

    @pytest.mark.asyncio
    async def test_list_variables_empty(self):
        """Test listing variables when empty."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        vars = await repl.list_variables()

        assert isinstance(vars, list)

    @pytest.mark.asyncio
    async def test_list_variables_with_data(self):
        """Test listing variables with data."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        await repl.execute("x = 100")
        await repl.execute("name = 'test'")

        vars = await repl.list_variables()
        assert len(vars) == 2

        names = [v["name"] for v in vars]
        assert "x" in names
        assert "name" in names


class TestPyReplSelfReference:
    """Test SelfReference integration in PyRepl namespace."""

    def test_attach_self_reference_exposes_global(self):
        """Attached self_reference should be available in REPL globals."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        repl = PyRepl()
        repl.attach_self_reference(self_reference)

        assert repl.namespace.get("self_reference") is self_reference

    def test_bind_history_delegates_to_self_reference(self):
        """PyRepl bind_history should update attached SelfReference store."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        repl = PyRepl(self_reference=self_reference)

        repl.bind_history("agent_main", [{"role": "user", "content": "hello"}])

        assert self_reference.list_history_keys() == ["agent_main"]
        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "hello"}
        ]

    @pytest.mark.asyncio
    async def test_execute_can_mutate_memory_via_self_reference_handle(self):
        """execute_code should mutate memory through self_reference proxy methods."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        repl = PyRepl(self_reference=self_reference)

        result = await repl.execute(
            'self_reference.memory["agent_main"].append('
            '{"role": "assistant", "content": "ok"})\n_ = 1'
        )

        assert result["success"] is True
        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "ok"},
        ]

    @pytest.mark.asyncio
    async def test_reset_keeps_attached_self_reference(self):
        """reset_repl should preserve attached self_reference global object."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        repl = PyRepl(self_reference=self_reference)

        await repl.execute("x = 1")
        await repl.reset()

        assert repl.namespace.get("self_reference") is self_reference

    @pytest.mark.asyncio
    async def test_reset_does_not_delete_self_reference_memory(self):
        """reset_repl should not clear SelfReference history store."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history(
            "agent_main",
            [{"role": "user", "content": "remember me"}],
        )

        repl = PyRepl(self_reference=self_reference)
        await repl.execute("x = 1")
        await repl.reset()

        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "remember me"}
        ]

    def test_bind_history_requires_attached_self_reference(self):
        """Convenience bind API should fail when no self_reference is attached."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        with pytest.raises(RuntimeError, match="No self_reference attached"):
            repl.bind_history("agent_main", [{"role": "user", "content": "x"}])


class TestPyReplStreaming:
    """Test PyRepl streaming with event_emitter."""

    @pytest.mark.asyncio
    async def test_execute_with_event_emitter(self):
        """Test that event_emitter receives stdout events."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl()
        emitter = ToolEventEmitter()

        result = await repl.execute("print('hello')", event_emitter=emitter)

        assert result["success"] is True

        await asyncio.sleep(0.1)

        events = await emitter.get_events()
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_streaming_multiple_lines(self):
        """Test streaming with multiple print statements."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl()
        emitter = ToolEventEmitter()

        result = await repl.execute(
            "import time\nfor i in range(3):\n    print(f'line {i}')",
            event_emitter=emitter,
        )

        assert result["success"] is True

        await asyncio.sleep(0.1)

        events = await emitter.get_events()
        assert len(events) >= 3

    @pytest.mark.asyncio
    async def test_event_contains_correct_data(self):
        """Test that emitted events contain correct data."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
        from SimpleLLMFunc.hooks.events import CustomEvent

        repl = PyRepl()
        emitter = ToolEventEmitter()

        await repl.execute("print('test')", event_emitter=emitter)

        await asyncio.sleep(0.1)

        events = await emitter.get_events()

        stdout_events = []
        for event_yield in events:
            event = event_yield.event
            if isinstance(event, CustomEvent) and event.event_name == "kernel_stdout":
                stdout_events.append(event)
        assert len(stdout_events) > 0
        assert "test" in str(stdout_events[0].data)


class TestPyReplEventLoopSafety:
    """Test PyRepl does not block asyncio event loop."""

    @pytest.mark.asyncio
    async def test_execute_does_not_block_event_loop(self):
        """execute_code should not freeze the loop during long-running code."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        tick_count = 0
        running = True

        async def ticker() -> None:
            nonlocal tick_count
            while running:
                tick_count += 1
                await asyncio.sleep(0.01)

        ticker_task = asyncio.create_task(ticker())
        try:
            result = await repl.execute("import time\ntime.sleep(0.12)")
            assert result["success"] is True
        finally:
            running = False
            await ticker_task

        assert tick_count >= 3


class TestPyReplTimeout:
    """Test PyRepl timeout policy for execution and interactive input."""

    @pytest.mark.asyncio
    async def test_execute_timeout_is_configurable(self):
        """Execution should honor configured timeout duration."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(execution_timeout_seconds=0.2)
        result = await repl.execute("import time\ntime.sleep(0.5)")

        assert result["success"] is False
        assert result["error"] is not None
        assert "0.2 seconds" in result["error"]

    @pytest.mark.asyncio
    async def test_waiting_for_input_does_not_consume_timeout(self):
        """input() waiting time should be excluded from timeout budget."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl(execution_timeout_seconds=0.2, input_idle_timeout_seconds=2)
        emitter = ToolEventEmitter()

        run_task = asyncio.create_task(
            repl.execute(
                "name = input('Name: ')\nprint(f'Hello, {name}!')",
                event_emitter=emitter,
            )
        )

        request_id, _prompt = await _wait_for_input_request(emitter)

        await asyncio.sleep(0.35)
        assert not run_task.done()

        assert PyRepl.submit_input(request_id, "Alice") is True
        result = await asyncio.wait_for(run_task, timeout=2)

        assert result["success"] is True
        assert "Hello, Alice!" in result["stdout"]

    @pytest.mark.asyncio
    async def test_timeout_is_reset_after_each_input_submission(self):
        """Each accepted input value should reset execution timeout window."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl(execution_timeout_seconds=0.2, input_idle_timeout_seconds=2)
        emitter = ToolEventEmitter()

        run_task = asyncio.create_task(
            repl.execute(
                """
first = input('First: ')
import time
time.sleep(0.15)
second = input('Second: ')
print(first + second)
""",
                event_emitter=emitter,
            )
        )

        seen_request_ids: set[str] = set()
        request_1, prompt_1 = await _wait_for_input_request(emitter, seen_request_ids)
        assert prompt_1 == "First: "
        seen_request_ids.add(request_1)

        await asyncio.sleep(0.3)
        assert PyRepl.submit_input(request_1, "A") is True

        request_2, prompt_2 = await _wait_for_input_request(emitter, seen_request_ids)
        assert prompt_2 == "Second: "

        await asyncio.sleep(0.3)
        assert not run_task.done()
        assert PyRepl.submit_input(request_2, "B") is True

        result = await asyncio.wait_for(run_task, timeout=2)
        assert result["success"] is True
        assert "AB" in result["stdout"]

    @pytest.mark.asyncio
    async def test_input_idle_timeout_is_enforced(self):
        """Tool-input requests should fail after configured idle timeout."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl(execution_timeout_seconds=2, input_idle_timeout_seconds=0.2)
        emitter = ToolEventEmitter()

        run_task = asyncio.create_task(
            repl.execute("value = input('Value: ')", event_emitter=emitter)
        )

        request_id, prompt = await _wait_for_input_request(emitter)
        assert prompt == "Value: "

        result = await asyncio.wait_for(run_task, timeout=2)

        assert result["success"] is False
        assert result["error"] == "Input request timed out after 0.2 seconds"
        assert "Input request timed out after 0.2 seconds" in result["stderr"]
        assert PyRepl.submit_input(request_id, "late") is False


class TestPyReplInputHook:
    """Test PyRepl interactive input() bridge."""

    def test_submit_input_returns_false_for_unknown_request(self):
        """Submitting to an unknown request id should fail gracefully."""
        from SimpleLLMFunc.builtin import PyRepl

        assert PyRepl.submit_input("unknown-request", "value") is False

    @pytest.mark.asyncio
    async def test_execute_supports_input_roundtrip_via_events(self):
        """execute should emit input request and accept UI-provided response."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl()
        emitter = ToolEventEmitter()

        run_task = asyncio.create_task(
            repl.execute(
                "name = input('Name: ')\nprint(f'Hello, {name}!')",
                event_emitter=emitter,
            )
        )

        request_id, prompt = await _wait_for_input_request(emitter)
        assert prompt == "Name: "
        assert PyRepl.submit_input(request_id, "Alice") is True

        result = await asyncio.wait_for(run_task, timeout=2)
        assert result["success"] is True
        assert "Hello, Alice!" in result["stdout"]
