"""Tests for PyRepl builtin tool."""

from __future__ import annotations

import pytest
import asyncio


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

        stdout_events = [
            e
            for e in events
            if isinstance(e.event, CustomEvent)
            and e.event.event_name == "kernel_stdout"
        ]
        assert len(stdout_events) > 0
        assert "test" in str(stdout_events[0].event.data)


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
        from SimpleLLMFunc.hooks.events import CustomEvent

        repl = PyRepl()
        emitter = ToolEventEmitter()

        run_task = asyncio.create_task(
            repl.execute(
                "name = input('Name: ')\nprint(f'Hello, {name}!')",
                event_emitter=emitter,
            )
        )

        request_id = None
        prompt = ""
        deadline = asyncio.get_running_loop().time() + 2
        while asyncio.get_running_loop().time() < deadline and request_id is None:
            events = await emitter.get_events()
            for event_yield in events:
                event = event_yield.event
                if (
                    isinstance(event, CustomEvent)
                    and event.event_name == "kernel_input_request"
                ):
                    data = getattr(event, "data", None)
                    if isinstance(data, dict):
                        request_id = data.get("request_id")
                        prompt = data.get("prompt", "")
                    break

            if request_id is None:
                await asyncio.sleep(0.01)

        assert isinstance(request_id, str) and request_id
        assert prompt == "Name: "
        assert PyRepl.submit_input(request_id, "Alice") is True

        result = await asyncio.wait_for(run_task, timeout=2)
        assert result["success"] is True
        assert "Hello, Alice!" in result["stdout"]
