"""Tests for PyRepl builtin tool."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from SimpleLLMFunc.hooks.events import CustomEvent


async def _wait_for_input_request(
    emitter,
    seen_request_ids: set[str] | None = None,
    timeout: float = 5.0,
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

    def test_close_releases_worker_queue_handles(self):
        """close() should release multiprocessing queue handles promptly."""
        from unittest.mock import MagicMock

        from SimpleLLMFunc.builtin import PyRepl

        class _DeadProcess:
            def is_alive(self) -> bool:
                return False

        repl = PyRepl()
        command_queue = MagicMock()
        event_queue = MagicMock()

        repl._process = _DeadProcess()
        repl._command_queue = command_queue
        repl._event_queue = event_queue

        repl.close()

        command_queue.close.assert_called_once()
        command_queue.join_thread.assert_called_once()
        event_queue.close.assert_called_once()
        event_queue.join_thread.assert_called_once()

    def test_self_reference_exported_from_builtin_primitive(self):
        """SelfReference should be importable from builtin.primitive."""
        from SimpleLLMFunc.builtin.primitive import (
            SelfReference as BuiltinSelfReference,
        )
        from SimpleLLMFunc.self_reference import SelfReference

        assert BuiltinSelfReference is SelfReference

    def test_self_reference_exported_from_builtin_package(self):
        """SelfReference should be re-exported from builtin package."""
        from SimpleLLMFunc.builtin import SelfReference as BuiltinSelfReference
        from SimpleLLMFunc.self_reference import SelfReference

        assert BuiltinSelfReference is SelfReference

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
        assert "timeout_seconds" in description
        assert "runtime.list_primitives()" in description
        assert "runtime.list_primitive_specs()" in description
        assert "does not delete runtime memory managed by registered primitives" in (
            description
        )

    def test_execute_tool_schema_exposes_timeout_seconds(self):
        """execute_code tool schema should expose per-call timeout controls."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        function_schema = execute_tool.to_openai_tool()["function"]
        params_schema = function_schema["parameters"]
        properties = params_schema["properties"]

        assert "timeout_seconds" in properties
        timeout_type = properties["timeout_seconds"].get("type")
        if isinstance(timeout_type, list):
            assert "number" in timeout_type
        else:
            assert timeout_type == "number"

        required = params_schema.get("required", [])
        assert "timeout_seconds" not in required

    def test_all_tool_descriptions_are_english_guidance(self):
        """Builtin tool descriptions should be explicit English guidance."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        descriptions = {tool.name: tool.description for tool in repl.toolset}

        assert "Reset REPL runtime variables" in descriptions["reset_repl"]
        assert (
            "preserves registered runtime primitive backends"
            in descriptions["reset_repl"]
        )
        assert "List user-defined variables" in descriptions["list_variables"]
        assert "excluding private names and runtime" in descriptions["list_variables"]


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

    @pytest.mark.asyncio
    async def test_execute_runtime_error_includes_structured_details(self):
        """Runtime errors should provide line-aware structured diagnostics."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("x = 1\ny = 0\nx / y")

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert "ZeroDivisionError" in result["error"]

        details = result["error_details"]
        assert isinstance(details, dict)
        assert details["error_type"] == "ZeroDivisionError"
        assert details["line"] == 3
        assert details["snippet"] == "x / y"

    @pytest.mark.asyncio
    async def test_execute_syntax_error_includes_snippet_and_pointer(self):
        """Syntax errors should expose exact snippet location information."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("for i in range(2)\n    print(i)")

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert "SyntaxError" in result["error"]

        details = result["error_details"]
        assert isinstance(details, dict)
        assert details["error_type"] == "SyntaxError"
        assert details["line"] == 1
        assert details["column"] == 18
        assert details["snippet"] == "for i in range(2)"
        assert details["pointer"] == " " * 17 + "^"

    @pytest.mark.asyncio
    async def test_execute_handles_stderr_with_invalid_fileno(self, monkeypatch):
        """Worker startup should survive environments where stderr has fileno=-1."""
        from SimpleLLMFunc.builtin import PyRepl

        class _InvalidStderr:
            def fileno(self) -> int:
                return -1

            def write(self, _text: str) -> int:
                return 0

            def flush(self) -> None:
                return None

        monkeypatch.setattr(sys, "stderr", _InvalidStderr())

        repl = PyRepl()
        try:
            result = await repl.execute("print('ok')")
        finally:
            repl.close()

        assert result["success"] is True
        assert "ok" in result["stdout"]


class TestPyReplAudit:
    """Test per-instance audit log persistence behavior."""

    @pytest.mark.asyncio
    async def test_each_instance_writes_isolated_audit_log(self, monkeypatch, tmp_path):
        """Each PyRepl instance should persist execution history separately."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.logger.logger_config import logger_config

        monkeypatch.setattr(logger_config, "LOG_DIR", str(tmp_path))

        repl_a = PyRepl()
        repl_b = PyRepl()

        try:
            result_a = await repl_a.execute("print('A')")
            result_b = await repl_b.execute("print('B')")
        finally:
            repl_a.close()
            repl_b.close()

        assert result_a["success"] is True
        assert result_b["success"] is True

        audit_dir_a = Path(repl_a.audit_log_dir)
        audit_dir_b = Path(repl_b.audit_log_dir)
        assert audit_dir_a != audit_dir_b
        assert audit_dir_a.parent == tmp_path / "pyrepl"
        assert audit_dir_b.parent == tmp_path / "pyrepl"

        audit_file_a = Path(repl_a.audit_log_file)
        audit_file_b = Path(repl_b.audit_log_file)
        assert audit_file_a.exists()
        assert audit_file_b.exists()

        records_a = [
            json.loads(line)
            for line in audit_file_a.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records_b = [
            json.loads(line)
            for line in audit_file_b.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert records_a[-1]["code"] == "print('A')"
        assert records_b[-1]["code"] == "print('B')"
        assert records_a[-1]["result"]["success"] is True
        assert records_b[-1]["result"]["success"] is True


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


class TestPyReplPrimitivePacks:
    """Test primitive-pack installation and runtime backend behavior."""

    def test_install_selfref_pack_registers_backend_and_primitives(self):
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        repl = PyRepl()

        assert "memory.keys" not in repl.list_primitives()
        assert repl.list_runtime_backends() == []

        repl.install_primitive_pack("selfref", backend=self_reference)

        assert repl.get_runtime_backend("selfref") is self_reference
        assert repl.list_runtime_backends() == ["selfref"]
        assert "selfref.history.keys" in repl.list_primitives()
        assert "selfref.fork.run" in repl.list_primitives()
        assert "memory.keys" not in repl.list_primitives()
        assert "fork.run" not in repl.list_primitives()

    def test_install_unknown_primitive_pack_raises(self):
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        with pytest.raises(KeyError, match="primitive pack"):
            repl.install_primitive_pack("unknown_pack")

    def test_install_legacy_self_reference_pack_name_raises(self):
        """Hard-cut migration: old pack name must be rejected."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        repl = PyRepl()
        with pytest.raises(KeyError, match="primitive pack"):
            repl.install_primitive_pack("self_reference", backend=SelfReference())

    def test_install_selfref_pack_registers_under_selfref_backend(self):
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        repl = PyRepl()

        repl.install_primitive_pack("selfref", backend=self_reference)

        assert repl.get_runtime_backend("selfref") is self_reference
        assert "selfref.history.keys" in repl.list_primitives()
        assert "selfref.fork.run" in repl.list_primitives()

    @pytest.mark.asyncio
    async def test_execute_can_mutate_memory_via_runtime_primitives(self):
        """execute_code should mutate memory through runtime primitives."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)

        result = await repl.execute(
            "runtime.selfref.history.append({'role': 'assistant', 'content': 'ok'})\n_ = 1"
        )

        assert result["success"] is True
        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "ok"},
        ]

    @pytest.mark.asyncio
    async def test_reset_keeps_registered_self_reference_backend(self):
        """reset_repl should preserve installed runtime backend registration."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)

        await repl.execute("x = 1")
        await repl.reset()

        assert repl.get_runtime_backend("selfref") is self_reference

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

        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)
        await repl.execute("x = 1")
        await repl.reset()

        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "remember me"}
        ]

    @pytest.mark.asyncio
    async def test_execute_can_fork_bound_agent_instance_with_memory_snapshot(self):
        """REPL runtime.selfref.fork.run should inherit memory as child context."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        observed_calls: list[dict[str, object]] = []

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
        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)

        result = await repl.execute(
            "fork_result = runtime.selfref.fork.run('sub-task')\n"
            "print(fork_result['source_memory_key'])\n"
            "print(fork_result['memory_key'])\n"
            "print(fork_result['response'])\n"
        )

        assert result["success"] is True
        assert "agent_main" in result["stdout"]
        assert "forked:sub-task" in result["stdout"]
        assert observed_calls[0]["history"] == [{"role": "user", "content": "seed"}]

        fork_keys = [
            key
            for key in self_reference.list_history_keys()
            if key.startswith("agent_main::fork::")
        ]
        assert len(fork_keys) == 1
        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "seed"}
        ]
        assert self_reference.snapshot_history(fork_keys[0]) == [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "child done"},
        ]

    @pytest.mark.asyncio
    async def test_execute_can_spawn_and_wait_fork_from_code_act(self):
        """Code-act fork should be runtime-hooked and support spawn/wait APIs."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                {"task": message, "runtime_pid": os.getpid()},
                [
                    *(history or []),
                    {"role": "assistant", "content": f"done:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)

        result = await repl.execute(
            "spawned = runtime.selfref.fork.spawn('task-a')\n"
            "print(spawned['status'])\n"
            "final = runtime.selfref.fork.wait(spawned['fork_id'])\n"
            "print(final['status'])\n"
            "print(final['response']['runtime_pid'])\n"
        )

        assert result["success"] is True
        assert "running" in result["stdout"]
        assert "completed" in result["stdout"]
        assert str(os.getpid()) in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_can_wait_all_spawned_forks(self):
        """Code-act runtime.selfref.fork.wait_all should collect spawned forks."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                message,
                [
                    *(history or []),
                    {"role": "assistant", "content": f"done:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)

        result = await repl.execute(
            "handles = [\n"
            "    runtime.selfref.fork.spawn('task-a'),\n"
            "    runtime.selfref.fork.spawn('task-b'),\n"
            "]\n"
            "ids = [item['fork_id'] for item in handles]\n"
            "all_results = runtime.selfref.fork.wait_all(ids)\n"
            "print(len(all_results))\n"
            "print(sorted(all_results.keys()) == sorted(ids))\n"
            "print(all(v['status'] == 'completed' for v in all_results.values()))\n"
        )

        assert result["success"] is True
        assert "2" in result["stdout"]
        assert "True" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_emits_fork_lifecycle_events(self):
        """Code-act fork should emit structured lifecycle custom events."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)
        emitter = ToolEventEmitter()

        result = await repl.execute(
            "handle = runtime.selfref.fork.spawn('task-a')\n"
            "_ = runtime.selfref.fork.wait(handle['fork_id'])\n",
            event_emitter=emitter,
        )

        assert result["success"] is True

        events = await emitter.get_events()
        event_names = [
            event.event.event_name
            for event in events
            if isinstance(event.event, CustomEvent)
        ]

        assert "selfref_fork_spawned" in event_names
        assert "selfref_fork_start" in event_names
        assert "selfref_fork_end" in event_names

    @pytest.mark.asyncio
    async def test_execute_emits_fork_stream_events(self):
        """Code-act fork should emit streaming child output custom events."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            yield (
                "child-a ",
                list(history or []),
            )
            await asyncio.sleep(0)
            yield (
                "child-b\n",
                list(history or []),
            )
            yield (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)
        emitter = ToolEventEmitter()

        result = await repl.execute(
            "fork_result = runtime.selfref.fork.run('task-a')\n"
            "print(fork_result['status'])\n",
            event_emitter=emitter,
        )

        assert result["success"] is True
        assert "completed" in result["stdout"]

        events = await emitter.get_events()
        custom_events = [
            event.event for event in events if isinstance(event.event, CustomEvent)
        ]
        event_names = [event.event_name for event in custom_events]

        assert "selfref_fork_stream_open" in event_names
        assert "selfref_fork_stream_delta" in event_names
        assert "selfref_fork_stream_close" in event_names

        delta_texts = [
            data.get("text")
            for data in [
                event.data
                for event in custom_events
                if event.event_name == "selfref_fork_stream_delta"
            ]
            if isinstance(data, dict)
        ]
        assert "child-a " in delta_texts
        assert "child-b\n" in delta_texts
        assert "done:task-a" in delta_texts


class TestPyReplRuntimePrimitives:
    """Test direct runtime primitive access inside execute_code."""

    @pytest.mark.asyncio
    async def test_execute_exposes_runtime_list_primitives(self):
        """runtime.list_primitives should list baseline runtime primitives."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            "names = runtime.list_primitives()\n"
            "print('runtime.list_primitives' in names)\n"
            "print('runtime.list_backends' in names)\n"
            "print('selfref.history.keys' in names)\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["True", "True", "False"]

    @pytest.mark.asyncio
    async def test_execute_exposes_runtime_list_primitive_specs(self):
        """runtime.list_primitive_specs should include structured contract metadata."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            "specs = runtime.list_primitive_specs()\n"
            "core = next(item for item in specs if item.get('name') == 'runtime.list_primitive_specs')\n"
            "print(isinstance(specs, list))\n"
            "print(any(item.get('name') == 'runtime.list_primitives' for item in specs))\n"
            "print(any(item.get('name') == 'runtime.list_primitive_specs' for item in specs))\n"
            "print(any(item.get('name') == 'runtime.list_backends' for item in specs))\n"
            "print(isinstance(core.get('input_type'), str))\n"
            "print(isinstance(core.get('output_type'), str))\n"
            "print(isinstance(core.get('parameters'), list))\n"
            "print(isinstance(core.get('best_practices'), list))\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == [
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
        ]

    @pytest.mark.asyncio
    async def test_execute_exposes_selfref_guide_and_best_practices(self):
        """Selfref pack should expose namespace guide with fork/memory best practices."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=SelfReference())
        result = await repl.execute(
            "guide = runtime.selfref.guide()\n"
            "specs = runtime.list_primitive_specs()\n"
            "guide_spec = next(item for item in specs if item.get('name') == 'selfref.guide')\n"
            "print('best_practices' in guide)\n"
            "print(len(guide.get('best_practices', [])) >= 5)\n"
            "print(isinstance(guide_spec.get('parameters'), list))\n"
            "print(isinstance(guide_spec.get('best_practices'), list))\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["True", "True", "True", "True"]

    @pytest.mark.asyncio
    async def test_execute_can_mutate_memory_via_runtime_primitive_calls(self):
        """runtime.selfref.history.* should proxy host self-reference operations."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)
        result = await repl.execute(
            "runtime.selfref.history.append({'role': 'assistant', 'content': 'ok'})\n"
            "print(runtime.selfref.history.count())\n"
            "print(runtime.selfref.history.get(1)['content'])\n"
            "print(runtime.selfref.history.active_key())\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["2", "ok", "agent_main"]
        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "ok"},
        ]

    @pytest.mark.asyncio
    async def test_execute_can_run_fork_via_runtime_primitive_calls(self):
        """runtime.selfref.fork.run should fork bound agent instance."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.self_reference import SelfReference

        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                f"runtime:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        repl = PyRepl()
        repl.install_primitive_pack("selfref", backend=self_reference)

        result = await repl.execute(
            "fork_result = runtime.selfref.fork.run('sub-task')\n"
            "print(fork_result['source_memory_key'])\n"
            "print(fork_result['response'])\n"
        )

        assert result["success"] is True
        assert "agent_main" in result["stdout"]
        assert "runtime:sub-task" in result["stdout"]


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
    async def test_execute_timeout_can_be_overridden_per_call(self):
        """Per-call timeout argument should override the REPL default timeout."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(execution_timeout_seconds=2.0)
        result = await repl.execute(
            "import time\ntime.sleep(0.5)",
            timeout_seconds=0.1,
        )

        assert result["success"] is False
        assert result["error"] is not None
        assert "0.1 seconds" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_rejects_non_positive_per_call_timeout(self):
        """Per-call timeout should reject non-positive values."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        with pytest.raises(ValueError, match="timeout_seconds"):
            await repl.execute("print('hello')", timeout_seconds=0)

    @pytest.mark.asyncio
    async def test_waiting_for_input_does_not_consume_timeout(self):
        """input() waiting time should be excluded from timeout budget."""
        import time
        from unittest.mock import patch

        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(execution_timeout_seconds=1.0, input_idle_timeout_seconds=10)

        def delayed_input(prompt: str = "") -> str:
            assert prompt == "Name: "
            time.sleep(1.3)
            return "Alice"

        with patch("builtins.input", side_effect=delayed_input):
            result = await repl.execute(
                "name = input('Name: ')\nprint(f'Hello, {name}!')"
            )

        assert result["success"] is True
        assert "Hello, Alice!" in result["stdout"]

    @pytest.mark.asyncio
    async def test_timeout_is_reset_after_each_input_submission(self):
        """Accepted input should reset execution timeout window."""
        import time
        from unittest.mock import patch

        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(execution_timeout_seconds=1.0, input_idle_timeout_seconds=10)

        prompts_seen: list[str] = []
        values = iter(["A", "B"])

        def delayed_input(prompt: str = "") -> str:
            prompts_seen.append(prompt)
            time.sleep(1.3)
            return next(values)

        with patch("builtins.input", side_effect=delayed_input):
            result = await repl.execute(
                """
first = input('First: ')
second = input('Second: ')
import time
time.sleep(0.4)
print(first + second)
"""
            )

        assert result["success"] is True
        assert prompts_seen == ["First: ", "Second: "]
        assert "AB" in result["stdout"]

    @pytest.mark.asyncio
    async def test_input_idle_timeout_is_enforced(self):
        """Tool-input requests should fail after configured idle timeout."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.events import CustomEvent
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl(execution_timeout_seconds=10, input_idle_timeout_seconds=0.2)
        emitter = ToolEventEmitter()

        run_task = asyncio.create_task(
            repl.execute("value = input('Value: ')", event_emitter=emitter)
        )

        result = await asyncio.wait_for(run_task, timeout=10)

        assert result["success"] is False
        assert result["error"] == "Input request timed out after 0.2 seconds"
        assert "Input request timed out after 0.2 seconds" in result["stderr"]

        request_id: str | None = None
        request_prompt: str | None = None
        for event_yield in await emitter.get_events():
            event = event_yield.event
            if not isinstance(event, CustomEvent):
                continue
            if event.event_name != "kernel_input_request":
                continue

            data = getattr(event, "data", None)
            if not isinstance(data, dict):
                continue

            maybe_id = data.get("request_id")
            maybe_prompt = data.get("prompt")
            if isinstance(maybe_id, str) and maybe_id:
                request_id = maybe_id
            if isinstance(maybe_prompt, str):
                request_prompt = maybe_prompt
            break

        if request_prompt is not None:
            assert request_prompt == "Value: "

        assert PyRepl.submit_input(request_id or "late-request-id", "late") is False


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
