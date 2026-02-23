"""PyRepl builtin tool for SimpleLLMFunc.

轻量级 Python REPL，基于 subprocess + IPython InteractiveShell。
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
from SimpleLLMFunc.logger.logger_config import logger_config
from SimpleLLMFunc.self_reference import SelfReference
from SimpleLLMFunc.tool import Tool

from .pyrepl_worker import (
    COMMAND_EXECUTE,
    COMMAND_INPUT_REPLY,
    COMMAND_LIST_VARIABLES,
    COMMAND_RESET,
    COMMAND_SELF_REFERENCE_RESPONSE,
    COMMAND_SHUTDOWN,
    EVENT_EXECUTE_RESULT,
    EVENT_INPUT_ACCEPTED,
    EVENT_INPUT_REQUEST,
    EVENT_LIST_VARIABLES_RESULT,
    EVENT_RESET_RESULT,
    EVENT_SELF_REFERENCE_CALL,
    EVENT_STDERR,
    EVENT_STDOUT,
    EVENT_WORKER_READY,
    EVENT_WORKER_ERROR,
    SELF_REFERENCE_GLOBAL_NAME,
    run_pyrepl_worker,
)


class PyRepl:
    """轻量级 Python REPL

    基于 subprocess + IPython InteractiveShell，支持：
    - 实时 stdout/stderr streaming
    - 变量跨调用持久化
    - 独立进程执行，支持更可靠中断

    Usage:
        repl = PyRepl()
        tools = repl.toolset

        @llm_chat(toolkit=tools + [...], ...)
        async def chat(message: str, history=None):
            '''Python 编程助手'''
    """

    _input_registry_lock = threading.Lock()
    _pending_input_queues: Dict[str, queue.Queue[str]] = {}

    DEFAULT_EXECUTION_TIMEOUT_SECONDS = 120.0
    DEFAULT_INPUT_IDLE_TIMEOUT_SECONDS = 300.0
    INTERRUPT_GRACE_SECONDS = 1.0

    EXECUTE_TOOL_DESCRIPTION = (
        "Run Python code in a persistent REPL session (state persists across "
        "calls). Write direct executable snippets, not standalone scripts. "
        'Do not include `if __name__ == "__main__":` blocks. Interactive '
        "`input()` is supported. `reset_repl` only clears REPL variables and "
        "does not delete self-reference conversation memory."
    )
    RESET_TOOL_DESCRIPTION = (
        "Reset REPL runtime variables in the current session. This clears "
        "Python variables only and preserves attached self_reference object."
    )
    LIST_VARIABLES_TOOL_DESCRIPTION = (
        "List user-defined variables currently available in REPL namespace "
        "(excluding private names and self_reference)."
    )

    def __init__(
        self,
        execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        input_idle_timeout_seconds: float = DEFAULT_INPUT_IDLE_TIMEOUT_SECONDS,
        self_reference: Optional[SelfReference] = None,
    ):
        execution_timeout = float(execution_timeout_seconds)
        if execution_timeout <= 0:
            raise ValueError("execution_timeout_seconds must be greater than 0")

        input_idle_timeout = float(input_idle_timeout_seconds)
        if input_idle_timeout <= 0:
            raise ValueError("input_idle_timeout_seconds must be greater than 0")

        self.execution_timeout_seconds = execution_timeout
        self.input_idle_timeout_seconds = input_idle_timeout

        self.namespace: Dict[str, Any] = {}
        self._self_reference: Optional[SelfReference] = None
        self._tools: Optional[List[Tool]] = None
        self._lock = threading.RLock()
        self._operation_lock = asyncio.Lock()

        self._ctx = mp.get_context("spawn")
        self._command_queue: Any = None
        self._event_queue: Any = None
        self._process: Any = None
        self._prefetched_events: List[dict[str, Any]] = []
        self._closed = False

        self._instance_id = uuid.uuid4().hex
        self._audit_lock = threading.Lock()
        self._audit_dir = Path(logger_config.LOG_DIR) / "pyrepl" / self._instance_id
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._audit_file = self._audit_dir / "executions.jsonl"

        if self_reference is not None:
            self.attach_self_reference(self_reference)

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def audit_log_dir(self) -> str:
        return str(self._audit_dir)

    @property
    def audit_log_file(self) -> str:
        return str(self._audit_file)

    def attach_self_reference(self, self_reference: SelfReference) -> None:
        """Attach shared self-reference object into REPL global namespace."""

        if not isinstance(self_reference, SelfReference):
            raise ValueError("self_reference must be a SelfReference instance")

        with self._lock:
            self._self_reference = self_reference
            self.namespace[SELF_REFERENCE_GLOBAL_NAME] = self_reference

    def detach_self_reference(self) -> None:
        """Detach previously attached self-reference object from namespace."""

        with self._lock:
            self._self_reference = None
            self.namespace.pop(SELF_REFERENCE_GLOBAL_NAME, None)

    @classmethod
    def _register_input_queue(cls, request_id: str) -> queue.Queue[str]:
        request_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        with cls._input_registry_lock:
            cls._pending_input_queues[request_id] = request_queue
        return request_queue

    @classmethod
    def _pop_input_queue(cls, request_id: str) -> Optional[queue.Queue[str]]:
        with cls._input_registry_lock:
            return cls._pending_input_queues.pop(request_id, None)

    @classmethod
    def submit_input(cls, request_id: str, value: str) -> bool:
        """Submit a response for a pending ``input()`` request.

        Args:
            request_id: Request ID emitted by ``kernel_input_request`` event.
            value: User-provided input text.

        Returns:
            True if delivered to a live request; False otherwise.
        """

        with cls._input_registry_lock:
            request_queue = cls._pending_input_queues.get(request_id)

        if request_queue is None:
            return False

        try:
            request_queue.put_nowait(value)
            return True
        except queue.Full:
            return False

    @property
    def toolset(self) -> List[Tool]:
        """返回绑定到该 repl 实例的 tool 列表"""
        if self._tools is None:
            self._tools = self._create_tools()
        return self._tools

    def _create_tools(self) -> List[Tool]:
        tools = []

        execute_tool = Tool(
            name="execute_code",
            description=self.EXECUTE_TOOL_DESCRIPTION,
            func=self.execute,
        )
        tools.append(execute_tool)

        reset_tool = Tool(
            name="reset_repl",
            description=self.RESET_TOOL_DESCRIPTION,
            func=self.reset,
        )
        tools.append(reset_tool)

        list_vars_tool = Tool(
            name="list_variables",
            description=self.LIST_VARIABLES_TOOL_DESCRIPTION,
            func=self.list_variables,
        )
        tools.append(list_vars_tool)

        return tools

    @staticmethod
    def _format_timeout_seconds(seconds: float) -> str:
        if float(seconds).is_integer():
            return str(int(seconds))
        return f"{seconds:g}"

    @staticmethod
    def _stream_fileno(stream: Any) -> Optional[int]:
        if stream is None:
            return None

        fileno = getattr(stream, "fileno", None)
        if fileno is None:
            return None

        try:
            value = fileno()
        except Exception:
            return None

        try:
            return int(value)
        except Exception:
            return None

    @contextlib.contextmanager
    def _temporary_valid_stderr(self):
        """Temporarily ensure ``sys.stderr`` has a valid POSIX file descriptor.

        In Textual runtime, ``sys.stderr`` can be a capture proxy with
        ``fileno() == -1``. multiprocessing.resource_tracker forwards
        ``sys.stderr.fileno()`` into ``fds_to_keep`` and crashes when it sees
        invalid values.
        """

        current_stderr = sys.stderr
        current_fd = self._stream_fileno(current_stderr)
        if current_fd is not None and current_fd >= 0:
            yield
            return

        temp_stream = None
        replacement = sys.__stderr__
        replacement_fd = self._stream_fileno(replacement)

        if replacement_fd is None or replacement_fd < 0:
            temp_stream = open(os.devnull, "w", encoding="utf-8")
            replacement = temp_stream

        sys.stderr = replacement
        try:
            yield
        finally:
            sys.stderr = current_stderr
            if temp_stream is not None:
                temp_stream.close()

    def _ensure_worker_locked(self) -> None:
        if self._closed:
            raise RuntimeError("PyRepl is closed")

        if self._process is not None and self._process.is_alive():
            return

        with self._temporary_valid_stderr():
            self._command_queue = self._ctx.Queue()
            self._event_queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=run_pyrepl_worker,
                args=(self._command_queue, self._event_queue),
                daemon=True,
            )
            process.start()
        self._process = process

        assert self._event_queue is not None
        startup_deadline = time.monotonic() + 10.0
        while time.monotonic() < startup_deadline:
            if not process.is_alive():
                raise RuntimeError("PyRepl worker exited before startup")

            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
            if event_type == EVENT_WORKER_READY:
                return

            if isinstance(event, dict):
                self._prefetched_events.append(event)

        raise RuntimeError("Timed out waiting for PyRepl worker startup")

    def _drain_event_queue_locked(self) -> None:
        self._prefetched_events.clear()

        if self._event_queue is None:
            return

        while True:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                return

    def _send_worker_command_locked(self, command: dict[str, Any]) -> None:
        self._ensure_worker_locked()
        assert self._command_queue is not None
        self._command_queue.put(command)

    async def _receive_worker_event(
        self,
        timeout_seconds: float,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            if self._prefetched_events:
                return self._prefetched_events.pop(0)
            event_queue = self._event_queue

        if event_queue is None:
            return None

        try:
            return await asyncio.to_thread(
                event_queue.get,
                True,
                timeout_seconds,
            )
        except queue.Empty:
            return None

    def _interrupt_worker_locked(self) -> None:
        process = self._process
        if process is None or not process.is_alive():
            return

        pid = process.pid
        if not pid:
            return

        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def _shutdown_worker_locked(self) -> None:
        process = self._process
        if process is None:
            return

        if process.is_alive():
            try:
                self._send_worker_command_locked({"type": COMMAND_SHUTDOWN})
            except Exception:
                pass

            process.join(timeout=1.0)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)

        self._process = None
        self._command_queue = None
        self._event_queue = None
        self._prefetched_events.clear()

    def _restart_worker_locked(self) -> None:
        self._shutdown_worker_locked()
        self._ensure_worker_locked()

    def _execute_self_reference_call(self, message: dict[str, Any]) -> dict[str, Any]:
        call_id = str(message.get("call_id", ""))
        operation = str(message.get("op", ""))
        key = message.get("key")
        args = message.get("args", [])

        if not isinstance(args, list):
            args = []

        try:
            if self._self_reference is None:
                raise RuntimeError("No self_reference attached")

            if operation == "keys":
                result = self._self_reference.memory.keys()
            else:
                if not isinstance(key, str):
                    raise ValueError("memory key must be a non-empty string")

                memory = self._self_reference.memory[key]

                if operation == "count":
                    result = memory.count()
                elif operation == "all":
                    result = memory.all()
                elif operation == "get":
                    result = memory.get(*args)
                elif operation == "append":
                    result = memory.append(*args)
                elif operation == "insert":
                    result = memory.insert(*args)
                elif operation == "update":
                    result = memory.update(*args)
                elif operation == "delete":
                    result = memory.delete(*args)
                elif operation == "replace":
                    result = memory.replace(*args)
                elif operation == "clear":
                    result = memory.clear()
                elif operation == "get_system_prompt":
                    result = memory.get_system_prompt()
                elif operation == "set_system_prompt":
                    result = memory.set_system_prompt(*args)
                elif operation == "append_system_prompt":
                    result = memory.append_system_prompt(*args)
                else:
                    raise ValueError(f"Unsupported self_reference op: {operation}")

            return {
                "type": COMMAND_SELF_REFERENCE_RESPONSE,
                "call_id": call_id,
                "ok": True,
                "result": result,
            }
        except Exception as exc:
            return {
                "type": COMMAND_SELF_REFERENCE_RESPONSE,
                "call_id": call_id,
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

    async def _emit_custom_event(
        self,
        event_emitter: Optional[ToolEventEmitter],
        event_name: str,
        data: dict[str, Any],
    ) -> None:
        if event_emitter is None:
            return
        await event_emitter.emit(event_name, data)

    @staticmethod
    def _build_timeout_error_details(
        message: str,
    ) -> dict[str, Any]:
        return {
            "error_type": "TimeoutError",
            "message": message,
            "filename": None,
            "line": None,
            "column": None,
            "snippet": None,
            "pointer": None,
            "summary": message,
            "user_traceback": "",
            "full_traceback": "",
        }

    def _append_audit_entry(self, payload: dict[str, Any]) -> None:
        with self._audit_lock:
            with self._audit_file.open("a", encoding="utf-8") as audit_stream:
                json.dump(payload, audit_stream, ensure_ascii=False, default=str)
                audit_stream.write("\n")

    async def execute(
        self,
        code: str,
        event_emitter: Optional[ToolEventEmitter] = None,
    ) -> Dict[str, Any]:
        """Execute Python snippets in a persistent REPL with streaming output.

        Guidance for LLM tool usage:
        - Write executable snippets directly.
        - Do not wrap code with ``if __name__ == "__main__":``.
        - Variables persist across multiple ``execute_code`` calls.
        - ``input()`` is supported. In event mode, callers can reply via
          ``PyRepl.submit_input(request_id, value)``.
        - ``reset_repl`` clears REPL variables only. Forgetting self-reference
          memory must be done via memory methods (for example ``delete`` /
          ``replace`` / ``clear`` on ``self_reference.memory["key"]``).

        Args:
            code: Python code to execute.
            event_emitter: Optional emitter for real-time stdout/stderr events.

        Returns:
            A dict containing success, stdout, stderr, return_value,
            error, error_details, and execution_time_ms.
        """

        async with self._operation_lock:
            start_time = time.time()
            timeout_seconds = self.execution_timeout_seconds

            stdout_parts: List[str] = []
            stderr_parts: List[str] = []
            error_message: Optional[str] = None
            error_details: Optional[dict[str, Any]] = None
            return_value: Optional[str] = None

            pending_input_requests: dict[str, queue.Queue[str]] = {}
            pending_input_waiters = 0

            poll_interval_seconds = 0.05
            execution_deadline = time.monotonic() + timeout_seconds

            timed_out = False
            interrupt_sent = False
            interrupt_deadline = 0.0
            received_execute_result = False

            execution_id = uuid.uuid4().hex

            try:
                with self._lock:
                    self._ensure_worker_locked()
                    self._drain_event_queue_locked()
                    self._send_worker_command_locked(
                        {
                            "type": COMMAND_EXECUTE,
                            "exec_id": execution_id,
                            "code": code,
                            "input_idle_timeout_seconds": self.input_idle_timeout_seconds,
                            "self_reference_enabled": self._self_reference is not None,
                        }
                    )

                while True:
                    for request_id, request_queue in list(
                        pending_input_requests.items()
                    ):
                        try:
                            submitted_value = request_queue.get_nowait()
                        except queue.Empty:
                            continue

                        with self._lock:
                            self._send_worker_command_locked(
                                {
                                    "type": COMMAND_INPUT_REPLY,
                                    "request_id": request_id,
                                    "value": submitted_value,
                                }
                            )
                        pending_input_requests.pop(request_id, None)
                        self._pop_input_queue(request_id)

                    event = await self._receive_worker_event(
                        timeout_seconds=poll_interval_seconds
                    )

                    if event is not None:
                        event_type = str(event.get("type", ""))

                        if event_type == EVENT_SELF_REFERENCE_CALL:
                            response = self._execute_self_reference_call(event)
                            with self._lock:
                                self._send_worker_command_locked(response)
                            continue

                        if event_type == EVENT_WORKER_ERROR:
                            message = str(event.get("message", "Worker error"))
                            stderr_parts.append(message + "\n")
                            await self._emit_custom_event(
                                event_emitter,
                                "kernel_stderr",
                                {"text": message + "\n"},
                            )
                            continue

                        event_exec_id = event.get("exec_id")
                        if event_exec_id != execution_id:
                            continue

                        if event_type == EVENT_STDOUT:
                            text = str(event.get("text", ""))
                            if text:
                                stdout_parts.append(text)
                                await self._emit_custom_event(
                                    event_emitter,
                                    "kernel_stdout",
                                    {"text": text},
                                )
                            continue

                        if event_type == EVENT_STDERR:
                            text = str(event.get("text", ""))
                            if text:
                                stderr_parts.append(text)
                                await self._emit_custom_event(
                                    event_emitter,
                                    "kernel_stderr",
                                    {"text": text},
                                )
                            continue

                        if event_type == EVENT_INPUT_REQUEST:
                            request_id = str(event.get("request_id", ""))
                            prompt = str(event.get("prompt", ""))

                            if not request_id:
                                continue

                            pending_input_waiters += 1
                            request_queue = self._register_input_queue(request_id)
                            pending_input_requests[request_id] = request_queue

                            await self._emit_custom_event(
                                event_emitter,
                                "kernel_input_request",
                                {
                                    "request_id": request_id,
                                    "prompt": prompt,
                                    "idle_timeout_seconds": self.input_idle_timeout_seconds,
                                },
                            )

                            if event_emitter is None:
                                input_value = await asyncio.to_thread(
                                    builtins.input, prompt
                                )
                                with self._lock:
                                    self._send_worker_command_locked(
                                        {
                                            "type": COMMAND_INPUT_REPLY,
                                            "request_id": request_id,
                                            "value": input_value,
                                        }
                                    )
                                pending_input_requests.pop(request_id, None)
                                self._pop_input_queue(request_id)

                            continue

                        if event_type == EVENT_INPUT_ACCEPTED:
                            request_id = str(event.get("request_id", ""))
                            if pending_input_waiters > 0:
                                pending_input_waiters -= 1
                            pending_input_requests.pop(request_id, None)
                            self._pop_input_queue(request_id)
                            execution_deadline = time.monotonic() + timeout_seconds
                            continue

                        if event_type == EVENT_EXECUTE_RESULT:
                            received_execute_result = True
                            raw_error = event.get("error")
                            error_message = (
                                str(raw_error)
                                if isinstance(raw_error, str)
                                else (str(raw_error) if raw_error is not None else None)
                            )
                            raw_error_details = event.get("error_details")
                            if isinstance(raw_error_details, dict):
                                error_details = raw_error_details
                            raw_return_value = event.get("return_value")
                            return_value = (
                                raw_return_value
                                if isinstance(raw_return_value, str)
                                else (
                                    str(raw_return_value)
                                    if raw_return_value is not None
                                    else None
                                )
                            )
                            break

                    now = time.monotonic()
                    if (
                        not timed_out
                        and pending_input_waiters == 0
                        and now >= execution_deadline
                    ):
                        timed_out = True
                        interrupt_sent = True
                        interrupt_deadline = now + self.INTERRUPT_GRACE_SECONDS
                        with self._lock:
                            self._interrupt_worker_locked()

                    if (
                        interrupt_sent
                        and not received_execute_result
                        and now >= interrupt_deadline
                    ):
                        with self._lock:
                            self._restart_worker_locked()
                        break

                for request_id in list(pending_input_requests.keys()):
                    pending_input_requests.pop(request_id, None)
                    self._pop_input_queue(request_id)
            except Exception as exc:
                error_message = f"PyRepl worker failed: {exc}"
                error_details = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "filename": None,
                    "line": None,
                    "column": None,
                    "snippet": None,
                    "pointer": None,
                    "summary": error_message,
                    "user_traceback": "",
                    "full_traceback": "",
                }
                stderr_parts.append(error_message + "\n")
                await self._emit_custom_event(
                    event_emitter,
                    "kernel_stderr",
                    {"text": error_message + "\n"},
                )
                for request_id in list(pending_input_requests.keys()):
                    pending_input_requests.pop(request_id, None)
                    self._pop_input_queue(request_id)

            if timed_out:
                timeout_message = (
                    "Execution timed out after "
                    f"{self._format_timeout_seconds(timeout_seconds)} seconds"
                )
                error_message = timeout_message
                error_details = self._build_timeout_error_details(timeout_message)
                if timeout_message + "\n" not in stderr_parts:
                    stderr_parts.append(timeout_message + "\n")
                    await self._emit_custom_event(
                        event_emitter,
                        "kernel_stderr",
                        {"text": timeout_message + "\n"},
                    )

            execution_time_ms = (time.time() - start_time) * 1000

            result = {
                "success": error_message is None,
                "stdout": "".join(stdout_parts),
                "stderr": "".join(stderr_parts),
                "return_value": return_value,
                "error": error_message,
                "error_details": error_details,
                "execution_time_ms": execution_time_ms,
            }

            self._append_audit_entry(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "instance_id": self._instance_id,
                    "execution_id": execution_id,
                    "code": code,
                    "result": result,
                    "timeout_seconds": timeout_seconds,
                    "input_idle_timeout_seconds": self.input_idle_timeout_seconds,
                    "self_reference_attached": self._self_reference is not None,
                }
            )

            return result

    async def reset(self) -> str:
        """Reset REPL runtime variables for this session."""
        request_id = uuid.uuid4().hex

        async with self._operation_lock:
            with self._lock:
                self.namespace.clear()
                if self._self_reference is not None:
                    self.namespace[SELF_REFERENCE_GLOBAL_NAME] = self._self_reference

                self._ensure_worker_locked()
                self._send_worker_command_locked(
                    {
                        "type": COMMAND_RESET,
                        "request_id": request_id,
                        "self_reference_enabled": self._self_reference is not None,
                    }
                )

            deadline = time.monotonic() + 5.0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with self._lock:
                        self._restart_worker_locked()
                    return "REPL 已重置，所有变量已清除"

                event = await self._receive_worker_event(min(0.1, remaining))
                if event is None:
                    continue

                event_type = str(event.get("type", ""))
                if event_type == EVENT_SELF_REFERENCE_CALL:
                    response = self._execute_self_reference_call(event)
                    with self._lock:
                        self._send_worker_command_locked(response)
                    continue

                if (
                    event_type == EVENT_RESET_RESULT
                    and str(event.get("request_id", "")) == request_id
                ):
                    message = event.get("message")
                    return (
                        str(message)
                        if isinstance(message, str)
                        else "REPL 已重置，所有变量已清除"
                    )

    async def list_variables(self) -> List[Dict[str, str]]:
        """List currently defined user variables in REPL namespace."""
        request_id = uuid.uuid4().hex

        async with self._operation_lock:
            with self._lock:
                self._ensure_worker_locked()
                self._send_worker_command_locked(
                    {
                        "type": COMMAND_LIST_VARIABLES,
                        "request_id": request_id,
                        "self_reference_enabled": self._self_reference is not None,
                    }
                )

            deadline = time.monotonic() + 5.0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with self._lock:
                        self._restart_worker_locked()
                    return []

                event = await self._receive_worker_event(min(0.1, remaining))
                if event is None:
                    continue

                event_type = str(event.get("type", ""))
                if event_type == EVENT_SELF_REFERENCE_CALL:
                    response = self._execute_self_reference_call(event)
                    with self._lock:
                        self._send_worker_command_locked(response)
                    continue

                if (
                    event_type == EVENT_LIST_VARIABLES_RESULT
                    and str(event.get("request_id", "")) == request_id
                ):
                    variables = event.get("variables")
                    if isinstance(variables, list):
                        return [
                            {
                                "name": str(item.get("name", "")),
                                "type": str(item.get("type", "")),
                            }
                            for item in variables
                            if isinstance(item, dict)
                        ]
                    return []

    def bind_history(self, key: str, history: List[Dict[str, Any]]) -> None:
        """Bind a history list to attached self-reference memory store."""

        with self._lock:
            self_reference = self._self_reference

        if self_reference is None:
            raise RuntimeError(
                "No self_reference attached. Call attach_self_reference()"
            )

        self_reference.bind_history(key, history)

    def unbind_history(self, key: str) -> None:
        """Unbind a history key from attached self-reference memory store."""

        with self._lock:
            self_reference = self._self_reference

        if self_reference is None:
            raise RuntimeError(
                "No self_reference attached. Call attach_self_reference()"
            )

        self_reference.unbind_history(key)

    def list_history_keys(self) -> List[str]:
        """List history keys from attached self-reference memory store."""

        with self._lock:
            self_reference = self._self_reference

        if self_reference is None:
            return []

        return self_reference.list_history_keys()

    def close(self) -> None:
        """Close worker process and release resources."""

        with self._lock:
            if self._closed:
                return
            self._shutdown_worker_locked()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["PyRepl"]
