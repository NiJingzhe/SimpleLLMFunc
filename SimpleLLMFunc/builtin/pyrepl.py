"""PyRepl builtin tool for SimpleLLMFunc.

轻量级 Python REPL，支持实时 streaming 输出，不依赖 jupyter_client。
"""

from __future__ import annotations

import asyncio
import builtins
import io
import queue
import sys
import threading
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
from SimpleLLMFunc.self_reference import SelfReference


class _LineCapture(io.TextIOBase):
    """按行缓冲，每完整一行触发一次 callback"""

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.callback(line)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self.callback(self._buf)
            self._buf = ""


class _InputRequestTimeoutError(TimeoutError):
    """Raised when waiting for a tool input request times out."""


class _InputRequestInterruptedError(Exception):
    """Raised when waiting for tool input is interrupted externally."""


class PyRepl:
    """轻量级 Python REPL

    基于 threading + sys.stdout 重定向实现，支持：
    - 实时 stdout/stderr streaming
    - 变量跨调用持久化
    - 独立线程执行，不阻塞主线程

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
    SELF_REFERENCE_GLOBAL_NAME = "self_reference"
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

        self.namespace: Dict[str, Any] = {}
        self._tools: Optional[List[Tool]] = None
        self._lock = threading.Lock()
        self.execution_timeout_seconds = execution_timeout
        self.input_idle_timeout_seconds = input_idle_timeout
        self._self_reference: Optional[SelfReference] = None

        if self_reference is not None:
            self.attach_self_reference(self_reference)

    def attach_self_reference(self, self_reference: SelfReference) -> None:
        """Attach shared self-reference object into REPL global namespace."""

        if not isinstance(self_reference, SelfReference):
            raise ValueError("self_reference must be a SelfReference instance")

        with self._lock:
            self._self_reference = self_reference
            self.namespace[self.SELF_REFERENCE_GLOBAL_NAME] = self_reference

    def detach_self_reference(self) -> None:
        """Detach previously attached self-reference object from namespace."""

        with self._lock:
            self._self_reference = None
            self.namespace.pop(self.SELF_REFERENCE_GLOBAL_NAME, None)

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
            error, and execution_time_ms.
        """
        import time

        start_time = time.time()
        loop = asyncio.get_running_loop()
        timeout_seconds = self.execution_timeout_seconds
        input_idle_timeout_seconds = self.input_idle_timeout_seconds

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        error_msg: Optional[str] = None
        return_value: Optional[str] = None
        done_event = threading.Event()
        stop_event = threading.Event()
        original_input = builtins.input
        timeout_state_lock = threading.Lock()
        timeout_deadline = time.monotonic() + timeout_seconds
        pending_input_waiters = 0

        def format_timeout_seconds(seconds: float) -> str:
            if float(seconds).is_integer():
                return str(int(seconds))
            return f"{seconds:g}"

        def mark_input_wait_start() -> None:
            nonlocal pending_input_waiters
            with timeout_state_lock:
                pending_input_waiters += 1

        def mark_input_wait_end(reset_timeout: bool) -> None:
            nonlocal pending_input_waiters, timeout_deadline
            with timeout_state_lock:
                if pending_input_waiters > 0:
                    pending_input_waiters -= 1
                if reset_timeout:
                    timeout_deadline = time.monotonic() + timeout_seconds

        def is_timeout_reached() -> bool:
            with timeout_state_lock:
                if pending_input_waiters > 0:
                    return False
                deadline = timeout_deadline
            return time.monotonic() >= deadline

        def emit_custom_event(event_name: str, data: Dict[str, Any]) -> None:
            if not event_emitter:
                return

            try:
                asyncio.run_coroutine_threadsafe(
                    event_emitter.emit(event_name, data), loop
                )
            except Exception:
                # 事件发射失败不应影响代码执行主流程
                pass

        def on_output(line: str) -> None:
            stdout_parts.append(line + "\n")
            emit_custom_event("kernel_stdout", {"text": line + "\n"})

        def on_error(line: str) -> None:
            stderr_parts.append(line + "\n")
            emit_custom_event("kernel_stderr", {"text": line + "\n"})

        def input_hook(prompt: str = "") -> str:
            if event_emitter is None:
                mark_input_wait_start()
                got_value = False
                try:
                    value = original_input(prompt)
                    got_value = True
                    return value
                finally:
                    mark_input_wait_end(reset_timeout=got_value)

            request_id = uuid.uuid4().hex
            request_queue = self._register_input_queue(request_id)
            emit_custom_event(
                "kernel_input_request",
                {
                    "request_id": request_id,
                    "prompt": prompt,
                    "idle_timeout_seconds": input_idle_timeout_seconds,
                },
            )

            mark_input_wait_start()
            got_value = False
            request_deadline = time.monotonic() + input_idle_timeout_seconds
            try:
                while not stop_event.is_set():
                    remaining_seconds = request_deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        raise _InputRequestTimeoutError(
                            "Input request timed out after "
                            f"{format_timeout_seconds(input_idle_timeout_seconds)} seconds"
                        )

                    try:
                        wait_timeout = min(0.1, remaining_seconds)
                        value = request_queue.get(timeout=wait_timeout)
                        got_value = True
                        return value
                    except queue.Empty:
                        continue
            finally:
                self._pop_input_queue(request_id)
                mark_input_wait_end(reset_timeout=got_value)

            raise _InputRequestInterruptedError("Input request interrupted")

        def run_code():
            nonlocal error_msg, return_value

            with self._lock:
                if self._self_reference is not None:
                    self.namespace[self.SELF_REFERENCE_GLOBAL_NAME] = (
                        self._self_reference
                    )

                old_out = sys.stdout
                old_err = sys.stderr
                old_input = builtins.input

                cap_out = _LineCapture(on_output)
                cap_err = _LineCapture(on_error)
                sys.stdout = cap_out
                sys.stderr = cap_err
                builtins.input = input_hook

                try:
                    code_obj = compile(code, "<input>", "exec")
                    exec(code_obj, self.namespace)
                    try:
                        code_obj = compile(code, "<input>", "eval")
                        result = eval(code_obj, self.namespace)
                        if result is not None:
                            return_value = repr(result)
                    except SyntaxError:
                        pass
                except SyntaxError:
                    try:
                        code_obj = compile(code, "<input>", "eval")
                        result = eval(code_obj, self.namespace)
                        if result is not None:
                            return_value = repr(result)
                            on_output(return_value)
                    except SyntaxError:
                        pass
                    except _InputRequestTimeoutError as exc:
                        error_msg = str(exc)
                        on_error(error_msg)
                    except _InputRequestInterruptedError:
                        if not stop_event.is_set():
                            error_msg = "Input request interrupted"
                            on_error(error_msg)
                    except Exception:
                        error_msg = traceback.format_exc()
                        on_error(error_msg)
                except _InputRequestTimeoutError as exc:
                    error_msg = str(exc)
                    on_error(error_msg)
                except _InputRequestInterruptedError:
                    if not stop_event.is_set():
                        error_msg = "Input request interrupted"
                        on_error(error_msg)
                except Exception:
                    error_msg = traceback.format_exc()
                    on_error(error_msg)
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err
                    builtins.input = old_input
                    done_event.set()

        thread = threading.Thread(target=run_code, daemon=True)
        thread.start()
        poll_interval_seconds = 0.1
        timed_out = False
        while not done_event.is_set():
            await asyncio.sleep(poll_interval_seconds)
            if done_event.is_set():
                break
            if is_timeout_reached():
                timed_out = True
                break

        if timed_out and error_msg is None:
            stop_event.set()
            await asyncio.to_thread(done_event.wait, 1)
            error_msg = (
                "Execution timed out after "
                f"{format_timeout_seconds(timeout_seconds)} seconds"
            )
            stderr_parts.append(error_msg + "\n")
            if event_emitter:
                await event_emitter.emit("kernel_stderr", {"text": error_msg + "\n"})

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "success": error_msg is None,
            "stdout": "".join(stdout_parts),
            "stderr": "".join(stderr_parts),
            "return_value": return_value,
            "error": error_msg,
            "execution_time_ms": execution_time_ms,
        }

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

    async def reset(self) -> str:
        """Reset REPL runtime variables for this session."""
        with self._lock:
            self.namespace.clear()
            if self._self_reference is not None:
                self.namespace[self.SELF_REFERENCE_GLOBAL_NAME] = self._self_reference
        return "REPL 已重置，所有变量已清除"

    async def list_variables(self) -> List[Dict[str, str]]:
        """List currently defined user variables in REPL namespace."""
        with self._lock:
            vars_list = [
                {"name": name, "type": type(value).__name__}
                for name, value in self.namespace.items()
                if not name.startswith("_") and name != self.SELF_REFERENCE_GLOBAL_NAME
            ]
        return vars_list


__all__ = ["PyRepl"]
