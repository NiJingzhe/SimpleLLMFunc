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

    def __init__(self):
        self.namespace: Dict[str, Any] = {}
        self._tools: Optional[List[Tool]] = None
        self._lock = threading.Lock()

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
            description=self.execute.__doc__ or "执行 Python 代码",
            func=self.execute,
        )
        tools.append(execute_tool)

        reset_tool = Tool(
            name="reset_repl",
            description=self.reset.__doc__ or "重置 REPL 状态",
            func=self.reset,
        )
        tools.append(reset_tool)

        list_vars_tool = Tool(
            name="list_variables",
            description=self.list_variables.__doc__ or "列出变量",
            func=self.list_variables,
        )
        tools.append(list_vars_tool)

        return tools

    async def execute(
        self,
        code: str,
        event_emitter: Optional[ToolEventEmitter] = None,
    ) -> Dict[str, Any]:
        """执行 Python 代码，支持实时 streaming

        Args:
            code: 要执行的 Python 代码
            event_emitter: 事件发射器，用于实时发射 stdout/stderr

        Returns:
            包含 success, stdout, stderr, return_value, error, execution_time_ms 的字典
        """
        import time

        start_time = time.time()
        loop = asyncio.get_running_loop()

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        error_msg: Optional[str] = None
        return_value: Optional[str] = None
        done_event = threading.Event()
        stop_event = threading.Event()
        original_input = builtins.input

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
                return original_input(prompt)

            request_id = uuid.uuid4().hex
            request_queue = self._register_input_queue(request_id)
            emit_custom_event(
                "kernel_input_request",
                {
                    "request_id": request_id,
                    "prompt": prompt,
                },
            )

            try:
                while not stop_event.is_set():
                    try:
                        return request_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
            finally:
                self._pop_input_queue(request_id)

            raise EOFError("Input request interrupted")

        def run_code():
            nonlocal error_msg, return_value

            with self._lock:
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
                    except Exception:
                        error_msg = traceback.format_exc()
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
        completed = await asyncio.to_thread(done_event.wait, 30)

        if not completed and error_msg is None:
            stop_event.set()
            await asyncio.to_thread(done_event.wait, 1)
            error_msg = "Execution timed out after 30 seconds"
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

    async def reset(self) -> str:
        """重置 REPL 状态（清除所有变量）"""
        with self._lock:
            self.namespace.clear()
        return "REPL 已重置，所有变量已清除"

    async def list_variables(self) -> List[Dict[str, str]]:
        """列出当前定义的变量"""
        with self._lock:
            vars_list = [
                {"name": name, "type": type(value).__name__}
                for name, value in self.namespace.items()
                if not name.startswith("_")
            ]
        return vars_list


__all__ = ["PyRepl"]
