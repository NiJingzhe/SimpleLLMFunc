"""PyRepl builtin tool for SimpleLLMFunc.

轻量级 Python REPL，支持实时 streaming 输出，不依赖 jupyter_client。
"""

from __future__ import annotations

import asyncio
import io
import sys
import threading
import traceback
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

    def __init__(self):
        self.namespace: Dict[str, Any] = {}
        self._tools: Optional[List[Tool]] = None
        self._lock = threading.Lock()

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

        def on_output(line: str) -> None:
            stdout_parts.append(line + "\n")
            if event_emitter and loop:
                asyncio.run_coroutine_threadsafe(
                    event_emitter.emit("kernel_stdout", {"text": line + "\n"}), loop
                )

        def on_error(line: str) -> None:
            stderr_parts.append(line + "\n")
            if event_emitter and loop:
                asyncio.run_coroutine_threadsafe(
                    event_emitter.emit("kernel_stderr", {"text": line + "\n"}), loop
                )

        def run_code():
            nonlocal error_msg, return_value

            with self._lock:
                old_out = sys.stdout
                old_err = sys.stderr

                cap_out = _LineCapture(on_output)
                cap_err = _LineCapture(on_error)
                sys.stdout = cap_out
                sys.stderr = cap_err

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
                    done_event.set()

        thread = threading.Thread(target=run_code, daemon=True)
        thread.start()
        completed = await asyncio.to_thread(done_event.wait, 30)

        if not completed and error_msg is None:
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
