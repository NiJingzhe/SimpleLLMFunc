"""iPython Kernel builtin tool for SimpleLLMFunc.

提供 iPython Kernel 代码执行能力，支持实时 streaming 输出。
"""

from __future__ import annotations

import uuid
import asyncio
from typing import Any, Dict, List, Optional

from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter


class Kernel:
    """iPython Kernel 管理类

    提供交互式 Python 代码执行能力，支持：
    - 实时 stdout/stderr streaming
    - 变量管理
    - Session 隔离

    Usage:
        kernel = Kernel()
        tools = kernel.toolset

        @llm_chat(toolkit=tools + [...], ...)
        async def chat(message: str, history=None):
            '''Python 编程助手'''
    """

    def __init__(
        self,
        kernel_name: str = "python3",
        timeout: int = 30,
    ):
        self.kernel_name = kernel_name
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())[:8]
        self._km = None
        self._client = None
        self._tools = None

    @property
    def toolset(self) -> List[Tool]:
        """返回绑定到该 kernel 实例的 tool 列表"""
        if self._tools is None:
            self._tools = self._create_tools()
        return self._tools

    def _create_tools(self) -> List[Tool]:
        """为该 kernel 实例创建 tool 列表"""
        tools = []

        execute_tool = Tool(
            name="execute_code",
            description=self.execute.__doc__ or "执行 Python 代码",
            func=self.execute,
        )
        tools.append(execute_tool)

        reset_tool = Tool(
            name="reset_kernel",
            description=self.reset.__doc__ or "重置 kernel 状态",
            func=self.reset,
        )
        tools.append(reset_tool)

        list_vars_tool = Tool(
            name="list_variables",
            description=self.list_variables.__doc__ or "列出变量",
            func=self.list_variables,
        )
        tools.append(list_vars_tool)

        close_tool = Tool(
            name="close_kernel",
            description=self.close.__doc__ or "关闭 kernel",
            func=self.close,
        )
        tools.append(close_tool)

        return tools

    async def _ensure_started(self):
        """确保 kernel 已启动"""
        if self._client is not None:
            return

        try:
            from jupyter_client import KernelManager
        except ImportError:
            raise ImportError(
                "jupyter-client is required for Kernel functionality. "
                "Install it with: pip install jupyter-client"
            )

        self._km = KernelManager(kernel_name=self.kernel_name)
        await asyncio.get_event_loop().run_in_executor(None, self._km.start_kernel)

        self._client = self._km.client()
        self._client.start_channels()

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client.wait_for_ready(timeout=self.timeout)
            )
        except Exception as e:
            await self.close()
            raise RuntimeError(f"Kernel failed to start: {e}")

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

        await self._ensure_started()

        stdout_parts = []
        stderr_parts = []
        return_value = None
        error_msg = None

        try:
            msg_id = self._client.execute(code)

            while True:
                try:
                    msg = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._client.get_iopub_msg(timeout=self.timeout)
                    )
                except Exception:
                    break

                msg_type = msg["header"]["msg_type"]
                content = msg["content"]

                if msg_type == "stream":
                    text = content.get("text", "")
                    name = content.get("name", "stdout")

                    if name == "stdout":
                        stdout_parts.append(text)
                        if event_emitter:
                            await event_emitter.emit(
                                "kernel_stdout",
                                {"text": text, "session_id": self.session_id},
                            )
                    else:
                        stderr_parts.append(text)
                        if event_emitter:
                            await event_emitter.emit(
                                "kernel_stderr",
                                {"text": text, "session_id": self.session_id},
                            )

                elif msg_type == "execute_result":
                    return_value = content.get("data", {}).get("text/plain")
                    if event_emitter:
                        await event_emitter.emit(
                            "kernel_result",
                            {"result": return_value, "session_id": self.session_id},
                        )

                elif msg_type == "error":
                    error_msg = f"{content.get('ename')}: {content.get('evalue')}"
                    if event_emitter:
                        await event_emitter.emit(
                            "kernel_error",
                            {"error": error_msg, "session_id": self.session_id},
                        )

                elif msg_type == "status":
                    if content.get("execution_state") == "idle":
                        break

        except Exception as e:
            error_msg = str(e)
            if event_emitter:
                await event_emitter.emit(
                    "kernel_error",
                    {"error": error_msg, "session_id": self.session_id},
                )

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
        """重置 kernel 状态（清除所有变量）"""
        await self._ensure_started()

        result = await self.execute(code="%reset -f")

        if result["success"]:
            return "Kernel 已重置，所有变量已清除"
        else:
            return f"重置失败: {result['error']}"

    async def list_variables(self) -> List[Dict[str, str]]:
        """列出当前定义的变量"""
        await self._ensure_started()

        result = await self.execute(code="%who_ls")

        if result["success"]:
            return [{"name": result["stdout"].strip()}]
        else:
            return []

    async def close(self) -> str:
        """关闭 kernel"""
        if self._client:
            self._client.stop_channels()
            self._client = None

        if self._km:
            await asyncio.get_event_loop().run_in_executor(
                None, self._km.shutdown_kernel
            )
            self._km = None

        return f"Kernel {self.session_id} 已关闭"


__all__ = ["Kernel"]
