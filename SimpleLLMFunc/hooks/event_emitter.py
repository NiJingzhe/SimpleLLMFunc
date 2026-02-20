"""Tool 事件发射器 - 允许在 tool 内部发射自定义事件

该模块提供 ToolEventEmitter 类，允许用户自定义的 tool 函数在执行过程中
发射自定义事件，这些事件会被汇入到 Event Stream 中，供外部观察者消费。

使用方式：
    @tool(name="my_tool", description="...")
    async def my_tool(arg1: str, event_emitter: ToolEventEmitter = None):
        if event_emitter:
            await event_emitter.emit("progress", {"percent": 50})
        # ... 执行逻辑
        return result
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, List, Optional, TYPE_CHECKING

from SimpleLLMFunc.hooks.events import (
    ReActEventType,
    CustomEvent,
)
from SimpleLLMFunc.logger.context_manager import get_current_trace_id
from SimpleLLMFunc.logger.logger import get_current_context_attribute

if TYPE_CHECKING:
    from SimpleLLMFunc.hooks.stream import EventYield


@dataclass
class ToolEventEmitter:
    """工具事件发射器

    允许在 tool 函数内部发射自定义事件，这些事件会被汇入到 Event Stream 中。

    Attributes:
        _queue: 事件队列，用于存储发射的事件
        _trace_id: 当前追踪 ID
        _func_name: 当前函数名称
        _iteration: 当前迭代次数
    """

    _queue: asyncio.Queue[EventYield] = field(default_factory=asyncio.Queue, repr=False)
    _trace_id: str = ""
    _func_name: str = ""
    _iteration: int = 0

    def __post_init__(self):
        """初始化后处理"""
        if not self._trace_id:
            self._trace_id = get_current_trace_id() or ""
        if not self._func_name:
            self._func_name = (
                get_current_context_attribute("function_name") or "unknown"
            )

    async def emit(
        self,
        event_name: str,
        data: Any = None,
    ) -> None:
        """发射自定义事件

        Args:
            event_name: 事件名称，应为有意义的标识符
            data: 事件数据，可以是任意可序列化的对象
        """
        event = CustomEvent(
            event_type=ReActEventType.CUSTOM_EVENT,
            timestamp=datetime.now(timezone.utc),
            trace_id=self._trace_id,
            func_name=self._func_name,
            iteration=self._iteration,
            event_name=event_name,
            data=data,
        )

        from SimpleLLMFunc.hooks.stream import EventYield

        event_yield = EventYield(event=event)
        await self._queue.put(event_yield)

    async def emit_batch(
        self,
        events: List[tuple[str, Any]],
    ) -> None:
        """批量发射自定义事件

        Args:
            events: 事件列表，每项为 (event_name, data) 元组
        """
        for event_name, data in events:
            await self.emit(event_name, data)

    async def get_events(self) -> List[EventYield]:
        """获取已发射的事件列表

        获取后清空队列。

        Returns:
            已发射的事件列表
        """
        events = []
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break
        return events

    def has_events(self) -> bool:
        """检查是否有待处理的事件

        Returns:
            是否有事件
        """
        return not self._queue.empty()


@dataclass
class NoOpEventEmitter:
    """空操作事件发射器

    当 event stream 未启用时使用的默认发射器，
    不会产生任何实际事件，用于避免 tool 函数中的空值检查。
    """

    async def emit(self, event_name: str, data: Any = None) -> None:
        """空操作，不发射任何事件"""
        pass

    async def emit_batch(self, events: List[tuple[str, Any]]) -> None:
        """空操作，不发射任何事件"""
        pass

    async def get_events(self) -> List[Any]:
        """返回空列表"""
        return []

    def has_events(self) -> bool:
        """返回 False"""
        return False


__all__ = [
    "ToolEventEmitter",
    "NoOpEventEmitter",
]
