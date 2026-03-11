"""ReAct 事件系统

提供 ReAct 循环的事件定义和事件流支持。
"""

from SimpleLLMFunc.hooks.events import (
    ReActEvent,
    ReActEventType,
    ReactStartEvent,
    ReactIterationStartEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    LLMCallEndEvent,
    LLMCallErrorEvent,
    ToolCallsBatchStartEvent,
    ToolCallStartEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallEndEvent,
    ToolCallErrorEvent,
    ToolCallsBatchEndEvent,
    ReactIterationEndEvent,
    ReactEndEvent,
    CustomEvent,
)
from SimpleLLMFunc.hooks.event_emitter import (
    ToolEventEmitter,
    NoOpEventEmitter,
)
from SimpleLLMFunc.hooks.event_bus import EventBus
from SimpleLLMFunc.hooks.input_stream import (
    AgentInputRouter,
    InputRouteResult,
    ToolInputRequest,
    UserInputEvent,
)
from SimpleLLMFunc.hooks.stream import (
    EventYield,
    EventOrigin,
    ReactOutput,
    ResponseYield,
    events_only,
    filter_events,
    is_event_yield,
    is_response_yield,
    responses_only,
    with_event_observer,
)

__all__ = [
    # 基础类型
    "ReActEvent",
    "ReActEventType",
    # 事件类型
    "ReactStartEvent",
    "ReactIterationStartEvent",
    "LLMCallStartEvent",
    "LLMChunkArriveEvent",
    "LLMCallEndEvent",
    "LLMCallErrorEvent",
    "ToolCallsBatchStartEvent",
    "ToolCallStartEvent",
    "ToolCallArgumentsDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallErrorEvent",
    "ToolCallsBatchEndEvent",
    "ReactIterationEndEvent",
    "ReactEndEvent",
    "CustomEvent",
    # Stream 类型
    "ResponseYield",
    "EventYield",
    "EventOrigin",
    "ReactOutput",
    # 类型守卫
    "is_response_yield",
    "is_event_yield",
    # 过滤器
    "responses_only",
    "events_only",
    "filter_events",
    # 装饰器
    "with_event_observer",
    # 事件发射器
    "ToolEventEmitter",
    "NoOpEventEmitter",
    "EventBus",
    # 输入流
    "UserInputEvent",
    "ToolInputRequest",
    "InputRouteResult",
    "AgentInputRouter",
]
