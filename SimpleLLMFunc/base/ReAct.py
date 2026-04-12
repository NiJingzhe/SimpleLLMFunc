"""Core execution pipeline handling LLM and tool-call orchestration.

This module implements the ReAct (Reasoning and Acting) pattern for orchestrating
LLM calls with tool usage. It manages:
1. Initial LLM invocation with or without streaming
2. Tool call extraction and execution
3. Iterative LLM-tool interaction loops
4. Message history management and context preservation
5. Maximum tool call limit enforcement
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import copy
import inspect
import json
import time

from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.type import (
    MessageList,
    ToolDefinitionList,
    ToolCall,
    ToolCallArguments,
    ToolResult,
)
from SimpleLLMFunc.logger import app_log, push_debug
from SimpleLLMFunc.logger.logger import get_current_context_attribute, get_location
from SimpleLLMFunc.logger.context_manager import get_current_trace_id
from SimpleLLMFunc.hooks.event_bus import EventBus
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.hooks.stream import EventYield, ReactOutput, ResponseYield
from SimpleLLMFunc.hooks.events import (
    ReActEventType,
    ReactStartEvent,
    ReactIterationStartEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    LLMCallEndEvent,
    ToolCallsBatchStartEvent,
    ToolCallStartEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallEndEvent,
    ToolCallErrorEvent,
    ToolCallsBatchEndEvent,
    ToolCallResult,
    ReactIterationEndEvent,
    ReactEndEvent,
)
from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter, NoOpEventEmitter
from SimpleLLMFunc.type.tool_call import dict_to_tool_call, tool_call_to_dict
from SimpleLLMFunc.base.messages import (
    build_assistant_response_message,
    build_assistant_tool_message,
    extract_usage_from_response,
)
from SimpleLLMFunc.base.post_process import (
    extract_content_from_response,
    extract_content_from_stream_response,
)
from SimpleLLMFunc.base.tool_call import (
    accumulate_tool_calls_from_chunks,
    extract_reasoning_details,
    extract_reasoning_details_from_stream,
    parse_tool_call_arguments,
    extract_tool_calls,
    extract_tool_calls_from_stream_response,
    process_tool_calls,
)
from SimpleLLMFunc.base.react_hooks import ReActState, run_react_hook

from SimpleLLMFunc.observability.langfuse_client import (
    coerce_langfuse_metadata,
    get_langfuse_trace_context,
    langfuse_client,
)


async def _yield_pending_event_bus_events(
    event_bus: EventBus,
) -> AsyncGenerator[EventYield, None]:
    while not event_bus.empty():
        yield await event_bus.get()


_ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool", "function"}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clone_message_list(messages: MessageList) -> MessageList:
    return [copy.deepcopy(message) for message in messages]


def _read_context_token_counters() -> tuple[int, int]:
    return (
        _as_int(get_current_context_attribute("input_tokens") or 0),
        _as_int(get_current_context_attribute("output_tokens") or 0),
    )


def _usage_from_context_delta(
    input_before: int,
    output_before: int,
) -> Optional[CompletionUsage]:
    input_after, output_after = _read_context_token_counters()
    prompt_tokens = max(0, input_after - input_before)
    completion_tokens = max(0, output_after - output_before)

    if prompt_tokens == 0 and completion_tokens == 0:
        return None

    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _message_to_dict(message: Any, index: int) -> Dict[str, Any]:
    if isinstance(message, dict):
        return copy.deepcopy(message)

    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=False)
        if isinstance(dumped, dict):
            return dumped

    raise ValueError(f"message at index {index} must be a dict-like chat message")


def _is_valid_content_for_role(role: str, content: Any) -> bool:
    if role == "system":
        return isinstance(content, str)
    if role == "user":
        return isinstance(content, (str, list))
    if role == "assistant":
        return content is None or isinstance(content, (str, list))
    if role in {"tool", "function"}:
        return isinstance(content, str)
    return False


def _validate_message_shape(message: Dict[str, Any], index: int) -> None:
    role = message.get("role")
    if not isinstance(role, str) or role not in _ALLOWED_MESSAGE_ROLES:
        raise ValueError(f"Invalid message role at index {index}: {role!r}")

    if "content" not in message:
        raise ValueError(
            f"Message at index {index} is missing required field 'content'"
        )

    if not _is_valid_content_for_role(role, message.get("content")):
        raise ValueError(
            f"Invalid content for role '{role}' at index {index}: "
            f"{type(message.get('content')).__name__}"
        )

    if role == "assistant" and "tool_calls" in message:
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise ValueError("assistant.tool_calls must be a list when present")
        if isinstance(tool_calls, list):
            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    raise ValueError(
                        "assistant.tool_calls entries must be dict objects "
                        f"(index {index}, tool_call {call_index})"
                    )
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(
                        "assistant.tool_calls entries must contain non-empty id "
                        f"(index {index}, tool_call {call_index})"
                    )

    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ValueError("tool messages must contain non-empty tool_call_id")


def _validate_tool_linkage(messages: MessageList) -> None:
    pending_tool_call_ids: List[str] = []

    for index, message in enumerate(messages):
        role = message.get("role")

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if pending_tool_call_ids and not tool_calls:
                raise ValueError(
                    "Missing tool results before next assistant message "
                    f"(index {index})"
                )
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    call_id = tool_call.get("id")
                    if isinstance(call_id, str) and call_id:
                        pending_tool_call_ids.append(call_id)
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not pending_tool_call_ids:
                raise ValueError(
                    "Tool message appears without preceding assistant tool_calls "
                    f"(index {index})"
                )
            if tool_call_id not in pending_tool_call_ids:
                raise ValueError(
                    "tool_call_id does not match pending assistant tool_calls "
                    f"(index {index})"
                )
            pending_tool_call_ids.remove(tool_call_id)
            continue

        if pending_tool_call_ids:
            raise ValueError(
                "Pending assistant tool_calls must be followed by matching tool "
                f"messages before role '{role}' (index {index})"
            )

    if pending_tool_call_ids:
        raise ValueError("Unmatched assistant tool_calls without tool results")


def _normalize_and_validate_messages(messages: MessageList) -> MessageList:
    normalized = [
        _message_to_dict(message, index) for index, message in enumerate(messages)
    ]
    for index, message in enumerate(normalized):
        _validate_message_shape(message, index)
    _validate_tool_linkage(normalized)
    return normalized


@dataclass
class _SingleLLMCallResult:
    messages: MessageList
    last_response: Any
    content: str
    tool_calls: List[Dict[str, Any]]
    reasoning_details: List[Dict[str, Any]]
    usage: Optional[CompletionUsage]
    aborted: bool


@dataclass
class _StreamingToolCallState:
    """Streaming state per tool-call index while parsing argument deltas."""

    tool_call_id: str = ""
    tool_name: str = ""
    arguments: str = ""
    emitted_argument_values: Dict[str, str] = field(default_factory=dict)


def _stringify_argument_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _collect_stream_argument_deltas(
    state: _StreamingToolCallState,
    parsed_arguments: ToolCallArguments,
) -> List[Tuple[str, str]]:
    """Compute per-argument text deltas from parsed arguments snapshot."""

    deltas: List[Tuple[str, str]] = []

    for argname, argvalue in parsed_arguments.items():
        current_value = _stringify_argument_value(argvalue)
        previous_value = state.emitted_argument_values.get(argname, "")

        if current_value == previous_value:
            continue

        if previous_value and previous_value.startswith(current_value):
            # Ignore temporary parser regression caused by partial payloads.
            continue

        if previous_value and current_value.startswith(previous_value):
            delta = current_value[len(previous_value) :]
        else:
            delta = current_value

        if delta:
            deltas.append((argname, delta))

        state.emitted_argument_values[argname] = current_value

    return deltas


def _collect_tool_argument_delta_payloads(
    tool_call_chunks: List[Dict[str, Any]],
    stream_states: Dict[int, _StreamingToolCallState],
) -> List[Tuple[str, str, str, str]]:
    """Collect (tool_name, tool_call_id, argname, arg_delta) payloads from chunks."""

    payloads: List[Tuple[str, str, str, str]] = []

    for chunk in tool_call_chunks:
        index = chunk.get("index")
        if index is None:
            continue

        state = stream_states.setdefault(index, _StreamingToolCallState())

        chunk_id = chunk.get("id")
        if isinstance(chunk_id, str) and chunk_id:
            state.tool_call_id = chunk_id

        function_chunk = chunk.get("function")
        if isinstance(function_chunk, dict):
            chunk_name = function_chunk.get("name")
            if isinstance(chunk_name, str) and chunk_name:
                state.tool_name = chunk_name

            argument_delta = function_chunk.get("arguments")
            if isinstance(argument_delta, str) and argument_delta:
                state.arguments += argument_delta

        if not state.arguments or not state.tool_call_id:
            continue

        parsed_arguments = parse_tool_call_arguments(
            state.arguments,
            allow_closure=True,
        )
        if parsed_arguments is None:
            continue

        deltas = _collect_stream_argument_deltas(state, parsed_arguments)
        for argname, argcontent_delta in deltas:
            payloads.append(
                (
                    state.tool_name,
                    state.tool_call_id,
                    argname,
                    argcontent_delta,
                )
            )

    return payloads


async def _process_tool_calls_with_events_gen(
    tool_calls: List[Dict[str, Any]],
    messages: MessageList,
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    enable_event: bool,
    event_bus: Optional[EventBus],
    trace_id: str,
    func_name: str,
    iteration: int,
    abort_signal: Optional[AbortSignal] = None,
) -> AsyncGenerator[Union[EventYield, MessageList], None]:
    """处理工具调用并发射事件（异步生成器版本）

    Args:
        tool_calls: 工具调用列表
        messages: 消息列表
        tool_map: 工具映射
        enable_event: 是否启用事件
        trace_id: 追踪ID
        func_name: 函数名
        iteration: 迭代次数

    Yields:
        EventYield: 事件对象
        MessageList: 更新后的消息列表（作为最后一个 yield）
    """

    def _abort_requested() -> bool:
        return abort_signal is not None and abort_signal.is_aborted

    if not tool_calls:
        yield messages
        return

    if _abort_requested():
        yield messages
        return

    trace_context = get_langfuse_trace_context()

    # 准备执行环境
    from SimpleLLMFunc.base.tool_call.execution import _execute_single_tool_call
    import asyncio

    fallback_event_queue: asyncio.Queue[EventYield] = asyncio.Queue()

    async def _publish_tool_event(
        event: Any,
        *,
        origin_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if event_bus is not None:
            await event_bus.emit_event(
                cast(Any, event),
                origin_overrides=origin_overrides,
            )
            return

        await fallback_event_queue.put(EventYield(event=event))

    # 发射工具调用批次开始事件
    if enable_event:
        try:
            tool_calls_typed: List[ToolCall] = [
                dict_to_tool_call(tc) for tc in tool_calls
            ]
            batch_start_event = ToolCallsBatchStartEvent(
                event_type=ReActEventType.TOOL_CALLS_BATCH_START,
                timestamp=datetime.now(timezone.utc),
                trace_id=trace_id,
                func_name=func_name,
                iteration=iteration,
                tool_calls=tool_calls_typed,
                batch_size=len(tool_calls),
            )
            if event_bus is not None:
                yield await event_bus.emit_and_get(batch_start_event)
            else:
                yield EventYield(event=batch_start_event)
        except Exception:
            pass

    async def _execute_with_events_task(
        tool_call: Dict[str, Any],
    ) -> tuple[
        Dict[str, Any],
        List[Dict[str, Any]],
        bool,
        Optional[ToolResult],
        Optional[Exception],
        float,
    ]:
        """执行单个工具调用并将事件放入队列"""
        tool_call_id = tool_call.get("id", "")
        function_call = tool_call.get("function", {})
        tool_name = function_call.get("name", "")
        arguments_str = function_call.get("arguments", "{}")

        # 发射工具调用开始事件
        if enable_event:
            try:
                parsed_arguments_start = parse_tool_call_arguments(
                    arguments_str,
                    allow_closure=True,
                )
                arguments_start: ToolCallArguments = (
                    parsed_arguments_start if parsed_arguments_start is not None else {}
                )
                tool_call_typed_start = dict_to_tool_call(tool_call)
                await _publish_tool_event(
                    ToolCallStartEvent(
                        event_type=ReActEventType.TOOL_CALL_START,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=trace_id,
                        func_name=func_name,
                        iteration=iteration,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        arguments=arguments_start,
                        tool_call=tool_call_typed_start,
                    ),
                    origin_overrides={
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                    },
                )
            except Exception:
                pass

        # 执行工具调用
        tool_start_time = time.time()
        tool_result: Optional[ToolResult] = None
        tool_error: Optional[Exception] = None

        # 为每个 tool call 创建独立 emitter，携带 tool 上下文用于 UI 定位
        tool_event_emitter = ToolEventEmitter(
            _queue=cast(Any, fallback_event_queue),
            _trace_id=trace_id,
            _func_name=func_name,
            _iteration=iteration,
            _tool_name=tool_name,
            _tool_call_id=tool_call_id,
            _event_bus=event_bus,
        )

        try:
            (
                tool_call_dict,
                messages_to_append,
                is_multimodal,
            ) = await _execute_single_tool_call(
                tool_call,
                tool_map,
                event_emitter=tool_event_emitter,
                trace_context=trace_context,
            )

            # 从消息中提取工具结果
            if messages_to_append:
                tool_message = messages_to_append[0]
                if tool_message.get("role") == "tool":
                    content = tool_message.get("content", "")
                    try:
                        parsed = json.loads(content)
                        tool_result = (
                            parsed if isinstance(parsed, (str, dict, list)) else content
                        )
                    except Exception:
                        tool_result = content
        except Exception as e:
            tool_error = e
            # 如果执行失败，创建错误消息
            tool_call_dict = tool_call
            messages_to_append = [
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                }
            ]
            is_multimodal = False

        tool_execution_time = time.time() - tool_start_time

        # 发射工具调用结束或错误事件
        if enable_event:
            try:
                parsed_arguments_end = parse_tool_call_arguments(
                    arguments_str,
                    allow_closure=True,
                )
                arguments_end: ToolCallArguments = (
                    parsed_arguments_end if parsed_arguments_end is not None else {}
                )
                if tool_error:
                    await _publish_tool_event(
                        ToolCallErrorEvent(
                            event_type=ReActEventType.TOOL_CALL_ERROR,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=trace_id,
                            func_name=func_name,
                            iteration=iteration,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            arguments=arguments_end,
                            error=tool_error,
                            error_message=str(tool_error),
                            error_type=type(tool_error).__name__,
                            execution_time=tool_execution_time,
                        ),
                        origin_overrides={
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                        },
                    )
                else:
                    await _publish_tool_event(
                        ToolCallEndEvent(
                            event_type=ReActEventType.TOOL_CALL_END,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=trace_id,
                            func_name=func_name,
                            iteration=iteration,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            arguments=arguments_end,
                            result=tool_result if tool_result is not None else "",
                            execution_time=tool_execution_time,
                            success=tool_error is None,
                        ),
                        origin_overrides={
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                        },
                    )
            except Exception:
                pass

        return (
            tool_call_dict,
            messages_to_append,
            is_multimodal,
            tool_result,
            tool_error,
            tool_execution_time,
        )

    # 启动所有任务
    tasks = [asyncio.create_task(_execute_with_events_task(tc)) for tc in tool_calls]
    task_tool_calls = dict(zip(tasks, tool_calls))
    batch_start_time = time.time()

    # 实时消费事件流（与工具执行并发）
    if enable_event:
        while True:
            if _abort_requested():
                for task in tasks:
                    if task.done():
                        continue
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                abort_reason = abort_signal.reason if abort_signal is not None else ""
                for task, tool_call in task_tool_calls.items():
                    if task.done() and not task.cancelled():
                        continue
                    tool_call_id = tool_call.get("id", "")
                    function_call = tool_call.get("function", {})
                    tool_name = function_call.get("name", "")
                    arguments_str = function_call.get("arguments", "{}")
                    try:
                        parsed_arguments_end = parse_tool_call_arguments(
                            arguments_str,
                            allow_closure=True,
                        )
                        arguments_end: ToolCallArguments = (
                            parsed_arguments_end
                            if parsed_arguments_end is not None
                            else {}
                        )
                        await _publish_tool_event(
                            ToolCallErrorEvent(
                                event_type=ReActEventType.TOOL_CALL_ERROR,
                                timestamp=datetime.now(timezone.utc),
                                trace_id=trace_id,
                                func_name=func_name,
                                iteration=iteration,
                                tool_name=tool_name,
                                tool_call_id=tool_call_id,
                                arguments=arguments_end,
                                error=RuntimeError(abort_reason or "Tool call aborted"),
                                error_message=abort_reason or "Tool call aborted",
                                error_type="CancelledError",
                                execution_time=0.0,
                            ),
                            origin_overrides={
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                            },
                        )
                    except Exception:
                        continue
                yield messages
                return
            queue_empty = (
                event_bus.empty()
                if event_bus is not None
                else fallback_event_queue.empty()
            )
            if all(task.done() for task in tasks) and queue_empty:
                break

            try:
                if event_bus is not None:
                    event = await asyncio.wait_for(event_bus.get(), timeout=0.01)
                else:
                    event = await asyncio.wait_for(
                        fallback_event_queue.get(),
                        timeout=0.01,
                    )
            except asyncio.TimeoutError:
                continue

            yield event

    if _abort_requested():
        for task in tasks:
            if task.done():
                continue
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        yield messages
        return

    # 收集结果
    tool_results: List[ToolCallResult] = []
    # 这里我们只收集成功完成的结果，异常已经在 task 内部捕获
    results = [t.result() for t in tasks if not t.cancelled() and not t.exception()]

    # 收集工具调用结果用于批次结束事件
    for (
        tool_call_dict,
        messages_to_append,
        is_multimodal,
        tool_result,
        tool_error,
        exec_time,
    ) in results:
        tool_call_id = tool_call_dict.get("id", "")
        function_call = tool_call_dict.get("function", {})
        tool_name = function_call.get("name", "")

        tool_result_dict: ToolCallResult = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "result": tool_result if tool_result is not None else "",
            "execution_time": exec_time,
            "success": tool_error is None,
        }
        if tool_error:
            tool_result_dict["error"] = tool_error
        tool_results.append(tool_result_dict)

    # 发射工具调用批次结束事件
    total_execution_time = time.time() - batch_start_time
    if enable_event:
        try:
            success_count = sum(1 for tr in tool_results if tr["success"])
            error_count = len(tool_results) - success_count
            batch_end_event = ToolCallsBatchEndEvent(
                event_type=ReActEventType.TOOL_CALLS_BATCH_END,
                timestamp=datetime.now(timezone.utc),
                trace_id=trace_id,
                func_name=func_name,
                iteration=iteration,
                tool_results=tool_results,
                batch_size=len(tool_calls),
                total_execution_time=total_execution_time,
                success_count=success_count,
                error_count=error_count,
            )
            if event_bus is not None:
                yield await event_bus.emit_and_get(batch_end_event)
            else:
                yield EventYield(event=batch_end_event)
        except Exception:
            pass

    # =========================================================================
    # 后处理：构建消息列表 (替代 process_tool_calls 以避免重复执行)
    # =========================================================================

    # 分类结果
    normal_results: List[List[Dict[str, Any]]] = []
    multimodal_results: List[tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    multimodal_tool_call_ids: set[str] = set()

    for tool_call_dict, messages_to_append, is_multimodal, _, _, _ in results:
        if is_multimodal:
            multimodal_results.append((tool_call_dict, messages_to_append))
            tool_call_id = tool_call_dict.get("id")
            if tool_call_id:
                multimodal_tool_call_ids.add(tool_call_id)
        else:
            normal_results.append(messages_to_append)

    current_messages = messages.copy()

    # 阶段 1: 移除多模态 tool_calls
    if multimodal_tool_call_ids:
        for i in range(len(current_messages) - 1, -1, -1):
            msg = current_messages[i]
            if msg.get("role") == "assistant" and "tool_calls" in msg:  # type: ignore
                original_tool_calls = msg["tool_calls"]  # type: ignore
                filtered_tool_calls = [
                    tc
                    for tc in original_tool_calls  # type: ignore
                    if tc.get("id") not in multimodal_tool_call_ids
                ]

                if not filtered_tool_calls:
                    del msg["tool_calls"]  # type: ignore
                    if msg.get("content") is None:  # type: ignore
                        msg["content"] = ""  # type: ignore
                else:
                    msg["tool_calls"] = filtered_tool_calls  # type: ignore
                break

    # 阶段 2: 追加普通结果
    for msgs in normal_results:
        current_messages.extend(msgs)  # type: ignore

    # 阶段 3: 处理多模态结果
    for tool_call_dict, user_messages in multimodal_results:
        tool_name = tool_call_dict.get("function", {}).get("name", "unknown")
        arguments = tool_call_dict.get("function", {}).get("arguments", "{}")

        assistant_message = {
            "role": "assistant",
            "content": f"我将求助用户使用 {tool_name} 工具来获取结果，使用参数为：{arguments}，请用户按照工具的描述和参数要求，提供符合要求的结果。",
        }
        current_messages.append(assistant_message)  # type: ignore
        current_messages.extend(user_messages)  # type: ignore

    yield cast(MessageList, current_messages)


async def _stream_single_llm_call_outputs(
    *,
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: ToolDefinitionList,
    stream: bool,
    llm_kwargs: Dict[str, Any],
    func_name: str,
    trace_id: str,
    iteration: int,
    emit_event: Callable[..., Awaitable[EventYield]],
    abort_signal: Optional[AbortSignal] = None,
) -> AsyncGenerator[ReactOutput, None]:
    llm_kwargs_filtered = dict(llm_kwargs)
    if not tools:
        llm_kwargs_filtered.pop("tool_choice", None)

    async def _close_stream(stream_obj: Any) -> None:
        close_method = getattr(stream_obj, "aclose", None)
        if callable(close_method):
            try:
                result = close_method()
                if inspect.isawaitable(result):
                    await cast(Awaitable[Any], result)
            except Exception:
                pass

    async def _next_stream_item(stream_iter: Any) -> Any:
        if abort_signal is None:
            return await stream_iter.__anext__()

        abort_task = asyncio.create_task(abort_signal.wait())
        next_task = asyncio.create_task(stream_iter.__anext__())
        done, _ = await asyncio.wait(
            {abort_task, next_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
            abort_task.cancel()
            raise asyncio.CancelledError
        abort_task.cancel()
        return next_task.result()

    def _abort_requested() -> bool:
        return abort_signal is not None and abort_signal.is_aborted

    llm_call_start_time = time.time()
    llm_input_tokens_before, llm_output_tokens_before = _read_context_token_counters()

    yield await emit_event(
        LLMCallStartEvent(
            event_type=ReActEventType.LLM_CALL_START,
            timestamp=datetime.now(timezone.utc),
            trace_id=trace_id,
            func_name=func_name,
            iteration=iteration,
            messages=messages.copy(),
            tools=tools,
            llm_kwargs=llm_kwargs,
            stream=stream,
        )
    )

    content = ""
    tool_calls: List[Dict[str, Any]] = []
    tool_call_chunks: List[Dict[str, Any]] = []
    reasoning_details: List[Dict[str, Any]] = []
    last_response: Any = None
    usage_info: Optional[CompletionUsage] = None
    aborted = False

    if stream:
        reasoning_details_list: List[Dict[str, Any]] = []
        stream_tool_call_states: Dict[int, _StreamingToolCallState] = {}
        chunk_index = 0
        accumulated_content = ""
        stream_response = llm_interface.chat_stream(
            messages=cast(List[Dict[str, Any]], messages),
            tools=tools,
            **llm_kwargs_filtered,
        )
        stream_iter = stream_response.__aiter__()
        while True:
            try:
                chunk = await _next_stream_item(stream_iter)
            except StopAsyncIteration:
                break
            except asyncio.CancelledError:
                aborted = True
                await _close_stream(stream_response)
                break

            chunk_content = extract_content_from_stream_response(chunk, func_name)
            content += chunk_content
            accumulated_content += chunk_content
            chunk_tool_call_chunks = extract_tool_calls_from_stream_response(chunk)
            tool_call_chunks.extend(chunk_tool_call_chunks)
            reasoning_details_list.extend(extract_reasoning_details_from_stream(chunk))  # type: ignore[arg-type]
            last_response = chunk

            yield await emit_event(
                LLMChunkArriveEvent(
                    event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                    timestamp=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    func_name=func_name,
                    iteration=iteration,
                    chunk=chunk,
                    accumulated_content=accumulated_content,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            if chunk_tool_call_chunks:
                payloads = _collect_tool_argument_delta_payloads(
                    chunk_tool_call_chunks,
                    stream_tool_call_states,
                )
                for tool_name, tool_call_id, argname, argcontent_delta in payloads:
                    origin_overrides: Dict[str, Any] = {"tool_call_id": tool_call_id}
                    if tool_name:
                        origin_overrides["tool_name"] = tool_name
                    yield await emit_event(
                        ToolCallArgumentsDeltaEvent(
                            event_type=ReActEventType.TOOL_CALL_ARGUMENTS_DELTA,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=trace_id,
                            func_name=func_name,
                            iteration=iteration,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            argname=argname,
                            argcontent_delta=argcontent_delta,
                        ),
                        origin_overrides=origin_overrides,
                    )

            yield ResponseYield(
                type="response",
                response=chunk,
                messages=messages.copy(),
            )

            if _abort_requested():
                aborted = True
                await _close_stream(stream_response)
                break

        tool_calls = accumulate_tool_calls_from_chunks(tool_call_chunks)
        reasoning_details = reasoning_details_list
    else:
        if _abort_requested():
            aborted = True
        else:
            response = await llm_interface.chat(
                messages=cast(List[Dict[str, Any]], messages),
                tools=tools,
                **llm_kwargs_filtered,
            )
            content = extract_content_from_response(response, func_name)
            tool_calls = extract_tool_calls(response)
            reasoning_details = extract_reasoning_details(response)  # type: ignore[assignment]
            last_response = response

            yield ResponseYield(
                type="response",
                response=response,
                messages=messages.copy(),
            )

    usage_info = extract_usage_from_response(last_response)
    if usage_info is None:
        usage_info = _usage_from_context_delta(
            llm_input_tokens_before,
            llm_output_tokens_before,
        )

    if not aborted or last_response is not None or content:
        typed_tool_calls = (
            [dict_to_tool_call(tc) for tc in tool_calls] if tool_calls else []
        )
        yield await emit_event(
            LLMCallEndEvent(
                event_type=ReActEventType.LLM_CALL_END,
                timestamp=datetime.now(timezone.utc),
                trace_id=trace_id,
                func_name=func_name,
                iteration=iteration,
                response=last_response,
                messages=messages.copy(),
                tool_calls=typed_tool_calls,
                execution_time=time.time() - llm_call_start_time,
                content=content,
                reasoning_details=reasoning_details,
                usage=usage_info,
            )
        )


async def execute_single_llm_call(
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: ToolDefinitionList = None,
    *,
    stream: bool = True,
    trace_id: str = "",
    func_name: str = "single_llm_call",
    abort_signal: Optional[AbortSignal] = None,
    iteration: int = 0,
    **llm_kwargs: Any,
) -> AsyncGenerator[Any, None]:
    normalized_messages = _normalize_and_validate_messages(messages)
    current_trace_id = (
        trace_id or get_current_trace_id() or f"trace_{int(time.time() * 1000)}"
    )
    event_bus = EventBus(
        session_id=current_trace_id,
        agent_call_id=f"single_call_{current_trace_id}",
    )

    async def _emit_event(
        event: Any,
        *,
        origin_overrides: Optional[Dict[str, Any]] = None,
    ) -> EventYield:
        return await event_bus.emit_and_get(
            cast(Any, event),
            origin_overrides=origin_overrides,
        )

    async for output in _stream_single_llm_call_outputs(
        llm_interface=llm_interface,
        messages=normalized_messages,
        tools=tools,
        stream=stream,
        llm_kwargs=dict(llm_kwargs),
        func_name=func_name,
        trace_id=current_trace_id,
        iteration=iteration,
        emit_event=_emit_event,
        abort_signal=abort_signal,
    ):
        if isinstance(output, EventYield):
            yield output.event


async def execute_llm(
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: ToolDefinitionList,
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    max_tool_calls: Optional[int],
    stream: bool = False,
    enable_event: bool = False,
    trace_id: str = "",
    user_task_prompt: str = "",
    abort_signal: Optional[AbortSignal] = None,
    hooks: Any = None,
    **llm_kwargs,
) -> AsyncGenerator[Union[Tuple[Any, MessageList], ReactOutput], None]:
    """Execute LLM calls and orchestrate iterative tool usage.

    Implements the ReAct (Reasoning and Acting) pattern by:
    1. Making an initial LLM call (streaming or non-streaming)
    2. Extracting tool calls from the response
    3. Executing the requested tools via tool_map
    4. Feeding tool results back to the LLM
    5. Repeating steps 2-4 until no more tools are called or ``max_tool_calls`` is reached

    Args:
            llm_interface: The LLM service interface for making chat requests.
            messages: Initial message history to send to the LLM.
            tools: Optional list of tool definitions available to the LLM.
            tool_map: Mapping of tool names to their async callable implementations.
            max_tool_calls: Optional maximum number of tool-call iterations before
                forcing termination. ``None`` means no framework-imposed cap.
            stream: Whether to stream responses or return complete responses.
            **llm_kwargs: Additional keyword arguments to pass to the LLM interface.

    Yields:
            Tuple of (response, updated_messages) where:
            - response: Responses from the LLM (either complete responses or stream chunks)
            - updated_messages: Current message history including tool call results
    """

    func_name = get_current_context_attribute("function_name") or "Unknown Function"
    current_trace_id = (
        trace_id or get_current_trace_id() or f"trace_{int(time.time() * 1000)}"
    )

    current_messages = messages.copy()  # 创建副本以避免修改原始列表
    call_count = 0
    iteration = 0
    total_llm_calls = 0
    total_tool_calls = 0
    start_time = time.time()
    event_bus = EventBus(
        session_id=current_trace_id,
        agent_call_id=f"agent_{current_trace_id}",
    )

    async def _emit_event(
        event: Any,
        *,
        origin_overrides: Optional[Dict[str, Any]] = None,
    ) -> EventYield:
        return await event_bus.emit_and_get(
            cast(Any, event),
            origin_overrides=origin_overrides,
        )

    def _self_reference_has_pending_forks() -> bool:
        self_reference = getattr(hooks, "self_reference", None)
        if self_reference is None:
            return False

        instance = getattr(self_reference, "instance", None)
        has_pending = getattr(instance, "has_pending_fork_tasks", None)
        if not callable(has_pending):
            return False

        try:
            return bool(has_pending(event_bus))
        except Exception:
            return False

    async def _flush_event_bus_until_idle() -> AsyncGenerator[EventYield, None]:
        while True:
            emitted = False
            async for queued_output in _yield_pending_event_bus_events(event_bus):
                emitted = True
                yield queued_output

            if not _self_reference_has_pending_forks():
                break

            if not emitted:
                await asyncio.sleep(0.01)

    async def _emit_tool_argument_delta_events(
        tool_call_chunks_batch: List[Dict[str, Any]],
        stream_states: Dict[int, _StreamingToolCallState],
        *,
        current_iteration: int,
    ) -> List[EventYield]:
        if not enable_event:
            return []

        payloads = _collect_tool_argument_delta_payloads(
            tool_call_chunks_batch,
            stream_states,
        )
        emitted: List[EventYield] = []

        for tool_name, tool_call_id, argname, argcontent_delta in payloads:
            try:
                origin_overrides: Dict[str, Any] = {"tool_call_id": tool_call_id}
                if tool_name:
                    origin_overrides["tool_name"] = tool_name

                emitted.append(
                    await _emit_event(
                        ToolCallArgumentsDeltaEvent(
                            event_type=ReActEventType.TOOL_CALL_ARGUMENTS_DELTA,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=current_trace_id,
                            func_name=func_name,
                            iteration=current_iteration,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            argname=argname,
                            argcontent_delta=argcontent_delta,
                        ),
                        origin_overrides=origin_overrides,
                    )
                )
            except Exception:
                continue

        return emitted

    def _abort_requested() -> bool:
        return abort_signal is not None and abort_signal.is_aborted

    state = ReActState(
        trace_id=current_trace_id,
        func_name=func_name,
        user_task_prompt=user_task_prompt,
        messages=current_messages,
        llm_kwargs=dict(llm_kwargs),
        stream=stream,
    )

    def _abort_reason() -> str:
        return abort_signal.reason if abort_signal is not None else ""

    async def _close_stream(stream_obj: Any) -> None:
        close_method = getattr(stream_obj, "aclose", None)
        if callable(close_method):
            try:
                result = close_method()
                if inspect.isawaitable(result):
                    await cast(Awaitable[Any], result)
            except Exception:
                pass

    async def _next_stream_item(
        stream_iter: Any,
    ) -> Any:
        if abort_signal is None:
            return await stream_iter.__anext__()

        abort_task = asyncio.create_task(abort_signal.wait())
        next_task = asyncio.create_task(stream_iter.__anext__())
        done, pending = await asyncio.wait(
            {abort_task, next_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
            abort_task.cancel()
            raise asyncio.CancelledError
        abort_task.cancel()
        return next_task.result()

    def _set_abort_metadata(event: ReactEndEvent) -> None:
        event_extra = getattr(event, "extra", None)
        if not isinstance(event_extra, dict):
            event_extra = {}
            setattr(event, "extra", event_extra)
        event_extra["aborted"] = True
        abort_reason = _abort_reason()
        if abort_reason:
            event_extra["abort_reason"] = abort_reason

    async def _apply_before_finalize(
        final_messages: MessageList,
        final_response: str,
        *,
        current_iteration: int,
    ) -> tuple[MessageList, str]:
        state.iteration = current_iteration
        state.messages = final_messages
        state.final_response = final_response
        await run_react_hook(hooks, "before_finalize", state)
        return state.messages, state.final_response or ""

    def _reconcile_tool_batch_messages(
        base_messages_before_tools: MessageList,
        returned_messages_after_tools: MessageList,
    ) -> MessageList:
        active_messages = state.messages
        memory_key = getattr(hooks, "memory_key", None)
        self_reference = getattr(hooks, "self_reference", None)
        if isinstance(memory_key, str) and self_reference is not None:
            try:
                if self_reference.consume_destructive_history_mutation(memory_key):
                    base_len = len(base_messages_before_tools)
                    if len(returned_messages_after_tools) <= base_len:
                        return active_messages

                    tail_messages = cast(
                        MessageList,
                        returned_messages_after_tools[base_len:],
                    )
                    reconciled = _clone_message_list(active_messages)
                    reconciled.extend(_clone_message_list(tail_messages))
                    return reconciled
            except Exception:
                pass

        if active_messages is returned_messages_after_tools:
            return returned_messages_after_tools

        if active_messages == base_messages_before_tools:
            return returned_messages_after_tools

        base_len = len(base_messages_before_tools)
        if len(returned_messages_after_tools) < base_len:
            return active_messages

        tail_messages = cast(MessageList, returned_messages_after_tools[base_len:])
        reconciled = _clone_message_list(active_messages)
        reconciled.extend(_clone_message_list(tail_messages))
        return reconciled

    push_debug(
        f"LLM 函数 '{func_name}' 开始执行，消息数: {len(current_messages)}",
        location=get_location(),
    )

    await run_react_hook(hooks, "on_run_start", state)
    current_messages = state.messages

    # 发射 ReAct 开始事件
    if enable_event:
        try:
            yield await _emit_event(
                ReactStartEvent(
                    event_type=ReActEventType.REACT_START,
                    timestamp=datetime.now(timezone.utc),
                    trace_id=current_trace_id,
                    func_name=func_name,
                    iteration=0,
                    user_task_prompt=user_task_prompt,
                    initial_messages=current_messages.copy(),
                    available_tools=tools,
                )
            )
        except Exception:
            # 事件发射失败不应影响主流程
            pass

    # Phase 1: Initial LLM call
    model_parameters = {k: v for k, v in llm_kwargs.items() if k not in ["retry_times"]}
    model_name = llm_interface.model_name

    content = ""
    tool_calls: List[Dict[str, Any]] = []
    reasoning_details: List[Dict[str, Any]] = []
    last_response: Any = None
    usage_info: Optional[CompletionUsage] = None
    aborted = False

    total_llm_calls += 1
    state.total_llm_calls = total_llm_calls
    state.iteration = iteration
    state.messages = current_messages
    await run_react_hook(hooks, "before_llm_call", state)
    current_messages = state.messages

    trace_context = get_langfuse_trace_context()

    with langfuse_client.start_as_current_observation(
        as_type="generation",
        name=f"{func_name}_initial_llm_call",
        input=current_messages,
        model=model_name,
        model_parameters=model_parameters,
        metadata=coerce_langfuse_metadata(
            {
                "stream": stream,
                "tools_available": len(tools) if tools else 0,
            }
        ),
        completion_start_time=datetime.now(timezone.utc),
        trace_context=trace_context,
    ) as generation_span:
        initial_end_event: Optional[LLMCallEndEvent] = None

        async for output in _stream_single_llm_call_outputs(
            llm_interface=llm_interface,
            messages=current_messages,
            tools=tools,
            stream=stream,
            llm_kwargs=dict(llm_kwargs),
            func_name=func_name,
            trace_id=current_trace_id,
            iteration=iteration,
            emit_event=_emit_event,
            abort_signal=abort_signal,
        ):
            if isinstance(output, EventYield) and isinstance(
                output.event, LLMCallEndEvent
            ):
                initial_end_event = output.event
                continue

            if enable_event:
                yield output
            elif isinstance(output, ResponseYield):
                yield output.response, cast(MessageList, output.messages.copy())

        aborted = _abort_requested()

        if initial_end_event is not None:
            content = initial_end_event.content
            tool_calls = [
                tool_call_to_dict(tool_call)
                for tool_call in initial_end_event.tool_calls
            ]
            reasoning_details = list(initial_end_event.reasoning_details)
            last_response = initial_end_event.response
            usage_info = initial_end_event.usage

        state.messages = current_messages
        state.last_response = last_response
        state.content = content
        state.tool_calls = list(tool_calls)
        state.reasoning_details = list(reasoning_details)
        state.usage = usage_info
        await run_react_hook(hooks, "after_llm_call", state)
        current_messages = state.messages
        last_response = state.last_response
        content = state.content
        tool_calls = list(state.tool_calls)
        reasoning_details = list(state.reasoning_details)
        usage_info = state.usage

        if enable_event and initial_end_event is not None:
            try:
                tool_calls_typed_initial: List[ToolCall] = (
                    [dict_to_tool_call(tc) for tc in tool_calls] if tool_calls else []
                )
                yield await _emit_event(
                    LLMCallEndEvent(
                        event_type=ReActEventType.LLM_CALL_END,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=iteration,
                        response=last_response,
                        messages=current_messages.copy(),
                        tool_calls=tool_calls_typed_initial,
                        execution_time=initial_end_event.execution_time,
                        content=content,
                        reasoning_details=reasoning_details,
                        usage=usage_info,
                    )
                )
            except Exception:
                pass

        usage_dict_initial: Optional[Dict[str, int]] = None
        if usage_info:
            usage_dict_initial = {
                "prompt_tokens": usage_info.prompt_tokens,
                "completion_tokens": usage_info.completion_tokens,
                "total_tokens": usage_info.total_tokens,
            }
        generation_span.update(
            output={"content": content, "tool_calls": tool_calls},
            usage_details=usage_dict_initial,
        )

        push_debug(
            f"LLM 函数 '{func_name}' 初始响应已获取，工具调用数: {len(tool_calls)}",
            location=get_location(),
        )

        if content.strip() != "":
            assistant_message = build_assistant_response_message(content)
            current_messages.append(cast(Any, assistant_message))

        if aborted:
            total_execution_time = time.time() - start_time
            if enable_event:
                try:
                    current_messages, final_content = await _apply_before_finalize(
                        current_messages,
                        content,
                        current_iteration=0,
                    )
                    react_end = ReactEndEvent(
                        event_type=ReActEventType.REACT_END,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=0,
                        final_response=final_content,
                        final_messages=current_messages.copy(),
                        total_iterations=0,
                        total_execution_time=total_execution_time,
                        total_tool_calls=0,
                        total_llm_calls=total_llm_calls,
                        total_token_usage=usage_info,
                    )
                    _set_abort_metadata(react_end)
                    yield await _emit_event(react_end)
                    async for queued_output in _flush_event_bus_until_idle():
                        yield queued_output
                except Exception:
                    pass

            app_log(
                f"LLM 函数 '{func_name}' aborted",
                location=get_location(),
            )
            return

        if len(tool_calls) != 0:
            assistant_tool_call_message = build_assistant_tool_message(
                tool_calls, reasoning_details if reasoning_details else None
            )
            current_messages.append(cast(Any, assistant_tool_call_message))
            state.messages = current_messages
        else:
            total_execution_time = time.time() - start_time
            if enable_event:
                try:
                    current_messages, final_content = await _apply_before_finalize(
                        current_messages,
                        content,
                        current_iteration=0,
                    )
                    yield await _emit_event(
                        ReactEndEvent(
                            event_type=ReActEventType.REACT_END,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=current_trace_id,
                            func_name=func_name,
                            iteration=0,
                            final_response=final_content,
                            final_messages=current_messages.copy(),
                            total_iterations=0,
                            total_execution_time=total_execution_time,
                            total_tool_calls=0,
                            total_llm_calls=total_llm_calls,
                            total_token_usage=usage_info,
                        )
                    )
                    async for queued_output in _flush_event_bus_until_idle():
                        yield queued_output
                except Exception:
                    pass

            app_log(
                f"LLM 函数 '{func_name}' 完成执行",
                location=get_location(),
            )
            return

    # Phase 2: Tool calling loop
    push_debug(
        f"LLM 函数 '{func_name}' 开始执行 {len(tool_calls)} 个工具调用",
        location=get_location(),
    )

    call_count += 1
    iteration = 1
    total_tool_calls += len(tool_calls)
    state.total_tool_calls = total_tool_calls
    tool_batch_base_messages = _clone_message_list(current_messages)

    # 使用支持事件的工具调用处理函数
    state.messages = current_messages
    await run_react_hook(hooks, "before_tool_batch", state)
    current_messages = state.messages
    if enable_event:
        # 使用异步生成器实时发射事件
        async for item in _process_tool_calls_with_events_gen(
            tool_calls=tool_calls,
            messages=current_messages,
            tool_map=tool_map,
            enable_event=enable_event,
            event_bus=event_bus,
            trace_id=current_trace_id,
            func_name=func_name,
            iteration=iteration,
            abort_signal=abort_signal,
        ):
            if isinstance(item, EventYield):
                yield item
            else:
                # 最后一个 yield 是 MessageList
                current_messages = item
    else:
        if abort_signal is None:
            result_messages_iteration = await process_tool_calls(
                tool_calls=tool_calls,
                messages=cast(List[Dict[str, Any]], current_messages),
                tool_map=tool_map,
                event_emitter=NoOpEventEmitter(),
            )
            current_messages = cast(MessageList, result_messages_iteration)
        else:
            tool_task = asyncio.create_task(
                process_tool_calls(
                    tool_calls=tool_calls,
                    messages=cast(List[Dict[str, Any]], current_messages),
                    tool_map=tool_map,
                    event_emitter=NoOpEventEmitter(),
                    abort_signal=abort_signal,
                )
            )
            abort_task = asyncio.create_task(abort_signal.wait())
            done, _ = await asyncio.wait(
                {tool_task, abort_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if abort_task in done:
                tool_task.cancel()
                await asyncio.gather(tool_task, return_exceptions=True)
            else:
                abort_task.cancel()
                result_messages_iteration = tool_task.result()
                current_messages = cast(MessageList, result_messages_iteration)

    current_messages = _reconcile_tool_batch_messages(
        tool_batch_base_messages,
        current_messages,
    )

    state.messages = current_messages
    await run_react_hook(hooks, "after_tool_batch", state)
    current_messages = state.messages

    if _abort_requested():
        total_execution_time = time.time() - start_time
        if enable_event:
            try:
                usage_info = extract_usage_from_response(last_response)
                if usage_info is None:
                    usage_info = _usage_from_context_delta(
                        llm_input_tokens_before,
                        llm_output_tokens_before,
                    )
                final_content = content
                current_messages, final_content = await _apply_before_finalize(
                    current_messages,
                    final_content,
                    current_iteration=iteration,
                )
                react_end = ReactEndEvent(
                    event_type=ReActEventType.REACT_END,
                    timestamp=datetime.now(timezone.utc),
                    trace_id=current_trace_id,
                    func_name=func_name,
                    iteration=iteration,
                    final_response=final_content,
                    final_messages=current_messages.copy(),
                    total_iterations=iteration,
                    total_execution_time=total_execution_time,
                    total_tool_calls=total_tool_calls,
                    total_llm_calls=total_llm_calls,
                    total_token_usage=usage_info,
                )
                _set_abort_metadata(react_end)
                yield await _emit_event(react_end)
                async for queued_output in _flush_event_bus_until_idle():
                    yield queued_output
            except Exception:
                pass
        return

    max_tool_calls_reached = False

    while True:
        if max_tool_calls is not None and call_count >= max_tool_calls:
            max_tool_calls_reached = True
            break

        # Phase 3: Iterative LLM-tool interaction
        iteration = call_count + 1

        # 发射迭代开始事件
        if enable_event:
            try:
                yield await _emit_event(
                    ReactIterationStartEvent(
                        event_type=ReActEventType.REACT_ITERATION_START,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=iteration,
                        current_messages=current_messages.copy(),
                    )
                )
            except Exception:
                pass

        push_debug(
            f"LLM 函数 '{func_name}' 工具调用循环 (次数: {call_count})",
            location=get_location(),
        )

        iteration_llm_start_time = time.time()
        total_llm_calls += 1
        state.total_llm_calls = total_llm_calls
        aborted = False
        state.iteration = iteration
        state.messages = current_messages
        await run_react_hook(hooks, "before_llm_call", state)
        current_messages = state.messages

        # 为迭代调用创建新的观测
        with langfuse_client.start_as_current_observation(
            as_type="generation",
            name=f"{func_name}_iteration_{call_count}_llm_call",
            input=current_messages,
            model=model_name,
            model_parameters=model_parameters,
            metadata=coerce_langfuse_metadata(
                {
                    "stream": stream,
                    "iteration": call_count,
                    "tools_available": len(tools) if tools else 0,
                }
            ),
            completion_start_time=datetime.now(timezone.utc),
            trace_context=trace_context,
        ) as iteration_generation_span:
            iteration_end_event: Optional[LLMCallEndEvent] = None

            async for output in _stream_single_llm_call_outputs(
                llm_interface=llm_interface,
                messages=current_messages,
                tools=tools,
                stream=stream,
                llm_kwargs=dict(llm_kwargs),
                func_name=func_name,
                trace_id=current_trace_id,
                iteration=iteration,
                emit_event=_emit_event,
                abort_signal=abort_signal,
            ):
                if isinstance(output, EventYield) and isinstance(
                    output.event, LLMCallEndEvent
                ):
                    iteration_end_event = output.event
                    continue

                if enable_event:
                    yield output
                elif isinstance(output, ResponseYield):
                    yield output.response, cast(MessageList, output.messages.copy())

            aborted = _abort_requested()

            if iteration_end_event is not None:
                content = iteration_end_event.content
                tool_calls = [
                    tool_call_to_dict(tool_call)
                    for tool_call in iteration_end_event.tool_calls
                ]
                reasoning_details = list(iteration_end_event.reasoning_details)
                last_response = iteration_end_event.response
                usage_info = iteration_end_event.usage
            else:
                content = ""
                tool_calls = []
                reasoning_details = []
                last_response = None
                usage_info = None

            state.messages = current_messages
            state.last_response = last_response
            state.content = content
            state.tool_calls = list(tool_calls)
            state.reasoning_details = list(reasoning_details)
            state.usage = usage_info
            await run_react_hook(hooks, "after_llm_call", state)
            current_messages = state.messages
            last_response = state.last_response
            content = state.content
            tool_calls = list(state.tool_calls)
            reasoning_details = list(state.reasoning_details)
            usage_info = state.usage
            if enable_event and iteration_end_event is not None:
                try:
                    tool_calls_typed_iteration: List[ToolCall] = (
                        [dict_to_tool_call(tc) for tc in tool_calls]
                        if tool_calls
                        else []
                    )
                    yield await _emit_event(
                        LLMCallEndEvent(
                            event_type=ReActEventType.LLM_CALL_END,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=current_trace_id,
                            func_name=func_name,
                            iteration=iteration,
                            response=last_response,
                            messages=current_messages.copy(),
                            tool_calls=tool_calls_typed_iteration,
                            execution_time=iteration_end_event.execution_time,
                            content=content,
                            reasoning_details=reasoning_details,
                            usage=usage_info,
                        )
                    )
                except Exception:
                    pass

            # 更新迭代生成观测数据
            usage_dict_iteration: Optional[Dict[str, int]] = None
            if usage_info:
                usage_dict_iteration = {
                    "prompt_tokens": usage_info.prompt_tokens,
                    "completion_tokens": usage_info.completion_tokens,
                    "total_tokens": usage_info.total_tokens,
                }
            iteration_generation_span.update(
                output={"content": content, "tool_calls": tool_calls},
                usage_details=usage_dict_iteration,
            )

        # Append new assistant response to message history
        if content.strip() != "":
            assistant_message = build_assistant_response_message(content)
            current_messages.append(cast(Any, assistant_message))

        if aborted or _abort_requested():
            total_execution_time = time.time() - start_time
            if enable_event:
                try:
                    final_content = (
                        content
                        if stream
                        else (
                            extract_content_from_response(last_response, func_name)
                            if last_response
                            else content
                        )
                    )
                    current_messages, final_content = await _apply_before_finalize(
                        current_messages,
                        final_content,
                        current_iteration=iteration,
                    )
                    react_end = ReactEndEvent(
                        event_type=ReActEventType.REACT_END,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=iteration,
                        final_response=final_content,
                        final_messages=current_messages.copy(),
                        total_iterations=iteration,
                        total_execution_time=total_execution_time,
                        total_tool_calls=total_tool_calls,
                        total_llm_calls=total_llm_calls,
                        total_token_usage=usage_info,
                    )
                    _set_abort_metadata(react_end)
                    yield await _emit_event(react_end)
                    async for queued_output in _flush_event_bus_until_idle():
                        yield queued_output
                except Exception:
                    pass
            return

        if len(tool_calls) != 0:
            assistant_tool_call_message = build_assistant_tool_message(
                tool_calls, reasoning_details if reasoning_details else None
            )
            current_messages.append(cast(Any, assistant_tool_call_message))
            state.messages = current_messages

        if len(tool_calls) == 0:
            # No more tool calls, exit loop
            push_debug(
                f"LLM 函数 '{func_name}' 无更多工具调用，返回最终结果",
                location=get_location(),
            )

            # 发射迭代结束事件
            iteration_time = time.time() - iteration_llm_start_time
            if enable_event:
                try:
                    yield await _emit_event(
                        ReactIterationEndEvent(
                            event_type=ReActEventType.REACT_ITERATION_END,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=current_trace_id,
                            func_name=func_name,
                            iteration=iteration,
                            messages=current_messages.copy(),
                            iteration_time=iteration_time,
                            tool_calls_count=0,
                        )
                    )
                except Exception:
                    pass

            # 发射 ReAct 结束事件
            total_execution_time = time.time() - start_time
            if enable_event:
                try:
                    if stream:
                        final_content = content
                    else:
                        final_content = (
                            extract_content_from_response(last_response, func_name)
                            if last_response
                            else ""
                        )
                    current_messages, final_content = await _apply_before_finalize(
                        current_messages,
                        final_content,
                        current_iteration=iteration,
                    )
                    yield await _emit_event(
                        ReactEndEvent(
                            event_type=ReActEventType.REACT_END,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=current_trace_id,
                            func_name=func_name,
                            iteration=iteration,
                            final_response=final_content,
                            final_messages=current_messages.copy(),
                            total_iterations=iteration,
                            total_execution_time=total_execution_time,
                            total_tool_calls=total_tool_calls,
                            total_llm_calls=total_llm_calls,
                            total_token_usage=usage_info,
                        )
                    )
                    async for queued_output in _flush_event_bus_until_idle():
                        yield queued_output
                except Exception:
                    pass

            app_log(
                f"LLM 函数 '{func_name}' 完成执行",
                location=get_location(),
            )
            # 注意：响应已经在上面 yield 过了，这里直接 return
            return

        # Continue with next iteration of tool calls
        push_debug(
            f"LLM 函数 '{func_name}' 发现 {len(tool_calls)} 个工具调用",
            location=get_location(),
        )

        total_tool_calls += len(tool_calls)
        state.total_tool_calls = total_tool_calls
        tool_batch_base_messages = _clone_message_list(current_messages)

        # 使用支持事件的工具调用处理函数
        state.messages = current_messages
        await run_react_hook(hooks, "before_tool_batch", state)
        current_messages = state.messages
        if enable_event:
            # 使用异步生成器实时发射事件
            async for item in _process_tool_calls_with_events_gen(
                tool_calls=tool_calls,
                messages=current_messages,
                tool_map=tool_map,
                enable_event=enable_event,
                event_bus=event_bus,
                trace_id=current_trace_id,
                func_name=func_name,
                iteration=iteration,
                abort_signal=abort_signal,
            ):
                if isinstance(item, EventYield):
                    yield item
                else:
                    # 最后一个 yield 是 MessageList
                    current_messages = item
        else:
            if abort_signal is None:
                result_messages = await process_tool_calls(
                    tool_calls=tool_calls,
                    messages=cast(List[Dict[str, Any]], current_messages),
                    tool_map=tool_map,
                    event_emitter=NoOpEventEmitter(),
                )
                current_messages = cast(MessageList, result_messages)
            else:
                tool_task = asyncio.create_task(
                    process_tool_calls(
                        tool_calls=tool_calls,
                        messages=cast(List[Dict[str, Any]], current_messages),
                        tool_map=tool_map,
                        event_emitter=NoOpEventEmitter(),
                        abort_signal=abort_signal,
                    )
                )
                abort_task = asyncio.create_task(abort_signal.wait())
                done, _ = await asyncio.wait(
                    {tool_task, abort_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort_task in done:
                    tool_task.cancel()
                    await asyncio.gather(tool_task, return_exceptions=True)
                else:
                    abort_task.cancel()
                    result_messages = tool_task.result()
                    current_messages = cast(MessageList, result_messages)

        current_messages = _reconcile_tool_batch_messages(
            tool_batch_base_messages,
            current_messages,
        )

        state.messages = current_messages
        await run_react_hook(hooks, "after_tool_batch", state)
        current_messages = state.messages

        if _abort_requested():
            total_execution_time = time.time() - start_time
            if enable_event:
                try:
                    usage_info = extract_usage_from_response(last_response)
                    if usage_info is None:
                        usage_info = _usage_from_context_delta(
                            iteration_input_tokens_before,
                            iteration_output_tokens_before,
                        )
                    final_content = content
                    react_end = ReactEndEvent(
                        event_type=ReActEventType.REACT_END,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=iteration,
                        final_response=final_content,
                        final_messages=current_messages.copy(),
                        total_iterations=iteration,
                        total_execution_time=total_execution_time,
                        total_tool_calls=total_tool_calls,
                        total_llm_calls=total_llm_calls,
                        total_token_usage=usage_info,
                    )
                    _set_abort_metadata(react_end)
                    yield await _emit_event(react_end)
                    async for queued_output in _flush_event_bus_until_idle():
                        yield queued_output
                except Exception:
                    pass
            return

        # 发射迭代结束事件
        iteration_time = time.time() - iteration_llm_start_time
        if enable_event:
            try:
                yield await _emit_event(
                    ReactIterationEndEvent(
                        event_type=ReActEventType.REACT_ITERATION_END,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=iteration,
                        messages=current_messages.copy(),
                        iteration_time=iteration_time,
                        tool_calls_count=len(tool_calls),
                    )
                )
            except Exception:
                pass

        call_count += 1

    if not max_tool_calls_reached:
        return

    # Phase 4: Handle max_tool_calls limit reached
    push_debug(
        f"LLM 函数 '{func_name}' 达到最大工具调用次数限制 ({max_tool_calls})",
        location=get_location(),
    )

    if _abort_requested():
        total_execution_time = time.time() - start_time
        if enable_event:
            try:
                usage_info = extract_usage_from_response(last_response)
                if usage_info is None:
                    usage_info = _usage_from_context_delta(
                        llm_input_tokens_before,
                        llm_output_tokens_before,
                    )
                current_messages, content = await _apply_before_finalize(
                    current_messages,
                    content,
                    current_iteration=call_count,
                )
                react_end = ReactEndEvent(
                    event_type=ReActEventType.REACT_END,
                    timestamp=datetime.now(timezone.utc),
                    trace_id=current_trace_id,
                    func_name=func_name,
                    iteration=call_count,
                    final_response=content,
                    final_messages=current_messages.copy(),
                    total_iterations=call_count,
                    total_execution_time=total_execution_time,
                    total_tool_calls=total_tool_calls,
                    total_llm_calls=total_llm_calls,
                    total_token_usage=usage_info,
                )
                _set_abort_metadata(react_end)
                yield await _emit_event(react_end)
            except Exception:
                pass
        return

    total_llm_calls += 1
    state.total_llm_calls = total_llm_calls
    state.iteration = call_count + 1
    state.messages = current_messages
    await run_react_hook(hooks, "before_llm_call", state)
    current_messages = state.messages

    # 为最终调用创建观测
    with langfuse_client.start_as_current_observation(
        as_type="generation",
        name=f"{func_name}_final_llm_call",
        input=current_messages,
        model=model_name,
        model_parameters=model_parameters,
        metadata=coerce_langfuse_metadata(
            {
                "stream": False,
                "reason": "max_tool_calls_reached",
                "call_count": call_count,
            }
        ),
        completion_start_time=datetime.now(timezone.utc),
        trace_context=trace_context,
    ) as final_generation_span:
        final_end_event: Optional[LLMCallEndEvent] = None

        async for output in _stream_single_llm_call_outputs(
            llm_interface=llm_interface,
            messages=current_messages,
            tools=None,
            stream=False,
            llm_kwargs=dict(llm_kwargs),
            func_name=func_name,
            trace_id=current_trace_id,
            iteration=call_count + 1,
            emit_event=_emit_event,
            abort_signal=abort_signal,
        ):
            if isinstance(output, EventYield) and isinstance(
                output.event, LLMCallEndEvent
            ):
                final_end_event = output.event
                continue

            if enable_event:
                yield output
            elif isinstance(output, ResponseYield):
                yield output.response, cast(MessageList, output.messages.copy())

        final_response = (
            final_end_event.response if final_end_event is not None else None
        )
        final_content = final_end_event.content if final_end_event is not None else ""
        usage_info = final_end_event.usage if final_end_event is not None else None
        state.iteration = call_count + 1
        state.messages = current_messages
        state.last_response = final_response
        state.content = final_content
        state.tool_calls = []
        state.reasoning_details = []
        state.usage = usage_info
        await run_react_hook(hooks, "after_llm_call", state)
        current_messages = state.messages
        final_response = state.last_response
        final_content = state.content
        usage_info = state.usage

        if enable_event and final_end_event is not None:
            try:
                yield await _emit_event(
                    LLMCallEndEvent(
                        event_type=ReActEventType.LLM_CALL_END,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=call_count + 1,
                        response=final_response,
                        messages=current_messages.copy(),
                        tool_calls=[],
                        execution_time=final_end_event.execution_time,
                        content=final_content,
                        reasoning_details=[],
                        usage=usage_info,
                    )
                )
            except Exception:
                pass

        # 更新最终观测数据
        usage_dict_final: Optional[Dict[str, int]] = None
        if usage_info:
            usage_dict_final = {
                "prompt_tokens": usage_info.prompt_tokens,
                "completion_tokens": usage_info.completion_tokens,
                "total_tokens": usage_info.total_tokens,
            }
        final_generation_span.update(
            output={"content": final_content, "tool_calls": []},
            usage_details=usage_dict_final,
        )

        # 发射响应
        if enable_event:
            try:
                yield ResponseYield(
                    type="response",
                    response=final_response,
                    messages=current_messages.copy(),
                )
            except Exception:
                pass
        else:
            yield final_response, cast(MessageList, current_messages.copy())

        # 发射 ReAct 结束事件
        total_execution_time = time.time() - start_time
        current_messages, final_content = await _apply_before_finalize(
            current_messages,
            final_content,
            current_iteration=call_count + 1,
        )
        if enable_event:
            try:
                yield await _emit_event(
                    ReactEndEvent(
                        event_type=ReActEventType.REACT_END,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=call_count + 1,
                        final_response=final_content,
                        final_messages=current_messages.copy(),
                        total_iterations=call_count + 1,
                        total_execution_time=total_execution_time,
                        total_tool_calls=total_tool_calls,
                        total_llm_calls=total_llm_calls,
                        total_token_usage=usage_info,
                    )
                )
                async for queued_output in _flush_event_bus_until_idle():
                    yield queued_output
            except Exception:
                pass

        app_log(
            f"LLM 函数 '{func_name}' 完成执行",
            location=get_location(),
        )


__all__ = ["execute_llm", "execute_single_llm_call"]
