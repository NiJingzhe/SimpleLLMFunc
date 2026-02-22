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
from datetime import datetime, timezone
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
    ToolCallEndEvent,
    ToolCallErrorEvent,
    ToolCallsBatchEndEvent,
    ToolCallResult,
    ReactIterationEndEvent,
    ReactEndEvent,
)
from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter, NoOpEventEmitter
from SimpleLLMFunc.type.tool_call import dict_to_tool_call
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
    extract_tool_calls,
    extract_tool_calls_from_stream_response,
    process_tool_calls,
)

from SimpleLLMFunc.observability.langfuse_client import langfuse_client


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


async def _process_tool_calls_with_events_gen(
    tool_calls: List[Dict[str, Any]],
    messages: MessageList,
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    enable_event: bool,
    trace_id: str,
    func_name: str,
    iteration: int,
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
    if not tool_calls:
        yield messages
        return

    # 准备执行环境
    from SimpleLLMFunc.base.tool_call.execution import _execute_single_tool_call
    import asyncio

    # 发射工具调用批次开始事件
    if enable_event:
        try:
            tool_calls_typed: List[ToolCall] = [
                dict_to_tool_call(tc) for tc in tool_calls
            ]
            yield EventYield(
                event=ToolCallsBatchStartEvent(
                    event_type=ReActEventType.TOOL_CALLS_BATCH_START,
                    timestamp=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    func_name=func_name,
                    iteration=iteration,
                    tool_calls=tool_calls_typed,
                    batch_size=len(tool_calls),
                )
            )
        except Exception:
            pass

    event_queue: asyncio.Queue[Optional[EventYield]] = asyncio.Queue()

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
                arguments_start: ToolCallArguments = json.loads(arguments_str)
                tool_call_typed_start = dict_to_tool_call(tool_call)
                event_queue.put_nowait(
                    EventYield(
                        event=ToolCallStartEvent(
                            event_type=ReActEventType.TOOL_CALL_START,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=trace_id,
                            func_name=func_name,
                            iteration=iteration,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            arguments=arguments_start,
                            tool_call=tool_call_typed_start,
                        )
                    )
                )
            except Exception:
                pass

        # 执行工具调用
        tool_start_time = time.time()
        tool_result: Optional[ToolResult] = None
        tool_error: Optional[Exception] = None

        # 为每个 tool call 创建独立 emitter，携带 tool 上下文用于 UI 定位
        tool_event_emitter = ToolEventEmitter(
            _queue=cast(Any, event_queue),
            _trace_id=trace_id,
            _func_name=func_name,
            _iteration=iteration,
            _tool_name=tool_name,
            _tool_call_id=tool_call_id,
        )

        try:
            (
                tool_call_dict,
                messages_to_append,
                is_multimodal,
            ) = await _execute_single_tool_call(
                tool_call, tool_map, event_emitter=tool_event_emitter
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
                arguments_end: ToolCallArguments = json.loads(arguments_str)
                if tool_error:
                    event_queue.put_nowait(
                        EventYield(
                            event=ToolCallErrorEvent(
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
                            )
                        )
                    )
                else:
                    event_queue.put_nowait(
                        EventYield(
                            event=ToolCallEndEvent(
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
                            )
                        )
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
    batch_start_time = time.time()

    # 创建一个任务来监控所有工具任务的完成，并发送终止信号
    async def monitor_completion():
        await asyncio.gather(*tasks, return_exceptions=True)
        await event_queue.put(None)

    # 启动监控任务（不等待它）
    monitor_task = asyncio.create_task(monitor_completion())

    # 实时消费事件流（与工具执行并发）
    # 当工具执行时，事件会被放入队列，我们立即 yield 出去
    while True:
        event = await event_queue.get()
        if event is None:
            break
        yield event

    # 确保监控任务完成
    await monitor_task

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
            yield EventYield(
                event=ToolCallsBatchEndEvent(
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
            )
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


async def execute_llm(
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: ToolDefinitionList,
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    max_tool_calls: int,
    stream: bool = False,
    enable_event: bool = False,
    trace_id: str = "",
    user_task_prompt: str = "",
    **llm_kwargs,
) -> AsyncGenerator[Union[Tuple[Any, MessageList], ReactOutput], None]:
    """Execute LLM calls and orchestrate iterative tool usage.

    Implements the ReAct (Reasoning and Acting) pattern by:
    1. Making an initial LLM call (streaming or non-streaming)
    2. Extracting tool calls from the response
    3. Executing the requested tools via tool_map
    4. Feeding tool results back to the LLM
    5. Repeating steps 2-4 until no more tools are called or max_tool_calls is reached

    Args:
            llm_interface: The LLM service interface for making chat requests.
            messages: Initial message history to send to the LLM.
            tools: Optional list of tool definitions available to the LLM.
            tool_map: Mapping of tool names to their async callable implementations.
            max_tool_calls: Maximum number of tool call iterations before forcing termination.
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

    push_debug(
        f"LLM 函数 '{func_name}' 开始执行，消息数: {len(current_messages)}",
        location=get_location(),
    )

    # 发射 ReAct 开始事件
    if enable_event:
        try:
            yield EventYield(
                event=ReactStartEvent(
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
    # 准备 Langfuse 观测数据
    model_parameters = {k: v for k, v in llm_kwargs.items() if k not in ["retry_times"]}
    model_name = llm_interface.model_name

    # 声明变量
    content = ""
    tool_calls: List[Dict[str, Any]] = []
    tool_call_chunks: List[Dict[str, Any]] = []
    reasoning_details: List[Dict[str, Any]] = []
    last_response: Any = None

    # 发射 LLM 调用开始事件
    llm_call_start_time = time.time()
    if enable_event:
        try:
            yield EventYield(
                event=LLMCallStartEvent(
                    event_type=ReActEventType.LLM_CALL_START,
                    timestamp=datetime.now(timezone.utc),
                    trace_id=current_trace_id,
                    func_name=func_name,
                    iteration=iteration,
                    messages=current_messages.copy(),
                    tools=tools,
                    llm_kwargs=llm_kwargs,
                    stream=stream,
                )
            )
        except Exception:
            pass

    total_llm_calls += 1

    # 如果没有tools，移除tool_choice参数（如果存在）
    llm_kwargs_filtered = llm_kwargs.copy()
    if not tools:
        # 如果没有传递任何 tool，不应该设置 tool_choice
        llm_kwargs_filtered.pop("tool_choice", None)

    llm_input_tokens_before, llm_output_tokens_before = _read_context_token_counters()

    with langfuse_client.start_as_current_observation(
        as_type="generation",
        name=f"{func_name}_initial_llm_call",
        input=current_messages,
        model=model_name,
        model_parameters=model_parameters,
        metadata={"stream": stream, "tools_available": len(tools) if tools else 0},
        completion_start_time=datetime.now(timezone.utc),
    ) as generation_span:
        if stream:
            # Handle streaming response
            reasoning_details_list: List[Dict[str, Any]] = []
            chunk_index = 0
            accumulated_content = ""
            async for chunk in llm_interface.chat_stream(
                messages=cast(List[Dict[str, Any]], current_messages),
                tools=tools,
                **llm_kwargs_filtered,
            ):
                chunk_content = extract_content_from_stream_response(chunk, func_name)
                content += chunk_content
                accumulated_content += chunk_content
                tool_call_chunks.extend(extract_tool_calls_from_stream_response(chunk))
                reasoning_details_list.extend(
                    extract_reasoning_details_from_stream(chunk)
                )  # type: ignore
                last_response = chunk

                # 发射 chunk 事件
                if enable_event:
                    try:
                        yield EventYield(
                            event=LLMChunkArriveEvent(
                                event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                                timestamp=datetime.now(timezone.utc),
                                trace_id=current_trace_id,
                                func_name=func_name,
                                iteration=iteration,
                                chunk=chunk,
                                accumulated_content=accumulated_content,
                                chunk_index=chunk_index,
                            )
                        )
                    except Exception:
                        pass
                    chunk_index += 1

                # 发射响应
                if enable_event:
                    try:
                        yield ResponseYield(
                            type="response",
                            response=chunk,
                            messages=current_messages.copy(),
                        )
                    except Exception:
                        pass
                else:
                    yield chunk, cast(MessageList, current_messages.copy())

            tool_calls = accumulate_tool_calls_from_chunks(tool_call_chunks)
            reasoning_details = reasoning_details_list
        else:
            # Handle non-streaming response
            initial_response = await llm_interface.chat(
                messages=cast(List[Dict[str, Any]], current_messages),
                tools=tools,
                **llm_kwargs_filtered,
            )

            content = extract_content_from_response(initial_response, func_name)
            tool_calls = extract_tool_calls(initial_response)
            reasoning_details = extract_reasoning_details(initial_response)  # type: ignore
            last_response = initial_response

            # 发射响应
            if enable_event:
                try:
                    yield ResponseYield(
                        type="response",
                        response=initial_response,
                        messages=current_messages.copy(),
                    )
                except Exception:
                    pass
            else:
                yield initial_response, cast(MessageList, current_messages.copy())

        # 发射 LLM 调用结束事件
        llm_call_execution_time = time.time() - llm_call_start_time
        if enable_event:
            try:
                usage_info = extract_usage_from_response(last_response)
                if usage_info is None:
                    usage_info = _usage_from_context_delta(
                        llm_input_tokens_before,
                        llm_output_tokens_before,
                    )
                tool_calls_typed_initial: List[ToolCall] = (
                    [dict_to_tool_call(tc) for tc in tool_calls] if tool_calls else []
                )

                yield EventYield(
                    event=LLMCallEndEvent(
                        event_type=ReActEventType.LLM_CALL_END,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=iteration,
                        response=last_response,
                        messages=current_messages.copy(),
                        tool_calls=tool_calls_typed_initial,
                        usage=usage_info,
                        execution_time=llm_call_execution_time,
                    )
                )
            except Exception:
                pass

        push_debug(
            f"LLM 函数 '{func_name}' 初始响应已获取，工具调用数: {len(tool_calls)}",
            location=get_location(),
        )

        # Append assistant response to message history
        if content.strip() != "":
            assistant_message = build_assistant_response_message(content)
            current_messages.append(cast(Any, assistant_message))

        if len(tool_calls) != 0:
            assistant_tool_call_message = build_assistant_tool_message(
                tool_calls, reasoning_details if reasoning_details else None
            )
            current_messages.append(cast(Any, assistant_tool_call_message))
        else:
            # No tool calls, return final result
            # 发射 ReAct 结束事件（无工具调用的情况）
            total_execution_time = time.time() - start_time
            if enable_event:
                try:
                    usage_info = extract_usage_from_response(last_response)
                    if usage_info is None:
                        usage_info = _usage_from_context_delta(
                            llm_input_tokens_before,
                            llm_output_tokens_before,
                        )
                    final_content = (
                        extract_content_from_response(last_response, func_name)
                        if last_response
                        else content
                    )
                    yield EventYield(
                        event=ReactEndEvent(
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
                except Exception:
                    pass

            app_log(
                f"LLM 函数 '{func_name}' 完成执行",
                location=get_location(),
            )

            # 更新观测数据
            usage_info = extract_usage_from_response(last_response)
            if usage_info is None:
                usage_info = _usage_from_context_delta(
                    llm_input_tokens_before,
                    llm_output_tokens_before,
                )
            usage_dict_no_tools: Optional[Dict[str, int]] = None
            if usage_info:
                usage_dict_no_tools = {
                    "prompt_tokens": usage_info.prompt_tokens,
                    "completion_tokens": usage_info.completion_tokens,
                    "total_tokens": usage_info.total_tokens,
                }
            generation_span.update(
                output={"content": content, "tool_calls": []},
                usage_details=usage_dict_no_tools,
            )
            # 注意：响应已经在上面 yield 过了，这里直接 return
            return

        # 更新观测数据
        usage_info = extract_usage_from_response(last_response)
        if usage_info is None:
            usage_info = _usage_from_context_delta(
                llm_input_tokens_before,
                llm_output_tokens_before,
            )
        usage_dict_with_tools: Optional[Dict[str, int]] = None
        if usage_info:
            usage_dict_with_tools = {
                "prompt_tokens": usage_info.prompt_tokens,
                "completion_tokens": usage_info.completion_tokens,
                "total_tokens": usage_info.total_tokens,
            }
        generation_span.update(
            output={"content": content, "tool_calls": tool_calls},
            usage_details=usage_dict_with_tools,
        )

    # Phase 2: Tool calling loop
    push_debug(
        f"LLM 函数 '{func_name}' 开始执行 {len(tool_calls)} 个工具调用",
        location=get_location(),
    )

    call_count += 1
    iteration = 1
    total_tool_calls += len(tool_calls)

    # 使用支持事件的工具调用处理函数
    if enable_event:
        # 使用异步生成器实时发射事件
        async for item in _process_tool_calls_with_events_gen(
            tool_calls=tool_calls,
            messages=current_messages,
            tool_map=tool_map,
            enable_event=enable_event,
            trace_id=current_trace_id,
            func_name=func_name,
            iteration=iteration,
        ):
            if isinstance(item, EventYield):
                yield item
            else:
                # 最后一个 yield 是 MessageList
                current_messages = item
    else:
        result_messages_iteration = await process_tool_calls(
            tool_calls=tool_calls,
            messages=cast(List[Dict[str, Any]], current_messages),
            tool_map=tool_map,
            event_emitter=NoOpEventEmitter(),
        )
        current_messages = cast(MessageList, result_messages_iteration)

    while call_count < max_tool_calls:
        # Phase 3: Iterative LLM-tool interaction
        iteration = call_count + 1

        # 发射迭代开始事件
        if enable_event:
            try:
                yield EventYield(
                    event=ReactIterationStartEvent(
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

        # 发射迭代中的 LLM 调用开始事件
        iteration_llm_start_time = time.time()
        if enable_event:
            try:
                yield EventYield(
                    event=LLMCallStartEvent(
                        event_type=ReActEventType.LLM_CALL_START,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=iteration,
                        messages=current_messages.copy(),
                        tools=tools,
                        llm_kwargs=llm_kwargs,
                        stream=stream,
                    )
                )
            except Exception:
                pass

        total_llm_calls += 1

        (
            iteration_input_tokens_before,
            iteration_output_tokens_before,
        ) = _read_context_token_counters()

        # 为迭代调用创建新的观测
        with langfuse_client.start_as_current_observation(
            as_type="generation",
            name=f"{func_name}_iteration_{call_count}_llm_call",
            input=current_messages,
            model=model_name,
            model_parameters=model_parameters,
            metadata={
                "stream": stream,
                "iteration": call_count,
                "tools_available": len(tools) if tools else 0,
            },
            completion_start_time=datetime.now(timezone.utc),
        ) as iteration_generation_span:
            last_response = None

            if stream:
                # Handle streaming response after tool calls
                content = ""
                tool_call_chunks = []  # Reset for iteration
                reasoning_details_list = []  # Reset for iteration
                chunk_index = 0
                accumulated_content = ""
                async for chunk in llm_interface.chat_stream(
                    messages=cast(List[Dict[str, Any]], current_messages),
                    tools=tools,
                    **llm_kwargs_filtered,
                ):
                    chunk_content = extract_content_from_stream_response(
                        chunk, func_name
                    )
                    content += chunk_content
                    accumulated_content += chunk_content
                    tool_call_chunks.extend(
                        extract_tool_calls_from_stream_response(chunk)
                    )
                    reasoning_details_list.extend(
                        extract_reasoning_details_from_stream(chunk)  # type: ignore
                    )
                    last_response = chunk

                    # 发射 chunk 事件
                    if enable_event:
                        try:
                            yield EventYield(
                                event=LLMChunkArriveEvent(
                                    event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                                    timestamp=datetime.now(timezone.utc),
                                    trace_id=current_trace_id,
                                    func_name=func_name,
                                    iteration=iteration,
                                    chunk=chunk,
                                    accumulated_content=accumulated_content,
                                    chunk_index=chunk_index,
                                )
                            )
                        except Exception:
                            pass
                        chunk_index += 1

                    # 发射响应
                    if enable_event:
                        try:
                            yield ResponseYield(
                                type="response",
                                response=chunk,
                                messages=current_messages.copy(),
                            )
                        except Exception:
                            pass
                    else:
                        yield chunk, cast(MessageList, current_messages.copy())
                tool_calls = accumulate_tool_calls_from_chunks(tool_call_chunks)
                reasoning_details = reasoning_details_list
            else:
                # Handle non-streaming response after tool calls
                response = await llm_interface.chat(
                    messages=cast(List[Dict[str, Any]], current_messages),
                    tools=tools,
                    **llm_kwargs_filtered,
                )

                content = extract_content_from_response(response, func_name)
                tool_calls = extract_tool_calls(response)
                reasoning_details = extract_reasoning_details(response)  # type: ignore
                last_response = response

                # 发射响应
                if enable_event:
                    try:
                        yield ResponseYield(
                            type="response",
                            response=response,
                            messages=current_messages.copy(),
                        )
                    except Exception:
                        pass
                else:
                    yield response, cast(MessageList, current_messages.copy())

            # 发射迭代中的 LLM 调用结束事件
            iteration_llm_execution_time = time.time() - iteration_llm_start_time
            if enable_event:
                try:
                    usage_info = extract_usage_from_response(last_response)
                    if usage_info is None:
                        usage_info = _usage_from_context_delta(
                            iteration_input_tokens_before,
                            iteration_output_tokens_before,
                        )
                    tool_calls_typed_iteration: List[ToolCall] = (
                        [dict_to_tool_call(tc) for tc in tool_calls]
                        if tool_calls
                        else []
                    )
                    yield EventYield(
                        event=LLMCallEndEvent(
                            event_type=ReActEventType.LLM_CALL_END,
                            timestamp=datetime.now(timezone.utc),
                            trace_id=current_trace_id,
                            func_name=func_name,
                            iteration=iteration,
                            response=last_response,
                            messages=current_messages.copy(),
                            tool_calls=tool_calls_typed_iteration,
                            usage=usage_info,
                            execution_time=iteration_llm_execution_time,
                        )
                    )
                except Exception:
                    pass

            # 更新迭代生成观测数据
            usage_info = extract_usage_from_response(last_response)
            if usage_info is None:
                usage_info = _usage_from_context_delta(
                    iteration_input_tokens_before,
                    iteration_output_tokens_before,
                )
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

        if len(tool_calls) != 0:
            assistant_tool_call_message = build_assistant_tool_message(
                tool_calls, reasoning_details if reasoning_details else None
            )
            current_messages.append(cast(Any, assistant_tool_call_message))

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
                    yield EventYield(
                        event=ReactIterationEndEvent(
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
                    usage_info = extract_usage_from_response(last_response)
                    if usage_info is None:
                        usage_info = _usage_from_context_delta(
                            iteration_input_tokens_before,
                            iteration_output_tokens_before,
                        )
                    final_content = (
                        extract_content_from_response(last_response, func_name)
                        if last_response
                        else ""
                    )
                    yield EventYield(
                        event=ReactEndEvent(
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

        # 使用支持事件的工具调用处理函数
        if enable_event:
            # 使用异步生成器实时发射事件
            async for item in _process_tool_calls_with_events_gen(
                tool_calls=tool_calls,
                messages=current_messages,
                tool_map=tool_map,
                enable_event=enable_event,
                trace_id=current_trace_id,
                func_name=func_name,
                iteration=iteration,
            ):
                if isinstance(item, EventYield):
                    yield item
                else:
                    # 最后一个 yield 是 MessageList
                    current_messages = item
        else:
            result_messages = await process_tool_calls(
                tool_calls=tool_calls,
                messages=cast(List[Dict[str, Any]], current_messages),
                tool_map=tool_map,
                event_emitter=NoOpEventEmitter(),
            )
            current_messages = cast(MessageList, result_messages)

        # 发射迭代结束事件
        iteration_time = time.time() - iteration_llm_start_time
        if enable_event:
            try:
                yield EventYield(
                    event=ReactIterationEndEvent(
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

    # Phase 4: Handle max_tool_calls limit reached
    push_debug(
        f"LLM 函数 '{func_name}' 达到最大工具调用次数限制 ({max_tool_calls})",
        location=get_location(),
    )

    # 发射最终 LLM 调用开始事件
    final_llm_start_time = time.time()
    if enable_event:
        try:
            yield EventYield(
                event=LLMCallStartEvent(
                    event_type=ReActEventType.LLM_CALL_START,
                    timestamp=datetime.now(timezone.utc),
                    trace_id=current_trace_id,
                    func_name=func_name,
                    iteration=call_count + 1,
                    messages=current_messages.copy(),
                    tools=None,
                    llm_kwargs=llm_kwargs,
                    stream=False,
                )
            )
        except Exception:
            pass

    total_llm_calls += 1

    final_input_tokens_before, final_output_tokens_before = (
        _read_context_token_counters()
    )

    # 为最终调用创建观测
    with langfuse_client.start_as_current_observation(
        as_type="generation",
        name=f"{func_name}_final_llm_call",
        input=current_messages,
        model=model_name,
        model_parameters=model_parameters,
        metadata={
            "stream": False,
            "reason": "max_tool_calls_reached",
            "call_count": call_count,
        },
        completion_start_time=datetime.now(timezone.utc),
    ) as final_generation_span:
        # 最终响应时不传递 tools，因为已经结束了
        llm_kwargs_final = llm_kwargs.copy()
        llm_kwargs_final.pop("tool_choice", None)

        final_response = await llm_interface.chat(
            messages=cast(List[Dict[str, Any]], current_messages),
            **llm_kwargs_final,
        )

        # 提取最终响应内容和用量
        final_content = extract_content_from_response(final_response, func_name)
        usage_info = extract_usage_from_response(final_response)
        if usage_info is None:
            usage_info = _usage_from_context_delta(
                final_input_tokens_before,
                final_output_tokens_before,
            )

        # 发射最终 LLM 调用结束事件
        final_llm_execution_time = time.time() - final_llm_start_time
        if enable_event:
            try:
                yield EventYield(
                    event=LLMCallEndEvent(
                        event_type=ReActEventType.LLM_CALL_END,
                        timestamp=datetime.now(timezone.utc),
                        trace_id=current_trace_id,
                        func_name=func_name,
                        iteration=call_count + 1,
                        response=final_response,
                        messages=current_messages.copy(),
                        tool_calls=[],
                        usage=usage_info,
                        execution_time=final_llm_execution_time,
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
        if enable_event:
            try:
                yield EventYield(
                    event=ReactEndEvent(
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
            except Exception:
                pass

        app_log(
            f"LLM 函数 '{func_name}' 完成执行",
            location=get_location(),
        )


__all__ = ["execute_llm"]
