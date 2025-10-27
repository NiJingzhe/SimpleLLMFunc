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

from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List

from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import app_log, push_debug
from SimpleLLMFunc.logger.logger import get_current_context_attribute, get_location
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
    extract_tool_calls,
    extract_tool_calls_from_stream_response,
    process_tool_calls,
)

from SimpleLLMFunc.observability.langfuse_client import langfuse_client


async def execute_llm(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    max_tool_calls: int,
    stream: bool = False,
    **llm_kwargs,
) -> AsyncGenerator[Any, None]:
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
            Responses from the LLM (either complete responses or stream chunks).
    """

    func_name = get_current_context_attribute("function_name") or "Unknown Function"

    current_messages = messages
    call_count = 0

    push_debug(
        f"LLM 函数 '{func_name}' 开始执行，消息数: {len(current_messages)}",
        location=get_location(),
    )

    # Phase 1: Initial LLM call
    # 准备 Langfuse 观测数据
    model_parameters = {k: v for k, v in llm_kwargs.items() if k not in ["retry_times"]}
    model_name = llm_interface.model_name

    # 声明变量
    content = ""
    tool_calls: List[Dict[str, Any]] = []
    tool_call_chunks: List[Dict[str, Any]] = []
    last_response: Any = None

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
            async for chunk in llm_interface.chat_stream(
                messages=current_messages,
                tools=tools,
                **llm_kwargs,
            ):
                content += extract_content_from_stream_response(chunk, func_name)
                tool_call_chunks.extend(extract_tool_calls_from_stream_response(chunk))
                last_response = chunk
                yield chunk

            tool_calls = accumulate_tool_calls_from_chunks(tool_call_chunks)
        else:
            # Handle non-streaming response
            initial_response = await llm_interface.chat(
                messages=current_messages,
                tools=tools,
                **llm_kwargs,
            )

            content = extract_content_from_response(initial_response, func_name)
            tool_calls = extract_tool_calls(initial_response)
            last_response = initial_response
            yield initial_response

        push_debug(
            f"LLM 函数 '{func_name}' 初始响应已获取，工具调用数: {len(tool_calls)}",
            location=get_location(),
        )

        # Append assistant response to message history
        if content.strip() != "":
            assistant_message = build_assistant_response_message(content)
            current_messages.append(assistant_message)

        if len(tool_calls) != 0:
            assistant_tool_call_message = build_assistant_tool_message(tool_calls)
            current_messages.append(assistant_tool_call_message)
        else:
            # No tool calls, return final result
            app_log(
                f"LLM 函数 '{func_name}' 完成执行",
                location=get_location(),
            )
            
            # 更新观测数据
            usage_info = extract_usage_from_response(last_response)
            generation_span.update(
                output={"content": content, "tool_calls": []},
                usage_details=usage_info,
            )
            return

        # 更新观测数据
        usage_info = extract_usage_from_response(last_response)
        generation_span.update(
            output={"content": content, "tool_calls": tool_calls},
            usage_details=usage_info,
        )

    # Phase 2: Tool calling loop
    push_debug(
        f"LLM 函数 '{func_name}' 开始执行 {len(tool_calls)} 个工具调用",
        location=get_location(),
    )

    call_count += 1

    current_messages = await process_tool_calls(
        tool_calls=tool_calls,
        messages=current_messages,
        tool_map=tool_map,
    )

    while call_count < max_tool_calls:
        # Phase 3: Iterative LLM-tool interaction
        push_debug(
            f"LLM 函数 '{func_name}' 工具调用循环 (次数: {call_count})",
            location=get_location(),
        )

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
                async for chunk in llm_interface.chat_stream(
                    messages=current_messages,
                    tools=tools,
                    **llm_kwargs,
                ):
                    content += extract_content_from_stream_response(chunk, func_name)
                    tool_call_chunks.extend(
                        extract_tool_calls_from_stream_response(chunk)
                    )
                    last_response = chunk
                    yield chunk
                tool_calls = accumulate_tool_calls_from_chunks(tool_call_chunks)
            else:
                # Handle non-streaming response after tool calls
                response = await llm_interface.chat(
                    messages=current_messages,
                    tools=tools,
                    **llm_kwargs,
                )

                content = extract_content_from_response(response, func_name)
                tool_calls = extract_tool_calls(response)
                last_response = response
                yield response

            # 更新迭代生成观测数据
            usage_info = extract_usage_from_response(last_response)
            iteration_generation_span.update(
                output={"content": content, "tool_calls": tool_calls},
                usage_details=usage_info,
            )

        # Append new assistant response to message history
        if content.strip() != "":
            assistant_message = build_assistant_response_message(content)
            current_messages.append(assistant_message)

        if len(tool_calls) != 0:
            assistant_tool_call_message = build_assistant_tool_message(tool_calls)
            current_messages.append(assistant_tool_call_message)

        if len(tool_calls) == 0:
            # No more tool calls, exit loop
            push_debug(
                f"LLM 函数 '{func_name}' 无更多工具调用，返回最终结果",
                location=get_location(),
            )
            app_log(
                f"LLM 函数 '{func_name}' 完成执行",
                location=get_location(),
            )
            return

        # Continue with next iteration of tool calls
        push_debug(
            f"LLM 函数 '{func_name}' 发现 {len(tool_calls)} 个工具调用",
            location=get_location(),
        )

        current_messages = await process_tool_calls(
            tool_calls=tool_calls,
            messages=current_messages,
            tool_map=tool_map,
        )

        call_count += 1

    # Phase 4: Handle max_tool_calls limit reached
    push_debug(
        f"LLM 函数 '{func_name}' 达到最大工具调用次数限制 ({max_tool_calls})",
        location=get_location(),
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
        final_response = await llm_interface.chat(
            messages=current_messages,
            **llm_kwargs,
        )

        # 提取最终响应内容和用量
        final_content = extract_content_from_response(final_response, func_name)
        usage_info = extract_usage_from_response(final_response)

        # 更新最终观测数据
        final_generation_span.update(
            output={"content": final_content, "tool_calls": []},
            usage_details=usage_info,
        )

        app_log(
            f"LLM 函数 '{func_name}' 完成执行",
            location=get_location(),
        )

        yield final_response


__all__ = ["execute_llm"]
