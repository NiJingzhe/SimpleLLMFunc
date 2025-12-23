"""Step 4: Execute ReAct loop for llm_function."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Union, cast

from SimpleLLMFunc.base.ReAct import execute_llm
from SimpleLLMFunc.base.post_process import extract_content_from_response
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import push_debug, push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.type.message import MessageList


from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.utils import get_last_item_of_async_generator
from SimpleLLMFunc.llm_decorator.utils import process_tools


def prepare_tools_for_execution(
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]],
    func_name: str,
) -> tuple[Optional[List[Dict[str, Any]]], Dict[str, Callable[..., Awaitable[Any]]]]:
    """准备工具供执行使用"""
    return process_tools(toolkit, func_name)


async def execute_llm_call(
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: Optional[List[Dict[str, Any]]],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    max_tool_calls: int,
    stream: bool = False,
    **llm_kwargs: Any,
) -> AsyncGenerator[Any, None]:
    """执行 LLM 调用"""
    # 类型转换：MessageList 兼容 List[Dict[str, Any]]
    # execute_llm 现在返回 (response, updated_messages) 元组，但我们只需要 response
    async for response, _ in execute_llm(
        llm_interface=llm_interface,
        messages=cast(List[Dict[str, Any]], messages),
        tools=tools,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=stream,
        **llm_kwargs,
    ):
        yield response


async def get_final_response(
    response_stream: AsyncGenerator[Any, None],
) -> Any:
    """从响应流中获取最后一个响应"""
    return await get_last_item_of_async_generator(response_stream)


def check_response_content_empty(response: Any, func_name: str) -> bool:
    """检查响应内容是否为空"""
    content = ""
    if hasattr(response, "choices") and len(response.choices) > 0:
        message = response.choices[0].message
        content = message.content if message and hasattr(message, "content") else ""

    return content == ""


async def retry_llm_call(
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: Optional[List[Dict[str, Any]]],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    max_tool_calls: int,
    retry_times: int,
    func_name: str,
    **llm_kwargs: Any,
) -> Any:
    """重试 LLM 调用"""
    final_response = None

    for attempt in range(retry_times + 1):
        if attempt > 0:
            push_debug(
                f"Async LLM function '{func_name}' retry attempt {attempt}...",
                location=get_location(),
            )

        # 执行 LLM 调用
        response_stream = execute_llm_call(
            llm_interface=llm_interface,
            messages=messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=max_tool_calls,
            stream=False,
            **llm_kwargs,
        )

        # 获取最终响应
        final_response = await get_final_response(response_stream)

        # 检查内容是否为空
        content = extract_content_from_response(final_response, func_name)
        if content != "":
            break

    # 最终检查
    if final_response:
        content = extract_content_from_response(final_response, func_name)
        if content == "":
            push_error(
                f"Async LLM function '{func_name}' response content still empty, "
                "retry attempts exhausted.",
                location=get_location(),
            )
            raise ValueError("LLM response content is empty after retries.")

    return final_response


async def execute_react_loop(
    llm_interface: LLM_Interface,
    messages: MessageList,
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]],
    max_tool_calls: int,
    llm_kwargs: Dict[str, Any],
    func_name: str,
) -> Any:
    """执行 ReAct 循环的完整流程（包含重试）"""
    # 1. 准备工具
    tool_param, tool_map = prepare_tools_for_execution(toolkit, func_name)

    # 2. 执行 LLM 调用
    response_stream = execute_llm_call(
        llm_interface=llm_interface,
        messages=messages,
        tools=tool_param,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=False,
        **llm_kwargs,
    )

    # 3. 获取最终响应
    final_response = await get_final_response(response_stream)

    # 4. 检查响应内容是否为空
    if check_response_content_empty(final_response, func_name):
        push_warning(
            f"Async LLM function '{func_name}' returned empty response content, "
            "will retry automatically.",
            location=get_location(),
        )

        # 5. 重试 LLM 调用
        retry_times = llm_kwargs.get("retry_times", 2)
        final_response = await retry_llm_call(
            llm_interface=llm_interface,
            messages=messages,
            tools=tool_param,
            tool_map=tool_map,
            max_tool_calls=max_tool_calls,
            retry_times=retry_times,
            func_name=func_name,
            **llm_kwargs,
        )

    # 6. 记录最终响应
    push_debug(
        f"Async LLM function '{func_name}' received response "
        f"{json.dumps(final_response, default=str, ensure_ascii=False, indent=2)}",
        location=get_location(),
    )

    return final_response

