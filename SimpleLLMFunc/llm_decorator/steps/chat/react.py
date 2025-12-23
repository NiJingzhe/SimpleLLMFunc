"""Step 4: Execute ReAct loop for llm_chat (streaming)."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from SimpleLLMFunc.base.ReAct import execute_llm
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.llm_decorator.utils import process_tools


def prepare_tools_for_execution(
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]],
    func_name: str,
) -> tuple[Optional[List[Dict[str, Any]]], Dict[str, Callable[..., Awaitable[Any]]]]:
    """准备工具供执行使用"""
    return process_tools(toolkit, func_name)


async def execute_llm_call(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    max_tool_calls: int,
    stream: bool = False,
    **llm_kwargs: Any,
) -> AsyncGenerator[Tuple[Any, List[Dict[str, Any]]], None]:
    """执行 LLM 调用，返回响应和更新后的消息"""
    async for response, updated_messages in execute_llm(
        llm_interface=llm_interface,
        messages=messages,
        tools=tools,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=stream,
        **llm_kwargs,
    ):
        yield response, updated_messages


async def execute_react_loop_streaming(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, Any]],
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]],
    max_tool_calls: int,
    stream: bool,
    llm_kwargs: Dict[str, Any],
    func_name: str,
) -> AsyncGenerator[Tuple[Any, List[Dict[str, Any]]], None]:
    """执行 ReAct 循环的流式版本（无重试），返回响应和更新后的消息"""
    # 1. 准备工具
    tool_param, tool_map = prepare_tools_for_execution(toolkit, func_name)

    # 2. 执行 LLM 调用（流式）
    response_stream = execute_llm_call(
        llm_interface=llm_interface,
        messages=messages,
        tools=tool_param,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=stream,
        **llm_kwargs,
    )

    # 3. 返回响应流和更新后的消息
    async for response, updated_messages in response_stream:
        yield response, updated_messages

