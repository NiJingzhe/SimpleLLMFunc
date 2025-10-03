"""Core execution pipeline handling LLM and tool-call orchestration."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Callable, Dict, List

from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import app_log, push_debug
from SimpleLLMFunc.logger.logger import get_current_context_attribute, get_location
from SimpleLLMFunc.base.messages import (
	build_assistant_response_message,
	build_assistant_tool_message,
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


async def execute_llm(
	llm_interface: LLM_Interface,
	messages: List[Dict[str, Any]],
	tools: List[Dict[str, Any]] | None,
	tool_map: Dict[str, Callable[..., Any]],
	max_tool_calls: int,
	stream: bool = False,
	**llm_kwargs,
) -> AsyncGenerator[Any, None]:
	"""Execute LLM calls and orchestrate iterative tool usage."""

	func_name = get_current_context_attribute("function_name") or "Unknown Function"

	current_messages = messages
	call_count = 0

	push_debug(
		f"LLM 函数 '{func_name}' 将要发起初始请求，消息数: {len(current_messages)}",
		location=get_location(),
	)

	if stream:
		push_debug(f"LLM 函数 '{func_name}' 使用流式响应", location=get_location())
		push_debug(f"LLM 函数 '{func_name}' 初始流式响应开始", location=get_location())

		content = ""
		tool_call_chunks: List[Dict[str, Any]] = []

		async for chunk in llm_interface.chat_stream(
			messages=current_messages,
			tools=tools,
			**llm_kwargs,
		):
			content += extract_content_from_stream_response(chunk, func_name)
			tool_call_chunks.extend(extract_tool_calls_from_stream_response(chunk))
			yield chunk

		tool_calls = accumulate_tool_calls_from_chunks(tool_call_chunks)
	else:
		push_debug(f"LLM 函数 '{func_name}' 使用非流式响应", location=get_location())
		initial_response = await llm_interface.chat(
			messages=current_messages,
			tools=tools,
			**llm_kwargs,
		)

		push_debug(
			f"LLM 函数 '{func_name}' 初始响应: {initial_response}",
			location=get_location(),
		)

		content = extract_content_from_response(initial_response, func_name)
		tool_calls = extract_tool_calls(initial_response)
		yield initial_response

	push_debug(
		f"LLM 函数 '{func_name}' 初始响应中抽取的content是: {content}",
		location=get_location(),
	)

	if content.strip() != "":
		assistant_message = build_assistant_response_message(content)
		current_messages.append(assistant_message)

	if len(tool_calls) != 0:
		assistant_tool_call_message = build_assistant_tool_message(tool_calls)
		current_messages.append(assistant_tool_call_message)
	else:
		push_debug("未发现工具调用，直接返回结果", location=get_location())
		app_log(
			f"LLM 函数 '{func_name}' 本次调用的完整messages: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
			location=get_location(),
		)
		return

	push_debug(
		f"LLM 函数 '{func_name}' 抽取工具后构建的完整消息: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
		location=get_location(),
	)

	push_debug(
		f"LLM 函数 '{func_name}' 发现 {len(tool_calls)} 个工具调用，开始执行工具",
		location=get_location(),
	)

	call_count += 1

	current_messages = process_tool_calls(
		tool_calls=tool_calls,
		messages=current_messages,
		tool_map=tool_map,
	)

	while call_count < max_tool_calls:
		push_debug(
			f"LLM 函数 '{func_name}' 工具调用循环: 第 {call_count}/{max_tool_calls} 次返回工具响应",
			location=get_location(),
		)

		if stream:
			push_debug(f"LLM 函数 '{func_name}' 使用流式响应", location=get_location())
			push_debug(
				f"LLM 函数 '{func_name}' 第 {call_count} 次工具调用返回后，LLM流式响应开始",
				location=get_location(),
			)

			content = ""
			tool_call_chunks = []
			async for chunk in llm_interface.chat_stream(
				messages=current_messages,
				tools=tools,
				**llm_kwargs,
			):
				content += extract_content_from_stream_response(chunk, func_name)
				tool_call_chunks.extend(
					extract_tool_calls_from_stream_response(chunk)
				)
				yield chunk
			tool_calls = accumulate_tool_calls_from_chunks(tool_call_chunks)
		else:
			push_debug(
				f"LLM 函数 '{func_name}' 使用非流式响应", location=get_location()
			)
			response = await llm_interface.chat(
				messages=current_messages,
				tools=tools,
				**llm_kwargs,
			)

			push_debug(
				f"LLM 函数 '{func_name}' 第 {call_count} 次工具调用返回后，LLM，响应: {response}",
				location=get_location(),
			)

			content = extract_content_from_response(response, func_name)
			tool_calls = extract_tool_calls(response)
			yield response

		push_debug(
			f"LLM 函数 '{func_name}' 初始响应中抽取的content是: {content}",
			location=get_location(),
		)

		if content.strip() != "":
			assistant_message = build_assistant_response_message(content)
			current_messages.append(assistant_message)

		if len(tool_calls) != 0:
			assistant_tool_call_message = build_assistant_tool_message(tool_calls)
			current_messages.append(assistant_tool_call_message)

		push_debug(
			f"LLM 函数 '{func_name}' 抽取工具后构建的完整消息: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
			location=get_location(),
		)

		if len(tool_calls) == 0:
			push_debug(
				f"LLM 函数 '{func_name}' 没有更多工具调用，返回最终响应",
				location=get_location(),
			)
			app_log(
				f"LLM 函数 '{func_name}' 本次调用的完整messages: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
				location=get_location(),
			)
			return

		push_debug(
			f"LLM 函数 '{func_name}' 发现 {len(tool_calls)} 个新的工具调用",
			location=get_location(),
		)

		current_messages = process_tool_calls(
			tool_calls=tool_calls,
			messages=current_messages,
			tool_map=tool_map,
		)

		call_count += 1

	push_debug(
		f"LLM 函数 '{func_name}' 达到最大工具调用次数 ({max_tool_calls})，强制结束并获取最终响应",
		location=get_location(),
	)

	final_response = await llm_interface.chat(
		messages=current_messages,
		**llm_kwargs,
	)

	app_log(
		f"LLM 函数 '{func_name}' 本次调用的完整messages: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
		location=get_location(),
	)

	yield final_response


__all__ = ["execute_llm"]
