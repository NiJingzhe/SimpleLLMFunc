"""Tool call extraction and execution helpers."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, TypedDict

from SimpleLLMFunc.logger import push_debug, push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.llm_decorator.multimodal_types import ImgPath, ImgUrl, Text


class ToolCallFunctionInfo(TypedDict):
    name: Optional[str]
    arguments: str


class AccumulatedToolCall(TypedDict):
    id: Optional[str]
    type: Optional[str]
    function: ToolCallFunctionInfo


def is_valid_tool_result(result: Any) -> bool:
    """Validate whether a tool return value is supported."""

    if isinstance(result, (ImgPath, ImgUrl)):
        return True

    if isinstance(result, str):
        return True

    if isinstance(result, tuple) and len(result) == 2:
        text_part, img_part = result
        if isinstance(text_part, str) and isinstance(img_part, (ImgPath, ImgUrl)):
            return True
        return False

    try:
        json.dumps(result)
        return True
    except (TypeError, ValueError):
        return False


def process_tool_calls(
    tool_calls: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    tool_map: Dict[str, Callable[..., Any]],
) -> List[Dict[str, Any]]:
    """Execute tool calls and append results to the message history."""

    current_messages = messages

    for tool_call in tool_calls:
        tool_call_id = tool_call.get("id")
        function_call = tool_call.get("function", {})
        tool_name = function_call.get("name")
        arguments_str = function_call.get("arguments", "{}")

        if tool_name not in tool_map:
            push_error(f"工具 '{tool_name}' 不在可用工具列表中")
            tool_error_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    {"error": f"找不到工具 '{tool_name}'"}, ensure_ascii=False, indent=2
                ),
            }
            current_messages.append(tool_error_message)
            continue

        try:
            arguments = json.loads(arguments_str)

            push_debug(f"执行工具 '{tool_name}' 参数: {arguments_str}")
            tool_func = tool_map[tool_name]
            tool_result = tool_func(**arguments)

            if not is_valid_tool_result(tool_result):
                push_warning(
                    f"工具 '{tool_name}' 返回了不支持的格式: {type(tool_result)}。支持的返回格式包括: str, JSON可序列化对象, ImgPath, ImgUrl, Tuple[str, ImgPath], Tuple[str, ImgUrl]",
                    location=get_location(),
                )
                tool_result_content_json: str = json.dumps(
                    str(tool_result), ensure_ascii=False, indent=2
                )
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_content_json,
                }
                current_messages.append(tool_message)
                continue

            from SimpleLLMFunc.base.messages import (
                create_image_path_content,
                create_image_url_content,
                create_text_content,
            )

            if isinstance(tool_result, ImgUrl):
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": tool_result.url,
                        "detail": tool_result.detail,
                    },
                }

                if (
                    current_messages
                    and current_messages[-1].get("role") == "assistant"
                    and current_messages[-1].get("tool_calls")
                ):
                    current_messages[-1] = {
                        "role": "assistant",
                        "content": f"我将会通过工具 '{tool_name}' 获取目标的图像",
                    }

                user_multimodal_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"这是工具 '{tool_name}' 返回的图像：",
                        },
                        image_content,
                    ],
                }
                current_messages.append(user_multimodal_message)
                continue

            if isinstance(tool_result, ImgPath):
                base64_img = tool_result.to_base64()
                mime_type = tool_result.get_mime_type()
                data_url = f"data:{mime_type};base64,{base64_img}"

                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": tool_result.detail,
                    },
                }

                if (
                    current_messages
                    and current_messages[-1].get("role") == "assistant"
                    and current_messages[-1].get("tool_calls")
                ):
                    current_messages[-1] = {
                        "role": "assistant",
                        "content": f"我将要调用工具 '{tool_name}' 获取图像文件",
                    }

                user_multimodal_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"这是工具 '{tool_name}' 返回的图像文件：",
                        },
                        image_content,
                    ],
                }
                current_messages.append(user_multimodal_message)
                continue

            if isinstance(tool_result, tuple) and len(tool_result) == 2:
                text_part, img_part = tool_result
                if isinstance(text_part, str) and isinstance(img_part, ImgUrl):
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": img_part.url,
                            "detail": img_part.detail,
                        },
                    }

                    if (
                        current_messages
                        and current_messages[-1].get("role") == "assistant"
                        and current_messages[-1].get("tool_calls")
                    ):
                        current_messages[-1] = {
                            "role": "assistant",
                            "content": f"我将会通过工具 '{tool_name}' 获取目标的图像，并提供说明文本",
                        }

                    user_multimodal_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"这是工具 '{tool_name}' 返回的图像和说明：{text_part}",
                            },
                            image_content,
                        ],
                    }
                    current_messages.append(user_multimodal_message)
                    continue

                if isinstance(text_part, str) and isinstance(img_part, ImgPath):
                    base64_img = img_part.to_base64()
                    mime_type = img_part.get_mime_type()
                    data_url = f"data:{mime_type};base64,{base64_img}"

                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": img_part.detail,
                        },
                    }

                    if (
                        current_messages
                        and current_messages[-1].get("role") == "assistant"
                        and current_messages[-1].get("tool_calls")
                    ):
                        current_messages[-1] = {
                            "role": "assistant",
                            "content": f"我将要调用工具 '{tool_name}' 获取图像文件，并提供说明文本",
                        }

                    user_multimodal_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"这是工具 '{tool_name}' 返回的图像文件和说明：{text_part}",
                            },
                            image_content,
                        ],
                    }
                    current_messages.append(user_multimodal_message)
                    continue

                tool_result_content_json = json.dumps(
                    tool_result, ensure_ascii=False, indent=2
                )
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_content_json,
                }
                current_messages.append(tool_message)
                push_debug(f"工具 '{tool_name}' 执行完成: {tool_result_content_json}")
                continue

            if isinstance(tool_result, (Text, str)):
                tool_result_content_json = json.dumps(
                    tool_result, ensure_ascii=False, indent=2
                )

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_content_json,
                }
            else:
                tool_result_content_json = json.dumps(
                    tool_result, ensure_ascii=False, indent=2
                )

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_content_json,
                }

            current_messages.append(tool_message)

            if isinstance(tool_result, (ImgUrl, ImgPath)):
                push_debug(
                    f"工具 '{tool_name}' 执行完成: image payload",
                    location=get_location(),
                )
            else:
                push_debug(
                    f"工具 '{tool_name}' 执行完成: {json.dumps(tool_result, ensure_ascii=False)}"
                )

        except Exception as exc:
            error_message = f"工具 '{tool_name}' 以参数 {arguments_str} 在执行或结果解析中出错，错误: {str(exc)}"
            push_error(error_message)

            tool_error_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    {"error": error_message}, ensure_ascii=False, indent=2
                ),
            }
            current_messages.append(tool_error_message)

    return current_messages


def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """Extract tool-call metadata from a synchronous response."""

    tool_calls: List[Dict[str, Any]] = []

    try:
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": getattr(tool_call, "type", "function"),
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
    except Exception as exc:
        push_error(f"提取工具调用时出错: {str(exc)}")
    finally:
        return tool_calls


def accumulate_tool_calls_from_chunks(
    tool_call_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge tool-call chunks emitted during streaming responses."""

    accumulated_calls: Dict[int, AccumulatedToolCall] = {}

    for chunk in tool_call_chunks:
        index = chunk.get("index")
        if index is None:
            push_warning(
                "工具调用 chunk 缺少 'index' 属性，已跳过处理",
                location=get_location(),
            )
            continue

        if index not in accumulated_calls:
            accumulated_calls[index] = AccumulatedToolCall(
                id=None,
                type=None,
                function=ToolCallFunctionInfo(name=None, arguments=""),
            )

        if chunk.get("id"):
            accumulated_calls[index]["id"] = chunk["id"]
        if chunk.get("type"):
            accumulated_calls[index]["type"] = chunk["type"]

        if "function" in chunk:
            function_chunk = chunk["function"]
            func_info = accumulated_calls[index]["function"]
            if function_chunk.get("name"):
                func_info["name"] = function_chunk["name"]
            if function_chunk.get("arguments"):
                func_info["arguments"] += function_chunk["arguments"]

    complete_tool_calls: List[Dict[str, Any]] = []
    for call in accumulated_calls.values():
        if call["id"] and call["function"]["name"]:
            if not call["type"]:
                call["type"] = "function"
            complete_tool_calls.append(
                {
                    "id": call["id"],
                    "type": call["type"],
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    },
                }
            )

    return complete_tool_calls


def extract_tool_calls_from_stream_response(chunk: Any) -> List[Dict[str, Any]]:
    """Extract tool-call fragments from a streaming chunk."""

    tool_call_chunks: List[Dict[str, Any]] = []

    try:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                delta = choice.delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        tool_call_chunk: Dict[str, Any] = {
                            "index": getattr(tool_call, "index", None),
                            "id": getattr(tool_call, "id", None),
                            "type": getattr(tool_call, "type", None),
                        }

                        if hasattr(tool_call, "function") and tool_call.function:
                            function_info: Dict[str, Any] = {}
                            if (
                                hasattr(tool_call.function, "name")
                                and tool_call.function.name
                            ):
                                function_info["name"] = tool_call.function.name
                            if (
                                hasattr(tool_call.function, "arguments")
                                and tool_call.function.arguments
                            ):
                                function_info["arguments"] = (
                                    tool_call.function.arguments
                                )

                            if function_info:
                                tool_call_chunk["function"] = function_info

                        tool_call_chunks.append(tool_call_chunk)
    except Exception as exc:
        push_error(f"提取流工具调用时出错: {str(exc)}")

    return tool_call_chunks


__all__ = [
    "is_valid_tool_result",
    "process_tool_calls",
    "extract_tool_calls",
    "accumulate_tool_calls_from_chunks",
    "extract_tool_calls_from_stream_response",
    "ToolCallFunctionInfo",
    "AccumulatedToolCall",
]
