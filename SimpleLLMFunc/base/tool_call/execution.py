"""Tool call execution helpers."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Awaitable, Callable, Dict, List, get_type_hints, get_origin, get_args, Union as TypingUnion

from SimpleLLMFunc.logger import push_debug, push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text
from SimpleLLMFunc.observability.langfuse_client import langfuse_client


def _convert_tool_arguments(
    arguments: Dict[str, Any],
    tool_func: Callable[..., Awaitable[Any]],
) -> Dict[str, Any]:
    """转换工具参数，将字符串列表转换为多模态对象列表。
    
    根据工具函数的类型注解，自动将 LLM 传递的字符串数组转换为对应的多模态对象数组。
    支持的类型：
    - List[ImgPath] -> List[ImgPath对象]
    - List[ImgUrl] -> List[ImgUrl对象]
    - List[Text] -> List[Text对象]
    - Optional[List[...]] -> 处理 None 值
    - Union 类型 -> 提取非 None 类型
    
    Args:
        arguments: LLM 传递的原始参数字典（JSON 解析后）
        tool_func: 工具函数对象
        
    Returns:
        转换后的参数字典
    """
    try:
        # 获取函数签名和类型注解
        signature = inspect.signature(tool_func)
        type_hints = get_type_hints(tool_func)
        
        converted_args = {}
        
        for param_name, param_value in arguments.items():
            if param_name not in signature.parameters:
                # 参数不在签名中，保持原样
                converted_args[param_name] = param_value
                continue
            
            param_type = type_hints.get(param_name, Any)
            
            # 处理 None 值
            if param_value is None:
                converted_args[param_name] = None
                continue
            
            # 处理 Optional 类型
            origin = get_origin(param_type)
            if origin is TypingUnion:
                args = get_args(param_type)
                # 提取非 None 类型
                non_none_types = [t for t in args if t is not type(None)]
                if non_none_types:
                    param_type = non_none_types[0]
                    origin = get_origin(param_type)
            
            # 处理列表类型
            if origin is list:
                args = get_args(param_type)
                if not args:
                    # List 没有类型参数，保持原样
                    converted_args[param_name] = param_value
                    continue
                
                element_type = args[0]
                
                # 检查是否为多模态列表类型
                if element_type is ImgPath:
                    if isinstance(param_value, list):
                        try:
                            converted_args[param_name] = [ImgPath(path) for path in param_value]
                        except Exception as e:
                            push_warning(
                                f"工具参数 '{param_name}' 转换为 List[ImgPath] 失败: {e}，使用原始值",
                                location=get_location(),
                            )
                            converted_args[param_name] = param_value
                    else:
                        converted_args[param_name] = param_value
                elif element_type is ImgUrl:
                    if isinstance(param_value, list):
                        try:
                            converted_args[param_name] = [ImgUrl(url) for url in param_value]
                        except Exception as e:
                            push_warning(
                                f"工具参数 '{param_name}' 转换为 List[ImgUrl] 失败: {e}，使用原始值",
                                location=get_location(),
                            )
                            converted_args[param_name] = param_value
                    else:
                        converted_args[param_name] = param_value
                elif element_type is Text:
                    if isinstance(param_value, list):
                        try:
                            converted_args[param_name] = [Text(text) for text in param_value]
                        except Exception as e:
                            push_warning(
                                f"工具参数 '{param_name}' 转换为 List[Text] 失败: {e}，使用原始值",
                                location=get_location(),
                            )
                            converted_args[param_name] = param_value
                    else:
                        converted_args[param_name] = param_value
                else:
                    # 非多模态列表，保持原样
                    converted_args[param_name] = param_value
            # 处理单个多模态类型
            elif param_type is ImgPath:
                if isinstance(param_value, str):
                    try:
                        converted_args[param_name] = ImgPath(param_value)
                    except Exception as e:
                        push_warning(
                            f"工具参数 '{param_name}' 转换为 ImgPath 失败: {e}，使用原始值",
                            location=get_location(),
                        )
                        converted_args[param_name] = param_value
                else:
                    converted_args[param_name] = param_value
            elif param_type is ImgUrl:
                if isinstance(param_value, str):
                    try:
                        converted_args[param_name] = ImgUrl(param_value)
                    except Exception as e:
                        push_warning(
                            f"工具参数 '{param_name}' 转换为 ImgUrl 失败: {e}，使用原始值",
                            location=get_location(),
                        )
                        converted_args[param_name] = param_value
                else:
                    converted_args[param_name] = param_value
            elif param_type is Text:
                if isinstance(param_value, str):
                    try:
                        converted_args[param_name] = Text(param_value)
                    except Exception as e:
                        push_warning(
                            f"工具参数 '{param_name}' 转换为 Text 失败: {e}，使用原始值",
                            location=get_location(),
                        )
                        converted_args[param_name] = param_value
                else:
                    converted_args[param_name] = param_value
            else:
                # 其他类型，保持原样
                converted_args[param_name] = param_value
        
        return converted_args
    except Exception as e:
        push_warning(
            f"工具参数转换过程中出错: {e}，使用原始参数",
            location=get_location(),
        )
        return arguments


async def _execute_single_tool_call(
    tool_call: Dict[str, Any],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    """Execute a single tool call and return its results.

    Returns:
        Tuple of (tool_call_dict, list_of_messages_to_append, is_multimodal)
    """

    tool_call_id = tool_call.get("id")
    function_call = tool_call.get("function", {})
    tool_name = function_call.get("name")
    arguments_str = function_call.get("arguments", "{}")
    messages_to_append: List[Dict[str, Any]] = []

    if tool_name not in tool_map:
        push_error(f"工具 '{tool_name}' 不在可用工具列表中")
        tool_error_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(
                {"error": f"找不到工具 '{tool_name}'"}, ensure_ascii=False, indent=2
            ),
        }
        messages_to_append.append(tool_error_message)
        return (tool_call, messages_to_append, False)

    # 使用 Langfuse 观测工具调用
    with langfuse_client.start_as_current_observation(
        as_type="tool",
        name=tool_name,
        input={"raw_arguments": arguments_str},
        metadata={"tool_call_id": tool_call_id},
    ) as tool_span:
        try:
            arguments = json.loads(arguments_str)

            # 更新为解析后的参数
            tool_span.update(input=arguments)

            push_debug(f"执行工具 '{tool_name}' 参数: {arguments_str}")

            tool_func = tool_map[tool_name]
            
            # 转换参数：将字符串列表转换为多模态对象列表
            converted_arguments = _convert_tool_arguments(arguments, tool_func)
            
            tool_result = await tool_func(**converted_arguments)

            # 更新工具调用观测数据，序列化输出以便langfuse记录
            from SimpleLLMFunc.base.tool_call.validation import (
                is_valid_tool_result,
                serialize_tool_output_for_langfuse,
            )

            serialized_output = serialize_tool_output_for_langfuse(tool_result)
            tool_span.update(output=serialized_output)

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
                messages_to_append.append(tool_message)
                return (tool_call, messages_to_append, False)

            if isinstance(tool_result, ImgUrl):
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": tool_result.url,
                        "detail": tool_result.detail,
                    },
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
                messages_to_append.append(user_multimodal_message)
                return (tool_call, messages_to_append, True)

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
                messages_to_append.append(user_multimodal_message)
                return (tool_call, messages_to_append, True)

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
                    messages_to_append.append(user_multimodal_message)
                    return (tool_call, messages_to_append, True)

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
                    messages_to_append.append(user_multimodal_message)
                    return (tool_call, messages_to_append, True)

                tool_result_content_json = json.dumps(
                    tool_result, ensure_ascii=False, indent=2
                )
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_content_json,
                }
                messages_to_append.append(tool_message)
                push_debug(f"工具 '{tool_name}' 执行完成: {tool_result_content_json}")
                return (tool_call, messages_to_append, False)

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

            messages_to_append.append(tool_message)

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

            # 记录错误到langfuse
            tool_span.update(
                output={"error": error_message, "exception_type": type(exc).__name__},
                level="ERROR",
            )

            tool_error_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    {"error": error_message}, ensure_ascii=False, indent=2
                ),
            }
            messages_to_append.append(tool_error_message)

    return (tool_call, messages_to_append, False)


async def process_tool_calls(
    tool_calls: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
) -> List[Dict[str, Any]]:
    """Execute tool calls concurrently and append results to the message history.

    All tool calls are executed in parallel using structured concurrency with asyncio.gather(),
    then results are appended to messages in the original order.
    
    对于多模态工具调用，会先插入一个 assistant message 说明将使用该工具，
    然后再插入工具结果的 user message。
    """

    if not tool_calls:
        return messages

    # Execute all tool calls concurrently
    tasks = [_execute_single_tool_call(tool_call, tool_map) for tool_call in tool_calls]
    results = await asyncio.gather(*tasks)

    # 分类结果：普通工具调用和多模态工具调用
    normal_results: List[List[Dict[str, Any]]] = []
    multimodal_results: List[tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    multimodal_tool_call_ids: set[str] = set()
    
    for tool_call_dict, messages_to_append, is_multimodal in results:
        if is_multimodal:
            multimodal_results.append((tool_call_dict, messages_to_append))
            tool_call_id = tool_call_dict.get("id")
            if tool_call_id:
                multimodal_tool_call_ids.add(tool_call_id)
        else:
            normal_results.append(messages_to_append)

    # 从原始messages中移除多模态工具调用的tool_calls
    if multimodal_tool_call_ids:
        # 找到最后一个包含tool_calls的assistant message
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                original_tool_calls = msg["tool_calls"]
                # 过滤掉多模态工具调用
                filtered_tool_calls = [
                    tc for tc in original_tool_calls
                    if tc.get("id") not in multimodal_tool_call_ids
                ]
                
                if not filtered_tool_calls:
                    # 如果所有tool_calls都是多模态的，移除tool_calls字段
                    del msg["tool_calls"]
                    # 如果content为None，设置为空字符串
                    if msg.get("content") is None:
                        msg["content"] = ""
                else:
                    # 否则更新为过滤后的tool_calls
                    msg["tool_calls"] = filtered_tool_calls
                break

    # Append普通工具调用结果
    # 创建messages的副本以避免修改原始列表
    current_messages = messages.copy()
    for msgs in normal_results:
        current_messages.extend(msgs)
    
    # 处理多模态工具调用：先插入assistant message，再插入user message
    for tool_call_dict, user_messages in multimodal_results:
        tool_name = tool_call_dict.get("function", {}).get("name", "unknown")
        arguments = tool_call_dict.get("function", {}).get("arguments", "{}")
        
        # 创建assistant message说明将使用该工具
        assistant_message = {
            "role": "assistant",
            "content": f"我将求助用户使用 {tool_name} 工具来获取结果，使用参数为：{arguments}，请用户按照工具的描述和参数要求，提供符合要求的结果。",
        }
        current_messages.append(assistant_message)
        
        # 添加工具返回的user message（包含多模态内容）
        current_messages.extend(user_messages)

    return current_messages

