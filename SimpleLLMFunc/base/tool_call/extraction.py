"""Tool call extraction helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Literal

from SimpleLLMFunc.logger import push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location


class ReasoningDetail(TypedDict):
    """推理细节类型（如 Google Gemini 的 reasoning_details）"""
    id: str
    format: str
    index: int
    type: Literal["reasoning.encrypted"]
    data: str


class ToolCallFunctionInfo(TypedDict):
    name: Optional[str]
    arguments: str


class AccumulatedToolCall(TypedDict):
    id: Optional[str]
    type: Optional[str]
    function: ToolCallFunctionInfo


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


def extract_reasoning_details(response: Any) -> List[ReasoningDetail]:
    """从非流式响应中提取 reasoning_details。
    
    Args:
        response: LLM 响应对象
        
    Returns:
        reasoning_details 列表
    """
    reasoning_details: List[ReasoningDetail] = []
    
    try:
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            
            # 尝试从 message 中获取 reasoning_details
            reasoning_details_raw = None
            if hasattr(message, "reasoning_details"):
                reasoning_details_raw = message.reasoning_details
            elif isinstance(message, dict) and "reasoning_details" in message:
                reasoning_details_raw = message["reasoning_details"]
            
            if reasoning_details_raw:
                for detail in reasoning_details_raw:
                    # 处理 detail 可能是字典或对象的情况
                    if isinstance(detail, dict):
                        reasoning_details.append(
                            ReasoningDetail(
                                id=detail.get("id", ""),
                                format=detail.get("format", ""),
                                index=detail.get("index", 0),
                                type=detail.get("type", "reasoning.encrypted"),
                                data=detail.get("data", ""),
                            )
                        )
                    else:
                        # detail 是对象，尝试获取属性
                        reasoning_details.append(
                            ReasoningDetail(
                                id=getattr(detail, "id", "") if hasattr(detail, "id") else "",
                                format=getattr(detail, "format", "") if hasattr(detail, "format") else "",
                                index=getattr(detail, "index", 0) if hasattr(detail, "index") else 0,
                                type=getattr(detail, "type", "reasoning.encrypted") if hasattr(detail, "type") else "reasoning.encrypted",
                                data=getattr(detail, "data", "") if hasattr(detail, "data") else "",
                            )
                        )
    except Exception as exc:
        push_error(f"提取 reasoning_details 时出错: {str(exc)}")
        import traceback
        push_error(f"详细错误: {traceback.format_exc()}")
    
    return reasoning_details


def extract_reasoning_details_from_stream(chunk: Any) -> List[ReasoningDetail]:
    """从流式响应 chunk 中提取 reasoning_details。
    
    Args:
        chunk: 流式响应的一个 chunk
        
    Returns:
        reasoning_details 列表
    """
    reasoning_details: List[ReasoningDetail] = []
    
    try:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                delta = choice.delta
                
                # 尝试从 delta 中获取 reasoning_details
                reasoning_details_raw = None
                if hasattr(delta, "reasoning_details"):
                    reasoning_details_raw = delta.reasoning_details
                elif isinstance(delta, dict) and "reasoning_details" in delta:
                    reasoning_details_raw = delta["reasoning_details"]
                
                if reasoning_details_raw:
                    for detail in reasoning_details_raw:
                        # 处理 detail 可能是字典或对象的情况
                        if isinstance(detail, dict):
                            reasoning_details.append(
                                ReasoningDetail(
                                    id=detail.get("id", ""),
                                    format=detail.get("format", ""),
                                    index=detail.get("index", 0),
                                    type=detail.get("type", "reasoning.encrypted"),
                                    data=detail.get("data", ""),
                                )
                            )
                        else:
                            # detail 是对象，尝试获取属性
                            reasoning_details.append(
                                ReasoningDetail(
                                    id=getattr(detail, "id", "") if hasattr(detail, "id") else "",
                                    format=getattr(detail, "format", "") if hasattr(detail, "format") else "",
                                    index=getattr(detail, "index", 0) if hasattr(detail, "index") else 0,
                                    type=getattr(detail, "type", "reasoning.encrypted") if hasattr(detail, "type") else "reasoning.encrypted",
                                    data=getattr(detail, "data", "") if hasattr(detail, "data") else "",
                                )
                            )
    except Exception as exc:
        push_error(f"提取流式 reasoning_details 时出错: {str(exc)}")
        import traceback
        push_error(f"详细错误: {traceback.format_exc()}")
    
    return reasoning_details

