"""Tool call extraction helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from SimpleLLMFunc.logger import push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location


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

