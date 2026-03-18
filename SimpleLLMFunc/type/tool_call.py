"""工具调用相关的类型定义

优先使用 OpenAI SDK 类型，最小化自定义类型。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, TypeAlias, TypedDict
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function as OpenAIFunction,
)

# ============================================================================
# 工具调用类型（直接使用 OpenAI SDK）
# ============================================================================

# 工具调用对象（直接使用 OpenAI 类型）
ToolCall: TypeAlias = ChatCompletionMessageFunctionToolCall
"""
工具调用对象，直接使用 OpenAI SDK 类型

属性:
    - id: str - 工具调用 ID
    - type: str - 类型（通常为 "function"）
    - function: Function - 函数信息
        - name: str - 函数名称
        - arguments: str - JSON 字符串格式的参数
"""

# 工具调用函数信息（直接使用 OpenAI 类型）
ToolCallFunction: TypeAlias = OpenAIFunction
"""
工具调用的函数信息

属性:
    - name: str - 函数名称
    - arguments: str - JSON 字符串格式的参数
"""

# 工具调用参数（JSON 解析后）
ToolCallArguments = Dict[str, Any]
"""工具调用的参数（JSON 解析后的字典）"""

# ============================================================================
# 工具定义类型（OpenAI Tool Schema）
# ============================================================================


class ToolFunctionDefinition(TypedDict):
    """OpenAI Tool Function 定义 Schema"""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 格式


class ToolDefinition(TypedDict):
    """OpenAI Tool 定义 Schema"""

    type: Literal["function"]
    function: ToolFunctionDefinition


ToolDefinitionList = Optional[List[ToolDefinition]]
"""工具定义列表类型（可选）"""

# ============================================================================
# 内部使用类型（用于流式累积）
# ============================================================================


class ToolCallFunctionInfo(TypedDict):
    """
    内部使用：流式响应中累积工具调用时的临时结构

    注意：这不是最终的工具调用类型，最终会转换为 ToolCall
    """

    name: Optional[str]
    arguments: str  # 逐步累积的 JSON 字符串


class AccumulatedToolCall(TypedDict):
    """
    内部使用：流式响应中累积的工具调用

    注意：这不是最终的工具调用类型，最终会转换为 ToolCall
    """

    id: Optional[str]
    type: Optional[str]
    function: ToolCallFunctionInfo


# ============================================================================
# 类型转换辅助函数
# ============================================================================


def dict_to_tool_call(data: Dict[str, Any]) -> ToolCall:
    """
    将字典转换为 ToolCall 对象

    Args:
        data: 字典格式的工具调用数据

    Returns:
        OpenAI SDK 的函数工具调用对象
    """
    return ChatCompletionMessageFunctionToolCall(
        id=data["id"],
        type=data.get("type", "function"),
        function=OpenAIFunction(
            name=data["function"]["name"],
            arguments=data["function"]["arguments"],
        ),
    )


def tool_call_to_dict(tool_call: ToolCall) -> Dict[str, Any]:
    """
    将 ToolCall 对象转换为字典

    Args:
        tool_call: OpenAI SDK 的工具调用对象

    Returns:
        字典格式的工具调用数据
    """
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


__all__ = [
    # 主要类型
    "ToolCall",
    "ToolCallFunction",
    "ToolCallArguments",
    "ToolDefinition",
    "ToolFunctionDefinition",
    "ToolDefinitionList",
    # 内部类型
    "ToolCallFunctionInfo",
    "AccumulatedToolCall",
    # 辅助函数
    "dict_to_tool_call",
    "tool_call_to_dict",
]
