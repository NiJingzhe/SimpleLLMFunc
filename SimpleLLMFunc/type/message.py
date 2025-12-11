"""OpenAI API message type definitions.

直接使用 OpenAI SDK 定义的消息类型，确保类型安全。
消息类型结构是固定的：
- role: "system" | "user" | "assistant" | "tool" | "function"
- content: str | List[Dict[str, Any]] | None (取决于 role)
- reasoning_details: 可选字段，某些模型（如 Google Gemini）会返回 reasoning 信息
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, NotRequired, TypeAlias, TypedDict

# 导入 OpenAI SDK 的消息类型
from openai.types.chat.chat_completion_message import ChatCompletionMessage

# Reasoning detail 的类型定义
class ReasoningDetail(TypedDict):
    """推理细节的类型定义（用于 Google Gemini 等模型）"""
    id: str
    format: str
    index: int
    type: Literal["reasoning.encrypted"]
    data: str

# 扩展的消息参数类型，包含所有可能的字段
class ExtendedMessageParam(TypedDict, total=False):
    """扩展的消息参数类型，支持额外的字段如 reasoning_details
    
    包含 ChatCompletionMessage 的所有字段，并添加 reasoning_details 支持
    """
    # 基础字段
    role: str
    content: str | List[Dict[str, Any]] | None
    
    # OpenAI 标准字段
    refusal: NotRequired[str | None]
    annotations: NotRequired[List[Dict[str, Any]] | None]
    audio: NotRequired[Dict[str, Any] | None]
    function_call: NotRequired[Dict[str, Any] | None]
    tool_calls: NotRequired[List[Dict[str, Any]] | None]
    
    # 扩展字段（如 Google Gemini 的 reasoning_details）
    reasoning_details: NotRequired[List[ReasoningDetail]]

# 使用扩展类型作为主要消息类型
MessageParam: TypeAlias = ChatCompletionMessage | ExtendedMessageParam

# 消息列表类型
MessageList: TypeAlias = List[MessageParam]

