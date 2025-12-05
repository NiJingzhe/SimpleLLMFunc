"""OpenAI API message type definitions.

直接使用 OpenAI SDK 定义的消息类型，确保类型安全。
消息类型结构是固定的：
- role: "system" | "user" | "assistant" | "tool" | "function"
- content: str | List[Dict[str, Any]] | None (取决于 role)
"""

from __future__ import annotations

from typing import List, TypeAlias

# 导入 OpenAI SDK 的消息类型
# 这是 OpenAI SDK 定义的 TypedDict Union 类型，包含了所有可能的消息格式
from openai.types.chat import ChatCompletionMessageParam

# 使用 OpenAI SDK 的实际类型定义
MessageParam: TypeAlias = ChatCompletionMessageParam

# 消息列表类型
MessageList: TypeAlias = List[MessageParam]

