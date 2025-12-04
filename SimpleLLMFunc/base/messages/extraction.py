"""Response information extraction helpers."""

from __future__ import annotations

from typing import Dict, Optional, Union

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


def extract_usage_from_response(
    response: Union[ChatCompletion, ChatCompletionChunk, None],
) -> Dict[str, int] | None:
    """从LLM响应中提取用量信息。

    Args:
        response: OpenAI API的ChatCompletion或ChatCompletionChunk响应对象

    Returns:
        包含用量信息的字典 {"input": int, "output": int, "total": int}，
        如果无法提取则返回None
    """
    if response is None:
        return None

    try:
        if hasattr(response, "usage") and response.usage:
            return {
                "input": getattr(response.usage, "prompt_tokens", 0),
                "output": getattr(response.usage, "completion_tokens", 0),
                "total": getattr(response.usage, "total_tokens", 0),
            }
    except (AttributeError, TypeError):
        pass
    return None

