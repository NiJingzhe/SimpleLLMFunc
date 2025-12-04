"""Step 5: Process chat response stream."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Tuple

from SimpleLLMFunc.base.post_process import (
    extract_content_from_response,
    extract_content_from_stream_response,
)
from SimpleLLMFunc.logger import app_log
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.type.decorator import HistoryList


def extract_stream_response_content(chunk: Any, func_name: str) -> str:
    """从流式响应 chunk 中提取内容"""
    return extract_content_from_stream_response(chunk, func_name)


def process_single_chat_response(
    response: Any,
    return_mode: Literal["text", "raw"],
    stream: bool,
    func_name: str,
) -> Any:
    """处理单个响应"""
    if return_mode == "raw":
        return response

    # text 模式：提取内容
    if stream:
        return extract_stream_response_content(response, func_name)
    else:
        return extract_content_from_response(response, func_name) or ""


async def process_chat_response_stream(
    response_stream: AsyncGenerator[Any, None],
    return_mode: Literal["text", "raw"],
    messages: List[Dict[str, Any]],
    func_name: str,
    stream: bool,
) -> AsyncGenerator[Tuple[Any, HistoryList], None]:
    """处理流式响应的完整流程"""
    async for response in response_stream:
        # 记录响应日志
        app_log(
            f"LLM Chat '{func_name}' received response:"
            f"\n{json.dumps(response, default=str, ensure_ascii=False, indent=2)}",
            location=get_location(),
        )

        # 处理单个响应
        content = process_single_chat_response(
            response,
            return_mode,
            stream,
            func_name,
        )

        # Yield 响应和当前历史
        yield content, messages

    # 流结束标记（text 模式）
    if return_mode == "text":
        yield "", messages

