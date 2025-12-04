"""Step 5: Parse and validate response for llm_function."""

from __future__ import annotations

from typing import Any

from SimpleLLMFunc.base.post_process import process_response


def extract_response_content(response: Any, func_name: str) -> str:
    """从响应对象中提取文本内容"""
    from SimpleLLMFunc.base.post_process import extract_content_from_response

    return extract_content_from_response(response, func_name)


def parse_response_to_type(response: Any, return_type: Any) -> Any:
    """将响应解析为目标返回类型"""
    return process_response(response, return_type)


def parse_and_validate_response(
    response: Any,
    return_type: Any,
    func_name: str,
) -> Any:
    """解析和验证响应的完整流程"""
    return parse_response_to_type(response, return_type)

