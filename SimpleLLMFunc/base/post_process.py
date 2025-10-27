"""LLM response post-processing helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar, cast

from SimpleLLMFunc.logger import push_debug, push_error, push_warning
from SimpleLLMFunc.logger.logger import get_current_context_attribute, get_location

T = TypeVar("T")


def process_response(response: Any, return_type: Optional[Type[T]]) -> T:
    """Convert an LLM response into the expected return type."""

    func_name = get_current_context_attribute("function_name") or "Unknown Function"
    content = extract_content_from_response(response, func_name)

    if content is None:
        content = ""

    if return_type is None or return_type is str:
        return cast(T, content)

    if return_type in (int, float, bool):
        return cast(T, _convert_to_primitive_type(content, return_type))

    if return_type is dict or getattr(return_type, "__origin__", None) is dict:
        return cast(T, _convert_to_dict(content, func_name))

    if return_type and hasattr(return_type, "model_validate_json"):
        return cast(T, _convert_to_pydantic_model(content, return_type, func_name))

    try:
        return cast(T, content)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"无法将 LLM 响应转换为所需类型: {content}") from exc


def extract_content_from_response(response: Any, func_name: str) -> str:
    """Extract textual content from a normal LLM response."""

    content = ""
    try:
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            if message and hasattr(message, "content") and message.content is not None:
                content = message.content
            else:
                content = ""
        else:
            push_error(
                f"LLM 函数 '{func_name}': 未知响应格式: {type(response)}，将直接转换为字符串",
                location=get_location(),
            )
            content = ""
    except Exception as exc:
        push_error(f"提取响应内容时出错: {str(exc)}")
        content = ""

    push_debug(f"LLM 函数 '{func_name}' 提取的内容:\n{content}")
    return content


def extract_content_from_stream_response(chunk: Any, func_name: str) -> str:
    """Extract textual content from a streaming LLM chunk."""

    content = ""
    if not chunk:
        push_warning(
            f"LLM 函数 '{func_name}': 检测到空的流响应 chunk，返回空字符串",
            location=get_location(),
        )
        return content
    try:
        if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                delta = choice.delta
                if hasattr(delta, "content") and delta.content is not None:
                    content = delta.content
                else:
                    content = ""
            else:
                content = ""
        else:
            push_debug(
                f"LLM 函数 '{func_name}': 检测到流响应格式: {type(chunk)}，内容为: {chunk}，预估不包含content，将会返回空串",
                location=get_location(),
            )
            content = ""
    except Exception as exc:
        push_error(f"提取流响应内容时出错: {str(exc)}")
        content = ""

    return content


def _convert_to_primitive_type(content: str, return_type: Type) -> Any:
    """Cast textual content to primitive Python types."""

    try:
        if return_type is int:
            return int(content.strip())
        if return_type is float:
            return float(content.strip())
        if return_type is bool:
            return content.strip().lower() in ("true", "yes", "1")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"无法将 LLM 响应 '{content}' 转换为 {return_type.__name__} 类型"
        ) from exc
    raise ValueError(f"不支持的基本类型转换: {return_type}")


def _convert_to_dict(content: str, func_name: str) -> Dict[str, Any]:
    """Parse textual content into a JSON dictionary."""

    try:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, content)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)

            cleaned_content = content.strip()
            if cleaned_content.startswith("```") and cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[3:-3].strip()
            return json.loads(cleaned_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法将 LLM 响应解析为有效的 JSON: {content}") from exc


def _convert_to_pydantic_model(content: str, model_class: Type, func_name: str) -> Any:
    """Parse textual content into a Pydantic model instance."""

    try:
        if content.strip():
            try:
                parsed_content = json.loads(content)
                clean_json_str = json.dumps(parsed_content)
                return model_class.model_validate_json(clean_json_str)
            except json.JSONDecodeError:
                json_pattern = r"```json\s*([\s\S]*?)\s*```"
                match = re.search(json_pattern, content)
                if match:
                    json_str = match.group(1)
                    parsed_json = json.loads(json_str)
                    clean_json_str = json.dumps(parsed_json)
                    return model_class.model_validate_json(clean_json_str)
                return model_class.model_validate_json(content)
        raise ValueError("收到空响应")
    except Exception as exc:
        push_error(f"解析错误详情: {str(exc)}, 内容: {content}")
        raise ValueError(f"无法解析为 Pydantic 模型: {str(exc)}") from exc


__all__ = [
    "process_response",
    "extract_content_from_response",
    "extract_content_from_stream_response",
]
