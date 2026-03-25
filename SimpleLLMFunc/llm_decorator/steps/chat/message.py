"""Step 3: Build chat messages for llm_chat."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from SimpleLLMFunc.base.messages import build_multimodal_content
from SimpleLLMFunc.base.type_resolve.multimodal import has_multimodal_content
from SimpleLLMFunc.logger import push_warning
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.type import HistoryList
from SimpleLLMFunc.llm_decorator.steps.common.types import FunctionSignature
from SimpleLLMFunc.llm_decorator.utils import process_tools

# Constants
HISTORY_PARAM_NAMES: List[str] = ["history", "chat_history"]


def extract_conversation_history(
    arguments: Dict[str, Any],
    func_name: str,
    history_param_names: Optional[List[str]] = None,
) -> Optional[HistoryList]:
    """提取并验证对话历史"""
    if history_param_names is None:
        history_param_names = HISTORY_PARAM_NAMES

    # 查找历史参数
    history_param_name = None
    for param_name in history_param_names:
        if param_name in arguments:
            history_param_name = param_name
            break

    if not history_param_name:
        push_warning(
            f"LLM Chat '{func_name}' missing history parameter "
            f"(parameter name should be one of {history_param_names}). "
            "History will not be passed.",
            location=get_location(),
        )
        return None

    custom_history = arguments[history_param_name]

    # 验证历史格式
    if not (
        isinstance(custom_history, list)
        and all(isinstance(item, dict) for item in custom_history)
    ):
        push_warning(
            f"LLM Chat '{func_name}' history parameter should be List[Dict[str, str]] type. "
            "History will not be passed.",
            location=get_location(),
        )
        return None

    return custom_history


def build_chat_user_message_content(
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    has_multimodal: bool,
    exclude_params: List[str],
) -> Union[str, List[Dict[str, Any]]]:
    """构建用户消息内容"""
    if has_multimodal:
        return build_multimodal_content(
            arguments,
            type_hints,
            exclude_params=exclude_params,
        )
    else:
        # 构建文本消息，排除历史参数
        message_parts = [
            f"{param_name}: {param_value}"
            for param_name, param_value in arguments.items()
            if param_name not in exclude_params
        ]
        return "\n\t".join(message_parts)


def build_chat_system_prompt(
    docstring: str,
    history_system_prompt: Optional[str] = None,
) -> Optional[str]:
    """构建聊天系统提示"""
    base_prompt = history_system_prompt if history_system_prompt else docstring

    if not base_prompt:
        return None

    return base_prompt


def extract_history_system_prompt(history: Optional[HistoryList]) -> Optional[str]:
    """Extract latest valid system prompt from history list."""

    if not history:
        return None

    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "system":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            return content

    return None


def filter_history_messages(
    history: HistoryList,
    func_name: str,
) -> HistoryList:
    """过滤历史消息，排除 system 消息"""
    filtered = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            if msg["role"] not in ["system"]:
                filtered.append(msg)
        else:
            push_warning(
                f"Skipping malformed history item: {msg}",
                location=get_location(),
            )
    return filtered


def build_chat_messages(
    signature: FunctionSignature,
    toolkit: Optional[List[Union[Tool, Any]]],
    exclude_params: List[str],
) -> HistoryList:
    """构建聊天消息列表的完整流程"""
    messages: HistoryList = []

    # 1. 准备工具
    _tool_param, _tool_map = process_tools(toolkit, signature.func_name)

    # 2. 提取对话历史
    custom_history = extract_conversation_history(
        signature.bound_args.arguments,
        signature.func_name,
    )

    # 3. 构建系统提示（history 中的 system prompt 优先）
    system_content = build_chat_system_prompt(
        signature.docstring,
        history_system_prompt=extract_history_system_prompt(custom_history),
    )
    if system_content:
        messages.append({"role": "system", "content": system_content})

    # 4. 过滤并添加非 system 历史消息
    if custom_history:
        filtered_history = filter_history_messages(custom_history, signature.func_name)
        messages.extend(filtered_history)

    # 5. 检查多模态内容
    has_multimodal = has_multimodal_content(
        signature.bound_args.arguments,
        signature.type_hints,
        exclude_params=exclude_params,
    )

    # 6. 构建用户消息内容
    user_message_content = build_chat_user_message_content(
        signature.bound_args.arguments,
        signature.type_hints,
        has_multimodal,
        exclude_params,
    )

    # 7. 添加用户消息
    if user_message_content:
        messages.append({"role": "user", "content": user_message_content})

    return messages
