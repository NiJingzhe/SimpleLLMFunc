"""Step 3: Build initial prompts for llm_function."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from SimpleLLMFunc.base.messages import build_multimodal_content
from SimpleLLMFunc.base.type_resolve.multimodal import has_multimodal_content
from SimpleLLMFunc.base.type_resolve.description import (
    get_detailed_type_description,
)
from SimpleLLMFunc.logger import push_debug
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.llm_decorator.steps.common.prompt import (
    extract_parameter_type_hints,
    process_docstring_template,
)
from SimpleLLMFunc.llm_decorator.steps.common.types import FunctionSignature

# Default prompt templates
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
Your task is to provide results that meet the requirements based on the **function description** 
and the user's request.

- Function Description:
    {function_description}

- You will receive the following parameters:
    {parameters_description}

- The type of content you need to return:
    {return_type_description}

Execution Requirements:
1. Use available tools to assist in completing the task if needed
2. Do not wrap results in markdown format or code blocks; directly output the expected content or JSON representation
"""

DEFAULT_USER_PROMPT_TEMPLATE = """
The parameters provided are:
    {parameters}

Return the result directly without any explanation or formatting.
"""


def build_parameter_type_descriptions(
    param_type_hints: Dict[str, Any],
) -> List[str]:
    """构建参数类型描述列表"""
    descriptions = []
    for param_name, param_type in param_type_hints.items():
        type_str = (
            get_detailed_type_description(param_type)
            if param_type
            else "Unknown Type"
        )
        descriptions.append(f"  - {param_name}: {type_str}")
    return descriptions


def build_return_type_description(return_type: Any) -> str:
    """构建返回类型描述"""
    if return_type is None:
        return "未知类型"
    return get_detailed_type_description(return_type)


def build_text_messages(
    processed_docstring: str,
    param_type_descriptions: List[str],
    return_type_description: str,
    arguments: Dict[str, Any],
    system_template: str,
    user_template: str,
) -> List[Dict[str, str]]:
    """构建文本消息列表"""
    # 构建 system prompt
    system_prompt = system_template.format(
        function_description=processed_docstring,
        parameters_description="\n".join(param_type_descriptions),
        return_type_description=return_type_description,
    )

    # 构建 user prompt
    user_param_values = [
        f"  - {param_name}: {param_value}"
        for param_name, param_value in arguments.items()
    ]
    user_prompt = user_template.format(parameters="\n".join(user_param_values))

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    push_debug(f"System prompt: {system_prompt}", location=get_location())
    push_debug(f"User prompt: {user_prompt}", location=get_location())

    return messages


def build_multimodal_messages(
    system_prompt: str,
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """构建多模态消息列表"""
    # 构建多模态用户消息内容
    user_content = build_multimodal_content(arguments, type_hints)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    push_debug(f"System prompt: {system_prompt}", location=get_location())
    push_debug(
        f"Multimodal user message contains {len(user_content)} content blocks",
        location=get_location(),
    )

    return messages


def build_initial_prompts(
    signature: FunctionSignature,
    system_prompt_template: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    template_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """构建初始提示的完整流程"""
    # 1. 处理 docstring 模板参数
    processed_docstring = process_docstring_template(
        signature.docstring,
        template_params,
    )

    # 2. 提取参数类型提示
    param_type_hints = extract_parameter_type_hints(signature.type_hints)

    # 3. 构建参数类型描述
    param_type_descriptions = build_parameter_type_descriptions(param_type_hints)

    # 4. 构建返回类型描述
    return_type_description = build_return_type_description(signature.return_type)

    # 5. 检查多模态内容
    has_multimodal = has_multimodal_content(
        signature.bound_args.arguments,
        signature.type_hints,
    )

    # 6. 选择模板
    system_template = system_prompt_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE
    user_template = user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE

    # 7. 构建消息列表
    if has_multimodal:
        # 先构建文本 system prompt
        text_messages = build_text_messages(
            processed_docstring,
            param_type_descriptions,
            return_type_description,
            signature.bound_args.arguments,
            system_template,
            user_template,
        )
        system_prompt = text_messages[0]["content"]

        # 构建多模态消息
        messages = build_multimodal_messages(
            system_prompt,
            signature.bound_args.arguments,
            signature.type_hints,
        )
    else:
        # 构建文本消息
        messages = build_text_messages(
            processed_docstring,
            param_type_descriptions,
            return_type_description,
            signature.bound_args.arguments,
            system_template,
            user_template,
        )

    return messages

