"""Step 1: Parse function signature."""

from __future__ import annotations

import inspect
import uuid
from typing import Any, Callable, Dict, Optional, Tuple, get_type_hints

from SimpleLLMFunc.logger.logger import get_current_trace_id
from SimpleLLMFunc.llm_decorator.steps.common.types import FunctionSignature


def extract_template_params(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 kwargs 中提取模板参数"""
    return kwargs.pop("_template_params", None)


def extract_function_metadata(
    func: Callable,
) -> Tuple[inspect.Signature, Dict[str, Any], Any, str, str]:
    """提取函数的元数据"""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return")
    docstring = func.__doc__ or ""
    func_name = func.__name__

    return signature, type_hints, return_type, docstring, func_name


def generate_trace_id(func_name: str) -> str:
    """生成唯一的追踪 ID"""
    context_trace_id = get_current_trace_id()
    current_trace_id = f"{func_name}_{uuid.uuid4()}"
    if context_trace_id:
        current_trace_id += f"_{context_trace_id}"
    return current_trace_id


def bind_function_arguments(
    signature: inspect.Signature,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> inspect.BoundArguments:
    """绑定函数参数并应用默认值"""
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args


def build_function_signature(
    func_name: str,
    trace_id: str,
    bound_args: inspect.BoundArguments,
    signature: inspect.Signature,
    type_hints: Dict[str, Any],
    return_type: Any,
    docstring: str,
) -> FunctionSignature:
    """构建函数签名对象"""
    return FunctionSignature(
        func_name=func_name,
        trace_id=trace_id,
        bound_args=bound_args,
        signature=signature,
        type_hints=type_hints,
        return_type=return_type,
        docstring=docstring,
    )


def parse_function_signature(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[FunctionSignature, Optional[Dict[str, Any]]]:
    """解析函数签名的完整流程"""
    # 1. 提取模板参数
    template_params = extract_template_params(kwargs)

    # 2. 提取函数元数据
    signature, type_hints, return_type, docstring, func_name = extract_function_metadata(
        func
    )

    # 3. 生成追踪 ID
    trace_id = generate_trace_id(func_name)

    # 4. 绑定函数参数
    bound_args = bind_function_arguments(signature, args, kwargs)

    # 5. 构建函数签名对象
    function_signature = build_function_signature(
        func_name=func_name,
        trace_id=trace_id,
        bound_args=bound_args,
        signature=signature,
        type_hints=type_hints,
        return_type=return_type,
        docstring=docstring,
    )

    return function_signature, template_params

