"""Step 2: Setup log context."""

from __future__ import annotations

import json
from typing import Any, AsyncContextManager, Dict

from SimpleLLMFunc.logger import app_log, async_log_context
from SimpleLLMFunc.logger.logger import get_location


def log_function_call(func_name: str, arguments: Dict[str, Any]) -> None:
    """记录函数调用日志"""
    args_str = json.dumps(arguments, default=str, ensure_ascii=False, indent=4)
    app_log(
        f"Async LLM function '{func_name}' called with arguments: {args_str}",
        location=get_location(),
    )


def create_log_context_manager(
    func_name: str, trace_id: str
) -> AsyncContextManager[None]:
    """创建日志上下文管理器"""
    return async_log_context(
        trace_id=trace_id,
        function_name=func_name,
        input_tokens=0,
        output_tokens=0,
    )


def setup_log_context(
    func_name: str,
    trace_id: str,
    arguments: Dict[str, Any],
) -> AsyncContextManager[None]:
    """设置日志上下文的完整流程"""
    # 1. 记录函数调用日志
    log_function_call(func_name, arguments)

    # 2. 创建并返回日志上下文管理器
    return create_log_context_manager(func_name, trace_id)

