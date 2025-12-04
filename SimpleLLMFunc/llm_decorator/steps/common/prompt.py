"""Shared prompt processing helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from SimpleLLMFunc.logger import push_warning
from SimpleLLMFunc.logger.logger import get_location


def process_docstring_template(
    docstring: str, template_params: Optional[Dict[str, Any]]
) -> str:
    """处理 docstring 模板参数替换"""
    if not template_params:
        return docstring

    try:
        return docstring.format(**template_params)
    except KeyError as e:
        push_warning(
            f"DocString template parameter substitution failed: missing parameter {e}. "
            "Using original DocString.",
            location=get_location(),
        )
        return docstring
    except Exception as e:
        push_warning(
            f"Error during DocString template parameter substitution: {str(e)}. "
            "Using original DocString.",
            location=get_location(),
        )
        return docstring


def extract_parameter_type_hints(type_hints: Dict[str, Any]) -> Dict[str, Any]:
    """提取参数类型提示（排除返回类型）"""
    return {k: v for k, v in type_hints.items() if k != "return"}

