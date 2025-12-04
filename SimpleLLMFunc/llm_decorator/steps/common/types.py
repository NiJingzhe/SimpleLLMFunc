"""Internal type definitions for decorator steps."""

from typing import Any, Dict, NamedTuple
import inspect


class FunctionSignature(NamedTuple):
    """函数签名信息（内部使用）"""

    func_name: str
    trace_id: str
    bound_args: inspect.BoundArguments
    signature: inspect.Signature
    type_hints: Dict[str, Any]
    return_type: Any
    docstring: str

