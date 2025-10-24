"""
工具处理公用模块

此模块提供统一的工具解析和处理逻辑，供 llm_chat_decorator 和 llm_function_decorator 使用。
"""

import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from SimpleLLMFunc.logger import get_location, push_debug, push_warning
from SimpleLLMFunc.tool import Tool


def process_tools(
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]] = None,
    func_name: str = "unknown_function",
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Callable[..., Awaitable[Any]]]]:
    """
    处理工具列表，返回 API 所需的工具参数和工具映射。

    此函数是 llm_chat_decorator 和 llm_function_decorator 中工具处理逻辑的统一实现。
    统一将所有 @tool 装饰的函数映射到 tool_obj.run 方法。

    ## 工具类型支持
    - `Tool` 对象：直接使用，要求 `run` 方法为 `async` 函数
    - `@tool` 装饰的异步函数：会被转换为相应的工具映射

    ## 处理流程
    1. 验证输入工具列表
    2. 遍历工具，区分 Tool 对象和 @tool 装饰的函数
    3. 检查类型合法性（必须为 async）
    4. 构建工具对象列表和工具名称到函数的映射
    5. 序列化工具以供 LLM API 使用
    6. 返回序列化的工具参数和工具映射字典

    Args:
        toolkit: 工具列表，可以是 Tool 对象或被 @tool 装饰的异步函数，为 None 或空列表时返回 (None, {})
        func_name: 函数名称，用于日志记录和错误信息

    Returns:
        (tool_param_for_api, tool_map) 元组：
            - tool_param_for_api: 序列化后的工具参数列表，供 LLM API 使用，如果无工具则为 None
            - tool_map: 工具名称到异步函数的映射字典，用于工具调用时的查找

    Raises:
        TypeError: 当工具的 run 方法或被装饰的函数不是 async 时抛出

    Examples:
        ```python
        from SimpleLLMFunc.tool import Tool

        # 示例1：使用 Tool 对象
        my_tool = Tool(name="get_weather", ...)
        tool_param, tool_map = process_tools([my_tool], "my_func")

        # 示例2：使用 @tool 装饰的函数
        @tool(name="calculate")
        async def calculate(a: int, b: int) -> int:
            return a + b

        tool_param, tool_map = process_tools([calculate], "my_func")

        # 示例3：混合使用
        tools = [my_tool, calculate]
        tool_param, tool_map = process_tools(tools, "my_func")
        ```

    Note:
        - 所有工具的 run 方法或函数本体必须是异步的（async）
        - 工具名称通过 Tool.name 属性获取
        - 序列化使用 Tool.serialize_tools() 方法
        - 所有 @tool 装饰的函数都统一映射到其 tool_obj.run 方法
    """
    if not toolkit:
        return None, {}

    tool_objects: List[Union[Tool, Callable[..., Awaitable[Any]]]] = []
    tool_map: Dict[str, Callable[..., Awaitable[Any]]] = {}

    for tool in toolkit:
        if isinstance(tool, Tool):
            # 处理 Tool 对象
            _process_tool_object(tool, func_name, tool_objects, tool_map)
        elif callable(tool) and hasattr(tool, "_tool"):
            # 处理 @tool 装饰的函数
            _process_decorated_function(tool, func_name, tool_objects, tool_map)
        else:
            push_warning(
                f"LLM 函数 '{func_name}': 不支持的工具类型 {type(tool)}，"
                "工具必须是 Tool 对象或被 @tool 装饰的函数",
                location=get_location(),
            )

    # 序列化工具以供 LLM API 使用
    tool_param_for_api: Optional[List[Dict[str, Any]]] = (
        Tool.serialize_tools(tool_objects) if tool_objects else None
    )

    push_debug(
        f"LLM 函数 '{func_name}' 加载了 {len(tool_objects)} 个工具",
        location=get_location(),
    )

    return tool_param_for_api, tool_map


def _process_tool_object(
    tool: Tool,
    func_name: str,
    tool_objects: List[Union[Tool, Callable[..., Awaitable[Any]]]],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
) -> None:
    """
    处理 Tool 对象。

    Args:
        tool: Tool 实例
        func_name: 函数名，用于日志记录
        tool_objects: 工具对象列表（会被修改）
        tool_map: 工具名称到函数的映射（会被修改）

    Raises:
        TypeError: 当工具的 run 方法不是 async 时抛出
    """
    if not inspect.iscoroutinefunction(tool.run):
        raise TypeError(
            f"LLM 函数 '{func_name}': Tool '{tool.name}' 必须实现 async run 方法"
        )
    tool_objects.append(tool)
    tool_map[tool.name] = tool.run


def _process_decorated_function(
    tool: Callable[..., Awaitable[Any]],
    func_name: str,
    tool_objects: List[Union[Tool, Callable[..., Awaitable[Any]]]],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
) -> None:
    """
    处理被 @tool 装饰的函数。

    统一将被 @tool 装饰的函数映射到其 tool_obj.run 方法。

    Args:
        tool: 被 @tool 装饰的异步函数
        func_name: 函数名，用于日志记录
        tool_objects: 工具对象列表（会被修改）
        tool_map: 工具名称到函数的映射（会被修改）

    Raises:
        TypeError: 当函数不是 async 时抛出
    """
    if not inspect.iscoroutinefunction(tool):
        raise TypeError(
            f"LLM 函数 '{func_name}': 被 @tool 装饰的函数 '{tool.__name__}' 必须是 async 函数"
        )

    tool_obj = getattr(tool, "_tool", None)
    assert isinstance(
        tool_obj, Tool
    ), "这一定是一个Tool对象，不会是None！是None我赤石"

    # 添加 Tool 对象到列表（用于序列化）
    tool_objects.append(tool_obj)

    # 统一映射到 tool_obj.run
    tool_map[tool_obj.name] = tool_obj.run
