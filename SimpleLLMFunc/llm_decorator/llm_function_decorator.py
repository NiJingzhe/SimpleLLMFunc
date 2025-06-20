"""
LLM 函数装饰器模块

本模块提供了 LLM 函数装饰器，可以将普通 Python 函数的执行委托给大语言模型。
使用此装饰器，只需要定义函数签名（参数和返回类型），然后在文档字符串中描述函数的执行策略即可。

数据流程:
1. 用户定义函数签名和文档字符串
2. 装饰器捕获函数调用，提取参数和类型信息
3. 构建系统提示和用户提示
4. 调用 LLM 进行推理
5. 处理工具调用（如有必要）
6. 将 LLM 响应转换为指定的返回类型
7. 返回结果给调用者

示例:
```python
@llm_function(llm_interface=my_llm)
def generate_summary(text: str) -> str:
    \"\"\"根据输入文本生成一个简洁的摘要，应该包含文本的主要观点。\"\"\"
    pass
```
"""

import inspect
from functools import wraps
import json
import asyncio
from typing import (
    List,
    Callable,
    TypeVar,
    Dict,
    Any,
    cast,
    get_type_hints,
    Optional,
    Union,
    Tuple,
    NamedTuple,
    Awaitable,
)

import uuid

from SimpleLLMFunc.logger.logger import push_debug
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import (
    app_log,
    push_warning,
    push_error,
    get_location,
    log_context,
    get_current_trace_id,
    async_log_context,
)

# 从utils模块导入工具函数
from SimpleLLMFunc.llm_decorator.utils import (
    execute_llm,
    process_response,
    get_detailed_type_description,
    has_multimodal_content,
    build_multimodal_content,
)

from SimpleLLMFunc.utils import (
    get_last_item_of_async_generator,
)
from SimpleLLMFunc.llm_decorator.utils import extract_content_from_response

# 定义一个类型变量，用于函数的返回类型
T = TypeVar("T")


# ===== 数据结构定义 =====


class FunctionCallContext(NamedTuple):
    """函数调用上下文信息"""

    func_name: str
    trace_id: str
    bound_args: Any  # inspect.BoundArguments
    signature: Any  # inspect.Signature
    type_hints: Dict[str, Any]
    return_type: Any
    docstring: str


class LLMCallParams(NamedTuple):
    """LLM 调用参数"""

    messages: List[Dict[str, Any]]
    tool_param: Optional[List[Dict[str, Any]]]  # 修正为正确的类型
    tool_map: Dict[str, Callable]
    llm_kwargs: Dict[str, Any]


def llm_function(
    llm_interface: LLM_Interface,
    toolkit: Optional[List[Union[Tool, Callable]]] = None,
    max_tool_calls: int = 5,  # 最大工具调用次数，防止无限循环
    system_prompt_template: Optional[str] = None,  # 自定义系统提示模板
    user_prompt_template: Optional[str] = None,  # 自定义用户提示模板
    **llm_kwargs,  # 额外的关键字参数，将直接传递给LLM接口
):
    """
    LLM 函数装饰器，将函数的执行委托给大语言模型。

    ## 使用方法
    1. 定义一个函数，包括参数和返回类型标注
    2. 在函数的文档字符串中描述函数的功能或执行策略
    3. 使用 @llm_function 装饰该函数

    ## 参数传递流程
    1. 当用户调用被装饰的函数时，装饰器捕获所有实际参数
    2. 这些参数被格式化为文本提示，发送给 LLM
    3. 函数的文档字符串作为系统提示，指导 LLM 如何执行
    4. 每个参数名称和值都作为用户提示的一部分

    ## 工具使用
    - 如果提供了工具集 (toolkit)，LLM 可以使用这些工具来辅助完成任务
    - 支持 Tool 对象或被 @tool 装饰的函数作为工具

    ## 自定义提示模板
    自定义模板中必须包含以下占位符：

    ### 系统提示模板 (system_prompt_template)
    - `{function_description}`: 函数文档字符串内容
    - `{parameters_description}`: 函数参数及其类型的描述
    - `{return_type_description}`: 返回值类型的描述

    ### 用户提示模板 (user_prompt_template)
    - `{parameters}`: 格式化后的参数名称和值

    如果不提供自定义模板，则使用默认模板。自定义模板必须包含上述占位符，否则格式化时会出错。

    ## 返回值处理
    - LLM 的响应会根据函数声明的返回类型进行转换
    - 支持基本类型 (str, int, float, bool)、字典和 Pydantic 模型

    ## LLM接口参数
    - 通过`**llm_kwargs`传递的参数将直接传递给LLM接口
    - 可以用于设置temperature、top_p等模型参数，而无需修改LLM接口类

    Args:
        llm_interface: LLM 接口实例，用于与大语言模型通信
        toolkit: 可选的工具列表，可以是 Tool 对象或被 @tool 装饰的函数
        max_tool_calls: 最大工具调用次数，防止无限循环，默认为 5
        system_prompt_template: 可选的自定义系统提示模板，用于替代默认模板
        user_prompt_template: 可选的自定义用户提示模板，用于替代默认模板
        **llm_kwargs: 额外的关键字参数，将直接传递给LLM接口调用（如temperature、top_p等）

    Returns:
        装饰后的函数，返回值类型与原函数的返回类型标注一致

    示例1:
        ```python
        # 基本用法
        @llm_function(llm_interface=my_llm)
        def summarize_text(text: str, max_words: int = 100) -> str:
            \"\"\"生成输入文本的摘要，摘要不超过指定的词数。\"\"\"
            pass

        # 调用函数
        summary = summarize_text(long_text, max_words=50)
        ```
    示例2:
        ```python
        # 使用自定义系统提示模板
        custom_system_template = \"\"\"
        你是一名专业的文本处理专家，请根据以下信息执行任务:
        - 参数信息:
        {parameters_description}
        - 返回类型: {return_type_description}

        任务描述:
        {function_description}
        \"\"\"

        @llm_function(
            llm_interface=my_llm,
            system_prompt_template=custom_system_template
        )
        def analyze_text(text: str) -> Dict[str, Any]:
            \"\"\"分析文本并返回关键信息，包括主题、情感和关键词。\"\"\"
            pass

        ```
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # 获取函数的签名、类型提示和文档字符串
        signature = inspect.signature(func)
        docstring = func.__doc__ or ""
        func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # ===== 第一步：准备函数调用 =====
            context = _prepare_function_call(func, args, kwargs)

            with log_context(
                trace_id=context.trace_id,
                function_name=context.func_name,
                input_tokens=0,
                output_tokens=0,
            ):
                # 记录详细参数
                args_str = json.dumps(
                    context.bound_args.arguments,
                    default=str,
                    ensure_ascii=False,
                    indent=4,
                )

                app_log(
                    f"LLM 函数 '{context.func_name}' 被调用，参数: {args_str}",
                    location=get_location(),
                )

                # ===== 第二步：构建消息 =====
                messages = _build_messages(
                    context=context,
                    system_prompt_template=system_prompt_template,
                    user_prompt_template=user_prompt_template,
                )

                # ===== 第三步：处理工具 =====
                tool_param, tool_map = _prepare_tools_for_llm(
                    toolkit, context.func_name
                )

                # ===== 第四步：准备 LLM 调用参数 =====
                llm_params = LLMCallParams(
                    messages=messages,
                    tool_param=tool_param,
                    tool_map=tool_map,
                    llm_kwargs=llm_kwargs,
                )

                # ===== 第五步：执行 LLM 调用 =====
                try:
                    final_response = _execute_llm_with_retry(
                        llm_interface=llm_interface,
                        context=context,
                        llm_params=llm_params,
                        max_tool_calls=max_tool_calls,
                    )

                    # ===== 第六步：处理响应 =====
                    result = _process_final_response(
                        final_response, context.return_type
                    )
                    return result

                except Exception as e:
                    # 记录错误并重新抛出
                    push_error(
                        f"LLM 函数 '{context.func_name}' 执行时出错: {str(e)}",
                        location=get_location(),
                    )
                    raise

        # 保留原始函数的元数据
        wrapper.__name__ = func_name
        wrapper.__doc__ = docstring
        wrapper.__annotations__ = func.__annotations__
        wrapper.__signature__ = signature  # type: ignore

        return cast(Callable[..., T], wrapper)

    return decorator


def async_llm_function(
    llm_interface: LLM_Interface,
    toolkit: Optional[List[Union[Tool, Callable]]] = None,
    max_tool_calls: int = 5,  # 最大工具调用次数，防止无限循环
    system_prompt_template: Optional[str] = None,  # 自定义系统提示模板
    user_prompt_template: Optional[str] = None,  # 自定义用户提示模板
    **llm_kwargs,  # 额外的关键字参数，将直接传递给LLM接口
):
    """
    异步 LLM 函数装饰器，将函数的执行委托给大语言模型（异步版本）。

    此装饰器与 llm_function 功能相同，但使用原生异步实现，
    避免在 LLM 调用期间阻塞事件循环。

    ## 使用方法
    1. 定义一个异步函数，包括参数和返回类型标注
    2. 在函数的文档字符串中描述函数的功能或执行策略
    3. 使用 @async_llm_function 装饰该函数

    ## 异步特性
    - LLM 调用直接使用异步接口，不会阻塞主事件循环
    - 支持并发调用多个 LLM 函数
    - 可以与其他异步操作配合使用（如 asyncio.gather）

    ## 参数传递流程
    与同步版本相同：
    1. 当用户调用被装饰的函数时，装饰器捕获所有实际参数
    2. 这些参数被格式化为文本提示，发送给 LLM
    3. 函数的文档字符串作为系统提示，指导 LLM 如何执行
    4. 每个参数名称和值都作为用户提示的一部分

    ## 工具使用
    - 如果提供了工具集 (toolkit)，LLM 可以使用这些工具来辅助完成任务
    - 支持 Tool 对象或被 @tool 装饰的函数作为工具

    ## 自定义提示模板
    与同步版本相同，支持自定义系统提示和用户提示模板。

    ## 返回值处理
    - LLM 的响应会根据函数声明的返回类型进行转换
    - 支持基本类型 (str, int, float, bool)、字典和 Pydantic 模型

    ## LLM接口参数
    - 通过`**llm_kwargs`传递的参数将直接传递给LLM接口
    - 可以用于设置temperature、top_p等模型参数

    Args:
        llm_interface: LLM 接口实例，用于与大语言模型通信
        toolkit: 可选的工具列表，可以是 Tool 对象或被 @tool 装饰的函数
        max_tool_calls: 最大工具调用次数，防止无限循环，默认为 5
        system_prompt_template: 可选的自定义系统提示模板，用于替代默认模板
        user_prompt_template: 可选的自定义用户提示模板，用于替代默认模板
        **llm_kwargs: 额外的关键字参数，将直接传递给LLM接口调用（如temperature、top_p等）

    Returns:
        装饰后的异步函数，返回值类型与原函数的返回类型标注一致

    示例1:
        ```python
        # 基本用法
        @async_llm_function(llm_interface=my_llm)
        async def summarize_text(text: str, max_words: int = 100) -> str:
            \"\"\"生成输入文本的摘要，摘要不超过指定的词数。\"\"\"
            pass

        # 调用函数
        summary = await summarize_text(long_text, max_words=50)
        ```

    示例2:
        ```python
        # 并发调用
        @async_llm_function(llm_interface=my_llm)
        async def analyze_sentiment(text: str) -> str:
            \"\"\"分析文本的情感倾向。\"\"\"
            pass

        @async_llm_function(llm_interface=my_llm)
        async def extract_keywords(text: str) -> List[str]:
            \"\"\"提取文本中的关键词。\"\"\"
            pass

        # 并发调用
        texts = ["text1", "text2", "text3"]
        results = await asyncio.gather(*[
            analyze_sentiment(text) for text in texts
        ])
        ```

    示例3:
        ```python
        # 与其他异步操作配合
        @async_llm_function(llm_interface=my_llm)
        async def process_data(data: str) -> Dict[str, Any]:
            \"\"\"处理数据并返回结构化结果。\"\"\"
            pass

        async def main():
            # 同时进行多个操作
            tasks = [
                process_data(data1),
                process_data(data2),
                some_other_async_operation(),
            ]
            results = await asyncio.gather(*tasks)
            return results
        ```
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        # 获取函数的签名、类型提示和文档字符串
        signature = inspect.signature(func)
        docstring = func.__doc__ or ""
        func_name = func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # ===== 第一步：准备函数调用 =====
            context = _prepare_function_call(func, args, kwargs)

            async with async_log_context(
                trace_id=context.trace_id,
                function_name=context.func_name,
                input_tokens=0,
                output_tokens=0,
            ):
                # 记录详细参数
                args_str = json.dumps(
                    context.bound_args.arguments,
                    default=str,
                    ensure_ascii=False,
                    indent=4,
                )

                app_log(
                    f"异步 LLM 函数 '{context.func_name}' 被调用，参数: {args_str}",
                    location=get_location(),
                )

                # ===== 第二步：构建消息 =====
                messages = _build_messages(
                    context=context,
                    system_prompt_template=system_prompt_template,
                    user_prompt_template=user_prompt_template,
                )

                # ===== 第三步：处理工具 =====
                tool_param, tool_map = _prepare_tools_for_llm(
                    toolkit, context.func_name
                )

                # ===== 第四步：准备 LLM 调用参数 =====
                llm_params = LLMCallParams(
                    messages=messages,
                    tool_param=tool_param,
                    tool_map=tool_map,
                    llm_kwargs=llm_kwargs,
                )

                # ===== 第五步：异步执行 LLM 调用 =====
                try:
                    final_response = await _execute_llm_with_retry_async(
                        llm_interface=llm_interface,
                        context=context,
                        llm_params=llm_params,
                        max_tool_calls=max_tool_calls,
                    )

                    # ===== 第六步：处理响应 =====
                    result = _process_final_response(
                        final_response, context.return_type
                    )
                    return result

                except Exception as e:
                    # 记录错误并重新抛出
                    push_error(
                        f"异步 LLM 函数 '{context.func_name}' 执行时出错: {str(e)}",
                        location=get_location(),
                    )
                    raise

        # 保留原始函数的元数据
        async_wrapper.__name__ = func_name
        async_wrapper.__doc__ = docstring
        async_wrapper.__annotations__ = func.__annotations__
        async_wrapper.__signature__ = signature  # type: ignore

        return cast(Callable[..., Awaitable[T]], async_wrapper)

    return decorator


# ===== 默认提示模板 =====

# 默认系统提示模板
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
你的任务是按照以下的**功能描述**，根据用户的要求，给出符合要求的结果。

- 功能描述:
    {function_description}

- 你会接受到以下参数：
    {parameters_description}

- 你需要返回内容的类型: 
    {return_type_description}

执行要求:
1. 如果有工具可用，可以使用工具来辅助完成任务
2. 不要用 markdown 格式或代码块包裹结果，请直接输出期望的内容或者对应的JSON表示
"""

# 默认用户提示模板
DEFAULT_USER_PROMPT_TEMPLATE = """
给定的参数如下:
    {parameters}

直接返回结果，不需要任何解释或格式化。
"""


# ===== 内部辅助函数 =====


def _prepare_function_call(
    func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> FunctionCallContext:
    """
    准备函数调用，处理参数绑定和上下文创建

    Args:
        func: 被装饰的函数
        args: 位置参数
        kwargs: 关键字参数

    Returns:
        FunctionCallContext: 函数调用上下文信息
    """
    # 获取函数的签名、类型提示和文档字符串
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return")
    docstring = func.__doc__ or ""
    func_name = func.__name__

    # 构建追踪 ID，用于日志关联
    context_current_trace_id = get_current_trace_id()
    current_trace_id = f"{func_name}_{uuid.uuid4()}" + (
        f"_{context_current_trace_id}" if context_current_trace_id else ""
    )

    # 将参数绑定到函数签名，应用默认值
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    return FunctionCallContext(
        func_name=func_name,
        trace_id=current_trace_id,
        bound_args=bound_args,
        signature=signature,
        type_hints=type_hints,
        return_type=return_type,
        docstring=docstring,
    )


def _build_messages(
    context: FunctionCallContext,
    system_prompt_template: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    构建发送给 LLM 的消息列表，支持多模态内容

    Args:
        context: 函数调用上下文
        system_prompt_template: 自定义系统提示模板
        user_prompt_template: 自定义用户提示模板

    Returns:
        消息列表
    """
    # 检查是否包含多模态内容
    has_multimodal = _has_multimodal_content(
        context.bound_args.arguments, context.type_hints
    )

    if has_multimodal:
        # 构建多模态消息
        messages = _build_multimodal_messages(
            context, system_prompt_template, user_prompt_template
        )
    else:
        # 构建传统的文本消息
        system_prompt, user_prompt = _build_prompts(
            docstring=context.docstring,
            arguments=context.bound_args.arguments,
            type_hints=context.type_hints,
            custom_system_template=system_prompt_template,
            custom_user_template=user_prompt_template,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        push_debug(f"系统提示: {system_prompt}", location=get_location())
        push_debug(f"用户提示: {user_prompt}", location=get_location())

    return messages


def _prepare_tools_for_llm(
    toolkit: Optional[List[Union[Tool, Callable]]], func_name: str
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Callable]]:
    """
    处理工具准备，返回工具参数和工具映射

    Args:
        toolkit: 工具列表
        func_name: 函数名，用于日志

    Returns:
        (tool_param, tool_map) 元组
    """
    tool_param = None
    tool_map = {}  # 工具名称到函数的映射

    if toolkit:
        # 处理工具列表
        tool_objects, tool_map = _prepare_tools(toolkit, func_name)

        if tool_objects:
            # 序列化工具以供 LLM 使用
            tool_param = Tool.serialize_tools(tool_objects)

    return tool_param, tool_map


def _execute_llm_with_retry(
    llm_interface: LLM_Interface,
    context: FunctionCallContext,
    llm_params: LLMCallParams,
    max_tool_calls: int,
) -> Any:
    """
    执行 LLM 调用并处理重试逻辑（同步版本，内部使用异步调用）

    Args:
        llm_interface: LLM 接口实例
        context: 函数调用上下文
        llm_params: LLM 调用参数
        max_tool_calls: 最大工具调用次数

    Returns:
        最终的 LLM 响应
    """

    async def _async_execution():
        """异步执行逻辑"""
        return await _execute_llm_with_retry_async(
            llm_interface=llm_interface,
            context=context,
            llm_params=llm_params,
            max_tool_calls=max_tool_calls,
        )

    # 参考 llm_chat_decorator 的实现方式
    try:
        # 创建一个新的事件循环来处理异步操作
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_async_execution())
            return result
        finally:
            # 确保事件循环被正确关闭
            loop.close()

    except Exception as e:
        push_error(
            f"LLM 函数 '{context.func_name}' 执行出错: {str(e)}",
            location=get_location(),
        )
        raise


def _process_final_response(response: Any, return_type: Any) -> Any:
    """
    处理最终响应，转换为指定的返回类型

    Args:
        response: LLM 响应
        return_type: 期望的返回类型

    Returns:
        转换后的结果
    """
    return process_response(response, return_type)


def _build_prompts(
    docstring: str,
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    custom_system_template: Optional[str] = None,
    custom_user_template: Optional[str] = None,
) -> Tuple[str, str]:
    """
    构建发送给 LLM 的系统提示和用户提示

    流程:
    1. 从类型提示中提取参数类型
    2. 构建参数类型描述
    3. 获取返回类型的详细描述
    4. 使用模板格式化系统提示和用户提示

    Args:
        func_name: 函数名
        docstring: 函数文档字符串
        arguments: 函数参数值
        type_hints: 类型提示
        custom_system_template: 自定义系统提示模板
        custom_user_template: 自定义用户提示模板

    Returns:
        (system_prompt, user_prompt) 元组
    """
    # 移除返回类型提示，只保留参数类型
    param_type_hints = {k: v for k, v in type_hints.items() if k != "return"}

    # 构建参数类型描述（用于系统提示）
    param_type_descriptions = []
    for param_name, param_type in param_type_hints.items():
        type_str = (
            get_detailed_type_description(param_type) if param_type else "未知类型"
        )
        param_type_descriptions.append(f"  - {param_name}: {type_str}")

    # 获取返回类型的详细描述
    return_type = type_hints.get("return", None)
    return_type_description = get_detailed_type_description(return_type)

    # 使用自定义模板或默认模板
    system_template = custom_system_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE
    user_template = custom_user_template or DEFAULT_USER_PROMPT_TEMPLATE

    # 构建系统提示
    system_prompt = system_template.format(
        function_description=docstring,
        parameters_description="\n".join(param_type_descriptions),
        return_type_description=return_type_description,
    )

    # 构建用户提示（包含参数值）
    user_param_values = []
    for param_name, param_value in arguments.items():
        user_param_values.append(f"  - {param_name}: {param_value}")

    user_prompt = user_template.format(
        parameters="\n".join(user_param_values),
    )

    return system_prompt.strip(), user_prompt.strip()


def _prepare_tools(
    toolkit: List[Union[Tool, Callable]], func_name: str
) -> Tuple[List[Tool | Callable], Dict[str, Callable]]:
    """
    准备工具列表和工具映射

    流程:
    1. 遍历工具列表
    2. 将每个工具转换为 Tool 对象
    3. 创建工具名称到函数的映射

    Args:
        toolkit: 工具列表，可以是 Tool 对象或被 @tool 装饰的函数
        func_name: 函数名，用于日志

    Returns:
        (tool_objects, tool_map) 元组，包含 Tool 对象列表和工具名称到函数的映射
    """
    tool_objects = []
    tool_map = {}

    for tool in toolkit:
        if isinstance(tool, Tool):
            # 如果是 Tool 对象，直接添加
            tool_objects.append(tool)
            # 添加到工具映射
            tool_map[tool.name] = tool.run
        elif callable(tool) and hasattr(tool, "_tool"):
            # 如果是被 @tool 装饰的函数，获取其 _tool 属性
            tool_obj = tool._tool  # type: ignore
            tool_objects.append(tool_obj)
            # 添加到工具映射（使用原始函数）
            tool_map[tool_obj.name] = tool
        else:
            push_warning(
                f"LLM 函数 '{func_name}': "
                f"不支持的工具类型: {type(tool)}。"
                "工具必须是 Tool 对象或被 @tool 装饰的函数。",
                location=get_location(),
            )

    return tool_objects, tool_map


# ===== 异步版本的装饰器和辅助函数 =====


async def _execute_llm_with_retry_async(
    llm_interface: LLM_Interface,
    context: FunctionCallContext,
    llm_params: LLMCallParams,
    max_tool_calls: int,
) -> Any:
    """
    异步执行 LLM 调用并处理重试逻辑

    Args:
        llm_interface: LLM 接口实例
        context: 函数调用上下文
        llm_params: LLM 调用参数
        max_tool_calls: 最大工具调用次数

    Returns:
        最终的 LLM 响应
    """
    push_debug("开始异步 LLM 调用...", location=get_location())

    # 调用异步 LLM 并获取最终响应
    response_generator = execute_llm(
        llm_interface=llm_interface,
        messages=llm_params.messages,
        tools=llm_params.tool_param,
        tool_map=llm_params.tool_map,
        max_tool_calls=max_tool_calls,
        **llm_params.llm_kwargs,  # 传递额外的关键字参数
    )

    # 获取最后一个响应作为最终结果
    final_response = await get_last_item_of_async_generator(response_generator)

    # 检查final response中的content字段是否为空
    retry_times = llm_params.llm_kwargs.get("retry_times", 2)
    content = ""
    if hasattr(final_response, "choices") and len(final_response.choices) > 0:  # type: ignore
        message = final_response.choices[0].message  # type: ignore
        content = message.content if message and hasattr(message, "content") else ""

    if content == "":
        # 如果响应内容为空，记录警告并重试
        push_warning(
            f"异步 LLM 函数 '{context.func_name}' 返回的响应内容为空，将会自动重试。",
            location=get_location(),
        )
        # 重新调用 LLM
        while (
            retry_times > 0
            and hasattr(final_response.choices[0].message, "content")  # type: ignore
            and final_response.choices[0].message.content == ""  # type: ignore
        ):
            retry_times -= 1
            push_debug(
                f"异步 LLM 函数 '{context.func_name}' 重试第 {llm_params.llm_kwargs.get('retry_times', 2) - retry_times + 1} 次...",
                location=get_location(),
            )
            response_generator = execute_llm(
                llm_interface=llm_interface,
                messages=llm_params.messages,
                tools=llm_params.tool_param,
                tool_map=llm_params.tool_map,
                max_tool_calls=max_tool_calls,
                **llm_params.llm_kwargs,  # 传递额外的关键字参数
            )
            final_response = await get_last_item_of_async_generator(response_generator)

            content = extract_content_from_response(final_response, context.func_name)
            if content != "":  # type: ignore
                break

    content = extract_content_from_response(final_response, context.func_name)

    if content == "":
        push_error(
            f"异步 LLM 函数 '{context.func_name}' 返回的响应内容仍然为空，重试次数已用完。",
            location=get_location(),
        )
        raise ValueError("LLM response content is empty after retries.")

    # 记录最终响应
    push_debug(
        f"异步 LLM 函数 '{context.func_name}' 收到response {json.dumps(final_response, default=str, ensure_ascii=False, indent=2)}",
        location=get_location(),
    )

    return final_response


def _has_multimodal_content(
    arguments: Dict[str, Any], type_hints: Dict[str, Any]
) -> bool:
    """
    检查参数中是否包含多模态内容

    Args:
        arguments: 函数参数值
        type_hints: 类型提示

    Returns:
        是否包含多模态内容
    """
    return has_multimodal_content(arguments, type_hints)


def _build_multimodal_messages(
    context: FunctionCallContext,
    system_prompt_template: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    构建多模态消息列表

    Args:
        context: 函数调用上下文
        system_prompt_template: 自定义系统提示模板
        user_prompt_template: 自定义用户提示模板

    Returns:
        消息列表
    """
    # 构建系统提示（仍然是纯文本）
    system_prompt, _ = _build_prompts(
        docstring=context.docstring,
        arguments=context.bound_args.arguments,
        type_hints=context.type_hints,
        custom_system_template=system_prompt_template,
        custom_user_template=user_prompt_template,
    )

    # 构建多模态用户消息内容
    user_content = build_multimodal_content(
        context.bound_args.arguments, context.type_hints
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    push_debug(f"系统提示: {system_prompt}", location=get_location())
    push_debug(
        f"多模态用户消息包含 {len(user_content)} 个内容块", location=get_location()
    )

    return messages
