import asyncio
import inspect
import json
import uuid
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.llm_decorator.utils import (
    execute_llm,
    extract_content_from_stream_response,
    has_multimodal_content,
    build_multimodal_content,
)
from SimpleLLMFunc.logger import (
    app_log,
    async_log_context,
    get_current_trace_id,
    get_location,
    log_context,
    push_debug,
    push_error,
    push_warning,
)
from SimpleLLMFunc.tool import Tool

# 类型变量定义
T = TypeVar("T")
P = ParamSpec("P")

# 常量定义
HISTORY_PARAM_NAMES = ["history", "chat_history"]  # 历史记录参数名列表
DEFAULT_MAX_TOOL_CALLS = 5  # 默认最大工具调用次数


def llm_chat(
    llm_interface: LLM_Interface,
    toolkit: Optional[List[Union[Tool, Callable]]] = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    stream: bool = False,
    **llm_kwargs,
):
    """
    LLM聊天装饰器，用于实现与大语言模型的对话功能，支持工具调用和历史记录管理。

    这是同步版本的装饰器，内部使用 asyncio.run 来调用异步的 LLM 接口。
    对于需要原生异步支持的场景，请使用 @async_llm_chat 装饰器。

    ## 功能特性
    - 自动管理对话历史记录
    - 支持工具调用和函数执行
    - 支持多模态内容（文本、图片URL、本地图片）
    - 支持流式响应
    - 自动过滤和清理历史记录

    ## 参数传递规则
    - 装饰器会将函数参数以 `参数名: 参数值` 的形式作为用户消息传递给LLM
    - `history`/`chat_history` 参数作为特殊参数处理，不会包含在用户消息中
    - 函数的文档字符串会作为系统提示传递给LLM

    ## 历史记录格式要求
    ```python
    [
        {"role": "user", "content": "用户消息"},
        {"role": "assistant", "content": "助手回复"},
        {"role": "system", "content": "系统消息"}
    ]
    ```

    ## 返回值格式
    ```python
    Generator[Tuple[str, List[Dict[str, str]]], None, None]
    ```
    - `str`: 助手的响应内容
    - `List[Dict[str, str]]`: 过滤后的对话历史记录（不含工具调用信息）

    Args:
        llm_interface: LLM接口实例，用于与大语言模型通信
        toolkit: 可选的工具列表，可以是Tool对象或被@tool装饰的函数
        max_tool_calls: 最大工具调用次数，防止无限循环
        stream: 是否使用流式响应
        **llm_kwargs: 额外的关键字参数，将直接传递给LLM接口

    Returns:
        装饰后的函数，返回生成器，每次迭代返回(响应内容, 更新后的历史记录)

    Example:
        ```python
        @llm_chat(llm_interface=my_llm)
        def chat_with_llm(message: str, history: List[Dict[str, str]] = []):
            '''系统提示信息'''
            pass

        response, updated_history = next(chat_with_llm("你好", history=[]))
        ```
    """

    def decorator(
        func: Callable[P, Any],
    ) -> Callable[P, Generator[Tuple[str, List[Dict[str, str]]], None, None]]:
        # 获取函数元信息
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        docstring = func.__doc__ or ""
        func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            
            # 生成唯一的追踪ID
            context_trace_id = get_current_trace_id()
            current_trace_id = f"{func_name}_{uuid.uuid4()}"
            if context_trace_id:
                current_trace_id += f"_{context_trace_id}"

            # 使用同步的日志上下文
            with log_context(
                trace_id=current_trace_id,
                function_name=func_name,
                input_tokens=0,
                output_tokens=0,
            ):
                # 使用内部异步函数处理实际逻辑
                async def _async_chat_logic():
                    async for result in _async_llm_chat_impl(
                        func_name=func_name,
                        signature=signature,
                        type_hints=type_hints,
                        docstring=docstring,
                        args=args,
                        kwargs=kwargs,
                        llm_interface=llm_interface,
                        toolkit=toolkit,
                        max_tool_calls=max_tool_calls,
                        stream=stream,
                        use_log_context=False,  # 不在异步实现中使用日志上下文
                        **llm_kwargs,
                    ):
                        yield result

                # 将异步生成器转换为同步生成器
                try:
                    # 创建一个新的事件循环来处理整个异步生成器
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        async_gen = _async_chat_logic()

                        # 使用 run_until_complete 来逐个获取异步生成器的值
                        while True:
                            try:
                                result = loop.run_until_complete(async_gen.__anext__())
                                yield result
                            except StopAsyncIteration:
                                break
                    finally:
                        loop.close()

                except Exception as e:
                    push_error(
                        f"LLM Chat '{func_name}' 执行出错: {str(e)}",
                        location=get_location(),
                    )
                    raise

        # 保留原始函数的元数据
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__
        wrapper.__signature__ = signature  # type: ignore

        return cast(
            Callable[P, Generator[Tuple[str, List[Dict[str, str]]], None, None]],
            wrapper,
        )

    return decorator


def async_llm_chat(
    llm_interface: LLM_Interface,
    toolkit: Optional[List[Union[Tool, Callable]]] = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    stream: bool = False,
    **llm_kwargs,
):
    """
    异步LLM聊天装饰器，用于实现与大语言模型的异步对话功能，支持工具调用和历史记录管理。

    这是原生异步版本的装饰器，提供完全的异步支持，返回 AsyncGenerator。
    对于不需要异步的场景，请使用 @llm_chat 装饰器。

    ## 功能特性
    - 自动管理对话历史记录
    - 支持工具调用和函数执行
    - 支持多模态内容（文本、图片URL、本地图片）
    - 支持流式响应
    - 自动过滤和清理历史记录
    - 原生异步支持，无阻塞执行

    ## 参数传递规则
    - 装饰器会将函数参数以 `参数名: 参数值` 的形式作为用户消息传递给LLM
    - `history`/`chat_history` 参数作为特殊参数处理，不会包含在用户消息中
    - 函数的文档字符串会作为系统提示传递给LLM

    ## 历史记录格式要求
    ```python
    [
        {"role": "user", "content": "用户消息"},
        {"role": "assistant", "content": "助手回复"},
        {"role": "system", "content": "系统消息"}
    ]
    ```

    ## 返回值格式
    ```python
    AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]
    ```
    - `str`: 助手的响应内容
    - `List[Dict[str, str]]`: 过滤后的对话历史记录（不含工具调用信息）

    Args:
        llm_interface: LLM接口实例，用于与大语言模型通信
        toolkit: 可选的工具列表，可以是Tool对象或被@tool装饰的函数
        max_tool_calls: 最大工具调用次数，防止无限循环
        stream: 是否使用流式响应
        **llm_kwargs: 额外的关键字参数，将直接传递给LLM接口

    Returns:
        装饰后的函数，返回异步生成器，每次迭代返回(响应内容, 更新后的历史记录)

    Example:
        ```python
        @async_llm_chat(llm_interface=my_llm)
        async def chat_with_llm(message: str, history: List[Dict[str, str]] = []):
            '''系统提示信息'''
            pass

        async for response, updated_history in chat_with_llm("你好", history=[]):
            print(response)
        ```
    """

    def decorator(
        func: Callable[P, Any],
    ) -> Callable[P, AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]]:
        # 获取函数元信息
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        docstring = func.__doc__ or ""
        func_name = func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成唯一的追踪ID
            context_trace_id = get_current_trace_id()
            current_trace_id = f"{func_name}_{uuid.uuid4()}"
            if context_trace_id:
                current_trace_id += f"_{context_trace_id}"

            # 使用异步的日志上下文
            async with async_log_context(
                trace_id=current_trace_id,
                function_name=func_name,
                input_tokens=0,
                output_tokens=0,
            ):
                async for result in _async_llm_chat_impl(
                    func_name=func_name,
                    signature=signature,
                    type_hints=type_hints,
                    docstring=docstring,
                    args=args,
                    kwargs=kwargs,
                    llm_interface=llm_interface,
                    toolkit=toolkit,
                    max_tool_calls=max_tool_calls,
                    stream=stream,
                    use_log_context=False,  # 不在异步实现中使用日志上下文
                    **llm_kwargs,
                ):
                    yield result

        # 保留原始函数的元数据
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__
        wrapper.__signature__ = signature  # type: ignore

        return cast(
            Callable[P, AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]],
            wrapper,
        )

    return decorator


async def _async_llm_chat_impl(
    func_name: str,
    signature: inspect.Signature,
    type_hints: Dict[str, Any],
    docstring: str,
    args: tuple,
    kwargs: dict,
    llm_interface: LLM_Interface,
    toolkit: Optional[List[Union[Tool, Callable]]],
    max_tool_calls: int,
    stream: bool,
    use_log_context: bool = True,
    **llm_kwargs,
) -> AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]:
    """
    共享的异步LLM聊天实现逻辑

    Args:
        func_name: 函数名称
        signature: 函数签名
        type_hints: 类型提示
        docstring: 文档字符串
        args: 位置参数
        kwargs: 关键字参数
        llm_interface: LLM接口
        toolkit: 工具列表
        max_tool_calls: 最大工具调用次数
        stream: 是否流式响应
        use_log_context: 是否使用异步日志上下文
        **llm_kwargs: 额外的LLM参数

    Yields:
        (响应内容, 更新后的历史记录) 元组
    """
    # 绑定参数到函数签名
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    async def _execute_impl():
        # 1. 处理工具
        tool_param_for_api, tool_map = _process_tools(toolkit, func_name)

        # 2. 检查多模态内容
        has_multimodal = has_multimodal_content(
            bound_args.arguments, type_hints, exclude_params=HISTORY_PARAM_NAMES
        )

        # 3. 构建用户消息
        user_message_content = _build_user_message_content(
            bound_args.arguments, type_hints, has_multimodal
        )

        # 4. 处理历史记录
        custom_history = _extract_history_from_args(bound_args.arguments, func_name)

        # 5. 构建完整消息列表
        current_messages = _build_messages(
            docstring,
            custom_history,
            user_message_content,
            tool_param_for_api,
            has_multimodal,
        )

        # 6. 记录调试信息
        push_debug(
            f"LLM Chat '{func_name}' 将使用以下消息执行:"
            f"\n{json.dumps(current_messages, ensure_ascii=False, indent=2)}",
            location=get_location(),
        )

        # 7. 执行LLM调用并处理响应
        complete_content = ""
        response_flow = execute_llm(
            llm_interface=llm_interface,
            messages=current_messages,
            tools=tool_param_for_api,
            tool_map=tool_map,
            max_tool_calls=max_tool_calls,
            stream=stream,
            **llm_kwargs,
        )

        # 8. 处理响应流（异步迭代）
        async for response in response_flow:
            app_log(
                f"LLM Chat '{func_name}' 收到响应:"
                f"\n{json.dumps(response, default=str, ensure_ascii=False, indent=2)}",
                location=get_location(),
            )

            content = extract_content_from_stream_response(response, func_name)
            complete_content += content
            yield content, current_messages

        # 9. 添加最终响应到历史记录
        # current_messages.append({"role": "assistant", "content": complete_content})
        yield "", current_messages

    if use_log_context:
        # 生成唯一的追踪ID
        context_trace_id = get_current_trace_id()
        current_trace_id = f"{func_name}_{uuid.uuid4()}"
        if context_trace_id:
            current_trace_id += f"_{context_trace_id}"

        async with async_log_context(
            trace_id=current_trace_id,
            function_name=func_name,
            input_tokens=0,
            output_tokens=0,
        ):
            try:
                async for result in _execute_impl():
                    yield result
            except Exception as e:
                push_error(
                    f"LLM Chat '{func_name}' 执行出错: {str(e)}",
                    location=get_location(),
                )
                raise
    else:
        # 不使用日志上下文，直接执行
        try:
            async for result in _execute_impl():
                yield result
        except Exception as e:
            push_error(
                f"LLM Chat '{func_name}' 执行出错: {str(e)}",
                location=get_location(),
            )
            raise


# ===== 核心辅助函数 =====


def _process_tools(
    toolkit: Optional[List[Union[Tool, Callable]]], func_name: str
) -> Tuple[Optional[List], Dict]:
    """
    处理工具列表，返回API所需的工具参数和工具映射

    Args:
        toolkit: 工具列表
        func_name: 函数名，用于日志记录

    Returns:
        (tool_param_for_api, tool_map): API工具参数和工具名称到函数的映射
    """
    if not toolkit:
        return None, {}

    tool_objects = []
    tool_map = {}

    for tool in toolkit:
        if isinstance(tool, Tool):
            # Tool对象直接添加
            tool_objects.append(tool)
            tool_map[tool.name] = tool.run
        elif callable(tool) and hasattr(tool, "_tool"):
            # @tool装饰的函数
            tool_obj = getattr(tool, "_tool", None)
            assert isinstance(
                tool_obj, Tool
            ), "这一定是一个Tool对象，不会是None！是None我赤石"
            tool_objects.append(tool_obj)
            tool_map[tool_obj.name] = tool_obj.run
        else:
            push_warning(
                f"LLM函数 '{func_name}': 不支持的工具类型 {type(tool)}，"
                "工具必须是Tool对象或被@tool装饰的函数",
                location=get_location(),
            )

    tool_param_for_api = Tool.serialize_tools(tool_objects) if tool_objects else None

    push_debug(
        f"LLM Chat '{func_name}' 加载了 {len(tool_objects)} 个工具",
        location=get_location(),
    )

    return tool_param_for_api, tool_map


def _extract_history_from_args(
    arguments: Dict[str, Any], func_name: str
) -> Optional[List[Dict]]:
    """
    从函数参数中提取历史记录

    Args:
        arguments: 函数参数字典
        func_name: 函数名，用于日志记录

    Returns:
        历史记录列表或None
    """
    # 查找历史记录参数
    history_param_name = None
    for param_name in HISTORY_PARAM_NAMES:
        if param_name in arguments:
            history_param_name = param_name
            break

    if not history_param_name:
        push_warning(
            f"LLM Chat '{func_name}' 缺少历史记录参数"
            f"（参数名应为 {HISTORY_PARAM_NAMES} 之一），将不传递历史记录",
            location=get_location(),
        )
        return None

    custom_history = arguments[history_param_name]

    # 验证历史记录格式
    if not (
        isinstance(custom_history, list)
        and all(isinstance(item, dict) for item in custom_history)
    ):
        push_warning(
            f"LLM Chat '{func_name}' 历史记录参数应为 List[Dict[str, str]] 类型，"
            "将不传递历史记录",
            location=get_location(),
        )
        return None

    return custom_history


def _build_user_message_content(
    arguments: Dict[str, Any], type_hints: Dict[str, Any], has_multimodal: bool
) -> Union[str, List[Dict[str, Any]]]:
    """
    构建用户消息内容

    Args:
        arguments: 函数参数字典
        type_hints: 类型提示字典
        has_multimodal: 是否包含多模态内容

    Returns:
        用户消息内容（文本或多模态内容列表）
    """
    if has_multimodal:
        return build_multimodal_content(
            arguments, type_hints, exclude_params=HISTORY_PARAM_NAMES
        )
    else:
        # 构建传统文本消息，排除历史记录参数
        message_parts = []
        for param_name, param_value in arguments.items():
            if param_name not in HISTORY_PARAM_NAMES:
                message_parts.append(f"{param_name}: {param_value}")
        return "\n\t".join(message_parts)


def _build_messages(
    docstring: str,
    custom_history: Optional[List[Dict]],
    user_message_content: Union[str, List[Dict[str, Any]]],
    tool_objects: Optional[List],
    has_multimodal: bool,
) -> List[Dict[str, Any]]:
    """
    构建完整的消息列表

    Args:
        docstring: 函数文档字符串
        custom_history: 用户提供的历史记录
        user_message_content: 用户消息内容
        tool_objects: 工具列表
        has_multimodal: 是否为多模态消息

    Returns:
        完整的消息列表
    """
    messages = []

    # 1. 添加系统消息
    if docstring:
        system_content = docstring
        if tool_objects:
            system_content += "\n\n你需要灵活使用以下工具：\n\t" + "\n\t".join(
                (
                    f"- {tool.name}: {tool.description}"
                    if isinstance(tool, Tool)
                    else (
                        f"- {getattr(getattr(tool, '_tool'), 'name')}: {getattr(getattr(tool, '_tool'), 'description')}"
                        if isinstance(tool, Callable) and hasattr(tool, "_tool")
                        else f"- {tool}"
                    )
                )  # 处理其他类型的工具
                for tool in tool_objects
            )
        messages.append({"role": "system", "content": system_content})

    # 2. 添加历史记录
    if custom_history:
        for msg in custom_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["role"] not in ["system"]:
                    messages.append(msg)
            else:
                push_warning(
                    f"跳过格式不正确的历史记录项: {msg}",
                    location=get_location(),
                )

    # 3. 添加当前用户消息
    if user_message_content:
        if has_multimodal and isinstance(user_message_content, list):
            user_msg = {"role": "user", "content": user_message_content}
        else:
            user_msg = {"role": "user", "content": user_message_content}
        messages.append(user_msg)

    return messages
