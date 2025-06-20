import asyncio
import inspect
import json
import uuid
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Callable, 
    Concatenate,
    Dict,
    Generator,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.llm_decorator.multimodal_types import ImgPath, ImgUrl, Text
from SimpleLLMFunc.llm_decorator.utils import (
    execute_llm,
    extract_content_from_stream_response,
)
from SimpleLLMFunc.logger import (
    app_log,
    get_current_trace_id,
    get_location,
    log_context,
    push_debug,
    push_error,
    push_warning,
)
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.utils import get_last_item_of_async_generator

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
                    **llm_kwargs,
                ):
                    yield result

            # 将异步生成器转换为同步生成器
            try:
                async_gen = _async_chat_logic()
                
                while True:
                    try:
                        # 使用 asyncio.run 获取下一个值
                        result = asyncio.run(async_gen.__anext__())
                        yield result
                    except StopAsyncIteration:
                        break
                        
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
        **llm_kwargs: 额外的LLM参数
        
    Yields:
        (响应内容, 更新后的历史记录) 元组
    """
    # 生成唯一的追踪ID
    context_trace_id = get_current_trace_id()
    current_trace_id = f"{func_name}_{uuid.uuid4()}"
    if context_trace_id:
        current_trace_id += f"_{context_trace_id}"

    # 绑定参数到函数签名
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    with log_context(
        trace_id=current_trace_id,
        function_name=func_name,
        input_tokens=0,
        output_tokens=0,
    ):
        try:
            # 1. 处理工具
            tool_param_for_api, tool_map = _process_tools(toolkit, func_name)
            
            # 2. 检查多模态内容
            has_multimodal = _has_multimodal_content_chat(
                bound_args.arguments, type_hints
            )
            
            # 3. 构建用户消息
            user_message_content = _build_user_message_content(
                bound_args.arguments, type_hints, has_multimodal
            )
            
            # 4. 处理历史记录
            custom_history = _extract_history_from_args(
                bound_args.arguments, func_name
            )
            
            # 5. 构建完整消息列表
            current_messages = _build_messages(
                docstring, custom_history, user_message_content, 
                tool_param_for_api, has_multimodal
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
            current_messages.append({
                "role": "assistant", 
                "content": complete_content
            })
            yield "", current_messages

        except Exception as e:
            push_error(
                f"LLM Chat '{func_name}' 执行出错: {str(e)}",
                location=get_location(),
            )
            raise


# ===== 核心辅助函数 =====

def _process_tools(toolkit: Optional[List[Union[Tool, Callable]]], func_name: str) -> Tuple[Optional[List], Dict]:
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
            tool_obj = tool._tool
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


def _extract_history_from_args(arguments: Dict[str, Any], func_name: str) -> Optional[List[Dict]]:
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
    if not (isinstance(custom_history, list) and 
            all(isinstance(item, dict) for item in custom_history)):
        push_warning(
            f"LLM Chat '{func_name}' 历史记录参数应为 List[Dict[str, str]] 类型，"
            "将不传递历史记录",
            location=get_location(),
        )
        return None
    
    return custom_history


def _build_user_message_content(
    arguments: Dict[str, Any], 
    type_hints: Dict[str, Any], 
    has_multimodal: bool
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
        return _build_multimodal_user_message_chat(arguments, type_hints)
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
    has_multimodal: bool
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
                f"- {tool.name}: {tool.description}"   
                if isinstance(tool, Tool) else  
                f"- {tool._tool.name}: {tool._tool.description}" 
                if isinstance(tool, Callable) and hasattr(tool, "_tool") 
                else f"- {tool}"  # 处理其他类型的工具
                for tool in tool_objects
            )
        messages.append({"role": "system", "content": system_content})
    
    # 2. 添加历史记录
    if custom_history:
        for msg in custom_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
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

# ===== 多模态支持辅助函数 =====

def _has_multimodal_content_chat(arguments: Dict[str, Any], type_hints: Dict[str, Any]) -> bool:
    """
    检查聊天参数中是否包含多模态内容（自动排除历史记录参数）
    
    Args:
        arguments: 函数参数值
        type_hints: 类型提示
        
    Returns:
        是否包含多模态内容
    """
    for param_name, param_value in arguments.items():
        # 跳过历史记录参数
        if param_name in HISTORY_PARAM_NAMES:
            continue
            
        if param_name in type_hints:
            annotation = type_hints[param_name]
            if _is_multimodal_type_chat(param_value, annotation):
                return True
    return False


def _is_multimodal_type_chat(value: Any, annotation: Any) -> bool:
    """
    检查值和类型注解是否为多模态类型
    
    Args:
        value: 参数值
        annotation: 类型注解
        
    Returns:
        是否为多模态类型
    """
    # 检查直接的多模态类型实例
    if isinstance(value, (Text, ImgUrl, ImgPath)):
        return True
    
    # 检查类型注解
    if annotation in (Text, ImgUrl, ImgPath):
        return True
    
    # 检查泛型类型
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    # 处理List类型
    if origin in (list, List) and args:
        element_type = args[0]
        if element_type in (Text, ImgUrl, ImgPath):
            return True
        # 检查列表中的实际值
        if isinstance(value, (list, tuple)):
            return any(isinstance(item, (Text, ImgUrl, ImgPath)) for item in value)
    
    # 处理Union类型
    if origin is Union:
        return any(arg in (Text, ImgUrl, ImgPath) for arg in args)
    
    return False


def _build_multimodal_user_message_chat(
    arguments: Dict[str, Any], 
    type_hints: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    为聊天函数构建多模态用户消息（自动排除历史记录参数）
    
    Args:
        arguments: 函数参数值
        type_hints: 类型提示
        
    Returns:
        多模态消息内容列表
    """
    content = []
    
    for param_name, param_value in arguments.items():
        # 跳过历史记录参数
        if param_name in HISTORY_PARAM_NAMES:
            continue
            
        if param_name in type_hints:
            annotation = type_hints[param_name]
            parsed_content = _parse_parameter_chat(param_value, annotation, param_name)
            content.extend(parsed_content)
        else:
            # 没有类型注解的参数，默认作为文本处理
            content.append(_create_text_content_chat(param_value, param_name))
    
    return content


def _parse_parameter_chat(value: Any, annotation: Any, param_name: str) -> List[Dict[str, Any]]:
    """
    递归解析聊天参数，返回OpenAI内容格式列表
    
    Args:
        value: 参数值
        annotation: 类型注解
        param_name: 参数名称（用于日志）
        
    Returns:
        OpenAI格式内容列表
    """
    if value is None:
        return [_create_text_content_chat("None", param_name)]
    
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    # 处理List类型
    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            push_warning(
                f"参数 {param_name} 应为列表类型，但获得 {type(value)}", 
                location=get_location()
            )
            return [_create_text_content_chat(value, param_name)]
        
        element_type = args[0] if args else Any
        content = []
        for i, item in enumerate(value):
            item_content = _parse_parameter_chat(item, element_type, f"{param_name}[{i}]")
            content.extend(item_content)
        return content
    
    # 处理基础多模态类型
    if annotation in (Text, str):
        return [_create_text_content_chat(value, param_name)]
    elif annotation is ImgUrl:
        return [_create_image_url_content_chat(value, param_name)]
    elif annotation is ImgPath:
        return [_create_image_path_content_chat(value, param_name)]
    
    # 处理Union类型
    if origin is Union:
        return [_handle_union_type_chat(value, args, param_name)]
    
    # 默认作为文本处理
    return [_create_text_content_chat(value, param_name)]


def _create_text_content_chat(value: Any, param_name: str) -> Dict[str, Any]:
    """创建文本内容格式"""
    if isinstance(value, Text):
        text = value.content
    else:
        text = str(value)
    
    return {
        "type": "text", 
        "text": f"{param_name}: {text}"
    }


def _create_image_url_content_chat(value: Any, param_name: str) -> Dict[str, Any]:
    """创建图片URL内容格式"""
    if value is None:
        return _create_text_content_chat("None", param_name)
    
    if isinstance(value, ImgUrl):
        url = value.url
        detail = value.detail
    else:
        url = str(value)
        detail = "auto"
    
    push_debug(
        f"添加图片URL: {param_name} = {url} (detail: {detail})", 
        location=get_location()
    )
    
    image_url_data = {"url": url}
    if detail != "auto":
        image_url_data["detail"] = detail
    
    return {
        "type": "image_url",
        "image_url": image_url_data
    }


def _create_image_path_content_chat(value: Any, param_name: str) -> Dict[str, Any]:
    """创建本地图片内容格式"""
    if value is None:
        return _create_text_content_chat("None", param_name)
    
    if isinstance(value, ImgPath):
        img_path = value
        detail = value.detail
    else:
        img_path = ImgPath(value)
        detail = "auto"
    
    # 转换为base64编码的data URL
    base64_img = img_path.to_base64()
    mime_type = img_path.get_mime_type()
    data_url = f"data:{mime_type};base64,{base64_img}"
    
    push_debug(
        f"添加本地图片: {param_name} = {img_path.path} (detail: {detail})", 
        location=get_location()
    )
    
    image_url_data = {"url": data_url}
    if detail != "auto":
        image_url_data["detail"] = detail
    
    return {
        "type": "image_url", 
        "image_url": image_url_data
    }


def _handle_union_type_chat(value: Any, union_args: tuple, param_name: str) -> Dict[str, Any]:
    """处理Union类型，尝试匹配最合适的类型"""
    # 优先检查实际值的类型
    if isinstance(value, Text):
        return _create_text_content_chat(value, param_name)
    elif isinstance(value, ImgUrl):
        return _create_image_url_content_chat(value, param_name)
    elif isinstance(value, ImgPath):
        return _create_image_path_content_chat(value, param_name)
    
    # 尝试根据Union中的类型进行转换
    for arg_type in union_args:
        try:
            if arg_type is Text and isinstance(value, str):
                return _create_text_content_chat(Text(value), param_name)
            elif arg_type is ImgUrl and isinstance(value, str):
                return _create_image_url_content_chat(ImgUrl(value), param_name)
            elif arg_type is ImgPath and isinstance(value, str):
                return _create_image_path_content_chat(ImgPath(value), param_name)
        except Exception as e:
            push_debug(
                f"尝试将参数 {param_name} 转换为 {arg_type} 失败: {e}", 
                location=get_location()
            )
            continue
    
    # 默认作为文本处理
    return _create_text_content_chat(value, param_name)
