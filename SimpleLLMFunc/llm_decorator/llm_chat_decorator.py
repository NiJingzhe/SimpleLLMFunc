import inspect
import json
from typing import (
    Concatenate,
    Generator,
    List,
    Callable,
    TypeVar,
    ParamSpec,
    Dict,
    get_type_hints,
    Optional,
    Union,
    Tuple,
)
import uuid
from xml.sax.xmlreader import Locator

from SimpleLLMFunc.logger.logger import push_critical, push_error
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import (
    app_log,
    push_warning,
    get_location,
    log_context,
    get_current_trace_id,
)

# 从utils模块导入工具函数 - 修正导入路径
from SimpleLLMFunc.llm_decorator.utils import (
    execute_llm,
    process_response,
)

# 定义一个类型变量，用于函数的返回类型
T = TypeVar("T")


def llm_chat(
    llm_interface: LLM_Interface,
    toolkit: Optional[List[Union[Tool, Callable]]] = None,
    max_tool_calls: int = 5,  # 最大工具调用次数，防止无限循环
):
    """
    LLM聊天装饰器，用于实现聊天功能

    我们将以 key: value 的形式将入参作为user message传递给API

    入参中 **必须要包含** 的 history/chat_history 参数不会被作为user prompt的一部分，而是会被视作自定义的历史记录。

    这些参数需要满足于一定的格式，如下:

    1. [{"role": "user", "content": "xxx"}]，即具有role和content两个键和字符串值的字典构成的列表

    如果不存在历史记录参数，那么我们会只使用一个system prompt和user prompt来请求LLM

    如果历史记录参数存在，但是其中有不正确格式的item，那么这个item会被忽略
    
    被该装饰器装饰的函数必须被标注这样的返回值:

        - 无标注
        
        - Generator[Tuple[str, List[Dict[str, str]]], None, None]  
        
    但是装饰器返回的结果一定具有 Generator[Tuple[str, List[Dict[str, str]]], None, None] 的返回类型

    Args:
        llm_interface: LLM接口
        tools: 可选的工具列表，可以是Tool对象或被@tool装饰的函数
        max_tool_calls: 最大工具调用次数，防止无限循环，默认为5

    Returns:
        装饰后的函数, 装饰器会给函数加上一个返回值，这个返回值是对话历史记录。
    """

    def decorator(  
        func: Callable[Concatenate[List[Dict[str, str]], ...], Generator[Tuple[str, List[Dict[str, str]]], None, None] | None],
    ) -> Callable[
        Concatenate[List[Dict[str, str]], ...],
        Generator[Tuple[str, List[Dict[str, str]]], None, None],
    ]:
        # 获取函数的签名
        signature = inspect.signature(func)
        # 获取函数的文档字符串
        docstring = func.__doc__ or ""

        func_name = func.__name__

        def wrapper(*args, **kwargs):

            context_current_trace_id = get_current_trace_id()

            # 当前 trace id 的构建逻辑：
            # 为了确保能够从上下文继承语义，同时有保证每次function调用能够被区分
            # 我们会对上下文中的trace id进行拼接
            # function name _ uuid4 _ context_trace_id(if have) or "" (if not have)
            current_trace_id = f"{func.__name__}_{uuid.uuid4()}" + (
                f"_{context_current_trace_id}" if context_current_trace_id else ""
            )

            # 绑定参数到函数签名
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            possible_history_param_name = ["history", "chat_history"]

            with log_context(trace_id=current_trace_id, function_name=func_name):
                # 检查是否有messages参数，这会被直接作为API的messages参数。
                user_message = ""

                # 获得函数的入参列表
                input_param_names = list(bound_args.arguments.keys())
                input_param_values = list(bound_args.arguments.values())

                # 历史记录参数不会被作为user message的一部分
                user_message = "\n\t".join(
                    [
                        f"{param}: {value}"
                        for param, value in zip(input_param_names, input_param_values)
                        if param not in possible_history_param_name
                    ]
                )

                custom_history = None
                # 将函数的参数列表和possible_history_param_name求交集
                intersect_of_function_params_and_history_param = [
                    param
                    for param in possible_history_param_name
                    if param in input_param_names
                ]

                # 检查是否为空
                if len(intersect_of_function_params_and_history_param) == 0:
                    push_warning(
                        f"LLM Chat '{func.__name__}' doesn't have correct history parameter"
                        " with name 'history' or 'chat_history', which is required for LLM chat function."
                        " No history will be passed to llm.",
                        location=get_location(),
                    )
                else:
                    # 获取第一个匹配的历史记录参数
                    history_param_name = intersect_of_function_params_and_history_param[
                        0
                    ]
                    # 获取对应的值
                    custom_history = bound_args.arguments[history_param_name]

                    if not (
                        isinstance(custom_history, list)
                        and all(isinstance(item, dict) for item in custom_history)
                    ):
                        push_warning(
                            f"LLM Chat '{func.__name__}' history parameter should be a List[Dict[str, str]]."
                            " No history will be passed to llm.",
                            location=get_location(),
                        )
                        custom_history = None

                # 经过这样的设计后，custom history中只可能是正确的历史记录或者None

                # 准备消息列表
                current_messages = []

                nonlocal docstring
                # 添加系统消息
                if docstring != "":
                    current_messages.append({"role": "system", "content": docstring})

                # 使用自定义历史或者函数的专属历史
                formatted_history = []
                if custom_history is not None:
                    # 使用用户提供的历史
                    formatted_history = []
                    for msg in custom_history[1:]:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            formatted_history.append(msg)
                        elif isinstance(msg, tuple) and len(msg) == 2:
                            role, content = msg
                            formatted_history.append({"role": role, "content": content})
                        else:
                            app_log(
                                f"LLM Chat '{func.__name__}' Skip history item with incorrect format: {msg}",
                                location=get_location(),
                            )
                else:
                    # 那就当作没有历史记录了
                    formatted_history = []

                current_messages.extend(formatted_history)

                # 添加当前用户消息
                if user_message:
                    user_msg = {"role": "user", "content": user_message + "\n\n务必思考是否要使用工具"}
                    current_messages.append(user_msg)

                # 记录当前消息
                app_log(
                    f"LLM Chat '{func.__name__}' will execute llm with messages:"
                    f"\n{json.dumps(current_messages, ensure_ascii=False, indent=4)}",
                    location=get_location(),
                )

                # 处理tools参数
                tool_param = None
                tool_map = {}  # 工具名称到函数的映射

                if toolkit:
                    tool_objects = []
                    for tool in toolkit:
                        if isinstance(tool, Tool):
                            # 如果是Tool对象，直接添加
                            tool_objects.append(tool)
                            # 添加到工具映射
                            tool_map[tool.name] = tool.run
                        elif callable(tool) and hasattr(tool, "_tool"):
                            # 如果是被@tool装饰的函数，获取其_tool属性
                            tool_obj = tool._tool
                            tool_objects.append(tool_obj)
                            # 添加到工具映射（使用run方法以保持一致性）
                            tool_map[tool_obj.name] = tool_obj.run
                        else:
                            push_warning(
                                f"LLM Chat '{func.__name__}':"
                                f" Unsupported tool type: {type(tool)}."
                                " Tool must be a Tool object or a function decorated with @tool.",
                                location=get_location(),
                            )

                    if tool_objects:
                        tool_param = Tool.serialize_tools(tool_objects)

                try:
                    # 调用LLM
                    response_flow = execute_llm(
                        llm_interface=llm_interface,
                        messages=current_messages,
                        tools=tool_param,
                        tool_map=tool_map,
                        max_tool_calls=max_tool_calls,
                    )

                    # 处理一次调用可能会产生的一系列response(因为ToolCall迭代)
                    for response in response_flow:

                        # 记录响应
                        app_log(
                            f"LLM Chat '{func.__name__}' got response:"
                            f"\n{json.dumps(response, default=str, ensure_ascii=False, indent=4)}",
                            location=get_location(),
                        )

                        # 提取响应内容
                        content = process_response(response, str)

                        # 在有正确历史参数传入的时候
                        if len(formatted_history) != 0:
                            # 将响应内容添加到历史记录中
                            current_messages.append(
                                {"role": "assistant", "content": content}
                            )

                        yield content, [item for item in current_messages if item["content"] != ""]

                except Exception as e:
                    # 修复：在log_context环境中不再传递trace_id参数
                    push_error(
                        f"LLM Chat '{func.__name__}' Got error: {str(e)}",
                        location=get_location(),  # 明确指定为location参数
                    )
                    raise

        # 保留原始函数的元数据
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__

        return wrapper

    return decorator
