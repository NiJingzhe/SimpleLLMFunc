import inspect
from functools import wraps
import json
from typing import (
    Concatenate,
    Generator,
    List,
    Callable,
    TypeVar,
    Dict,
    Optional,
    Union,
    Tuple,
    cast
)
import uuid

from SimpleLLMFunc.logger.logger import push_error
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import (
    app_log,
    push_warning,
    get_location,
    log_context,
    get_current_trace_id,
)

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
    **llm_kwargs,  # 额外的关键字参数，将直接传递给LLM接口
):
    """
    LLM聊天装饰器，用于实现与大语言模型的对话功能，支持工具调用和历史记录管理。

    ## 参数传递
    - 装饰器会将函数参数以 key: value 的形式作为 user message 传递给 LLM API
    - history/chat_history 参数作为特殊参数处理，不会包含在 user message 中

    ## 历史记录要求
    history/chat_history 参数必须符合以下格式：
    ```
    [{"role": "user", "content": "用户消息"}, {"role": "assistant", "content": "助手回复"}]
    ```
    即包含 role 和 content 键的字典列表，role 可以是 user, assistant 或 system
    要求历史记录必须是chat函数的第一个参数。

    ## 历史记录处理
    - 如果不存在历史记录参数：仅使用 system prompt 和当前 user prompt 请求 LLM
    - 如果历史记录参数格式不正确：相关项会被忽略
    - 内部处理时，工具调用过程会生成完整历史记录（包含工具调用信息）
    - 返回给用户的历史记录会被过滤，仅保留 user、assistant 和 system 消息，不包含工具调用信息和结果

    ## 返回值格式
    装饰器修改后的函数将返回 `Generator[Tuple[str, List[Dict[str, str]]], None, None]` 类型：
    - `str`: 助手的响应内容
    - `List[Dict[str, str]]`: 过滤后的对话历史记录

    ## LLM接口参数
    - 通过`**llm_kwargs`传递的参数将直接传递给LLM接口
    - 可以用于设置temperature、top_p等模型参数，而无需修改LLM接口类

    Args:
        llm_interface: LLM接口实例，用于与大语言模型通信
        toolkit: 可选的工具列表，可以是 Tool 对象或被 @tool 装饰的函数, tool的信息会被解析并被加入到系统提示中
        max_tool_calls: 最大工具调用次数，防止无限循环，默认为 5
        **llm_kwargs: 额外的关键字参数，将直接传递给LLM接口调用（如temperature、top_p等）

    Returns:
        内置生成器的函数，生成器每次迭代返回: Tuple[str, List[Dict[str, str]]]

    示例:
        ```python
        @llm_chat(llm_interface=my_llm)
        def chat_with_llm(message: str, history: List[Dict[str, str]] = []):
            \"\"\"这里的文档字符串会作为系统提示发送给LLM，不建议在这里撰写Tool use相关信息，建议在Tool中就写详细的description\"\"\"
            pass

        response, updated_history = next(chat_with_llm("你好", history=[]))
        ```
    """

    def decorator(
        func: Callable[
            Concatenate[List[Dict[str, str]], ...],
            Generator[Tuple[str, List[Dict[str, str]]], None, None] | None,
        ],
    ) -> Callable[
        Concatenate[List[Dict[str, str]], ...],
        Generator[Tuple[str, List[Dict[str, str]]], None, None],
    ]:
        # 获取函数的签名
        signature = inspect.signature(func)
        # 获取函数的文档字符串
        docstring = func.__doc__ or ""

        # 获取func name用于优质log
        func_name = func.__name__

        @wraps(func)
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

            with log_context(
                trace_id=current_trace_id, 
                function_name=func_name,
                input_tokens=0,
                output_tokens=0
            ):

                # 处理tools参数
                tool_param_for_api = None  # 序列化后的工具参数，用于传递给API
                tool_map = {}  # 工具名称到函数的映射


                tool_objects = []
                if toolkit:
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
                        tool_param_for_api = Tool.serialize_tools(tool_objects)

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
                    current_messages.append(
                        {
                            "role": "system",
                            "content": f"{docstring}" + ("\n\n你需要灵活的使用以下工具：\n\t" if tool_objects != [] else '' + '\n\t'.join([f"- {tool.name}: {tool.description}" for tool in tool_objects]))
                        }
                    )

                formatted_history = None
                if custom_history is not None:
                    # 使用用户提供的历史
                    formatted_history = []
                    for msg in custom_history[1:]:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            formatted_history.append(msg)
                        else:
                            app_log(
                                f"LLM Chat '{func.__name__}' Skip history item with incorrect format: {msg}",
                                location=get_location(),
                            )

                if formatted_history is not None:
                    current_messages.extend(formatted_history)

                # 添加当前用户消息
                if user_message:
                    user_msg = {
                        "role": "user",
                        "content": user_message,
                    }

                    current_messages.append(user_msg)

                # 记录当前消息
                app_log(
                    f"LLM Chat '{func.__name__}' will execute llm with messages:"
                    f"\n{json.dumps(current_messages, ensure_ascii=False, indent=4)}",
                    location=get_location(),
                )

                try:
                    # 调用LLM
                    response_flow = execute_llm(
                        llm_interface=llm_interface,
                        messages=current_messages,
                        tools=tool_param_for_api,
                        tool_map=tool_map,
                        max_tool_calls=max_tool_calls,
                        **llm_kwargs,  # 传递额外的关键字参数
                    )

                    # 处理一次调用可能会产生的一系列response(因为ToolCall迭代)
                    complete_content = ""
                    for response in response_flow:

                        # 记录响应
                        app_log(
                            f"LLM Chat '{func.__name__}' got response:"
                            f"\n{json.dumps(response, default=str, ensure_ascii=False, indent=4)}",
                            location=get_location(),
                        )

                        # 提取响应内容
                        content = process_response(response, str)

                        complete_content += content

                        yield content, current_messages

                    current_messages.append(
                        {"role": "assistant", "content": complete_content}
                    )

                    yield "", current_messages

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
        wrapper.__signature__ = signature  # type: ignore

        return cast(
            Callable[
                Concatenate[List[Dict[str, str]], ...],
                Generator[Tuple[str, List[Dict[str, str]]], None, None],
            ],
            wrapper,
        )

    return decorator
