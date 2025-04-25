import inspect
import json
from typing import (
    List,
    Callable,
    TypeVar,
    Dict,
    Any,
    get_type_hints,
    Optional,
    Union,
    Tuple,
)
import uuid

from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import (
    app_log,
    push_warning,
    get_location,
    log_context,
    get_current_trace_id,
)

# 从utils模块导入工具函数
from SimpleLLMFunc.llm_decorator.utils import (
    execute_with_tools,
    process_response,
    get_detailed_type_description,
)

# 定义一个类型变量，用于函数的返回类型
T = TypeVar("T")


def llm_function(
    llm_interface: LLM_Interface,
    tools: Optional[List[Union[Tool, Callable]]] = None,
    trace_id: Optional[str] = None,
    max_tool_calls: int = 5,  # 最大工具调用次数，防止无限循环
):
    """
    LLM函数装饰器，将函数的执行委托给LLM

    Args:
        llm_interface: LLM接口
        tools: 可选的工具列表，可以是Tool对象或被@tool装饰的函数
        system_prompt: 可选的系统提示
        trace_id: 可选的追踪ID，用于日志。如果不指定，会自动生成，也可以通过log_context上下文管理器传递
        max_tool_calls: 最大工具调用次数，防止无限循环，默认为5

    Returns:
        装饰后的函数
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # 获取函数的签名
        signature = inspect.signature(func)
        # 获取函数的类型提示
        type_hints = get_type_hints(func)
        # 获取返回类型
        return_type = type_hints.get("return")
        # 获取函数的文档字符串
        docstring = func.__doc__ or ""

        def wrapper(*args, **kwargs):

            context_current_trace_id = get_current_trace_id()

            # 使用优先级: 1.参数传入的trace_id 2.上下文中的trace_id 3.自动生成的trace_id
            current_trace_id = f"{func.__name__}_{uuid.uuid4()}" + (
                f"_{trace_id}"
                if trace_id
                else (
                    f"_{context_current_trace_id}"
                    if context_current_trace_id
                    else f"_{uuid.uuid4()}"
                )
            )
            location = get_location()

            # 绑定参数到函数签名
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            with log_context(trace_id=current_trace_id):

                # 构建system prompt和user prompt
                system_template, user_template = _build_prompts(
                    func.__name__,
                    docstring,
                    bound_args.arguments,
                    type_hints,
                )

                # 修改：使用ensure_ascii=False来正确显示中文字符
                app_log(
                    f"LLM Function '{func.__name__}' called with arguments: {json.dumps(bound_args.arguments, default=str, ensure_ascii=False)}"
                    # 不需要显式传递trace_id，因为在log_context上下文中
                )

                # 准备messages
                messages = []

                # 添加系统提示
                messages.append({"role": "system", "content": system_template})

                # 添加用户提示
                messages.append({"role": "user", "content": user_template})

                # 处理tools参数，支持Tool对象和被@tool装饰的函数
                tool_param = None
                tool_map = {}  # 工具名称到函数的映射

                if tools:
                    tool_objects = []
                    for tool in tools:
                        if isinstance(tool, Tool):
                            # 如果是Tool对象，直接添加
                            tool_objects.append(tool)
                            # 添加到工具映射
                            tool_map[tool.name] = tool.run
                        elif callable(tool) and hasattr(tool, "_tool"):
                            # 如果是被@tool装饰的函数，获取其_tool属性
                            tool_obj = tool._tool
                            tool_objects.append(tool_obj)
                            # 添加到工具映射（使用原始函数）
                            tool_map[tool_obj.name] = tool
                        elif callable(tool) and hasattr(tool, "tool"):
                            # 兼容某些工具可能使用.tool属性而不是_tool
                            tool_obj = tool.tool
                            tool_objects.append(tool_obj)
                            # 添加到工具映射
                            tool_map[tool_obj.name] = tool
                        else:
                            push_warning(
                                f"Unsupported tool type: {type(tool)}. Tool must be a Tool object or a function decorated with @tool.",
                                location,
                                # 不需要显式传递trace_id
                            )

                    if tool_objects:
                        tool_param = Tool.serialize_tools(tool_objects)

                try:
                    # 简化的工具调用逻辑
                    if tool_param:
                        # 有工具参数，进入工具调用流程
                        final_response = execute_with_tools(
                            llm_interface=llm_interface,
                            messages=messages,
                            tools=tool_param,
                            tool_map=tool_map,
                            max_tool_calls=max_tool_calls,
                            func_name=func.__name__,
                        )
                    else:
                        # 无工具参数，直接调用LLM
                        final_response = llm_interface.chat(
                            messages=messages,
                            trace_id=current_trace_id,
                        )

                        # 记录响应
                        app_log(
                            f"LLM Function '{func.__name__}' received response: {json.dumps(final_response, default=str, ensure_ascii=False)}"
                            # 不需要显式传递trace_id
                        )

                    # 处理最终响应
                    result = process_response(final_response, return_type)
                    return result

                except Exception as e:
                    push_warning(
                        f"LLM Function '{func.__name__}' encountered an error: {str(e)}",
                        location,
                        # 不需要显式传递trace_id
                    )
                    raise

        # 保留原始函数的元数据
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__

        return wrapper

    return decorator


# 定义系统提示模板
LLM_FUNCTION_SYSTEM_PROMPT = (
    "作为函数执行者，你的任务是按照以下的函数说明，在给定的输入下给出这个函数的输出结果。\n"
    "- 函数名称: {function_name}\n"
    "- 函数描述: {function_description}\n"
    "- 参数类型:\n"
    "\t{parameters_description}\n"
    "请根据用户提供的参数值执行此函数并返回结果。返回格式必须符合指定的返回类型。"
    "如果返回类型是Pydantic模型，请以JSON格式返回符合模型规范的数据。\n"
    "期望返回类型: {return_type_description}\n"
    "如果提供了工具调用，可以考虑使用工具来辅助完成任务。"
)

# 定义用户提示模板
LLM_FUNCTION_USER_PROMPT = (
    "请使用以下参数值执行函数 {function_name}:\n"
    "\t{parameters}\n"
    "请直接输出函数执行的结果,也不要用任何markdown格式包裹结果。直接输出结果即可。"
)


def _build_prompts(
    func_name: str,
    docstring: str,
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
) -> Tuple[str, str]:
    """
    构建发送给LLM的system prompt和user prompt

    Args:
        func_name: 函数名
        docstring: 函数文档字符串
        arguments: 函数参数
        type_hints: 类型提示
        custom_system_prompt: 自定义系统提示

    Returns:
        (system_prompt, user_prompt)的元组
    """
    # 移除返回类型提示
    param_type_hints = {k: v for k, v in type_hints.items() if k != "return"}

    # 构建参数类型描述（用于system prompt）
    param_type_descriptions = []
    for param_name, param_type in param_type_hints.items():
        type_str = str(param_type) if param_type else "未知类型"
        param_type_descriptions.append(f"- {param_name} ({type_str})")

    # 构建返回类型描述
    return_type = type_hints.get("return", None)
    return_type_description = get_detailed_type_description(return_type)

    # 构建system prompt
    system_prompt = LLM_FUNCTION_SYSTEM_PROMPT.format(
        function_name=func_name,
        function_description=docstring,
        parameters_description="\n\t".join(param_type_descriptions),
        return_type_description=return_type_description,
    )

    # 构建user prompt（只包含参数值）
    user_param_values = []
    for param_name, param_value in arguments.items():
        user_param_values.append(f"- {param_name}: {param_value}")

    user_prompt = LLM_FUNCTION_USER_PROMPT.format(
        function_name=func_name,
        parameters="\n\t".join(user_param_values),
    )

    return system_prompt.strip(), user_prompt.strip()
