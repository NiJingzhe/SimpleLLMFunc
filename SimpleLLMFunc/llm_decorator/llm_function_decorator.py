import inspect
import json
from typing import (
    List,
    Callable,
    TypeVar,
    Dict,
    Any,
    Type,
    get_type_hints,
    Optional,
    Union,
    cast,
    Tuple,
)
import uuid
from pydantic import BaseModel

from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import (
    app_log,
    push_warning,
    get_location,
    log_context,
    push_error,
    push_debug,
    get_current_trace_id,
)

# 定义一个类型变量，用于函数的返回类型
T = TypeVar("T")


def llm_function(
    llm_interface: LLM_Interface,
    tools: Optional[List[Union[Tool, Callable]]] = None,
    system_prompt: Optional[str] = None,
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
                    system_prompt,
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
                        final_response = _execute_with_tools(
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
                    result = _process_response(final_response, return_type)
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


def _execute_with_tools(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    tool_map: Dict[str, Callable],
    max_tool_calls: int,
    func_name: str,
    # 移除trace_id参数，将从上下文获取
) -> Any:
    """
    执行LLM调用并处理工具调用

    Args:
        llm_interface: LLM接口
        messages: 消息历史
        tools: 序列化后的工具参数
        tool_map: 工具名称到函数的映射
        max_tool_calls: 最大工具调用次数
        func_name: 函数名称（用于日志）

    Returns:
        最终的LLM响应
    """
    call_count = 0
    current_messages = messages.copy()

    # 首次调用LLM
    initial_response = llm_interface.chat(
        messages=current_messages,
        tools=tools,
        
    )

    app_log(
        f"LLM Function '{func_name}' received initial response: {json.dumps(initial_response, default=str, ensure_ascii=False)}"
        # 不需要显式传递trace_id
    )

    # 提取初始响应中的工具调用
    tool_calls = _extract_tool_calls(initial_response)

    # 如果没有工具调用，直接返回初始响应
    if not tool_calls:
        app_log(f"No tool calls found in the response, returning directly")
        return initial_response

    # 有工具调用，进入工具调用循环
    app_log(f"Found {len(tool_calls)} tool calls, executing...")

    # 记录首次调用
    call_count += 1

    # 处理初始工具调用
    current_messages = _process_tool_calls(
        tool_calls=tool_calls,
        response=initial_response,
        messages=current_messages,
        tool_map=tool_map,
        # 不需要显式传递trace_id
    )

    # 继续处理可能的后续工具调用
    while call_count < max_tool_calls:
        # 调用LLM获取下一步响应
        response = llm_interface.chat(
            messages=current_messages,
            tools=tools,
            # 不需要显式传递trace_id
        )

        app_log(
            f"LLM tool calling loop: received response (call {call_count+1}/{max_tool_calls}): {json.dumps(response, default=str, ensure_ascii=False)}"
            # 不需要显式传递trace_id
        )

        # 检查是否有更多工具调用
        tool_calls = _extract_tool_calls(response)

        if not tool_calls:
            # 没有更多工具调用，返回最终响应
            push_debug(f"No more tool calls, returning final response")
            return response

        # 处理新的工具调用
        app_log(f"Found {len(tool_calls)} additional tool calls to execute")

        # 处理工具调用并更新消息历史
        current_messages = _process_tool_calls(
            tool_calls=tool_calls,
            response=response,
            messages=current_messages,
            tool_map=tool_map,
            # 不需要显式传递trace_id
        )

        # 增加调用计数
        call_count += 1

    # 如果达到最大调用次数但仍未返回，获取最终结果
    final_response = llm_interface.chat(
        messages=current_messages
        # 不需要显式传递trace_id
    )

    app_log(
        f"Reached maximum tool calls ({max_tool_calls}). Getting final response: {json.dumps(final_response, default=str, ensure_ascii=False)}"
        # 不需要显式传递trace_id
    )

    return final_response


def _process_tool_calls(
    tool_calls: List[Dict[str, Any]],
    response: Any,
    messages: List[Dict[str, Any]],
    tool_map: Dict[str, Callable],
    # 移除trace_id参数，将从上下文获取
) -> List[Dict[str, Any]]:
    """
    处理工具调用并返回更新后的消息历史

    Args:
        tool_calls: 工具调用列表
        response: LLM响应
        messages: 当前消息历史
        tool_map: 工具名称到函数的映射

    Returns:
        更新后的消息历史
    """
    current_messages = messages.copy()

    # 创建助手消息，包含工具调用
    assistant_message = _create_assistant_message(response)
    current_messages.append(assistant_message)

    # 处理每个工具调用
    for tool_call in tool_calls:
        tool_call_id = tool_call.get("id")
        function_call = tool_call.get("function", {})
        tool_name = function_call.get("name")
        arguments_str = function_call.get("arguments", "{}")

        # 检查工具是否存在
        if tool_name not in tool_map:
            push_error(f"Tool '{tool_name}' not found in available tools")
            # 创建工具调用出错的响应
            tool_error_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": f"Tool '{tool_name}' not found"}),
            }
            current_messages.append(tool_error_message)
            continue

        try:
            # 解析参数
            arguments = json.loads(arguments_str)

            # 执行工具
            app_log(f"Executing tool '{tool_name}' with arguments: {arguments_str}")
            tool_func = tool_map[tool_name]
            tool_result = tool_func(**arguments)

            # 创建工具响应消息
            tool_result_str = json.dumps(tool_result, ensure_ascii=False)
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result_str,
            }
            current_messages.append(tool_message)

            app_log(f"Tool '{tool_name}' execution completed: {tool_result_str}")

        except Exception as e:
            # 处理工具执行错误
            error_message = f"Error executing tool '{tool_name}': {str(e)}"
            push_error(error_message)

            # 创建工具错误响应消息
            tool_error_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": error_message}),
            }
            current_messages.append(tool_error_message)

    return current_messages


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """
    从LLM响应中提取工具调用

    Args:
        response: LLM响应

    Returns:
        工具调用列表
    """
    tool_calls = []

    try:
        # 检查是否有tool_calls属性（OpenAI API格式）
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                # 将对象格式转换为字典
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
        # 尝试从字典格式中提取
        elif isinstance(response, dict) and "choices" in response:
            choices = response["choices"]
            if choices and "message" in choices[0]:
                message = choices[0]["message"]
                if "tool_calls" in message and message["tool_calls"]:
                    tool_calls = message["tool_calls"]
    except Exception as e:
        push_error(f"Error extracting tool calls: {str(e)}")

    return tool_calls


def _create_assistant_message(response: Any) -> Dict[str, Any]:
    """
    从LLM响应创建助手消息

    Args:
        response: LLM响应

    Returns:
        助手消息字典
    """
    message = {"role": "assistant"}

    try:
        # 处理对象格式响应
        if hasattr(response, "choices") and len(response.choices) > 0:
            assistant_message = response.choices[0].message

            # 复制content（如果有）
            if hasattr(assistant_message, "content") and assistant_message.content:
                message["content"] = assistant_message.content
            else:
                message["content"] = None

            # 复制tool_calls（如果有）
            if (
                hasattr(assistant_message, "tool_calls")
                and assistant_message.tool_calls
            ):
                message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ]

        # 处理字典格式响应
        elif isinstance(response, dict) and "choices" in response:
            choices = response["choices"]
            if choices and "message" in choices[0]:
                assistant_message = choices[0]["message"]

                # 复制所有字段
                for key, value in assistant_message.items():
                    message[key] = value

    except Exception as e:
        push_error(
            f"Error creating assistant message: {str(e)}",
            trace_id="message_creation_error",
        )
        # 确保至少有content字段
        if "content" not in message:
            message["content"] = ""

    return message


def _build_prompts(
    func_name: str,
    docstring: str,
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    custom_system_prompt: Optional[str],
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
    return_type_description = _get_detailed_type_description(return_type)

    # 构建system prompt
    system_prompt = custom_system_prompt or ""
    system_prompt += f"""
你是一个函数执行助手，你的任务是执行以下函数：

函数: {func_name}

描述:
{docstring}

参数类型:
{chr(10).join(param_type_descriptions)}

期望返回类型: {return_type_description}

请根据用户提供的参数值执行此函数并返回结果。返回格式必须符合指定的返回类型。
如果返回类型是Pydantic模型，请以JSON格式返回符合模型规范的数据。
"""

    # 构建user prompt（只包含参数值）
    user_param_values = []
    for param_name, param_value in arguments.items():
        user_param_values.append(f"- {param_name}: {param_value}")

    user_prompt = f"""
请使用以下参数值执行函数 {func_name}:

{chr(10).join(user_param_values)}

请直接输出函数执行的结果,也不要用任何markdown格式包裹结果。直接输出结果即可。
"""

    return system_prompt.strip(), user_prompt.strip()


def _get_detailed_type_description(type_hint: Any) -> str:
    """
    获取类型的详细描述，特别是对Pydantic模型进行更详细的展开

    Args:
        type_hint: 类型提示

    Returns:
        类型的详细描述
    """
    if type_hint is None:
        return "未知类型"

    # 检查是否为Pydantic模型类
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        model_name = type_hint.__name__
        schema = type_hint.model_json_schema()

        # 提取属性信息
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        fields_desc = []
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "unknown")
            field_desc = field_info.get("description", "")
            is_required = field_name in required

            req_marker = "必填" if is_required else "可选"

            # 如果字段有额外属性，如最小/最大值等，也可以添加
            extra_info = ""
            if "minimum" in field_info:
                extra_info += f", 最小值: {field_info['minimum']}"
            if "maximum" in field_info:
                extra_info += f", 最大值: {field_info['maximum']}"
            if "default" in field_info:
                extra_info += f", 默认值: {field_info['default']}"

            fields_desc.append(
                f"  - {field_name} ({field_type}, {req_marker}): {field_desc}{extra_info}"
            )

        # 构建Pydantic模型的描述
        model_desc = f"{model_name} (Pydantic模型) 包含以下字段:\n" + "\n".join(
            fields_desc
        )
        return model_desc

    # 检查是否为列表或字典类型
    origin = getattr(type_hint, "__origin__", None)
    if origin is list or origin is List:
        args = getattr(type_hint, "__args__", [])
        if args:
            item_type_desc = _get_detailed_type_description(args[0])
            return f"List[{item_type_desc}]"
        return "List"

    if origin is dict or origin is Dict:
        args = getattr(type_hint, "__args__", [])
        if len(args) >= 2:
            key_type_desc = _get_detailed_type_description(args[0])
            value_type_desc = _get_detailed_type_description(args[1])
            return f"Dict[{key_type_desc}, {value_type_desc}]"
        return "Dict"

    # 对于其他类型，简单返回字符串表示
    return str(type_hint)


def _process_response(response: Dict[Any, Any], return_type: Optional[Type[T]]) -> T:
    """
    处理LLM的响应，将其转换为指定的返回类型

    Args:
        response: LLM的响应
        return_type: 期望的返回类型

    Returns:
        转换后的结果
    """
    # 从response中提取内容
    content = ""

    # 检查是否有工具调用但没有被处理
    tool_calls = _extract_tool_calls(response)
    if tool_calls:
        push_warning(
            f"Warning: Response contains unprocessed tool calls. Consider setting auto_tool_execution=True."
        )

    # 处理不同LLM接口返回的响应格式
    try:
        # 处理返回的是OpenAI API响应对象的情况
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            content = message.content if message and hasattr(message, "content") else ""
        # 处理其他情况
        else:
            push_warning(f"未知的响应格式: {type(response)}")
            # 尝试转换为字符串
            content = str(response)
    except Exception as e:
        push_error(f"提取响应内容时出错: {str(e)}")
        # 尝试将整个响应转换为字符串
        content = str(response)

    app_log(f"提取的内容: {content}")

    # 如果内容为None，转换为空字符串
    if content is None:
        content = ""

    # 如果没有返回类型或返回类型是str，直接返回内容
    if return_type is None or return_type == str:
        return cast(T, content)

    # 如果返回类型是基本类型，尝试转换
    if return_type in (int, float, bool):
        try:
            if return_type == int:
                return cast(T, int(content.strip()))
            elif return_type == float:
                return cast(T, float(content.strip()))
            elif return_type == bool:
                return cast(T, content.strip().lower() in ("true", "yes", "1"))
        except (ValueError, TypeError):
            raise ValueError(
                f"无法将LLM响应 '{content}' 转换为 {return_type.__name__} 类型"
            )

    # 如果返回类型是字典，尝试解析JSON
    if return_type == dict or getattr(return_type, "__origin__", None) is dict:
        try:
            # 尝试从内容中提取JSON
            # 首先尝试直接解析
            try:
                result = json.loads(content)
                return cast(T, result)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试查找内容中的JSON部分
                import re

                json_pattern = r"```json\s*([\s\S]*?)\s*```"
                match = re.search(json_pattern, content)
                if match:
                    json_str = match.group(1)
                    result = json.loads(json_str)
                    return cast(T, result)
                else:
                    # 如果没有找到JSON块，再次尝试直接解析
                    # 这次做一些清理
                    cleaned_content = content.strip()
                    # 移除可能的 markdown 标记
                    if cleaned_content.startswith("```") and cleaned_content.endswith(
                        "```"
                    ):
                        cleaned_content = cleaned_content[3:-3].strip()
                    result = json.loads(cleaned_content)
                    return cast(T, result)
        except json.JSONDecodeError:
            raise ValueError(f"无法将LLM响应解析为有效的JSON: {content}")

    # 如果返回类型是Pydantic模型，使用model_validate_json解析
    if return_type and hasattr(return_type, "model_validate_json"):
        try:
            # 处理可能的JSON字符串转义问题
            # 首先尝试直接解析内容
            try:
                # 关键修改：首先尝试解析为Python对象，然后再转换为JSON字符串
                # 这样可以处理内容中的转义字符问题
                if content.strip():
                    # 尝试先解析内容中的JSON，然后再转换为标准JSON字符串
                    try:
                        # 这里处理内容可能是字符串形式的JSON对象
                        parsed_content = json.loads(content)
                        # 将解析后的对象重新转换为标准JSON字符串
                        clean_json_str = json.dumps(parsed_content)
                        return return_type.model_validate_json(clean_json_str)
                    except json.JSONDecodeError:
                        # 如果直接解析失败，尝试查找内容中的JSON部分
                        import re

                        json_pattern = r"```json\s*([\s\S]*?)\s*```"
                        match = re.search(json_pattern, content)
                        if match:
                            json_str = match.group(1)
                            # 确保这是有效的JSON
                            parsed_json = json.loads(json_str)
                            clean_json_str = json.dumps(parsed_json)
                            return return_type.model_validate_json(clean_json_str)
                        else:
                            # 如果没有找到JSON块，尝试使用原始内容
                            return return_type.model_validate_json(content)
                else:
                    raise ValueError("收到空响应")
            except Exception as e:
                push_error(f"解析错误详情: {str(e)}, 内容: {content}")
                raise ValueError(f"无法解析JSON: {str(e)}")
        except Exception as e:
            raise ValueError(f"无法将LLM响应解析为Pydantic模型: {str(e)}")

    # 最后尝试直接转换
    try:
        return cast(T, content)
    except (ValueError, TypeError):
        raise ValueError(f"无法将LLM响应转换为所需类型: {content}")
