"""
这里定义了每种LLM装饰器都会需要用到的通用函数

1. excute_with_tools: 执行LLM调用并处理工具调用
2. get_detailed_type_description: 获取类型的详细描述，特别是对Pydantic模型进行更详细的展开
3. process_response: 处理LLM的响应，将其转换为指定的返回类型

"""

import json
from typing import List, Dict, Any, Type, Optional, TypeVar, cast, Callable
from pydantic import BaseModel

from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import (
    app_log,
    push_warning,
    push_error,
    push_debug,
)
from SimpleLLMFunc.logger.logger import get_current_context_attribute, get_location

# 定义一个类型变量，用于函数的返回类型
T = TypeVar("T")

# ======================================= 以下函数会被导出使用 =============================================

def execute_llm(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]] | None,
    tool_map: Dict[str, Callable],
    max_tool_calls: int,
) -> Any:
    """
    执行LLM调用，会自动处理Tool Use，最终返回一个包含返回文本的LLM Response

    Args:
        llm_interface: LLM接口
        messages: 消息历史, 会被直接传递给LLM Interface
        tools: 序列化后的工具信息，会被直接传递给LLM Interface
        tool_map: 工具名称到实际实现工具功能函数的映射
        max_tool_calls: 最大工具调用次数
        func_name: 函数名称（用于日志）

    Returns:
        最终的LLM响应
    """

    func_name = get_current_context_attribute("function_name")
    
    if func_name is None:
        func_name = "Unknow Function"
    
    call_count = 0
    # copy 是为了不影响传入的message，主要是为了避免加入很多Tool Use迭代的消息。
    current_messages = messages.copy()

    # 发送LLM请求，首次
    initial_response = llm_interface.chat(
        messages=current_messages,
        tools=tools,
    )

    # initial response 可能是一个Tool Use Response

    app_log(
        f"LLM Function '{func_name}' received initial response: {json.dumps(initial_response, default=str, ensure_ascii=False)}",
        location=get_location()
    )

    # 提取初始响应中的工具调用
    tool_calls = _extract_tool_calls(initial_response)

    # 如果没有工具调用，直接返回初始响应
    if not tool_calls:
        push_debug(f"No tool calls found in the response, returning directly", location=get_location())
        return initial_response  #  <--------------------------------------------------------------------------- 第一个返回分支

    # 有工具调用，进入工具调用循环
    app_log(
        f"LLM Function '{func_name}' Found {len(tool_calls)} tool calls, executing..."
    )

    # 记录首次调用
    call_count += 1

    # 处理初始工具调用，执行工具并将执行结果添加到message副本中
    current_messages = _process_tool_calls(
        tool_calls=tool_calls,
        response=initial_response,
        messages=current_messages,
        tool_map=tool_map,
    )

    # 继续处理可能的后续工具调用
    while call_count < max_tool_calls:
        # 将包含工具调用结果的消息发送给LLM
        response = llm_interface.chat(
            messages=current_messages,
            tools=tools,
        )

        app_log(
            (
                f"LLM Function: '{func_name}': LLM tool calling loop:"
                f" received response (call {call_count+1}/{max_tool_calls}):"
                f"\n{json.dumps(response, default=str, ensure_ascii=False, indent=4)}"
            ),
            location=get_location()
        )

        # 检查是否有更多工具调用
        tool_calls = _extract_tool_calls(response)

        if not tool_calls:
            # 没有更多工具调用，返回最终响应
            push_debug(f"LLM Function '{func_name}': No more tool calls, returning final response", location=get_location())
            return response  #  <------------------------------------------------------------------------------ 第二个返回分支

        # 处理新的工具调用
        app_log(f"LLM Function '{func_name}' Found {len(tool_calls)} additional tool calls to execute", location=get_location())

        # 处理工具调用并更新消息历史
        current_messages = _process_tool_calls(
            tool_calls=tool_calls,
            response=response,
            messages=current_messages,
            tool_map=tool_map,
        )

        # 增加调用计数
        call_count += 1

    # 如果达到最大调用次数但仍未返回，获取最终结果
    final_response = llm_interface.chat(messages=current_messages)

    app_log(
        (
            f"LLM Function '{func_name}' Reached maximum tool calls ({max_tool_calls})."
            f" Getting final response:"
            f"\n{json.dumps(final_response, default=str, ensure_ascii=False, indent=4)}"
        ),
        location=get_location()
    )

    return final_response  #    #  <--------------------------------------------------------------------------- 第三个返回分支



def process_response(response: Dict[Any, Any], return_type: Optional[Type[T]]) -> T:
    """
    处理LLM的响应，将其转换为指定的返回类型

    Args:
        response: LLM的响应
        return_type: 期望的返回类型

    Returns:
        转换后的结果
    """
    func_name = get_current_context_attribute("function_name")
    
    if func_name is None:
        func_name = "Unknown Function"
    
    # 从response中提取内容
    content = ""

    # 检查是否有工具调用但没有被处理
    # 因为我们一定会先执行 execute_with_tools
    # 这几乎可以保证返回的内容中不包含工具调用响应
    # 除非超过Tool Use的最大轮次但是依然在使用工具
    
    # 但是如果你单独执行了这个函数，那么这个检查就非常有必要了
    tool_calls = _extract_tool_calls(response)
    if tool_calls:
        push_warning(
            f"LLM Function '{func_name}' Warning: Response contains unprocessed tool calls. Consider setting auto_tool_execution=True."
        )

    # 从API Response中提取文本内容
    try:
        if hasattr(response, "choices") and len(response.choices) > 0: # type: ignore
            message = response.choices[0].message  # type: ignore
            content = message.content if message and hasattr(message, "content") else ""
        # 处理其他情况
        else:
            push_warning(
                f"LLM Function '{func_name}': Unknown response format: {type(response)},"
                " response would be directly converted into string", 
                location=get_location()
            )
            # 尝试转换为字符串
            content = str(response)
    except Exception as e:
        push_error(f"提取响应内容时出错: {str(e)}")
        # 尝试将整个响应转换为字符串
        content = str(response)

    push_debug(f"LLM Function '{func_name}' Extracted Content:\n{content}")

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
                        return return_type.model_validate_json(clean_json_str)  # type: ignore
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
                            return return_type.model_validate_json(clean_json_str)  # type: ignore
                        else:
                            # 如果没有找到JSON块，尝试使用原始内容
                            return return_type.model_validate_json(content)  # type: ignore
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



def get_detailed_type_description(type_hint: Any) -> str:
    """
    获取类型的详细描述，特别是对Pydantic模型进行更详细的展开
    
    可以考虑用来获得类型注解的详细信息，拼接在prompt中

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
            item_type_desc = get_detailed_type_description(args[0])
            return f"List[{item_type_desc}]"
        return "List"

    if origin is dict or origin is Dict:
        args = getattr(type_hint, "__args__", [])
        if len(args) >= 2:
            key_type_desc = get_detailed_type_description(args[0])
            value_type_desc = get_detailed_type_description(args[1])
            return f"Dict[{key_type_desc}, {value_type_desc}]"
        return "Dict"

    # 对于其他类型，简单返回字符串表示
    return str(type_hint)

# ======================================= 以上函数会被导出使用 =============================================


# ======================================= 以下函数是内部工具函数 =============================================
def _process_tool_calls(
    tool_calls: List[Dict[str, Any]],
    response: Any,
    messages: List[Dict[str, Any]],
    tool_map: Dict[str, Callable],
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
            error_message = f"Error executing tool '{tool_name}' with arguments {arguments_str} : {str(e)}"
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
    message: Dict[str, Any] = {"role": "assistant"}

    try:
        # 处理对象格式响应
        if hasattr(response, "choices") and len(response.choices) > 0:
            assistant_message = response.choices[0].message

            # 复制content（如果有）
            if hasattr(assistant_message, "content") and assistant_message.content:
                message["content"] = assistant_message.content
            else:
                message["content"] = ""

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





__all__ = [
    "execute_llm",
    "get_detailed_type_description",
    "process_response",
]
