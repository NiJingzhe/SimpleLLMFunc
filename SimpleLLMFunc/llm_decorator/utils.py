"""
LLM 装饰器通用工具函数模块

本模块提供了 LLM 装饰器所需的核心功能，包括：
1. LLM 执行与工具调用 - 处理与 LLM 的交互和工具调用逻辑
2. 响应处理 - 将 LLM 响应转换为所需的返回类型
3. 类型描述 - 获取类型的详细描述，特别是对 Pydantic 模型进行展开

这些功能被设计为相互独立的组件，每个组件负责特定的职责。
"""

import json
from typing import List, Dict, Any, Type, Optional, TypeVar, cast, Callable, AsyncGenerator
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

# ======================= 数据流相关函数 =======================


async def execute_llm(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]] | None,
    tool_map: Dict[str, Callable],
    max_tool_calls: int,
    stream: bool = False,
    **llm_kwargs,  # 添加llm_kwargs参数接收额外的LLM配置
) -> AsyncGenerator[Any, None]:
    """
    执行 LLM 调用并处理工具调用流程

    数据流程:
    1. 以初始消息列表调用 LLM
    2. 检查响应中是否包含工具调用
    3. 如有工具调用，执行工具并将结果添加到消息列表
    4. 使用更新后的消息列表再次调用 LLM
    5. 重复步骤 2-4 直到没有更多工具调用或达到最大调用次数
    6. 返回最终响应

    Args:
        llm_interface: LLM 接口
        messages: 初始消息历史，将直接传递给 LLM API
        tools: 序列化后的工具信息，将传递给 LLM API
        tool_map: 工具名称到实际实现函数的映射
        max_tool_calls: 最大工具调用次数，防止无限循环
        **llm_kwargs: 额外的关键字参数，将直接传递给LLM接口

    Returns:
        生成器，产生 LLM 响应，最后一个响应是最终结果
    """
    func_name = get_current_context_attribute("function_name") or "Unknown Function"

    # 创建消息历史副本，避免修改原始消息列表
    current_messages = list(messages)

    # 记录调用次数
    call_count = 0

    # 第一次调用 LLM，获取初始响应
    push_debug(
        f"LLM 函数 '{func_name}' 将要发起初始请求，消息数: {len(current_messages)}",
        location=get_location(),
    )

    # 如果 stream 为 True，使用流式响应
    if stream:
        push_debug(f"LLM 函数 '{func_name}' 使用流式响应", location=get_location())
        
        push_debug(
            f"LLM 函数 '{func_name}' 初始流式响应开始", location=get_location()
        )

        # 处理流式响应
        content = ""
        tool_call_chunks = []  # 累积工具调用片段
        # 流式响应：在一次遍历中同时提取内容和工具调用片段
        async for chunk in llm_interface.chat_stream(
            messages=current_messages,
            tools=tools,
            **llm_kwargs,  # 传递额外的关键字参数
        ):
            content += extract_content_from_stream_response(chunk, func_name)
            tool_call_chunks.extend(_extract_tool_calls_from_stream_response(chunk))
            yield chunk  # 如果是流式响应，逐个返回 chunk
        # 合并工具调用片段为完整的工具调用
        tool_calls = _accumulate_tool_calls_from_chunks(tool_call_chunks)
    else:
        push_debug(f"LLM 函数 '{func_name}' 使用非流式响应", location=get_location())
        # 使用非流式响应
        initial_response = await llm_interface.chat(
            messages=current_messages,
            tools=tools,
            **llm_kwargs,  # 传递额外的关键字参数
        )

        push_debug(
            f"LLM 函数 '{func_name}' 初始响应: {initial_response}", location=get_location()
        )

        # 处理非流式响应
        content = extract_content_from_response(initial_response, func_name)
        tool_calls = _extract_tool_calls(initial_response)
        yield initial_response

    push_debug(
        f"LLM 函数 '{func_name}' 初始响应中抽取的content是: {content}",
        location=get_location(),
    )

    # 根据content是否为空决定是否要构造助手中间输出
    if content.strip() != "":
        assistant_message = _build_assistant_response_message(content)
        current_messages.append(assistant_message)

    # 根据工具调用是否为空决定是否要构造助手工具调用消息
    if len(tool_calls) != 0:
        assistant_tool_call_message = _build_assistant_tool_message(tool_calls)
        current_messages.append(assistant_tool_call_message)
    else:
        push_debug(f"未发现工具调用，直接返回结果", location=get_location())
        # app_log 记录全过程messages
        app_log(
            f"LLM 函数 '{func_name}' 本次调用的完整messages: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
            location=get_location(),
        )
        return

    push_debug(
        f"LLM 函数 '{func_name}' 抽取工具后构建的完整消息: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
        location=get_location(),
    )

    # === 工具调用循环 ===
    push_debug(
        f"LLM 函数 '{func_name}' 发现 {len(tool_calls)} 个工具调用，开始执行工具",
        location=get_location(),
    )

    # 记录首次调用
    call_count += 1

    # 处理初始工具调用，执行工具并将结果添加到消息历史
    current_messages = _process_tool_calls(
        tool_calls=tool_calls,
        messages=current_messages,
        tool_map=tool_map,
    )

    # 继续处理可能的后续工具调用
    while call_count < max_tool_calls:
        push_debug(
            f"LLM 函数 '{func_name}' 工具调用循环: 第 {call_count}/{max_tool_calls} 次返回工具响应",
            location=get_location(),
        )

        # 使用更新后的消息历史再次调用 LLM
        # 如果 stream 为 True，使用流式响应
        if stream:
            push_debug(f"LLM 函数 '{func_name}' 使用流式响应", location=get_location())

            push_debug(
                f"LLM 函数 '{func_name}' 第 {call_count} 次工具调用返回后，LLM流式响应开始",
                location=get_location(),
            )

            # 处理流式响应
            content = ""
            tool_call_chunks = []  # 累积工具调用片段
            # 流式响应：在一次遍历中同时提取内容和工具调用片段
            async for chunk in llm_interface.chat_stream(
                messages=current_messages,
                tools=tools,
                **llm_kwargs,  # 传递额外的关键字参数
            ):
                content += extract_content_from_stream_response(chunk, func_name)
                tool_call_chunks.extend(_extract_tool_calls_from_stream_response(chunk))
                yield chunk  # 如果是流式响应，逐个返回 chunk
            # 合并工具调用片段为完整的工具调用
            tool_calls = _accumulate_tool_calls_from_chunks(tool_call_chunks)
        else:
            push_debug(
                f"LLM 函数 '{func_name}' 使用非流式响应", location=get_location()
            )
            # 使用非流式响应
            response = await llm_interface.chat(
                messages=current_messages,
                tools=tools,
                **llm_kwargs,  # 传递额外的关键字参数
            )

            push_debug(
                f"LLM 函数 '{func_name}' 第 {call_count} 次工具调用返回后，LLM，响应: {response}",
                location=get_location(),
            )

            # 处理非流式响应
            content = extract_content_from_response(response, func_name)
            tool_calls = _extract_tool_calls(response)
            yield response

        push_debug(
            f"LLM 函数 '{func_name}' 初始响应中抽取的content是: {content}",
            location=get_location(),
        )

        # 根据content是否为空决定是否要构造助手中间输出
        if content.strip() != "":
            assistant_message = _build_assistant_response_message(content)
            current_messages.append(assistant_message)

        # 将助手消息添加到消息历史
        if len(tool_calls) != 0:
            assistant_tool_call_message = _build_assistant_tool_message(tool_calls)
            current_messages.append(assistant_tool_call_message)

        push_debug(
            f"LLM 函数 '{func_name}' 抽取工具后构建的完整消息: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
            location=get_location(),
        )

        if len(tool_calls) == 0:
            # 没有更多工具调用，返回最终响应
            push_debug(
                f"LLM 函数 '{func_name}' 没有更多工具调用，返回最终响应",
                location=get_location(),
            )
            app_log(
                f"LLM 函数 '{func_name}' 本次调用的完整messages: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
                location=get_location(),
            )
            return

        # 处理新的工具调用
        push_debug(
            f"LLM 函数 '{func_name}' 发现 {len(tool_calls)} 个新的工具调用",
            location=get_location(),
        )

        # 处理工具调用并更新消息历史
        current_messages = _process_tool_calls(
            tool_calls=tool_calls,
            messages=current_messages,
            tool_map=tool_map,
        )

        # 增加调用计数
        call_count += 1

    # 如果达到最大调用次数但仍未完成所有工具调用
    push_debug(
        f"LLM 函数 '{func_name}' 达到最大工具调用次数 ({max_tool_calls})，强制结束并获取最终响应",
        location=get_location(),
    )

    # 最后一次调用 LLM 获取最终结果
    final_response = await llm_interface.chat(
        messages=current_messages,
        **llm_kwargs,  # 传递额外的关键字参数
    )
    
    # app_log 记录全过程messages
    app_log(
        f"LLM 函数 '{func_name}' 本次调用的完整messages: {json.dumps(current_messages, ensure_ascii=False, indent=2)}",
        location=get_location(),
    )

    # 产生最终响应
    yield final_response


def process_response(response: Any, return_type: Optional[Type[T]]) -> T:
    """
    处理 LLM 的响应，将其转换为指定的返回类型

    数据流程:
    1. 从 LLM 响应中提取纯文本内容
    2. 根据指定的返回类型进行相应转换:
       - 基本类型 (str, int, float, bool): 直接转换
       - 字典类型: 解析 JSON
       - Pydantic 模型: 使用 model_validate_json 解析

    Args:
        response: LLM 的原始响应对象
        return_type: 期望的返回类型

    Returns:
        转换后的结果，类型为 T
    """
    func_name = get_current_context_attribute("function_name") or "Unknown Function"

    # 步骤 1: 从 API 响应中提取文本内容
    content = extract_content_from_response(response, func_name)

    # 步骤 2: 根据返回类型进行适当的转换
    # 如果内容为 None，转换为空字符串
    if content is None:
        content = ""

    # 如果没有返回类型或返回类型是 str，直接返回内容
    if return_type is None or return_type == str:
        return cast(T, content)

    # 如果返回类型是基本类型，尝试转换
    if return_type in (int, float, bool):
        return _convert_to_primitive_type(content, return_type)

    # 如果返回类型是字典，尝试解析 JSON
    if return_type == dict or getattr(return_type, "__origin__", None) is dict:
        return _convert_to_dict(content, func_name)  # type: ignore

    # 如果返回类型是 Pydantic 模型，使用 model_validate_json 解析
    if return_type and hasattr(return_type, "model_validate_json"):
        return _convert_to_pydantic_model(content, return_type, func_name)

    # 最后尝试直接转换
    try:
        return cast(T, content)
    except (ValueError, TypeError):
        raise ValueError(f"无法将 LLM 响应转换为所需类型: {content}")


def get_detailed_type_description(type_hint: Any) -> str:
    """
    获取类型的详细描述，特别是对 Pydantic 模型进行更详细的展开

    这个函数用于生成类型的人类可读描述，以便在提示中使用。
    对于 Pydantic 模型，会展开其字段结构；对于容器类型，会递归描述其元素类型。

    Args:
        type_hint: 类型提示对象

    Returns:
        类型的详细描述字符串
    """
    if type_hint is None:
        return "未知类型"

    # 检查是否为 Pydantic 模型类
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        return _describe_pydantic_model(type_hint)

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


def extract_content_from_response(response: Any, func_name: str) -> str:
    """从 LLM 响应中提取文本内容"""
    content = ""
    try:
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            if message and hasattr(message, "content") and message.content is not None:
                content = message.content
            else:
                content = ""
        else:
            push_error(
                f"LLM 函数 '{func_name}': 未知响应格式: {type(response)}，将直接转换为字符串",
                location=get_location(),
            )
            content = ""
    except Exception as e:
        push_error(f"提取响应内容时出错: {str(e)}")
        content = ""

    push_debug(f"LLM 函数 '{func_name}' 提取的内容:\n{content}")
    return content


def extract_content_from_stream_response(chunk: Any, func_name: str) -> str:
    """从流返回中抽取一个chunk的内容

    Args:
        chunk (Any): 流响应chunk对象, chunk对象的内容在delta中
        func_name (str): 函数名称

    Returns:
        str: 提取的文本内容
    """

    content = ""  # 初始化内容为空字符串
    if not chunk:
        push_warning(
            f"LLM 函数 '{func_name}': 检测到空的流响应 chunk，返回空字符串",
            location=get_location(),
        )
        return content
    try:
        # 检查是否为OpenAI ChatCompletionChunk格式
        if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            # 检查是否有delta属性（流式响应）
            if hasattr(choice, "delta") and choice.delta:
                delta = choice.delta
                if hasattr(delta, "content") and delta.content is not None:
                    content = delta.content
                else:
                    content = ""
            else:
                content = ""
        else:
            # 尝试其他可能的格式
            push_debug(
                f"LLM 函数 '{func_name}': 检测到流响应格式: {type(chunk)}，内容为: {chunk}，预估不包含content，将会返回空串",
                location=get_location(),
            )
            content = ""
    except Exception as e:
        push_error(f"提取流响应内容时出错: {str(e)}")
        content = ""

    push_debug(f"LLM 函数 '{func_name}' 提取的流内容:\n{content}")
    return content


# ======================= 类型转换辅助函数 =======================


def _convert_to_primitive_type(content: str, return_type: Type) -> Any:
    """将文本内容转换为基本类型 (int, float, bool)"""
    try:
        if return_type == int:
            return int(content.strip())
        elif return_type == float:
            return float(content.strip())
        elif return_type == bool:
            return content.strip().lower() in ("true", "yes", "1")
    except (ValueError, TypeError):
        raise ValueError(
            f"无法将 LLM 响应 '{content}' 转换为 {return_type.__name__} 类型"
        )


def _convert_to_dict(content: str, func_name: str) -> Dict:
    """将文本内容转换为字典 (解析 JSON)"""
    try:
        # 首先尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试查找内容中的 JSON 部分
            import re

            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, content)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            else:
                # 尝试清理后再解析
                cleaned_content = content.strip()
                # 移除可能的 markdown 标记
                if cleaned_content.startswith("```") and cleaned_content.endswith(
                    "```"
                ):
                    cleaned_content = cleaned_content[3:-3].strip()
                return json.loads(cleaned_content)
    except json.JSONDecodeError:
        raise ValueError(f"无法将 LLM 响应解析为有效的 JSON: {content}")


def _convert_to_pydantic_model(content: str, model_class: Type, func_name: str) -> Any:
    """将文本内容转换为 Pydantic 模型"""
    try:
        if content.strip():
            try:
                # 先解析内容中的 JSON，然后再转换为标准 JSON 字符串
                parsed_content = json.loads(content)
                clean_json_str = json.dumps(parsed_content)
                return model_class.model_validate_json(clean_json_str)
            except json.JSONDecodeError:
                # 尝试查找内容中的 JSON 部分
                import re

                json_pattern = r"```json\s*([\s\S]*?)\s*```"
                match = re.search(json_pattern, content)
                if match:
                    json_str = match.group(1)
                    parsed_json = json.loads(json_str)
                    clean_json_str = json.dumps(parsed_json)
                    return model_class.model_validate_json(clean_json_str)
                else:
                    # 尝试使用原始内容
                    return model_class.model_validate_json(content)
        else:
            raise ValueError("收到空响应")
    except Exception as e:
        push_error(f"解析错误详情: {str(e)}, 内容: {content}")
        raise ValueError(f"无法解析为 Pydantic 模型: {str(e)}")


def _describe_pydantic_model(model_class: Type[BaseModel]) -> str:
    """生成 Pydantic 模型的详细描述"""
    model_name = model_class.__name__
    schema = model_class.model_json_schema()

    # 提取属性信息
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields_desc = []
    for field_name, field_info in properties.items():
        field_type = field_info.get("type", "unknown")
        field_desc = field_info.get("description", "")
        is_required = field_name in required

        req_marker = "必填" if is_required else "可选"

        # 添加额外属性信息
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

    # 构建 Pydantic 模型的描述
    model_desc = f"{model_name} (Pydantic模型) 包含以下字段:\n" + "\n".join(fields_desc)
    return model_desc


# ======================= 类型转换辅助函数 =======================

# ======================= 工具调用处理函数 =======================


def _process_tool_calls(
    tool_calls: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    tool_map: Dict[str, Callable],
) -> List[Dict[str, Any]]:
    """
    处理工具调用并返回更新后的消息历史

    工作流程:
    1. 为每个工具调用创建一个助手消息
    2. 对每个工具调用:
       a. 提取工具名称和参数
       b. 检查工具是否存在
       c. 执行工具并获取结果
       d. 创建工具响应消息并添加到消息历史

    Args:
        tool_calls: 工具调用列表
        response: LLM 响应
        messages: 当前消息历史
        tool_map: 工具名称到函数的映射

    Returns:
        更新后的消息历史
    """
    # 创建消息历史副本
    current_messages = list(messages)

    # 处理每个工具调用
    for tool_call in tool_calls:
        tool_call_id = tool_call.get("id")
        function_call = tool_call.get("function", {})
        tool_name = function_call.get("name")
        arguments_str = function_call.get("arguments", "{}")

        # 检查工具是否存在
        if tool_name not in tool_map:
            push_error(f"工具 '{tool_name}' 不在可用工具列表中")
            # 创建工具调用出错的响应
            tool_error_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": f"找不到工具 '{tool_name}'"}, ensure_ascii=False, indent=2),
            }
            current_messages.append(tool_error_message)
            continue

        try:
            # 解析参数
            arguments = json.loads(arguments_str)

            # 执行工具
            push_debug(f"执行工具 '{tool_name}' 参数: {arguments_str}")
            tool_func = tool_map[tool_name]
            tool_result = tool_func(**arguments)

            # 创建工具响应消息
            tool_result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result_str,
            }
            current_messages.append(tool_message)

            push_debug(f"工具 '{tool_name}' 执行完成: {tool_result_str}")

        except Exception as e:
            # 处理工具执行错误
            error_message = (
                f"执行工具 '{tool_name}' 出错，参数 {arguments_str}: {str(e)}"
            )
            push_error(error_message)

            # 创建工具错误响应消息
            tool_error_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": error_message}, ensure_ascii=False, indent=2),
            }
            current_messages.append(tool_error_message)

    return current_messages


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """
    从 LLM 响应中提取工具调用

    Args:
        response: LLM 响应

    Returns:
        工具调用列表
    """
    tool_calls = []

    try:
        # 检查对象格式 (OpenAI API 格式)
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                # 将对象格式转换为字典
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": getattr(tool_call, "type", "function"),
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
    except Exception as e:
        push_error(f"提取工具调用时出错: {str(e)}")
    finally:
        return tool_calls


def _accumulate_tool_calls_from_chunks(
    tool_call_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    累积并合并流式响应中的工具调用片段

    在流式响应中，工具调用信息会分散在多个chunk中：
    - 第一个chunk可能包含id和type
    - 后续chunks包含function name和arguments的不同部分

    Args:
        tool_call_chunks: 从多个chunk中提取的工具调用片段列表

    Returns:
        合并后的完整工具调用列表
    """
    # 使用字典来按index累积工具调用
    accumulated_calls = {}

    for chunk in tool_call_chunks:
        index = chunk.get("index")
        if index is None:
            push_warning(
                "工具调用 chunk 缺少 'index' 属性，已跳过处理", location=get_location()
            )
            continue

        if index not in accumulated_calls:
            accumulated_calls[index] = {
                "id": None,
                "type": None,
                "function": {"name": None, "arguments": ""},
            }

        # 累积基本信息
        if chunk.get("id"):
            accumulated_calls[index]["id"] = chunk["id"]
        if chunk.get("type"):
            accumulated_calls[index]["type"] = chunk["type"]

        # 累积function信息
        if "function" in chunk:
            function_chunk = chunk["function"]
            if function_chunk.get("name"):
                accumulated_calls[index]["function"]["name"] = function_chunk["name"]
            if function_chunk.get("arguments"):
                # 累积arguments字符串
                accumulated_calls[index]["function"]["arguments"] += function_chunk[
                    "arguments"
                ]

    # 过滤出完整的工具调用（至少有id和function name）
    complete_tool_calls = []
    for call in accumulated_calls.values():
        if call["id"] and call["function"]["name"]:
            # 设置默认type
            if not call["type"]:
                call["type"] = "function"
            complete_tool_calls.append(call)

    return complete_tool_calls


def _extract_tool_calls_from_stream_response(chunk: Any) -> List[Dict[str, Any]]:
    """
    从流响应中提取工具调用片段

    注意：流式响应中工具调用信息会分散在多个chunk中，
    这个函数只提取当前chunk中的部分信息，需要在上层进行累积。

    Args:
        chunk: 流响应的一个 chunk

    Returns:
        工具调用片段列表，每个元素包含当前chunk的部分信息
    """
    tool_call_chunks = []

    try:
        # 检查对象格式 (OpenAI API 格式)
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                delta = choice.delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    # 将对象格式转换为字典，保留chunk中的所有信息
                    for tool_call in delta.tool_calls:
                        tool_call_chunk = {
                            "index": getattr(tool_call, "index", None),
                            "id": getattr(tool_call, "id", None),
                            "type": getattr(tool_call, "type", None),
                        }

                        # 处理 function 部分
                        if hasattr(tool_call, "function") and tool_call.function:
                            function_info = {}
                            if (
                                hasattr(tool_call.function, "name")
                                and tool_call.function.name
                            ):
                                function_info["name"] = tool_call.function.name
                            if (
                                hasattr(tool_call.function, "arguments")
                                and tool_call.function.arguments
                            ):
                                function_info["arguments"] = (
                                    tool_call.function.arguments
                                )

                            if function_info:
                                tool_call_chunk["function"] = function_info

                        tool_call_chunks.append(tool_call_chunk)
    except Exception as e:
        push_error(f"提取流工具调用时出错: {str(e)}")

    return tool_call_chunks


def _build_assistant_tool_message(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    构造 assistant message，包含 tool_calls 字段
    """
    if tool_calls:
        return {"role": "assistant", "content": None, "tool_calls": tool_calls}
    else:
        return {}


def _build_assistant_response_message(content: str) -> Dict[str, Any]:
    """
    构造 assistant message，包含 content 和 tool_calls 字段
    """
    return {
        "role": "assistant",
        "content": content,
    }


# ======================= 工具调用处理函数 =======================


# 导出公共函数
__all__ = [
    "execute_llm",
    "get_detailed_type_description",
    "process_response",
    "extract_content_from_response",
    "extract_content_from_stream_response",
]
