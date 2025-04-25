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

# 从utils模块导入工具函数 - 修正导入路径
from SimpleLLMFunc.llm_decorator.utils import (
    execute_with_tools,
    process_response,
)

# 定义一个类型变量，用于函数的返回类型
T = TypeVar("T")


def llm_chat(
    llm_interface: LLM_Interface,
    toolkit: Optional[List[Union[Tool, Callable]]] = None,
    trace_id: Optional[str] = None,
    max_tool_calls: int = 5,  # 最大工具调用次数，防止无限循环
    max_memory_length: int = 50  # 最大记忆对话条目，默认50条
):
    """
    LLM聊天装饰器，用于实现聊天功能
    
    如果被装饰的函数包含 message 参数，这个message会直接被传递到API
    
    否则我们将以 key: value 的形式将入参作为user message传递给API
    
    入参中的 history/chat_history/messages 参数会被跳过传递，而是会被视作自定义的历史记录。
    这些参数需要满足于一定的格式，有两种格式可以选择:
    1. [{"role": "user", "content": "xxx"}]，即具有role和content两个键和字符串值的字典构成的列表
    2. [("user", "xxxx")]，即元组列表，元组的第一个元素是角色，第二个元素是内容
    否则将无法正确解析历史记录。

    Args:
        llm_interface: LLM接口
        tools: 可选的工具列表，可以是Tool对象或被@tool装饰的函数
        system_prompt: 可选的系统提示
        trace_id: 可选的追踪ID，用于日志。如果不指定，会自动生成，也可以通过log_context上下文管理器传递
        max_tool_calls: 最大工具调用次数，防止无限循环，默认为5
        max_memory_length: 最大记忆对话条目，默认50条

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
        
        # 为每个函数创建一个属性字典，用于存储函数特有的状态
        # 这样每个装饰的函数都有自己独立的历史记录
        function_state = {
            "chat_history": []
        }
        
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
            
            # 检查是否有message参数，这是用户的输入消息
            user_message = ""
            message_param = bound_args.arguments.get("message", None)
            
            # 如果有明确的message参数，使用它
            if message_param is not None:
                user_message = str(message_param)
            else:
                # 否则，将所有参数格式化为消息
                param_strings = []
                for param_name, param_value in bound_args.arguments.items():
                    # 跳过可能存在的消息历史参数
                    if param_name in ["history", "chat_history", "messages"]:
                        continue
                    param_strings.append(f"{param_name}: {param_value}")
                
                if param_strings:
                    user_message = "\n".join(param_strings)
                else:
                    # 如果没有参数，使用空消息
                    user_message = ""
            
            # 检查是否传入了自定义消息历史
            custom_history = None
            for history_param in ["history", "chat_history", "messages"]:
                if history_param in bound_args.arguments:
                    custom_history = bound_args.arguments[history_param]
                    break
            
            with log_context(trace_id=current_trace_id):
                # 准备消息列表
                current_messages = []
                
                nonlocal docstring 
                # 添加系统消息
                if docstring != "":
                    docstring += "\n\n此后输出时，请先输出你的思考，然后再输出最终的结果。" 
                    current_messages.append({"role": "system", "content": docstring})
                
                # 使用自定义历史或者函数的专属历史
                if custom_history is not None:
                    # 使用用户提供的历史
                    if isinstance(custom_history, list):
                        # 确保历史是正确的格式
                        formatted_history = []
                        for msg in custom_history:
                            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                formatted_history.append(msg)
                            elif isinstance(msg, tuple) and len(msg) == 2:
                                role, content = msg
                                formatted_history.append({"role": role, "content": content})
                            else:
                                app_log(f"跳过格式不正确的历史消息: {msg}")
                        
                        # 添加格式化后的历史
                        current_messages.extend(formatted_history)
                    else:
                        push_warning(f"提供的历史记录格式不正确，应为列表: {type(custom_history)}")
                else:
                    # 使用函数专属的历史
                    chat_history = function_state["chat_history"]
                    
                    # 如果超过最大长度，裁剪历史
                    if len(chat_history) > max_memory_length:
                        # 保留系统消息
                        system_messages = [msg for msg in chat_history if msg.get("role") == "system"]
                        other_messages = [msg for msg in chat_history if msg.get("role") != "system"]
                        
                        # 保留最近的消息
                        kept_messages = other_messages[-(max_memory_length - len(system_messages)):]
                        chat_history.clear()
                        chat_history.extend(system_messages + kept_messages)
                    
                    current_messages.extend(chat_history)
                
                # 添加当前用户消息
                if user_message:
                    user_msg = {"role": "user", "content": user_message}
                    current_messages.append(user_msg)
                    
                    # 同时添加到历史中(如果没有使用自定义历史)
                    if custom_history is None:
                        function_state["chat_history"].append(user_msg)
                
                # 记录当前消息
                app_log(f"LLM Chat '{func.__name__}' 发送消息列表: {json.dumps(current_messages, ensure_ascii=False)}")
                
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
                            # 添加到工具映射（使用原始函数）
                            tool_map[tool_obj.name] = tool
                        else:
                            push_warning(
                                f"Unsupported tool type: {type(tool)}. Tool must be a Tool object or a function decorated with @tool.",
                                location,
                            )

                    if tool_objects:
                        tool_param = Tool.serialize_tools(tool_objects)

                try:
                    # 调用LLM
                    if tool_param:
                        # 有工具参数，进入工具调用流程
                        response = execute_with_tools(
                            llm_interface=llm_interface,
                            messages=current_messages,
                            tools=tool_param,
                            tool_map=tool_map,
                            max_tool_calls=max_tool_calls,
                            func_name=func.__name__,
                        )
                    else:
                        # 无工具参数，直接调用LLM
                        response = llm_interface.chat(
                            messages=current_messages,
                            trace_id=current_trace_id,
                        )

                    # 记录响应
                    app_log(f"LLM Chat '{func.__name__}' 收到响应: {json.dumps(response, default=str, ensure_ascii=False)}")
                    
                    # 提取响应内容
                    content = process_response(response, str)
                    
                    # 添加到历史(如果没有使用自定义历史)
                    if custom_history is None:
                        assistant_msg = {"role": "assistant", "content": content}
                        function_state["chat_history"].append(assistant_msg)
                    
                    # 根据返回类型处理结果
                    if return_type is None or return_type == str:
                        # 直接返回内容
                        return content
                    elif return_type == dict or str(return_type).startswith("dict"):
                        # 返回结构化响应
                        return {
                            "content": content,
                            "history": list(function_state["chat_history"]),  # 创建历史的副本
                            "raw_response": response
                        }
                    else:
                        # 处理其他返回类型
                        return process_response(response, return_type)

                except Exception as e:
                    # 修复：在log_context环境中不再传递trace_id参数
                    push_warning(
                        f"LLM Chat '{func.__name__}' 遇到错误: {str(e)}",
                        location=location,  # 明确指定为location参数
                    )
                    raise

        # 添加清理历史的方法
        def clear_history():
            function_state["chat_history"].clear()
            return "聊天历史已清除"
        
        # 添加获取历史的方法
        def get_history():
            return list(function_state["chat_history"])  # 返回历史的副本
        
        # 将方法附加到包装函数
        wrapper.clear_history = clear_history
        wrapper.get_history = get_history
        
        # 保留原始函数的元数据
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__

        return wrapper

    return decorator

