# Helper Function 详细设计文档

本文档详细分析每个步骤函数的具体实现，提取公共函数，并给出文件规划。

---

## Step 1: `parse_function_signature`

### 职责
解析函数签名，提取所有元数据（函数名、参数类型、返回类型、docstring、绑定参数等）

### 具体执行步骤

#### 1.1 提取模板参数
**函数名**: `extract_template_params`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/signature.py`

**功能**: 从 kwargs 中提取并移除 `_template_params` 参数

**Pseudo Code**:
```python
def extract_template_params(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从 kwargs 中提取模板参数
    
    Args:
        kwargs: 函数的关键字参数字典（会被修改）
    
    Returns:
        模板参数字典，如果不存在则返回 None
    """
    return kwargs.pop("_template_params", None)
```

---

#### 1.2 提取函数元数据
**函数名**: `extract_function_metadata`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/signature.py`

**功能**: 从函数对象中提取签名、类型提示、docstring、函数名

**Pseudo Code**:
```python
def extract_function_metadata(func: Callable) -> Tuple[
    inspect.Signature,
    Dict[str, Any],  # type_hints
    Any,  # return_type
    str,  # docstring
    str,  # func_name
]:
    """
    提取函数的元数据
    
    Returns:
        (signature, type_hints, return_type, docstring, func_name)
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return")
    docstring = func.__doc__ or ""
    func_name = func.__name__
    
    return signature, type_hints, return_type, docstring, func_name
```

---

#### 1.3 生成追踪 ID
**函数名**: `generate_trace_id`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/signature.py`

**功能**: 生成唯一的 trace_id

**Pseudo Code**:
```python
def generate_trace_id(func_name: str) -> str:
    """
    生成唯一的追踪 ID
    
    Args:
        func_name: 函数名
    
    Returns:
        格式: {func_name}_{uuid}_{parent_trace_id}
    """
    context_trace_id = get_current_trace_id()
    current_trace_id = f"{func_name}_{uuid.uuid4()}"
    if context_trace_id:
        current_trace_id += f"_{context_trace_id}"
    return current_trace_id
```

---

#### 1.4 绑定函数参数
**函数名**: `bind_function_arguments`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/signature.py`

**功能**: 将 args 和 kwargs 绑定到函数签名，并应用默认值

**Pseudo Code**:
```python
def bind_function_arguments(
    signature: inspect.Signature,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> inspect.BoundArguments:
    """
    绑定函数参数并应用默认值
    
    Returns:
        绑定后的参数对象
    """
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args
```

---

#### 1.5 构建函数签名对象
**函数名**: `build_function_signature`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/signature.py`

**功能**: 组装所有信息为 FunctionSignature 对象

**Pseudo Code**:
```python
def build_function_signature(
    func_name: str,
    trace_id: str,
    bound_args: inspect.BoundArguments,
    signature: inspect.Signature,
    type_hints: Dict[str, Any],
    return_type: Any,
    docstring: str,
) -> FunctionSignature:
    """
    构建函数签名对象
    
    Returns:
        FunctionSignature 对象
    """
    return FunctionSignature(
        func_name=func_name,
        trace_id=trace_id,
        bound_args=bound_args,
        signature=signature,
        type_hints=type_hints,
        return_type=return_type,
        docstring=docstring,
    )
```

---

### Step 1 主函数
**函数名**: `parse_function_signature`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/signature.py`

**Pseudo Code**:
```python
def parse_function_signature(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[FunctionSignature, Optional[Dict[str, Any]]]:
    """
    解析函数签名的完整流程
    
    Returns:
        (FunctionSignature, template_params)
    """
    # 1. 提取模板参数
    template_params = extract_template_params(kwargs)
    
    # 2. 提取函数元数据
    signature, type_hints, return_type, docstring, func_name = extract_function_metadata(func)
    
    # 3. 生成追踪 ID
    trace_id = generate_trace_id(func_name)
    
    # 4. 绑定函数参数
    bound_args = bind_function_arguments(signature, args, kwargs)
    
    # 5. 构建函数签名对象
    function_signature = build_function_signature(
        func_name=func_name,
        trace_id=trace_id,
        bound_args=bound_args,
        signature=signature,
        type_hints=type_hints,
        return_type=return_type,
        docstring=docstring,
    )
    
    return function_signature, template_params
```

---

## Step 2: `setup_log_context`

### 职责
设置日志上下文，记录函数调用信息

### 具体执行步骤

#### 2.1 记录函数调用日志
**函数名**: `log_function_call`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/log_context.py`

**功能**: 记录函数调用时的参数信息

**Pseudo Code**:
```python
def log_function_call(
    func_name: str,
    arguments: Dict[str, Any],
) -> None:
    """
    记录函数调用日志
    
    Args:
        func_name: 函数名
        arguments: 函数参数值
    """
    args_str = json.dumps(
        arguments,
        default=str,
        ensure_ascii=False,
        indent=4,
    )
    app_log(
        f"Async LLM function '{func_name}' called with arguments: {args_str}",
        location=get_location(),
    )
```

---

#### 2.2 创建日志上下文管理器
**函数名**: `create_log_context_manager`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/log_context.py`

**功能**: 创建并返回 async_log_context 管理器

**Pseudo Code**:
```python
def create_log_context_manager(
    func_name: str,
    trace_id: str,
) -> AsyncContextManager[None]:
    """
    创建日志上下文管理器
    
    Returns:
        async_log_context 管理器
    """
    return async_log_context(
        trace_id=trace_id,
        function_name=func_name,
        input_tokens=0,
        output_tokens=0,
    )
```

---

### Step 2 主函数
**函数名**: `setup_log_context`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/log_context.py`

**Pseudo Code**:
```python
def setup_log_context(
    func_name: str,
    trace_id: str,
    arguments: Dict[str, Any],
) -> AsyncContextManager[None]:
    """
    设置日志上下文的完整流程
    
    Returns:
        日志上下文管理器
    """
    # 1. 记录函数调用日志
    log_function_call(func_name, arguments)
    
    # 2. 创建并返回日志上下文管理器
    return create_log_context_manager(func_name, trace_id)
```

---

## Step 3: `build_initial_prompts` (llm_function)

### 职责
根据函数元数据和实际参数构建初始的 SystemPrompt 和 UserPrompt

### 具体执行步骤

#### 3.1 处理 Docstring 模板参数
**函数名**: `process_docstring_template`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/prompt.py`

**功能**: 替换 docstring 中的模板参数

**Pseudo Code**:
```python
def process_docstring_template(
    docstring: str,
    template_params: Optional[Dict[str, Any]],
) -> str:
    """
    处理 docstring 模板参数替换
    
    Returns:
        处理后的 docstring
    """
    if not template_params:
        return docstring
    
    try:
        return docstring.format(**template_params)
    except KeyError as e:
        push_warning(
            f"DocString template parameter substitution failed: missing parameter {e}. "
            "Using original DocString.",
            location=get_location(),
        )
        return docstring
    except Exception as e:
        push_warning(
            f"Error during DocString template parameter substitution: {str(e)}. "
            "Using original DocString.",
            location=get_location(),
        )
        return docstring
```

---

#### 3.2 提取参数类型提示
**函数名**: `extract_parameter_type_hints`
**位置**: `SimpleLLMFunc/llm_decorator/steps/common/prompt.py`

**功能**: 从 type_hints 中提取参数类型（排除 return）

**Pseudo Code**:
```python
def extract_parameter_type_hints(
    type_hints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    提取参数类型提示（排除返回类型）
    
    Returns:
        参数名 -> 参数类型的字典
    """
    return {k: v for k, v in type_hints.items() if k != "return"}
```

---

#### 3.3 构建参数类型描述
**函数名**: `build_parameter_type_descriptions`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/prompt.py`

**功能**: 为每个参数构建类型描述字符串

**Pseudo Code**:
```python
def build_parameter_type_descriptions(
    param_type_hints: Dict[str, Any],
) -> List[str]:
    """
    构建参数类型描述列表
    
    Returns:
        格式: ["  - param_name: type_description", ...]
    """
    descriptions = []
    for param_name, param_type in param_type_hints.items():
        type_str = (
            get_detailed_type_description(param_type)
            if param_type
            else "Unknown Type"
        )
        descriptions.append(f"  - {param_name}: {type_str}")
    return descriptions
```

---

#### 3.4 构建返回类型描述
**函数名**: `build_return_type_description`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/prompt.py`

**功能**: 构建返回类型的详细描述（支持简单类型和复杂类型）

**Pseudo Code**:
```python
def build_return_type_description(return_type: Any) -> str:
    """
    构建返回类型描述
    
    对于简单类型：使用文本描述
    对于复杂类型（BaseModel, List, Dict, Union）：使用 JSON 格式 + 示例
    """
    if return_type is None:
        return "未知类型"
    
    if return_type in (str, int, float, bool, type(None)):
        # 简单类型
        return get_detailed_type_description(return_type)
    
    # 复杂类型：检查是否为 BaseModel、List、Dict、Union
    is_complex = _is_complex_type(return_type)
    
    if is_complex:
        type_json_obj = build_type_description_json(return_type)
        example_obj = generate_example_object(return_type)
        return (
            "Type Description (JSON):\n"
            + json.dumps(type_json_obj, ensure_ascii=False, indent=2)
            + "\n\nExample JSON:\n"
            + json.dumps(example_obj, ensure_ascii=False, indent=2)
        )
    else:
        return get_detailed_type_description(return_type)
```

**辅助函数**: `_is_complex_type`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/prompt.py`

**Pseudo Code**:
```python
def _is_complex_type(return_type: Any) -> bool:
    """判断是否为复杂类型"""
    from typing import get_origin, Union as TypingUnion
    
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return True
    
    origin = getattr(return_type, "__origin__", None) or get_origin(return_type)
    return origin in (list, List, dict, Dict, TypingUnion)
```

---

#### 3.5 检查多模态内容
**函数名**: `check_multimodal_content`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/prompt.py` (llm_function) 或 `SimpleLLMFunc/llm_decorator/steps/chat/message.py` (llm_chat)

**功能**: 检查参数中是否包含多模态内容（调用 base 中的函数）

**Pseudo Code**:
```python
def check_multimodal_content(
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    exclude_params: Optional[List[str]] = None,
) -> bool:
    """
    检查参数中是否包含多模态内容
    
    直接调用 base/type_resolve.py 中的 has_multimodal_content
    
    Returns:
        True 如果包含多模态内容
    """
    from SimpleLLMFunc.base.type_resolve import has_multimodal_content
    return has_multimodal_content(arguments, type_hints, exclude_params)
```

---

#### 3.6 构建文本消息列表
**函数名**: `build_text_messages`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/prompt.py`

**功能**: 构建纯文本的 system 和 user 消息

**Pseudo Code**:
```python
def build_text_messages(
    processed_docstring: str,
    param_type_descriptions: List[str],
    return_type_description: str,
    arguments: Dict[str, Any],
    system_template: str,
    user_template: str,
) -> List[Dict[str, str]]:
    """
    构建文本消息列表
    
    Returns:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    # 构建 system prompt
    system_prompt = system_template.format(
        function_description=processed_docstring,
        parameters_description="\n".join(param_type_descriptions),
        return_type_description=return_type_description,
    )
    
    # 构建 user prompt
    user_param_values = [
        f"  - {param_name}: {param_value}"
        for param_name, param_value in arguments.items()
    ]
    user_prompt = user_template.format(
        parameters="\n".join(user_param_values),
    )
    
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
```

---

#### 3.7 构建多模态消息列表
**函数名**: `build_multimodal_messages`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/prompt.py`

**功能**: 构建包含多模态内容的消息列表（调用 base 中的函数）

**Pseudo Code**:
```python
def build_multimodal_messages(
    system_prompt: str,
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    构建多模态消息列表
    
    调用 base/messages.py 中的 build_multimodal_content
    
    Returns:
        [{"role": "system", "content": "..."}, {"role": "user", "content": [...]}]
    """
    from SimpleLLMFunc.base.messages import build_multimodal_content
    
    # 构建多模态用户消息内容
    user_content = build_multimodal_content(arguments, type_hints)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
```

---

### Step 3 主函数 (llm_function)
**函数名**: `build_initial_prompts`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/prompt.py`

**Pseudo Code**:
```python
def build_initial_prompts(
    signature: FunctionSignature,
    system_prompt_template: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    template_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    构建初始提示的完整流程
    """
    # 1. 处理 docstring 模板参数
    processed_docstring = process_docstring_template(
        signature.docstring,
        template_params,
    )
    
    # 2. 提取参数类型提示
    param_type_hints = extract_parameter_type_hints(signature.type_hints)
    
    # 3. 构建参数类型描述
    param_type_descriptions = build_parameter_type_descriptions(param_type_hints)
    
    # 4. 构建返回类型描述
    return_type_description = build_return_type_description(signature.return_type)
    
    # 5. 检查多模态内容
    has_multimodal = check_multimodal_content(
        signature.bound_args.arguments,
        signature.type_hints,
    )
    
    # 6. 选择模板
    system_template = system_prompt_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE
    user_template = user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
    
    # 7. 构建消息列表
    if has_multimodal:
        # 先构建文本 system prompt
        text_messages = build_text_messages(
            processed_docstring,
            param_type_descriptions,
            return_type_description,
            signature.bound_args.arguments,
            system_template,
            user_template,
        )
        system_prompt = text_messages[0]["content"]
        
        # 构建多模态消息
        messages = build_multimodal_messages(
            system_prompt,
            signature.bound_args.arguments,
            signature.type_hints,
        )
    else:
        # 构建文本消息
        messages = build_text_messages(
            processed_docstring,
            param_type_descriptions,
            return_type_description,
            signature.bound_args.arguments,
            system_template,
            user_template,
        )
    
    return messages
```

---

## Step 3: `build_chat_messages` (llm_chat)

### 职责
构建聊天消息列表（System + History + User）

### 具体执行步骤

#### 3.1 提取对话历史
**函数名**: `extract_conversation_history`
**位置**: `SimpleLLMFunc/llm_decorator/steps/chat/message.py`

**功能**: 从参数中提取并验证对话历史

**Pseudo Code**:
```python
def extract_conversation_history(
    arguments: Dict[str, Any],
    func_name: str,
    history_param_names: List[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    提取并验证对话历史
    
    Args:
        history_param_names: 历史参数名列表，默认 ["history", "chat_history"]
    
    Returns:
        对话历史列表，如果不存在或无效则返回 None
    """
    if history_param_names is None:
        history_param_names = ["history", "chat_history"]
    
    # 查找历史参数
    history_param_name = None
    for param_name in history_param_names:
        if param_name in arguments:
            history_param_name = param_name
            break
    
    if not history_param_name:
        push_warning(
            f"LLM Chat '{func_name}' missing history parameter "
            f"(parameter name should be one of {history_param_names}). "
            "History will not be passed.",
            location=get_location(),
        )
        return None
    
    custom_history = arguments[history_param_name]
    
    # 验证历史格式
    if not (
        isinstance(custom_history, list)
        and all(isinstance(item, dict) for item in custom_history)
    ):
        push_warning(
            f"LLM Chat '{func_name}' history parameter should be List[Dict[str, str]] type. "
            "History will not be passed.",
            location=get_location(),
        )
        return None
    
    return custom_history
```

---

#### 3.2 构建用户消息内容
**函数名**: `build_chat_user_message_content`
**位置**: `SimpleLLMFunc/llm_decorator/steps/chat/message.py`

**功能**: 构建用户消息内容（文本或多模态）

**Pseudo Code**:
```python
def build_chat_user_message_content(
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    has_multimodal: bool,
    exclude_params: List[str],
) -> Union[str, List[Dict[str, Any]]]:
    """
    构建用户消息内容
    
    Returns:
        文本字符串或多模态内容列表
    """
    if has_multimodal:
        return build_multimodal_content(
            arguments,
            type_hints,
            exclude_params=exclude_params,
        )
    else:
        # 构建文本消息，排除历史参数
        message_parts = [
            f"{param_name}: {param_value}"
            for param_name, param_value in arguments.items()
            if param_name not in exclude_params
        ]
        return "\n\t".join(message_parts)
```

---

#### 3.3 构建系统提示（含工具描述）
**函数名**: `build_chat_system_prompt`
**位置**: `SimpleLLMFunc/llm_decorator/steps/chat/message.py`

**功能**: 构建系统提示，如果提供工具则添加工具描述

**Pseudo Code**:
```python
def build_chat_system_prompt(
    docstring: str,
    tool_objects: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    """
    构建聊天系统提示
    
    Returns:
        系统提示内容，如果 docstring 为空则返回 None
    """
    if not docstring:
        return None
    
    system_content = docstring
    
    # 如果提供工具，添加工具描述
    if tool_objects:
        tool_descriptions = "\n\t".join(
            f"- {tool['function']['name']}: {tool['function']['description']}"
            for tool in tool_objects
        )
        system_content = (
            "\n\nYou can use the following tools flexibly according to the real case and tool description:\n\t"
            + tool_descriptions
            + "\n\n"
            + system_content.strip()
        )
    
    return system_content
```

---

#### 3.4 过滤历史消息
**函数名**: `filter_history_messages`
**位置**: `SimpleLLMFunc/llm_decorator/steps/chat/message.py`

**功能**: 过滤历史消息，排除 system 消息

**Pseudo Code**:
```python
def filter_history_messages(
    history: List[Dict[str, Any]],
    func_name: str,
) -> List[Dict[str, Any]]:
    """
    过滤历史消息，排除 system 消息
    
    Returns:
        过滤后的历史消息列表
    """
    filtered = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            if msg["role"] not in ["system"]:
                filtered.append(msg)
        else:
            push_warning(
                f"Skipping malformed history item: {msg}",
                location=get_location(),
            )
    return filtered
```

---

### Step 3 主函数 (llm_chat)
**函数名**: `build_chat_messages`
**位置**: `SimpleLLMFunc/llm_decorator/steps/chat/message.py`

**Pseudo Code**:
```python
def build_chat_messages(
    signature: FunctionSignature,
    toolkit: Optional[List[Union[Tool, Callable]]],
    exclude_params: List[str],
) -> List[Dict[str, Any]]:
    """
    构建聊天消息列表的完整流程
    """
    messages = []
    
    # 1. 准备工具
    tool_param, tool_map = process_tools(toolkit, signature.func_name)
    
    # 2. 构建系统提示
    system_content = build_chat_system_prompt(
        signature.docstring,
        tool_param,
    )
    if system_content:
        messages.append({"role": "system", "content": system_content})
    
    # 3. 提取对话历史
    custom_history = extract_conversation_history(
        signature.bound_args.arguments,
        signature.func_name,
    )
    
    # 4. 过滤并添加历史消息
    if custom_history:
        filtered_history = filter_history_messages(custom_history, signature.func_name)
        messages.extend(filtered_history)
    
    # 5. 检查多模态内容
    has_multimodal = check_multimodal_content(
        signature.bound_args.arguments,
        signature.type_hints,
        exclude_params=exclude_params,
    )
    
    # 6. 构建用户消息内容
    user_message_content = build_chat_user_message_content(
        signature.bound_args.arguments,
        signature.type_hints,
        has_multimodal,
        exclude_params,
    )
    
    # 7. 添加用户消息
    if user_message_content:
        messages.append({"role": "user", "content": user_message_content})
    
    return messages
```

---

## Step 4: `execute_react_loop` (llm_function)

### 职责
执行 ReAct 循环，返回最终响应（包含重试逻辑）

### 具体执行步骤

#### 4.1 准备工具
**函数名**: `prepare_tools_for_execution`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/react.py` (llm_function) 或 `SimpleLLMFunc/llm_decorator/steps/chat/react.py` (llm_chat)

**功能**: 准备工具参数和映射（调用 utils 中的 `process_tools`）

**Pseudo Code**:
```python
def prepare_tools_for_execution(
    toolkit: Optional[List[Union[Tool, Callable]]],
    func_name: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Callable]]:
    """
    准备工具供执行使用
    
    直接调用 llm_decorator/utils/tools.py 中的 process_tools
    
    Returns:
        (tool_param, tool_map)
    """
    from SimpleLLMFunc.llm_decorator.utils.tools import process_tools
    return process_tools(toolkit, func_name)
```

---

#### 4.2 执行 LLM 调用
**函数名**: `execute_llm_call`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/react.py` (llm_function) 或 `SimpleLLMFunc/llm_decorator/steps/chat/react.py` (llm_chat)

**功能**: 调用 base 中的 `execute_llm` 函数

**Pseudo Code**:
```python
async def execute_llm_call(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    tool_map: Dict[str, Callable],
    max_tool_calls: int,
    stream: bool = False,
    **llm_kwargs: Any,
) -> AsyncGenerator[Any, None]:
    """
    执行 LLM 调用
    
    直接调用 base/ReAct.py 中的 execute_llm
    
    Returns:
        LLM 响应流
    """
    from SimpleLLMFunc.base.ReAct import execute_llm
    
    return execute_llm(
        llm_interface=llm_interface,
        messages=messages,
        tools=tools,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=stream,
        **llm_kwargs,
    )
```

---

#### 4.3 获取最终响应
**函数名**: `get_final_response`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/react.py`

**功能**: 从响应流中获取最后一个响应

**Pseudo Code**:
```python
async def get_final_response(
    response_stream: AsyncGenerator[Any, None],
) -> Any:
    """
    从响应流中获取最后一个响应
    
    Returns:
        最后一个响应对象
    """
    return await get_last_item_of_async_generator(response_stream)
```

---

#### 4.4 检查响应内容是否为空
**函数名**: `check_response_content_empty`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/react.py`

**功能**: 检查响应内容是否为空

**Pseudo Code**:
```python
def check_response_content_empty(
    response: Any,
    func_name: str,
) -> bool:
    """
    检查响应内容是否为空
    
    Returns:
        True 如果内容为空
    """
    content = ""
    if hasattr(response, "choices") and len(response.choices) > 0:
        message = response.choices[0].message
        content = message.content if message and hasattr(message, "content") else ""
    
    return content == ""
```

---

#### 4.5 重试 LLM 调用
**函数名**: `retry_llm_call`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/react.py`

**功能**: 重试 LLM 调用直到获得非空响应或重试次数耗尽

**Pseudo Code**:
```python
async def retry_llm_call(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    tool_map: Dict[str, Callable],
    max_tool_calls: int,
    retry_times: int,
    func_name: str,
    **llm_kwargs: Any,
) -> Any:
    """
    重试 LLM 调用
    
    Returns:
        最终响应对象
    
    Raises:
        ValueError: 如果重试后仍然为空
    """
    final_response = None
    
    for attempt in range(retry_times + 1):
        if attempt > 0:
            push_debug(
                f"Async LLM function '{func_name}' retry attempt {attempt}...",
                location=get_location(),
            )
        
        # 执行 LLM 调用
        response_stream = await execute_llm_call(
            llm_interface=llm_interface,
            messages=messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=max_tool_calls,
            stream=False,
            **llm_kwargs,
        )
        
        # 获取最终响应
        final_response = await get_final_response(response_stream)
        
        # 检查内容是否为空
        content = extract_content_from_response(final_response, func_name)
        if content != "":
            break
    
    # 最终检查
    if final_response:
        content = extract_content_from_response(final_response, func_name)
        if content == "":
            push_error(
                f"Async LLM function '{func_name}' response content still empty, "
                "retry attempts exhausted.",
                location=get_location(),
            )
            raise ValueError("LLM response content is empty after retries.")
    
    return final_response
```

---

### Step 4 主函数 (llm_function)
**函数名**: `execute_react_loop`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/react.py`

**Pseudo Code**:
```python
async def execute_react_loop(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, Any]],
    toolkit: Optional[List[Union[Tool, Callable]]],
    max_tool_calls: int,
    llm_kwargs: Dict[str, Any],
    func_name: str,
) -> Any:
    """
    执行 ReAct 循环的完整流程（包含重试）
    """
    # 1. 准备工具
    tool_param, tool_map = prepare_tools_for_execution(toolkit, func_name)
    
    # 2. 执行 LLM 调用
    response_stream = execute_llm_call(
        llm_interface=llm_interface,
        messages=messages,
        tools=tool_param,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=False,
        **llm_kwargs,
    )
    
    # 3. 获取最终响应
    final_response = await get_final_response(response_stream)
    
    # 4. 检查响应内容是否为空
    if check_response_content_empty(final_response, func_name):
        push_warning(
            f"Async LLM function '{func_name}' returned empty response content, "
            "will retry automatically.",
            location=get_location(),
        )
        
        # 5. 重试 LLM 调用
        retry_times = llm_kwargs.get("retry_times", 2)
        final_response = await retry_llm_call(
            llm_interface=llm_interface,
            messages=messages,
            tools=tool_param,
            tool_map=tool_map,
            max_tool_calls=max_tool_calls,
            retry_times=retry_times,
            func_name=func_name,
            **llm_kwargs,
        )
    
    # 6. 记录最终响应
    push_debug(
        f"Async LLM function '{func_name}' received response "
        f"{json.dumps(final_response, default=str, ensure_ascii=False, indent=2)}",
        location=get_location(),
    )
    
    return final_response
```

---

## Step 4: `execute_react_loop_streaming` (llm_chat)

### 职责
执行 ReAct 循环，返回响应流（无重试逻辑）

### Step 4 主函数 (llm_chat)
**函数名**: `execute_react_loop_streaming`
**位置**: `SimpleLLMFunc/llm_decorator/steps/chat/react.py`

**Pseudo Code**:
```python
async def execute_react_loop_streaming(
    llm_interface: LLM_Interface,
    messages: List[Dict[str, Any]],
    toolkit: Optional[List[Union[Tool, Callable]]],
    max_tool_calls: int,
    stream: bool,
    llm_kwargs: Dict[str, Any],
    func_name: str,
) -> AsyncGenerator[Any, None]:
    """
    执行 ReAct 循环的流式版本（无重试）
    """
    # 1. 准备工具
    tool_param, tool_map = prepare_tools_for_execution(toolkit, func_name)
    
    # 2. 执行 LLM 调用（流式）
    response_stream = execute_llm_call(
        llm_interface=llm_interface,
        messages=messages,
        tools=tool_param,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=stream,
        **llm_kwargs,
    )
    
    # 3. 直接返回响应流（不进行重试）
    async for response in response_stream:
        yield response
```

---

## Step 5: `parse_and_validate_response` (llm_function)

### 职责
解析和验证 LLM 响应，转换为目标返回类型

### 具体执行步骤

#### 5.1 提取响应内容
**函数名**: `extract_response_content`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/response.py` (llm_function) 或 `SimpleLLMFunc/llm_decorator/steps/chat/response.py` (llm_chat)

**功能**: 从响应对象中提取文本内容（调用 base 中的函数）

**Pseudo Code**:
```python
def extract_response_content(
    response: Any,
    func_name: str,
) -> str:
    """
    从响应对象中提取文本内容
    
    直接调用 base/post_process.py 中的 extract_content_from_response
    
    Returns:
        响应文本内容
    """
    from SimpleLLMFunc.base.post_process import extract_content_from_response
    return extract_content_from_response(response, func_name)
```

---

#### 5.2 解析响应为返回类型
**函数名**: `parse_response_to_type`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/response.py`

**功能**: 将响应内容解析为目标返回类型（调用 base 中的 `process_response`）

**Pseudo Code**:
```python
def parse_response_to_type(
    response: Any,
    return_type: Any,
) -> Any:
    """
    将响应解析为目标返回类型
    
    直接调用 base/post_process.py 中的 process_response
    
    Returns:
        解析后的结果
    
    Raises:
        ValueError: 如果无法解析为指定类型
    """
    from SimpleLLMFunc.base.post_process import process_response
    return process_response(response, return_type)
```

---

### Step 5 主函数 (llm_function)
**函数名**: `parse_and_validate_response`
**位置**: `SimpleLLMFunc/llm_decorator/steps/function/response.py`

**Pseudo Code**:
```python
def parse_and_validate_response(
    response: Any,
    return_type: Any,
    func_name: str,
) -> Any:
    """
    解析和验证响应的完整流程
    """
    # 1. 提取响应内容（内部会调用 extract_content_from_response）
    # 2. 解析为返回类型（内部会调用 process_response）
    return parse_response_to_type(response, return_type)
```

---

## Step 5: `process_chat_response_stream` (llm_chat)

### 职责
处理流式响应，返回 (content, history) 元组流

### 具体执行步骤

#### 5.1 提取流式响应内容
**函数名**: `extract_stream_response_content`
**位置**: `SimpleLLMFunc/llm_decorator/steps/chat/response.py`

**功能**: 从流式响应 chunk 中提取内容（调用 base 中的函数）

**Pseudo Code**:
```python
def extract_stream_response_content(
    chunk: Any,
    func_name: str,
) -> str:
    """
    从流式响应 chunk 中提取内容
    
    直接调用 base/post_process.py 中的 extract_content_from_stream_response
    
    Returns:
        文本内容
    """
    from SimpleLLMFunc.base.post_process import extract_content_from_stream_response
    return extract_content_from_stream_response(chunk, func_name)
```

---

#### 5.2 处理单个响应
**函数名**: `process_single_chat_response`
**位置**: `SimpleLLMFunc/llm_decorator/steps/chat/response.py`

**功能**: 处理单个响应，根据 return_mode 返回相应格式

**Pseudo Code**:
```python
def process_single_chat_response(
    response: Any,
    return_mode: Literal["text", "raw"],
    stream: bool,
    func_name: str,
) -> Any:
    """
    处理单个响应
    
    Returns:
        - "raw" 模式: 原始响应对象
        - "text" 模式: 文本内容字符串
    """
    if return_mode == "raw":
        return response
    
    # text 模式：提取内容
    if stream:
        return extract_stream_response_content(response, func_name)
    else:
        return extract_response_content(response, func_name) or ""
```

---

### Step 5 主函数 (llm_chat)
**函数名**: `process_chat_response_stream`
**位置**: `SimpleLLMFunc/llm_decorator/steps/chat/response.py`

**Pseudo Code**:
```python
async def process_chat_response_stream(
    response_stream: AsyncGenerator[Any, None],
    return_mode: Literal["text", "raw"],
    messages: List[Dict[str, Any]],
    func_name: str,
    stream: bool,
) -> AsyncGenerator[Tuple[Any, List[Dict[str, Any]]], None]:
    """
    处理流式响应的完整流程
    """
    complete_content = ""
    
    async for response in response_stream:
        # 记录响应日志
        app_log(
            f"LLM Chat '{func_name}' received response:"
            f"\n{json.dumps(response, default=str, ensure_ascii=False, indent=2)}",
            location=get_location(),
        )
        
        # 处理单个响应
        content = process_single_chat_response(
            response,
            return_mode,
            stream,
            func_name,
        )
        
        # 累计内容（text 模式）
        if return_mode == "text":
            complete_content += content
        
        # Yield 响应和当前历史
        yield content, messages
    
    # 流结束标记（text 模式）
    if return_mode == "text":
        yield "", messages
```

---

## 文件规划

### 职责划分原则

#### base 文件夹的职责
- **核心的、可复用的底层实现**
- **不依赖装饰器特定的逻辑**
- **可以被多个模块使用**

保留在 `base` 的功能：
- `base/ReAct.py` - ReAct 循环的核心实现 (`execute_llm`)
- `base/tool_call.py` - 工具调用的核心实现 (`process_tool_calls`, `extract_tool_calls` 等)
- `base/messages.py` - 消息构建的核心实现 (`build_multimodal_content`, `build_assistant_response_message` 等)
- `base/post_process.py` - 响应后处理的核心实现 (`process_response`, `extract_content_from_response` 等)
- `base/type_resolve.py` - 类型解析的核心实现 (`get_detailed_type_description`, `build_type_description_json` 等)

#### llm_decorator/steps 的职责
- **装饰器特定的编排逻辑**
- **调用 base 中的函数**
- **处理装饰器特定的需求**（重试、日志、trace、函数签名解析等）

### 目录结构

```
SimpleLLMFunc/
├── base/                          # 核心底层实现（保持不变）
│   ├── __init__.py
│   ├── ReAct.py                   # ReAct 循环核心实现
│   │   └── execute_llm            # ✅ 保留在 base
│   ├── tool_call.py               # 工具调用核心实现
│   │   ├── process_tool_calls     # ✅ 保留在 base
│   │   ├── extract_tool_calls     # ✅ 保留在 base
│   │   └── ...                    # ✅ 保留在 base
│   ├── messages.py                # 消息构建核心实现
│   │   ├── build_multimodal_content  # ✅ 保留在 base
│   │   ├── build_assistant_response_message  # ✅ 保留在 base
│   │   └── ...                    # ✅ 保留在 base
│   ├── post_process.py            # 响应后处理核心实现
│   │   ├── process_response       # ✅ 保留在 base
│   │   ├── extract_content_from_response  # ✅ 保留在 base
│   │   ├── extract_content_from_stream_response  # ✅ 保留在 base
│   │   └── ...                    # ✅ 保留在 base
│   └── type_resolve.py            # 类型解析核心实现
│       ├── get_detailed_type_description  # ✅ 保留在 base
│       ├── build_type_description_json    # ✅ 保留在 base
│       ├── has_multimodal_content         # ✅ 保留在 base
│       └── ...                    # ✅ 保留在 base
│
├── llm_decorator/
│   ├── __init__.py
│   ├── llm_function_decorator.py      # llm_function 装饰器（重构后）
│   ├── llm_chat_decorator.py          # llm_chat 装饰器（重构后）
│   ├── steps/                         # 步骤函数模块（装饰器特定）
│   │   ├── __init__.py
│   │   ├── common/                    # 共享步骤（装饰器特定）
│   │   │   ├── __init__.py
│   │   │   ├── signature.py           # Step 1: 函数签名解析（装饰器特定）
│   │   │   │   ├── extract_template_params
│   │   │   │   ├── extract_function_metadata
│   │   │   │   ├── generate_trace_id
│   │   │   │   ├── bind_function_arguments
│   │   │   │   ├── build_function_signature
│   │   │   │   └── parse_function_signature (主函数)
│   │   │   ├── log_context.py         # Step 2: 日志上下文（装饰器特定）
│   │   │   │   ├── log_function_call
│   │   │   │   ├── create_log_context_manager
│   │   │   │   └── setup_log_context (主函数)
│   │   │   ├── prompt.py              # 共享的 prompt 处理（装饰器特定）
│   │   │   │   ├── process_docstring_template
│   │   │   │   └── extract_parameter_type_hints
│   │   │   └── types.py               # 数据结构定义
│   │   │       └── FunctionSignature
│   │   ├── function/                  # llm_function 专用步骤
│   │   │   ├── __init__.py
│   │   │   ├── prompt.py              # Step 3: 构建提示（装饰器特定）
│   │   │   │   ├── build_parameter_type_descriptions
│   │   │   │   ├── build_return_type_description
│   │   │   │   ├── _is_complex_type
│   │   │   │   ├── build_text_messages
│   │   │   │   └── build_initial_prompts (主函数)
│   │   │   │   # 调用 base/type_resolve.py 中的函数
│   │   │   │   # 调用 base/messages.py 中的函数
│   │   │   ├── react.py               # Step 4: ReAct 循环（装饰器特定）
│   │   │   │   ├── get_final_response
│   │   │   │   ├── check_response_content_empty
│   │   │   │   ├── retry_llm_call
│   │   │   │   └── execute_react_loop (主函数)
│   │   │   │   # 调用 base/ReAct.py 中的 execute_llm
│   │   │   │   # 调用 base/post_process.py 中的函数
│   │   │   └── response.py            # Step 5: 响应解析（装饰器特定）
│   │   │       ├── parse_response_to_type
│   │   │       └── parse_and_validate_response (主函数)
│   │   │       # 调用 base/post_process.py 中的 process_response
│   │   └── chat/                      # llm_chat 专用步骤
│   │       ├── __init__.py
│   │       ├── message.py             # Step 3: 构建消息（装饰器特定）
│   │       │   ├── extract_conversation_history
│   │       │   ├── build_chat_user_message_content
│   │       │   ├── build_chat_system_prompt
│   │       │   ├── filter_history_messages
│   │       │   └── build_chat_messages (主函数)
│   │       │   # 调用 base/messages.py 中的 build_multimodal_content
│   │       │   # 调用 base/type_resolve.py 中的 has_multimodal_content
│   │       ├── react.py               # Step 4: ReAct 循环（流式，装饰器特定）
│   │       │   └── execute_react_loop_streaming (主函数)
│   │       │   # 调用 base/ReAct.py 中的 execute_llm
│   │       └── response.py            # Step 5: 流式响应处理（装饰器特定）
│   │           ├── process_single_chat_response
│   │           └── process_chat_response_stream (主函数)
│   │           # 调用 base/post_process.py 中的函数
│   └── utils/
│       └── tools.py                    # 工具处理（保持不变）
│           └── process_tools           # ✅ 保留在 utils
```

### 数据结构定义

**位置**: `SimpleLLMFunc/llm_decorator/steps/common/types.py`

```python
from typing import NamedTuple, Any, Dict
import inspect

class FunctionSignature(NamedTuple):
    """函数签名信息"""
    func_name: str
    trace_id: str
    bound_args: inspect.BoundArguments
    signature: inspect.Signature
    type_hints: Dict[str, Any]
    return_type: Any
    docstring: str
```

---

## 总结

### 公共函数提取

1. **Step 1**: 完全共享，所有函数都在 `common/signature.py`
2. **Step 2**: 完全共享，所有函数都在 `common/log_context.py`
3. **Step 3**: 
   - 共享: `process_docstring_template`, `extract_parameter_type_hints`, `check_multimodal_content`
   - llm_function 专用: `build_parameter_type_descriptions`, `build_return_type_description`, `build_text_messages`
   - llm_chat 专用: `extract_conversation_history`, `build_chat_user_message_content`, `build_chat_system_prompt`
4. **Step 4**:
   - 共享: `prepare_tools_for_execution`, `execute_llm_call`
   - llm_function 专用: `get_final_response`, `check_response_content_empty`, `retry_llm_call`
   - llm_chat 专用: 流式处理逻辑
5. **Step 5**:
   - 共享: `extract_response_content`, `extract_stream_response_content`
   - llm_function 专用: `parse_response_to_type`
   - llm_chat 专用: `process_single_chat_response`

### 复用现有代码（保留在 base 和 utils）

#### base 文件夹（核心底层实现）
- **`base/ReAct.py`**:
  - `execute_llm` - ReAct 循环核心实现 ✅
  
- **`base/tool_call.py`**:
  - `process_tool_calls` - 工具调用执行 ✅
  - `extract_tool_calls` - 工具调用提取 ✅
  - `extract_tool_calls_from_stream_response` - 流式工具调用提取 ✅
  - `accumulate_tool_calls_from_chunks` - 工具调用累积 ✅

- **`base/messages.py`**:
  - `build_multimodal_content` - 多模态内容构建 ✅
  - `build_assistant_response_message` - 助手响应消息构建 ✅
  - `build_assistant_tool_message` - 工具调用消息构建 ✅
  - `extract_usage_from_response` - 用量信息提取 ✅

- **`base/post_process.py`**:
  - `process_response` - 响应解析和类型转换 ✅
  - `extract_content_from_response` - 响应内容提取 ✅
  - `extract_content_from_stream_response` - 流式响应内容提取 ✅

- **`base/type_resolve.py`**:
  - `has_multimodal_content` - 多模态内容检查 ✅
  - `get_detailed_type_description` - 类型描述生成 ✅
  - `build_type_description_json` - 类型 JSON 描述构建 ✅
  - `generate_example_object` - 示例对象生成 ✅
  - `is_multimodal_type` - 多模态类型判断 ✅

#### utils 文件夹
- **`utils.py`**:
  - `get_last_item_of_async_generator` - 获取异步生成器最后一项 ✅

- **`llm_decorator/utils/tools.py`**:
  - `process_tools` - 工具列表处理 ✅

### 调用关系

```
llm_decorator/steps/
    ├── 调用 base/ReAct.py::execute_llm
    ├── 调用 base/post_process.py::process_response
    ├── 调用 base/post_process.py::extract_content_from_response
    ├── 调用 base/messages.py::build_multimodal_content
    ├── 调用 base/type_resolve.py::has_multimodal_content
    ├── 调用 base/type_resolve.py::get_detailed_type_description
    ├── 调用 base/type_resolve.py::build_type_description_json
    ├── 调用 llm_decorator/utils/tools.py::process_tools
    └── 调用 utils.py::get_last_item_of_async_generator
```

