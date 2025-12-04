# LLM Function Decorator 重构计划

## 重构目标

1. **理清代码逻辑**：将装饰器内部逻辑拆分为清晰的5个步骤
2. **功能分层**：每个步骤职责单一，便于维护和测试
3. **最小化暴露**：装饰器内部直接暴露的接口非常少，主要通过5个步骤函数调用完成

## 当前代码结构分析

### 现有代码流程（`llm_function_decorator.py`）

当前装饰器的执行流程：

```178:267:SimpleLLMFunc/llm_decorator/llm_function_decorator.py
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Prepare function call context and extract template parameters
            context, call_time_template_params = _prepare_function_call(
                func, args, kwargs
            )

            async with async_log_context(
                trace_id=context.trace_id,
                function_name=context.func_name,
                input_tokens=0,
                output_tokens=0,
            ):
                # Log function invocation with arguments
                args_str = json.dumps(
                    context.bound_args.arguments,
                    default=str,
                    ensure_ascii=False,
                    indent=4,
                )

                app_log(
                    f"Async LLM function '{context.func_name}' called with arguments: {args_str}",
                    location=get_location(),
                )

                # Build message list (system prompt + user prompt)
                messages = _build_messages(
                    context=context,
                    system_prompt_template=system_prompt_template,
                    user_prompt_template=user_prompt_template,
                    template_params=call_time_template_params,
                )

                # Prepare tools for LLM
                tool_param, tool_map = _prepare_tools_for_llm(
                    toolkit, context.func_name
                )

                # Package LLM call parameters
                llm_params = LLMCallParams(
                    messages=messages,
                    tool_param=tool_param,
                    tool_map=tool_map,
                    llm_kwargs=llm_kwargs,
                )

                # 创建 Langfuse parent span 用于追踪整个函数调用
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"{context.func_name}_function_call",
                    input=context.bound_args.arguments,
                    metadata={
                        "function_name": context.func_name,
                        "trace_id": context.trace_id,
                        "tools_available": len(toolkit) if toolkit else 0,
                        "max_tool_calls": max_tool_calls,
                    },
                ) as function_span:
                    try:
                        # Execute LLM call with retry logic
                        final_response = await _execute_llm_with_retry_async(
                            llm_interface=llm_interface,
                            context=context,
                            llm_params=llm_params,
                            max_tool_calls=max_tool_calls,
                        )
                        # Convert response to specified return type
                        result = _process_final_response(
                            final_response, context.return_type
                        )

                        # 更新 span 输出信息
                        function_span.update(
                            output={
                                "result": result,
                                "return_type": str(context.return_type),
                            },
                        )

                        return result
                    except Exception as exc:
                        # 更新 span 错误信息
                        function_span.update(
                            output={"error": str(exc)},
                        )
                        push_error(
                            f"Async LLM function '{context.func_name}' execution failed: {str(exc)}",
                            location=get_location(),
                        )
                        raise
```

### 现有辅助函数

- `_prepare_function_call`: 解析函数签名，创建上下文
- `_build_messages`: 构建消息列表（system + user prompt）
- `_prepare_tools_for_llm`: 准备工具参数和映射
- `_execute_llm_with_retry_async`: 执行 LLM 调用（包含重试逻辑）
- `_process_final_response`: 处理最终响应，转换为返回类型

## 重构后的顶层设计

### 核心原则

装饰器内部只调用5个步骤函数，每个步骤函数职责单一，内部实现细节（helper functions）由步骤函数内部管理。

### 五个步骤函数设计

#### Step 1: `parse_function_signature`

**职责**：解析函数签名，提取所有元数据

**输入**：
- `func`: 被装饰的函数
- `args`: 位置参数
- `kwargs`: 关键字参数

**输出**：
- `FunctionSignature`: 包含以下信息的数据结构
  - `func_name: str` - 函数名
  - `param_types: Dict[str, Any]` - 参数类型表（参数名 -> 类型）
  - `return_type: Any` - 返回类型
  - `docstring: str` - Docstring
  - `bound_args: inspect.BoundArguments` - 绑定后的参数（包含默认值）
  - `template_params: Optional[Dict[str, Any]]` - 模板参数（如果有）

**说明**：
- 提取函数签名、类型提示、docstring
- 绑定参数并应用默认值
- 提取模板参数（`_template_params`）
- 生成 trace_id

**当前对应代码**：
- `_prepare_function_call` 的主要逻辑
- `FunctionCallContext` 的部分字段

---

#### Step 2: `setup_log_context`

**职责**：设置日志上下文

**输入**：
- `func_name: str` - 函数名
- `trace_id: str` - 追踪 ID
- `arguments: Dict[str, Any]` - 函数参数值

**输出**：
- `AsyncContextManager` - 日志上下文管理器

**说明**：
- 创建 `async_log_context`
- 记录函数调用日志（参数信息）
- 返回上下文管理器供外部使用

**当前对应代码**：
- `async_log_context` 的创建和使用
- 函数调用日志记录

---

#### Step 3: `build_initial_prompts`

**职责**：根据函数元数据和实际参数构建初始的 SystemPrompt 和 UserPrompt

**输入**：
- `signature: FunctionSignature` - 函数签名信息
- `system_prompt_template: Optional[str]` - 自定义系统提示模板
- `user_prompt_template: Optional[str]` - 自定义用户提示模板

**输出**：
- `List[Dict[str, Any]]` - 消息列表（包含 system 和 user 消息）
  - 支持多模态内容（如果参数中包含图片等）

**说明**：
- 根据参数类型表构建参数描述
- 根据返回类型表构建返回类型描述
- 处理 docstring 模板参数替换
- 构建 system prompt 和 user prompt
- 处理多模态内容（如果存在）

**当前对应代码**：
- `_build_messages` 的主要逻辑
- `_build_prompts` 的逻辑
- `_build_multimodal_messages` 的逻辑
- `_has_multimodal_content` 的检查

---

#### Step 4: `execute_react_loop`

**职责**：执行 ReAct 循环，与 LLM 交互并处理工具调用

**输入**：
- `llm_interface: LLM_Interface` - LLM 接口
- `messages: List[Dict[str, Any]]` - 初始消息列表
- `toolkit: Optional[List[Union[Tool, Callable]]]` - 工具列表
- `max_tool_calls: int` - 最大工具调用次数
- `llm_kwargs: Dict[str, Any]` - LLM 额外参数
- `func_name: str` - 函数名（用于日志和追踪）

**输出**：
- `Any` - 最终的 LLM 响应对象

**说明**：
- 调用 `execute_llm`（ReAct 循环）
- 处理重试逻辑（如果响应为空）
- 返回最终响应

**当前对应代码**：
- `_execute_llm_with_retry_async` 的主要逻辑
- `execute_llm` 的调用（来自 `base.ReAct`）
- `_prepare_tools_for_llm` 的逻辑（工具准备）

---

#### Step 5: `parse_and_validate_response`

**职责**：解析和验证 LLM 响应，转换为目标返回类型

**输入**：
- `response: Any` - LLM 响应对象
- `return_type: Any` - 目标返回类型
- `func_name: str` - 函数名（用于错误信息）

**输出**：
- `T` - 解析后的结果（类型由 return_type 决定）

**异常**：
- `ValueError`: 当响应无法解析为指定类型时抛出

**说明**：
- 从响应中提取内容
- 根据返回类型进行类型转换
- 支持基本类型、dict、Pydantic 模型等
- 抛出清晰的错误信息

**当前对应代码**：
- `_process_final_response` 的逻辑
- `process_response` 的调用（来自 `base.post_process`）

---

## Langfuse Trace 集成

### 当前实现

Langfuse trace 在装饰器中通过以下方式集成：

```225:267:SimpleLLMFunc/llm_decorator/llm_function_decorator.py
                # 创建 Langfuse parent span 用于追踪整个函数调用
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"{context.func_name}_function_call",
                    input=context.bound_args.arguments,
                    metadata={
                        "function_name": context.func_name,
                        "trace_id": context.trace_id,
                        "tools_available": len(toolkit) if toolkit else 0,
                        "max_tool_calls": max_tool_calls,
                    },
                ) as function_span:
                    try:
                        # Execute LLM call with retry logic
                        final_response = await _execute_llm_with_retry_async(
                            llm_interface=llm_interface,
                            context=context,
                            llm_params=llm_params,
                            max_tool_calls=max_tool_calls,
                        )
                        # Convert response to specified return type
                        result = _process_final_response(
                            final_response, context.return_type
                        )

                        # 更新 span 输出信息
                        function_span.update(
                            output={
                                "result": result,
                                "return_type": str(context.return_type),
                            },
                        )

                        return result
                    except Exception as exc:
                        # 更新 span 错误信息
                        function_span.update(
                            output={"error": str(exc)},
                        )
                        push_error(
                            f"Async LLM function '{context.func_name}' execution failed: {str(exc)}",
                            location=get_location(),
                        )
                        raise
```

### 重构后的集成方式

**方案**：Langfuse trace 应该在装饰器的最外层（Step 2 和 Step 4 之间）创建，包裹整个执行流程。

**位置**：在 `setup_log_context` 返回的上下文管理器内部，但在调用 `execute_react_loop` 之前创建 Langfuse span。

**数据流**：
1. Step 1 提供 `func_name`、`trace_id`、`arguments`（用于 span 的 input）
2. Step 4 执行后，使用最终响应更新 span 的 output
3. Step 5 执行后，使用解析结果更新 span 的 output（最终结果）

**注意**：
- ReAct 循环内部（`execute_llm`）已经创建了 generation 类型的观测，这些会自动成为 span 的子观测
- 装饰器层的 span 是 parent span，用于追踪整个函数调用

---

## Log Context 集成

### 当前实现

Log context 通过 `async_log_context` 管理：

```184:189:SimpleLLMFunc/llm_decorator/llm_function_decorator.py
            async with async_log_context(
                trace_id=context.trace_id,
                function_name=context.func_name,
                input_tokens=0,
                output_tokens=0,
            ):
```

### 重构后的集成方式

**方案**：Log context 由 Step 2 (`setup_log_context`) 负责创建和管理。

**职责**：
- 创建 `async_log_context`
- 记录函数调用日志（参数信息）
- 返回上下文管理器供装饰器使用

**数据流**：
- Step 1 提供 `trace_id` 和 `func_name`
- Step 2 创建上下文并记录初始日志
- 后续步骤在上下文中执行，自动继承 trace_id 等信息

---

## 重构后的装饰器结构（伪代码）

```python
@wraps(func)
async def async_wrapper(*args: Any, **kwargs: Any) -> T:
    # Step 1: 解析函数签名
    signature = parse_function_signature(func, args, kwargs)
    
    # Step 2: 设置日志上下文
    async with setup_log_context(
        func_name=signature.func_name,
        trace_id=signature.trace_id,
        arguments=signature.bound_args.arguments,
    ):
        # 创建 Langfuse parent span
        with langfuse_client.start_as_current_observation(...) as function_span:
            try:
                # Step 3: 构建初始提示
                messages = build_initial_prompts(
                    signature=signature,
                    system_prompt_template=system_prompt_template,
                    user_prompt_template=user_prompt_template,
                )
                
                # Step 4: 执行 ReAct 循环
                final_response = await execute_react_loop(
                    llm_interface=llm_interface,
                    messages=messages,
                    toolkit=toolkit,
                    max_tool_calls=max_tool_calls,
                    llm_kwargs=llm_kwargs,
                    func_name=signature.func_name,
                )
                
                # Step 5: 解析和验证响应
                result = parse_and_validate_response(
                    response=final_response,
                    return_type=signature.return_type,
                    func_name=signature.func_name,
                )
                
                # 更新 Langfuse span
                function_span.update(output={"result": result, ...})
                return result
                
            except Exception as exc:
                function_span.update(output={"error": str(exc)})
                raise
```

---

## 数据流图

```
装饰器入口 (async_wrapper)
    │
    ├─> Step 1: parse_function_signature
    │   └─> 返回: FunctionSignature
    │
    ├─> Step 2: setup_log_context
    │   └─> 返回: AsyncContextManager
    │       │
    │       └─> [进入上下文]
    │           │
    │           ├─> 创建 Langfuse parent span
    │           │
    │           ├─> Step 3: build_initial_prompts
    │           │   └─> 返回: List[Dict[str, Any]] (messages)
    │           │
    │           ├─> Step 4: execute_react_loop
    │           │   └─> 调用 execute_llm (ReAct 循环)
    │           │   └─> 返回: Any (final_response)
    │           │
    │           ├─> Step 5: parse_and_validate_response
    │           │   └─> 返回: T (result)
    │           │
    │           └─> 更新 Langfuse span
    │
    └─> [退出上下文]
```

---

## 关键设计决策

### 1. 数据结构设计

**FunctionSignature**：
- 替代当前的 `FunctionCallContext`
- 包含所有函数签名相关的信息
- 不包含执行时的动态信息（如 trace_id 可以单独传递）

### 2. 错误处理

- Step 5 负责类型转换错误
- Step 4 负责 LLM 调用错误（包括重试失败）
- 装饰器层统一捕获异常并更新 Langfuse span

### 3. 工具准备

- 工具准备逻辑可以放在 Step 4 内部，作为 `execute_react_loop` 的实现细节
- 或者作为 Step 3 和 Step 4 之间的一个独立步骤（如果工具信息需要用于 prompt 构建）

### 4. 多模态支持

- 多模态检查和处理逻辑放在 Step 3 (`build_initial_prompts`) 内部
- 作为实现细节，不暴露给装饰器层

### 5. 重试逻辑

- 重试逻辑放在 Step 4 (`execute_react_loop`) 内部
- 作为实现细节，不暴露给装饰器层

---

## 下一步工作

1. **设计 FunctionSignature 数据结构**：定义清晰的数据结构来替代 `FunctionCallContext`
2. **实现 Step 1**：`parse_function_signature` 函数
3. **实现 Step 2**：`setup_log_context` 函数
4. **实现 Step 3**：`build_initial_prompts` 函数
5. **实现 Step 4**：`execute_react_loop` 函数
6. **实现 Step 5**：`parse_and_validate_response` 函数
7. **重构装饰器**：使用5个步骤函数重构 `async_wrapper`
8. **测试**：确保功能完整性，特别是 Langfuse trace 和 log context

---

## 注意事项

1. **向后兼容**：确保重构后的装饰器行为与现有版本一致
2. **性能**：保持或提升性能，避免不必要的开销
3. **可测试性**：每个步骤函数应该易于单独测试
4. **可扩展性**：为未来功能扩展预留空间（如新的 prompt 模板、新的返回类型等）

---

# LLM Chat Decorator 重构计划

## llm_chat 与 llm_function 的对比分析

### 相同点

1. **都需要解析函数签名**：提取函数名、参数类型、docstring 等
2. **都需要设置日志上下文**：使用 `async_log_context` 管理 trace_id
3. **都需要构建消息列表**：准备发送给 LLM 的消息
4. **都需要执行 ReAct 循环**：调用 `execute_llm` 进行 LLM 交互和工具调用
5. **都需要 Langfuse trace 集成**：追踪整个调用过程

### 核心差异

| 特性 | llm_function | llm_chat |
|------|-------------|----------|
| **返回类型** | 单个值（根据返回类型解析） | `AsyncGenerator[Tuple[Any, HistoryList], None]` |
| **消息构建** | System prompt（基于 docstring + 参数类型 + 返回类型）+ User prompt（参数值） | System prompt（docstring）+ History + User message（参数值） |
| **历史管理** | 不需要 | 需要提取和管理 `history`/`chat_history` 参数 |
| **响应处理** | 等待最终响应，解析为返回类型 | 流式处理响应，每次 yield (content, history) |
| **多模态处理** | 在构建 prompt 时处理 | 在构建 user message 时处理，需排除 history 参数 |
| **重试逻辑** | 有（响应为空时重试） | 无（流式响应不需要） |
| **返回模式** | 固定（根据返回类型） | 支持 `text` 和 `raw` 两种模式 |

### 关键差异点详解

#### 1. 消息构建差异

**llm_function**:
- System prompt: 包含函数描述、参数类型描述、返回类型描述
- User prompt: 参数值列表
- 支持自定义模板（`system_prompt_template`, `user_prompt_template`）

**llm_chat**:
- System prompt: 仅 docstring（可能包含工具描述）
- History: 从参数中提取的对话历史
- User message: 参数值（排除 history 参数）
- 不支持自定义模板

#### 2. 响应处理差异

**llm_function**:
```python
final_response = await execute_react_loop(...)
result = parse_and_validate_response(final_response, return_type)
return result
```

**llm_chat**:
```python
async for response in execute_react_loop(...):
    if return_mode == "raw":
        yield response, current_messages
    else:
        content = extract_content(response)
        yield content, current_messages
yield "", current_messages  # 流结束标记
```

#### 3. 历史管理

**llm_chat 特有**:
- 需要从参数中提取 `history` 或 `chat_history`
- 验证历史格式
- 将历史插入到消息列表中
- 每次响应都返回更新后的历史

---

## llm_chat 装饰器的顶层流程设计

### 设计原则

1. **复用共同步骤**：Step 1 和 Step 2 可以完全复用
2. **差异化处理**：Step 3、Step 4、Step 5 需要针对 chat 场景调整
3. **流式处理**：Step 4 和 Step 5 需要支持流式返回

### 五个步骤函数设计（llm_chat 版本）

#### Step 1: `parse_function_signature` ✅ **可复用**

**职责**：解析函数签名，提取所有元数据

**输入**：
- `func`: 被装饰的函数
- `args`: 位置参数
- `kwargs`: 关键字参数

**输出**：
- `FunctionSignature`: 与 llm_function 相同的数据结构

**说明**：
- 与 llm_function 的 Step 1 完全相同
- 提取函数签名、类型提示、docstring
- 绑定参数并应用默认值
- 生成 trace_id

---

#### Step 2: `setup_log_context` ✅ **可复用**

**职责**：设置日志上下文

**输入**：
- `func_name: str` - 函数名
- `trace_id: str` - 追踪 ID
- `arguments: Dict[str, Any]` - 函数参数值

**输出**：
- `AsyncContextManager` - 日志上下文管理器

**说明**：
- 与 llm_function 的 Step 2 完全相同
- 创建 `async_log_context`
- 记录函数调用日志

---

#### Step 3: `build_chat_messages` ⚠️ **需要调整**

**职责**：根据函数元数据和实际参数构建聊天消息列表（System + History + User）

**输入**：
- `signature: FunctionSignature` - 函数签名信息
- `exclude_params: List[str]` - 需要排除的参数名（如 `history`, `chat_history`）

**输出**：
- `List[Dict[str, Any]]` - 消息列表
  - 包含 system message（docstring）
  - 包含 conversation history（如果有）
  - 包含 user message（参数值，排除 history）
  - 支持多模态内容

**说明**：
- 与 llm_function 的 Step 3 不同：
  - **不需要**构建参数类型描述和返回类型描述
  - **需要**提取和验证对话历史
  - **需要**排除 history 参数构建 user message
  - **需要**处理工具描述（如果提供工具，添加到 system prompt）

**当前对应代码**：
- `_build_messages` (llm_chat 版本)
- `_extract_history_from_args`
- `_build_user_message_content`
- `_process_tools`（工具准备）

---

#### Step 4: `execute_react_loop_streaming` ⚠️ **需要调整**

**职责**：执行 ReAct 循环，流式返回响应

**输入**：
- `llm_interface: LLM_Interface` - LLM 接口
- `messages: List[Dict[str, Any]]` - 初始消息列表
- `toolkit: Optional[List[Union[Tool, Callable]]]` - 工具列表
- `max_tool_calls: int` - 最大工具调用次数
- `stream: bool` - 是否流式返回
- `llm_kwargs: Dict[str, Any]` - LLM 额外参数
- `func_name: str` - 函数名

**输出**：
- `AsyncGenerator[Any, None]` - LLM 响应流

**说明**：
- 与 llm_function 的 Step 4 不同：
  - **不需要**重试逻辑（流式响应不需要）
  - **返回** `AsyncGenerator` 而不是单个响应
  - **需要**传递 `stream` 参数给 `execute_llm`

**当前对应代码**：
- `execute_llm` 的调用（来自 `base.ReAct`）
- `_process_tools` 的调用

---

#### Step 5: `process_chat_response_stream` ⚠️ **需要调整**

**职责**：处理流式响应，提取内容并返回（content, history）元组

**输入**：
- `response_stream: AsyncGenerator[Any, None]` - LLM 响应流
- `return_mode: Literal["text", "raw"]` - 返回模式
- `messages: List[Dict[str, Any]]` - 当前消息列表（用于返回 history）
- `func_name: str` - 函数名

**输出**：
- `AsyncGenerator[Tuple[Any, HistoryList], None]` - (响应内容, 历史) 元组流

**说明**：
- 与 llm_function 的 Step 5 完全不同：
  - **流式处理**：遍历响应流，每次 yield 一个结果
  - **返回模式**：
    - `"raw"` 模式：直接返回原始响应
    - `"text"` 模式：提取文本内容
  - **历史管理**：每次返回当前的消息列表作为历史
  - **流结束标记**：在流结束时 yield 空字符串（text 模式）

**当前对应代码**：
- `extract_content_from_response` / `extract_content_from_stream_response`
- 响应流的处理逻辑

---

## 重构后的 llm_chat 装饰器结构（伪代码）

```python
@wraps(func)
async def wrapper(*args, **kwargs) -> AsyncGenerator[Tuple[Any, HistoryList], None]:
    # Step 1: 解析函数签名（可复用）
    signature = parse_function_signature(func, args, kwargs)
    
    # Step 2: 设置日志上下文（可复用）
    async with setup_log_context(
        func_name=signature.func_name,
        trace_id=signature.trace_id,
        arguments=signature.bound_args.arguments,
    ):
        # 创建 Langfuse parent span
        with langfuse_client.start_as_current_observation(...) as chat_span:
            try:
                # Step 3: 构建聊天消息（chat 专用）
                messages = build_chat_messages(
                    signature=signature,
                    exclude_params=HISTORY_PARAM_NAMES,
                )
                
                # Step 4: 执行 ReAct 循环（流式版本）
                response_stream = execute_react_loop_streaming(
                    llm_interface=llm_interface,
                    messages=messages,
                    toolkit=toolkit,
                    max_tool_calls=max_tool_calls,
                    stream=stream,
                    llm_kwargs=llm_kwargs,
                    func_name=signature.func_name,
                )
                
                # Step 5: 处理响应流（chat 专用）
                collected_responses = []
                final_history = None
                
                async for content, history in process_chat_response_stream(
                    response_stream=response_stream,
                    return_mode=return_mode,
                    messages=messages,
                    func_name=signature.func_name,
                ):
                    collected_responses.append(content)
                    final_history = history
                    yield content, history
                
                # 更新 Langfuse span
                chat_span.update(
                    output={
                        "responses": collected_responses,
                        "final_history": final_history,
                        "total_responses": len(collected_responses),
                    },
                )
                
            except Exception as exc:
                chat_span.update(output={"error": str(exc)})
                raise
```

---

## 统一设计：共享步骤 vs 专用步骤

### 共享步骤（可完全复用）

1. **Step 1: `parse_function_signature`**
   - 两个装饰器完全相同
   - 可以放在共享模块中

2. **Step 2: `setup_log_context`**
   - 两个装饰器完全相同
   - 可以放在共享模块中

### 专用步骤（需要分别实现）

#### llm_function 专用

3. **Step 3: `build_initial_prompts`**
   - 构建包含参数类型和返回类型描述的 system prompt
   - 支持自定义模板
   - 处理 docstring 模板参数

4. **Step 4: `execute_react_loop`**
   - 返回单个最终响应
   - 包含重试逻辑

5. **Step 5: `parse_and_validate_response`**
   - 解析单个响应为返回类型
   - 类型转换和验证

#### llm_chat 专用

3. **Step 3: `build_chat_messages`**
   - 构建简单的 system prompt（docstring）
   - 提取和管理对话历史
   - 构建 user message（排除 history）

4. **Step 4: `execute_react_loop_streaming`**
   - 返回响应流（AsyncGenerator）
   - 无重试逻辑

5. **Step 5: `process_chat_response_stream`**
   - 流式处理响应
   - 支持 text/raw 两种模式
   - 返回 (content, history) 元组

---

## 代码组织建议

### 方案 1: 共享模块 + 专用模块

```
SimpleLLMFunc/llm_decorator/
├── __init__.py
├── llm_function_decorator.py      # llm_function 装饰器
├── llm_chat_decorator.py         # llm_chat 装饰器
├── steps/                        # 步骤函数模块
│   ├── __init__.py
│   ├── common/                   # 共享步骤
│   │   ├── __init__.py
│   │   ├── parse_signature.py    # Step 1
│   │   └── setup_log_context.py  # Step 2
│   ├── function/                 # llm_function 专用步骤
│   │   ├── __init__.py
│   │   ├── build_prompts.py      # Step 3
│   │   ├── execute_react.py      # Step 4
│   │   └── parse_response.py     # Step 5
│   └── chat/                     # llm_chat 专用步骤
│       ├── __init__.py
│       ├── build_messages.py      # Step 3
│       ├── execute_streaming.py  # Step 4
│       └── process_stream.py     # Step 5
└── utils/
    └── tools.py
```

### 方案 2: 统一接口，内部实现分离

保持当前文件结构，但在每个装饰器内部使用统一的步骤函数命名和接口：

```python
# llm_function_decorator.py
from .steps.common import parse_function_signature, setup_log_context
from .steps.function import build_initial_prompts, execute_react_loop, parse_and_validate_response

# llm_chat_decorator.py
from .steps.common import parse_function_signature, setup_log_context
from .steps.chat import build_chat_messages, execute_react_loop_streaming, process_chat_response_stream
```

---

## 数据流对比

### llm_function 数据流

```
装饰器入口
    │
    ├─> Step 1: parse_function_signature → FunctionSignature
    ├─> Step 2: setup_log_context → AsyncContextManager
    │   └─> [进入上下文]
    │       ├─> Step 3: build_initial_prompts → List[Dict] (messages)
    │       ├─> Step 4: execute_react_loop → Any (final_response)
    │       └─> Step 5: parse_and_validate_response → T (result)
    └─> [退出上下文] → return result
```

### llm_chat 数据流

```
装饰器入口
    │
    ├─> Step 1: parse_function_signature → FunctionSignature
    ├─> Step 2: setup_log_context → AsyncContextManager
    │   └─> [进入上下文]
    │       ├─> Step 3: build_chat_messages → List[Dict] (messages)
    │       ├─> Step 4: execute_react_loop_streaming → AsyncGenerator[Any]
    │       └─> Step 5: process_chat_response_stream → AsyncGenerator[Tuple[Any, HistoryList]]
    │           └─> async for (content, history) in stream:
    │               └─> yield content, history
    └─> [退出上下文]
```

---

## 关键设计决策

### 1. Step 1 和 Step 2 的复用

**决策**：完全复用，放在共享模块中

**理由**：
- 两个装饰器的签名解析和日志上下文设置逻辑完全相同
- 减少代码重复，提高可维护性

### 2. Step 3 的差异化

**决策**：分别实现 `build_initial_prompts` 和 `build_chat_messages`

**理由**：
- 消息构建逻辑差异较大
- llm_function 需要类型描述，llm_chat 不需要
- llm_chat 需要历史管理，llm_function 不需要

### 3. Step 4 的差异化

**决策**：分别实现 `execute_react_loop` 和 `execute_react_loop_streaming`

**理由**：
- llm_function 需要重试逻辑，llm_chat 不需要
- llm_chat 需要流式返回，llm_function 返回单个值
- 可以共享底层 `execute_llm` 调用，但包装逻辑不同

### 4. Step 5 的差异化

**决策**：完全不同的实现

**理由**：
- llm_function: 解析单个响应为返回类型
- llm_chat: 流式处理响应，返回 (content, history) 元组
- 逻辑完全不同，无法复用

---

## 总结

### 可复用性评估

| 步骤 | llm_function | llm_chat | 复用性 |
|------|-------------|----------|--------|
| Step 1 | ✅ | ✅ | **完全复用** |
| Step 2 | ✅ | ✅ | **完全复用** |
| Step 3 | ✅ | ⚠️ | **部分复用**（可共享多模态处理逻辑） |
| Step 4 | ✅ | ⚠️ | **部分复用**（共享 `execute_llm` 调用） |
| Step 5 | ✅ | ❌ | **无法复用**（逻辑完全不同） |

### 重构建议

1. **优先实现共享步骤**：Step 1 和 Step 2 可以立即提取到共享模块
2. **分别实现专用步骤**：Step 3、Step 4、Step 5 需要针对每个装饰器分别实现
3. **共享底层逻辑**：多模态处理、工具准备等可以提取为共享 helper 函数
4. **统一接口设计**：确保两个装饰器的步骤函数接口风格一致，便于维护

---

## 下一步工作（llm_chat）

1. **提取共享步骤**：将 Step 1 和 Step 2 提取到共享模块
2. **实现 Step 3**：`build_chat_messages` 函数
3. **实现 Step 4**：`execute_react_loop_streaming` 函数
4. **实现 Step 5**：`process_chat_response_stream` 函数
5. **重构装饰器**：使用5个步骤函数重构 `wrapper`
6. **测试**：确保流式响应、历史管理等功能正常

