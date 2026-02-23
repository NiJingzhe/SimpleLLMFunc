# PyRepl 代码执行

SimpleLLMFunc 提供内置的 PyRepl 支持，允许 LLM 在一个连续上下文中执行 Python 代码。与传统的一次性代码执行不同，PyRepl 保持变量和状态，让 LLM 可以分步执行复杂任务。

## 功能特性

- **IPython subprocess backend**：每个 `PyRepl` 实例对应一个独立子进程，内部运行 `IPython InteractiveShell`
- **连续上下文**：变量在多次调用间持久化，LLM 可以分步执行任务
- **实时 Streaming**：通过 `event_emitter` 实时获取 stdout/stderr 输出
- **异步不阻塞**：代码执行在独立线程运行，不阻塞主事件循环（适合 TUI/流式 UI）
- **Session 隔离**：不同的 PyRepl 实例相互独立，互不影响
- **完整工具集**：提供 execute_code、reset_repl、list_variables 等工具
- **超时保护**：单次 `execute_code` 默认 120 秒活动执行超时；等待 `input()` 不计时，且每次收到输入后会重置超时窗口；单次 `input()` 默认 300 秒空闲超时
- **Self-reference support**: expose controlled memory handles through `SelfReference` and `self_reference.memory["key"]`

## 快速开始

### 基本用法

```python
import asyncio
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.builtin import PyRepl

# 创建 PyRepl 实例
repl = PyRepl()

# 获取工具集
tools = repl.toolset

# 使用 repl 工具创建聊天机器人
@llm_chat(
    llm_interface=llm,
    toolkit=tools,
    enable_event=True,
)
async def python_assistant(message: str, history=None):
    """
    你是一个 Python 编程助手。
    用户会给你编程任务，你需要编写代码来完成。
    记住：变量会在后续调用中保持。
    """

# 使用
async for output in python_assistant("创建一个列表并计算均值"):
    # 处理输出
    pass
```

### 多 PyRepl 隔离

```python
# 创建两个独立的 repl
repl1 = PyRepl()
repl2 = PyRepl()

# 分别使用
@llm_chat(toolkit=repl1.toolset, ...)
async def chat1(message: str, history=None):
    """使用 repl1 的助手"""

@llm_chat(toolkit=repl2.toolset, ...)
async def chat2(message: str, history=None):
    """使用 repl2 的助手"""
```

## 工具详解

### execute_code

执行 Python 代码，返回执行结果。

> 说明：`execute_code` 默认有 120 秒活动执行超时保护。等待 `input()` 期间不会计入超时；每次 `input()` 成功回填后会重置超时计时。同时，单次 `input()` 默认 300 秒空闲超时。任一超时触发时 `success=False`，并在 `error`/`stderr` 中返回超时信息。

> Agent guidance in tool description (sent to model): write direct snippets instead of standalone scripts, **do not** add `if __name__ == "__main__":`, `input()` is supported, and `reset_repl` does **not** delete self-reference memory.

**参数：**

| 参数 | 类型 | 描述 |
|------|------|------|
| code | str | 要执行的 Python 代码 |
| event_emitter | ToolEventEmitter | 可选，事件发射器用于实时输出 |

**返回值：**

```python
{
    "success": bool,              # 是否成功执行
    "stdout": str,                # 标准输出
    "stderr": str,                # 标准错误
    "return_value": Any,          # 最后表达式的值
    "error": str | None,          # 错误信息（可直接定位到输入代码行）
    "error_details": dict | None, # 结构化错误详情（行号/列号/代码片段/指针等）
    "execution_time_ms": float    # 执行时间（毫秒）
}
```

### 错误定位增强

`execute_code` 会尽量直接返回输入代码的定位信息，而不是仅显示框架内部 `exec` 栈。

典型字段（在 `error_details` 中）：

- `error_type`: 异常类型（如 `SyntaxError`、`ZeroDivisionError`）
- `message`: 异常原始消息
- `line` / `column`: 出错行列（若可解析）
- `snippet`: 出错行源码
- `pointer`: 列指针（例如 `^`）
- `summary`: 面向模型/用户的简洁可读报错摘要
- `user_traceback`: 聚焦用户代码栈的 traceback 文本

示例：

```python
result = await execute(code="for i in range(2)\n    print(i)")
if not result["success"]:
    print(result["error"])
    print(result["error_details"])
```

### reset_repl

重置 repl 状态，清除所有变量。

> Model-facing tool description is in English and explicitly states: reset only affects REPL variables and preserves attached `self_reference` object.

```python
result = await repl.reset()
# 返回: "REPL 已重置，所有变量已清除"
```

### list_variables

列出当前 repl 中定义的所有变量。

> Model-facing tool description is in English and clarifies that private names and `self_reference` are excluded.

```python
variables = await repl.list_variables()
# 返回: [{"name": "x", "type": "int"}, {"name": "data", "type": "list"}]
```

## Streaming 事件

当 `enable_event=True` 时，`execute_code` 会实时发射以下事件：

| 事件名 | data 字段 | 描述 |
|--------|-----------|------|
| `kernel_stdout` | `{text: str}` | 标准输出 |
| `kernel_stderr` | `{text: str}` | 标准错误 |
| `kernel_input_request` | `{request_id: str, prompt: str, idle_timeout_seconds: float}` | `input()` 请求用户输入（含本次输入空闲超时） |

### 捕获 Streaming 事件

```python
from SimpleLLMFunc.hooks import is_event_yield, CustomEvent

async for output in llm_chat_function(message):
    if is_event_yield(output):
        event = output.event
        if isinstance(event, CustomEvent):
            if event.event_name == "kernel_stdout":
                print(f"[stdout] {event.data['text']}", end="")
            elif event.event_name == "kernel_stderr":
                print(f"[stderr] {event.data['text']}", end="", file=sys.stderr)

# 说明：当 event_name == "kernel_input_request" 时，
# 你可以把用户输入通过 PyRepl.submit_input(request_id, value) 回填。
```

## 使用示例

### 数据分析助手

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.hooks import is_event_yield, CustomEvent
import sys

repl = PyRepl()

@llm_chat(
    llm_interface=llm,
    toolkit=repl.toolset,
    enable_event=True,
)
async def data_helper(message: str, history=None):
    """
    你是一个数据分析助手。
    使用 Python 代码完成数据分析任务。
    每次只执行一小段代码，使用 print() 输出结果。
    """

# 执行任务
async for output in data_helper(
    "创建一个包含100个随机数的列表，计算均值和标准差"
):
    if is_event_yield(output):
        event = output.event
        if isinstance(event, CustomEvent):
            if event.event_name == "kernel_stdout":
                print(event.data['text'], end="")
```

### 连续编程上下文

```python
repl = PyRepl()
tools = repl.toolset
execute = next(t for t in tools if t.name == "execute_code")

# 第一次调用：定义数据
result1 = await execute(code="""
import random
data = [random.randint(1, 100) for _ in range(10)]
print(f"创建了 {len(data)} 个随机数")
print(f"数据: {data}")
""")

# 第二次调用：使用之前的数据
result2 = await execute(code="""
mean = sum(data) / len(data)
print(f"均值: {mean}")
""")

# 变量 data 仍然可用！
print(result2['stdout'])  # "均值: 52.3"
```

## 配置选项

### PyRepl 构造函数参数

```python
# 默认活动执行超时为 120 秒
repl = PyRepl()

# 可按需调整活动执行超时（单位：秒）
repl = PyRepl(execution_timeout_seconds=180)

# 也可调整 input 空闲超时（单位：秒，默认 300）
repl = PyRepl(input_idle_timeout_seconds=300)

# 两者都可配置
repl = PyRepl(execution_timeout_seconds=180, input_idle_timeout_seconds=300)
```

## SelfReference Binding

Use explicit wiring for `SelfReference` as a framework-level memory object:

```python
from SimpleLLMFunc import SelfReference, llm_chat
from SimpleLLMFunc.builtin import PyRepl

self_reference = SelfReference()
repl = PyRepl()
repl.attach_self_reference(self_reference)

@llm_chat(
    llm_interface=llm,
    toolkit=repl.toolset,
    self_reference=self_reference,
    self_reference_key="agent_main",
)
async def agent(message: str, history=None):
    ...
```

When `@llm_chat(...)` receives `self_reference`, the framework automatically appends a deduplicated **SelfReference Memory Contract** to the end of the current system prompt. This contract tells the agent how to operate memory via `self_reference.memory["<key>"]`.

For first-turn safety, if this memory key is empty, the current durable system prompt is seeded into `self_reference` before tool execution, so `self_reference.memory["<key>"].all()` is not empty and includes a system message.

Inside `execute_code`, the agent accesses memory through the controlled proxy:

```python
# Run inside execute_code
mem = self_reference.memory["agent_main"]
print(mem.count())
mem.append({"role": "user", "content": "[plan] step 1"})
```

### Append durable memory into system prompt

This is the most common and recommended pattern: persist user preferences in system prompt memory.

```python
# Run inside execute_code
mem = self_reference.memory["agent_main"]
mem.append_system_prompt("User preference: answer in concise bullet points.")
mem.append_system_prompt("Always include one actionable next step.")

print(mem.get_system_prompt())
```

You can also overwrite the system prompt directly:

```python
mem = self_reference.memory["agent_main"]
mem.set_system_prompt("You are a concise coding assistant.")
```

### Common memory operation examples

```python
mem = self_reference.memory["agent_main"]

# Read
print(mem.count())
print(mem.all())
print(mem.get(0))

# Write
mem.append({"role": "user", "content": "remember this"})
mem.insert(1, {"role": "assistant", "content": "inserted"})
mem.update(1, {"role": "assistant", "content": "updated"})
mem.delete(1)

# Bulk
mem.replace([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "hello"},
])
```

Method reference (purpose of each method):

- `count()`: return current message count for this memory key.
- `all()`: return a deep-copied snapshot of all messages.
- `get(index)`: read one message by index.
- `append(message)`: append one message to the tail.
- `insert(index, message)`: insert one message at a specific index.
- `update(index, message)`: replace one message at a specific index.
- `delete(index)`: remove one message by index.
- `replace(messages)`: replace full history with a validated list.
- `clear()`: clear all messages for this memory key.
- `get_system_prompt()`: read the latest system prompt text.
- `set_system_prompt(text)`: overwrite current system prompt.
- `append_system_prompt(text)`: append text to system prompt using a newline.

Forgetting note:

- Forgetting memory is **not** `reset_repl`.
- `reset_repl` only clears Python variables in the REPL namespace.
- Forget memory by deleting message records with `delete(index)`, `replace(messages)`, or `clear()`.

All operations write into `SelfReference`'s internal store instead of exposing raw lists. Memory edits performed within a turn are merged into returned `updated_history` (and `ReactEndEvent.final_messages` in event mode) at turn end.

### Multi-agent sharing in one REPL

Use different `self_reference_key` values for different agents:

```python
@llm_chat(..., self_reference=self_reference, self_reference_key="agent_1")
async def agent_1(message: str, history=None):
    ...

@llm_chat(..., self_reference=self_reference, self_reference_key="agent_2")
async def agent_2(message: str, history=None):
    ...
```

This gives each agent an isolated memory space.

## 最佳实践

### 1. Session 隔离

```python
# 为不同任务创建独立的 repl
analysis_repl = PyRepl()
experiment_repl = PyRepl()

# 分析任务使用 analysis_repl
@llm_chat(toolkit=analysis_repl.toolset, ...)
async def analyze(message: str, history=None):
    pass

# 实验任务使用 experiment_repl
@llm_chat(toolkit=experiment_repl.toolset, ...)
async def experiment(message: str, history=None):
    pass
```

### 2. 错误处理

```python
result = await execute(code="可能出错的代码")

if not result['success']:
    print(f"执行错误: {result['error']}")
    # 可选：读取结构化定位信息
    print(result.get('error_details'))
else:
    print(result['stdout'])
```

### 5. 审计日志（每实例独立）

`PyRepl` 会把代码执行审计记录落盘到独立目录：

- 根目录来自 `.env` / 环境变量中的 `LOG_DIR`
- 每个实例独立子目录：`<LOG_DIR>/pyrepl/<instance_id>/`
- 审计文件：`executions.jsonl`

每条记录包含：执行时间、代码、执行结果、结构化错误详情、超时配置等。

```python
repl = PyRepl()
print(repl.instance_id)
print(repl.audit_log_dir)
print(repl.audit_log_file)
```

### 3. 实时反馈

启用 `event_emitter` 获取实时输出，提供更好的用户体验：

```python
from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

emitter = ToolEventEmitter()

result = await execute(
    code="for i in range(10): print(i); import time; time.sleep(0.5)",
    event_emitter=emitter
)

# 同时处理事件和最终结果
events = await emitter.get_events()
for event in events:
    print(event.event.data)
```

### 4. 重置状态

当需要重新开始时，可以使用 `reset_repl` 清除所有变量：

```python
# 重置 repl
result = await repl.reset()
print(result)  # "REPL 已重置，所有变量已清除"
```

## Related Links

- Example: `examples/pyrepl_example.py`
- Local SelfReference demo: `examples/self_reference_basic_example.py`
- TUI SelfReference demo: `examples/tui_self_reference_example.py`
