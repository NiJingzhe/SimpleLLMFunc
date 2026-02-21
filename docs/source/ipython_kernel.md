# iPython iPyKernel 代码执行

SimpleLLMFunc 提供内置的 iPython iPyKernel 支持，允许 LLM 在一个连续上下文中执行 Python 代码。与传统的一次性代码执行不同，iPython iPyKernel 保持变量和状态，让 LLM 可以分步执行复杂任务。

## 功能特性

- **连续上下文**: 变量在多次调用间持久化，LLM 可以分步执行任务
- **实时 Streaming**: 通过 `event_emitter` 实时获取 stdout/stderr 输出
- **Session 隔离**: 不同的 iPyKernel 实例相互独立，互不影响
- **完整工具集**: 提供 execute_code、reset、list_variables、close 等工具

## 安装

iPython iPyKernel 功能需要 `jupyter-client` 包：

```bash
# 使用 poetry（推荐）
poetry add jupyter-client

# 或使用 pip
pip install jupyter-client
```

还需要安装并注册 IPython 内核：

```bash
pip install ipykernel
python -m ipykernel install --user --name=python3
```

## 快速开始

### 基本用法

```python
import asyncio
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.builtin import iPyKernel

# 创建 iPyKernel 实例
kernel = iPyKernel()

# 获取工具集
tools = kernel.toolset

# 使用 kernel 工具创建聊天机器人
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

### 多 iPyKernel 隔离

```python
# 创建两个独立的 kernel
kernel1 = iPyKernel()
kernel2 = iPyKernel()

# 分别使用
@llm_chat(toolkit=kernel1.toolset, ...)
async def chat1(message: str, history=None):
    """使用 kernel1 的助手"""

@llm_chat(toolkit=kernel2.toolset, ...)
async def chat2(message: str, history=None):
    """使用 kernel2 的助手"""
```

## 工具详解

### execute_code

执行 Python 代码，返回执行结果。

**参数：**

| 参数 | 类型 | 描述 |
|------|------|------|
| code | str | 要执行的 Python 代码 |
| event_emitter | ToolEventEmitter | 可选，事件发射器用于实时输出 |

**返回值：**

```python
{
    "success": bool,              # 是否成功执行
    "stdout": str,               # 标准输出
    "stderr": str,               # 标准错误
    "return_value": Any,         # 最后表达式的值
    "error": str | None,         # 错误信息
    "execution_time_ms": float   # 执行时间（毫秒）
}
```

### reset_kernel

重置 kernel 状态，清除所有变量。

```python
result = await kernel.reset()
# 返回:置，所有变量已 "iPyKernel 已重清除"
```

### list_variables

列出当前 kernel 中定义的所有变量。

```python
variables = await kernel.list_variables()
# 返回: [{"name": "x"}, {"name": "data"}]
```

### close_kernel

关闭 kernel，释放资源。

```python
await kernel.close()
```

## Streaming 事件

当 `enable_event=True` 时，`execute_code` 会实时发射以下事件：

| 事件名 | data 字段 | 描述 |
|--------|-----------|------|
| `kernel_stdout` | `{text: str, session_id: str}` | 标准输出 |
| `kernel_stderr` | `{text: str, session_id: str}` | 标准错误 |
| `kernel_result` | `{result: Any, session_id: str}` | 返回值 |
| `kernel_error` | `{error: str, session_id: str}` | 执行错误 |

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
```

## 使用示例

### 数据分析助手

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.builtin import iPyKernel
from SimpleLLMFunc.hooks import is_event_yield, CustomEvent
import sys

kernel = iPyKernel()

@llm_chat(
    llm_interface=llm,
    toolkit=kernel.toolset,
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
kernel = iPyKernel()
tools = kernel.toolset
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

### iPyKernel 构造函数参数

```python
kernel = iPyKernel(
    kernel_name="python3",  # 内核名称，默认 "python3"
    timeout=30,            # 执行超时（秒），默认 30
)
```

### 可用内核

列出可用的 Jupyter 内核：

```bash
jupyter kernelspec list
```

创建新的内核：

```bash
python -m ipykernel install --user --name=myenv
```

然后使用：

```python
kernel = iPyKernel(kernel_name="myenv")
```

## 最佳实践

### 1. 资源管理

```python
kernel = iPyKernel()

try:
    # 使用 kernel
    result = await execute(code="...")
finally:
    # 总是关闭 kernel
    await kernel.close()
```

### 2. Session 隔离

```python
# 为不同任务创建独立的 kernel
analysis_kernel = iPyKernel()
experiment_kernel = iPyKernel()

# 分析任务使用 analysis_kernel
@llm_chat(toolkit=analysis_kernel.toolset, ...)
async def analyze(message: str, history=None):
    pass

# 实验任务使用 experiment_kernel
@llm_chat(toolkit=experiment_kernel.toolset, ...)
async def experiment(message: str, history=None):
    pass
```

### 3. 错误处理

```python
result = await execute(code="可能出错的代码")

if not result['success']:
    print(f"执行错误: {result['error']}")
else:
    print(result['stdout'])
```

### 4. 实时反馈

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

## 故障排除

### 内核未找到

```
NoSuchiPyKernel: python3
```

解决方案：
```bash
pip install ipykernel
python -m ipykernel install --user --name=python3
```

### 超时错误

增加超时时间：
```python
kernel = iPyKernel(timeout=120)  # 2分钟
```

### 变量未定义

每次 `execute_code` 调用是独立的，但共享同一个 kernel 状态。确保：
1. 变量已在之前的调用中定义
2. 没有调用 `reset_kernel` 清除变量

## 相关链接

- [Jupyter Client 文档](https://jupyter-client.readthedocs.io/)
- [IPython 内核](https://ipython.readthedocs.io/)
- 示例文件：`examples/ipython_kernel_example.py`
