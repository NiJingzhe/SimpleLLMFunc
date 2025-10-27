# llm_chat 装饰器

本文档介绍 SimpleLLMFunc 库中的聊天装饰器 `llm_chat`。该装饰器专门用于实现与大语言模型的对话功能，支持多轮对话、历史记录管理和工具调用。

## llm_chat 装饰器概述

### 装饰器作用

`llm_chat` 装饰器用于构建对话式应用，特别适合以下场景：

- **多轮对话**: 自动管理对话历史，支持上下文连续性
- **流式响应**: 支持实时流式返回响应内容
- **智能助手**: 集成工具调用能力，让 LLM 可以执行外部操作
- **聊天机器人**: 适合构建实时交互的聊天应用

### 主要功能特性

- **多轮对话支持**: 自动管理对话历史记录，保持上下文
- **流式响应**: 返回异步生成器，支持实时流式输出
- **工具集成**: 支持在对话中调用工具，扩展 LLM 的能力范围
- **灵活参数处理**: 智能处理历史记录参数和用户消息
- **完整的日志记录**: 与框架日志系统集成，自动追踪对话

## 装饰器用法

> ⚠️ **重要说明**：`llm_chat` 可以装饰 `async def` 定义的异步函数，也可以装饰 `def` 定义的同步函数，但是返回的结果一定是一个 `async` 函数。如果要使用这个函数，要使用`await`或者`asyncio.run()`来执行。

### 基本语法

```python
from typing import AsyncGenerator, List, Dict, Tuple
from SimpleLLMFunc import llm_chat

@llm_chat(
    llm_interface=llm_interface,           # LLM 接口实例（必需）
    toolkit=None,                          # 工具列表（可选）
    max_tool_calls=5,                      # 最大工具调用次数（可选）
    stream=True,                           # 是否启用流式模式（可选）
    **llm_kwargs                           # 其他 LLM 参数
)
async def your_chat_function(
    message: str,
    history: List[Dict[str, str]] | None = None,
) -> AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]:
    """
    在这里描述聊天助手的角色和行为规则。
    这个 docstring 会被作为系统提示传递给 LLM。
    """
    yield "", history or []
```

### 参数说明

- **llm_interface** (必需): LLM 接口实例，用于与大语言模型通信
- **toolkit** (可选): 工具列表，可以是 Tool 对象或被 @tool 装饰的函数
- **max_tool_calls** (可选): 最大工具调用次数，防止无限循环，默认为 5
- **stream** (可选): 是否启用流式模式，默认为 True
- **return_mode** (可选): 返回模式，可选值为 "text"（默认）或 "raw"
- ****llm_kwargs**: 额外的关键字参数，将直接传递给 LLM 接口（如 temperature、top_p 等）

### 返回值

`llm_chat` 装饰的函数返回一个异步生成器，每次迭代返回：

- `chunk` (str): 响应内容的一部分（流式模式）或完整响应（非流式）
- `updated_history` (List[Dict[str, str]]): 更新后的对话历史

## 使用示例

### 示例 1: 基础聊天助手

最简单的对话助手实现：

```python
import asyncio
from typing import AsyncGenerator, Dict, List, Tuple
from SimpleLLMFunc import llm_chat, OpenAICompatible

# 初始化 LLM 接口
llm = OpenAICompatible.load_from_json_file("provider.json")["openai"]["gpt-3.5-turbo"]

# 创建聊天函数
@llm_chat(llm_interface=llm, stream=True)
async def simple_chat(
    message: str,
    history: List[Dict[str, str]] | None = None,
) -> AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]:
    """你是一个友好的聊天助手，善于回答各种问题。"""
    yield "", history or []

# 使用示例
async def main():
    history = []
    user_message = "你好，请介绍一下你自己"

    print(f"用户: {user_message}")
    print("助手: ", end="", flush=True)

    # 流式获取响应
    async for chunk, updated_history in simple_chat(user_message, history):
        if chunk:
            print(chunk, end="", flush=True)
        history = updated_history

    print()  # 换行

asyncio.run(main())
```

### 示例 2: 带工具调用的聊天助手

展示如何在对话中使用工具：

```python
import asyncio
from typing import AsyncGenerator, Dict, List, Tuple
from SimpleLLMFunc import llm_chat, tool, OpenAICompatible

# 定义工具
@tool(name="get_weather", description="获取指定城市的天气信息")
async def get_weather(city: str) -> Dict[str, str]:
    """
    获取指定城市的天气信息

    Args:
        city: 城市名称

    Returns:
        包含温度、湿度和天气状况的字典
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temperature": "25°C", "humidity": "60%", "condition": "晴朗"},
        "上海": {"temperature": "28°C", "humidity": "75%", "condition": "多云"},
        "广州": {"temperature": "30°C", "humidity": "80%", "condition": "小雨"}
    }
    return weather_data.get(city, {"temperature": "20°C", "humidity": "50%", "condition": "未知"})

# 初始化 LLM
llm = OpenAICompatible.load_from_json_file("provider.json")["openai"]["gpt-3.5-turbo"]

# 创建带工具的聊天函数
@llm_chat(llm_interface=llm, toolkit=[get_weather], stream=True)
async def weather_chat(
    message: str,
    history: List[Dict[str, str]] | None = None,
) -> AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]:
    """
    你是一个天气助手，可以查询城市天气信息。
    当用户询问天气时，使用 get_weather 工具来获取实时信息。
    """
    yield "", history or []

# 使用示例
async def main():
    history = []
    query = "北京今天天气怎么样？"

    print(f"用户: {query}")
    print("助手: ", end="", flush=True)

    async for chunk, updated_history in weather_chat(query, history):
        if chunk:
            print(chunk, end="", flush=True)
        history = updated_history

    print()

asyncio.run(main())
```

### 示例 3: 交互式多轮对话

展示如何维护完整的对话会话：

```python
import asyncio
from typing import AsyncGenerator, Dict, List, Tuple
from SimpleLLMFunc import llm_chat, OpenAICompatible

llm = OpenAICompatible.load_from_json_file("provider.json")["openai"]["gpt-3.5-turbo"]

@llm_chat(llm_interface=llm, stream=True)
async def multi_turn_chat(
    message: str,
    history: List[Dict[str, str]] | None = None,
) -> AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]:
    """你是一个专业的编程助手，精通 Python 和 JavaScript。"""
    yield "", history or []

async def interactive_chat_session():
    """运行交互式聊天会话"""
    history = []

    print("=== 编程助手（输入 'quit' 退出）===\n")

    # 这里使用 input() 只是为了演示，实际应用中应使用异步输入
    while True:
        # 在实际应用中，应该使用更好的异步输入方法
        user_input = input("你: ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        print("助手: ", end="", flush=True)

        response_text = ""
        async for chunk, updated_history in multi_turn_chat(user_input, history):
            if chunk:
                print(chunk, end="", flush=True)
                response_text += chunk
            history = updated_history

        print("\n")

# 非交互式演示（避免阻塞 input()）
async def demo():
    """演示版本，不使用交互式输入"""
    history = []

    messages = [
        "Python 中什么是列表推导式？",
        "如何使用异步编程？",
    ]

    for user_message in messages:
        print(f"\n用户: {user_message}")
        print("助手: ", end="", flush=True)

        async for chunk, updated_history in multi_turn_chat(user_message, history):
            if chunk:
                print(chunk, end="", flush=True)
            history = updated_history

        print()

asyncio.run(demo())
```

## 高级特性

### 返回模式

`return_mode` 参数控制返回的数据类型：

```python
# 返回文本（默认）
@llm_chat(llm_interface=llm, stream=True, return_mode="text")
async def text_mode_chat(message: str, history=None):
    """聊天函数"""
    yield "", history or []

# 返回原始响应对象（用于获取 token 使用量等详细信息）
@llm_chat(llm_interface=llm, stream=True, return_mode="raw")
async def raw_mode_chat(message: str, history=None):
    """聊天函数"""
    yield "", history or []
```

### 并发聊天会话

使用 `asyncio.gather` 处理多个并发的聊天会话：

```python
async def concurrent_chats():
    """并发处理多个聊天会话"""

    @llm_chat(llm_interface=llm, stream=True)
    async def chat(message: str, history=None):
        """通用聊天函数"""
        yield "", history or []

    # 定义多个会话
    sessions = [
        {"user_id": "user_1", "message": "你好"},
        {"user_id": "user_2", "message": "如何学习Python？"},
        {"user_id": "user_3", "message": "告诉我一个笑话"},
    ]

    async def handle_session(session):
        """处理单个会话"""
        history = []
        results = []

        async for chunk, updated_history in chat(session["message"], history):
            if chunk:
                results.append(chunk)
            history = updated_history

        return session["user_id"], "".join(results)

    # 并发执行所有会话
    results = await asyncio.gather(
        *[handle_session(session) for session in sessions]
    )

    for user_id, response in results:
        print(f"{user_id}: {response}\n")

asyncio.run(concurrent_chats())
```

## 最佳实践

### 1. 错误处理

```python
async def robust_chat():
    history = []
    try:
        async for chunk, updated_history in multi_turn_chat("测试", history):
            if chunk:
                print(chunk, end="", flush=True)
            history = updated_history
    except Exception as e:
        print(f"聊天出错: {e}")
```

### 2. 超时控制

```python
async def chat_with_timeout():
    history = []
    try:
        async with asyncio.timeout(30):  # Python 3.11+
            async for chunk, updated_history in multi_turn_chat("测试", history):
                if chunk:
                    print(chunk, end="", flush=True)
                history = updated_history
    except asyncio.TimeoutError:
        print("聊天超时")
```

### 3. 历史记录限制

为避免上下文过长，限制历史记录长度：

```python
MAX_HISTORY_LENGTH = 10

def trim_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """保留最近的 N 条消息"""
    if len(history) > MAX_HISTORY_LENGTH:
        return history[-MAX_HISTORY_LENGTH:]
    return history

async def chat_with_limited_history():
    history = []

    messages = ["第一条消息", "第二条消息", "第三条消息"]

    for msg in messages:
        # 限制历史记录
        history = trim_history(history)

        async for chunk, updated_history in multi_turn_chat(msg, history):
            if chunk:
                print(chunk, end="", flush=True)
            history = updated_history
        print()

asyncio.run(chat_with_limited_history())
```

### 4. 日志与调试

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# SimpleLLMFunc 日志
logger = logging.getLogger("SimpleLLMFunc")
logger.setLevel(logging.DEBUG)
```

## 常见问题

### Q: 如何保存和恢复对话历史？

```python
import json

def save_history(history: List[Dict[str, str]], filename: str):
    """保存对话历史到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(filename: str) -> List[Dict[str, str]]:
    """从文件加载对话历史"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# 使用
history = load_history("chat_history.json")
# ... 继续对话 ...
save_history(history, "chat_history.json")
```

### Q: 如何处理 LLM 拒绝或无效响应？

```python
async def robust_chat_with_retry():
    history = []
    max_retries = 3

    for attempt in range(max_retries):
        try:
            collected = ""
            async for chunk, updated_history in multi_turn_chat("测试", history):
                if chunk:
                    collected += chunk
                history = updated_history

            if collected.strip():
                print(f"成功: {collected}")
                break
            else:
                print(f"尝试 {attempt + 1}: 收到空响应，重试...")
        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {e}")
            if attempt == max_retries - 1:
                raise
```

---

通过这些示例和最佳实践，你可以构建功能强大的对话应用。`llm_chat` 装饰器提供了简洁而强大的方式来实现复杂的对话逻辑。
