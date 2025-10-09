# LLM Chat 装饰器

本文档介绍 SimpleLLMFunc 库中的聊天装饰器 `llm_chat`。该装饰器专门用于实现与大语言模型的对话功能，支持多轮对话、历史记录管理和工具调用。与 `llm_function` 装饰器不同，`llm_chat` 更适合构建聊天应用、助手系统和需要保持上下文的交互场景。

## llm_chat 装饰器

### 装饰器作用

`llm_chat` 装饰器专门用于实现与大语言模型的对话功能，支持多轮对话、历史记录管理和工具调用。

### 主要功能特性
- **多轮对话支持**: 自动管理对话历史记录，支持上下文连续性
- **流式响应**: 返回生成器，支持实时响应流
- **历史记录过滤**: 自动过滤工具调用信息，只保留用户和助手的对话内容
- **工具集成**: 支持在对话中调用工具，扩展 LLM 的能力
- **灵活参数处理**: 智能处理历史记录参数，其他参数作为用户消息
- **错误处理**: 完善的异常处理和日志记录机制

## 装饰器用法

> ⚠️ **重要说明**：`llm_chat` 只能装饰 `async def` 定义的异步函数，调用时需要在异步上下文中使用 `await`。

### 基本语法

def your_chat_function(message: str, history: List[Dict[str, str]] = []) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
```python
from typing import AsyncGenerator, Dict, List, Tuple

from SimpleLLMFunc.llm_decorator import llm_chat


@llm_chat(
    llm_interface=llm_interface,
    toolkit=None,
    max_tool_calls=5,
    **llm_kwargs,
)
async def your_chat_function(
    message: str,
    history: List[Dict[str, str]] | None = None,
) -> AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]:
    """在这里描述聊天助手的角色和行为规则"""
    yield "", history or []
```

### 参数说明

## 异步使用示例

`llm_chat` 是原生异步实现，以下示例展示了不同场景的用法：

### 示例 1: 基本聊天

```python
import asyncio
from typing import AsyncGenerator, Dict, List, Tuple

from SimpleLLMFunc.llm_decorator import llm_chat


@llm_chat(llm_interface=llm, stream=True)
async def chat_assistant(
    history: List[Dict[str, str]] | None,
    message: str,
) -> AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]:
    """你是一个友好的聊天助手，善于回答各种问题。"""
    pass


async def main():
    async for chunk, updated_history in chat_assistant([], "你好！"):
        if chunk:
            print(chunk, end="")


asyncio.run(main())
```

### 示例 2: 工具调用与原始响应

```python
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Tuple

from SimpleLLMFunc.tool import tool


@tool(name="lookup_weather", description="查询指定城市的天气")
async def lookup_weather(city: str) -> str:
    return f"{city} 晴，25℃"


@llm_chat(
    llm_interface=llm,
    toolkit=[lookup_weather],
    stream=True,
    return_mode="raw",
)
async def weather_assistant(
    history: List[Dict[str, str]] | None,
    message: str,
) -> AsyncGenerator[Tuple[Any, List[Dict[str, str]]], None]:
    """你是一个天气助手，可以查询城市天气。"""
    pass


async def main():
    async for raw, _ in weather_assistant([], "今天北京天气怎么样？"):
        print(raw)


asyncio.run(main())
```

### 示例 3: 多会话并发

```python
import asyncio
from typing import AsyncGenerator, Dict, List, Tuple


@llm_chat(llm_interface=llm, stream=True)
async def chat_session(
    history: List[Dict[str, str]] | None,
    message: str,
) -> AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]:
    """你是一个支持多会话的客服助手。"""
    pass


async def handle_session(session_id: int, question: str):
    collected = ""
    async for chunk, _ in chat_session([], question):
        if chunk:
            collected += chunk
    print(f"会话 {session_id}: {collected}")


async def main():
    questions = [
        "介绍下SimpleLLMFunc",
        "请用中文解释LLM Function",
        "列举几个使用场景"
    ]

    await asyncio.gather(*[
        handle_session(i, q) for i, q in enumerate(questions, start=1)
    ])


asyncio.run(main())
```

这些示例展示了如何使用 `llm_chat` 在异步环境中构建实时聊天体验。
    timeout=600
)
def GLaDos(history: List[Dict[str, str]], query: str):  # type: ignore
    """
    你是GLaDos，一为全能AI助手。

    由于你不能和控制台交互，所有的测试都需要首先使用unittest编写专门的测试脚本，并通过mock输入的方法来绕开控制台输入。

    使用工具前请务必说明你要用什么工具做什么。


    首先需要分析用户的需求，然后使用execute_command工具查看当前的工作环境，然后
    建议遵循以下过程：
        1. 使用file_operator工具创建TODO.md文档，用checkbox的形式将用户需求拆解成多个详细描述的小任务，并记录。
            任务拆分务必拆分到最细致的粒度，推荐任何任务都拆分到10个子任务以上。
        2. 使用file_operator工具读取TODO.md文档，检查任务列表
        3. 逐步执行计划
        4. 撰写每个部分的代码和测试代码（如果是代码任务）
        5. 根据结果反思执行效果，并继续下一步或者作出弥补
        6. 使用file_operator工具更新TODO.md文档

    直到你认为任务已经完成，输出"<<任务完成>>"字样

    """
    pass

if __name__ == "__main__":
    # 测试流式响应
    history = []
    query = "请帮我完成一个Python项目的开发"
    
    for response_chunk, updated_history in GLaDos(history, query):
        if response_chunk:
            stdout.write(response_chunk)
            stdout.flush()
            time.sleep(0.1) 
        else:
            history = updated_history
            break

    print()  # 换行
```

### 示例 6: 在Class中使用装饰器

```python
class CADAgent:

    def __init__(
        self,
        llm_interface: LLM_Interface,
        max_tool_iterations: int = 50,
        max_memory_length: int = 10,
    ):

        self.llm_interface: LLM_Interface = llm_interface
        self.max_tool_iterations: int = max_tool_iterations

        self.memory: list[dict[str, str]] = []
        self.max_memory_length: int = max_memory_length

        # 在实例化后应用装饰器
        self.chat = llm_chat(
            llm_interface=self.llm_interface,
            max_tool_calls=self.max_tool_iterations,
            toolkit=[
                file_operator,
                execute_command,
                get_current_time_and_date,
                write_code,
                make_user_query_more_detailed,
            ],
            timeout=600,
        )(self.chat_impl)

    @staticmethod
    def chat_impl(history: List[Dict[str, str]], user_requirement: str):  # type: ignore
        """
        ### 身份：
        你是一位专业的CAD设计师，同时精通PythonOCC框架。

        ### 任务：
        - 你需要根据用户的需求(user_requirement)，和用户进行亲切的对话，回答问题或生成高质量的PythonOCC代码

        ### 输出格式：
        每次使用工具前，说明你要做什么。
        每次工具使用之后，说明达到了什么效果或者目的。

        ## 提醒：
        1.  善用查看当前文件夹下的文件的能力，看看有没有什么能够帮助你的文件
        2.  尽可能自动的完成从完善的需求到写代码到导出文件的全过程
        """
        pass

    def run(self, query: str) -> Generator[str, None, None]:
        """
        运行CADAgent，处理用户的查询。
        """
        # 处理内存长度
        if len(self.memory) > self.max_memory_length:
            # 保留第一条，然后pop掉第二条
            self.memory.pop(1)

        query = query.strip() + "。请务必不要忘记使用工具, 以及将代码写入本地文件。"
## 异步使用示例

# 使用异步聊天代理
async def use_async_chat_agent():
    agent = AsyncChatAgent(llm_interface=llm)
    
    queries = [
        "现在几点了？",
        "计算一下 100 + 200",
        "北京今天天气怎么样？"
    ]
    
    for query in queries:
        print(f"\n用户: {query}")
        print("助手: ", end="")
        
        async for response in agent.run(query):
            print(response, end="", flush=True)
        print()

# 运行异步聊天代理
asyncio.run(use_async_chat_agent())
```

## 最佳实践

### 1. 错误处理
```python
async def robust_async_chat():
    try:
        async for content, history in your_async_chat_function(history=[], message="测试"):
            if content:
                print(content, end="")
            else:
                break
    except Exception as e:
        print(f"聊天调用失败: {e}")
```

### 2. 超时控制
```python
async def chat_with_timeout():
    try:
        # 为整个聊天会话设置超时
        async def chat_session():
            async for content, history in async_chat_function(history=[], message="测试"):
                if content:
                    yield content
                else:
                    break
        
        async for content in asyncio.wait_for(chat_session(), timeout=30.0):
            print(content, end="")
            
    except asyncio.TimeoutError:
        print("聊天会话超时")
```

### 3. 并发控制
```python
# 使用信号量控制并发数量
semaphore = asyncio.Semaphore(3)  # 最多3个并发聊天

async def controlled_concurrent_chat(message: str):
    async with semaphore:
        async for content, history in async_chat_function(history=[], message=message):
            if content:
                yield content
            else:
                break
```

---

通过这些示例可以看出，`llm_chat` 装饰器在异步场景下同样能够提供强大的对话能力，同时保持了良好的易用性和功能完整性，适合构建需要同时处理多个会话的聊天机器人或客服系统。