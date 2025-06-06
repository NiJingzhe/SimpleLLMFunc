# llm_chat 装饰器

## 装饰器作用

`llm_chat` 装饰器专门用于实现与大语言模型的对话功能，支持多轮对话、历史记录管理和工具调用。与 `llm_function` 装饰器不同，`llm_chat` 更适合构建聊天应用、助手系统和需要保持上下文的交互场景。

### 主要功能特性
- **多轮对话支持**: 自动管理对话历史记录，支持上下文连续性
- **流式响应**: 返回生成器，支持实时响应流
- **历史记录过滤**: 自动过滤工具调用信息，只保留用户和助手的对话内容
- **工具集成**: 支持在对话中调用工具，扩展 LLM 的能力
- **灵活参数处理**: 智能处理历史记录参数，其他参数作为用户消息
- **错误处理**: 完善的异常处理和日志记录机制

## 装饰器用法

### 基本语法

```python
from SimpleLLMFunc.llm_decorator import llm_chat

@llm_chat(
    llm_interface=llm_interface,
    toolkit=None,
    max_tool_calls=5,
    **llm_kwargs
)
def your_chat_function(message: str, history: List[Dict[str, str]] = []) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
    """在这里描述聊天助手的角色和行为规则"""
    pass
```

### 参数说明

- **llm_interface** (必需): LLM 接口实例，用于与大语言模型通信
- **toolkit** (可选): 工具列表，可以是 Tool 对象或被 @tool 装饰的函数
- **max_tool_calls** (可选): 最大工具调用次数，防止无限循环，默认为 5
- **stream**:  是否启用流式响应，默认为 False
- ****llm_kwargs**: 额外的关键字参数，将直接传递给 LLM 接口（如 temperature、top_p 等）

### 函数参数要求

#### 历史记录参数
函数必须包含名为 `history` 或 `chat_history` 的参数，格式要求：
```python
List[Dict[str, str]]
# 每个字典必须包含 'role' 和 'content' 键
# role 可以是: 'user', 'assistant', 'system'
```

#### 其他参数
除历史记录参数外的所有参数都会被格式化为用户消息内容。

### 返回值格式
装饰器修改后的函数返回 `Generator[Tuple[str, List[Dict[str, str]]], None, None]`：
- 第一个元素 `str`: 助手的响应内容（可能是部分内容，支持流式输出）
- 第二个元素 `List[Dict[str, str]]`: 更新后的对话历史记录（已过滤工具调用信息）

需要说明的是无论流式还是非流返回，都是同样的格式，非流返回中的 `str` 可能是完整的响应内容，而流式返回则是每个`delta`中的内容。
对于历史记录，我们只要关注生成器最后一个返回中的 `List[Dict[str, str]]`，它包含了更新好的对话历史记录。

## 被装饰的函数有要求：
- 函数必须包含 `history` 或 `chat_history` 参数，用于存储对话历史记录，且必须是第一个参数，且必须是 `List[Dict[str, str]]` 类型。
- 函数的必须有文档字符串（docstring），这将作为系统消息传递给 LLM，但是不能有函数体
- 函数可以接受其他参数，这些参数将被格式化为用户消息内容。

## 装饰器行为

### 参数处理流程

1. **参数绑定**: 将函数调用参数绑定到函数签名
2. **历史记录提取**: 识别并提取 `history` 或 `chat_history` 参数
3. **用户消息构建**: 将其他参数格式化为 `key: value` 形式的用户消息
4. **历史记录验证**: 验证历史记录格式，过滤无效项

### 消息构建流程

1. **系统消息**: 使用函数文档字符串作为系统提示
2. **工具信息**: 如果有工具，将工具描述添加到系统消息中
3. **历史消息**: 添加有效的历史记录消息
4. **当前消息**: 添加当前用户输入消息

### 响应处理流程

1. **LLM 调用**: 发送消息给 LLM，处理可能的工具调用
2. **流式生成**: 通过生成器逐步返回响应内容
3. **历史更新**: 将完整响应添加到历史记录中
4. **内容过滤**: 移除工具调用信息，只保留用户和助手消息

### 特殊处理机制

#### 历史记录过滤
- **输入过滤**: 忽略格式不正确的历史记录项
- **输出过滤**: 移除工具调用相关的消息，只保留 user、assistant、system 消息
- **工具调用透明**: 用户无需了解内部工具调用细节

#### 错误容错
- **参数缺失**: 如果没有历史记录参数，会发出警告但继续执行
- **格式错误**: 历史记录格式错误时自动忽略，不中断执行
- **工具错误**: 不支持的工具类型会被忽略并记录警告

## 示例

### 示例 1: 基本聊天功能

```python
from SimpleLLMFunc.llm_decorator import llm_chat
from SimpleLLMFunc.interface import OpenAICompatible
from typing import List, Dict, Generator, Tuple

# 初始化 LLM 接口
llm = OpenAICompatible(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo"
)

@llm_chat(llm_interface=llm)
def simple_chat(history: List[Dict[str, str]] = [], message: str) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
    """
    你是一个友好的助手，擅长回答各种问题。
    请保持对话自然流畅，回答要准确有用。
    """
    pass

# 使用示例
history = []
user_input = "你好，请介绍一下自己"

# 获取响应和更新的历史记录
for response_chunk, updated_history in simple_chat(user_input, history):
    if response_chunk:  # 非空响应
        print(response_chunk)
    else:  # 空响应表示对话结束
        history = updated_history
        break

print()  # 换行
print(f"历史记录: {history}")
```

### 示例 2: 多参数聊天

```python
@llm_chat(llm_interface=llm, temperature=0.7)
def multi_param_chat(
    history: List[Dict[str, str]] = [],
    question: str, 
    context: str, 
    language: str = "中文",
) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
    """
    你是一个专业的问答助手。根据提供的上下文信息回答用户问题。
    请确保回答准确、相关，并使用用户指定的语言。
    """
    pass

# 使用示例
context = "Python是一种高级编程语言，由Guido van Rossum在1989年发明。"
question = "Python是什么时候发明的？"

for response, history in multi_param_chat(
    question=question,
    context=context,
    language="中文",
    history=[]
):
    if response:
        print(response, end="")
    else:
        break
```

### 示例 3: 带工具的聊天助手

```python
from SimpleLLMFunc.tool import tool
import requests
from datetime import datetime

@tool
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_weather(city: str) -> str:
    """查询指定城市的天气信息"""
    # 这里是模拟的天气查询
    return f"{city}今天天气晴朗，温度22°C"

@tool
def calculate_math(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "计算出错，请检查表达式格式"

@llm_chat(
    llm_interface=llm,
    toolkit=[get_current_time, search_weather, calculate_math],
    temperature=0.3
)
def assistant_with_tools(
    history: List[Dict[str, str]] = [],
    message: str, 
) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
    """
    你是一个智能助手，可以帮助用户获取时间、查询天气和进行数学计算。
    当用户需要这些信息时，请主动使用相应的工具。
    保持友好和专业的对话风格。
    """
    pass

# 使用示例
def chat_session():
    history = []
    
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['quit', 'exit', '退出']:
            break
            
        print("助手: ", end="")
        
        for response, updated_history in assistant_with_tools(user_input, history):
            if response:
                print(response)
            else:
                history = updated_history
                break
        
        print()  # 换行

# 启动聊天
# chat_session()
```

### 示例 4: 角色扮演聊天

```python
@llm_chat(llm_interface=llm, temperature=0.8, top_p=0.9)
def role_play_chat(
    chat_history: List[Dict[str, str]] = [],
    user_message: str,
    character_name: str = "小助手",
    character_traits: str = "友好、幽默、博学",
) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
    """
    你需要扮演指定的角色与用户对话。
    根据角色名称和特征来调整你的说话风格和回应方式。
    保持角色的一致性，让对话更加生动有趣。
    """
    pass

# 使用示例
def role_play_session():
    history = []
    character = "孙悟空"
    traits = "勇敢、机智、有点顽皮，说话带有古代风格"
    
    print(f"开始与{character}对话...")
    
    while True:
        user_input = input(f"\n你: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print(f"{character}: ", end="")
        
        for response, updated_history in role_play_chat(
            user_message=user_input,
            character_name=character,
            character_traits=traits,
            chat_history=history
        ):
            if response:
                print(response)
            else:
                history = updated_history
                break
        
        print()

# role_play_session()
```

### 示例 5: 流式响应处理

```python
@llm_chat(
    llm_interface=GPT_4o_Interface,
    toolkit=[
        calc,
        get_current_time_and_date,
        file_operator,
        execute_command,
        interactive_terminal,
    ],
    stream=True,
    max_tool_calls=500,
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

        llm_response_flow = self.chat(self.memory, query)
        for response_str, history in llm_response_flow:
            self.memory = history
            yield response_str
```
---

通过这些示例可以看出，`llm_chat` 装饰器提供了强大而灵活的对话功能，支持多种使用场景，从简单的问答到复杂的多会话管理都能很好地处理。装饰器的流式响应特性使得用户能够获得实时的交互体验。