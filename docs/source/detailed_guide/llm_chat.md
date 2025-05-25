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
def simple_chat(message: str, history: List[Dict[str, str]] = []) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
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
        print(response_chunk, end="", flush=True)
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
    question: str, 
    context: str, 
    language: str = "中文",
    history: List[Dict[str, str]] = []
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
    message: str, 
    history: List[Dict[str, str]] = []
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
                print(response, end="", flush=True)
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
    user_message: str,
    character_name: str = "小助手",
    character_traits: str = "友好、幽默、博学",
    chat_history: List[Dict[str, str]] = []
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
                print(response, end="", flush=True)
            else:
                history = updated_history
                break
        
        print()

# role_play_session()
```

### 示例 5: 流式响应处理

```python
@llm_chat(llm_interface=llm)
def streaming_chat(
    message: str, 
    history: List[Dict[str, str]] = []
) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
    """
    你是一个能提供详细解释的助手。
    当回答复杂问题时，请给出充分的解释和例子。
    """
    pass

def handle_streaming_response(message: str, history: List[Dict[str, str]]):
    """处理流式响应的示例函数"""
    print("助手正在回复: ", end="")
    
    complete_response = ""
    final_history = []
    
    for response_chunk, updated_history in streaming_chat(message, history):
        if response_chunk:
            # 实时显示响应片段
            print(response_chunk, end="", flush=True)
            complete_response += response_chunk
        else:
            # 响应完成，获取最终历史记录
            final_history = updated_history
            break
    
    print()  # 换行
    
    return complete_response, final_history

# 使用示例
history = []
response, history = handle_streaming_response(
    "请详细解释什么是机器学习", 
    history
)

print(f"\n完整回复: {response}")
print(f"对话轮数: {len([msg for msg in history if msg['role'] == 'user'])}")
```

### 示例 6: 自定义历史记录管理

```python
class ChatManager:
    """聊天管理器，提供高级历史记录管理功能"""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.sessions = {}  # 存储多个会话的历史记录
    
    @llm_chat(llm_interface=llm, max_tool_calls=3)
    def managed_chat(
        self, 
        message: str,
        session_id: str = "default",
        history: List[Dict[str, str]] = []
    ) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
        """
        你是一个智能助手，能够管理多个独立的对话会话。
        每个会话都有独立的上下文和历史记录。
        请根据对话历史提供连贯和相关的回复。
        """
        pass
    
    def chat(self, message: str, session_id: str = "default") -> str:
        """发送消息并获取回复"""
        # 获取会话历史记录
        history = self.sessions.get(session_id, [])
        
        complete_response = ""
        
        # 调用装饰的聊天函数
        for response_chunk, updated_history in self.managed_chat(
            message=message,
            session_id=session_id,
            history=history
        ):
            if response_chunk:
                complete_response += response_chunk
            else:
                # 更新会话历史记录
                self.sessions[session_id] = updated_history
                break
        
        return complete_response
    
    def get_session_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """获取指定会话的历史记录"""
        return self.sessions.get(session_id, [])
    
    def clear_session(self, session_id: str = "default"):
        """清除指定会话的历史记录"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# 使用示例
chat_manager = ChatManager(llm)

# 会话1
response1 = chat_manager.chat("你好，我想了解Python编程", "python_session")
print(f"Python会话: {response1}")

# 会话2  
response2 = chat_manager.chat("推荐一些好看的电影", "movie_session")
print(f"电影会话: {response2}")

# 继续Python会话
response3 = chat_manager.chat("请给我一个简单的例子", "python_session")
print(f"Python会话继续: {response3}")

# 查看历史记录
python_history = chat_manager.get_session_history("python_session")
print(f"Python会话历史: {len(python_history)} 条消息")
```

---

通过这些示例可以看出，`llm_chat` 装饰器提供了强大而灵活的对话功能，支持多种使用场景，从简单的问答到复杂的多会话管理都能很好地处理。装饰器的流式响应特性使得用户能够获得实时的交互体验。