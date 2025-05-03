# SimpleLLMFunc 使用教程

## 0.1.4版本新增功能亮点

SimpleLLMFunc近期新增了两个重要功能：

1. **OpenAICompatible通用接口** - 简化了不同LLM供应商的接入，无需为每个供应商创建专门的实现
2. **装饰器自定义参数** 

优化内容：
1. 优化了LLM Chat中对于历史记录的管理策略。针对一个包含多伦ToolCall的response，我们会将每一次伴随tool call的response content进行记录，最终会将多轮tool call的response content和最终的response content进行拼接，形成最终的response content。


## 框架设计理念

SimpleLLMFunc 是一个轻量级的LLM调用和工具集成框架，其核心设计理念是：

1. **以函数为中心** - 将LLM能力封装为普通Python函数，使开发者能以最熟悉的编程方式集成AI能力
2. **DocString即Prompt** - 将提示词设计自然地融入函数文档，一举两得，既提高代码可读性，又明确了LLM的执行期望
3. **类型安全** - 通过类型注解和Pydantic模型确保输入输出的结构化和可预测性
4. **最大自由度** - 不限制开发者使用复杂流程（如有向环图），保持Python原生的灵活性
5. **恰到好处的抽象** - 屏蔽LLM调用细节，但不过度抽象，保持框架简单易懂

## 装饰器设计核心

SimpleLLMFunc 提供了两种核心装饰器：

1. **@llm_function** - 用于创建"无状态"LLM函数，每次调用相互独立
2. **@llm_chat** - 用于创建"有状态"对话函数，自动维护对话历史

这两种装饰器的设计使开发者能够：

- 专注于"函数应该做什么"而非"如何调用LLM API"
- 通过类型注解自动处理输入验证和输出转换
- 将函数文档字符串自动转换为系统提示，让原本即应该是对功能和流程的表述出现在最像函数实现的位置上
- 无缝集成工具调用能力


## 1. 装饰器详细使用指南

### @llm_function 装饰器

`@llm_function` 装饰器将普通Python函数转换为由LLM驱动的函数。它通过以下步骤工作：

1. 捕获函数签名、类型标注和文档字符串
2. 将这些信息转换为系统提示和用户提示
3. 调用LLM进行推理
4. 处理工具调用（如果配置了工具）
5. 将LLM响应转换为指定的返回类型

#### 基本语法

```python
from SimpleLLMFunc import llm_function
from SimpleLLMFunc.interface import OpenAICompatible

@llm_function(
    llm_interface=your_llm_interface,    # 必需：LLM接口
    toolkit=None,                        # 可选：工具列表
    max_tool_calls=5,                    # 可选：最大工具调用次数
    system_prompt_template=None,         # 可选：自定义系统提示模板
    user_prompt_template=None,           # 可选：自定义用户提示模板
    **llm_kwargs                         # 可选：传递给LLM的额外参数
)
def your_function(param1: type1, param2: type2 = default_value) -> return_type:
    """
    函数详细描述，这将作为系统提示的一部分发送给LLM。
    应当清晰地描述函数的功能、预期行为和任何特殊要求。
    
    Args:
        param1: 第一个参数的描述
        param2: 第二个参数的描述，有默认值
        
    Returns:
        返回值的描述
    """
    pass  # 函数体可以为空，实际执行由LLM完成
```

#### DocString的重要性

在`@llm_function`中，函数的文档字符串（DocString）扮演着至关重要的角色：

1. 它被用作系统提示的主要内容，指导LLM如何执行函数
2. `Args`部分的参数描述帮助LLM理解每个参数的含义和用途
3. `Returns`部分描述预期的输出格式和内容

编写优质的DocString是确保LLM正确理解任务的关键。

#### 返回类型处理

装饰器会自动将LLM的响应转换为函数签名中指定的返回类型：

- **基本类型**（`str`, `int`, `float`, `bool`）：直接转换
- **字典**（`Dict[str, Any]`）：解析JSON
- **Pydantic模型**：使用`model_validate_json`解析并验证

使用Pydantic模型作为返回类型是推荐的做法，它提供了自动验证和清晰的结构定义。

#### 自定义提示模板

您可以通过`system_prompt_template`和`user_prompt_template`参数自定义提示模板：

```python
custom_system_template = """
你是一名专业的{function_name}专家，请根据以下要求执行任务:

功能描述:
{function_description}

参数信息:
{parameters_description}

返回要求:
你需要返回符合{return_type_description}格式的结果
"""

@llm_function(
    llm_interface=llm,
    system_prompt_template=custom_system_template
)
def analyze_text(text: str) -> Dict[str, Any]:
    """分析文本内容，提取关键信息。"""
    pass
```

#### 工具集成

`@llm_function`支持工具集成，使LLM能够调用外部函数：

```python
from SimpleLLMFunc.tool import tool

@tool(name="calculate", description="执行数学计算")
def calculate(expression: str) -> float:
    """
    计算数学表达式的结果
    
    Args:
        expression: 要计算的数学表达式
        
    Returns:
        计算结果
    """
    return eval(expression)  # 注意：实际应用中请安全处理

@llm_function(
    llm_interface=llm,
    toolkit=[calculate]  # 传递工具列表
)
def solve_math_problem(problem: str) -> str:
    """
    解决数学问题，可以使用calculate工具进行计算
    
    Args:
        problem: 数学问题描述
        
    Returns:
        问题的解答过程和结果
    """
    pass
```

#### 自定义LLM参数

通过传递额外的关键字参数，您可以为每个函数自定义LLM参数：

```python
@llm_function(
    llm_interface=llm,
    temperature=0.7,        # 控制创造性
    max_tokens=500,         # 限制输出长度
    presence_penalty=0.6    # 增加话题多样性
)
def generate_story(theme: str, length: str = "medium") -> str:
    """生成一个创意故事"""
    pass
```

### @llm_chat 装饰器

`@llm_chat`装饰器用于创建对话函数，它维护对话历史并生成响应：

#### 基本语法

```python
from SimpleLLMFunc import llm_chat
from typing import List, Dict

@llm_chat(
    llm_interface=your_llm_interface,    # 必需：LLM接口
    toolkit=None,                        # 可选：工具列表
    max_tool_calls=5,                    # 可选：最大工具调用次数
    **llm_kwargs                         # 可选：传递给LLM的额外参数
)
def chat_function(
    message: str,                        # 当前用户消息
    history: List[Dict[str, str]] = []   # 对话历史，必须包含此参数
) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
    """
    这里的文档字符串会作为系统提示发送给LLM。
    描述助手的角色、行为和能力。
    Tools描述不需要在这里撰写，会根据传入的tools的name 和description自己生成并拼接在最后。
    """
    pass
```

#### 历史记录格式

`history`/`chat_history`参数必须是包含消息的列表，每条消息是带有`role`和`content`字段的字典：

```python
history = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
]
```

#### 使用对话函数

由于`@llm_chat`返回一个生成器，您需要使用迭代来获取结果：

使用生成器返回是因为多轮Tool Call Reponse会自动进行ReAct模式的处理，所以我们希望每一次Tool Call都能及时被展示，而不是等整个ReAct过程都结束了再显示所有内容。

```python
# 或者使用循环处理多轮对话
for response, updated_history in chat_function(history,"分析这段代码的性能问题"):
    print(response)
```

#### 工具集成

与`@llm_function`类似，`@llm_chat`也支持工具集成：

```python
@tool(name="search", description="搜索信息")
def search(query: str) -> List[Dict[str, str]]:
    """搜索相关信息"""
    # 模拟搜索结果
    return [{"title": "结果1", "snippet": "内容摘要"}]

@llm_chat(
    llm_interface=llm,
    toolkit=[search],
    round_to_emphasize=3  # 3轮对话后提醒LLM使用工具
)
def research_assistant(message: str, history: List[Dict[str, str]] = []):
    """你是一个研究助手，能够使用搜索工具查找信息来回答问题。"""
    pass
```

## 2. OpenAICompatible通用接口

### 设计理念

OpenAICompatible接口提供了一个通用的封装，可以连接任何兼容OpenAI API格式的大语言模型服务。这使您可以：

- 轻松切换不同的LLM提供商，只需更改base_url和模型名称
- 无需为每个供应商编写特定的接口实现
- 保持一致的API调用方式

### 基本用法

```python
from SimpleLLMFunc.interface import OpenAICompatible, APIKeyPool

# 创建API密钥池
api_key_pool = APIKeyPool(
    api_keys=["your-api-key1", "your-api-key2"],
    provider_id="custom-provider"
)

# 创建通用LLM接口
custom_llm = OpenAICompatible(
    api_key_pool=api_key_pool,
    model_name="model-name",
    base_url="https://api.yourprovider.com/v1"
)

# 在LLM函数装饰器中使用
from SimpleLLMFunc import llm_function

@llm_function(llm_interface=custom_llm)
def generate_summary(text: str) -> str:
    """生成文本摘要"""
    pass
```

### 支持的主要参数

OpenAICompatible接口支持以下主要参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| api_key_pool | APIKeyPool | API密钥池，用于管理和负载均衡API密钥 |
| model_name | str | 要使用的模型名称 |
| base_url | str | API基础URL，如"https://api.openai.com/v1" |
| max_retries | int | 请求失败时的最大重试次数 |
| retry_delay | float | 重试之间的延迟时间（秒） |
| allowed_models | List[str] | 可选的允许模型列表，建议用法是在配置文件中配置模型名称列表，用于在运行时验证所有Interface创建时传入的 model name 是否有效。|

### 对接不同供应商示例

#### OpenAI API

```python
openai_llm = OpenAICompatible(
    api_key_pool=APIKeyPool(["sk-..."], "openai"),
    model_name="gpt-4-turbo",
    base_url="https://api.openai.com/v1"
)
```

#### Azure OpenAI

```python
azure_llm = OpenAICompatible(
    api_key_pool=APIKeyPool(["your-azure-key"], "azure"),
    model_name="gpt-4",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment-name"
)
```

#### 智谱AI

```python
zhipu_llm = OpenAICompatible(
    api_key_pool=APIKeyPool(["your-zhipu-key"], "zhipu"),
    model_name="glm-4",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
```

## 3. 完整示例：结合使用装饰器和OpenAICompatible

下面是一个结合了装饰器和OpenAICompatible接口的完整示例：

```python
from typing import Dict, List
from pydantic import BaseModel, Field
from SimpleLLMFunc import llm_function, llm_chat
from SimpleLLMFunc.interface import OpenAICompatible, APIKeyPool
from SimpleLLMFunc.tool import tool

# 创建API密钥池
api_key_pool = APIKeyPool(
    api_keys=["your-api-key"],
    provider_id="openai"
)

# 创建OpenAICompatible实例
llm = OpenAICompatible(
    api_key_pool=api_key_pool,
    model_name="gpt-4-turbo",
    base_url="https://api.openai.com/v1"
)

# 定义工具
@tool(name="calculate", description="执行数学计算")
def calculate(expression: str) -> float:
    """
    计算数学表达式的结果
    
    Args:
        expression: 要计算的数学表达式，例如"2 + 2"或"sin(30) * 2"
        
    Returns:
        计算结果
    """
    return eval(expression)  # 注意：实际应用中应当进行安全处理

# 定义输出模型
class AnalysisResult(BaseModel):
    main_points: List[str] = Field(..., description="文本中的主要观点")
    sentiment: str = Field(..., description="情感倾向：positive, negative, neutral")
    keywords: List[str] = Field(..., description="关键词列表")
    summary: str = Field(..., description="简短摘要")

# 带工具和自定义参数的LLM函数
@llm_function(
    llm_interface=llm,
    tools=[calculate],
    temperature=0.3,  # 较低的temperature，提高一致性
    max_tokens=1500   # 控制输出长度
)
def analyze_text(text: str) -> AnalysisResult:
    """
    分析文本内容，提取主要观点、情感、关键词并生成摘要
    
    Args:
        text: 要分析的文本内容
        
    Returns:
        包含分析结果的结构化数据
    """
    pass

# 聊天函数，带自定义参数
@llm_chat(
    llm_interface=llm,
    temperature=0.8,
    presence_penalty=0.7,
    frequency_penalty=0.3
)
def creative_assistant(message: str, history: List[Dict[str, str]] = []):
    """一个有创意的助手，能提供富有想象力和创造性的回复，善于讲故事和提供创新想法。"""
    pass

# 主函数
def main():
    # 测试文本分析
    sample_text = """
    人工智能技术在2023年取得了显著进展。大型语言模型如GPT-4展示了惊人的能力，
    不仅能生成高质量文本，还能解决复杂问题和编写代码。然而，这些进步也带来了
    关于AI安全和伦理的担忧。研究人员警告说，没有足够的监管可能导致风险。
    尽管如此，AI技术在医疗、教育和气候研究等领域的应用前景依然广阔。
    """
    
    try:
        print("=== 文本分析结果 ===")
        result = analyze_text(sample_text)
        print(f"主要观点: {result.main_points}")
        print(f"情感倾向: {result.sentiment}")
        print(f"关键词: {result.keywords}")
        print(f"摘要: {result.summary}")
    except Exception as e:
        print(f"分析失败: {e}")
    
    # 测试创意助手
    try:
        print("\n=== 创意助手对话 ===")
        message = "请为我构思一个关于太空探索的短篇科幻故事开头"
        history = []
        
        for response, history in creative_assistant(message, history):
            print(f"助手: {response}")
    except Exception as e:
        print(f"对话失败: {e}")

if __name__ == "__main__":
    main()
```

## 4. 最佳实践与建议

### 装饰器最佳实践

1. **高质量DocString**：详细描述函数功能、参数和返回值，这直接影响LLM的输出质量
2. **使用Pydantic模型**：为复杂输出使用Pydantic模型，提供结构验证和自动文档
3. **分离功能**：每个LLM函数应专注于单一明确的任务，避免过于复杂的功能组合
4. **工具设计**：为LLM提供清晰、原子化的工具，帮助它完成复杂任务的子步骤
5. **错误处理**：考虑LLM可能失败的情况，实现适当的错误处理和重试机制


### 调试技巧

- 利用SimpleLLMFunc的日志系统跟踪每次调用的参数和响应
- 在开发阶段使用较小的max_tokens值以加快响应速度

## 5. 总结

SimpleLLMFunc的装饰器设计提供了一种优雅而强大的方式，将LLM能力无缝集成到Python函数中：

1. **简单直观** - 通过装饰普通函数将其转换为LLM驱动的函数
2. **保持Python原生风格** - 不引入新的DSL或复杂抽象
3. **类型安全** - 利用Python类型注解确保输入输出的正确性
4. **工具集成** - 使LLM能够使用自定义工具扩展其能力
5. **灵活配置** - 通过自定义参数微调每个函数的LLM行为

结合OpenAICompatible接口，SimpleLLMFunc为开发者提供了一个灵活、强大且简洁的框架，让AI应用开发变得更加简单和高效。