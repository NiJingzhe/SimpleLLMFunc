# llm_function 装饰器

## 装饰器作用

`llm_function` 装饰器是 SimpleLLMFunc 库的核心功能，它能够将普通 Python 函数的执行委托给大语言模型（LLM）。通过这个装饰器，开发者只需要定义函数签名（参数和返回类型）并在文档字符串中描述函数的执行策略，LLM 就会根据描述自动完成函数的实际执行。

### 主要功能特性
- **智能参数传递**: 自动将函数参数转换为 LLM 可理解的文本提示
- **类型安全**: 支持类型提示，自动将 LLM 响应转换为指定的返回类型
- **工具集成**: 支持为 LLM 提供工具，扩展其能力范围
- **灵活配置**: 支持自定义提示模板和 LLM 参数
- **错误处理**: 内置重试机制和详细的日志记录

## 装饰器用法

### 基本语法

```python
from SimpleLLMFunc.llm_decorator import llm_function

@llm_function(
    llm_interface=llm_interface,
    toolkit=None,
    max_tool_calls=5,
    system_prompt_template=None,
    user_prompt_template=None,
    **llm_kwargs
)
def your_function(param1: Type1, param2: Type2) -> ReturnType:
    """在这里描述函数的功能和执行策略"""
    pass
```

### 参数说明

- **llm_interface** (必需): LLM 接口实例，用于与大语言模型通信
- **toolkit** (可选): 工具列表，可以是 Tool 对象或被 @tool 装饰的函数
- **max_tool_calls** (可选): 最大工具调用次数，防止无限循环，默认为 5
- **system_prompt_template** (可选): 自定义系统提示模板
- **user_prompt_template** (可选): 自定义用户提示模板
- ****llm_kwargs**: 额外的关键字参数，将直接传递给 LLM 接口（如 temperature、top_p 等）

### 自定义提示模板

#### 系统提示模板占位符
- `{function_description}`: 函数文档字符串内容
- `{parameters_description}`: 函数参数及其类型的描述
- `{return_type_description}`: 返回值类型的描述

#### 用户提示模板占位符
- `{parameters}`: 格式化后的参数名称和值

## 装饰器行为

### 数据流程

1. **函数调用捕获**: 当用户调用被装饰的函数时，装饰器捕获所有实际参数
2. **类型信息提取**: 从函数签名中提取参数类型和返回类型信息
3. **提示构建**: 
   - 将函数文档字符串作为系统提示的核心
   - 将参数信息格式化为用户提示
   - 应用自定义模板（如果提供）
4. **LLM 调用**: 发送构建好的提示给 LLM
5. **工具处理**: 如果 LLM 需要使用工具，自动处理工具调用
6. **响应转换**: 将 LLM 的文本响应转换为指定的返回类型
7. **结果返回**: 返回转换后的结果给调用者

### 内置功能

#### 类型转换支持
- 基本类型：`str`, `int`, `float`, `bool`
- 容器类型：`List`, `Dict`, `Tuple`
- Pydantic 模型
- 自定义类型（通过 JSON 序列化）

#### 错误处理机制
- **空响应重试**: 当 LLM 返回空内容时自动重试
- **异常捕获**: 完整的异常处理和日志记录
- **类型转换错误**: 优雅处理类型转换失败的情况

#### 日志记录
- 详细的执行日志，包括参数、提示内容和响应
- 支持追踪 ID，便于调试复杂的调用链
- 分级日志（调试、信息、警告、错误）

## 示例

### 示例 1: 基本文本处理

```python
from SimpleLLMFunc.llm_decorator import llm_function
from SimpleLLMFunc.interface import OpenAICompatible

# 初始化 LLM 接口
llm = OpenAICompatible(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo"
)

@llm_function(llm_interface=llm)
def summarize_text(text: str, max_words: int = 100) -> str:
    """根据输入文本生成一个简洁的摘要，摘要不超过指定的词数。"""
    pass

# 使用函数
long_text = "这是一段很长的文本..."
summary = summarize_text(long_text, max_words=50)
print(summary)
```

### 示例 2: 结构化数据返回

```python
from typing import Dict, Any, List

@llm_function(llm_interface=llm)
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    分析文本的情感倾向，返回包含以下字段的字典：
    - sentiment: 情感标签（positive/negative/neutral）
    - confidence: 置信度（0-1之间的浮点数）
    - keywords: 关键词列表
    """
    pass

# 使用函数
result = analyze_sentiment("我今天心情很好，天气也很棒！")
print(result)
# 输出: {'sentiment': 'positive', 'confidence': 0.95, 'keywords': ['心情', '好', '天气', '棒']}
```

### 示例 3: 使用工具集

```python
from SimpleLLMFunc.tool import tool

@tool
def search_web(query: str) -> str:
    """在网络上搜索信息"""
    # 实现网络搜索逻辑
    return f"搜索结果: {query}"

@tool
def calculate(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

@llm_function(
    llm_interface=llm,
    toolkit=[search_web, calculate]
)
def research_and_calculate(topic: str, calculation: str) -> str:
    """
    根据主题搜索相关信息，并进行指定的计算，
    最后整合信息给出综合报告。
    """
    pass

# 使用函数
result = research_and_calculate(
    topic="Python编程语言", 
    calculation="2023 - 1991"
)
print(result)
```

### 示例 4: 自定义提示模板

```python
# 自定义系统提示模板
custom_system_template = """
你是一名专业的数据分析师，请根据以下信息执行任务:

参数信息:
{parameters_description}

返回类型: {return_type_description}

任务描述:
{function_description}

请确保分析结果准确、客观，并提供数据支持。
"""

# 自定义用户提示模板
custom_user_template = """
请分析以下数据:
{parameters}

请直接提供分析结果，格式要求为指定的返回类型。
"""

@llm_function(
    llm_interface=llm,
    system_prompt_template=custom_system_template,
    user_prompt_template=custom_user_template,
    temperature=0.7,  # 通过 llm_kwargs 传递模型参数
    top_p=0.9
)
def analyze_data(data: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
    """
    对给定的数据集进行指定类型的分析，
    支持的分析类型包括：趋势分析、异常检测、统计摘要等。
    """
    pass

# 使用函数
sample_data = [
    {"date": "2023-01-01", "value": 100},
    {"date": "2023-01-02", "value": 120},
    {"date": "2023-01-03", "value": 95}
]

analysis_result = analyze_data(sample_data, "趋势分析")
print(analysis_result)
```

### 示例 5: Pydantic 模型返回

```python
from pydantic import BaseModel
from typing import List

class TaskResult(BaseModel):
    success: bool
    message: str
    tasks: List[str]
    estimated_time: int

@llm_function(llm_interface=llm)
def create_project_plan(project_description: str, deadline_days: int) -> TaskResult:
    """
    根据项目描述和截止时间，制定详细的项目计划。
    返回包含任务列表、预估时间和执行建议的结构化结果。
    """
    pass

# 使用函数
plan = create_project_plan(
    project_description="开发一个简单的待办事项应用",
    deadline_days=30
)

print(f"计划制定成功: {plan.success}")
print(f"建议: {plan.message}")
print(f"任务列表: {plan.tasks}")
print(f"预估时间: {plan.estimated_time}天")
```

---

通过这些示例可以看出，`llm_function` 装饰器提供了一种简洁而强大的方式来利用 LLM 的能力，同时保持了 Python 代码的类型安全性和可读性。
