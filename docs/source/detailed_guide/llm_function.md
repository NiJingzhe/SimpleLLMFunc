# LLM 函数装饰器

本文档介绍 SimpleLLMFunc 库中的核心装饰器 `llm_function`。该装饰器能够将异步 Python 函数的执行委托给大语言模型（LLM），开发者只需要定义函数签名（参数和返回类型）并在文档字符串中描述函数的执行策略，LLM 就会根据描述自动完成函数的实际执行。

## llm_function 装饰器

### 装饰器作用

`llm_function` 装饰器是 SimpleLLMFunc 库的核心功能之一，它提供了原生异步的 LLM 函数调用能力。通过这个装饰器，开发者可以轻松地将异步函数转换为由 LLM 执行的智能函数。

### 主要功能特性
- **智能参数传递**: 自动将函数参数转换为 LLM 可理解的文本提示
- **类型安全**: 支持类型提示，自动将 LLM 响应转换为指定的返回类型
- **工具集成**: 支持为 LLM 提供工具，扩展其能力范围
- **灵活配置**: 支持自定义提示模板和 LLM 参数
- **错误处理**: 内置重试机制和详细的日志记录

### 0.4.0 版本要点
- **步骤化执行链**：`llm_decorator/steps` 拆分了 Prompt 构建、签名解析、ReAct 执行和响应解析，便于扩展与调试。
- **类型解析增强**：`base/type_resolve` 新增描述、示例和多模态解析子模块，对嵌套类型、多模态联合类型的推断更稳健。
- **多模态工具链优化**：`base/tool_call/execution` 改进工具调用入参与返回的多模态处理，确保图片/文本组合的传递和校验更可靠。

## 重要说明

> ⚠️ `llm_function` 只能装饰 `async def` 定义的异步函数，返回的也是可 `await` 的协程；请在异步上下文中调用，或在脚本入口使用 `asyncio.run()`。


## 装饰器用法

### 基本语法

```python
from SimpleLLMFunc import llm_function

@llm_function(
    llm_interface=llm_interface,
    toolkit=None,
    max_tool_calls=5,
    system_prompt_template=None,
    user_prompt_template=None,
    **llm_kwargs
)
async def your_function(param1: Type1, param2: Type2) -> ReturnType:
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

### 动态模板参数

SimpleLLMFunc 支持在函数调用时通过 `_template_params` 参数动态设置 DocString 模板参数。这个功能让同一个函数可以适应不同的使用场景，大大提高了代码的复用性。

#### 使用方法

1. **在 DocString 中使用占位符**：
```python
@llm_function(llm_interface=llm)
async def flexible_function(text: str) -> str:
    """作为{role}，请{action}以下文本，输出风格为{style}。"""
    pass
```

2. **调用时传入模板参数**：
```python
import asyncio


async def main():
    # 编辑角色
    result1 = await flexible_function(
        text,
        _template_params={
            'role': '专业编辑',
            'action': '润色',
            'style': '学术'
        }
    )

    # 翻译角色
    result2 = await flexible_function(
        text,
        _template_params={
            'role': '翻译专家',
            'action': '翻译',
            'style': '商务'
        }
    )

    return result1, result2


result1, result2 = asyncio.run(main())
```

#### 核心特性

- **动态角色切换**：同一个函数可以扮演不同的角色
- **灵活任务适配**：根据调用上下文调整任务类型
- **透明处理**：`_template_params` 不会传递给 LLM，仅用于模板处理
- **错误处理**：当模板参数不完整时，系统会发出警告并使用原始 DocString

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
import asyncio
from SimpleLLMFunc import llm_function, OpenAICompatible

# 初始化 LLM 接口（推荐方式：从配置文件加载）
models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openai"]["gpt-3.5-turbo"]

@llm_function(llm_interface=llm)
async def summarize_text(text: str, max_words: int = 100) -> str:
    """根据输入文本生成一个简洁的摘要，摘要不超过指定的词数。"""
    pass

# 使用函数
long_text = "这是一段很长的文本..."


async def main():
    summary = await summarize_text(long_text, max_words=50)
    print(summary)


asyncio.run(main())
```

### 示例 2: 结构化数据返回

```python
import asyncio
from typing import Dict, Any, List

@llm_function(llm_interface=llm)
async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    分析文本的情感倾向，返回包含以下字段的字典：
    - sentiment: 情感标签（positive/negative/neutral）
    - confidence: 置信度（0-1之间的浮点数）
    - keywords: 关键词列表
    """
    pass

# 使用函数


async def main():
    result = await analyze_sentiment("我今天心情很好，天气也很棒！")
    print(result)


asyncio.run(main())
# 输出: {'sentiment': 'positive', 'confidence': 0.95, 'keywords': ['心情', '好', '天气', '棒']}
```

### 示例 3: 使用工具集

```python
import asyncio
from SimpleLLMFunc import tool, llm_function, OpenAICompatible

# 加载模型接口
models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openai"]["gpt-3.5-turbo"]

@tool
async def search_web(query: str) -> str:
    """在网络上搜索信息"""
    # 实现网络搜索逻辑
    return f"搜索结果: {query}"

@tool
async def calculate(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

@llm_function(
    llm_interface=llm,
    toolkit=[search_web, calculate]
)
async def research_and_calculate(topic: str, calculation: str) -> str:
    """
    根据主题搜索相关信息，并进行指定的计算，
    最后整合信息给出综合报告。
    """
    pass

# 使用函数
async def main():
    result = await research_and_calculate(
        topic="Python编程语言",
        calculation="2023 - 1991"
    )
    print(result)


asyncio.run(main())
```

### 示例 4: 自定义提示模板

```python
import asyncio

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
async def analyze_data(data: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
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

async def main():
    analysis_result = await analyze_data(sample_data, "趋势分析")
    print(analysis_result)


asyncio.run(main())
```

### 示例 5: Pydantic 模型返回

```python
import asyncio
from pydantic import BaseModel
from typing import List

class TaskResult(BaseModel):
    success: bool
    message: str
    tasks: List[str]
    estimated_time: int

@llm_function(llm_interface=llm)
async def create_project_plan(project_description: str, deadline_days: int) -> TaskResult:
    """
    根据项目描述和截止时间，制定详细的项目计划。
    返回包含任务列表、预估时间和执行建议的结构化结果。
    """
    pass

# 使用函数


async def main():
    plan = await create_project_plan(
        project_description="开发一个简单的待办事项应用",
        deadline_days=30
    )

    print(f"计划制定成功: {plan.success}")
    print(f"建议: {plan.message}")
    print(f"任务列表: {plan.tasks}")
    print(f"预估时间: {plan.estimated_time}天")


asyncio.run(main())
```

### 示例 6: 动态模板参数

这个示例展示了如何使用动态模板参数让同一个函数适应不同的使用场景：

```python
import asyncio


@llm_function(llm_interface=llm)
async def analyze_code(code: str) -> str:
    """以{style}的方式分析{language}代码，重点关注{focus}。"""
    pass


@llm_function(llm_interface=llm)
async def process_text(text: str) -> str:
    """作为{role}，请{action}以下文本，输出风格为{style}。"""
    pass


# 使用示例
python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""


async def main():
    # 不同的分析方式
    performance_analysis = await analyze_code(
        python_code,
        _template_params={
            'style': '详细',
            'language': 'Python',
            'focus': '性能优化'
        }
    )

    code_style_analysis = await analyze_code(
        python_code,
        _template_params={
            'style': '简洁',
            'language': 'Python',
            'focus': '代码规范'
        }
    )

    # 不同的文本处理角色
    sample_text = "人工智能技术正在快速发展，对各行各业产生深远影响。"

    edited_text = await process_text(
        sample_text,
        _template_params={
            'role': '专业编辑',
            'action': '润色',
            'style': '学术'
        }
    )

    translated_text = await process_text(
        sample_text,
        _template_params={
            'role': '翻译专家',
            'action': '翻译成英文',
            'style': '商务'
        }
    )

    print("性能分析结果:", performance_analysis)
    print("代码规范分析:", code_style_analysis)
    print("编辑润色结果:", edited_text)
    print("翻译结果:", translated_text)


asyncio.run(main())
```

这个示例展示了动态模板参数的强大功能：
- **一个函数，多种场景**：`analyze_code` 可以用于性能分析、规范检查等不同目的
- **动态角色切换**：`process_text` 可以扮演编辑、翻译等不同角色
- **灵活任务适配**：根据调用时的参数动态调整任务类型和输出风格

---

通过这些示例可以看出，`llm_function` 装饰器提供了一种简洁而强大的方式来利用 LLM 的能力，同时保持了 Python 代码的类型安全性和可读性。

## 异步使用示例

`llm_function` 自身即为原生异步实现。以下示例演示如何在不同场景下使用它：

### 示例 1: 基本异步用法

```python
import asyncio
from SimpleLLMFunc import llm_function, OpenAICompatible

# 从配置文件加载
models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openai"]["gpt-3.5-turbo"]


@llm_function(llm_interface=llm)
async def summarize_text_async(text: str, max_words: int = 100) -> str:
    """根据输入文本生成一个简洁的摘要，摘要不超过指定的词数。"""
    pass


async def main():
    long_text = "这是一段很长的文本..."
    summary = await summarize_text_async(long_text, max_words=50)
    print(summary)


asyncio.run(main())
```

### 示例 2: 并发处理多个请求

```python
import asyncio


@llm_function(llm_interface=llm)
async def translate_text_async(text: str, target_language: str = "English") -> str:
    """将输入文本翻译成指定语言。"""
    pass


@llm_function(llm_interface=llm)
async def analyze_sentiment_async(text: str) -> str:
    """分析文本的情感倾向。"""
    pass


async def process_texts_concurrently():
    texts = [
        "今天天气很好",
        "我感到很沮丧",
        "这个产品质量很棒"
    ]

    translation_tasks = [translate_text_async(text) for text in texts]
    sentiment_tasks = [analyze_sentiment_async(text) for text in texts]

    translations, sentiments = await asyncio.gather(
        asyncio.gather(*translation_tasks),
        asyncio.gather(*sentiment_tasks)
    )

    for i, (text, translation, sentiment) in enumerate(zip(texts, translations, sentiments)):
        print(f"文本 {i + 1}: {text}")
        print(f"翻译: {translation}")
        print(f"情感: {sentiment}")


asyncio.run(process_texts_concurrently())
```

### 示例 3: 与其他异步操作配合

```python
import aiohttp
import asyncio


@llm_function(llm_interface=llm)
async def process_content_async(content: str) -> str:
    """处理从网络获取的内容。"""
    pass


async def fetch_and_process_url(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        content = await response.text()

    processed = await process_content_async(content[:1000])
    return processed


async def process_multiple_urls():
    urls = [
        "https://example1.com",
        "https://example2.com",
        "https://example3.com"
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_process_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                print(f"处理 {url} 时出错: {result}")
            else:
                print(f"{url}: {result}")


asyncio.run(process_multiple_urls())
```

这些示例展示了如何使用 `llm_function` 在异步环境中构建高并发的 LLM 调用逻辑。

## 最佳实践

### 1. 错误处理
```python
async def robust_llm_call():
    try:
        result = await your_llm_function("input")
        return result
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return "默认值"
```

### 2. 超时控制
```python
async def llm_call_with_timeout():
    try:
        result = await asyncio.wait_for(
            your_llm_function("input"),
            timeout=30.0  # 30秒超时
        )
        return result
    except asyncio.TimeoutError:
        print("LLM 调用超时")
        return "超时默认值"
```

---

通过这些示例可以看出，`llm_function` 装饰器在异步场景下同样能够提供高性能的 LLM 调用能力，并保持了良好的易用性与功能完整性。
