# LLM 函数装饰器

本文档介绍 SimpleLLMFunc 库中的两个核心装饰器：`llm_function` 和 `async_llm_function`。这两个装饰器能够将普通 Python 函数的执行委托给大语言模型（LLM），开发者只需要定义函数签名（参数和返回类型）并在文档字符串中描述函数的执行策略，LLM 就会根据描述自动完成函数的实际执行。

## llm_function 装饰器

### 装饰器作用

`llm_function` 装饰器是 SimpleLLMFunc 库的核心功能之一，它提供了同步的 LLM 函数调用能力。通过这个装饰器，开发者可以轻松地将普通函数转换为由 LLM 执行的智能函数。

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

# 初始化 LLM 接口（推荐方式：从配置文件加载）
llm = OpenAICompatible.load_from_json_file("provider.json")["provider_name"]["model_name"]

# 或者直接创建接口
# from SimpleLLMFunc.interface import APIKeyPool
# key_pool = APIKeyPool(
#     api_keys=["your-api-key-1", "your-api-key-2"],
#     provider_id="openai_gpt-3.5-turbo"
# )
# llm = OpenAICompatible(
#     api_key_pool=key_pool,
#     model_name="gpt-3.5-turbo",
#     base_url="https://api.openai.com/v1"
# )

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

# async_llm_function 装饰器

## 装饰器作用

`async_llm_function` 装饰器是 `llm_function` 的异步版本，它同样能够将普通 Python 函数的执行委托给大语言模型（LLM），但通过线程池实现异步执行，避免在 LLM 调用期间阻塞事件循环。这使得它特别适合需要并发处理多个 LLM 请求或与其他异步操作配合使用的场景。

### 主要功能特性
- **异步执行**: LLM 调用在线程池中执行，不会阻塞主事件循环
- **并发支持**: 支持并发调用多个 LLM 函数，提高处理效率
- **线程池管理**: 支持自定义线程池执行器，提供更好的资源控制
- **完整兼容**: 与同步版本功能完全相同，包括类型安全、工具集成等
- **异步操作集成**: 可以与其他异步操作配合使用（如 asyncio.gather）

## 装饰器用法

### 基本语法

```python
from SimpleLLMFunc.llm_decorator import async_llm_function

@async_llm_function(
    llm_interface=llm_interface,
    toolkit=None,
    max_tool_calls=5,
    system_prompt_template=None,
    user_prompt_template=None,
    executor=None,
    **llm_kwargs
)
async def your_async_function(param1: Type1, param2: Type2) -> ReturnType:
    """在这里描述函数的功能和执行策略"""
    pass
```

### 参数说明

- **llm_interface** (必需): LLM 接口实例，用于与大语言模型通信
- **toolkit** (可选): 工具列表，可以是 Tool 对象或被 @tool 装饰的函数
- **max_tool_calls** (可选): 最大工具调用次数，防止无限循环，默认为 5
- **system_prompt_template** (可选): 自定义系统提示模板
- **user_prompt_template** (可选): 自定义用户提示模板
- **executor** (可选): 线程池执行器，用于执行 LLM 调用。如果为 None，会为每次调用创建临时线程池
- ****llm_kwargs**: 额外的关键字参数，将直接传递给 LLM 接口（如 temperature、top_p 等）

### 线程池管理

#### 默认行为
如果不提供 `executor` 参数，每次调用会创建临时的线程池，这对于偶尔的异步调用是合适的。

#### 共享线程池（推荐）
对于频繁的异步调用，建议为多个函数共享同一个线程池执行器以提高性能：

```python
from concurrent.futures import ThreadPoolExecutor

# 创建共享的线程池
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="LLM-Async-")

@async_llm_function(llm_interface=llm, executor=executor)
async def function1(text: str) -> str:
    """功能描述1"""
    pass

@async_llm_function(llm_interface=llm, executor=executor)
async def function2(text: str) -> str:
    """功能描述2"""
    pass
```

## 装饰器行为

### 数据流程

异步版本的数据流程与同步版本基本相同，主要区别在于执行方式：

1. **函数调用捕获**: 当用户调用被装饰的异步函数时，装饰器捕获所有实际参数
2. **类型信息提取**: 从函数签名中提取参数类型和返回类型信息
3. **提示构建**: 
   - 将函数文档字符串作为系统提示的核心
   - 将参数信息格式化为用户提示
   - 应用自定义模板（如果提供）
4. **异步 LLM 调用**: 在线程池中发送构建好的提示给 LLM
5. **工具处理**: 如果 LLM 需要使用工具，在线程池中自动处理工具调用
6. **响应转换**: 将 LLM 的文本响应转换为指定的返回类型
7. **结果返回**: 返回转换后的结果给调用者

### 异步执行机制

- **线程池隔离**: LLM 调用在独立的线程池中执行，不会阻塞事件循环
- **上下文传播**: 支持 contextvars 的正确传播，保持日志追踪等上下文信息
- **异常处理**: 完整的异步异常处理机制

### 内置功能

异步版本继承了同步版本的所有内置功能：

#### 类型转换支持
- 基本类型：`str`, `int`, `float`, `bool`
- 容器类型：`List`, `Dict`, `Tuple`
- Pydantic 模型
- 自定义类型（通过 JSON 序列化）

#### 错误处理机制
- **空响应重试**: 当 LLM 返回空内容时自动重试
- **异步异常捕获**: 完整的异步异常处理和日志记录
- **类型转换错误**: 优雅处理类型转换失败的情况

#### 日志记录
- 详细的执行日志，包括参数、提示内容和响应
- 支持异步日志上下文，便于调试复杂的调用链
- 分级日志（调试、信息、警告、错误）

## 示例

### 示例 1: 基本异步用法

```python
from SimpleLLMFunc.llm_decorator import async_llm_function
from SimpleLLMFunc.interface import OpenAICompatible
import asyncio

# 初始化 LLM 接口
llm = OpenAICompatible(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo"
)

@async_llm_function(llm_interface=llm)
async def summarize_text_async(text: str, max_words: int = 100) -> str:
    """根据输入文本生成一个简洁的摘要，摘要不超过指定的词数。"""
    pass

# 使用异步函数
async def main():
    long_text = "这是一段很长的文本..."
    summary = await summarize_text_async(long_text, max_words=50)
    print(summary)

# 运行异步函数
asyncio.run(main())
```

### 示例 2: 并发处理多个请求

```python
import asyncio
from typing import List

@async_llm_function(llm_interface=llm)
async def translate_text_async(text: str, target_language: str = "English") -> str:
    """将输入文本翻译成指定语言。"""
    pass

@async_llm_function(llm_interface=llm)
async def analyze_sentiment_async(text: str) -> str:
    """分析文本的情感倾向。"""
    pass

async def process_texts_concurrently():
    texts = [
        "今天天气很好",
        "我感到很沮丧",
        "这个产品质量很棒"
    ]
    
    # 并发执行翻译和情感分析
    translation_tasks = [translate_text_async(text) for text in texts]
    sentiment_tasks = [analyze_sentiment_async(text) for text in texts]
    
    # 等待所有任务完成
    translations, sentiments = await asyncio.gather(
        asyncio.gather(*translation_tasks),
        asyncio.gather(*sentiment_tasks)
    )
    
    # 处理结果
    for i, (text, translation, sentiment) in enumerate(zip(texts, translations, sentiments)):
        print(f"文本 {i+1}: {text}")
        print(f"翻译: {translation}")
        print(f"情感: {sentiment}")
        print()

asyncio.run(process_texts_concurrently())
```

### 示例 3: 使用共享线程池

```python
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from typing import List

class AnalysisResult(BaseModel):
    keywords: List[str]
    sentiment: str
    summary: str

# 创建共享的线程池执行器
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="LLM-Async-")

@async_llm_function(llm_interface=llm, executor=executor)
async def extract_keywords_async(text: str) -> List[str]:
    """从文本中提取关键词。"""
    pass

@async_llm_function(llm_interface=llm, executor=executor)
async def analyze_sentiment_async(text: str) -> str:
    """分析文本情感。"""
    pass

@async_llm_function(llm_interface=llm, executor=executor)
async def summarize_async(text: str) -> str:
    """生成文本摘要。"""
    pass

async def comprehensive_analysis(text: str) -> AnalysisResult:
    """对文本进行综合分析"""
    # 并发执行多个分析任务
    keywords, sentiment, summary = await asyncio.gather(
        extract_keywords_async(text),
        analyze_sentiment_async(text),
        summarize_async(text)
    )
    
    return AnalysisResult(
        keywords=keywords,
        sentiment=sentiment,
        summary=summary
    )

async def main():
    text = "这是一段需要分析的长文本..."
    result = await comprehensive_analysis(text)
    print(f"关键词: {result.keywords}")
    print(f"情感: {result.sentiment}")
    print(f"摘要: {result.summary}")

asyncio.run(main())
```

### 示例 4: 与其他异步操作配合

```python
import aiohttp
import asyncio

@async_llm_function(llm_interface=llm)
async def process_content_async(content: str) -> str:
    """处理从网络获取的内容。"""
    pass

async def fetch_and_process_url(session: aiohttp.ClientSession, url: str) -> str:
    """获取URL内容并进行处理"""
    # 异步获取网页内容
    async with session.get(url) as response:
        content = await response.text()
    
    # 异步处理内容
    processed = await process_content_async(content[:1000])  # 只处理前1000字符
    return processed

async def process_multiple_urls():
    urls = [
        "https://example1.com",
        "https://example2.com", 
        "https://example3.com"
    ]
    
    async with aiohttp.ClientSession() as session:
        # 并发处理多个URL
        tasks = [fetch_and_process_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                print(f"处理 {url} 时出错: {result}")
            else:
                print(f"{url}: {result}")

asyncio.run(process_multiple_urls())
```

### 示例 5: 使用工具集的异步函数

```python
from SimpleLLMFunc.tool import tool

@tool
def search_database(query: str) -> str:
    """在数据库中搜索信息"""
    # 数据库搜索逻辑
    return f"数据库搜索结果: {query}"

@async_llm_function(
    llm_interface=llm,
    toolkit=[search_database],
    executor=executor
)
async def intelligent_query_async(user_question: str) -> str:
    """
    根据用户问题智能查询数据库并生成回答。
    可以使用search_database工具来获取相关信息。
    """
    pass

async def main():
    questions = [
        "用户管理相关的功能有哪些？",
        "如何设置权限？",
        "数据备份策略是什么？"
    ]
    
    # 并发处理多个问题
    answers = await asyncio.gather(*[
        intelligent_query_async(question) for question in questions
    ])
    
    for question, answer in zip(questions, answers):
        print(f"问题: {question}")
        print(f"回答: {answer}")
        print()

asyncio.run(main())
```

### 示例 6: 性能比较（串行 vs 并发）

```python
import time
import asyncio

@async_llm_function(llm_interface=llm, executor=executor)
async def process_item_async(item: str) -> str:
    """处理单个项目"""
    pass

async def test_sequential_vs_concurrent():
    items = [f"项目{i}" for i in range(1, 6)]
    
    # 测试串行执行
    print("串行执行:")
    start_time = time.time()
    sequential_results = []
    for item in items:
        result = await process_item_async(item)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"串行执行耗时: {sequential_time:.2f}秒")
    
    # 测试并发执行
    print("\n并发执行:")
    start_time = time.time()
    concurrent_results = await asyncio.gather(*[
        process_item_async(item) for item in items
    ])
    concurrent_time = time.time() - start_time
    print(f"并发执行耗时: {concurrent_time:.2f}秒")
    
    # 性能提升
    speedup = sequential_time / concurrent_time
    print(f"\n性能提升: {speedup:.2f}倍")

asyncio.run(test_sequential_vs_concurrent())
```

## 最佳实践

### 1. 线程池管理
- 对于频繁调用，使用共享线程池而不是为每次调用创建临时线程池
- 根据 LLM 接口的并发限制设置合适的线程池大小
- 记得在应用结束时关闭线程池

### 2. 错误处理
```python
async def robust_llm_call():
    try:
        result = await your_async_llm_function("input")
        return result
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return "默认值"
```

### 3. 超时控制
```python
async def llm_call_with_timeout():
    try:
        result = await asyncio.wait_for(
            your_async_llm_function("input"),
            timeout=30.0  # 30秒超时
        )
        return result
    except asyncio.TimeoutError:
        print("LLM 调用超时")
        return "超时默认值"
```

### 4. 资源清理
```python
# 在应用结束时清理资源
async def cleanup():
    if executor:
        executor.shutdown(wait=True)
```

---

通过这些示例可以看出，`async_llm_function` 装饰器为需要高性能和并发处理的应用提供了强大的异步 LLM 调用能力，同时保持了与同步版本相同的易用性和功能完整性。
