# 使用指南

本指南将帮助你快速上手 SimpleLLMFunc 框架，从安装到实现第一个 LLM（大语言模型）应用。

## 安装

### 通过 pip 安装
本项目发表在 PyPI，可以直接通过 `pip` 安装（安装前请根据需要创建相应虚拟环境）：

```bash
pip install SimpleLLMFunc
```

注意：发表在 PyPI 的版本可能不是最新的开发版本，可能会缺少一些新功能或修复。

### 从源代码安装

首先您需要安装 [Poetry](https://python-poetry.org/) 作为包管理工具（根据 Poetry 的官方指南，请勿将 Poetry 安装至本项目的虚拟或者任何欲使用其管理的虚拟环境中）。然后可以通过以下命令克隆项目并安装依赖：

```bash
git clone https://github.com/NiJingzhe/SimpleLLMFunc.git
cd SimpleLLMFunc
poetry install
```


## 详细使用说明
```{toctree}
:maxdepth: 1
:caption: 目录：

llm_interface 接口 <detailed_guide/llm_interface>
llm_function 装饰器 <detailed_guide/llm_function>
llm_chat 装饰器 <detailed_guide/llm_chat>
tool 装饰器 <detailed_guide/tool>
```


## 配置

### 环境变量配置

SimpleLLMFunc 使用环境变量和 `.env` 文件进行配置。你可以在项目根目录创建一个 `.env` 文件，或者直接设置环境变量。

#### 日志相关配置

- `LOG_DIR`：日志文件存放目录，默认为当前目录 `./`
- `LOG_FILE`：日志文件名，默认为 `agent.log`
- `LOG_LEVEL`：日志级别（DEBUG、INFO、WARNING、ERROR、CRITICAL 五级，详细程度依次递减）

示例：

```
# 日志相关配置
LOG_DIR=./logs
LOG_FILE=my_agent.log
LOG_LEVEL=WARNING
```


## 基本用法

### 设置 LLM 接口

首先，创建一个 LLM 接口实例，用于与大语言模型服务通信：

- 通过配置文件创建接口（推荐）：使用 `OpenAICompatible.load_from_json_file` 函数
```python
from SimpleLLMFunc import OpenAICompatible
# 加载 provider.json 配置文件，获取 LLM 接口实例
llm_interface = OpenAICompatible.load_from_json_file("provider.json")["provider_name"]["model_name"]
```
- 直接创建接口
```python
from SimpleLLMFunc import OpenAICompatible, APIKeyPool

key_pool = APIKeyPool(
    api_keys=["your-api-key-1", "your-api-key-2"],  # 替换为你的API密钥
    provider_id="openai_model-name"
)

llm_interface = OpenAICompatible(
    api_key_pool=key_pool,
    base_url="https://api.example.com/v1",
    model="model-name"
)
```

### 创建并使用 LLM 函数

使用 `@llm_function` 装饰器以创建 LLM 函数，具体步骤如下：

1. 使用 Pydantic 定义输入参数和返回值类型，返回值类型必须是可json序列化的类型，推荐Pydantic Model或者基础类型；
2. 定义函数，并添加使用 `@llm_function` 装饰器；
3. 在函数的 docstring 中描述函数的功能、参数和返回值；
4. 函数体留空，实际操作由 LLM 完成，返回的字符串由框架自动尝试解析为指定格式，若解析失败则抛出异常。
5. 调用方法与普通 Python 函数相同。

`@llm_function` 装饰器的参数包括：

- `llm_interface`：指定使用的 LLM 接口实例；
- `toolkit`：可选，指定工具函数列表；
- `system_prompt_template`：可选，系统提示模板；
- `user_prompt_template`：可选，用户提示模板；
- 额外的关键字参数将直接传递给 LLM 接口。

示例：

```python
from typing import List
from pydantic import BaseModel, Field
from SimpleLLMFunc import llm_function

# 定义返回类型模型 (可选但推荐)
class ProductReview(BaseModel):
    rating: int = Field(..., description="产品评分，1-5分")
    pros: List[str] = Field(..., description="产品优点列表")
    cons: List[str] = Field(..., description="产品缺点列表")
    summary: str = Field(..., description="评价总结")

# 创建 LLM 函数
@llm_function(llm_interface=my_llm_interface)
def analyze_product_review(product_name: str, review_text: str) -> ProductReview:
    """
    分析产品评论，提取关键信息并生成结构化评测报告

    Args:
        product_name: 产品名称
        review_text: 用户评论文本

    Returns:
        包含评分、优缺点和总结的产品评测报告
    """
    pass  # 函数体为空，实际执行由LLM完成

# 使用函数
result = analyze_product_review("无线耳机", "这款耳机音质不错，但电池续航较差...")
print(f"评分: {result.rating}")
print(f"优点: {', '.join(result.pros)}")
```

### 创建可供 LLM 函数使用的工具

在 LLM 函数中使用工具函数，可以让 LLM 访问外部 API 或执行特定任务。

使用 `@tool` 装饰器以创建工具函数，具体步骤如下：

1. 定义工具函数，并添加 `@tool` 装饰器；
2. 撰写工具函数的 docstring，描述函数的功能、参数和返回值，返回值必须是可序列化的有语义的类型，建议是Pydantic Model或者基础类型；

`@tool` 装饰器的参数包括：

- `name`：工具名称；
- `description`：工具简略描述，将与该函数的 docstring **一并**传递给 LLM 作为描述；
- 额外的关键字参数将直接传递给 LLM 接口。

示例：

```python
from SimpleLLMFunc import tool

@tool(name="get_weather", description="获取指定城市的天气信息")
def get_weather(city: str) -> Dict[str, str]:
    """
    获取指定城市的天气信息

    Args:
        city: 城市名称

    Returns:
        包含温度、湿度和天气状况的字典
    """
    # 实际调用天气API的代码，此处略
    return {"temperature": "25°C", "humidity": "60%", "condition": "晴天"}
```

### 创建使用工具的 LLM 函数

在 `@llm_function` 装饰器中指定 `toolkit` 参数中传入工具函数列表即可，其余步骤与 [创建 LLM 函数](#创建 LLM 函数) 一节所述相同。

```python
@llm_function(llm_interface=my_llm_interface, toolkit=[get_weather])
def get_daily_recommendation(city: str) -> WeatherInfo:
    """
    获取指定城市的今日活动建议

    Args:
        city: 城市名称

    Returns:
        WeatherInfo对象，包含天气信息和活动建议
    """
    pass
```

### 创建async的 LLM 函数
如果需要创建异步的 LLM 函数，可以在函数定义前添加 `async` 关键字，并使用 `@async_llm_function` 装饰器。
```python
from SimpleLLMFunc import async_llm_function

@async_llm_function(llm_interface=my_llm_interface)
async def async_analyze_product_review(product_name: str, review_text: str) -> ProductReview:
    """    分析产品评论，提取关键信息并生成结构化评测报告  
    Args:
        product_name: 产品名称
        review_text: 用户评论文本
    Returns:
        包含评分、优缺点和总结的产品评测报告
    """
    pass  # 函数体为空，实际执行由LLM完成

```

### 创建对话型应用

使用 `@llm_chat` 装饰器以创建对话式应用，具体步骤如下：

```python
from SimpleLLMFunc import llm_chat

@llm_chat(llm_interface=my_llm_interface, toolkit=[get_weather])
def chat_assistant(history: List[Dict[str, str]], message: str):
    """
    你是一个友好的助手，可以回答用户问题并提供帮助。
    你可以使用工具来获取实时信息，例如天气状况。
    """
    pass

# 使用对话函数
history = []  # 初始化空的对话历史
response, updated_history = next(chat_assistant("今天北京天气怎么样？", history))
print(response)
# 继续对话，传入更新后的历史记录
next_response, updated_history = next(chat_assistant("那我应该穿什么衣服？", updated_history))
```

### 创建异步的对话型应用
如果需要创建异步的对话型应用，可以在函数定义前添加 `async` 关键字，并使用 `@async_llm_chat` 装饰器。
```python
from SimpleLLMFunc import async_llm_chat    
@async_llm_chat(llm_interface=my_llm_interface, toolkit=[get_weather])
async def async_chat_assistant(history: List[Dict[str, str]], message: str):
    """你是一个友好的助手，可以回答用户问题并提供帮助。
    你可以使用工具来获取实时信息，例如天气状况。
    """
    pass
```

## 最佳实践

1. **合理设计函数签名**：明确定义参数类型和返回类型，便于LLM理解任务要求
2. **编写详细的docstring**：在函数文档中清晰描述任务目标和预期输出
3. **使用 Pydantic 模型定义输出结构**：确保输出符合预期格式
4. **工具函数保持简单明确**：每个工具只负责一个明确的任务
5. **处理异常情况**：为LLM可能的错误输出添加错误处理逻辑
6. **合理组合多个LLM函数**：将复杂任务拆分为多个子任务，由不同函数负责
7. **使用日志**：利用日志系统记录函数调用和LLM交互的详细信息，便于后续分析和调试
8. **尝试从日志中清洗出训练数据用于微调模型**：定期分析日志，提取有价值的对话示例，构建高质量的训练数据集
