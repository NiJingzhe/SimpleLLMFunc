# 使用指南

本指南将帮助你快速上手 SimpleLLMFunc 框架，从安装到实现第一个 LLM 应用。

## 安装

### 通过 pip 安装

```bash
pip install SimpleLLMFunc
```

### 从源代码安装

```bash
git clone https://github.com/NiJingzhe/SimpleLLMFunc.git
cd SimpleLLMFunc
pip install -e .
```

## 配置

### 环境变量配置

SimpleLLMFunc 使用环境变量和 `.env` 文件进行配置。你可以创建一个 `.env` 文件在项目根目录，或者直接设置环境变量。

基本配置示例 (`.env` 文件):

```
# 日志相关配置
LOG_DIR=./logs
LOG_FILE=agent.log
LOG_LEVEL=DEBUG
```

## 基本用法

### 1. 设置 LLM 接口

首先，你需要创建一个 LLM 接口实例，用于与大语言模型服务通信：

```python
from SimpleLLMFunc import OpenAICompatible

# 通过配置文件方式创建接口
llm_interface = OpenAICompatible.load_from_json_file("provider.json")["provider_name"]["model_name"]

# 或者直接创建接口
llm_interface = OpenAICompatible(
    api_key="your-api-key",
    base_url="https://api.example.com/v1",
    model="model-name"
)
```

### 2. 创建 LLM 函数

使用 `@llm_function` 装饰器将普通 Python 函数转换为 LLM 驱动的函数：

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

### 3. 创建工具

使用 `@tool` 装饰器定义工具函数，让 LLM 能够调用外部服务或API：

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
    # 实际调用天气API的代码
    return {"temperature": "25°C", "humidity": "60%", "condition": "晴天"}
```

### 4. 创建使用工具的 LLM 函数

将工具与 LLM 函数结合使用：

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

### 5. 创建对话型应用

使用 `@llm_chat` 装饰器创建对话式应用：

```python
from SimpleLLMFunc import llm_chat

@llm_chat(llm_interface=my_llm_interface, toolkit=[get_weather])
def chat_assistant(message: str, history: List[Dict[str, str]] = None):
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

## 高级用法

### 自定义提示模板

你可以通过参数定制系统提示和用户提示的模板：

```python
@llm_function(
    llm_interface=my_llm_interface,
    system_prompt_template="你是一个专业的{field}专家，请根据以下信息提供分析:\n{function_description}\n参数: {parameters_description}\n返回: {return_type_description}",
    user_prompt_template="请分析以下数据:\n{parameters}\n请给出专业的分析结果。"
)
def analyze_data(field: str, data: Dict[str, Any]) -> AnalysisResult:
    """详细分析提供的数据，生成结构化报告"""
    pass
```

### API 密钥池管理

使用密钥池实现负载均衡和容错：

```python
from SimpleLLMFunc import APIKeyPool, OpenAICompatible

# 创建密钥池
key_pool = APIKeyPool()
key_pool.add_key("key1")
key_pool.add_key("key2")
key_pool.add_key("key3")

# 创建使用密钥池的LLM接口
llm_interface = OpenAICompatible(
    api_key_pool=key_pool,
    base_url="https://api.example.com/v1",
    model="model-name"
)
```

### 日志查询

通过 trace_id 查询特定函数调用的完整日志：

```python
from SimpleLLMFunc import get_logs_by_trace_id

# 获取指定trace_id的日志
logs = get_logs_by_trace_id("aa123-bb456-cc789")
for log in logs:
    print(f"[{log['timestamp']}] {log['message']}")
```

## 最佳实践

1. **合理设计函数签名**：明确定义参数类型和返回类型，便于LLM理解任务要求
2. **编写详细的docstring**：在函数文档中清晰描述任务目标和预期输出
3. **使用 Pydantic 模型定义输出结构**：确保输出符合预期格式
4. **工具函数保持简单明确**：每个工具只负责一个明确的任务
5. **处理异常情况**：为LLM可能的错误输出添加错误处理逻辑
6. **合理组合多个LLM函数**：将复杂任务拆分为多个子任务，由不同函数负责
