# SimpleLLMFunc

![SimpleLLMFunc](img/repocover.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/NiJingzhe/SimpleLLMFunc/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)

一个轻量级的LLM调用和工具集成框架，支持类型安全的LLM函数装饰器、多种模型接口和强大的日志跟踪系统。

## 特性

- **LLM函数装饰器**：简化LLM调用，支持类型安全的函数定义和返回值处理
- **多模型支持**：支持多种LLM提供商接口（目前支持智谱AI）
- **API密钥管理**：自动化API密钥负载均衡，优化资源利用
- **结构化输出**：使用Pydantic模型定义结构化返回类型
- **强大的日志系统**：支持trace_id跟踪和搜索，方便调试和监控
- **工具系统**：支持Agent与外部环境交互，易于扩展

## 项目结构

```
SimpleLLMFunc/
├── SimpleLLMFunc/            # 核心包
│   ├── interface/             # LLM 接口
│   │   ├── llm_interface.py   # LLM 接口抽象类
│   │   ├── key_pool.py        # API 密钥管理
│   │   └── zhipu.py           # 智谱 AI 接口实现
│   ├── llm_function/          # LLM函数装饰器
│   │   └── llm_function_decorator.py # 函数装饰器实现
│   ├── logger/                # 日志系统
│   │   ├── logger.py          # 日志核心实现
│   │   └── logger_config.py   # 日志配置
│   ├── tool/                  # 工具系统
│   │   ├── tool.py            # 工具抽象基类
│   │   └── schemas.py         # 工具参数模型
│   └── config.py              # 全局配置
└── examples/                  # 示例代码
    └── llm_function_example.py # LLM函数示例
```

## 配置管理

SimpleLLMFunc使用分层配置系统：

- 环境变量：最高优先级
- `.env` 文件：次优先级
- `config.py` 默认值：最低优先级

### 配置示例 (.env)

```
ZHIPU_API_KEYS=["your-api-key-1", "your-api-key-2"]
LOG_DIR=./
LOG_FILE=agent.log
LOG_LEVEL=DEBUG
```

## LLM函数装饰器

SimpleLLMFunc的核心特性是LLM函数装饰器，它允许您只通过声明带有类型标注的函数和撰写DocString来实现一个函数。

```python
"""
使用LLM函数装饰器的示例
"""
from typing import Dict, List
from pydantic import BaseModel, Field

from SimpleLLMFunc.llm_decorator.llm_function_decorator import llm_function
from SimpleLLMFunc.interface import ZhipuAI_glm_4_flash_Interface
from SimpleLLMFunc.logger.logger import app_log
from SimpleLLMFunc.tool import tool

# 定义一个Pydantic模型作为返回类型
class ProductReview(BaseModel):
    rating: int = Field(..., description="产品评分，1-5分")
    pros: List[str] = Field(..., description="产品优点列表")
    cons: List[str] = Field(..., description="产品缺点列表")
    summary: str = Field(..., description="评价总结")

# 使用装饰器创建一个LLM函数
@llm_function(
    llm_interface=ZhipuAI_glm_4_flash_Interface,
    system_prompt="你是一个专业的产品评测专家，可以客观公正地评价各种产品。"
)
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


@tool(name="天气查询", description="获取指定城市的天气信息")
def get_weather(city: str) -> Dict[str, str]:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称
        
    Returns:
        包含温度、湿度和天气状况的字典
    """
    return {
        "temperature": "32°C",
        "humidity": "80%",
        "condition": "Cloudy"
    }

class WeatherInfo(BaseModel):
    city: str = Field(..., description="城市名称")
    temperature: str = Field(..., description="当前温度")
    humidity: str = Field(..., description="当前湿度")
    condition: str = Field(..., description="天气状况")

@llm_function(
    llm_interface=ZhipuAI_glm_4_flash_Interface,
    tools=[get_weather]
)
def weather(city: str) -> WeatherInfo:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称
        
    Returns:
        WeatherInfo对象，包含温度、湿度和天气状况
    例如：{"city": "L.A.", "temperature": "25°C", "humidity": "60%", "condition": "晴天"}
    """
    pass


def main():
    
    app_log("开始运行示例代码")
    # 测试产品评测分析
    product_name = "XYZ无线耳机"
    review_text = """
    我买了这款XYZ无线耳机已经使用了一个月。音质非常不错，尤其是低音部分表现出色，
    佩戴也很舒适，可以长时间使用不感到疲劳。电池续航能力也很强，充满电后可以使用约8小时。
    不过连接偶尔会有些不稳定，有时候会突然断开。另外，触控操作不够灵敏，经常需要点击多次才能响应。
    总的来说，这款耳机性价比很高，适合日常使用，但如果你需要用于专业音频工作可能还不够。
    """
    
    try:
        print("\n===== 产品评测分析 =====")
        result = analyze_product_review(product_name, review_text)
        print(f"评分: {result.rating}/5")
        print("优点:")
        for pro in result.pros:
            print(f"- {pro}")
        print("缺点:")
        for con in result.cons:
            print(f"- {con}")
        print(f"总结: {result.summary}")
    except Exception as e:
        print(f"产品评测分析失败: {e}")
        
    # 测试天气查询
    city = "Hangzhou"
    try:
        print("\n===== 天气查询 =====")
        weather_info = weather(city)
        print(f"城市: {city}")
        print(f"温度: {weather_info.temperature}")
        print(f"湿度: {weather_info.humidity}")
        print(f"天气状况: {weather_info.condition}")
    except Exception as e:
        print(f"天气查询失败: {e}")
    
        

if __name__ == "__main__":
    main()

```
Output:
```bash
===== 产品评测分析 =====
评分: 4/5
优点:
- 音质非常不错，尤其是低音部分表现出色
- 佩戴也很舒适，可以长时间使用不感到疲劳
- 电池续航能力也很强，充满电后可以使用约8小时
- 性价比很高，适合日常使用
缺点:
- 连接偶尔会有些不稳定，有时候会突然断开
- 触控操作不够灵敏，经常需要点击多次才能响应
- 如果需要用于专业音频工作可能还不够
总结: 音质和续航表现优秀，佩戴舒适，但连接稳定性不足，触控操作不够灵敏，适合日常使用，但不适合专业音频工作。

===== 天气查询 =====
城市: Hangzhou
温度: 32°C
湿度: 80%
天气状况: Cloudy
```


正如这个例子展现的，只需要声明一个函数，声明返回类型，写好DocString，剩下的交给装饰器即可。



### 装饰器特性

- **类型安全**：根据函数签名自动识别参数和返回类型
- **Pydantic集成**：支持Pydantic模型作为返回类型，确保结果符合预定义结构
- **提示词自动构建**：基于函数文档和类型标注自动构建提示词
- **系统提示分离**：将系统提示与用户输入分开，优化LLM的响应质量

## LLM接口

LLM接口的封装是为了能够分隔供应商，如果是OpenAI SDK Compatiable的模型，可以省去设置BASE URL的重复工作，同时具有更好的类型提示。

同样的也能够支持某些非OpenAI SDK Compatiable的模型。

SimpleLLMFunc的LLM接口设计原则：

- 简单、无状态的函数调用
- 支持普通和流式两种调用模式
- API密钥负载均衡

### 示例用法

这里展示了接口的两种暴露接口，但实际使用过程中用户并不会接触到这样的直接调用。用户只会将接口对象作为参数传入装饰器。

```python
from SimpleLLMFunc.interface import ZhipuAI_glm_4_flash_Interface

# 非流式调用
response = ZhipuAI_glm_4_flash_Interface.chat(
    trace_id="unique_trace_id",
    messages=[{"role": "user", "content": "你好"}]
)

# 流式调用
for chunk in ZhipuAI_glm_4_flash_Interface.chat_stream(
    trace_id="unique_trace_id",
    messages=[{"role": "user", "content": "你好"}]
):
    print(chunk)
```

## 日志系统

SimpleLLMFunc包含强大的日志系统，支持：

- 不同级别的日志（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- 按trace_id跟踪和搜索相关日志
- 自动记录代码位置信息
- 彩色控制台输出
- JSON格式文件日志，便于解析

后续计划加入对每个llm function的性能监控，让开发者能够更好的追踪输入输出与响应时间，以进行Prompt调优和工作流效率优化。

### 日志使用示例

```python
from SimpleLLMFunc.logger import app_log, push_error, search_logs_by_trace_id

# 记录信息日志
app_log("操作成功完成", trace_id="operation_123")

# 记录错误日志
push_error("操作失败", trace_id="operation_123", exc_info=True)

# 使用日志上下文注入统一字段
with log_context(trace_id = "unified traceid")

    push_error("操作失败") # 不需要显式指定trace id，会自动获得上下文中的trace id

```

## 工具系统

SimpleLLMFunc实现了可扩展的工具系统，使LLM能够与外部环境交互。工具系统支持两种定义方式：函数装饰器方式（推荐）和类继承方式（向后兼容）。

### 函数装饰器方式（推荐）

使用`@tool`装饰器将普通Python函数转换为工具，非常简洁直观，对于参数的描述一部分可以来源于`Pydantic Model`的`description`字段，函数入参的`description`则来自DocString。你需要在DocString中包含`Args:`或者`Parameters:`字样，然后每一行写一个`[param name]: [description]`，正如你在下面的例子中看到的这样。

```python
from pydantic import BaseModel, Field
from SimpleLLMFunc.tool import tool

# 定义复杂参数的Pydantic模型
class Location(BaseModel):
    latitude: float = Field(..., description="纬度")
    longitude: float = Field(..., description="经度")

# 使用装饰器创建工具
@tool(name="get_weather", description="获取指定位置的天气信息")
def get_weather(location: Location, days: int = 1) -> dict:
    """
    获取指定位置的天气预报
    
    Args:
        location: 位置信息，包含经纬度
        days: 预报天数，默认为1天
        
    Returns:
        天气预报信息
    """
    # 实际实现会调用天气API
    return {
        "location": f"{location.latitude},{location.longitude}",
        "forecast": [{"day": i, "temp": 25, "condition": "晴朗"} for i in range(days)]
    }
```

这种方式具有以下优势：
- 直接使用Python原生类型和Pydantic模型进行参数标注
- 自动从函数签名和文档字符串提取参数信息
- 装饰后的函数仍可直接调用，便于测试

### 类继承方式（向后兼容）

也可以通过继承`Tool`类并实现`run`方法来创建工具：

```python
from SimpleLLMFunc.tool import Tool

class WebSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="在互联网上搜索信息"
        )
    
    def run(self, query: str, max_results: int = 5):
        """
        执行网络搜索
        
        Args:
            query: 搜索查询词
            max_results: 返回结果数量，默认为5
            
        Returns:
            搜索结果列表
        """
        # 搜索逻辑实现
        return {"results": ["结果1", "结果2", "结果3"]}
```

### 与LLM函数集成

使用装饰器方式定义的工具可以直接传递给LLM函数装饰器：

```python
from SimpleLLMFunc.llm_decorator import llm_function
from SimpleLLMFunc.interface import ZhipuAI_glm_4_flash_Interface

@llm_function(
    llm_interface=ZhipuAI_glm_4_flash_Interface,
    tools=[get_weather, search_web],  # 直接传递被@tool装饰的函数
    system_prompt="你是一个助手，可以使用工具来帮助用户。"
)
def answer_with_tools(question: str) -> str:
    """
    回答用户问题，必要时使用工具获取信息
    
    Args:
        question: 用户问题
        
    Returns:
        回答内容
    """
    pass
```

两种方式可以混合使用：

```python
@llm_function(
    llm_interface=ZhipuAI_glm_4_flash_Interface,
    tools=[get_weather, WebSearchTool()],  # 混合使用两种方式定义的工具
    system_prompt="你是一个助手，可以使用工具来帮助用户。"
)
def answer_with_mixed_tools(question: str) -> str:
    """回答用户问题，必要时使用工具获取信息"""
    pass
```

## API密钥管理

SimpleLLMFunc使用`APIKeyPool`类管理多个API密钥，实现负载均衡：

- 自动选择最少负载的API密钥
- 单例模式确保每个提供商只有一个密钥池，密钥池使用小根堆来进行负载均衡，每次取出load最低的KEY
- 自动跟踪每个密钥的使用情况

## 安装和使用

1. 克隆此仓库
2. 创建`.env`文件并配置您的API密钥
3. 使用Poetry安装依赖：`poetry install`
4. 导入并使用SimpleLLMFunc的各个组件

## 许可证

MIT