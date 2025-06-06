# Tool 工具系统

## 实现功能

SimpleLLMFunc 的工具系统为大语言模型提供了调用外部函数和 API 的能力，让 LLM 能够执行计算、查询数据、调用服务等操作。工具系统支持两种创建方式，并能自动将 Python 函数转换为 LLM 可理解的工具描述格式。

### 核心功能特性
- **智能类型推断**: 自动从函数签名中提取参数类型和描述信息
- **文档字符串解析**: 支持从 docstring 中解析参数描述
- **JSON Schema 生成**: 自动生成符合 OpenAI Function Calling API 的工具描述
- **类型安全**: 支持基本类型、容器类型、Pydantic 模型等多种类型
- **灵活创建**: 支持装饰器和继承两种创建方式
- **批量序列化**: 支持将多个工具一次性序列化为 API 格式

### 支持的数据类型
- **基本类型**: `str`, `int`, `float`, `bool`
- **容器类型**: `List[T]`, `Dict[K, V]`, `Set[T]`
- **Pydantic 模型**: 自动解析模型字段和验证规则
- **可选参数**: 支持带默认值的可选参数
- **复杂嵌套**: 支持嵌套的容器类型和复杂对象

**⚠️ 注意： 由于LLM提供给工具的参数只能是json支持的类型，所以实际上`Tuple`, `Set`等容器是无法作为工具参数的，但是可以作为返回值类型**

## 使用方法

### 基本语法

#### 方式一：装饰器方式（推荐）

```python
from SimpleLLMFunc.tool import tool

@tool(name="工具名称", description="工具简短描述")
def your_function(param1: Type1, param2: Type2 = default_value) -> ReturnType:
    """
    详细的函数说明，这部分会被包含在工具描述中
    
    Args:
        param1: 参数1的详细描述
        param2: 参数2的详细描述
        
    Returns:
        返回值描述
    """
    # 函数实现
    pass
```

#### 方式二：继承方式（兼容旧版本）

```python
from SimpleLLMFunc.tool import Tool

class YourTool(Tool):
    def __init__(self):
        super().__init__(
            name="工具名称",
            description="工具描述"
        )
    
    def run(self, *args, **kwargs):
        # 工具执行逻辑
        pass
```

### 参数说明

#### @tool 装饰器参数
- **name** (必需): 工具名称，应该简洁明了，符合函数命名规范
- **description** (必需): 工具的简短描述，说明工具的主要功能

#### 函数要求
- **类型标注**: 建议为所有参数添加类型标注，以便自动生成准确的 JSON Schema
- **文档字符串**: 建议编写详细的 docstring，特别是 Args 部分的参数描述
- **返回类型**: 建议添加返回类型标注

### 工具调用流程

1. **工具注册**: 使用 `@tool` 装饰器或继承 `Tool` 类创建工具
2. **参数解析**: 系统自动从函数签名中提取参数信息
3. **Schema 生成**: 自动生成符合 OpenAI API 的工具描述
4. **LLM 调用**: LLM 根据工具描述决定是否调用工具
5. **参数验证**: 系统验证 LLM 提供的参数是否符合要求
6. **函数执行**: 调用原始 Python 函数并返回结果

## 实现方法

### 核心类结构

#### Parameter 类
```python
class Parameter:
    """工具参数的包装类"""
    def __init__(self, name, description, type_annotation, required, default=None, example=None):
        self.name = name                    # 参数名
        self.description = description      # 参数描述
        self.type_annotation = type_annotation  # Python 类型标注
        self.required = required            # 是否必需
        self.default = default             # 默认值
        self.example = example             # 示例值
```

#### Tool 类
```python
class Tool(ABC):
    """抽象工具基类"""
    def __init__(self, name, description, func=None):
        self.name = name
        self.description = description
        self.func = func                   # 关联的函数
        self.parameters = self._extract_parameters()  # 参数列表
    
    def run(self, *args, **kwargs):
        """执行工具"""
        
    def to_openai_tool(self):
        """转换为 OpenAI 工具格式"""
        
    @staticmethod
    def serialize_tools(tools):
        """批量序列化工具"""
```

### 关键实现细节

#### 参数提取机制
系统通过以下步骤提取函数参数信息：

1. **签名分析**: 使用 `inspect.signature()` 获取函数签名
2. **类型提示**: 使用 `get_type_hints()` 获取类型标注
3. **文档解析**: 使用正则表达式解析 docstring 中的参数描述
4. **默认值处理**: 识别可选参数和默认值

#### 类型转换规则
```python
# 基本类型映射
str  -> {"type": "string"}
int  -> {"type": "integer"}
float -> {"type": "number"}
bool -> {"type": "boolean"}

# 容器类型
List[T] -> {"type": "array", "items": schema_of_T}
Dict[K, V] -> {"type": "object", "additionalProperties": schema_of_V}

# Pydantic 模型
BaseModel -> {"type": "object", "properties": model_json_schema}
```

#### 文档字符串解析
系统支持标准的 docstring 格式：

```python
def example_function(param1: str, param2: int = 10):
    """
    函数的主要描述
    
    Args:
        param1: 第一个参数的描述
        param2: 第二个参数的描述，可选
        
    Returns:
        返回值描述
    """
```

## 兼容写法

### 装饰器方式示例

```python
from SimpleLLMFunc.tool import tool
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 示例1：基本数据类型
@tool(name="calculate", description="执行数学计算")
def calculate(expression: str) -> float:
    """
    计算数学表达式的值
    
    Args:
        expression: 要计算的数学表达式，如 "2 + 3 * 4"
        
    Returns:
        计算结果
    """
    return eval(expression)

# 示例2：带可选参数
@tool(name="search_web", description="搜索网络信息")
def search_web(query: str, max_results: int = 10, language: str = "zh") -> List[Dict[str, str]]:
    """
    在网络上搜索信息
    
    Args:
        query: 搜索关键词
        max_results: 最大返回结果数量，默认10条
        language: 搜索语言，默认中文
        
    Returns:
        搜索结果列表，每个结果包含标题和链接
    """
    # 模拟搜索实现
    return [
        {"title": f"搜索结果 {i}", "url": f"http://example.com/{i}"} 
        for i in range(max_results)
    ]

# 示例3：使用 Pydantic 模型
class Location(BaseModel):
    latitude: float = Field(..., description="纬度")
    longitude: float = Field(..., description="经度")
    name: Optional[str] = Field(None, description="位置名称")

@tool(name="get_weather", description="获取天气信息")
def get_weather(location: Location, days: int = 1) -> Dict[str, Any]:
    """
    获取指定位置的天气预报
    
    Args:
        location: 位置信息，包含经纬度坐标
        days: 预报天数，1-7天，默认1天
        
    Returns:
        天气预报数据，包含温度、湿度、天气状况等
    """
    return {
        "location": location.name or f"{location.latitude},{location.longitude}",
        "forecast": [
            {
                "day": i + 1,
                "temperature": 25,
                "humidity": 60,
                "condition": "晴朗"
            } for i in range(days)
        ]
    }

# 示例4：复杂数据处理
@tool(name="analyze_data", description="分析数据集")
def analyze_data(
    data: List[Dict[str, Any]], 
    analysis_type: str,
    include_charts: bool = False
) -> Dict[str, Any]:
    """
    对数据集进行统计分析
    
    Args:
        data: 要分析的数据集，每个元素是一个包含字段的字典
        analysis_type: 分析类型，可选值：summary, trend, correlation
        include_charts: 是否包含图表数据，默认否
        
    Returns:
        分析结果，包含统计信息和可选的图表数据
    """
    result = {
        "type": analysis_type,
        "record_count": len(data),
        "summary": "数据分析完成"
    }
    
    if include_charts:
        result["charts"] = {"type": "bar", "data": "chart_data"}
    
    return result
```

### 继承方式示例

```python
from SimpleLLMFunc.tool import Tool
import requests
from typing import Dict, Any

class WebSearchTool(Tool):
    """网络搜索工具的继承实现方式"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="在网络上搜索信息并返回相关结果"
        )
    
    def run(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        执行网络搜索
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数量
            
        Returns:
            搜索结果列表
        """
        # 实际的搜索逻辑
        results = []
        for i in range(max_results):
            results.append({
                "title": f"搜索结果 {i+1}: {query}",
                "url": f"https://example.com/result/{i+1}",
                "snippet": f"关于{query}的相关信息..."
            })
        return results

class APICallTool(Tool):
    """API 调用工具"""
    
    def __init__(self, api_base_url: str):
        super().__init__(
            name="api_call",
            description="调用外部 API 获取数据"
        )
        self.api_base_url = api_base_url
    
    def run(self, endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        调用 API 端点
        
        Args:
            endpoint: API 端点路径
            method: HTTP 方法
            data: 请求数据
            
        Returns:
            API 响应数据
        """
        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
        
        if method.upper() == "GET":
            response = requests.get(url, params=data)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"不支持的 HTTP 方法: {method}")
        
        return response.json()
```

### 工具使用示例

```python
from SimpleLLMFunc.llm_decorator import llm_function, llm_chat
from SimpleLLMFunc.interface import OpenAICompatible

# 初始化 LLM
llm = OpenAICompatible(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo"
)

# 在 llm_function 中使用工具
@llm_function(
    llm_interface=llm,
    toolkit=[calculate, search_web, get_weather]
)
def intelligent_assistant(query: str) -> str:
    """
    智能助手，可以进行计算、搜索和查询天气。
    根据用户查询的内容，选择合适的工具来提供准确的答案。
    """
    pass

# 在 llm_chat 中使用工具
@llm_chat(
    llm_interface=llm,
    toolkit=[calculate, search_web, get_weather, analyze_data]
)
def chat_with_tools(message: str, history: List[Dict[str, str]] = []):
    """
    支持工具调用的聊天助手。
    可以执行计算、搜索网络、查询天气和分析数据。
    """
    pass

# 使用示例
if __name__ == "__main__":
    # 使用 llm_function
    result = intelligent_assistant("帮我计算 25 * 4 + 18 的结果")
    print(f"计算结果: {result}")
    
    # 使用 llm_chat
    history = []
    for response, updated_history in chat_with_tools("北京今天天气怎么样？", history):
        if response:
            print(response, end="")
        else:
            history = updated_history
            break
```

### 高级用法

#### 工具序列化和检查

```python
from SimpleLLMFunc.tool import Tool
import json

# 创建工具列表
tools = [calculate, search_web, get_weather]

# 序列化为 OpenAI 格式
openai_tools = Tool.serialize_tools(tools)

# 查看生成的工具描述
for tool_spec in openai_tools:
    print(json.dumps(tool_spec, indent=2, ensure_ascii=False))
```

#### 动态工具创建

```python
def create_tool_from_config(config: Dict[str, Any]):
    """根据配置动态创建工具"""
    
    @tool(name=config["name"], description=config["description"])
    def dynamic_tool(**kwargs):
        # 根据配置执行相应逻辑
        return config["handler"](kwargs)
    
    return dynamic_tool

# 配置示例
tool_config = {
    "name": "custom_processor",
    "description": "自定义数据处理器",
    "handler": lambda data: {"processed": True, "data": data}
}

custom_tool = create_tool_from_config(tool_config)
```

#### 工具链组合

```python
@tool(name="multi_step_analysis", description="多步骤数据分析")
def multi_step_analysis(data: List[Dict[str, Any]], steps: List[str]) -> Dict[str, Any]:
    """
    执行多步骤数据分析流程
    
    Args:
        data: 原始数据
        steps: 分析步骤列表，如 ["clean", "analyze", "visualize"]
        
    Returns:
        分析结果
    """
    results = {"steps_completed": []}
    
    for step in steps:
        if step == "clean":
            # 数据清洗
            results["cleaned_records"] = len(data)
        elif step == "analyze":
            # 数据分析
            results["analysis"] = {"mean": 0, "std": 0}
        elif step == "visualize":
            # 数据可视化
            results["charts"] = ["bar", "line", "pie"]
        
        results["steps_completed"].append(step)
    
    return results
```

---

工具系统提供了强大而灵活的扩展机制，让 LLM 能够调用各种外部功能。通过装饰器方式，开发者可以轻松地将现有函数转换为 LLM 可用的工具，而继承方式则提供了更多的自定义控制。系统自动处理类型转换和参数验证，确保工具调用的安全性和准确性。