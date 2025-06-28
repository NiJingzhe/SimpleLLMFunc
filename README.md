![SimpleLLMFunc](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/img/repocover_new.png?raw=true)

<center>
<h2 style="font-size:2em;">LLM as Function, Prompt as Code</h2>
</center>

----

![Github Stars](https://img.shields.io/github/stars/NiJingzhe/SimpleLLMFunc.svg?style=social)
![Github Forks](https://img.shields.io/github/forks/NiJingzhe/SimpleLLMFunc.svg?style=social)


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/NiJingzhe/SimpleLLMFunc/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NiJingzhe/SimpleLLMFunc/pulls)

### 更新说明 (0.2.9 Latest)
### Look here: [Change Log](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/CHANGELOG.md)

### 文档（ReadtheDoc）
### Look here: [Docs](https://simplellmfunc.readthedocs.io/zh-cn/latest/introduction.html)

-----
## What & why

一个轻量级的LLM应用开发框架，支持类型安全的`llm_function`装饰器用于设计Workflow步骤，同时也支持`llm_chat`装饰器用于设计Agent系统。同时具有可配置的供应商和强大的日志跟踪系统。

做过LLM开发的同志们或许都经历过这样的困境：

  1. 为了一些定制化功能，不得不用一些抽象trick，于是让一个本身主打低代码好理解的流变得抽象
  2. 使用低代码框架制作Workflow一时爽，但是发现又没有类型定义又没有代码提示，复杂流到后面的时候忘记了前面返回的格式
  3. 我只想要一个非常非常简单的无状态功能，但是用LangChain还得阅读一堆文档，创建一堆节点。
  4. 不管是LangChain还是Dify，居然都不能构建有向有环的逻辑？？？？（虽然Dify新出了Condition Loop但并不是理想的形式）
  5. 但是不用框架的话又要自己写LLM API Call，每次都要写一遍这个Call代码很麻烦。而且Prompt作为变量形式存在没有那么直观的体现逻辑和在程序中的作用。
   
**这时候就有人问了，啊主播主播这些框架啊什么的都太复杂了，而不用框架有又很麻烦，有没有一种又简单又方便又快速的方法呢?**

### 有的兄弟，有的！！

**SimpleLLMFunc** 的目标就是提供一个简单的恰到好处的框架，帮你实现了繁琐的API CALL撰写，帮你做了一点点Prompt工程，同时保留最大的自由度。

基础功能单元是函数，让你以最 “Coding” 的方式，快速集成LLM能力到你的应用中，同时不会受到只能创建DAG的约束，能自由的构建流程。

Prompt会以DocString的形式存在，一方面强制你撰写良好的函数功能说明，让其他协作者对于函数功能一目了然，另一方面这就好像是用自然语言写了一段代码，功能描述就这样出现在了最合适的位置上，再也不用为了看一个函数的功能而到处跳转找到Prompt变量了。

-----

## 特性

- **LLM函数装饰器**：简化LLM调用，支持类型安全的函数定义和返回值处理（但是小模型有很大概率无法输出正确的json格式）
- **异步支持**：提供 `async_llm_function` 和 `async_llm_chat` 装饰器，支持原生异步调用
- **多模态支持**：支持文本、图片URL和本地图片路径的多模态输入处理
- **通用模型接口**：支持任何符合OpenAI API格式的模型服务，无需针对每个供应商开发专门实现
- **API密钥管理**：自动化API密钥负载均衡，优化资源利用
- **流量控制**：集成令牌桶算法，实现智能流量平滑
- **结构化输出**：使用Pydantic模型定义结构化返回类型
- **强大的日志系统**：支持trace_id跟踪和搜索，方便调试和监控，即将支持token用量统计
- **工具系统**：支持Agent与外部环境交互，易于扩展

## 项目结构

```
SimpleLLMFunc/
├── SimpleLLMFunc/            # 核心包
│   ├── interface/             # LLM 接口
│   │   ├── llm_interface.py   # LLM 接口抽象类
│   │   ├── key_pool.py        # API 密钥管理
│   │   ├── openai_compatible.py # OpenAI Compatible 通用接口实现
│   │   └── token_bucket.py    # 流量控制令牌桶实现
│   ├── llm_decorator/         # LLM装饰器
│   │   ├── llm_chat_decorator.py     # 对话函数装饰器实现
│   │   ├── llm_function_decorator.py # 无状态函数装饰器实现
│   │   ├── multimodal_types.py       # 多模态类型定义
│   │   └── utils.py           # 装饰器工具函数
│   ├── logger/                # 日志系统
│   │   ├── logger.py          # 日志核心实现
│   │   └── logger_config.py   # 日志配置
│   ├── tool/                  # 工具系统
│   │   └── tool.py            # 工具基类和工具函数装饰器定义
│   ├── type/                  # 类型定义
│   │   └── __init__.py        # 多模态类型导出
│   ├── config.py              # 全局配置
│   └── utils.py               # 通用工具函数
└── examples/                  # 示例代码
    ├── llm_function_example.py  # LLM函数示例
    ├── llm_chat_example.py      # 对话函数示例
    ├── async_llm_func.py        # 异步LLM函数示例
    └── simple_manus.py          # 包含多种工具和对话函数的综合示例
```
## 配置管理

SimpleLLMFunc使用分层配置系统：

- 环境变量：最高优先级
- `.env` 文件：次优先级

### 日志配置 (.env)

```bash
# 日志相关配置
LOG_DIR=./logs
LOG_FILE=agent.log
LOG_LEVEL=DEBUG
```

## LLM函数装饰器 - Prompt As Code

- ### llm_function

SimpleLLMFunc的核心理念是 **"Everything is Function, Prompt is Code"**。通过将Prompt直接编写在函数的文档字符串（DocString）中，我们实现了：

1. **更好的代码可读性** - Prompt与其作用的函数紧密结合，一目了然
2. **类型安全** - 使用Python类型标注和Pydantic模型确保输入输出的正确性
3. **智能提示** - IDE可以提供完整的代码补全和类型检查
4. **文档即Prompt，Prompt即代码，代码即文档** - DocString既是函数文档，也是LLM的Prompt

```python
"""
使用LLM函数装饰器的示例
"""
from typing import List
from pydantic import BaseModel, Field
from SimpleLLMFunc import llm_function, OpenAICompatible, app_log

# 定义一个Pydantic模型作为返回类型
class ProductReview(BaseModel):
    rating: int = Field(..., description="产品评分，1-5分")
    pros: List[str] = Field(..., description="产品优点列表")
    cons: List[str] = Field(..., description="产品缺点列表")
    summary: str = Field(..., description="评价总结")

# 使用装饰器创建一个LLM函数
@llm_function(
    llm_interface=OpenAICompatible.load_from_json_file("provider.json")["volc_engine"]["deepseek-v3-250324"]
)
def analyze_product_review(product_name: str, review_text: str) -> ProductReview:
    """你是一个专业的产品评测专家，需要客观公正地分析以下产品评论，并生成一份结构化的评测报告。
    
    报告应该包括：
    1. 产品总体评分（1-5分）
    2. 产品的主要优点列表
    3. 产品的主要缺点列表
    4. 总结性评价
    
    评分规则：
    - 5分：完美，几乎没有缺点
    - 4分：优秀，优点明显大于缺点
    - 3分：一般，优缺点基本持平
    - 2分：较差，缺点明显大于优点
    - 1分：很差，几乎没有优点
    
    Args:
        product_name: 要评测的产品名称
        review_text: 用户对产品的评论内容
        
    Returns:
        一个结构化的ProductReview对象，包含评分、优点列表、缺点列表和总结
    """
    pass  # Prompt as Code, Code as Doc

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
        # result is directly a Pydantic model instance
        # no need to deserialize
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

if __name__ == "__main__":
    main()

```
Output:

```text
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
```


正如这个例子展现的，只需要声明一个函数，声明返回类型，写好DocString，剩下的交给装饰器即可。
函数直接返回的就是一个`Pydantic`对象，不需要做额外的反序列化操作。

- ### llm chat

同样的我们也支持创建**对话类函数**，以下是一个简单的对话函数的例子：[Simple Manus](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/simple_manus.py)。

这个例子实现了一些工具和一个对话函数，能够实现代码专精的Manus类似物

- ### 异步装饰器支持

SimpleLLMFunc 提供了完整的异步支持，包括 `async_llm_function` 和 `async_llm_chat` 装饰器：

```python
from SimpleLLMFunc import async_llm_function, async_llm_chat

# 异步LLM函数
@async_llm_function(llm_interface=my_llm_interface)
async def async_analyze_text(text: str) -> str:
    """异步分析文本内容"""
    pass

# 异步对话函数
@async_llm_chat(llm_interface=my_llm_interface, stream=True)
async def async_chat(message: str, history: List[Dict[str, str]]) -> AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]:
    """异步对话功能，支持流式响应"""
    pass

# 使用示例
async def main():
    result = await async_analyze_text("需要分析的文本")
    
    async for response, updated_history in async_chat("你好", []):
        print(response)
```

- ### 多模态支持

SimpleLLMFunc 支持多模态输入，可以处理文本、图片URL和本地图片：

```python
from SimpleLLMFunc import llm_function
from SimpleLLMFunc.type import Text, ImgUrl, ImgPath

@llm_function(llm_interface=my_llm_interface)
def analyze_image(
    description: Text,           # 文本描述
    web_image: ImgUrl,          # 网络图片URL
    local_image: ImgPath        # 本地图片路径
) -> str:
    """分析图像并根据描述提供详细说明
    
    Args:
        description: 对图像分析的具体要求
        web_image: 要分析的网络图片URL
        local_image: 要对比的本地参考图片路径
        
    Returns:
        详细的图像分析结果
    """
    pass

# 使用示例
result = analyze_image(
    description=Text("请详细描述这两张图片的区别"),
    web_image=ImgUrl("https://example.com/image.jpg"),
    local_image=ImgPath("./reference.jpg")
)
```

### 装饰器特性

- **类型安全**：根据函数签名自动识别参数和返回类型
- **异步支持**：提供 `async_llm_function` 和 `async_llm_chat` 装饰器，支持原生异步调用
- **多模态处理**：支持 `Text`、`ImgUrl`、`ImgPath` 类型的多模态输入
- **Pydantic集成**：支持Pydantic模型作为返回类型，确保结果符合预定义结构，对于能力较弱的模型有较大概率在自动重试后也无法输出正确的json格式
- **提示词自动构建**：基于函数文档和类型标注自动构建提示词

## LLM供应商接口

SimpleLLMFunc 提供了灵活的 LLM 接口支持，主要包括：

1. **OpenAI Compatible 通用接口** - 支持任何符合 OpenAI API 格式的模型服务，推荐通过`provider.json`配置文件来管理不同供应商的模型接口。
2. **自定义接口扩展** - 通过继承 `LLM_Interface` 基类实现自定义的模型接口。

### OpenAI Compatible 接口示例

```python
from SimpleLLMFunc import OpenAICompatible

# 从配置文件加载模型接口
provider_interfaces = OpenAICompatible.load_from_json_file("provider.json")
deepseek_interface = provider_interfaces["volc_engine"]["deepseek-v3-250324"]

# 在装饰器中使用
@llm_function(llm_interface=deepseek_interface)
def my_function():
    pass
```

### provider.json 配置示例

```json
{
    "volc_engine": {
        "deepseek-v3-250324": {
            "api_keys": ["your-api-key"],
            "base_url": "https://api.volc.example.com/v1",
            "model": "deepseek-chat"
        }
    }
}
```

SimpleLLMFunc的LLM接口设计原则：

- 简单统一的接口定义
- 支持普通和流式两种调用模式
- 支持自动的 API Key 负载均衡
- 完整的类型提示支持

## 日志系统

SimpleLLMFunc包含强大的日志系统，融合了结构化日志、自动追踪和聚合分析的能力：

### 1. 基本特性

- 多级别日志支持（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- 自动记录代码位置和执行环境信息
- JSON格式文件日志，便于程序化分析
- 彩色控制台输出，提升可读性

### 2. 智能日志关联

每个 LLM 函数调用会自动生成唯一的 `trace_id`，例如：`GLaDos_c790a5cc-e629-4cbd-b454-ab102c42d125`。这个ID会关联该调用产生的所有日志，包括：

- 函数调用的输入参数
- LLM请求和响应内容
- 工具调用记录
- 错误和警告信息
- Token usage statistics
- 执行时间和性能数据(Not Supported Yet)

### 3. 自动日志聚合

所有日志会被自动整理到 `log_indices/trace_index.json`，按 trace_id 分类聚合。这意味着：

- 可以轻松查看某次调用的完整执行流程
- 方便进行问题诊断和性能分析
- 有助于Prompt调优和工作流优化

### 日志使用示例

```python
from SimpleLLMFunc.logger import app_log, push_error, search_logs_by_trace_id, log_context

# 1. 基础日志记录
app_log("开始处理请求", trace_id="request_123")
push_error("发生错误", trace_id="request_123", exc_info=True)

# 2. 使用上下文管理器自动关联日志
with log_context(trace_id="task_456", function_name="analyze_text"):
    app_log("开始分析文本")  # 自动继承上下文的trace_id
    try:
        # 执行操作...
        app_log("分析完成")
    except Exception as e:
        push_error("分析失败", exc_info=True)  # 同样自动继承trace_id

# 3. 查看某次调用的所有相关日志
logs = search_logs_by_trace_id("GLaDos_c790a5cc-e629-4cbd-b454-ab102c42d125")
```

后续计划加入更多功能：
- LLM函数调用的性能指标面板
- 交互式日志分析工具
- 自动化Prompt优化建议

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
- **支持多模态返回**：工具可以返回 `ImgPath`（本地图片）或 `ImgUrl`（网络图片），实现多模态工具调用
- 当然，任何`llm_function`或者`llm_chat`装饰的函数，也可以接着被`tool`装饰器装饰以变成"智能工具"

### 多模态工具示例

```python
from SimpleLLMFunc.tool import tool
from SimpleLLMFunc.type import ImgPath, ImgUrl

@tool(name="generate_chart", description="根据数据生成图表")
def generate_chart(data: str, chart_type: str = "bar") -> ImgPath:
    """
    根据提供的数据生成图表
    
    Args:
        data: CSV格式的数据
        chart_type: 图表类型，默认为柱状图
        
    Returns:
        生成的图表文件路径
    """
    # 实际实现会生成图表并保存到本地
    chart_path = "./generated_chart.png"
    # ... 图表生成逻辑
    return ImgPath(chart_path)

@tool(name="search_web_image", description="搜索网络图片")
def search_web_image(query: str) -> ImgUrl:
    """
    搜索网络图片
    
    Args:
        query: 搜索关键词
        
    Returns:
        找到的图片URL
    """
    # 实际实现会调用图片搜索API
    image_url = "https://example.com/search_result.jpg"
    return ImgUrl(image_url)
```

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
from SimpleLLMFunc import llm_function

@llm_function(
    llm_interface=my_llm_interface,
    toolkit=[get_weather, search_web],  # 直接传递被@tool装饰的函数
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
    llm_interface=my_llm_interface,
    toolkit=[get_weather, WebSearchTool()],  # 混合使用两种方式定义的工具
)
def answer_with_mixed_tools(question: str) -> str:
    """回答用户问题，必要时使用工具获取信息"""
    pass
```

## API密钥管理

SimpleLLMFunc提供了完善的API密钥和流量管理机制：

### API密钥负载均衡
使用`APIKeyPool`类通过小根堆管理多个API密钥，实现负载均衡：

- 自动选择最少负载的API密钥
- 单例模式确保每个提供商只有一个密钥池，密钥池使用小根堆来进行负载均衡，每次取出load最低的KEY
- 自动跟踪每个密钥的使用情况

### 流量控制
集成了令牌桶算法（TokenBucket）实现智能流量平滑：

- 防止API调用频率过高触发限制
- 支持突发流量的缓冲处理
- 可在`provider.json`中配置每个模型的流量控制参数
- 与API密钥池协同工作，提供更稳定的服务

## 安装和使用

### 1. 源码安装
1. 克隆此仓库
2. 根据`env_template`创建`.env`文件并配置您的API密钥
3. 使用Poetry安装依赖：`poetry install`
4. 导入并使用`SimpleLLMFunc`的各个组件

### 2. PyPI安装

```bash
pip install SimpleLLMFunc
```

## Star History

<a href="https://www.star-history.com/#NiJingzhe/SimpleLLMFunc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=NiJingzhe/SimpleLLMFunc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=NiJingzhe/SimpleLLMFunc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=NiJingzhe/SimpleLLMFunc&type=Date" />
 </picture>
</a>

## Citation
如果您在研究或项目中使用了SimpleLLMFunc，请引用以下信息：

```bibtex
@software{Jingzhe_SimpleLLMFunc_2025,
  author = {Jingzhe Ni},
  month = {June},
  title = {{SimpleLLMFunc: A New Approach to Build LLM Applications}},
  url = {https://github.com/NiJingzhe/SimpleLLMFunc},
  version = {0.2.9},
  year = {2025}
}
```
## 许可证

MIT

