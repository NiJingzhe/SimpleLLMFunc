![cover](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/img/repocover_new.png?raw=true)

# 项目介绍

## SimpleLLMFunc 是什么?

SimpleLLMFunc 是一个轻量级的大语言模型（Large Language Model, LLM）应用开发框架，旨在简化 LLM 在应用中的集成过程。本框架的设计理念是“**Everything is Function, Prompt is Code**”，提供类型安全的装饰器，让开发者能以一种自然、直观的方式利用大语言模型的能力。

## 为什么需要 SimpleLLMFunc?

在开发基于大语言模型的应用时，我们常常面临以下挑战：

- 需要不断编写重复的 API 调用代码
- Prompt 作为字符串变量存在于代码中，不够直观
- 流程编排受到框架约束，缺乏灵活性
- 调试和监控 LLM 调用过程困难

SimpleLLMFunc 旨在解决这些问题，使得开发者可以：

- 用函数式编程的方式定义 LLM 能力
- 将 Prompt 直接放在函数的文档字符串中，使代码更加清晰
- 灵活构建应用流程，不受框架约束
- 获得全面的日志记录和调试信息

## 快速开始

下面是一个简单的示例，展示了 SimpleLLMFunc 的基本用法：

```python
from SimpleLLMFunc import llm_function, OpenAICompatible
from pydantic import BaseModel, Field
from typing import List

# 定义返回类型
class ProductAnalysis(BaseModel):
    pros: List[str] = Field(..., description="产品优点")
    cons: List[str] = Field(..., description="产品缺点")
    rating: int = Field(..., description="评分（1-5分）")

# 配置 LLM 接口
llm_interface = OpenAICompatible.load_from_json_file("provider.json")["provider"]["model"]

# 创建 LLM 函数
@llm_function(llm_interface=llm_interface)
def analyze_product(product_name: str, review: str) -> ProductAnalysis:
    """
    分析产品评论，提取优缺点并给出评分。
    
    Args:
        product_name: 产品名称
        review: 用户评论
        
    Returns:
        产品分析结果
    """
    pass  # Prompt as Code, Code as Doc

# 使用函数
result = analyze_product("无线耳机", "音质不错但连接不稳定")
print(f"优点: {result.pros}")
print(f"缺点: {result.cons}") 
print(f"评分: {result.rating}/5")
```

### 异步支持示例

```python
from SimpleLLMFunc import async_llm_function

@async_llm_function(llm_interface=llm_interface)
async def async_translate(text: str, target_language: str) -> str:
    """
    异步翻译文本到目标语言
    
    Args:
        text: 要翻译的文本
        target_language: 目标语言
        
    Returns:
        翻译结果
    """
    pass

# 异步调用
result = await async_translate("Hello world", "中文")
```

### 多模态支持示例

```python
from SimpleLLMFunc import llm_function
from SimpleLLMFunc.type import Text, ImgPath

@llm_function(llm_interface=llm_interface)
def analyze_image(description: Text, image: ImgPath) -> str:
    """
    分析图像内容
    
    Args:
        description: 分析要求描述
        image: 本地图片路径
        
    Returns:
        图像分析结果
    """
    pass

# 使用多模态输入
result = analyze_image(
    description=Text("描述这张图片中的主要内容"),
    image=ImgPath("./photo.jpg")
)
```

## 核心特性

- **装饰器驱动**: 使用 `@llm_function`、`@llm_chat` 及其异步版本装饰器轻松创建 LLM 驱动的功能。
- **DocString 即 Prompt**: 直接在函数文档中定义 Prompt，提高代码可读性。
- **类型安全**: 支持 Python 类型注解和 Pydantic 模型，确保数据结构正确。
- **异步支持**: 提供 `@async_llm_function` 和 `@async_llm_chat` 装饰器，支持原生异步调用。
- **多模态支持**: 支持文本、图片URL和本地图片路径的多模态输入处理，同时创新性支持工具的多模态返回。
- **通用模型接口**: 兼容任何符合 OpenAI API 格式的模型服务，并且定义了 LLM Interface 抽象类，便于扩展。
- **API 密钥管理**: 智能负载均衡多个 API 密钥。
- **流量控制**: 集成令牌桶算法，实现智能流量平滑。
- **工具系统**: 支持 LLM tool use，具有简单易用的工具定义和调用机制，支持多模态工具返回。
- **日志完备**: 支持 `trace_id` 跟踪和搜索，方便调试和监控。

## 项目架构

SimpleLLMFunc 的目录结构如下：

```
SimpleLLMFunc/
├── __init__.py                     # 包初始化文件
├── config.py                       # 全局配置
├── utils.py                        # 通用工具函数
├── interface/                      # LLM 接口
│   ├── __init__.py                 # 包初始化文件
│   ├── llm_interface.py            # LLM 接口抽象类
│   ├── key_pool.py                 # API 密钥池
│   ├── openai_compatible.py        # OpenAI 兼容接口实现
│   └── token_bucket.py             # 流量控制令牌桶实现
├── llm_decorator/                  # LLM装饰器
│   ├── __init__.py                 # 包初始化文件
│   ├── llm_function_decorator.py   # 函数装饰器
│   ├── llm_chat_decorator.py       # 对话装饰器
│   ├── multimodal_types.py         # 多模态类型定义
│   └── utils.py                    # 装饰器工具函数
├── logger/                         # 日志系统
│   ├── __init__.py                 # 包初始化文件
│   ├── logger.py                   # 日志核心功能
│   └── logger_config.py            # 日志配置
├── tool/                           # 工具系统
│   ├── __init__.py                 # 包初始化文件
│   └── tool.py                     # 工具定义和装饰器
└── type/                           # 类型定义
    └── __init__.py                 # 多模态类型导出
```

### 模块介绍

#### LLM 接口模块

`interface` 模块提供了与各种 LLM 服务通信的标准接口。它支持任何符合 OpenAI API 格式的服务，包括 OpenAI 自身、Azure OpenAI、各种开源模型的兼容 API 等。新增的 `token_bucket.py` 提供了流量控制功能，防止API调用频率过高。

#### LLM 装饰器模块

`llm_decorator` 模块是框架的核心，提供了四种主要装饰器：

- `@llm_function`: 用于创建无状态的 LLM 功能，适合单次查询和转换任务
- `@llm_chat`: 用于创建对话式 LLM 功能，支持历史记录管理和多轮交互
- `@async_llm_function`: `@llm_function` 的异步版本，支持原生异步调用
- `@async_llm_chat`: `@llm_chat` 的异步版本，支持原生异步对话

该模块还包含 `multimodal_types.py`，定义了 `Text`、`ImgUrl`、`ImgPath` 等多模态类型，支持处理文本和图像的混合输入。

#### 类型定义模块

`type` 模块专门用于导出多模态类型定义，使开发者可以方便地使用类型标注来创建支持多模态输入的 LLM 函数。

#### 日志系统

`logger` 模块提供了全面的日志记录功能，包括 trace_id 跟踪、token 使用统计、系统和用户提示的记录等。特别地，日志系统会自动记录所有 LLM 的输入输出对话，生成结构化的 trace 索引文件以按照函数调用归类日志，开发者可以直接从这些日志中快速整理出高质量的对话语料，用于后续的模型微调和优化。

#### 工具系统

`tool` 模块允许 LLM 访问外部工具和服务，增强其解决问题的能力。工具可以是任何 Python 函数，通过 `@tool` 装饰器进行标记。该模块现在支持多模态工具返回，工具可以返回图片、文本或其组合。

## 与其他框架的比较

相比其他 LLM 开发框架，SimpleLLMFunc 具有以下优势：

- **更轻量级**: 核心依赖少，启动快速，学习曲线平缓
- **更直观**: Prompt 直接嵌入函数文档，而非单独的变量
- **更灵活**: 不强制使用特定的框架结构，可以自由组合功能
- **更易用**: 装饰器语法简洁明了，减少样板代码
- **更易调试**: 完整的日志系统，便于追踪问题和优化 Prompt
- **原生异步**: 提供真正的异步支持，而非基于线程池的伪异步
- **多模态友好**: 原生支持文本、图像等多模态输入和输出
- **智能流控**: 集成令牌桶算法，自动处理API限流问题
- **类型安全**: 完整的类型提示支持，IDE 友好，减少运行时错误

## 适用场景

SimpleLLMFunc 特别适合以下场景：

- **快速原型开发**: 需要快速验证 LLM 应用想法
- **企业级应用**: 需要稳定、可维护的 LLM 集成方案
- **多模态应用**: 需要处理文本和图像混合输入的场景
- **高并发场景**: 需要异步处理大量 LLM 请求
- **调试优化**: 需要详细日志追踪和 Prompt 优化的场景
