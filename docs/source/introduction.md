![cover](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/img/repocover_new.png?raw=true)

# 项目介绍

## SimpleLLMFunc 是什么?

SimpleLLMFunc 是一个轻量级的大语言模型（Large Language Model, LLM）应用开发框架，旨在简化 LLM 在应用中的集成过程。本框架的设计理念是“**LLM as Function, Prompt as Code**”，提供类型安全的装饰器，让开发者能以一种自然、直观的方式利用大语言模型的能力。

## 为什么需要 SimpleLLMFunc?

在开发基于大语言模型的应用时，我们常常面临以下挑战：

- 需要不断编写重复的 API 调用代码
- Prompt 作为字符串变量存在于代码中，不够直观
- 流程编排受到框架约束，缺乏灵活性
- 调试和监控 LLM 调用过程困难

SimpleLLMFunc 旨在解决这些问题，使得开发者可以：

- 装饰器驱动：提供 `@llm_function`、`@llm_chat` 等装饰器，所有装饰器仅支持异步函数 (`async def`) 并原生适配异步调用。
- Prompt 即逻辑：Prompt 就是代码，是这个函数的逻辑实现。
- 类型安全：支持 Python 类型注解和 Pydantic 模型，确保数据结构正确。
- 多模态支持：支持文本、图片 URL 和本地图片路径的混合输入，创新性地支持工具的多模态返回。
- 通用模型接口：兼容任何符合 OpenAI API 格式的模型服务，易于扩展。
- API 密钥管理：智能负载均衡多个 API 密钥。
- 流量控制：集成令牌桶算法，实现智能流量平滑。
- 工具系统：支持 LLM 工具使用，具有简单易用的工具定义和调用机制，支持多模态工具返回。
- 日志完备：支持 trace_id 跟踪和搜索，方便调试和监控。

| 特性           | SimpleLLMFunc | LangChain | Dify |
| -------------- | :-----------: | :-------: | :--: |
| 易用性（学习曲线） |      ✅       |     ❌     |  ✅  |
| 直观性         |      ✅       |     ❌     |  ⭕️  |
| 灵活性         |      ✅       |     ✅     |  ⭕️  |
| 开发速度       |      ✅       |     ❌     |  ✅  |
| 调试性         |      ✅       |     ❌     |  ✅  |
| 异步支持       |      ✅       |     ✅     |  ⭕️  |
| 多模态支持     |      ✅       |     ⭕️     |  ⭕️  |
| 流量控制       |      ✅       |     ⭕️     |  ⭕️  |
| 类型安全       |      ✅       |     ⭕️     |  ❌  |
| 工具集成       |      ✅🌟      |     ✅     |  ✅  |
| 社区与生态系统 |      ⭕️       |     ✅     |  ✅  |

## 样例展示 

下面是一个简单的示例，展示了 SimpleLLMFunc 的基本用法：

> ⚠️ SimpleLLMFunc 中的 `@llm_function`、`@llm_chat`、`@tool` 等装饰器只能装饰 `async def` 定义的函数，请在异步上下文中通过 `await` 调用。

```python
import asyncio
from typing import List

from pydantic import BaseModel, Field

from SimpleLLMFunc import llm_function, OpenAICompatible

# 定义返回类型
class ProductAnalysis(BaseModel):
    pros: List[str] = Field(..., description="产品优点")
    cons: List[str] = Field(..., description="产品缺点")
    rating: int = Field(..., description="评分（1-5分）")

# 配置 LLM 接口
llm_interface = OpenAICompatible.load_from_json_file("provider.json")["provider"]["model"]

# 创建 LLM 函数
@llm_function(llm_interface=llm_interface)
async def analyze_product(product_name: str, review: str) -> ProductAnalysis:
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


async def main():
    result = await analyze_product("无线耳机", "音质不错但连接不稳定")
    print(f"优点: {result.pros}")
    print(f"缺点: {result.cons}")
    print(f"评分: {result.rating}/5")


asyncio.run(main())
```

### 异步支持示例

```python
import asyncio

from SimpleLLMFunc import llm_function


@llm_function(llm_interface=llm_interface)
async def async_translate(text: str, target_language: str) -> str:
    """
    将输入文本翻译为目标语言。

    Args:
        text: 要翻译的文本
        target_language: 目标语言

    Returns:
        翻译结果
    """
    pass


async def main():
    result = await async_translate("Hello world", "中文")
    print(result)


asyncio.run(main())
```

### 多模态支持示例

```python
import asyncio

from SimpleLLMFunc import llm_function
from SimpleLLMFunc.type import Text, ImgPath

@llm_function(llm_interface=llm_interface)
async def analyze_image(description: Text, image: ImgPath) -> str:
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
async def run():
    result = await analyze_image(
        description=Text("描述这张图片中的主要内容"),
        image=ImgPath("./photo.jpg")
    )
    print(result)


asyncio.run(run())
```

### 动态模板参数示例

```python
import asyncio

from SimpleLLMFunc import llm_function

# 万能的代码分析函数
@llm_function(llm_interface=llm_interface)
async def analyze_code(code: str) -> str:
    """以{style}的方式分析{language}代码，重点关注{focus}。"""
    pass


# 万能的文本处理函数
@llm_function(llm_interface=llm_interface)
async def process_text(text: str) -> str:
    """作为{role}，请{action}以下文本，输出风格为{style}。"""
    pass


async def main():
    # 不同的调用方式，适应不同场景
    performance_analysis = await analyze_code(
        python_code,
        _template_params={
            'style': '详细',
            'language': 'Python',
            'focus': '性能优化'
        }
    )

    code_review = await analyze_code(
        js_code,
        _template_params={
            'style': '简洁',
            'language': 'JavaScript',
            'focus': '代码规范'
        }
    )

    # 同一个函数，不同角色
    edited_text = await process_text(
        text,
        _template_params={
            'role': '专业编辑',
            'action': '润色',
            'style': '学术'
        }
    )

    translated_text = await process_text(
        text,
        _template_params={
            'role': '翻译专家',
            'action': '翻译成英文',
            'style': '商务'
        }
    )

    print(performance_analysis, code_review, edited_text, translated_text)


asyncio.run(main())
```

## 核心特性

- **装饰器驱动**: 使用 `@llm_function`、`@llm_chat` 构建 LLM 驱动的功能，均为原生异步实现。
- **DocString 即 Prompt**: 直接在函数文档中定义 Prompt，提高代码可读性。
- **动态模板参数**: 支持通过 `_template_params` 在函数调用时动态设置 DocString 模板参数，让一个函数适应多种场景。
- **类型安全**: 支持 Python 类型注解和 Pydantic 模型，确保数据结构正确。
- **异步支持**: `@llm_function` 与 `@llm_chat` 原生支持异步调用，无需额外别名。
- **多模态支持**: 支持文本、图片URL和本地图片路径的多模态输入处理，同时创新性支持工具的多模态返回。
- **步骤化装饰器流水线**：`llm_decorator/steps` 将 Prompt 构建、签名解析、ReAct 和响应解析拆分为可组合的步骤，便于调试与扩展。
- **基础引擎模块化**：`base/messages`、`base/tool_call`、`base/type_resolve` 独立演进，类型解析和多模态处理更稳健。
- **通用模型接口**: 兼容任何符合 OpenAI API 格式的模型服务，并且定义了 LLM Interface 抽象类，便于扩展。
- **API 密钥管理**: 智能负载均衡多个 API 密钥。
- **流量控制**: 集成令牌桶算法，实现智能流量平滑。
- **工具系统**: 支持 LLM tool use，具有简单易用的工具定义和调用机制，支持多模态工具返回。
- **日志完备**: 支持 `trace_id` 跟踪和搜索，方便调试和监控。

## 项目架构

SimpleLLMFunc 的目录结构如下：

```
SimpleLLMFunc/
├── __init__.py                  # 包初始化
├── config.py                    # 全局配置
├── utils.py                     # 通用工具函数
├── base/                        # 核心执行引擎
│   ├── messages/                # 消息构建与多模态内容生成
│   ├── tool_call/               # 工具调用参数转换、执行与校验
│   ├── type_resolve/            # 类型描述、示例与多模态类型解析
│   ├── post_process.py          # 响应解析与类型转换
│   └── ReAct.py                 # ReAct 协调器
├── interface/                   # LLM 接口层
│   ├── llm_interface.py         # 抽象基类
│   ├── openai_compatible.py     # OpenAI 兼容实现
│   ├── key_pool.py              # API 密钥负载均衡
│   └── token_bucket.py          # 令牌桶流量控制
├── llm_decorator/               # 装饰器与步骤化流水线
│   ├── llm_function_decorator.py
│   ├── llm_chat_decorator.py
│   ├── multimodal_types.py
│   ├── steps/                   # Prompt/签名/执行/响应拆分
│   │   ├── common/
│   │   ├── function/
│   │   └── chat/
│   └── utils/tools.py
├── tool/                        # 工具定义与序列化
│   └── tool.py
├── type/                        # 类型与多模态辅助
│   ├── decorator.py             # 装饰器相关的类型定义
│   ├── message.py               # 消息片段类型
│   └── multimodal.py            # Text / ImgUrl / ImgPath 等
├── logger/                      # 日志与可观测性
├── observability/               # Langfuse 等集成
└── py.typed
```

### 模块介绍

#### LLM 接口模块

`interface` 模块提供了与各种 LLM 服务通信的标准接口，支持任何符合 OpenAI API 格式的服务，包括 OpenAI/Deepseek/Claude 及自建兼容 API。`token_bucket.py` 负责流量控制，`key_pool.py` 负责密钥负载均衡。

#### LLM 装饰器模块

`llm_decorator` 模块是框架的核心，提供 `@llm_function` 与 `@llm_chat`，并在 `steps/` 中将 Prompt 构建、签名解析、ReAct 执行、响应解析等环节拆分为可组合步骤。`multimodal_types.py` 定义了 `Text`、`ImgUrl`、`ImgPath` 等多模态类型，支持处理文本和图片混合输入。

#### 类型定义模块

`type` 模块专门用于导出多模态类型定义，使开发者可以方便地使用类型标注来创建支持多模态输入的 LLM 函数。

#### 日志系统

`logger` 模块提供了全面的日志记录功能，包括 trace_id 跟踪、token 使用统计、系统和用户提示的记录等。特别地，日志系统会自动记录所有 LLM 的输入输出对话，生成结构化的 trace 索引文件以按照函数调用归类日志，开发者可以直接从这些日志中快速整理出高质量的对话语料，用于后续的模型微调和优化。

#### 工具系统

`tool` 模块允许 LLM 访问外部工具和服务，工具通过 `@tool` 装饰器标记，并支持多模态返回（文本、图片或组合）。

## 适用人群

SimpleLLMFunc 特别适合以下朋友：

- **LLM应用开发的入门创客玩家**: 学习曲线平缓，内容简单，快速上手，直观易懂。
- **快速原型开发的创业者**: 需要快速验证 LLM 应用想法，缩短开发周期和迭代时间。
- **会Python的PM**: 需要快速实现 LLM 应用原型，验证产品想法。

当然我们也欢迎任何一位对 LLM 应用开发感兴趣的小白，老手或者专家加入我们的社区，一起探索 LLM 应用的无限可能！
