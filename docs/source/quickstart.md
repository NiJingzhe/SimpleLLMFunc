# 快速开始

欢迎使用 SimpleLLMFunc！本指南将帮助您从零开始，快速搭建并运行您的第一个 LLM 应用。

## 环境要求

- Python 3.10 或更高版本
- 支持的操作系统：Windows、macOS、Linux

## 第一步：环境准备

### 1.1 创建虚拟环境

推荐使用虚拟环境来管理项目依赖：

```bash
# 使用 venv 创建虚拟环境
python -m venv simplellmfunc_env

# 激活虚拟环境
# Windows
simplellmfunc_env\Scripts\activate
# macOS/Linux
source simplellmfunc_env/bin/activate
```

### 1.2 安装 SimpleLLMFunc

有两种安装方式：

**方式一：从 PyPI 安装（推荐）**
```bash
pip install SimpleLLMFunc
```

**方式二：从源码安装**
```bash
# 克隆项目
git clone https://github.com/NiJingzhe/SimpleLLMFunc.git
cd SimpleLLMFunc

# 使用 Poetry 安装依赖
poetry install

# 或者使用 pip 安装
pip install -e .
```

## 第二步：配置 API 密钥

### 2.1 创建配置文件

在项目根目录创建 `.env` 文件：

```bash
# 复制环境模板
cp env_template .env
```

编辑 `.env` 文件，配置日志设置：

```bash
LOG_LEVEL=WARNING
```

### 2.2 创建模型配置文件

创建 `provider.json` 文件来配置您的 LLM 模型：

```json
{
    "openai": [
        {
            "model_name": "gpt-3.5-turbo",
            "api_keys": ["your-openai-api-key"],
            "base_url": "https://api.openai.com/v1"
        }
    ],
    "volc_engine": [
        {
            "model_name": "deepseek-v3-250324",
            "api_keys": ["your-volc-engine-api-key"],
            "base_url": "https://ark.cn-beijing.volces.com/api/v3/",
            "max_retries": 3,
            "retry_delay": 1
        }
    ]
}
```

**重要提示：**
- 请将 `openai` 和 `volc_engine` 替换为您的实际模型提供商
- 请将 `model_name` 替换为您的实际模型名称
- 请将 `your-openai-api-key` 和 `your-volc-engine-api-key` 替换为您的实际 API 密钥
- 支持多个 API 密钥，系统会自动进行负载均衡
- 可以配置多个不同的模型提供商

## 第三步：创建第一个 Demo

### 3.1 基础文本分析示例

创建一个名为 `first_demo.py` 的文件：

> ⚠️ SimpleLLMFunc 中的所有装饰器（如 `@llm_function`、`@llm_chat`、`@tool`）都要求装饰 `async def` 定义的函数；使用时请在异步上下文中通过 `await` 调用，或使用 `asyncio.run` 启动顶层任务。

```python
import asyncio
from typing import List
from pydantic import BaseModel, Field
from SimpleLLMFunc import llm_function, OpenAICompatible

# 定义返回类型
class TextAnalysis(BaseModel):
    sentiment: str = Field(..., description="情感分析结果：积极、消极或中性")
    keywords: List[str] = Field(..., description="提取的关键词列表")
    summary: str = Field(..., description="文本摘要")

# 加载模型接口
provider_interfaces = OpenAICompatible.load_from_json_file("provider.json")
gpt_interface = provider_interfaces["openai"]["gpt-3.5-turbo"]

# 创建 LLM 函数
@llm_function(llm_interface=gpt_interface)
async def analyze_text(text: str) -> TextAnalysis:
    """分析给定文本的情感、关键词和摘要
    
    Args:
        text: 需要分析的文本内容
        
    Returns:
        包含情感分析、关键词和摘要的结构化结果
    """
    pass

async def main():
    # 测试文本
    test_text = """
    今天天气非常好，阳光明媚，温度适宜。我决定去公园散步，
    看到很多人在户外活动，孩子们在玩耍，老人们在下棋。
    整个公园充满了生机和活力，让人心情愉悦。
    """
    
    print("=== 文本分析 Demo ===")
    print(f"输入文本: {test_text.strip()}")
    print("\n分析结果:")
    
    try:
        result = await analyze_text(test_text)
        print(f"情感: {result.sentiment}")
        print(f"关键词: {', '.join(result.keywords)}")
        print(f"摘要: {result.summary}")
    except Exception as e:
        print(f"分析失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3.2 运行 Demo

```bash
python first_demo.py
```

您应该看到类似以下的输出：

```
=== 文本分析 Demo ===
输入文本: 今天天气非常好，阳光明媚，温度适宜。我决定去公园散步，看到很多人在户外活动，孩子们在玩耍，老人们在下棋。整个公园充满了生机和活力，让人心情愉悦。

分析结果:
情感: 积极
关键词: 天气, 阳光, 公园, 散步, 户外活动, 生机, 活力
摘要: 描述了一个阳光明媚的日子，作者在公园散步时看到各种户外活动，感受到生机和活力，心情愉悦。
```

## 第四步：进阶示例

### 4.1 动态模板参数示例

这个示例展示了如何使用 `_template_params` 让同一个函数适应不同的使用场景：

```python
import asyncio

from SimpleLLMFunc import llm_function, OpenAICompatible

# 加载模型接口
provider_interfaces = OpenAICompatible.load_from_json_file("provider.json")
gpt_interface = provider_interfaces["openai"]["gpt-3.5-turbo"]

# 万能的代码分析函数
@llm_function(llm_interface=gpt_interface)
async def analyze_code(code: str) -> str:
    """以{style}的方式分析{language}代码，重点关注{focus}。"""
    pass

# 万能的文本处理函数
@llm_function(llm_interface=gpt_interface)
async def process_text(text: str) -> str:
    """作为{role}，请{action}以下文本，输出风格为{style}。"""
    pass

async def main():
    print("=== 动态模板参数 Demo ===")
    
    # 测试代码
    python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    # 不同的分析方式
    print("\n1. 代码分析 - 性能优化:")
    try:
    performance_result = await analyze_code(
            python_code,
            _template_params={
                'style': '详细',
                'language': 'Python',
                'focus': '性能优化'
            }
        )
        print(performance_result)
    except Exception as e:
        print(f"分析失败: {e}")
    
    print("\n2. 代码分析 - 代码规范:")
    try:
    style_result = await analyze_code(
            python_code,
            _template_params={
                'style': '简洁',
                'language': 'Python',
                'focus': '代码规范'
            }
        )
        print(style_result)
    except Exception as e:
        print(f"分析失败: {e}")
    
    # 不同的文本处理角色
    sample_text = "人工智能技术正在快速发展，对各行各业产生深远影响。"
    
    print("\n3. 文本处理 - 编辑润色:")
    try:
    edited_result = await process_text(
            sample_text,
            _template_params={
                'role': '专业编辑',
                'action': '润色',
                'style': '学术'
            }
        )
        print(edited_result)
    except Exception as e:
        print(f"处理失败: {e}")
    
    print("\n4. 文本处理 - 翻译转换:")
    try:
    translated_result = await process_text(
            sample_text,
            _template_params={
                'role': '翻译专家',
                'action': '翻译成英文',
                'style': '商务'
            }
        )
        print(translated_result)
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

这个示例展示了动态模板参数的核心优势：
- **一个函数定义，多种使用场景**
- **调用时动态指定角色和任务**
- **代码复用性大大提高**

### 4.2 带工具调用的示例

创建一个更复杂的示例，展示工具调用功能：

```python
import asyncio
from typing import Dict
from pydantic import BaseModel, Field
from SimpleLLMFunc import llm_function, tool, OpenAICompatible

# 定义工具
@tool(name="get_weather", description="获取指定城市的天气信息")
async def get_weather(city: str) -> Dict[str, str]:
    """获取指定城市的天气信息
    
    Args:
        city: 城市名称
        
    Returns:
        包含温度、湿度和天气状况的字典
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temperature": "25°C", "humidity": "60%", "condition": "晴朗"},
        "上海": {"temperature": "28°C", "humidity": "75%", "condition": "多云"},
        "广州": {"temperature": "30°C", "humidity": "80%", "condition": "小雨"}
    }
    return weather_data.get(city, {"temperature": "20°C", "humidity": "50%", "condition": "未知"})

# 定义返回类型
class TravelRecommendation(BaseModel):
    city: str = Field(..., description="推荐的城市")
    weather_info: Dict[str, str] = Field(..., description="天气信息")
    activities: List[str] = Field(..., description="推荐的活动")
    reason: str = Field(..., description="推荐理由")

# 创建带工具调用的 LLM 函数
@llm_function(llm_interface=gpt_interface, toolkit=[get_weather])
async def recommend_travel(preference: str) -> TravelRecommendation:
    """根据用户偏好推荐旅游目的地和活动
    
    Args:
        preference: 用户的旅游偏好描述
        
    Returns:
        包含推荐城市、天气信息、活动和理由的结构化结果
    """
    pass

async def main():
    preference = "我喜欢温暖的气候，想要进行户外活动，最好是能看到美丽的风景"
    
    print("=== 旅游推荐 Demo ===")
    print(f"用户偏好: {preference}")
    print("\n推荐结果:")
    
    try:
        result = await recommend_travel(preference)
        print(f"推荐城市: {result.city}")
        print(f"天气信息: {result.weather_info}")
        print(f"推荐活动: {', '.join(result.activities)}")
        print(f"推荐理由: {result.reason}")
    except Exception as e:
        print(f"推荐失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 第五步：查看日志

SimpleLLMFunc 提供了强大的日志系统。运行 Demo 后，您可以：

1. 查看控制台输出的彩色日志
2. 检查 `./agent.log` 文件中的详细日志
3. 查看 `log_indices/trace_index.json` 中的结构化日志索引

每个函数调用都会生成唯一的 `trace_id`，方便追踪和调试。

## 常见问题

### Q: API 密钥配置错误怎么办？
A: 请检查 `provider.json` 文件中的 API 密钥是否正确，确保密钥有效且有足够的配额。

### Q: 模型返回格式不正确怎么办？
A: SimpleLLMFunc 会自动重试，但如果小模型无法输出正确的 JSON 格式，建议使用更强大的模型如 GPT-4。

### Q: 如何添加更多工具？
A: 使用 `@tool` 装饰器定义工具函数，然后在 `@llm_function` 的 `toolkit` 参数中传入工具列表。

### Q: 如何实现异步调用？
A: SimpleLLMFunc 中的装饰器（如 `@llm_function`、`@llm_chat`、`@tool`）只支持 `async def` 定义的函数，因此需要在异步上下文中通过 `await` 调用，或在脚本入口使用 `asyncio.run`。

### Q: 什么是动态模板参数？
A: 动态模板参数允许您在函数调用时通过 `_template_params` 参数动态设置 DocString 中的占位符，让同一个函数适应不同的使用场景。

### Q: 如何使用动态模板参数？
A: 在 DocString 中使用 `{变量名}` 占位符，调用时通过 `_template_params` 传入变量值。例如：
```python
@llm_function(llm_interface=llm)
async def analyze_code(code: str) -> str:
    """以{style}的方式分析{language}代码，重点关注{focus}。"""
    pass

import asyncio


async def main():
    result = await analyze_code(code, _template_params={
        'style': '详细', 'language': 'Python', 'focus': '性能优化'
    })
    print(result)


asyncio.run(main())
```

## 下一步

恭喜！您已经成功运行了第一个 SimpleLLMFunc 应用。接下来您可以：

1. 探索更多示例代码（在 `examples/` 目录中）
2. 阅读详细文档了解高级功能
3. 尝试创建自己的 LLM 应用
4. 参与社区讨论和贡献

祝您使用愉快！
