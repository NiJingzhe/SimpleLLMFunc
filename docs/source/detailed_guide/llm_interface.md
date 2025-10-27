# LLM 接口层

本文档介绍 SimpleLLMFunc 的 LLM 接口层设计，包括接口抽象、密钥管理和流量控制等核心组件。

## 概述

SimpleLLMFunc 提供了完整的 LLM 接口框架，旨在统一不同 LLM 服务提供商的调用方式。该框架包含三个核心组件：

1. **LLM_Interface** - 抽象基类，定义标准接口规范
2. **APIKeyPool** - API 密钥池，实现负载均衡
3. **TokenBucket** - 令牌桶算法，提供流量控制

## 快速开始

### 最简单的使用方式

```python
from SimpleLLMFunc import OpenAICompatible

# 从配置文件加载 LLM 接口
llm = OpenAICompatible.load_from_json_file("provider.json")["openai"]["gpt-3.5-turbo"]

# 现在可以在装饰器中使用
from SimpleLLMFunc import llm_function

@llm_function(llm_interface=llm)
async def my_task(text: str) -> str:
    """处理文本任务"""
    pass
```

### 配置文件格式

创建 `provider.json` 文件：

```json
{
  "openai": {
    "gpt-3.5-turbo": {
      "api_keys": ["sk-key1", "sk-key2"],
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-3.5-turbo",
      "max_retries": 5,
      "retry_delay": 1.0,
      "rate_limit_capacity": 20,
      "rate_limit_refill_rate": 3.0
    },
    "gpt-4": {
      "api_keys": ["sk-key3"],
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4"
    }
  },
  "deepseek": {
    "v3-turbo": {
      "api_keys": ["your-deepseek-key"],
      "base_url": "https://api.deepseek.com/v1",
      "model": "deepseek-chat"
    }
  }
}
```

## LLM_Interface 基类

### 设计理念

`LLM_Interface` 是所有 LLM 实现的抽象基类，定义了统一的接口规范。所有具体实现（如 `OpenAICompatible`）都继承自这个类。

### 核心特性

- **标准化接口**: 定义了 `chat()` 和 `chat_stream()` 两个核心方法
- **类型安全**: 使用 Python 类型注解确保类型正确性
- **异步原生**: 完全基于异步编程，支持高并发
- **可扩展性**: 通过继承轻松添加新的 LLM 服务支持

### 接口定义

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, List

class LLM_Interface(ABC):
    """LLM 接口抽象基类"""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        timeout: int | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行非流式对话请求

        Args:
            messages: 消息列表，每条消息为 {"role": "user"|"assistant"|"system", "content": "..."}
            stream: 流式标志（False 表示非流式）
            timeout: 请求超时时间（秒）
            **kwargs: 其他 LLM 参数（temperature、top_p 等）

        Returns:
            LLM 响应数据
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        timeout: int | None = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行流式对话请求

        Args:
            messages: 消息列表
            stream: 流式标志（True 表示流式）
            timeout: 请求超时时间（秒）
            **kwargs: 其他 LLM 参数

        Yields:
            LLM 响应数据块
        """
        if False:
            yield {}
```

## OpenAICompatible 实现

### 什么是 OpenAICompatible

`OpenAICompatible` 是 `LLM_Interface` 的具体实现，支持任何兼容 OpenAI API 格式的服务，包括：

- OpenAI (GPT-4, GPT-3.5)
- Deepseek
- Anthropic Claude
- 火山引擎 Ark
- 百度千帆
- 本地 LLM (Ollama, vLLM)
- 任何兼容 OpenAI API 的服务

### 创建接口

#### 方式 1: 从配置文件加载（推荐）

```python
from SimpleLLMFunc import OpenAICompatible

# 从 JSON 配置文件加载
all_models = OpenAICompatible.load_from_json_file("provider.json")

# 访问具体的模型
gpt35 = all_models["openai"]["gpt-3.5-turbo"]
gpt4 = all_models["openai"]["gpt-4"]
deepseek = all_models["deepseek"]["v3-turbo"]
```

#### 方式 2: 直接创建

```python
from SimpleLLMFunc import OpenAICompatible, APIKeyPool

# 创建 API 密钥池
key_pool = APIKeyPool(
    api_keys=["sk-key1", "sk-key2", "sk-key3"],
    provider_id="openai-gpt35"
)

# 创建 LLM 接口
llm = OpenAICompatible(
    api_key_pool=key_pool,
    base_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo",
    max_retries=5,
    retry_delay=1.0,
    rate_limit_capacity=20,
    rate_limit_refill_rate=3.0
)
```

### 核心特性

- **自动重试**: 请求失败时自动重试（可配置次数）
- **令牌统计**: 自动统计 input 和 output token 数量
- **流量控制**: 使用令牌桶算法防止速率限制
- **密钥轮换**: 自动在多个密钥间负载均衡

## APIKeyPool - 密钥管理

### 设计理念

`APIKeyPool` 使用**小根堆**数据结构实现 API 密钥的负载均衡，确保请求均匀分布到不同的密钥上。

### 核心特性

- **小根堆轮换**: 自动选择负载最低的密钥
- **负载均衡**: 实时跟踪每个密钥的任务数量
- **线程安全**: 使用锁保护并发访问
- **单例模式**: 相同 provider_id 的密钥池共享状态

### 工作原理

```
堆结构：[(任务数, API密钥), ...]
堆顶永远是任务数最少的密钥

示例：
heap = [(0, "key1"), (1, "key2"), (2, "key3")]
获取最低负载密钥 → "key1"
key1 任务完成后重新整理堆...
```

### 使用示例

```python
from SimpleLLMFunc.interface import APIKeyPool

# 创建密钥池
key_pool = APIKeyPool(
    api_keys=["sk-key1", "sk-key2", "sk-key3"],
    provider_id="my-provider"
)

# 获取最低负载的密钥
key = key_pool.get_least_loaded_key()

# 任务开始时增加计数
key_pool.increment_task_count(key)

# 任务完成后减少计数
key_pool.decrement_task_count(key)
```

## TokenBucket - 流量控制

### 设计理念

`TokenBucket` 实现经典的**令牌桶算法**，用于 API 请求的流量控制和速率限制。

### 算法原理

令牌桶工作原理：

1. **令牌生成**: 以固定速率向桶中添加令牌
2. **令牌消费**: 每次请求消费一个或多个令牌
3. **容量限制**: 桶有最大容量，多余令牌会被丢弃
4. **突发支持**: 当桶中有足够令牌时，允许突发请求

示例：
```
容量=5，补充速率=2 tokens/秒

时间: 0s      1s      2s      3s
桶:  [●●●●●] [●●●●●] [●●●●●] [●●●]
请求:  0个    0个    2个    2个
```

### 核心特性

- **平滑流量**: 避免突发请求冲击后端 API
- **可配置参数**: 支持自定义容量和补充速率
- **异步支持**: 非阻塞的令牌获取，支持超时
- **线程安全**: 统一的锁保护所有操作

### 配置参数

| 参数 | 类型 | 说明 | 推荐值 |
|------|------|------|--------|
| `capacity` | int | 令牌桶容量（最大令牌数） | 10-50 |
| `refill_rate` | float | 令牌补充速率（tokens/秒） | 0.5-5.0 |

### 使用场景

```python
# 高频 API（如 OpenAI）
capacity=20, refill_rate=3.0

# 标准 API
capacity=10, refill_rate=1.0

# 受限 API（如百度千帆）
capacity=5, refill_rate=0.5
```

## 完整的生产级示例

### 1. 多模型配置

```python
from SimpleLLMFunc import OpenAICompatible, llm_function

# 加载多个模型
models = OpenAICompatible.load_from_json_file("provider.json")

# 定义不同的任务用不同的模型
fast_llm = models["openai"]["gpt-3.5-turbo"]  # 快速、便宜
powerful_llm = models["openai"]["gpt-4"]      # 强大、昂贵
deepseek_llm = models["deepseek"]["v3-turbo"] # 国内服务

@llm_function(llm_interface=fast_llm)
async def simple_task(text: str) -> str:
    """简单任务使用快速模型"""
    pass

@llm_function(llm_interface=powerful_llm)
async def complex_task(text: str) -> str:
    """复杂任务使用强大模型"""
    pass
```

### 2. 错误处理和重试

```python
import asyncio
from SimpleLLMFunc import llm_function

@llm_function(llm_interface=llm)
async def robust_call(text: str) -> str:
    """可靠的 LLM 调用"""
    pass

async def call_with_fallback():
    """带备选方案的调用"""
    try:
        result = await robust_call("test")
        return result
    except Exception as e:
        print(f"主模型失败: {e}")
        # 使用备选模型
        return await backup_call("test")
```

### 3. 监控和调试

```python
# 检查 token 使用量
print(f"输入 tokens: {llm.input_token_count}")
print(f"输出 tokens: {llm.output_token_count}")

# 检查密钥池状态
least_loaded = llm.key_pool.get_least_loaded_key()
print(f"最低负载密钥: {least_loaded}")

# 检查令牌桶状态
print(f"可用令牌: {llm.token_bucket.tokens}")
```

## 最佳实践

### 1. 密钥管理

```python
# ✅ 推荐：为每个项目/环境配置独立的密钥
{
  "openai": {
    "prod": {
      "api_keys": ["prod-key1", "prod-key2"],
      "max_retries": 5
    },
    "dev": {
      "api_keys": ["dev-key1"],
      "max_retries": 2
    }
  }
}

# ✅ 推荐：为不同模型配置不同的速率限制
{
  "openai": {
    "gpt-4": {
      "rate_limit_capacity": 10,  # 小容量，保守
      "rate_limit_refill_rate": 1.0
    },
    "gpt-3.5-turbo": {
      "rate_limit_capacity": 50,  # 大容量，激进
      "rate_limit_refill_rate": 5.0
    }
  }
}
```

### 2. 流量控制

```python
# ✅ 推荐：根据 API 限制调整参数
# OpenAI：rate_limit_capacity=10-20, refill_rate=1.0-3.0
# Deepseek：rate_limit_capacity=10-15, refill_rate=2.0-3.0
# 本地 LLM：capacity=100, refill_rate=10.0（几乎无限）
```

### 3. 错误处理

```python
import asyncio
from typing import Optional

async def call_with_exponential_backoff(
    llm_call,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> Optional[str]:
    """带指数退避的重试"""
    for attempt in range(max_retries):
        try:
            return await llm_call()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"第 {attempt + 1} 次失败，等待 {delay} 秒后重试...")
            await asyncio.sleep(delay)
```

## 故障排除

### 常见问题

1. **"Rate limit exceeded" 错误**
   - 增加 `rate_limit_capacity` 或 `rate_limit_refill_rate`
   - 检查配置中的速率限制是否与 API 限制匹配

2. **密钥持续失败**
   - 检查 API 密钥是否有效且有足够配额
   - 验证 `base_url` 是否正确

3. **Token 统计不准确**
   - 某些 API 可能不返回 token 统计信息
   - 框架会尽力估算，但可能不完全准确

### 调试技巧

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("SimpleLLMFunc")
logger.setLevel(logging.DEBUG)

# 查看请求/响应详情
```

## 总结

SimpleLLMFunc 的接口层通过以下设计提供强大的 LLM 服务：

1. **LLM_Interface**: 统一的抽象基类
2. **OpenAICompatible**: 开箱即用的 OpenAI 兼容实现
3. **APIKeyPool**: 智能的密钥负载均衡
4. **TokenBucket**: 可靠的流量控制

这种设计既保证了易用性，又提供了企业级的功能和可靠性。
