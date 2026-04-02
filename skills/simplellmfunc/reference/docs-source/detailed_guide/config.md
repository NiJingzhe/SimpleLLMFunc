# 配置文件说明

## `.env` 文件

`.env` 用于存放环境变量，本框架主要读取日志配置与 Langfuse 配置。你可以在项目的工作目录创建 `.env`，也可以直接通过系统环境变量覆盖。

### 日志相关环境变量

```bash
# 日志级别（默认 DEBUG）
# 可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=DEBUG

# 日志目录（默认 logs）
LOG_DIR=logs
```

### 支持的环境变量

| 环境变量 | 说明 | 可选值 | 默认值 |
|---------|------|--------|--------|
| `LOG_LEVEL` | 控制台与文件日志级别 | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `DEBUG` |
| `LOG_DIR` | 日志输出目录 | 任意路径 | `logs` |

### 环境变量优先级

优先级顺序（从高到低）：

1. 运行时环境变量（如 `export LOG_LEVEL=INFO`）
2. `.env` 文件中的配置
3. 框架默认值

## `provider.json` 文件

`provider.json` 用于配置模型提供商、API 密钥与限流参数。`OpenAICompatible.load_from_json_file(...)` 读取此文件并返回一个二维字典：`providers[provider_id][model_name]`。

### 配置文件结构

`provider.json` 采用 **提供商 -> 模型配置列表** 的结构，每个模型配置包含 `model_name` 与密钥等参数：

```json
{
  "openai": [
    {
      "model_name": "gpt-3.5-turbo",
      "api_keys": ["sk-test-key-1", "sk-test-key-2"],
      "base_url": "https://api.openai.com/v1",
      "max_retries": 5,
      "retry_delay": 1.0,
      "rate_limit_capacity": 20,
      "rate_limit_refill_rate": 3.0
    },
    {
      "model_name": "gpt-4",
      "api_keys": ["sk-test-key-3"],
      "base_url": "https://api.openai.com/v1",
      "max_retries": 5,
      "retry_delay": 1.0,
      "rate_limit_capacity": 10,
      "rate_limit_refill_rate": 1.0
    }
  ],
  "zhipu": [
    {
      "model_name": "glm-4",
      "api_keys": ["zhipu-test-key-1", "zhipu-test-key-2"],
      "base_url": "https://open.bigmodel.cn/api/paas/v4/",
      "max_retries": 3,
      "retry_delay": 0.5,
      "rate_limit_capacity": 15,
      "rate_limit_refill_rate": 2.0
    }
  ]
}
```

### 配置参数说明

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `model_name` | 字符串 | 模型名称（作为索引 key） | `gpt-3.5-turbo` |
| `api_keys` | 数组 | API 密钥列表（支持负载均衡） | `["key1", "key2"]` |
| `base_url` | 字符串 | API 服务地址 | `https://api.openai.com/v1` |
| `max_retries` | 数字 | 最大重试次数 | `5` |
| `retry_delay` | 浮点数 | 重试间隔（秒） | `1.0` |
| `rate_limit_capacity` | 数字 | 令牌桶容量 | `20` |
| `rate_limit_refill_rate` | 浮点数 | 令牌补充速率（tokens/秒） | `3.0` |

### 加载和使用

```python
from SimpleLLMFunc import OpenAICompatible, llm_function

models = OpenAICompatible.load_from_json_file("provider.json")
gpt35 = models["openai"]["gpt-3.5-turbo"]

@llm_function(llm_interface=gpt35)
async def my_task(text: str) -> str:
    """处理文本的任务"""
    pass
```

### 最佳实践

1. **多密钥负载均衡**：同一模型配置多个 key，避免单 key 限流。
2. **按模型调整限流**：高成本模型设置更保守的 `rate_limit_capacity` 与 `rate_limit_refill_rate`。
3. **避免重复 `model_name`**：同一 provider 内的 `model_name` 将作为索引键，重复会覆盖。
