# 配置文件说明

## `.env` 文件

`.env` 文件用于存储环境变量，在本框架中主要用于配置日志相关设置。你可以在你项目最终的 `WORKING DIR` 下创建一个 `.env` 文件，或者直接在环境变量中设置这些值。

### 环境变量配置

```bash
# 日志相关配置

# LOG_LEVEL：控制台日志级别，默认为 WARNING
# 可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=WARNING

# 其他可选日志配置
# LOG_FILE：日志文件路径（如果需要）
# LOG_FILE=./logs/app.log
```

### 支持的环境变量

| 环境变量 | 说明 | 可选值 | 默认值 |
|---------|------|--------|--------|
| `LOG_LEVEL` | 日志级别 | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `WARNING` |

### 环境变量优先级

注意，直接 `export` 环境变量会覆盖 `.env` 文件中的设置，因此如果你在运行时设置了环境变量，这些设置将优先于 `.env` 文件中的配置。

优先级顺序（从高到低）：

1. 运行时设置的环境变量 (如 `export LOG_LEVEL=DEBUG`)
2. `.env` 文件中的配置
3. 框架默认值

## `provider.json` 文件

`provider.json` 文件用于配置 LLM 接口的相关信息，包括 API 密钥、提供商信息、模型名称等。你可以在项目根目录创建一个 `provider.json` 文件，内容示例如下：

### 配置文件结构

provider.json 使用嵌套结构：`提供商 -> 模型名 -> 配置参数`

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
      "api_keys": ["sk-test-key-3", "sk-test-key-4"],
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
  ],
  "claude": [
    {
      "model_name": "claude-3-sonnet",
      "api_keys": ["claude-test-key-1"],
      "base_url": "https://api.anthropic.com/v1",
      "max_retries": 5,
      "retry_delay": 1.0,
      "rate_limit_capacity": 8,
      "rate_limit_refill_rate": 0.5
    }
  ]
}
```

### 配置参数说明

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `api_keys` | 数组 | API 密钥列表，支持多个密钥用于负载均衡 | `["key1", "key2"]` |
| `base_url` | 字符串 | API 服务器地址 | `https://api.openai.com/v1` |
| `model` | 字符串 | 模型名称，与提供商对应 | `gpt-3.5-turbo` |
| `max_retries` | 数字 | 最大重试次数，默认 3 | `5` |
| `retry_delay` | 浮点数 | 重试延迟（秒），默认 1.0 | `1.0` |
| `rate_limit_capacity` | 数字 | 令牌桶容量，默认 10 | `20` |
| `rate_limit_refill_rate` | 浮点数 | 令牌补充速率（tokens/秒），默认 1.0 | `3.0` |

### 加载和使用

然后你可以使用这个json文件来加载所有的接口，例如：

```python
from SimpleLLMFunc import OpenAICompatible

# 加载所有模型
models = OpenAICompatible.load_from_json_file("provider.json")

# 获取特定模型
gpt35 = models["openai"]["gpt-3.5-turbo"]
gpt4 = models["openai"]["gpt-4"]
deepseek = models["deepseek"]["deepseek-chat"]
zhipu = models["zhipu"]["glm-4"]

# 在装饰器中使用
from SimpleLLMFunc import llm_function

@llm_function(llm_interface=gpt35)
async def my_task(text: str) -> str:
    """处理文本的任务"""
    pass
```

### 最佳实践

1. **多个 API 密钥**: 为了实现负载均衡和高可用性，建议为每个模型配置多个 API 密钥
2. **不同模型的限流策略**: 根据不同的 API 限制配置不同的 `rate_limit_capacity` 和 `rate_limit_refill_rate`
3. **环境区分**: 可以为开发环境和生产环境配置不同的 `max_retries` 和 `retry_delay`

