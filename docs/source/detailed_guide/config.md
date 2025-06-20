#  配置文件说明

## `.env` 文件

`.env` 文件用于存储环境变量，在本框架中主要用于配置日志相关设置。你可以在你项目最终的`WORKING DIR`下创建一个 `.env` 文件，或者直接在环境变量中设置这些值。

```bash
# 日志相关配置

# LOG_DIR：日志文件存放目录，默认为当前目录 `./`
LOG_DIR=./logs

# LOG_FILE：日志文件名，默认为 `agent.log`
# 在LOG目录下还会有另一个log indices文件夹，里面存放按照trace id聚类的日志文件
LOG_FILE=my_agent.log

# LOG_LEVEL：会打印到控制台的志级别，默认为 `WARNING`
LOG_LEVEL=WARNING
```

注意，直接`export`环境变量会覆盖 `.env` 文件中的设置，因此如果你在运行时设置了环境变量，这些设置将优先于 `.env` 文件中的配置。

## `provider.json` 文件

`provider.json` 文件用于配置 LLM 接口的相关信息，包括 API 密钥、提供商 ID、模型名称等。你可以在项目根目录创建一个 `provider.json` 文件，内容示例如下：

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

然后你可以使用这个json文件来加载所有的接口，例如：
```python
from SimpleLLMFunc import OpenAICompatible

all_model = OpenAICompatible.load_from_json_file("provider.json")
llm_interface = all_model["openai"]["gpt-3.5-turbo"]

# 或者直接指定模型
llm_interface = OpenAICompatible.load_from_json_file("provider.json")["zhipu"]["glm-4"]
```