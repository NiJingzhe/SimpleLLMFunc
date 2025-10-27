# Langfuse 集成指南

SimpleLLMFunc 框架已集成 Langfuse 可观测性平台，支持对 LLM 生成和工具调用进行全面追踪。

## 功能特性

- **LLM 生成追踪**: 自动记录所有 LLM 调用的输入、输出、模型参数和使用统计
- **工具调用观测**: 追踪工具调用的参数、执行结果和性能指标
- **嵌套跨度支持**: 支持复杂的多层调用链追踪
- **流式响应支持**: 兼容流式和非流式 LLM 响应
- **优雅降级**: 在 Langfuse 不可用时自动禁用，不影响核心功能

## 安装和配置

### 1. 安装 Langfuse

Langfuse 已包含在框架依赖中：

```bash
# 如果使用 poetry
poetry install

# 如果使用 pip
pip install langfuse
```

### 2. 获取 Langfuse 凭据

1. 访问 [Langfuse](https://langfuse.com) 并注册账户
2. 创建新项目
3. 获取项目的 Public Key 和 Secret Key

### 3. 配置环境变量

```bash
export LANGFUSE_PUBLIC_KEY="your_public_key"
export LANGFUSE_SECRET_KEY="your_secret_key"
export LANGFUSE_HOST="https://cloud.langfuse.com"  # 可选
export LANGFUSE_ENABLED="true"  # 可选
```

### 4. 初始化观测器

```python
from SimpleLLMFunc import configure_langfuse_from_env

# 从环境变量自动配置
observer = configure_langfuse_from_env()
```

或者直接在代码中配置：

```python
from SimpleLLMFunc import configure_langfuse

observer = configure_langfuse(
    public_key="your_public_key",
    secret_key="your_secret_key",
    host="https://cloud.langfuse.com",
    enabled=True
)
```

## 使用示例

### 基本 LLM 函数追踪

```python
from SimpleLLMFunc import llm_function, configure_langfuse_from_env, OpenAI_Interface

# 配置观测器
configure_langfuse_from_env()

# 配置 LLM 接口
llm = OpenAI_Interface(api_key="your_api_key")

@llm_function(llm_interface=llm)
async def analyze_text(text: str) -> str:
    """分析文本内容并提供摘要"""
    pass

# 使用函数 - 自动追踪到 Langfuse
result = await analyze_text("这是一段需要分析的文本...")
```

### 带工具调用的追踪

```python
from SimpleLLMFunc import llm_function, tool

@tool(name="calculate", description="执行数学计算")
async def calculate(expression: str) -> dict:
    """执行数学表达式计算"""
    result = eval(expression)  # 实际使用中应使用更安全的方法
    return {"expression": expression, "result": result}

@llm_function(
    llm_interface=llm,
    toolkit=[calculate],
    max_tool_calls=3
)
async def math_assistant(question: str) -> str:
    """数学助手，可以回答数学问题并进行计算"""
    pass

# 使用 - 工具调用也会被自动追踪
result = await math_assistant("计算 15 * 8 + 32 的结果")
```

### 聊天对话追踪

```python
from SimpleLLMFunc import llm_chat

@llm_chat(
    llm_interface=llm,
    toolkit=[calculate],
    max_tool_calls=2
)
async def chat_bot(message: str, history: list = None):
    """智能聊天机器人"""
    pass

# 使用 - 每轮对话都会被追踪
history = []
async for response, updated_history in chat_bot("你好，请帮我计算一些数学题", history):
    if response.strip():
        print(response)
history = updated_history
```

## 追踪数据结构

### Generation 追踪

每个 LLM 调用会创建一个 Generation 观测，包含：

- **输入**: 发送给 LLM 的消息列表
- **输出**: LLM 的响应内容和工具调用
- **模型信息**: 模型名称和参数
- **使用统计**: Token 使用量和成本信息
- **元数据**: 流式模式、可用工具数量等

### Tool 追踪

每个工具调用会创建一个 Tool 观测，包含：

- **输入**: 工具调用的参数
- **输出**: 工具执行的结果
- **元数据**: 工具调用 ID、执行时间等

### 层级结构

```
Function Call (Span)
├── Initial Generation
│   ├── Input: Messages
│   ├── Output: Response + Tool Calls
│   └── Usage: Token counts
├── Tool Call 1 (Tool)
│   ├── Input: Parameters
│   └── Output: Result
├── Tool Call 2 (Tool)
│   ├── Input: Parameters
│   └── Output: Result
└── Follow-up Generation
    ├── Input: Updated Messages
    ├── Output: Final Response
    └── Usage: Token counts
```

## 配置选项

### 环境变量

| 变量名 | 描述 | 默认值 | 必需 |
|--------|------|--------|------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse 公钥 | - | 是 |
| `LANGFUSE_SECRET_KEY` | Langfuse 私钥 | - | 是 |
| `LANGFUSE_HOST` | Langfuse 服务器地址 | `https://cloud.langfuse.com` | 否 |
| `LANGFUSE_ENABLED` | 是否启用观测 | `true` | 否 |

### 代码配置

```python
from SimpleLLMFunc.observability import LangfuseObserver

observer = LangfuseObserver(
    public_key="your_public_key",
    secret_key="your_secret_key",
    host="https://cloud.langfuse.com",
    enabled=True
)
```

## 最佳实践

### 1. 环境分离

```bash
# 开发环境
export LANGFUSE_ENABLED="false"

# 生产环境
export LANGFUSE_ENABLED="true"
export LANGFUSE_PUBLIC_KEY="prod_public_key"
export LANGFUSE_SECRET_KEY="prod_secret_key"
```

### 2. 错误处理

框架会自动处理 Langfuse 相关错误，但建议监控日志：

```python
import logging

# 启用 SimpleLLMFunc 日志
logging.getLogger("SimpleLLMFunc").setLevel(logging.INFO)
```

### 3. 性能考虑

- Langfuse 调用是异步的，不会阻塞主要业务逻辑
- 在高并发场景下，考虑设置合适的 Langfuse 客户端配置
- 可以通过 `LANGFUSE_ENABLED=false` 临时禁用观测

### 4. 数据隐私

- 确保敏感数据在发送到 Langfuse 前已脱敏
- 考虑使用自托管的 Langfuse 实例处理敏感数据

## 故障排除

### 常见问题

1. **观测器未启用**
   ```
   警告: Langfuse 观测功能未启用
   ```
   - 检查环境变量是否正确设置
   - 确认 Langfuse 包已安装

2. **连接失败**
   ```
   Langfuse 初始化失败: Connection error
   ```
   - 检查网络连接
   - 验证 API 密钥是否正确
   - 确认 LANGFUSE_HOST 设置正确

3. **数据未显示**
   - 检查 Langfuse 仪表板的项目设置
   - 确认使用的是正确的项目密钥
   - 等待数据同步（通常几秒钟）

### 调试模式

```python
import logging

# 启用详细日志
logging.getLogger("SimpleLLMFunc").setLevel(logging.DEBUG)

# 查看 Langfuse 相关日志
logging.getLogger("langfuse").setLevel(logging.DEBUG)
```

## 示例项目

查看 `examples/langfuse_integration_example.py` 获取完整的使用示例。

## 相关链接

- [Langfuse 官方文档](https://langfuse.com/docs)
- [SimpleLLMFunc 文档](../README.md)
- [API 参考](./api_reference.md)
