# 版本说明

本文档记录了 SimpleLLMFunc 项目的所有重要更改。

## 0.1.6.3版本 (最新)

**更新日期**: 2025-05-16

### 功能改进

1. **工具序列化方法优化**
   - 修复了 `Tool.serialize_tools()` 方法的使用体验
   - 现在可以直接将被 `@tool` 装饰的函数传递给序列化方法，无需访问内部 `_tool` 属性

### 用法对比

**之前的用法**:
```python
from SimpleLLMFunc import Tool, tool

@tool(name="shell", description="a shell")
def shell(command: str) -> tuple[str, str]:
    """
    Args:
        command: the command need to be executed in shell

    Returns:
        tuple[str, str]: stdout and stderr
    """
    # implement here

serialized = Tool.serialize_tools([shell._tool])[0]  # 需要访问内部 _tool 属性
```

**现在的用法**:
```python
from SimpleLLMFunc import Tool, tool

@tool(name="shell", description="a shell")
def shell(command: str) -> tuple[str, str]:
    """
    Args:
        command: the command need to be executed in shell

    Returns:
        tuple[str, str]: stdout and stderr
    """
    # implement here

serialized = Tool.serialize_tools([shell])[0]  # 直接使用装饰后的函数
```

## 0.1.6.2版本

**更新日期**: 2025-05-02

### 主要功能

1. **Token 用量统计**
   - 为 `llm_function` 和 `llm_chat` 装饰的函数添加了 token 用量统计功能
   - 用量信息会记录在日志中，可通过 function call trace id 查看
   - Token 统计仅限函数自身上下文，嵌套调用的 token 不会计入外层函数

### 日志示例

```json
"get_daily_recommendation_86cdc1b5-f279-4643-b2a0-13782ab30b26": [
    {
        "timestamp": "2025-05-16T20:23:02.428657+08:00",
        "level": "INFO",
        "location": "llm_function_example.py:main:116",
        "message": "LLM 函数 'get_daily_recommendation' 被调用，参数: {\n    \"city\": \"Hangzhou\"\n}",
        "input_tokens": 0,
        "output_tokens": 0
    },
    {
        "timestamp": "2025-05-16T20:23:04.016900+08:00",
        "level": "INFO",
        "location": "utils.py:get_last_item_of_generator:12",
        "message": "LLM 函数 'get_daily_recommendation' 收到初始响应",
        "input_tokens": 453,
        "output_tokens": 232
    }
]
```

## 0.1.6.1版本

**更新日期**: 2025-04-15

### 功能改进

1. **工具调用流程优化**
   - 改进了工具调用的错误处理
   - 添加了对工具执行超时的处理

2. **日志系统增强**
   - 添加了更详细的工具调用日志
   - 为关键操作添加了性能指标记录

### 错误修复

1. 修复了在特定情况下工具调用结果未正确传递给 LLM 的问题
2. 修复了多线程环境下日志上下文混乱的问题

## 0.1.6.0版本

**更新日期**: 2025-03-28

### 主要功能

1. **增加 `llm_chat` 装饰器**
   - 支持创建对话型助手
   - 内置历史记录管理
   - 支持流式响应

2. **改进 API 密钥管理**
   - 添加 APIKeyPool 实现负载均衡
   - 支持自动重试和失败转移

### 其他改进

1. 文档系统整体升级
2. 添加了更多示例代码
3. 性能优化

## 0.1.5.0版本

**更新日期**: 2025-03-10

### 主要功能

1. **工具系统**
   - 添加了 `@tool` 装饰器
   - 允许 LLM 函数调用外部工具和服务
   - 支持工具调用链

### 其他改进

1. 改进错误处理
2. 添加更多单元测试
3. 代码风格优化

## 0.1.0.0版本

**更新日期**: 2025-02-15

### 初始版本功能

1. 核心 `@llm_function` 装饰器
2. 通用 OpenAI 兼容接口
3. 基本日志系统
4. Pydantic 模型集成
