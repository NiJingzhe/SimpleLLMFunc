# 项目介绍

## SimpleLLMFunc 是什么?

SimpleLLMFunc 是一个轻量级的 LLM 应用开发框架，旨在简化大语言模型（LLM）在应用中的集成过程。本框架的设计理念是"**Everything is Function, Prompt is Code**"，提供类型安全的装饰器，让开发者能以最自然、直观的方式利用大语言模型的能力。

## 为什么需要 SimpleLLMFunc?

在开发基于大语言模型的应用时，我们常常面临以下挑战：

1. 需要不断编写重复的 API 调用代码
2. Prompt 作为字符串变量存在于代码中，不够直观
3. 流程编排受到框架约束，缺乏灵活性
4. 调试和监控 LLM 调用过程困难

SimpleLLMFunc 旨在解决这些问题，让你：
- 用函数式编程的方式定义 LLM 能力
- 将 Prompt 直接放在函数的文档字符串中，使代码更加清晰
- 灵活构建应用流程，不受框架约束
- 获得全面的日志记录和调试信息

## 核心特性

- **装饰器驱动**: 使用 `@llm_function` 和 `@llm_chat` 装饰器轻松创建 LLM 驱动的功能
- **DocString 即 Prompt**: 直接在函数文档中定义 Prompt，提高代码可读性
- **类型安全**: 支持 Python 类型注解和 Pydantic 模型，确保数据结构正确
- **通用模型接口**: 兼容任何符合 OpenAI API 格式的模型服务
- **API 密钥管理**: 智能负载均衡多个 API 密钥
- **工具系统**: 简单易用的工具定义和调用机制
- **强大日志**: 支持 trace_id 跟踪和搜索，方便调试和监控

## 项目架构

SimpleLLMFunc 的核心架构包括以下几个主要模块：

```
SimpleLLMFunc/
├── interface/            # LLM 接口
│   ├── llm_interface.py  # LLM 接口抽象类
│   ├── key_pool.py       # API 密钥管理
│   └── openai_compatible.py # 通用接口实现
├── llm_decorator/        # LLM装饰器
│   ├── llm_function_decorator.py # 函数装饰器
│   └── llm_chat_decorator.py     # 对话装饰器
├── logger/               # 日志系统
│   ├── logger.py         # 日志核心功能
│   └── logger_config.py  # 日志配置
├── tool/                 # 工具系统
│   └── tool.py           # 工具定义和装饰器
└── config.py             # 全局配置
```

### 模块介绍

#### LLM 接口模块

`interface` 模块提供了与各种 LLM 服务通信的标准接口。它支持任何符合 OpenAI API 格式的服务，包括 OpenAI 自身、Azure OpenAI、各种开源模型的兼容 API 等。

#### LLM 装饰器模块

`llm_decorator` 模块是框架的核心，提供了两种主要装饰器：

- `@llm_function`: 用于创建无状态的 LLM 功能，适合单次查询和转换任务
- `@llm_chat`: 用于创建对话式 LLM 功能，支持历史记录管理和多轮交互

#### 日志系统

`logger` 模块提供了全面的日志记录功能，包括 trace_id 跟踪、token 使用统计、系统和用户提示的记录等。

#### 工具系统

`tool` 模块允许 LLM 访问外部工具和服务，增强其解决问题的能力。工具可以是任何 Python 函数，通过 `@tool` 装饰器进行标记。

## 与其他框架的比较

相比其他 LLM 开发框架，SimpleLLMFunc 具有以下优势：

- **更轻量级**: 核心依赖少，启动快速，学习曲线平缓
- **更直观**: Prompt 直接嵌入函数文档，而非单独的变量
- **更灵活**: 不强制使用特定的框架结构，可以自由组合功能
- **更易用**: 装饰器语法简洁明了，减少样板代码
- **更易调试**: 完整的日志系统，便于追踪问题和优化 Prompt
