# 使用指南

本指南提供了 SimpleLLMFunc 框架各个功能模块的详细文档。请根据你的需求选择相应的文档进行阅读。

## 📚 详细文档导航

```{toctree}
:maxdepth: 1
:caption: 基础设施

配置与环境 <detailed_guide/config>
LLM 接口层 <detailed_guide/llm_interface>
```

```{toctree}
:maxdepth: 1
:caption: 开发体验

llm_function 装饰器 <detailed_guide/llm_function>
llm_chat 装饰器 <detailed_guide/llm_chat>
```

```{toctree}
:maxdepth: 1
:caption: Agent 主体逻辑

事件流系统 <detailed_guide/event_stream>
中断与取消 <detailed_guide/abort>
```

```{toctree}
:maxdepth: 1
:caption: 工具与运行时

工具系统 <detailed_guide/tool>
PyRepl运行时 <pyrepl>
```

```{toctree}
:maxdepth: 1
:caption: UI 与交互

终端TUI <tui>
```

```{toctree}
:maxdepth: 1
:caption: 集成与示例

Langfuse集成 <langfuse_integration>
示例代码 <examples>
```

## 🎯 按使用场景查找文档

### 我想要快速上手
👉 [快速开始](quickstart.md) - 5分钟内运行你的第一个示例

### 我想要配置 API 和环境
👉 [配置与环境](detailed_guide/config.md) - 学习如何设置 provider.json 和环境变量

### 我想要创建 LLM 函数
👉 [llm_function 装饰器](detailed_guide/llm_function.md) - 创建无状态的 LLM 驱动函数

### 我想要构建聊天应用
👉 [llm_chat 装饰器](detailed_guide/llm_chat.md) - 构建多轮对话和 Agent 应用

### 我想要观察执行过程
👉 [事件流系统](detailed_guide/event_stream.md) - 实时观察 ReAct 循环的执行过程

### 我想要中断正在执行的回合
👉 [中断与取消](detailed_guide/abort.md) - AbortSignal 用法与事件流收尾说明

### 我想要整合工具/API
👉 [工具系统](detailed_guide/tool.md) - 让 LLM 调用外部函数和 API

### 我想要理解接口设计
👉 [LLM 接口层](detailed_guide/llm_interface.md) - 了解密钥管理和流量控制

### 我想要查看代码示例
👉 [示例代码](examples.md) - 浏览各种使用场景的完整示例

## 🚀 推荐学习路径

### 初级用户（刚开始使用）
1. [快速开始](quickstart.md) - 环境配置和第一个示例
2. [llm_function 装饰器](detailed_guide/llm_function.md) - 基础功能
3. [示例代码](examples.md) - 学习实际用法

### 中级用户（已掌握基础）
1. [llm_chat 装饰器](detailed_guide/llm_chat.md) - 构建交互应用
2. [工具系统](detailed_guide/tool.md) - 整合外部能力
3. [配置与环境](detailed_guide/config.md) - 优化配置

### 高级用户（深入理解框架）
1. [LLM 接口层](detailed_guide/llm_interface.md) - 密钥管理和流量控制
2. 自定义 LLM 接口和工具
3. [示例代码](examples.md) - 生产级别的实现参考

## 📖 按功能模块查找

| 功能 | 文档 | 说明 |
|-----|------|------|
| 基础配置 | [配置与环境](detailed_guide/config.md) | API 密钥、环境变量、provider.json |
| 简单任务 | [llm_function 装饰器](detailed_guide/llm_function.md) | 无状态函数、文本处理、数据转换 |
| 对话应用 | [llm_chat 装饰器](detailed_guide/llm_chat.md) | 多轮对话、历史管理、流式响应 |
| 事件流 | [事件流系统](detailed_guide/event_stream.md) | 实时观察、工具调用监控、性能分析 |
| 中断控制 | [中断与取消](detailed_guide/abort.md) | 终止流式输出、取消工具调用 |
| 工具集成 | [工具系统](detailed_guide/tool.md) | 工具定义、调用、多模态返回 |
| 系统设计 | [LLM 接口层](detailed_guide/llm_interface.md) | 接口抽象、密钥池、流量控制 |
| 实战示例 | [示例代码](examples.md) | 各种场景的完整代码 |

## ❓ 常见问题速查

- **如何配置 API 密钥？** → [配置与环境](detailed_guide/config.md)
- **装饰器支持同步函数吗？** → [llm_function 装饰器 - 重要说明](detailed_guide/llm_function.md)
- **如何做多轮对话？** → [llm_chat 装饰器](detailed_guide/llm_chat.md)
- **如何中断当前回复？** → [中断与取消](detailed_guide/abort.md)
- **如何让 LLM 调用函数？** → [工具系统](detailed_guide/tool.md)
- **支持哪些 LLM 提供商？** → [LLM 接口层 - OpenAICompatible 实现](detailed_guide/llm_interface.md)
- **如何处理错误和重试？** → [LLM 接口层 - 故障排除](detailed_guide/llm_interface.md)

## 🔗 其他资源

- [项目介绍](introduction.md) - 了解 SimpleLLMFunc 的设计理念
- [示例代码](examples.md) - 各种场景的完整代码示例
- [贡献指南](contributing.md) - 如何为项目做出贡献
- [GitHub 仓库](https://github.com/NiJingzhe/SimpleLLMFunc) - 源代码和问题追踪

## 💡 提示

- 每个文档都包含完整的代码示例，可以直接复制使用
- 使用浏览器的搜索功能（Ctrl+F）快速定位内容
- 遇到问题时，先查看对应文档的"故障排除"或"常见问题"部分
- 所有示例代码都位于 [examples/](https://github.com/NiJingzhe/SimpleLLMFunc/tree/master/examples) 目录
