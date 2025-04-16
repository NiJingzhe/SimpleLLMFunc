# SimpleAgent

一个灵活、可扩展的智能代理框架，支持多种 LLM 接口和强大的日志跟踪系统。

## 特性

- 多种 LLM 提供商接口（目前支持 ZhipuAI）
- 自动化 API 密钥负载均衡
- 强大的日志系统，支持 trace_id 跟踪和搜索
- 灵活的配置管理

## 项目结构

```
SimpleAgent/
├── SimpleAgent/            # 核心包
│   ├── interface/          # LLM 接口
│   │   ├── llm_interface.py  # LLM 接口抽象类
│   │   ├── key_pool.py       # API 密钥管理
│   │   └── zhipu.py          # 智谱 AI 接口实现
│   ├── logger/             # 日志系统
│   │   ├── logger.py         # 日志核心实现
│   │   └── logger_config.py  # 日志配置
│   └── config.py           # 全局配置
└── logs/                   # 日志输出目录
    └── log_indices/        # 日志索引
```

## 配置管理

SimpleAgent 使用分层配置系统：

- 环境变量：最高优先级
- `.env` 文件：次优先级
- `config.py` 默认值：最低优先级

### 配置示例 (.env)

```
ZHIPU_API_KEYS=["your-api-key-1", "your-api-key-2"]
LOG_LEVEL=DEBUG
LOG_DIR=./logs
```

## LLM 接口

SimpleAgent 的 LLM 接口设计原则：

- 简单、无状态的函数调用
- 专注于提供 LLM 调用功能，不管理历史记录
- 提供普通调用和流式调用两种模式
- 返回未经处理的原始响应

### 示例用法

```python
from SimpleAgent.interface import ZhipuAI_glm_4_flash_Interface

# 非流式调用
response = ZhipuAI_glm_4_flash_Interface.chat(
    trace_id="unique_trace_id",
    messages=[{"role": "user", "content": "你好"}]
)

# 流式调用
for chunk in ZhipuAI_glm_4_flash_Interface.chat_stream(
    trace_id="unique_trace_id",
    messages=[{"role": "user", "content": "你好"}]
):
    print(chunk)
```

## 日志系统

SimpleAgent 包含强大的日志系统，支持：

- 不同级别的日志（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- 按 trace_id 跟踪和搜索相关日志
- 自动记录代码位置信息
- 彩色控制台输出
- JSON 格式文件日志，便于解析

### 日志使用示例

```python
from SimpleAgent.logger import push_info, push_error, search_logs_by_trace_id

# 记录信息日志
push_info("操作成功完成", trace_id="operation_123")

# 记录错误日志
push_error("操作失败", trace_id="operation_123", exc_info=True)

# 按 trace_id 搜索相关日志
logs = search_logs_by_trace_id("operation_123")
```

## API 密钥管理

SimpleAgent 使用 `APIKeyPool` 类管理多个 API 密钥，实现负载均衡：

- 自动选择最少负载的 API 密钥
- 单例模式确保每个提供商只有一个密钥池
- 自动跟踪每个密钥的使用情况

## Agent 状态管理

Agent 自身拥有程序化的状态管理机制，流程如下：

- **感知 (Perception)**: 从环境中读取上下文
- **规划 (Planning)**: 生成任务计划（产生子任务，基于提供的能力）
- **执行 (Execution)**: 执行计划中的任务
- **反馈 (Feedback)**: 分析执行结果并调整后续行为

## 开始使用

1. 克隆此仓库
2. 创建 `.env` 文件并配置您的 API 密钥
3. 使用 Poetry 安装依赖：`poetry install`
4. 导入并使用 SimpleAgent 的各个组件


