# SimpleAgent

一个灵活、可扩展的智能代理框架，支持多种 LLM 接口和强大的日志跟踪系统。

## 特性

- 多种 LLM 提供商接口（目前支持 ZhipuAI）
- 自动化 API 密钥负载均衡
- 强大的日志系统，支持 trace_id 跟踪和搜索
- 灵活的配置管理
- 圆桌会议式的交互环境，支持多 Agent 协作
- 基于观察者模式的事件驱动机制
- Agent 异步任务管理系统
- 工具系统，支持 Agent 与外部环境交互

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
│   ├── tool/               # 工具系统
│   │   ├── tool.py           # 工具抽象基类
│   │   └── schemas.py        # 工具参数模型
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

## Tools 工具系统

SimpleAgent 实现了强大的工具系统，使 Agent 能够与外部环境交互并执行各种操作。工具系统采用插件式设计，易于扩展和自定义。

### 核心概念

- **Tool**：表示 Agent 可以使用的一个能力或功能
- **ToolParameters**：工具参数的描述和验证模型
- **ParameterType**：参数类型枚举，支持基本类型和嵌套类型

### 工具系统设计

工具系统的核心是 `Tool` 抽象基类，具有以下特点：

- **统一接口**：所有工具继承自同一个抽象基类，确保接口一致性
- **自描述性**：每个工具包含名称、描述和参数列表，方便 Agent 理解如何使用
- **类型安全**：使用 Pydantic 模型进行参数验证和类型检查
- **OpenAI 兼容**：工具可以序列化为 OpenAI Function Calling 格式

### 参数类型系统

SimpleAgent 支持丰富的参数类型：

```python
class ParameterType(str, Enum):
    STRING = "string"      # 字符串
    INTEGER = "integer"    # 整数
    FLOAT = "float"        # 浮点数
    BOOLEAN = "boolean"    # 布尔值
    LIST = "list"          # 列表
    DICT = "dict"          # 字典
```

同时支持嵌套类型，如：
- `list<string>` - 字符串列表
- `dict<string,integer>` - 键为字符串、值为整数的字典

### 工具定义示例

```python
from SimpleAgent.tool import Tool, ToolParameters, ParameterType

class WebSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="在互联网上搜索信息",
            parameters=[
                ToolParameters(
                    name="query",
                    description="搜索查询词",
                    type=ParameterType.STRING,
                    required=True,
                    example="最新的人工智能研究"
                ),
                ToolParameters(
                    name="max_results",
                    description="返回结果数量",
                    type=ParameterType.INTEGER,
                    required=False,
                    default=5,
                    example=10
                )
            ]
        )
    
    def run(self, query: str, max_results: int = 5):
        # 搜索逻辑实现
        return {"results": [...]}
```

### OpenAI 工具格式转换

SimpleAgent 的工具可以自动转换为 OpenAI Function Calling 格式：

```python
tools = [WebSearchTool(), CalendarTool(), CalculatorTool()]
openai_tools = Tool.serialize_tools(tools)

# 使用 OpenAI 工具格式调用 LLM
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "帮我搜索最新的人工智能研究"}],
    tools=openai_tools
)
```

### 工具参数验证

`ToolParameters` 类基于 Pydantic，提供了参数验证和文档生成能力：

- **类型验证**：确保参数符合预期类型
- **必填检查**：标记必需参数
- **默认值**：为可选参数提供默认值
- **示例值**：为 LLM 提供使用示例

### 工具集成到 Agent

Agent 可以通过 `toolkit` 属性持有多个工具：

```python
agent = Agent(
    name="助手",
    description="帮助用户完成各种任务",
    toolkit=[WebSearchTool(), CalendarTool()],
    one_sentence_target="提供高质量的信息和服务"
)
```

## 圆桌会议式交互环境

SimpleAgent 实现了"圆桌会议"式的交互环境，使人类和多个 Agent 能够在共享的上下文空间中进行交流。

核心概念：
- **共享 Table（桌面）**：作为人类和多个 Agent 之间的共享上下文空间
- **Agent 独立记忆**：每个 Agent 维护自己的私有记忆系统
- **动态交互**：参与者可以自由加入或离开"圆桌"，对特定内容做出回应

### 观察者模式设计

SimpleAgent 采用观察者模式来设计 Agent 和 Table 之间的交互：

#### 核心组件

1. **Subject (被观察者)**:
   - Table 作为被观察者
   - 维护观察者列表
   - 在状态变化时通知所有观察者

2. **Observer (观察者)**:
   - Agent 作为观察者
   - 实现更新接口以响应 Table 的变化
   - 可以选择性地关注特定类型的事件

3. **Event (事件)**:
   - 表示 Table 上发生的变化
   - 包含变化的类型和相关数据
   - 提供上下文信息供观察者处理

#### 事件类型

```python
class TableEventType(Enum):
    ITEM_ADDED = "item_added"         # 新项目被添加
    ITEM_UPDATED = "item_updated"     # 项目被更新
    ITEM_REMOVED = "item_removed"     # 项目被删除
    PARTICIPANT_JOINED = "participant_joined"  # 新参与者加入
    PARTICIPANT_LEFT = "participant_left"      # 参与者离开
    TABLE_CLEARED = "table_cleared"   # 桌面被清空
```

#### 事件流程

1. 人类向 Table 添加内容
2. Table 创建事件并通知所有观察者（Agent）
3. 每个 Agent 异步接收到事件通知
4. Agent 根据自身状态和事件决定如何响应
5. Agent 创建新任务来处理事件
6. 处理完成后可能向 Table 添加新内容，触发新的事件循环

## Agent 异步任务机制

SimpleAgent 的 Agent 使用异步任务机制来处理事件和执行操作：

### 任务管理

Agent 维护一个基于优先级的任务队列，可以并行处理多个任务：

- **任务类型**：
  - 响应型任务：回应 Table 上的特定内容
  - 主动型任务：Agent 主动发起的行动
  - 后台型任务：长时间运行的处理过程

- **任务优先级**：
  - 基于重要性、紧急性和相关性确定
  - 高优先级任务可能中断正在进行的低优先级任务

- **中断和恢复**：
  - 任务可以被保存、暂停和恢复
  - 支持任务上下文的保存和加载

### 异步执行框架

使用 Python 的 `asyncio` 库实现异步执行：

```python
class AgentTaskManager:
    """管理 Agent 的任务队列和执行"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.task_queue = PriorityQueue()  # 优先级队列
        self.running_tasks = {}  # 正在运行的任务
        
    async def add_task(self, task: AgentTask) -> str:
        """添加新任务到队列"""
        await self.task_queue.put((task.priority, task))
        return task.task_id
        
    async def process_tasks(self) -> None:
        """处理任务队列"""
        while not self.task_queue.empty():
            _, task = await self.task_queue.get()
            
            # 执行任务
            await task.execute()
```

### 设计优势

1. **事件驱动**：人类可以在任何时候向 Table 添加内容，Agent 会自动响应
2. **异步处理**：Agent 可以并行处理多个任务，不会阻塞响应新事件
3. **解耦性**：Table 不需要知道具体有哪些 Agent 在观察它，Agent 也不需要直接依赖 Table 的内部实现
4. **自然交互**：模拟真实世界的对话模式，支持多方参与的复杂交互场景


## 开始使用

1. 克隆此仓库
2. 创建 `.env` 文件并配置您的 API 密钥
3. 使用 Poetry 安装依赖：`poetry install`
4. 导入并使用 SimpleAgent 的各个组件