# 终端 TUI（Textual）

SimpleLLMFunc 提供了开箱即用的 TUI 装饰器，基于 `textual` + `event stream` 构建。

你可以把它直接叠加在 `@llm_chat` 上，让 Agent 具备完整的终端输入循环、流式渲染和工具调用可视化。

> 提示：`@tui` 依赖事件流，务必在 `@llm_chat` 中设置 `enable_event=True`，并推荐开启 `stream=True` 以获得流式渲染体验。

## 安装依赖

`textual` 已作为框架依赖。

如果你在已有环境中升级，请重新安装依赖：

```bash
poetry install
```

## 快速开始

```python
from SimpleLLMFunc import llm_chat, tui


@tui()
@llm_chat(
    llm_interface=llm,
    toolkit=[...],
    stream=True,
    enable_event=True,
)
async def agent(message: str, history=None):
    """Your agent prompt."""


if __name__ == "__main__":
    agent()  # 启动 TUI + input loop
```

`@tui` 会自动识别输入参数：

- `history` / `chat_history` 作为历史参数
- 其余第一个参数作为用户输入

## UI 能力

- 用户消息与模型消息交替渲染
- 模型流式输出实时刷新，并支持 Markdown 渲染
- 流式输出期间消息区自动跟随到底部，优先展示最新内容
- 推理增量（reasoning delta）以灰色文本展示（当模型提供该字段时）
- 每次模型调用结束后，在消息下方展示耗时与 token 用量
- 工具调用开始时展示结构化参数（可读格式，不是裸 JSON 字符串）
- 工具执行期间消费 `CustomEvent` 并实时更新工具输出
- 当工具（如 PyRepl）触发 `input()` 时，输入框自动切换为工具输入模式并优先回填该请求
- 工具结束后展示结果与调用统计（耗时/状态）
- fork 任务自动拆分为独立列，基于 `origin.fork_id` 路由到对应分栏
- 聊天消息区支持超出视口滚动，消息项会随内容自适应高度

## 交互与退出

- 发送消息：输入后按 Enter
- 当存在待处理的工具输入请求时，Enter 会优先把输入提交给该请求
- 强制发送新一轮聊天：`/chat <message>`（可在有待处理工具输入时绕过优先路由）
- 复制完整转录：`/copy` 或 `Ctrl+Y`
- 退出命令：`/exit`、`/quit`、`/q`
- 快捷键退出：`Ctrl+Q`（同时保留 `Ctrl+C`）

## 自定义 Tool 事件 Hook

`@tui` 支持通过 `custom_event_hook` 注入自定义事件解析逻辑：

```python
from SimpleLLMFunc.hooks.events import CustomEvent
from SimpleLLMFunc.utils.tui import ToolEventRenderUpdate, ToolRenderSnapshot


def my_hook(
    event: CustomEvent,
    snapshot: ToolRenderSnapshot,
) -> ToolEventRenderUpdate | None:
    if event.event_name != "batch_progress" or not isinstance(event.data, dict):
        return None

    return ToolEventRenderUpdate(
        append_output=f"progress={event.data['percent']}%\n"
    )


@tui(custom_event_hook=[my_hook])
@llm_chat(..., enable_event=True, stream=True)
async def agent(message: str, history=None):
    ...
```

框架也内置了部分常见工具事件的默认解析（例如 PyRepl 的 `kernel_stdout`/`kernel_stderr`/`kernel_input_request`）。

## 运行示例

参考示例：`examples/tui_chat_example.py`
