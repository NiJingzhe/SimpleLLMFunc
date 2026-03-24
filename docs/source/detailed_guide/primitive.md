# Primitive 原语系统

Primitive 是 SimpleLLMFunc 的“运行时内置能力”（CodeAct 范式下的 builtin tool）。
它不是直接暴露给模型的 `@tool`，而是通过 `PyRepl` 的 `execute_code` 间接调用：

```
LLM -> execute_code -> runtime.<pack>.<primitive>(...)
```

## 定位与作用

在 CodeAct 模式下，模型会使用 `execute_code` 来执行一段 Python 代码。
Primitive 就是这段代码里可直接调用的“运行时 API”，例如：

```python
runtime.selfref.history.count()
runtime.github_repo.list_open_issues("owner/repo")
```

因此 Primitive 的定位是：

- **运行时能力入口**：不需要 import，直接从 `runtime` 命名空间访问
- **CodeAct 内置工具**：由 `PyRepl` 托管并运行
- **可被 LLM 发现和理解**：支持 `runtime.get_primitive_spec(...)` 与 `runtime.list_primitive_specs(...)`

## 关键概念

### PrimitivePack

PrimitivePack 是一组 runtime 原语的命名空间：

- pack 名称决定 runtime 前缀：`runtime.<pack_name>.<primitive_name>`
- pack 同时绑定一个默认 backend（见下文）

### Primitive

Primitive 是 pack 下面的一个运行时函数入口：

```text
runtime.<pack_name>.<primitive_name>(...)
```

Primitive 的实现就是一个普通 Python 函数，但它会接收一个注入的上下文 `PrimitiveCallContext`。

### Backend

Backend 是提供能力的 Python 对象（可选但推荐）：

- 可以是 dict / service / client / 自定义类实例
- 用来承载状态或真实能力
- 通过 `ctx.backend` 或 `ctx.get_backend(...)` 注入到 primitive handler

## 开发流程（推荐写法）

```python
from SimpleLLMFunc.builtin import PyRepl

class GitHubRepoAPI:
    def list_open_issues(self, repo: str) -> list[dict[str, str]]:
        # In production, call GitHub REST/GraphQL here.
        return [
            {"id": "42", "title": "Bug: tool timeout", "repo": repo},
            {"id": "57", "title": "Docs: update primitive guide", "repo": repo},
        ]

    def get_issue(self, repo: str, issue_id: str) -> dict[str, str]:
        return {"id": issue_id, "title": "Example issue", "repo": repo}


repl = PyRepl()

github_repo = repl.pack(
    "github_repo",
    backend=GitHubRepoAPI(),
)


@github_repo.primitive(
    "list_open_issues",
    description="List open issues from a GitHub repository.",
)
def list_open_issues(ctx, repo: str) -> list[dict[str, str]]:
    backend = ctx.backend
    if not isinstance(backend, GitHubRepoAPI):
        raise RuntimeError("backend must be a GitHubRepoAPI")
    return backend.list_open_issues(repo)


@github_repo.primitive(
    "get_issue",
    description="Read one issue by id from a GitHub repository.",
)
def get_issue(ctx, repo: str, issue_id: str) -> dict[str, str]:
    backend = ctx.backend
    if not isinstance(backend, GitHubRepoAPI):
        raise RuntimeError("backend must be a GitHubRepoAPI")
    return backend.get_issue(repo, issue_id)


repl.install_pack(github_repo)

# 在 execute_code 内调用：
# runtime.github_repo.list_open_issues("owner/repo")
# runtime.github_repo.get_issue("owner/repo", "42")
```

如果只是轻量扩展，也可以使用 `@repl.primitive(...)`：

```python
repo_backend = GitHubRepoAPI()

repl.register_runtime_backend("github_repo", repo_backend, replace=True)


@repl.primitive("github_repo.list_open_issues", backend="github_repo", replace=True)
def list_open_issues(ctx, repo: str) -> list[dict[str, str]]:
    backend = ctx.get_backend("github_repo")
    if not isinstance(backend, GitHubRepoAPI):
        raise RuntimeError("backend must be a GitHubRepoAPI")
    return backend.list_open_issues(repo)
```

## 在哪里被使用

- `PyRepl.execute_code`：运行时环境里提供 `runtime.*` 命名空间
- `llm_chat(toolkit=repl.toolset)`：模型调用 `execute_code` 后，才能访问 primitives
- 内置 selfref pack：`runtime.selfref.history.*` / `runtime.selfref.fork.*`

## Primitive 上下文注入

每个 primitive handler 都会收到 `PrimitiveCallContext`，包含：

- `primitive_name`：当前 primitive 的完整名称
- `call_id` / `execution_id`：执行链路标识
- `event_emitter`：可用于发送自定义事件
- `repl` / `registry`：运行时宿主与注册中心
- `backend_name`：当前 primitive 默认绑定的 backend 名
- `backend`：通过 backend_name 注入的 backend 对象

上下文的注入流程：

```
worker -> PyRepl._execute_primitive_call -> PrimitiveRegistry.call
-> context.backend_name / context.backend 填充
```

因此建议：**primitive handler 优先通过 `ctx.backend` / `ctx.get_backend(...)` 访问能力**。

## Contract 与可发现性

Primitive 支持结构化契约，用于模型发现：

- `runtime.get_primitive_spec(name)`：单条契约
- `runtime.list_primitive_specs(...)`：批量契约

默认格式为 XML，也可使用 `format='dict'` 返回结构化字段。
契约信息来自以下来源：

- handler 的 docstring（Use/Input/Output/Parse/Parameters/Best Practices）
- `PrimitiveContract` / `@primitive(...)` 装饰器显式参数

## Best Practices 与 Docstring

Primitive 的最佳实践 **只来自 docstring**，且为必填项。
缺失 `Best Practices` 会在注册时抛出错误。

### Docstring 结构规范

Docstring 采用“段落 + 冒号”的简洁格式，大小写不敏感：

- `Use:` / `Input:` / `Output:` / `Parse:`
- `Parameters:`（每行 `name: description` 或 `- name: description`）
- `Best Practices:`（每行 `- rule`）

没有标题的首段会自动当作 `Use`（description）。

推荐的完整模板：

```python
@pack.primitive("list_open_issues")
def list_open_issues(ctx, repo: str) -> list[dict[str, str]]:
    """
    Use: List open issues from a GitHub repository.
    Input: `repo: str` (format: owner/repo).
    Output: `list[dict]` with keys `id`, `title`, `repo`.
    Parse: Read only `id` + `title` unless you need full details.
    Parameters:
    - repo: Repository in owner/repo form.
    Best Practices:
    - If the list is long, call get_issue for only the top 3 IDs.
    - Avoid dumping full issue bodies into chat.
    """
    ...
```

**提示**：Best Practices 依赖 docstring 解析，请把规则写在 docstring。

这些最佳实践会进入 primitive contract，可以通过：

- `runtime.get_primitive_spec("github_repo.list_open_issues")`
- `runtime.list_primitive_specs(contains="github_repo.")`

来读取。

## 如何注入到提示词里

Primitive **本身不会直接注入 system prompt**。

系统会注入的是 **工具（Tool）的最佳实践**，例如 `execute_code` 工具：

- `llm_chat` / `llm_function` 会把工具的 `best_practices` 去重后拼成 `<tool_best_practices>` 块
- 这块内容被插入到 system prompt 顶部

因此 Primitive 的最佳实践通常通过两种方式“进入模型上下文”：

1. **execute_code 工具最佳实践**：提示模型使用 `runtime.get_primitive_spec(...)`
2. **运行时主动查询**：模型在 `execute_code` 中调用 `runtime.get_primitive_spec` 或 `runtime.selfref.guide()`

如果你希望把某个 primitive 的规则“强注入提示词”，建议把规则写到：

- `execute_code` 工具的 best practices（或自定义 Tool）
- 或通过 tool 的 `prompt_injection_builder` 输出固定指导文本

## Backend 生命周期与 fork/clone

如果 backend 有状态，建议实现 `RuntimePrimitiveBackend`：

- `clone_for_fork(context=...)`：fork child 时如何复制/共享 backend（默认共享）
- `on_install(repl)`：安装时回调
- `on_close(repl)`：REPL 关闭时回调（可做资源释放）

这让 fork 子 agent 时的 backend 行为更可控，也能保证状态清理。

## Primitive vs Tool

- **Tool**：`@tool` 暴露给模型的函数调用（OpenAI tool calling）
- **Primitive**：CodeAct 运行时内置 API，通过 `execute_code` 间接调用

你可以把 Primitive 理解为“内置在运行时的 builtin tool”。
