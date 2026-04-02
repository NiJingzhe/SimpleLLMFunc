# Decorators And Tools

## `@llm_function`

Use `@llm_function` for one-shot typed tasks.

Best practices:

- Keep the signature narrow and explicit.
- Use a typed return value instead of asking the model for hand-written JSON.
- Use `_template_params` only when one prompt pattern truly needs runtime role/style slots.
- Add a toolkit only for real external capability, not for tasks the model can do unaided.

## `@llm_chat`

Use `@llm_chat` for agent-like or conversational behavior.

Best practices:

- Use `stream=True` for chat UIs or incremental feedback.
- Name the history parameter `history` or `chat_history`.
- Keep `history` outside the function and feed it back in on the next turn.
- Use `strict_signature=True` when you want a stable agent signature for self-reference or fork-heavy flows.
- Set an explicit `max_tool_calls` only if you want a hard loop cap.

## `@tool`

Use `@tool` for capabilities the model may call.

Best practices:

- The function must be `async def`.
- Write a good docstring `Args:` section because parameter descriptions are extracted from it.
- Keep tool outputs concise unless the full payload is truly required.
- Use `best_practices=[...]` when the model needs durable guidance about how to use the tool.
- Use `too_long_to_file=True` for tools that may return massive text, such as code execution or large search results; the framework keeps roughly the first 20000 tokens in-chat and writes the full text to a temp file.

## `FileToolset`

`FileToolset` provides safe file operations inside one workspace.

Rules that matter in practice:

- Read before you write; edits are hash-guarded.
- `grep` requires a `path_pattern` and rejects overly broad `.*` searches.
- Hidden files and out-of-workspace paths are rejected.
- `echo_into` is for full-file replacement, not patch-style edits.

## Event mode and TUI

Turn on `enable_event=True` when you need execution visibility.

Key differences:

- `llm_function(enable_event=True)` yields `ReactOutput`; the final `ResponseYield.response` is the parsed Python result.
- `llm_chat(enable_event=True)` also yields `ReactOutput`, but response payloads stay closer to raw response or stream chunks.

Use the built-in TUI with the existing pattern:

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.utils.tui import tui


@tui()
@llm_chat(..., stream=True, enable_event=True)
async def agent(message: str, history=None):
    """Your agent prompt."""
    pass
```

When handling event mode manually, inspect `ResponseYield` and `EventYield` separately instead of assuming tuple outputs.
