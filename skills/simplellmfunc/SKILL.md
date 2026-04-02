---
name: simplellmfunc
description: "Use SimpleLLMFunc to build typed async LLM functions, chat agents, tools, event-stream consumers, and PyRepl/selfref workflows. Use when writing or editing app code that imports SimpleLLMFunc, configures provider.json, adds @llm_function/@llm_chat/@tool, or mounts PyRepl/FileToolset."
license: MIT
compatibility: "Python 3.12+ repo with async execution and OpenAI-compatible providers configured through provider.json."
metadata:
  project: SimpleLLMFunc
  version: "0.7.5"
---

# SimpleLLMFunc Usage

## When to use this skill
- Use this skill for application-level work built on top of SimpleLLMFunc.
- Use it when the task mentions `llm_function`, `llm_chat`, `tool`, `OpenAICompatible`, `provider.json`, `enable_event`, `PyRepl`, `SelfReference`, `FileToolset`, or the built-in TUI.
- Do not use this skill for framework-internal refactors; use `simplellmfunc-developer` for that.

## Core philosophy
- Treat the LLM call like a normal Python function call.
- Put the prompt in the function docstring.
- Let Python parameters and return annotations define the contract.
- Keep orchestration in Python instead of hiding it in giant prompt strings.
- Prefer small, typed, composable building blocks.

## How system prompts are really built

This is critical for writing good prompts in SimpleLLMFunc: your docstring is important, but it is usually not the whole final system prompt.

### `llm_function`
- Your docstring is first treated as `function_description`.
- If you pass `_template_params`, the docstring is formatted before prompt construction.
- The framework then wraps that docstring inside a system template that also adds:
  - parameter type descriptions
  - return-type instructions
  - plain-text or XML output constraints depending on the return type
- If tools are mounted, the framework prepends a deduplicated `<tool_best_practices>` block before the main system prompt.
- Runtime argument values are not put in the system prompt; they go into the user prompt.

Write `llm_function` docstrings as task policy, quality bar, constraints, and style guidance. Do not waste docstring space restating parameter schemas or low-level output formatting that the framework already injects.

### `llm_chat`
- Base system prompt is either:
  - the latest `system` message in `history`, if one exists, or
  - the function docstring otherwise
- Then the framework prepends `<tool_best_practices>` when tools exist.
- Then it appends a `<must_principles>` block that tells the model to use native structured tool calls instead of writing fake tool calls in assistant text.
- Older `system` messages from history are filtered out; only the latest one wins.
- Current turn data is added as the user message, not merged into the system prompt.

Write `llm_chat` docstrings as stable assistant policy and long-lived behavior. Put current task content in the function call arguments, not in the docstring.

### Prompt-writing implications
- For `llm_function`, think: function contract + execution strategy.
- For `llm_chat`, think: assistant identity + durable rules.
- Put tool-usage advice in tool `best_practices` when possible, not only in the main docstring.
- If you need to durably change a chat agent's policy mid-run, use the latest history `system` message or self-reference system-prompt helpers instead of trying to mutate old docstrings.

## Fast start modes
- Project mode: load models from `provider.json` with `OpenAICompatible.load_from_json_file(...)` when you have shared config or multiple models.
- Instant mode: write a tiny `python - <<'PY'` snippet, construct `APIKeyPool` + `OpenAICompatible` directly, decorate one function, and call it immediately.
- Prefer instant mode for quick shell usage, generated scripts, demos, and one-off local agents.

## Export the packaged skill

After installing `SimpleLLMFunc`, export the bundled Agent Skills with:

```bash
simplellmfunc-skill usage ~/.config/opencode/skills
simplellmfunc-skill developer ~/.config/opencode/skills
```

- `usage` exports the `simplellmfunc` folder.
- `developer` exports the `simplellmfunc-developer` folder.
- The second argument is the parent directory that receives the exported skill folder.
- Add `--force` if you need to overwrite an existing exported copy.

## Configuration essentials

### Minimal `provider.json` shape

SimpleLLMFunc expects `provider.json` to be:

```json
{
  "openrouter": [
    {
      "model_name": "z-ai/glm-5",
      "api_keys": ["sk-key-1", "sk-key-2"],
      "base_url": "https://openrouter.ai/api/v1",
      "max_retries": 5,
      "retry_delay": 1.0,
      "rate_limit_capacity": 20,
      "rate_limit_refill_rate": 3.0
    }
  ]
}
```

- Top level = provider id -> model config list.
- Lookup shape after loading = `providers[provider_id][model_name]`.
- Start from `examples/provider_template.json` when possible.

### `.env` and environment variables

The framework mainly reads `.env` / environment variables for logging and optional Langfuse observability.

```bash
LOG_LEVEL=WARNING
LOG_DIR=logs

LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_EXPORT_ALL_SPANS=true
LANGFUSE_ENABLED=true
```

- Precedence is: runtime environment variables -> `.env` -> framework defaults.
- Recommended default: `LOG_LEVEL=WARNING` to reduce noisy framework logs during normal agent usage.
- `provider.json` is the main model/provider config file; `.env` is not the primary place for provider definitions in project mode.
- For shell-first one-offs, direct constructor snippets are still fine even without `provider.json`.

## Default workflow
1. Choose the right surface:
   - `@llm_function` for one typed call.
   - `@llm_chat` for multi-turn agent behavior.
   - `@tool` for external capabilities the model may call.
   - `PyRepl` when the model needs persistent code execution or runtime primitives.
2. Define everything as `async def`.
3. Write a precise docstring prompt and leave the body as `pass`.
4. Use explicit parameter types and a typed return value; prefer Pydantic for structured outputs.
5. Build the model either from `provider.json` or directly with `APIKeyPool` + `OpenAICompatible`.
6. Add `toolkit=[...]` only when the task truly needs tools.
7. Validate with a focused example or event consumer.

## Best-practice patterns

### Instant shell-first usage (Recommended for one-offs)

Use the direct constructor path when you want to turn SimpleLLMFunc into a shell ability with almost no setup besides pasting your literal model settings.

```bash
python - <<'PY'
import asyncio

from SimpleLLMFunc import APIKeyPool, OpenAICompatible, llm_function


llm = OpenAICompatible(
    api_key_pool=APIKeyPool(
        api_keys=["sk-your-key"],
        provider_id="openrouter-z-ai-glm-5",
    ),
    model_name="z-ai/glm-5",
    base_url="https://openrouter.ai/api/v1",
)


@llm_function(llm_interface=llm)
async def answer(question: str) -> str:
    """Answer the question in a compact, practical way."""
    pass


print(asyncio.run(answer("Give me three uses of SimpleLLMFunc.")))
PY
```

For a shell agent with file tools and REPL, see `reference/instant-use.md` and `examples/instant_chat_agent.py`.

### Typed `llm_function` (Recommended)

```python
import asyncio

from pydantic import BaseModel, Field

from SimpleLLMFunc import OpenAICompatible, llm_function


class SentimentReport(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0.0 to 1.0 confidence score")
    summary: str = Field(description="one-sentence explanation")


models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openrouter"]["z-ai/glm-5"]


@llm_function(llm_interface=llm)
async def classify_sentiment(text: str) -> SentimentReport:
    """
    Classify the sentiment of the input text.

    Args:
        text: The user text to analyze.

    Returns:
        A structured sentiment report.
    """
    pass


async def main() -> None:
    result = await classify_sentiment("The setup was rough, but the product is excellent.")
    print(result.model_dump())


asyncio.run(main())
```

### Chat agent with a tool

```python
import asyncio

from SimpleLLMFunc import OpenAICompatible, llm_chat, tool


@tool
async def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: First factor.
        b: Second factor.

    Returns:
        Product of the two inputs.
    """
    return a * b


models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openrouter"]["z-ai/glm-5"]


@llm_chat(llm_interface=llm, toolkit=[multiply], stream=True)
async def tutor(message: str, history: list[dict[str, str]] | None = None):
    """
    You are a concise math tutor.
    Use the multiply tool when arithmetic is requested.
    """
    pass


async def main() -> None:
    history: list[dict[str, str]] = []
    async for chunk, history in tutor("What is 12.5 times 8?", history):
        if chunk:
            print(chunk, end="")


asyncio.run(main())
```

### Persistent runtime with `PyRepl`

```python
import asyncio

from SimpleLLMFunc.builtin import PyRepl, SelfReference


MEMORY_KEY = "agent_main"


async def main() -> None:
    repl = PyRepl()
    selfref = repl.get_runtime_backend("selfref")
    assert isinstance(selfref, SelfReference)

    selfref.bind_history(
        MEMORY_KEY,
        [{"role": "system", "content": "Answer in bullet points."}],
    )

    result = await repl.execute(
        "runtime.selfref.history.append({'role': 'user', 'content': 'remember this'})\n"
        "print(runtime.selfref.history.count())"
    )
    print(result["stdout"])


asyncio.run(main())
```

## Hard rules and gotchas
- Default to `async def` for all decorators. `@tool` enforces async directly.
- The function body does not implement behavior; the docstring does.
- The docstring is usually only the base material for prompt construction, not the entire final system prompt.
- For `llm_chat`, name the history parameter `history` or `chat_history`.
- The current direct-construction path for `OpenAICompatible` requires `APIKeyPool`; do not assume a simplified `api_key=` constructor exists.
- Instant snippets can call the decorated function directly at top level with `asyncio.run(...)`; no `__main__` guard is required.
- `max_tool_calls=None` means no framework-imposed tool-call cap. Set an explicit integer if you need a guardrail.
- Complex structured outputs are parsed from XML-oriented contracts internally. Do not manually force JSON unless you intentionally want plain-text behavior.
- `llm_chat(enable_event=True)` yields `ReactOutput` events and responses, not `(chunk, history)` tuples.
- `too_long_to_file=True` keeps roughly the first 20000 tokens in chat and writes the full tool result to a temp file.
- `PyRepl.reset()` clears REPL variables but keeps runtime backends and self-reference memory.
- `FileToolset` is workspace-scoped and read-before-write guarded.

## Load more context only when needed
- Philosophy and core concepts: `reference/philosophy-and-concepts.md`
- System prompt construction and prompt-writing rules: `reference/system-prompt-construction.md`
- Instant shell-first setup and constructor usage: `reference/instant-use.md`
- Provider and environment setup: `reference/configuration.md`
- Decorators, tools, file tools, and event mode: `reference/decorators-and-tools.md`
- PyRepl, runtime primitives, and selfref: `reference/pyrepl-runtime.md`
- Non-obvious behavior: `reference/gotchas.md`
- Mirrored repo docs: `reference/docs-source/quickstart.md`, `reference/docs-source/guide.md`, `reference/docs-source/detailed_guide/`
- Instant heredoc examples: `examples/instant_llm_function.py`, `examples/instant_chat_agent.py`
- Real repo examples: `examples/agent_as_tool_example.py`, `examples/llm_function_pydantic_example.py`, `examples/runtime_primitives_basic_example.py`, `examples/tui_general_agent_example.py`
