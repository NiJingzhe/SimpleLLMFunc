# Project Map

## Top-level structure

```text
SimpleLLMFunc/
├── SimpleLLMFunc/   # framework package
├── tests/           # pytest suite
├── examples/        # runnable examples
├── mintlify_docs/   # Mintlify documentation source
├── spec/            # repo-specific maps and conventions
└── skills/          # agent skills
```

## Package map

### `SimpleLLMFunc/llm_decorator/`

Primary user entrypoints:

- `llm_function_decorator.py`
- `llm_chat_decorator.py`
- `steps/common/`
- `steps/function/`
- `steps/chat/`
- `utils/tools.py`

This layer turns typed Python call signatures into prompt-building, ReAct execution, and response parsing.

### `SimpleLLMFunc/base/`

Core engine and lower-level primitives:

- `ReAct.py`
- `post_process.py`
- `messages/`
- `tool_call/`
- `type_resolve/`

When behavior seems "framework magical," the truth usually lives here.

### `SimpleLLMFunc/runtime/`

Runtime primitive system:

- primitive registry
- primitive contracts and docstring parsing
- backend lifecycle and fork behavior

This layer powers the `runtime.*` namespace used inside `PyRepl`.

### `SimpleLLMFunc/builtin/`

End-user built-ins:

- `pyrepl.py`
- `file_tools.py`
- `self_reference.py`

These builtins expose high-level behavior on top of lower-level runtime and tool plumbing.

### `SimpleLLMFunc/hooks/`

Streaming and event infrastructure:

- event types
- event emitters
- event wrappers such as `EventYield` and `ResponseYield`
- abort signaling

### `SimpleLLMFunc/interface/`

Provider-facing integration:

- `llm_interface.py`
- `openai_compatible.py`
- `key_pool.py`
- `token_bucket.py`

### `SimpleLLMFunc/logger/` and `SimpleLLMFunc/observability/`

Logging, trace propagation, and Langfuse support.

### `SimpleLLMFunc/utils/`

Support layers such as:

- `utils/tui/`
- `utils/stdio/`

## How to navigate changes

- Decorator behavior bug: start in `llm_decorator/`, then trace into `base/`.
- Tool schema or prompt-injection issue: start in `tool/tool.py` and `llm_decorator/utils/tools.py`.
- Runtime primitive issue: start in `runtime/primitives.py`, then inspect `builtin/pyrepl.py` or the relevant builtin backend.
- Event-stream issue: start in `hooks/`, then inspect decorator step modules.
- Provider transport issue: start in `interface/openai_compatible.py` plus related tests.
- User-facing docs mismatch: source code and tests win; then patch `mintlify_docs/` and maybe `README.md`.
