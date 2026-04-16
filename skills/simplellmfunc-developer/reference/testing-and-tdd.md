# Testing And TDD

## Repo expectations

The repo explicitly expects test-first or at least test-led changes.

Use this loop:

1. Understand the target behavior from docs, source, and existing tests.
2. Add or update a focused test.
3. Make the smallest implementation change to satisfy it.
4. Refactor while keeping tests green.

## Test organization

- Tests largely mirror source layout.
- Use targeted files under `tests/` instead of one giant integration test when possible.
- Async behavior is commonly tested with `@pytest.mark.asyncio`.
- Mocks are widely used for LLM transport and observability boundaries.

## Good test patterns in this repo

- Assert defaults explicitly, such as `max_tool_calls is None`.
- Capture subtle contract rules, such as `llm_chat` strict signature enforcement.
- Test lifecycle behavior for runtime backends and forks.
- Test failure modes, not just happy paths.
- Use fixtures from `tests/conftest.py` when they simplify repeated setup.
- For provider adapters, test adapter-specific request translation and stream-chunk normalization without moving provider-specific assertions into generic ReAct tests.

## What to test for common changes

### Decorator behavior
- default arguments
- prompt-building rules
- response-shape differences between normal mode and event mode
- self-reference integration
- history handling semantics
- lifecycle hook timing, especially tool-batch sync and `before_finalize`

### Tool behavior
- async enforcement
- docstring parameter extraction
- schema shape
- long-output truncation
- prompt-injection behavior

### Runtime primitives
- registration and contract parsing
- docstring requirements, especially `Best Practices`
- backend binding and lifecycle hooks
- fork cloning behavior
- pre-fork child history construction and terminal pending tool-call stripping when selfref fork behavior changes
- compact fork result payload shape, including `response` / `result` aliases and error payloads
- pure context parse/render helpers separately from stateful backend mutation when selfref behavior changes

### File tools
- workspace safety
- hash/stale-write protection
- hidden path rejection
- regex validation and path scoping

## Practical command guidance

Run focused tests first, then broaden when shared behavior is touched.

Typical examples:

```bash
poetry run pytest tests/test_llm_function_decorator.py
poetry run pytest tests/test_llm_chat_decorator.py
poetry run pytest tests/test_base/test_react.py
poetry run pytest tests/test_selfref_context_ops.py
poetry run pytest tests/test_self_reference.py
poetry run pytest tests/test_interface/test_openai_responses_compatible.py
poetry run pytest tests/test_runtime_primitives_docstring.py
poetry run pytest tests/test_builtin/test_file_tools.py
poetry run pytest
```

If a change also affects docs-facing behavior, run the relevant example manually when possible.
