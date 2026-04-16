# Framework Gotchas

## Source of truth

- Source and tests beat stale docs.
- Some user docs and README language are broader or older than the current implementation.

## Important mismatches and subtleties

- Structured `llm_function` outputs are XML-backed internally for complex types.
- The current direct `OpenAICompatible` and `OpenAIResponsesCompatible` constructors require `APIKeyPool`; if a README snippet shows `api_key=` / `model=`, treat it as stale and trust source code plus tests instead.
- Async guidance is stricter in docs and conventions than in every implementation guard, but async-first is still the right rule for new work.
- `max_tool_calls=None` is intentionally the default for both `llm_function` and `llm_chat`.
- `llm_chat` history behavior is opinionated: `history` and `chat_history` are special names, and the newest `system` message overrides the docstring prompt.
- `PyRepl.reset()` only clears REPL variables; it does not wipe runtime backends or self-reference memory.
- `runtime.selfref.context.compact(...)` is not a finalize-only effect anymore. The queue is committed after the current tool batch when possible so the next same-turn model step sees compacted context; finalize is the last-chance commit point.
- `llm_chat` selfref integration is split across layers now: decorator wiring, lifecycle sync hooks, and selfref state/context code. Avoid pushing all of that behavior back into one file.
- `OpenAIResponsesCompatible` is a separate adapter. Keep Responses API request shape, streamed tool-call chunk adaptation, system->`instructions` mapping, and reasoning handling there instead of pushing provider-specific logic into `ReAct`.
- Selfref fork child context is intentionally built from the pre-fork snapshot. Removing only one fork tool call is insufficient when the terminal assistant message contains multiple pending `tool_calls`; the whole terminal pending tool-calls message must stay out of child history.
- `runtime.selfref.fork.gather_all(...)` now exposes both `response` and `result`. Treat that alias as an ergonomics contract, not accidental duplication.
- `before_finalize` is a shared terminal hook contract. If you add a new ReAct terminal path, route it through the common finalize helper or you will regress non-event / abort / max-tool-cap consistency.
- Multimodal tool results may appear as assistant/user message pairs instead of plain tool messages because OpenAI tool messages cannot carry images.

## Runtime primitive contract rules

- Primitive names and pack names are normalized and validated.
- Primitive docstrings are not decorative; they are part of the runtime-discovery contract.
- Missing primitive `Best Practices` is a hard registration failure.

## Docs and sample config gotchas

- `examples/provider_template.json` is the safe reference template.
- Treat any local checked-in `provider.json` as environment-specific, not as the canonical sample.
- Docs source files are largely Chinese markdown; rely on code and tests for exact semantics when needed.
