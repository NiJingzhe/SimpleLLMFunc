---
name: simplellmfunc-developer
description: "Develop and maintain the SimpleLLMFunc framework itself. Use when changing framework internals, tests, docs, specs, runtime primitives, decorator behavior, tool plumbing, event streams, PyRepl integration, provider adapters such as OpenAICompatible/OpenAIResponsesCompatible, or contributor-facing project structure and conventions."
license: MIT
compatibility: "Python 3.12+ repo with pytest, Poetry, Mintlify docs, and SimpleLLMFunc source tree available."
metadata:
  project: SimpleLLMFunc
  version: "0.7.7"
---

# SimpleLLMFunc Framework Development

## When to use this skill
- Use this skill when the task changes the framework itself, not just an app built on it.
- Typical triggers: editing `SimpleLLMFunc/`, `tests/`, `docs/`, `spec/`, built-in tools, runtime primitives, decorator semantics, event-stream behavior, provider adapters, TUI utilities, or contributor docs.

## Core development philosophy
- Preserve the framework's function-first design: `LLM as Function`, `Prompt as Code`, `Code as Doc`.
- Keep public behavior explicit and typed.
- Favor small, composable modules over hidden orchestration.
- Prefer explicit boundaries between pure transforms, state mutation, and orchestration side effects. Recent selfref/ReAct work depends on keeping those lines sharp.
- Follow repo-grounded conventions instead of generic framework habits.
- When docs and code disagree, source and tests are the final authority.

## Default implementation workflow
1. Read the relevant docs, tests, and source before changing behavior.
2. Map the affected layer: decorator, base engine, runtime, tooling, interface, hooks, or docs/spec.
3. Write or update tests first for the behavior you are changing.
4. Make the smallest coherent implementation change.
5. Run targeted tests, then broader tests if the change touches shared behavior.
6. Update docs/examples/specs when user-facing behavior or architecture changed.

## TDD and validation loop
- Start with a failing or missing test that captures the new behavior.
- Use red -> green -> refactor.
- Prefer focused unit tests in the mirrored `tests/` location.
- Add or update a runnable example when the feature is user-facing.
- If behavior changes affect docs or spec, update them in the same change.

## Project map
- `SimpleLLMFunc/llm_decorator/`: public decorator entrypoints and stepwise orchestration.
- `SimpleLLMFunc/base/`: ReAct loop, message handling, structured parsing, tool-call execution.
- `SimpleLLMFunc/runtime/`: primitive registry, backend lifecycle, runtime call context, and selfref state/context transforms.
- `SimpleLLMFunc/builtin/`: user-facing builtins such as `PyRepl`, `FileToolset`, and `SelfReference`.
- `SimpleLLMFunc/hooks/`: events, event bus, stream wrappers, abort support.
- `SimpleLLMFunc/interface/`: model interface abstractions and provider adapters such as `OpenAICompatible` and `OpenAIResponsesCompatible`.
- `SimpleLLMFunc/logger/` and `SimpleLLMFunc/observability/`: logs, trace context, Langfuse.
- `SimpleLLMFunc/utils/`: TUI and stdio helpers.
- `tests/`: mirror of behavior and architecture; often the fastest place to infer conventions.
- `mintlify_docs/`: Mintlify documentation source, including locale pages such as `en/...`.
- `spec/`: higher-level project map and repo conventions.

## Naming and style rules
- File names: `snake_case`.
- Functions: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Public APIs should carry type annotations.
- Public user-facing callables should have useful docstrings.
- Follow the repo's existing formatting and PEP 8 style.

## Framework-specific development rules
- Prefer `async def` for decorated public patterns and tool implementations.
- Keep docstring-parsed contracts in sync with behavior. This matters for `@tool` and runtime primitives.
- Runtime primitive docstrings must include `Best Practices`; registration fails without them.
- Preserve history semantics in `llm_chat`: `history` and `chat_history` are special names.
- Preserve structured output parsing behavior unless the task explicitly changes it.
- For selfref work, keep pure context parsing/rendering in `runtime/selfref/context_ops.py`, stateful storage and mutation in `runtime/selfref/state.py`, and `llm_chat` lifecycle bridging in `llm_decorator/selfref_sync.py`.
- For provider work, keep wire-format differences in the adapter layer under `SimpleLLMFunc/interface/`; do not leak Responses-specific request/stream contracts into `ReAct` or decorator code unless the public framework contract is intentionally changing.
- `OpenAIResponsesCompatible` should remain a first-class adapter, not a special case hidden inside `ReAct`. System prompts map to Responses `instructions`, and Responses-specific reasoning/tool-stream handling belongs in the adapter.
- For ReAct work, treat `base/ReAct.py` as phase-based orchestration. New terminal behavior should flow through the shared finalize path so `before_finalize` stays consistent across event, non-event, abort, and max-tool-cap exits.
- For selfref fork work, child context should be built from the pre-fork snapshot. Do not reintroduce the parent's pending assistant tool-call message into child-visible history.
- Treat tests as executable API documentation for subtle cases like self-reference, event mode, and provider compatibility.

## Documentation and spec rules
- Update `mintlify_docs/` when user-facing behavior changes.
- Update `spec/` when module responsibilities, architecture map, or repo-wide guidance changes.
- Keep examples runnable and aligned with current behavior.
- Use progressive disclosure in skills and docs: concise guidance in the main file, details in reference docs.
- Keep `provider.json` format docs and `.env` / environment-variable docs aligned with actual loader and observability behavior.
- Keep the packaged `skills/` directory and the `simplellmfunc-skill` export CLI aligned so installed users can export the current skill contents correctly.
- When Responses adapter behavior or selfref fork behavior changes, update packaged `skills/` docs in the same change, not only `mintlify_docs/`.
- Treat `AGENTS.md` as a feedback-loop artifact: when recurring agent mistakes reveal missing environmental guidance, update the file so the fix lives in the system instead of only in maintainer memory.

## Read these reference docs as needed
- Architecture and contributor map: `reference/project-map.md`
- TDD, tests, and validation expectations: `reference/testing-and-tdd.md`
- Coding conventions and naming rules: `reference/style-and-spec.md`
- Framework-specific gotchas: `reference/framework-gotchas.md`
- Docs and examples workflow: `reference/docs-and-examples.md`
- Maintainer workflow notes: `reference/AGENT.md`
- Contributor guide: `reference/contributing.md`
- Mirrored repo spec: `reference/spec/project-map.md`, `reference/spec/overall-spec.md`, `reference/spec/primitive-dev-api-plan.md`, `reference/spec/meta.md`
- Developer examples: `examples/add_runtime_primitive_pattern.py`, `examples/test_first_decorator_change.md`, `examples/update_docs_checklist.md`
