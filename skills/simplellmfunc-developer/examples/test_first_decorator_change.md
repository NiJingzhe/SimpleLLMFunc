# Test-First Decorator Change

Use this checklist when changing `@llm_function` or `@llm_chat` behavior.

1. Identify the closest existing test file:
   - `tests/test_llm_function_decorator.py`
   - `tests/test_llm_chat_decorator.py`
2. Add a focused failing test for the new rule or regression.
3. Patch the smallest owning module first:
   - public decorator file if the rule is surface-level
   - step module if the rule belongs to prompt building, response parsing, or streaming
   - `base/` only if the behavior is truly engine-level
4. Re-run the targeted test file.
5. Run neighboring tests if the change touched shared logic.
6. Update docs if the user-visible contract changed.

Good smell:

- one new test or a small cluster of related tests
- one owning implementation change
- docs updated only if behavior changed externally

Bad smell:

- changing unrelated tests just to make them pass
- patching broad engine code when a step module owns the behavior
- skipping docs after changing visible semantics
