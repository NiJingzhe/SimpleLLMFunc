# Docs And Examples

## When docs must change

Update docs when you change:

- public decorator behavior
- tool semantics
- event-stream contracts
- provider configuration behavior
- provider adapter behavior such as OpenAI-compatible chat transport vs Responses transport
- `.env` / environment-variable behavior
- runtime primitive authoring or discovery
- PyRepl or self-reference workflows

## Where to update

- `mintlify_docs/`: user documentation
- `README.md`: high-level user-facing introduction and quick-start narrative
- `examples/`: runnable examples for visible features
- `spec/`: project map and repo-wide development guidance
- `AGENT.md`: contributor workflow guidance when relevant
- `skills/`: packaged agent-facing docs shipped by `simplellmfunc-skill`

Configuration-specific rule:

- If you change provider loading, model-selection behavior, or observability env handling, update both `provider.json` examples and `.env` / environment-variable docs together.
- If you change `OpenAIResponsesCompatible` or selfref fork contracts, update packaged examples and skill guidance together so exported skills stay truthful.

## Example quality rules

- Keep examples runnable.
- Use current APIs and current config-loading patterns.
- Favor small, focused examples over giant kitchen-sink scripts.
- If a feature needs an LLM provider, say so clearly.
- If a feature does not need a provider, prefer a no-provider example because it is easier to validate locally.
- If a feature has both chat/completions and Responses adapter paths, be explicit about which adapter the example is demonstrating.

## Documentation quality rules

- Be explicit about async usage.
- Call out gotchas near the feature, not only in one distant page.
- Prefer code snippets that match real repo conventions.
- If changing docs for one feature creates new inconsistency elsewhere, patch the nearby conflicting page in the same change if feasible.
- When fork result parsing changed, show the current field names (`status`, `response`, `result`) instead of leaving examples to guess.
