# Documentation Update Checklist

Use this when a framework change affects users or contributors.

## User-facing behavior changed

- Update the closest page under `mintlify_docs/`
- Update `README.md` if the high-level promise or quick-start path changed
- Update or add an example under `examples/`

## Architecture or contributor workflow changed

- Update `spec/project-map.md`
- Update `spec/overall-spec.md` if repo-wide conventions changed
- Update `AGENT.md` if contributor workflow changed

## Runtime primitive authoring changed

- Update `mintlify_docs/detailed_guide/primitive.mdx`
- Add or adjust a runtime example
- Verify tests still cover docstring contract rules

## Final consistency check

- Docs match current source behavior
- Examples use current API names and signatures
- No old wording contradicts new semantics nearby
