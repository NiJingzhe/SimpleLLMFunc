# Style And Spec

## Code style

- Follow PEP 8 and the repository's existing formatting.
- Preserve type hints on public APIs.
- Prefer clear names over clever abstractions.
- Keep related behavior near the owning module instead of scattering cross-cutting logic prematurely.

## Naming rules

- files: `snake_case`
- functions: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_SNAKE_CASE`

These conventions are stated in `spec/overall-spec.md` and reflected in the codebase.

## Docstring rules

- Public APIs should have useful docstrings.
- Tool parameter descriptions are parsed from docstring `Args:` or `Parameters:` blocks.
- Runtime primitive contracts are also docstring-driven.
- Primitive `Best Practices` are mandatory and enforced by tests.

## Spec rules

When editing `spec/` documents:

- Respect the `DOC_SUMMARY`, `DOC_MAP`, and optional `DOC_META` structure.
- Keep project map paths aligned with the actual tree.
- Update spec docs only when architecture, conventions, or module responsibilities actually changed.

## Documentation stack

- Sphinx
- MyST Markdown
- localized source/build setup
- Read the Docs configuration

When behavior changes for users, update `docs/source/` first; then align higher-level summaries like `README.md` if needed.
