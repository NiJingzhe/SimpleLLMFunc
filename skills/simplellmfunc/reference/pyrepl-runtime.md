# PyRepl Runtime

## When to use `PyRepl`

Use `PyRepl` when the model needs:

- persistent Python execution across turns
- runtime primitive discovery through `runtime.*`
- durable self-reference memory
- a forkable execution environment for sub-agents or parallel work

`PyRepl` is not just a one-shot shell command wrapper. It keeps state in a dedicated Python subprocess.

## Core workflow

Use this sequence whenever the model needs runtime help:

1. Discover capabilities with `runtime.list_primitives()`.
2. Inspect one contract with `runtime.get_primitive_spec(name)`.
3. Execute small, focused code snippets with `execute_code`.
4. Persist memory through `runtime.selfref.history.*` only when you truly need durable context.

## Self-reference basics

`PyRepl()` installs the built-in `selfref` backend by default.

Useful calls:

- `runtime.selfref.history.keys()`
- `runtime.selfref.history.active_key()`
- `runtime.selfref.history.all()`
- `runtime.selfref.history.append(message)`
- `runtime.selfref.history.set_system_prompt(text)`
- `runtime.selfref.history.append_system_prompt(text)`

Good pattern:

- Keep a stable memory key such as `agent_main`.
- Append durable preferences only when they should survive future turns.
- Keep normal per-turn reasoning out of durable memory.

## Forking

Fork-related primitives let an agent split work into child contexts.

Typical pattern:

1. Discover fork contract details with `runtime.get_primitive_spec("selfref.fork.spawn")`.
2. Spawn isolated subtasks.
3. Gather results with `runtime.selfref.fork.gather_all()`.

Use forks only for concrete, isolated, verifiable subtasks. Do not fork tiny inline work.

## Safety rules

- Execute small code blocks, not giant scripts.
- Print checkpoints and short summaries instead of dumping huge objects.
- Prefer runtime discovery before assuming a primitive contract.
- Treat REPL state as persistent and cumulative across calls.

## Important reset behavior

`reset_repl` clears REPL variables, but it does not wipe installed runtime backends or self-reference memory. If you need to forget durable memory, use the `runtime.selfref.history.*` APIs directly.
