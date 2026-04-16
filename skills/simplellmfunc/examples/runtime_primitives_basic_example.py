"""Local walkthrough for runtime context primitives.

Run:
    poetry run python examples/runtime_primitives_basic_example.py

This example does not require any LLM provider. It demonstrates how to:
1. Create and bind a SelfReference memory key.
2. Start PyRepl with its builtin selfref backend.
3. Declare a custom PrimitivePack with backend-aware handlers.
4. Append durable experience into system context via runtime.selfref.context.remember.
5. Inspect context and queue a milestone compaction through runtime.selfref.context.* primitives.

Runtime self-reference primitive quick reference:
- runtime.list_primitives()
- runtime.list_primitive_specs()
- runtime.selfref.context.inspect(key=None)
- runtime.selfref.context.remember(text, key=None)
- runtime.selfref.context.forget(experience_id, key=None)
- runtime.selfref.context.compact(..., key=None)
"""

from __future__ import annotations

import asyncio

from SimpleLLMFunc.builtin import PyRepl, SelfReference


MEMORY_KEY = "agent_main"


async def main() -> None:
    repl = PyRepl()
    self_reference = repl.get_runtime_backend("selfref")
    if not isinstance(self_reference, SelfReference):
        raise RuntimeError("PyRepl builtin selfref backend is not available")

    self_reference.bind_history(
        MEMORY_KEY,
        [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "hello from initial history"},
        ],
    )

    constants = repl.pack(
        "constants",
        backend={
            "app_name": "SimpleLLMFunc",
            "memory_key": MEMORY_KEY,
        },
    )

    @constants.primitive(
        "get",
        description="Read one value from constants runtime backend.",
    )
    def constants_get(ctx, key: str):
        """
        Use: Read one value from constants backend.
        Input: `key: str`.
        Output: `str | None`.
        Best Practices:
        - Keep lookups to single keys.
        """
        backend = ctx.backend
        if not isinstance(backend, dict):
            raise RuntimeError("runtime backend 'constants' must be a dict")
        return backend.get(key)

    repl.install_pack(constants)

    print("Before execute_code:")
    print(self_reference.snapshot_history(MEMORY_KEY))

    code_runtime_overview = """
print("backends:", runtime.list_backends())
print("has constants.get:", "constants.get" in runtime.list_primitives())
print(
    "has constants description:",
    any(
        item.get("name") == "constants.get"
        for item in runtime.list_primitive_specs(format='dict')
    ),
)
print("app_name:", runtime.constants.get("app_name"))
"""

    result_0 = await repl.execute(code_runtime_overview)

    print("\nStep 0 result (runtime backend/primitive registration):")
    print({"success": result_0["success"], "error": result_0["error"]})
    print(result_0["stdout"].rstrip())

    code_append_system_memory = """
runtime.selfref.context.remember(
    "User preference: answer in bullet points.",
)
runtime.selfref.context.remember(
    "Keep every answer concise.",
)
snapshot = runtime.selfref.context.inspect()
print("durable experiences:")
for item in snapshot['experiences']:
    print(item['id'], item['text'])
"""

    result_1 = await repl.execute(code_append_system_memory)

    print("\nStep 1 result (append durable experience):")
    print({"success": result_1["success"], "error": result_1["error"]})
    print(result_1["stdout"].rstrip())

    code_memory_crud = """
snapshot_before = runtime.selfref.context.inspect()
print("message count before compact:", len(snapshot_before['messages']))

payload = runtime.selfref.context.compact(
    goal="Demo context cleanup",
    instruction="Show the current runtime context state after using a few primitives.",
    discoveries=[
        "The selfref backend is installed by default.",
        "Context inspection returns the compiled message list.",
    ],
    completed=[
        "Read a custom runtime constant.",
        "Stored durable experience.",
    ],
    current_status="Ready for the next turn with a compact checkpoint.",
    likely_next_work=[
        "Start the next milestone from the retained summary.",
    ],
    relevant_files_directories=[
        "skills/simplellmfunc/examples/runtime_primitives_basic_example.py",
    ],
)

print("compaction status:", payload['status'])
print(payload['assistant_message'])
"""

    result_2 = await repl.execute(code_memory_crud)

    print("\nStep 2 result (context inspection + compaction queue):")
    print({"success": result_2["success"], "error": result_2["error"]})
    print(result_2["stdout"].rstrip())

    self_reference.commit_pending_compaction(MEMORY_KEY)

    print("\nAfter execute_code:")
    print(self_reference.snapshot_history(MEMORY_KEY))


if __name__ == "__main__":
    asyncio.run(main())
