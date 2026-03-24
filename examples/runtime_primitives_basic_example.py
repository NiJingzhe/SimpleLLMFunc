"""Local walkthrough for runtime memory primitives.

Run:
    poetry run python examples/runtime_primitives_basic_example.py

This example does not require any LLM provider. It demonstrates how to:
1. Create and bind a SelfReference memory key.
2. Install SelfReference primitive pack into PyRepl.
3. Declare a custom PrimitivePack with backend-aware handlers.
4. Append durable memory into system prompt via runtime.selfref.history.append_system_prompt.
5. Perform common memory CRUD operations through runtime.selfref.history.* primitives.

Runtime self-reference primitive quick reference:
- runtime.list_primitives()
- runtime.list_primitive_specs()
- runtime.selfref.history.keys()
- runtime.selfref.history.active_key()
- runtime.selfref.history.count(key=None)
- runtime.selfref.history.all(key=None)
- runtime.selfref.history.get(index, key=None)
- runtime.selfref.history.append(message, key=None)
- runtime.selfref.history.insert(index, message, key=None)
- runtime.selfref.history.update(index, message, key=None)
- runtime.selfref.history.delete(index, key=None)
- runtime.selfref.history.replace(messages, key=None)
- runtime.selfref.history.clear(key=None)
- runtime.selfref.history.get_system_prompt(key=None)
- runtime.selfref.history.set_system_prompt(text, key=None)
- runtime.selfref.history.append_system_prompt(text, key=None)
"""

from __future__ import annotations

import asyncio

from SimpleLLMFunc.builtin import PyRepl, SelfReference


MEMORY_KEY = "agent_main"


async def main() -> None:
    self_reference = SelfReference()
    self_reference.bind_history(
        MEMORY_KEY,
        [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "hello from initial history"},
        ],
    )

    repl = PyRepl()
    repl.install_primitive_pack("selfref", backend=self_reference)

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
runtime.selfref.history.append_system_prompt(
    "User preference: answer in bullet points.",
)
runtime.selfref.history.append_system_prompt(
    "Memory: keep every answer concise.",
)
print("updated system prompt:")
print(runtime.selfref.history.get_system_prompt())
"""

    result_1 = await repl.execute(code_append_system_memory)

    print("\nStep 1 result (append to system prompt memory):")
    print({"success": result_1["success"], "error": result_1["error"]})
    print(result_1["stdout"].rstrip())

    code_memory_crud = """
runtime.selfref.history.append(
    {"role": "assistant", "content": "Noted, I will follow your preferences."},
)
runtime.selfref.history.insert(
    1,
    {"role": "user", "content": "Inserted memory example"},
)
runtime.selfref.history.update(
    2,
    {"role": "user", "content": "Updated original user memory"},
)
runtime.selfref.history.delete(1)

count = runtime.selfref.history.count()
print("count:", count)
print("last message:", runtime.selfref.history.get(count - 1))
print("active key:", runtime.selfref.history.active_key())
print("keys:", runtime.selfref.history.keys())
"""

    result_2 = await repl.execute(code_memory_crud)

    print("\nStep 2 result (memory CRUD operations):")
    print({"success": result_2["success"], "error": result_2["error"]})
    print(result_2["stdout"].rstrip())

    print("\nAfter execute_code:")
    print(self_reference.snapshot_history(MEMORY_KEY))


if __name__ == "__main__":
    asyncio.run(main())
