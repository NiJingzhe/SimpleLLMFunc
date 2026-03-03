"""Local walkthrough for runtime memory primitives.

Run:
    poetry run python examples/runtime_primitives_basic_example.py

This example does not require any LLM provider. It demonstrates how to:
1. Create and bind a SelfReference memory key.
2. Install SelfReference primitive pack into PyRepl.
3. Register a custom runtime backend and primitive.
4. Append durable memory into system prompt via runtime.memory.append_system_prompt.
5. Perform common memory CRUD operations through runtime.memory.* primitives.

Runtime memory primitive quick reference:
- runtime.memory.keys()
- runtime.memory.count(key)
- runtime.memory.all(key)
- runtime.memory.get(key, index)
- runtime.memory.append(key, message)
- runtime.memory.insert(key, index, message)
- runtime.memory.update(key, index, message)
- runtime.memory.delete(key, index)
- runtime.memory.replace(key, messages)
- runtime.memory.clear(key)
- runtime.memory.get_system_prompt(key)
- runtime.memory.set_system_prompt(key, text)
- runtime.memory.append_system_prompt(key, text)
"""

from __future__ import annotations

import asyncio

from SimpleLLMFunc import SelfReference
from SimpleLLMFunc.builtin import PyRepl


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
    repl.install_primitive_pack("self_reference", backend=self_reference)

    repl.register_runtime_backend(
        "constants",
        {
            "app_name": "SimpleLLMFunc",
            "memory_key": MEMORY_KEY,
        },
        replace=True,
    )

    def constants_get(_ctx, key: str):
        constants = repl.get_runtime_backend("constants")
        if not isinstance(constants, dict):
            raise RuntimeError("runtime backend 'constants' must be a dict")
        return constants.get(key)

    repl.register_primitive(
        "constants.get",
        constants_get,
        description="Read one value from constants runtime backend.",
        replace=True,
    )

    print("Before execute_code:")
    print(self_reference.snapshot_history(MEMORY_KEY))

    code_runtime_overview = """
print("backends:", runtime.list_backends())
print("has constants.get:", "constants.get" in runtime.list_primitives())
print("app_name:", runtime.constants.get("app_name"))
"""

    result_0 = await repl.execute(code_runtime_overview)

    print("\nStep 0 result (runtime backend/primitive registration):")
    print({"success": result_0["success"], "error": result_0["error"]})
    print(result_0["stdout"].rstrip())

    code_append_system_memory = """
runtime.memory.append_system_prompt(
    "agent_main",
    "User preference: answer in bullet points.",
)
runtime.memory.append_system_prompt(
    "agent_main",
    "Memory: keep every answer concise.",
)
print("updated system prompt:")
print(runtime.memory.get_system_prompt("agent_main"))
"""

    result_1 = await repl.execute(code_append_system_memory)

    print("\nStep 1 result (append to system prompt memory):")
    print({"success": result_1["success"], "error": result_1["error"]})
    print(result_1["stdout"].rstrip())

    code_memory_crud = """
runtime.memory.append(
    "agent_main",
    {"role": "assistant", "content": "Noted, I will follow your preferences."},
)
runtime.memory.insert(
    "agent_main",
    1,
    {"role": "user", "content": "Inserted memory example"},
)
runtime.memory.update(
    "agent_main",
    2,
    {"role": "user", "content": "Updated original user memory"},
)
runtime.memory.delete("agent_main", 1)

count = runtime.memory.count("agent_main")
print("count:", count)
print("last message:", runtime.memory.get("agent_main", count - 1))
print("keys:", runtime.memory.keys())
"""

    result_2 = await repl.execute(code_memory_crud)

    print("\nStep 2 result (memory CRUD operations):")
    print({"success": result_2["success"], "error": result_2["error"]})
    print(result_2["stdout"].rstrip())

    print("\nAfter execute_code:")
    print(self_reference.snapshot_history(MEMORY_KEY))


if __name__ == "__main__":
    asyncio.run(main())
