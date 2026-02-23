"""Local walkthrough for SelfReference memory APIs.

Run:
    poetry run python examples/self_reference_basic_example.py

This example does not require any LLM provider. It demonstrates how to:
1. Create and bind a SelfReference memory key.
2. Attach SelfReference to PyRepl explicitly.
3. Append durable memory into system prompt with append_system_prompt.
4. Perform common memory CRUD operations through memory handles.

Memory method purpose quick reference:
- count(): number of messages in current key
- all(): deep-copied full snapshot
- get(index): read one message
- append(message): append one message
- insert(index, message): insert one message
- update(index, message): replace one message
- delete(index): remove one message
- replace(messages): replace whole history
- clear(): clear all messages
- get_system_prompt(): read system prompt memory
- set_system_prompt(text): overwrite system prompt memory
- append_system_prompt(text): append to system prompt memory
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

    repl = PyRepl(self_reference=self_reference)
    execute_tool = next(tool for tool in repl.toolset if tool.name == "execute_code")

    print("Before execute_code:")
    print(self_reference.snapshot_history(MEMORY_KEY))

    code_append_system_memory = """
mem = self_reference.memory["agent_main"]
mem.append_system_prompt("User preference: answer in bullet points.")
mem.append_system_prompt("Memory: keep every answer concise.")
print("updated system prompt:")
print(mem.get_system_prompt())
"""

    result_1 = await execute_tool.func(code=code_append_system_memory)

    print("\nStep 1 result (append to system prompt memory):")
    print({"success": result_1["success"], "error": result_1["error"]})
    print(result_1["stdout"].rstrip())

    code_memory_crud = """
mem = self_reference.memory["agent_main"]

mem.append({"role": "assistant", "content": "Noted, I will follow your preferences."})
mem.insert(1, {"role": "user", "content": "Inserted memory example"})
mem.update(2, {"role": "user", "content": "Updated original user memory"})
mem.delete(1)

print("count:", mem.count())
print("last message:", mem.get(mem.count() - 1))
print("keys:", self_reference.memory.keys())
"""

    result_2 = await execute_tool.func(code=code_memory_crud)

    print("\nStep 2 result (memory CRUD operations):")
    print({"success": result_2["success"], "error": result_2["error"]})
    print(result_2["stdout"].rstrip())

    print("\nAfter execute_code:")
    print(self_reference.snapshot_history(MEMORY_KEY))


if __name__ == "__main__":
    asyncio.run(main())
