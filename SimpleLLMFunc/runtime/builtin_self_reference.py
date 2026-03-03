"""Builtin runtime primitive pack backed by ``SelfReference``."""

from __future__ import annotations

from typing import Any, Callable, Optional

from SimpleLLMFunc.self_reference import SelfReference

from .primitives import PrimitiveCallContext, PrimitiveRegistry


def register_self_reference_primitives(
    registry: PrimitiveRegistry,
    *,
    get_self_reference: Callable[[], Optional[SelfReference]],
    replace: bool = False,
) -> None:
    """Register memory/fork primitives backed by ``SelfReference``.

    Notes:
    - This function only installs primitive handlers.
    - The caller controls lifecycle of the ``SelfReference`` instance via
      ``get_self_reference``.
    """

    def require_self_reference() -> SelfReference:
        self_reference = get_self_reference()
        if self_reference is None:
            raise RuntimeError(
                "No self_reference backend registered. "
                "Install the 'self_reference' primitive pack first."
            )
        return self_reference

    def get_memory_handle(key: str):
        return require_self_reference().memory[key]

    def memory_keys(_ctx: PrimitiveCallContext) -> list[str]:
        return require_self_reference().memory.keys()

    def memory_count(_ctx: PrimitiveCallContext, key: str) -> int:
        return get_memory_handle(key).count()

    def memory_all(_ctx: PrimitiveCallContext, key: str) -> list[dict[str, Any]]:
        return get_memory_handle(key).all()

    def memory_get(
        _ctx: PrimitiveCallContext,
        key: str,
        index: int,
    ) -> dict[str, Any]:
        return get_memory_handle(key).get(index)

    def memory_append(
        _ctx: PrimitiveCallContext,
        key: str,
        message: dict[str, Any],
    ) -> None:
        get_memory_handle(key).append(message)

    def memory_insert(
        _ctx: PrimitiveCallContext,
        key: str,
        index: int,
        message: dict[str, Any],
    ) -> None:
        get_memory_handle(key).insert(index, message)

    def memory_update(
        _ctx: PrimitiveCallContext,
        key: str,
        index: int,
        message: dict[str, Any],
    ) -> None:
        get_memory_handle(key).update(index, message)

    def memory_delete(
        _ctx: PrimitiveCallContext,
        key: str,
        index: int,
    ) -> None:
        get_memory_handle(key).delete(index)

    def memory_replace(
        _ctx: PrimitiveCallContext,
        key: str,
        messages: list[dict[str, Any]],
    ) -> None:
        get_memory_handle(key).replace(messages)

    def memory_clear(_ctx: PrimitiveCallContext, key: str) -> None:
        get_memory_handle(key).clear()

    def memory_get_system_prompt(
        _ctx: PrimitiveCallContext,
        key: str,
    ) -> Optional[str]:
        return get_memory_handle(key).get_system_prompt()

    def memory_set_system_prompt(
        _ctx: PrimitiveCallContext,
        key: str,
        text: str,
    ) -> None:
        get_memory_handle(key).set_system_prompt(text)

    def memory_append_system_prompt(
        _ctx: PrimitiveCallContext,
        key: str,
        text: str,
    ) -> None:
        get_memory_handle(key).append_system_prompt(text)

    def fork_is_bound(_ctx: PrimitiveCallContext) -> bool:
        return require_self_reference().instance.is_bound()

    async def fork_run(
        ctx: PrimitiveCallContext,
        *agent_args: Any,
        **agent_kwargs: Any,
    ) -> dict[str, Any]:
        call_kwargs = dict(agent_kwargs)
        call_kwargs.pop("_event_emitter", None)
        return await require_self_reference().instance.fork(
            *agent_args,
            _event_emitter=ctx.event_emitter,
            **call_kwargs,
        )

    async def fork_spawn(
        ctx: PrimitiveCallContext,
        *agent_args: Any,
        **agent_kwargs: Any,
    ) -> dict[str, Any]:
        call_kwargs = dict(agent_kwargs)
        call_kwargs.pop("_event_emitter", None)
        return await require_self_reference().instance.fork_spawn(
            *agent_args,
            _event_emitter=ctx.event_emitter,
            **call_kwargs,
        )

    async def fork_wait(
        _ctx: PrimitiveCallContext,
        fork_id: str,
    ) -> dict[str, Any]:
        return await require_self_reference().instance.fork_wait(fork_id)

    async def fork_wait_all(
        _ctx: PrimitiveCallContext,
        fork_ids: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Any]]:
        return await require_self_reference().instance.fork_wait_all(fork_ids)

    registry.register(
        "memory.keys",
        memory_keys,
        description="List all bound memory keys.",
        replace=replace,
    )
    registry.register(
        "memory.count",
        memory_count,
        description="Return message count for one memory key.",
        replace=replace,
    )
    registry.register(
        "memory.all",
        memory_all,
        description="Return all messages for one memory key.",
        replace=replace,
    )
    registry.register(
        "memory.get",
        memory_get,
        description="Read one message by index.",
        replace=replace,
    )
    registry.register(
        "memory.append",
        memory_append,
        description="Append one message to memory.",
        replace=replace,
    )
    registry.register(
        "memory.insert",
        memory_insert,
        description="Insert one message at index.",
        replace=replace,
    )
    registry.register(
        "memory.update",
        memory_update,
        description="Replace one message at index.",
        replace=replace,
    )
    registry.register(
        "memory.delete",
        memory_delete,
        description="Delete one message by index.",
        replace=replace,
    )
    registry.register(
        "memory.replace",
        memory_replace,
        description="Replace full message history.",
        replace=replace,
    )
    registry.register(
        "memory.clear",
        memory_clear,
        description="Clear all messages for one key.",
        replace=replace,
    )
    registry.register(
        "memory.get_system_prompt",
        memory_get_system_prompt,
        description="Get latest system prompt text.",
        replace=replace,
    )
    registry.register(
        "memory.set_system_prompt",
        memory_set_system_prompt,
        description="Overwrite system prompt text.",
        replace=replace,
    )
    registry.register(
        "memory.append_system_prompt",
        memory_append_system_prompt,
        description="Append system prompt text.",
        replace=replace,
    )
    registry.register(
        "fork.is_bound",
        fork_is_bound,
        description="Check whether recursive fork is available.",
        replace=replace,
    )
    registry.register(
        "fork.run",
        fork_run,
        description="Run one child fork and wait for result.",
        replace=replace,
    )
    registry.register(
        "fork.spawn",
        fork_spawn,
        description="Spawn one child fork asynchronously.",
        replace=replace,
    )
    registry.register(
        "fork.wait",
        fork_wait,
        description="Wait for one spawned fork result.",
        replace=replace,
    )
    registry.register(
        "fork.wait_all",
        fork_wait_all,
        description="Wait for multiple fork results.",
        replace=replace,
    )


__all__ = ["register_self_reference_primitives"]
