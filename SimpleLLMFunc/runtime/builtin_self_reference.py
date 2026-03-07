"""Builtin runtime primitive pack backed by ``SelfReference``."""

from __future__ import annotations

from typing import Any, Callable, Optional

from SimpleLLMFunc.builtin.self_reference import SelfReference

from .primitives import PrimitiveCallContext, PrimitiveRegistry


def register_self_reference_primitives(
    registry: PrimitiveRegistry,
    *,
    get_self_reference: Callable[[], Optional[SelfReference]],
    replace: bool = False,
) -> None:
    """Register selfref history/fork primitives backed by ``SelfReference``.

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
                "Install the 'selfref' primitive pack first."
            )
        return self_reference

    def resolve_history_key(key: Optional[str] = None) -> str:
        return require_self_reference().resolve_history_key(key)

    def get_history_handle(key: Optional[str] = None):
        self_reference = require_self_reference()
        resolved_key = self_reference.resolve_history_key(key)
        return self_reference.memory[resolved_key]

    def selfref_history_keys(_ctx: PrimitiveCallContext) -> list[str]:
        return require_self_reference().memory.keys()

    def selfref_history_active_key(_ctx: PrimitiveCallContext) -> str:
        return resolve_history_key(None)

    def selfref_history_count(
        _ctx: PrimitiveCallContext,
        key: Optional[str] = None,
    ) -> int:
        return get_history_handle(key).count()

    def selfref_history_all(
        _ctx: PrimitiveCallContext,
        key: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return get_history_handle(key).all()

    def selfref_history_get(
        _ctx: PrimitiveCallContext,
        index: int,
        key: Optional[str] = None,
    ) -> dict[str, Any]:
        return get_history_handle(key).get(index)

    def selfref_history_append(
        _ctx: PrimitiveCallContext,
        message: dict[str, Any],
        key: Optional[str] = None,
    ) -> None:
        get_history_handle(key).append(message)

    def selfref_history_insert(
        _ctx: PrimitiveCallContext,
        index: int,
        message: dict[str, Any],
        key: Optional[str] = None,
    ) -> None:
        get_history_handle(key).insert(index, message)

    def selfref_history_update(
        _ctx: PrimitiveCallContext,
        index: int,
        message: dict[str, Any],
        key: Optional[str] = None,
    ) -> None:
        get_history_handle(key).update(index, message)

    def selfref_history_delete(
        _ctx: PrimitiveCallContext,
        index: int,
        key: Optional[str] = None,
    ) -> None:
        get_history_handle(key).delete(index)

    def selfref_history_replace(
        _ctx: PrimitiveCallContext,
        messages: list[dict[str, Any]],
        key: Optional[str] = None,
    ) -> None:
        get_history_handle(key).replace(messages)

    def selfref_history_clear(
        _ctx: PrimitiveCallContext,
        key: Optional[str] = None,
    ) -> None:
        get_history_handle(key).clear()

    def selfref_history_get_system_prompt(
        _ctx: PrimitiveCallContext,
        key: Optional[str] = None,
    ) -> Optional[str]:
        return get_history_handle(key).get_system_prompt()

    def selfref_history_set_system_prompt(
        _ctx: PrimitiveCallContext,
        text: str,
        key: Optional[str] = None,
    ) -> None:
        get_history_handle(key).set_system_prompt(text)

    def selfref_history_append_system_prompt(
        _ctx: PrimitiveCallContext,
        text: str,
        key: Optional[str] = None,
    ) -> None:
        get_history_handle(key).append_system_prompt(text)

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

    def selfref_fork_is_bound(_ctx: PrimitiveCallContext) -> bool:
        return fork_is_bound(_ctx)

    async def selfref_fork_run(
        ctx: PrimitiveCallContext,
        *agent_args: Any,
        **agent_kwargs: Any,
    ) -> dict[str, Any]:
        return await fork_run(ctx, *agent_args, **agent_kwargs)

    async def selfref_fork_spawn(
        ctx: PrimitiveCallContext,
        *agent_args: Any,
        **agent_kwargs: Any,
    ) -> dict[str, Any]:
        return await fork_spawn(ctx, *agent_args, **agent_kwargs)

    async def selfref_fork_wait(
        _ctx: PrimitiveCallContext,
        fork_id: str,
    ) -> dict[str, Any]:
        return await fork_wait(_ctx, fork_id)

    async def selfref_fork_wait_all(
        _ctx: PrimitiveCallContext,
        fork_ids: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Any]]:
        return await fork_wait_all(_ctx, fork_ids)

    selfref_best_practices = [
        "Each layer focuses on planning for its own scope; delegate concrete execution to child forks.",
        "When tasks are independent (no content dependency), spawn forks in parallel.",
        "Before forking, review and trim memory; summarize irrelevant context or dump it to files.",
        "In fork prompts, define completion boundaries and require explicit acceptance criteria; prefer file-based handoff plus parent-agent messaging over dumping everything in chat.",
        "After each milestone, review and reorganize memory before moving forward.",
    ]

    def _optional_history_key_param() -> dict[str, Any]:
        return {
            "name": "key",
            "type": "str | None",
            "required": False,
            "description": "Optional history key. When omitted, resolves from active selfref context.",
            "kind": "keyword_only",
            "default": "None",
        }

    def _register_selfref_primitive(
        name: str,
        handler: Any,
        *,
        description: str,
        input_type: str,
        output_type: str,
        parameters: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        registry.register(
            name,
            handler,
            description=description,
            input_type=input_type,
            output_type=output_type,
            parameters=parameters or [],
            best_practices=selfref_best_practices,
            replace=replace,
        )

    def selfref_guide(_ctx: PrimitiveCallContext) -> dict[str, Any]:
        return {
            "namespace": "selfref",
            "overview": (
                "Self-reference primitives provide memory/history management and "
                "recursive fork delegation for agent workflows."
            ),
            "best_practices": list(selfref_best_practices),
        }

    # Self-reference first-class namespace.
    _register_selfref_primitive(
        "selfref.guide",
        selfref_guide,
        description="Return selfref namespace overview and fork/memory best practices.",
        input_type="No arguments.",
        output_type="dict[str, Any]",
        parameters=[],
    )
    _register_selfref_primitive(
        "selfref.history.keys",
        selfref_history_keys,
        description="List all selfref history keys.",
        input_type="No arguments.",
        output_type="list[str]",
        parameters=[],
    )
    _register_selfref_primitive(
        "selfref.history.active_key",
        selfref_history_active_key,
        description="Resolve current selfref history key from active context/defaults.",
        input_type="No arguments.",
        output_type="str",
        parameters=[],
    )
    _register_selfref_primitive(
        "selfref.history.count",
        selfref_history_count,
        description="Return message count for selfref history.",
        input_type="Optional history key.",
        output_type="int",
        parameters=[_optional_history_key_param()],
    )
    _register_selfref_primitive(
        "selfref.history.all",
        selfref_history_all,
        description="Return all messages for selfref history.",
        input_type="Optional history key.",
        output_type="list[dict[str, Any]]",
        parameters=[_optional_history_key_param()],
    )
    _register_selfref_primitive(
        "selfref.history.get",
        selfref_history_get,
        description="Read one selfref history message by index.",
        input_type="Message index and optional history key.",
        output_type="dict[str, Any]",
        parameters=[
            {
                "name": "index",
                "type": "int",
                "required": True,
                "description": "Zero-based message index.",
                "kind": "positional_or_keyword",
            },
            _optional_history_key_param(),
        ],
    )
    _register_selfref_primitive(
        "selfref.history.append",
        selfref_history_append,
        description="Append one message to selfref history.",
        input_type="Message payload and optional history key.",
        output_type="None",
        parameters=[
            {
                "name": "message",
                "type": "dict[str, Any]",
                "required": True,
                "description": "Message object with role/content fields.",
                "kind": "positional_or_keyword",
            },
            _optional_history_key_param(),
        ],
    )
    _register_selfref_primitive(
        "selfref.history.insert",
        selfref_history_insert,
        description="Insert one selfref history message at index.",
        input_type="Index, message payload, and optional history key.",
        output_type="None",
        parameters=[
            {
                "name": "index",
                "type": "int",
                "required": True,
                "description": "Insert position for the new message.",
                "kind": "positional_or_keyword",
            },
            {
                "name": "message",
                "type": "dict[str, Any]",
                "required": True,
                "description": "Message object with role/content fields.",
                "kind": "positional_or_keyword",
            },
            _optional_history_key_param(),
        ],
    )
    _register_selfref_primitive(
        "selfref.history.update",
        selfref_history_update,
        description="Replace one selfref history message at index.",
        input_type="Index, message payload, and optional history key.",
        output_type="None",
        parameters=[
            {
                "name": "index",
                "type": "int",
                "required": True,
                "description": "Target message index to overwrite.",
                "kind": "positional_or_keyword",
            },
            {
                "name": "message",
                "type": "dict[str, Any]",
                "required": True,
                "description": "Replacement message object.",
                "kind": "positional_or_keyword",
            },
            _optional_history_key_param(),
        ],
    )
    _register_selfref_primitive(
        "selfref.history.delete",
        selfref_history_delete,
        description="Delete one selfref history message by index.",
        input_type="Message index and optional history key.",
        output_type="None",
        parameters=[
            {
                "name": "index",
                "type": "int",
                "required": True,
                "description": "Zero-based message index to delete.",
                "kind": "positional_or_keyword",
            },
            _optional_history_key_param(),
        ],
    )
    _register_selfref_primitive(
        "selfref.history.replace",
        selfref_history_replace,
        description="Replace full selfref history.",
        input_type="Full message list and optional history key.",
        output_type="None",
        parameters=[
            {
                "name": "messages",
                "type": "list[dict[str, Any]]",
                "required": True,
                "description": "Complete replacement history message list.",
                "kind": "positional_or_keyword",
            },
            _optional_history_key_param(),
        ],
    )
    _register_selfref_primitive(
        "selfref.history.clear",
        selfref_history_clear,
        description="Clear all selfref history messages.",
        input_type="Optional history key.",
        output_type="None",
        parameters=[_optional_history_key_param()],
    )
    _register_selfref_primitive(
        "selfref.history.get_system_prompt",
        selfref_history_get_system_prompt,
        description="Get latest selfref system prompt text.",
        input_type="Optional history key.",
        output_type="str | None",
        parameters=[_optional_history_key_param()],
    )
    _register_selfref_primitive(
        "selfref.history.set_system_prompt",
        selfref_history_set_system_prompt,
        description="Overwrite selfref system prompt text.",
        input_type="Prompt text and optional history key.",
        output_type="None",
        parameters=[
            {
                "name": "text",
                "type": "str",
                "required": True,
                "description": "New system prompt content.",
                "kind": "positional_or_keyword",
            },
            _optional_history_key_param(),
        ],
    )
    _register_selfref_primitive(
        "selfref.history.append_system_prompt",
        selfref_history_append_system_prompt,
        description="Append text to selfref system prompt.",
        input_type="Prompt text and optional history key.",
        output_type="None",
        parameters=[
            {
                "name": "text",
                "type": "str",
                "required": True,
                "description": "Prompt suffix to append.",
                "kind": "positional_or_keyword",
            },
            _optional_history_key_param(),
        ],
    )
    _register_selfref_primitive(
        "selfref.fork.is_bound",
        selfref_fork_is_bound,
        description="Check whether selfref recursive fork is available.",
        input_type="No arguments.",
        output_type="bool",
        parameters=[],
    )
    _register_selfref_primitive(
        "selfref.fork.run",
        selfref_fork_run,
        description="Run one selfref child fork and wait for result.",
        input_type="Variadic agent args/kwargs forwarded to bound agent instance.",
        output_type="dict[str, Any]",
        parameters=[
            {
                "name": "*agent_args",
                "type": "Any",
                "required": False,
                "description": "Positional arguments forwarded to the child agent.",
                "kind": "var_positional",
            },
            {
                "name": "**agent_kwargs",
                "type": "Any",
                "required": False,
                "description": "Keyword arguments forwarded to the child agent, such as source_memory_key/fork_memory_key.",
                "kind": "var_keyword",
            },
        ],
    )
    _register_selfref_primitive(
        "selfref.fork.spawn",
        selfref_fork_spawn,
        description="Spawn one selfref child fork asynchronously.",
        input_type="Variadic agent args/kwargs forwarded to bound agent instance.",
        output_type="dict[str, Any]",
        parameters=[
            {
                "name": "*agent_args",
                "type": "Any",
                "required": False,
                "description": "Positional arguments forwarded to the child agent.",
                "kind": "var_positional",
            },
            {
                "name": "**agent_kwargs",
                "type": "Any",
                "required": False,
                "description": "Keyword arguments forwarded to the child agent, such as source_memory_key/fork_memory_key.",
                "kind": "var_keyword",
            },
        ],
    )
    _register_selfref_primitive(
        "selfref.fork.wait",
        selfref_fork_wait,
        description="Wait for one spawned selfref fork result.",
        input_type="Fork ID.",
        output_type="dict[str, Any]",
        parameters=[
            {
                "name": "fork_id",
                "type": "str",
                "required": True,
                "description": "Target fork identifier returned by spawn/run.",
                "kind": "positional_or_keyword",
            },
        ],
    )
    _register_selfref_primitive(
        "selfref.fork.wait_all",
        selfref_fork_wait_all,
        description="Wait for multiple selfref fork results.",
        input_type="Optional fork ID list.",
        output_type="dict[str, dict[str, Any]]",
        parameters=[
            {
                "name": "fork_ids",
                "type": "list[str] | None",
                "required": False,
                "description": "Specific fork IDs to wait for. Omit to wait all pending forks.",
                "kind": "positional_or_keyword",
                "default": "None",
            },
        ],
    )


__all__ = ["register_self_reference_primitives"]
