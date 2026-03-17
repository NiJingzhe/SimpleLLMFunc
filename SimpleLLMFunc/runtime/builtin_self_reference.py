"""Builtin runtime primitive pack backed by ``SelfReference``."""

from __future__ import annotations

from typing import Any, Callable, Optional

from SimpleLLMFunc.builtin.self_reference import SelfReference

from .primitives import PrimitiveCallContext, PrimitiveRegistry, primitive


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

    @primitive()
    def selfref_history_keys(_ctx: PrimitiveCallContext) -> list[str]:
        """
        Use: List all bound self-reference history keys.
        Input: No arguments.
        Output: `list[str]`. Each item is one history key.
        Parse: Iterate the string list directly. Use one returned key in other `selfref.history.*` calls.
        """
        return require_self_reference().memory.keys()

    @primitive()
    def selfref_history_active_key(_ctx: PrimitiveCallContext) -> str:
        """
        Use: Resolve the active history key for the current self-reference context.
        Input: No arguments.
        Output: `str`. The resolved history key.
        Parse: Treat the result as the default key that other history and fork primitives will use when `key` is omitted.
        """
        return resolve_history_key(None)

    @primitive()
    def selfref_history_count(
        _ctx: PrimitiveCallContext,
        *,
        key: Optional[str] = None,
    ) -> int:
        """
        Use: Count non-system messages in one self-reference history.
        Input: Keyword-only `key: str | None`. Omit it to use the active history key.
        Output: `int`. The non-system message count.
        Parse: Read the integer directly.
        Parameters:
        - key: Optional history key.
        """
        return get_history_handle(key).count()

    @primitive()
    def selfref_history_all(
        _ctx: PrimitiveCallContext,
        *,
        key: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Use: Read the full message history for one self-reference memory.
        Input: Keyword-only `key: str | None`. Omit it to use the active history key.
        Output: `list[dict[str, primitive]]`. Each item is one message dict with primitive fields such as `role:str`, `content:str | None`, and optional tool-related keys.
        Parse: Iterate the list in order. Each dict is one message.
        Parameters:
        - key: Optional history key.
        """
        return get_history_handle(key).all()

    @primitive()
    def selfref_history_get(
        _ctx: PrimitiveCallContext,
        index: int,
        *,
        key: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Use: Read one message from a self-reference history by zero-based index.
        Input: `index: int` plus keyword-only `key: str | None`.
        Output: `dict[str, primitive]`. One message dict.
        Parse: Read fields from the returned message dict, for example `message['role']` and `message['content']`.
        Parameters:
        - index: Zero-based message index.
        - key: Optional history key.
        """
        return get_history_handle(key).get(index)

    @primitive()
    def selfref_history_append(
        _ctx: PrimitiveCallContext,
        message: dict[str, Any],
        *,
        key: Optional[str] = None,
    ) -> None:
        """
        Use: Append one message to a self-reference history.
        Input: `message: dict[str, primitive]` plus keyword-only `key: str | None`. Message dicts should use primitive fields such as `role:str`, `content:str | None`, and optional tool metadata when needed.
        Output: `None`.
        Parse: No return value. Read history again if you need to verify the append.
        Parameters:
        - message: Message dict to append.
        - key: Optional history key.
        """
        get_history_handle(key).append(message)

    @primitive()
    def selfref_history_insert(
        _ctx: PrimitiveCallContext,
        index: int,
        message: dict[str, Any],
        *,
        key: Optional[str] = None,
    ) -> None:
        """
        Use: Insert one message into a self-reference history at a zero-based index.
        Input: `index: int`, `message: dict[str, primitive]`, and keyword-only `key: str | None`.
        Output: `None`.
        Parse: No return value. Read history again if you need the updated order.
        Parameters:
        - index: Insert position.
        - message: Message dict to insert.
        - key: Optional history key.
        """
        get_history_handle(key).insert(index, message)

    @primitive()
    def selfref_history_update(
        _ctx: PrimitiveCallContext,
        index: int,
        message: dict[str, Any],
        *,
        key: Optional[str] = None,
    ) -> None:
        """
        Use: Replace one message in a self-reference history.
        Input: `index: int`, `message: dict[str, primitive]`, and keyword-only `key: str | None`.
        Output: `None`.
        Parse: No return value. Read history again if you need the updated message.
        Parameters:
        - index: Target message index.
        - message: Replacement message dict.
        - key: Optional history key.
        """
        get_history_handle(key).update(index, message)

    @primitive()
    def selfref_history_delete(
        _ctx: PrimitiveCallContext,
        index: int,
        *,
        key: Optional[str] = None,
    ) -> None:
        """
        Use: Delete one message from a self-reference history.
        Input: `index: int` plus keyword-only `key: str | None`.
        Output: `None`.
        Parse: No return value. Read history again if you need the remaining messages.
        Parameters:
        - index: Zero-based message index to delete.
        - key: Optional history key.
        """
        get_history_handle(key).delete(index)

    @primitive()
    def selfref_history_replace(
        _ctx: PrimitiveCallContext,
        messages: list[dict[str, Any]],
        *,
        key: Optional[str] = None,
    ) -> None:
        """
        Use: Replace an entire self-reference history with a new message list.
        Input: `messages: list[dict[str, primitive]]` plus keyword-only `key: str | None`.
        Output: `None`.
        Parse: No return value. Read history again if you need the replacement snapshot.
        Parameters:
        - messages: Full replacement message list.
        - key: Optional history key.
        """
        get_history_handle(key).replace(messages)

    @primitive()
    def selfref_history_clear(
        _ctx: PrimitiveCallContext,
        *,
        key: Optional[str] = None,
    ) -> None:
        """
        Use: Clear non-system messages from a self-reference history while preserving the current system prompt.
        Input: Keyword-only `key: str | None`.
        Output: `None`.
        Parse: No return value. Read history again if you need to confirm what remains.
        Parameters:
        - key: Optional history key.
        """
        get_history_handle(key).clear()

    @primitive()
    def selfref_history_get_system_prompt(
        _ctx: PrimitiveCallContext,
        *,
        key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Use: Read the latest system prompt text from one self-reference history.
        Input: Keyword-only `key: str | None`.
        Output: `str | None`. Returns prompt text when present, otherwise `None`.
        Parse: Check for `None` before using the string.
        Parameters:
        - key: Optional history key.
        """
        return get_history_handle(key).get_system_prompt()

    @primitive()
    def selfref_history_set_system_prompt(
        _ctx: PrimitiveCallContext,
        text: str,
        *,
        key: Optional[str] = None,
    ) -> None:
        """
        Use: Replace the current system prompt text for one self-reference history.
        Input: `text: str` plus keyword-only `key: str | None`.
        Output: `None`.
        Parse: No return value. Call `selfref.history.get_system_prompt` to verify the new text.
        Parameters:
        - text: Full replacement system prompt text.
        - key: Optional history key.
        """
        get_history_handle(key).set_system_prompt(text)

    @primitive()
    def selfref_history_append_system_prompt(
        _ctx: PrimitiveCallContext,
        text: str,
        *,
        key: Optional[str] = None,
    ) -> None:
        """
        Use: Append text to the current system prompt for one self-reference history.
        Input: `text: str` plus keyword-only `key: str | None`.
        Output: `None`.
        Parse: No return value. Call `selfref.history.get_system_prompt` to read the combined prompt.
        Parameters:
        - text: Prompt suffix to append.
        - key: Optional history key.
        """
        get_history_handle(key).append_system_prompt(text)

    @primitive()
    def fork_is_bound(_ctx: PrimitiveCallContext) -> bool:
        """
        Use: Diagnostic helper for checking whether a self-reference agent callable is available for fork delegation.
        Input: No arguments.
        Output: `bool`.
        Parse: `True` means `selfref.fork.*` can delegate. Use this only for diagnostics, not as a routine preflight step.
        """
        return require_self_reference().instance.is_bound()

    async def fork_spawn(
        ctx: PrimitiveCallContext,
        message: str,
        **agent_kwargs: Any,
    ) -> dict[str, Any]:
        if "include_history" in agent_kwargs:
            raise TypeError(
                "selfref.fork.spawn does not accept include_history; "
                "call selfref.fork.gather_all(..., include_history=True) instead"
            )
        call_kwargs = dict(agent_kwargs)
        call_kwargs.pop("_event_emitter", None)
        return await require_self_reference().instance.fork_spawn(
            message=message,
            _event_emitter=ctx.event_emitter,
            **call_kwargs,
        )

    async def fork_gather_all(
        _ctx: PrimitiveCallContext,
        fork_ids: Optional[Any] = None,
        *,
        include_history: bool = False,
    ) -> dict[str, dict[str, Any]]:
        return await require_self_reference().instance.fork_gather_all(
            fork_ids,
            include_history=include_history,
        )

    @primitive()
    def selfref_fork_is_bound(_ctx: PrimitiveCallContext) -> bool:
        """
        Use: Diagnostic helper for checking whether recursive self-reference fork delegation is available.
        Input: No arguments.
        Output: `bool`.
        Parse: `True` means a child agent can be spawned. Use this only when debugging setup issues.
        """
        return fork_is_bound(_ctx)

    @primitive()
    @primitive()
    async def selfref_fork_spawn(
        ctx: PrimitiveCallContext,
        message: str,
        **agent_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Use: Spawn one child chat-style self-reference agent asynchronously and return a handle.
        Input: `message: str` plus `**agent_kwargs: primitive`.
        Output: `dict[str, primitive]` with keys `fork_id:str`, `parent_fork_id:str | None`, `depth:int`, `source_memory_key:str`, `memory_key:str`, and `status:'running'`.
        Parse: Save `fork_id`, then call `selfref.fork.gather_all`.
        Parameters:
        - message: User message string passed to the child agent as keyword argument `message=...`.
        - **agent_kwargs: Optional fork control kwargs such as `source_memory_key` / `fork_memory_key`.
        """
        return await fork_spawn(ctx, message, **agent_kwargs)

    @primitive()
    async def selfref_fork_gather_all(
        _ctx: PrimitiveCallContext,
        fork_ids: Optional[Any] = None,
        *,
        include_history: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Use: Gather results for spawned self-reference children.
        Input: Optional `fork_ids: str | dict[str, primitive] | list[str] | list[dict[str, primitive]] | None` plus keyword-only `include_history: bool = False`. Omit `fork_ids` to gather all pending children. If passing dicts, each must contain `fork_id`.
        Output: `dict[str, dict[str, primitive]]` keyed by `fork_id`. Each value includes `fork_id:str`, `parent_fork_id:str | None`, `depth:int`, `source_memory_key:str`, `memory_key:str`, `status:'completed' | 'error'`, `response:primitive`, `error_type:str` when status is `error`, `error_message:str` when status is `error`, `history_count:int`, `history_included:bool`, and optional `history:list[dict[str, primitive]]` when `include_history=True`.
        Parse: Iterate with `.items()`. For each value, check `status` first, then read `response` or `error_type` / `error_message`.
        Parameters:
        - fork_ids: Optional fork id or handle subset. You may pass fork handle dicts with `fork_id` instead of strings.
        - include_history: When `True`, include full child history in each result dict.
        """
        return await fork_gather_all(
            _ctx,
            fork_ids,
            include_history=include_history,
        )

    selfref_best_practices = [
        "Mental model: selfref = your agent state. (1) Your memory: message history via selfref.history.*. (2) Your clones: forked child agents via selfref.fork.*. Not philosophical self-reference, not Python self.",
        "Use selfref.history.* to read/write your conversation memory (messages). Prefer appending durable preferences into the system prompt via selfref.history.append_system_prompt(...).",
        "When `key` is omitted, selfref resolves an active memory key for the current execution context. Use selfref.history.active_key() to see which key is currently in scope.",
        "Safety: do not assume internal data structures; only interact via runtime primitives (runtime.selfref.history.* and runtime.selfref.fork.*). Do not invent fields in message dicts.",
        "Each layer focuses on planning for its own scope; delegate concrete execution to child forks.",
        "When tasks are independent (no content dependency), spawn forks in parallel (selfref.fork.spawn) and then gather results (selfref.fork.gather_all).",
        "Before forking, review and trim memory; summarize irrelevant context or dump it to files.",
        "runtime.selfref.history.clear clears non-system history only; the current system prompt is preserved.",
        "Call selfref.fork.spawn directly; do not use selfref.fork.is_bound as routine preflight. If fork status is error, read error_type/error_message and fix call arguments.",
        "Child args/kwargs must satisfy the current bound agent callable signature. For llm_chat agents, pass the main user input as the first positional arg or the declared input keyword (for example message=... or prompt=...). Do not invent unsupported kwargs such as goal=/scope= unless the agent actually accepts them.",
        "In fork prompts, define completion boundaries and require explicit acceptance criteria; prefer file-based handoff plus parent-agent messaging over dumping everything in chat.",
        "Fork results are compact by default (history omitted). Use include_history=True only when full child history is explicitly required; otherwise use memory_key + selfref.history.* to fetch details on demand.",
        "Fork result contract reminder: consume only required keys (fork_id/status/response/history_count/memory_key). Full child history is omitted unless include_history=True.",
        "If a fork result status is error, inspect error_type/error_message immediately; do not assume response/history_count explain the failure.",
        "NEVER print raw fork result dict (forbidden: print(result)). Read only the fields you need (status/response/memory_key) and fetch history explicitly when needed.",
        "After each milestone, review and reorganize memory before moving forward.",
    ]

    def _register_selfref_primitive(
        name: str,
        handler: Any,
    ) -> None:
        registry.register(
            name,
            handler,
            best_practices=selfref_best_practices,
            replace=replace,
        )

    @primitive()
    def selfref_guide(_ctx: PrimitiveCallContext) -> dict[str, Any]:
        """
        Use: Read the overview of the self-reference namespace.
        Input: No arguments.
        Output: `dict[str, primitive]` with keys `namespace:str`, `overview:str`, and `best_practices:list[str]`.
        Parse: Read `overview` first, then scan `best_practices` for fork and history rules.
        """
        return {
            "namespace": "selfref",
            "overview": (
                "selfref = your agent state. (1) Your memory: message history via selfref.history.*. "
                "(2) Your clones: child agents via selfref.fork.spawn/gather_all. "
                "You operate on your own memory and create your own clones. "
                "Not philosophical self-reference, not Python self."
            ),
            "best_practices": list(selfref_best_practices),
        }

    # Self-reference first-class namespace.
    _register_selfref_primitive(
        "selfref.guide",
        selfref_guide,
    )
    _register_selfref_primitive(
        "selfref.history.keys",
        selfref_history_keys,
    )
    _register_selfref_primitive(
        "selfref.history.active_key",
        selfref_history_active_key,
    )
    _register_selfref_primitive(
        "selfref.history.count",
        selfref_history_count,
    )
    _register_selfref_primitive(
        "selfref.history.all",
        selfref_history_all,
    )
    _register_selfref_primitive(
        "selfref.history.get",
        selfref_history_get,
    )
    _register_selfref_primitive(
        "selfref.history.append",
        selfref_history_append,
    )
    _register_selfref_primitive(
        "selfref.history.insert",
        selfref_history_insert,
    )
    _register_selfref_primitive(
        "selfref.history.update",
        selfref_history_update,
    )
    _register_selfref_primitive(
        "selfref.history.delete",
        selfref_history_delete,
    )
    _register_selfref_primitive(
        "selfref.history.replace",
        selfref_history_replace,
    )
    _register_selfref_primitive(
        "selfref.history.clear",
        selfref_history_clear,
    )
    _register_selfref_primitive(
        "selfref.history.get_system_prompt",
        selfref_history_get_system_prompt,
    )
    _register_selfref_primitive(
        "selfref.history.set_system_prompt",
        selfref_history_set_system_prompt,
    )
    _register_selfref_primitive(
        "selfref.history.append_system_prompt",
        selfref_history_append_system_prompt,
    )
    _register_selfref_primitive(
        "selfref.fork.is_bound",
        selfref_fork_is_bound,
    )
    _register_selfref_primitive(
        "selfref.fork.spawn",
        selfref_fork_spawn,
    )
    _register_selfref_primitive(
        "selfref.fork.gather_all",
        selfref_fork_gather_all,
    )


__all__ = ["register_self_reference_primitives"]
