"""Selfref runtime primitive pack backed by ``SelfReference``."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .state import SelfReference

from ..primitives import (
    PrimitiveCallContext,
    PrimitivePack,
    PrimitiveRegistry,
    primitive,
)


DEFAULT_SELF_REFERENCE_BACKEND_NAME = "selfref"
SELF_REFERENCE_PACK_GUIDANCE = (
    "selfref = your agent context plus your forked child agents. Use it for "
    "context compaction, durable experience cleanup, and parallel sub-agent "
    "decomposition."
)


def register_self_reference_primitives(
    registry: PrimitiveRegistry,
    *,
    backend_name: str = DEFAULT_SELF_REFERENCE_BACKEND_NAME,
    replace: bool = False,
) -> None:
    """Register selfref context/fork primitives backed by ``SelfReference``."""

    handlers = _build_self_reference_handlers(backend_name)

    for name, handler in handlers:
        registry.register(
            name,
            handler,
            backend_name=backend_name,
            replace=replace,
        )


def build_self_reference_pack(
    backend: SelfReference,
    *,
    backend_name: str = DEFAULT_SELF_REFERENCE_BACKEND_NAME,
    replace: bool = False,
) -> PrimitivePack:
    """Build a selfref PrimitivePack for installation into PyRepl."""

    pack = PrimitivePack(
        "selfref",
        backend=backend,
        backend_name=backend_name,
        guidance=SELF_REFERENCE_PACK_GUIDANCE,
    )
    handlers = _build_self_reference_handlers(backend_name)
    for name, handler in handlers:
        pack.add_primitive(
            name,
            handler,
            replace=replace,
        )
    return pack


def _append_best_practices_to_docstring(
    handler: Any,
    best_practices: Sequence[str],
) -> Any:
    if not best_practices:
        return handler

    docstring = handler.__doc__ or ""
    from ..primitives import _parse_primitive_docstring

    parsed_existing = _parse_primitive_docstring(docstring)
    if isinstance(parsed_existing.get("best_practices"), tuple):
        return handler

    block = "\n".join(f"- {item}" for item in best_practices)
    suffix = f"\n\nBest Practices:\n{block}\n"
    handler.__doc__ = f"{docstring.rstrip()}{suffix}" if docstring.strip() else suffix

    metadata = getattr(handler, "__simplellmfunc_primitive_spec__", None)
    if isinstance(metadata, dict):
        parsed = _parse_primitive_docstring(handler.__doc__)
        if isinstance(parsed.get("best_practices"), tuple):
            metadata["best_practices"] = parsed["best_practices"]

    return handler


def _build_self_reference_handlers(
    backend_name: str,
) -> list[tuple[str, Any]]:
    def require_self_reference(ctx: PrimitiveCallContext) -> SelfReference:
        backend = ctx.backend
        if backend is None:
            raise RuntimeError(
                "No self_reference backend registered. "
                "Install the 'selfref' primitive pack first."
            )
        if not isinstance(backend, SelfReference):
            raise RuntimeError(
                f"runtime backend '{backend_name}' must be SelfReference"
            )
        return backend

    def resolve_history_key(
        ctx: PrimitiveCallContext, key: Optional[str] = None
    ) -> str:
        return require_self_reference(ctx).resolve_history_key(key)

    @primitive()
    def selfref_context_inspect(
        ctx: PrimitiveCallContext,
        *,
        key: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Use: Inspect the current self-reference context as a read-only snapshot.
        Input: Keyword-only `key: str | None`. Omit it to use the active context key.
        Output: `dict[str, primitive]` with keys `active_key:str`, `experiences:list[dict[str, primitive]]`, `summary:dict[str, primitive] | None`, `messages:list[dict[str, primitive]]`, and `has_pending_compaction:bool`.
        Parse: Read `summary` and `experiences` first, then inspect `messages` when you need exact original message content or want to dump selected messages to disk.
        Parameters:
        - key: Optional context key.
        """
        self_reference = require_self_reference(ctx)
        resolved_key = self_reference.resolve_history_key(key)
        context_state = self_reference.parse_context_state(resolved_key)
        return {
            "active_key": resolved_key,
            "experiences": context_state.get("experiences", []),
            "summary": context_state.get("summary"),
            "messages": self_reference.snapshot_context_messages(resolved_key),
            "has_pending_compaction": self_reference.has_pending_compaction(
                resolved_key
            ),
        }

    @primitive()
    def selfref_context_remember(
        ctx: PrimitiveCallContext,
        text: str,
        *,
        key: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Use: Persist one durable experience into the system-context experience block.
        Input: `text: str` plus keyword-only `key: str | None`.
        Output: `dict[str, str]` with keys `id` and `text`.
        Parse: Save the returned `id` if you may want to forget this experience later.
        Parameters:
        - text: Short durable experience or rule that should survive future turns.
        - key: Optional context key.
        """
        self_reference = require_self_reference(ctx)
        resolved_key = self_reference.resolve_history_key(key)
        return self_reference.remember_experience(resolved_key, text)

    @primitive()
    def selfref_context_forget(
        ctx: PrimitiveCallContext,
        experience_id: str,
        *,
        key: Optional[str] = None,
    ) -> bool:
        """
        Use: Forget one durable experience by id.
        Input: `experience_id: str` plus keyword-only `key: str | None`.
        Output: `bool`. `True` when one experience was removed, otherwise `False`.
        Parse: If `False`, inspect the context first and use an exact experience id.
        Parameters:
        - experience_id: Experience id returned from `selfref.context.inspect()` or `selfref.context.remember()`.
        - key: Optional context key.
        """
        self_reference = require_self_reference(ctx)
        resolved_key = self_reference.resolve_history_key(key)
        return self_reference.forget_experience(resolved_key, experience_id)

    @primitive()
    def selfref_context_compact(
        ctx: PrimitiveCallContext,
        goal: str,
        instruction: str,
        discoveries: list[str],
        completed: list[str],
        current_status: str,
        likely_next_work: list[str],
        relevant_files_directories: list[str],
        *,
        remember: Optional[list[str]] = None,
        key: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Use: Queue a milestone compaction checkpoint. The framework will replace the current working transcript with the structured assistant summary at the end of this turn.
        Input: Structured summary fields plus optional keyword-only `remember: list[str] | None` and `key: str | None`.
        Output: `dict[str, primitive]` with keys `status:'queued'`, `active_key:str`, `summary:dict[str, primitive]`, `assistant_message:str`, and `remember:list[str]`.
        Parse: Print `assistant_message` if an operator should inspect the retained context. The actual context reset happens after the current turn finalizes.
        Parameters:
        - goal: Summary Goal section.
        - instruction: Summary Instruction section.
        - discoveries: Summary Discoveries bullets.
        - completed: Summary Completed bullets.
        - current_status: Summary Current Status section.
        - likely_next_work: Summary Likely next work bullets.
        - relevant_files_directories: Summary Relevant files/directories bullets.
        - remember: Optional durable experiences to append into the system experience block when compaction commits.
        - key: Optional context key.
        """
        self_reference = require_self_reference(ctx)
        resolved_key = self_reference.resolve_history_key(key)
        payload = self_reference.queue_context_compaction(
            resolved_key,
            {
                "goal": goal,
                "instruction": instruction,
                "discoveries": discoveries,
                "completed": completed,
                "current_status": current_status,
                "likely_next_work": likely_next_work,
                "relevant_files_directories": relevant_files_directories,
            },
            remember=remember,
        )
        return {
            "status": "queued",
            "active_key": resolved_key,
            "summary": payload["summary"],
            "assistant_message": payload["rendered_summary"],
            "remember": payload["remember"],
        }

    @primitive()
    def fork_is_bound(ctx: PrimitiveCallContext) -> bool:
        """
        Use: Diagnostic helper for checking whether a self-reference agent callable is available for fork delegation.
        Input: No arguments.
        Output: `bool`.
        Parse: `True` means `selfref.fork.*` can delegate. Use this for diagnostics and setup checks.
        """
        return require_self_reference(ctx).instance.is_bound()

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
        return await require_self_reference(ctx).instance.fork_spawn(
            message=message,
            _event_emitter=ctx.event_emitter,
            **call_kwargs,
        )

    async def fork_gather_all(
        ctx: PrimitiveCallContext,
        fork_ids: Optional[Any] = None,
        *,
        include_history: bool = False,
    ) -> dict[str, dict[str, Any]]:
        return await require_self_reference(ctx).instance.fork_gather_all(
            fork_ids,
            include_history=include_history,
        )

    @primitive()
    def selfref_fork_is_bound(ctx: PrimitiveCallContext) -> bool:
        """
        Use: Diagnostic helper for checking whether recursive self-reference fork delegation is available.
        Input: No arguments.
        Output: `bool`.
        Parse: `True` means a child agent can be spawned. Use this for setup diagnostics.
        """
        return fork_is_bound(ctx)

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
        ctx: PrimitiveCallContext,
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
        return await fork_gather_all(ctx, fork_ids, include_history=include_history)

    selfref_best_practices = [
        "Mental model: selfref = your agent context plus your forked child agents. Use selfref.context.* to inspect context, remember durable experience, forget stale experience, and queue milestone compaction.",
        "Use selfref.context.inspect() before context surgery. It returns a read-only full message snapshot plus parsed experiences and structured summary fields.",
        "Use selfref.context.remember(...) for durable lessons that belong in system context. Use selfref.context.forget(...) to remove wrong or obsolete experience entries by id.",
        "Use selfref.context.compact(...) only after you finish a milestone. Provide the required structured sections exactly and keep the retained summary concise but sufficient for the next turn.",
        "Compaction is queued during the current turn and committed at turn finalize. After commit, working transcript messages are cleared and only the assistant compaction summary remains outside the system prompt.",
        "Each layer focuses on planning for its own scope; delegate concrete execution to child forks.",
        "When tasks are independent (no content dependency), spawn forks in parallel (selfref.fork.spawn) and then gather results (selfref.fork.gather_all).",
        "Before forking, review and trim context; summarize irrelevant context or dump selected original messages to files using selfref.context.inspect() plus file tools.",
        "Call selfref.fork.spawn directly for fork creation. When fork status is error, inspect error_type/error_message and fix call arguments.",
        "Pass child args/kwargs that match the current bound agent callable signature. For llm_chat agents, pass the main user input as the first positional arg or the declared input keyword (for example message=... or prompt=...). Use supported kwargs from the agent signature.",
        "In fork prompts, define completion boundaries and require explicit acceptance criteria; prefer file-based handoff plus parent-agent messaging over dumping everything in chat.",
        "Fork results are compact by default (history omitted). Use include_history=True when full child history is required; use the returned memory_key + selfref.context.inspect(key=...) for on-demand detail reads.",
        "For fork results, read status/response/memory_key/history_count first; if status is error, inspect error_type/error_message before retrying.",
        "Read the fields you need from fork results (status/response/memory_key) and inspect child context explicitly when needed.",
        "After each milestone, compact aggressively: keep durable experience in system context, keep one structured assistant summary, and drop stale working transcript messages.",
    ]

    @primitive()
    def selfref_guide(_ctx: PrimitiveCallContext) -> dict[str, Any]:
        """
        Use: Read the overview of the self-reference namespace.
        Input: No arguments.
        Output: `dict[str, primitive]` with keys `namespace:str`, `overview:str`, and `best_practices:list[str]`.
        Parse: Read `overview` first, then scan `best_practices` for context compaction and fork rules.
        """
        return {
            "namespace": "selfref",
            "overview": SELF_REFERENCE_PACK_GUIDANCE,
            "best_practices": list(selfref_best_practices),
        }

    handlers = [
        ("selfref.guide", selfref_guide),
        ("selfref.context.inspect", selfref_context_inspect),
        ("selfref.context.remember", selfref_context_remember),
        ("selfref.context.forget", selfref_context_forget),
        ("selfref.context.compact", selfref_context_compact),
        ("selfref.fork.is_bound", selfref_fork_is_bound),
        ("selfref.fork.spawn", selfref_fork_spawn),
        ("selfref.fork.gather_all", selfref_fork_gather_all),
    ]

    for _, handler in handlers:
        _append_best_practices_to_docstring(handler, selfref_best_practices)

    return handlers


__all__ = [
    "DEFAULT_SELF_REFERENCE_BACKEND_NAME",
    "SELF_REFERENCE_PACK_GUIDANCE",
    "build_self_reference_pack",
    "register_self_reference_primitives",
]
