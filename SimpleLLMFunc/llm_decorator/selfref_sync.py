from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from SimpleLLMFunc.runtime.selfref.state import SelfReference


class SelfReferenceReActSyncHooks:
    def __init__(
        self,
        self_reference: SelfReference,
        memory_key: str,
    ) -> None:
        self._state_token = None
        self.self_reference = self_reference
        self.memory_key = memory_key

    async def on_run_start(self, state: Any) -> None:
        self._state_token = self.self_reference._set_active_react_state(state)

    async def before_finalize(self, state: Any) -> None:
        committed_messages = self.self_reference.commit_pending_compaction(
            self.memory_key,
            cast(List[Dict[str, Any]], state.messages),
        )
        if committed_messages is not None:
            state.messages = committed_messages
            return

        self.self_reference.set_context_messages(
            key=self.memory_key,
            messages=cast(List[Dict[str, Any]], state.messages),
        )

    async def before_tool_batch(self, state: Any) -> None:
        self.self_reference.bind_history(
            self.memory_key,
            cast(List[Dict[str, Any]], state.messages),
        )

    async def after_tool_batch(self, state: Any) -> None:
        committed_messages = self.self_reference.commit_pending_compaction(
            self.memory_key,
            cast(List[Dict[str, Any]], state.messages),
        )
        if committed_messages is not None:
            state.messages = committed_messages
            return

        self.self_reference.set_context_messages(
            key=self.memory_key,
            messages=cast(List[Dict[str, Any]], state.messages),
        )

    def close(self) -> None:
        if self._state_token is None:
            return

        self.self_reference._reset_active_react_state(self._state_token)
        self._state_token = None


def build_selfref_react_sync_hooks(
    self_reference: Optional[SelfReference],
    memory_key: Optional[str],
) -> Optional[SelfReferenceReActSyncHooks]:
    if self_reference is None or memory_key is None:
        return None

    return SelfReferenceReActSyncHooks(
        self_reference=self_reference,
        memory_key=memory_key,
    )


__all__ = [
    "SelfReferenceReActSyncHooks",
    "build_selfref_react_sync_hooks",
]
