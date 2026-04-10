"""Default tool-call card implementation."""

from __future__ import annotations

from SimpleLLMFunc.utils.tui.tool_cards.base import ToolCallCard


class DefaultToolCallCard(ToolCallCard):
    """Fallback card used for tools without specialized UI."""


__all__ = ["DefaultToolCallCard"]
