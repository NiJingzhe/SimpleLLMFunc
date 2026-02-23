"""Inbound input stream primitives for interactive agent sessions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Literal, Optional


@dataclass(frozen=True)
class UserInputEvent:
    """One inbound user-input event.

    Attributes:
        text: Raw text entered by user.
        request_id: Optional explicit target request for tool input.
        force_chat: If True, bypass pending tool-input requests and route to chat.
    """

    text: str
    request_id: Optional[str] = None
    force_chat: bool = False


@dataclass(frozen=True)
class ToolInputRequest:
    """A pending tool-input request emitted from tool events."""

    request_id: str
    tool_call_id: str
    prompt: str = ""


@dataclass(frozen=True)
class InputRouteResult:
    """Routing result for one inbound input event."""

    route: Literal["chat", "tool", "noop", "rejected"]
    chat_text: str = ""
    submitted_text: str = ""
    request: Optional[ToolInputRequest] = None
    reason: str = ""


class AgentInputRouter:
    """Routes inbound input events between chat and tool-input consumers.

    Pending tool-input requests are consumed with higher priority than new chat turns,
    unless ``UserInputEvent.force_chat`` is set.
    """

    def __init__(self, submit_tool_input: Callable[[str, str], bool]):
        self._submit_tool_input = submit_tool_input
        self._pending_by_request: "OrderedDict[str, ToolInputRequest]" = OrderedDict()
        self._request_ids_by_tool: dict[str, set[str]] = {}

    def register_tool_request(
        self,
        tool_call_id: str,
        request_id: str,
        prompt: str = "",
    ) -> None:
        request = ToolInputRequest(
            request_id=request_id,
            tool_call_id=tool_call_id,
            prompt=prompt,
        )
        self._pending_by_request[request_id] = request
        self._request_ids_by_tool.setdefault(tool_call_id, set()).add(request_id)

    def clear_tool_requests(self, tool_call_id: str) -> None:
        request_ids = self._request_ids_by_tool.pop(tool_call_id, set())
        for request_id in request_ids:
            self._pending_by_request.pop(request_id, None)

    def clear_all_requests(self) -> None:
        self._pending_by_request.clear()
        self._request_ids_by_tool.clear()

    def has_pending_tool_requests(self) -> bool:
        return bool(self._pending_by_request)

    def peek_pending_tool_request(self) -> Optional[ToolInputRequest]:
        if not self._pending_by_request:
            return None
        first_request_id = next(iter(self._pending_by_request.keys()))
        return self._pending_by_request.get(first_request_id)

    def route_input(self, event: UserInputEvent) -> InputRouteResult:
        """Route one inbound input event."""

        if self.has_pending_tool_requests() and not event.force_chat:
            return self._route_tool_input(event)

        chat_text = event.text.strip()
        if not chat_text:
            return InputRouteResult(route="noop")

        return InputRouteResult(route="chat", chat_text=chat_text)

    def _route_tool_input(self, event: UserInputEvent) -> InputRouteResult:
        candidate_request_ids: list[str]
        if event.request_id:
            candidate_request_ids = [event.request_id]
        else:
            candidate_request_ids = list(self._pending_by_request.keys())

        for request_id in candidate_request_ids:
            request = self._pending_by_request.get(request_id)
            if request is None:
                continue

            delivered = False
            try:
                delivered = self._submit_tool_input(request.request_id, event.text)
            except Exception:
                delivered = False

            if delivered:
                self._drop_request(request.request_id)
                return InputRouteResult(
                    route="tool",
                    request=request,
                    submitted_text=event.text,
                )

            # Request has likely expired; remove stale entry and try next one.
            self._drop_request(request.request_id)

        return InputRouteResult(
            route="rejected",
            reason="Tool input request expired before it could be submitted.",
        )

    def _drop_request(self, request_id: str) -> None:
        request = self._pending_by_request.pop(request_id, None)
        if request is None:
            return

        tool_ids = self._request_ids_by_tool.get(request.tool_call_id)
        if not tool_ids:
            return

        tool_ids.discard(request_id)
        if not tool_ids:
            self._request_ids_by_tool.pop(request.tool_call_id, None)


__all__ = [
    "UserInputEvent",
    "ToolInputRequest",
    "InputRouteResult",
    "AgentInputRouter",
]
