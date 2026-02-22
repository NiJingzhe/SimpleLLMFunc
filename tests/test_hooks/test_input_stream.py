"""Tests for inbound input-stream routing utilities."""

from __future__ import annotations

from SimpleLLMFunc.hooks.input_stream import AgentInputRouter, UserInputEvent


def test_route_chat_when_no_pending_tool_request() -> None:
    """Input should route to chat when no tool request is pending."""

    router = AgentInputRouter(submit_tool_input=lambda _req, _text: True)

    result = router.route_input(UserInputEvent(text="hello"))

    assert result.route == "chat"
    assert result.chat_text == "hello"


def test_route_tool_input_when_pending_request_exists() -> None:
    """Pending tool input request should consume user input first."""

    delivered: list[tuple[str, str]] = []

    def _submit_tool_input(request_id: str, value: str) -> bool:
        delivered.append((request_id, value))
        return True

    router = AgentInputRouter(submit_tool_input=_submit_tool_input)
    router.register_tool_request("call-1", "req-1", "Name: ")

    result = router.route_input(UserInputEvent(text="Alice"))

    assert result.route == "tool"
    assert result.request is not None
    assert result.request.request_id == "req-1"
    assert delivered == [("req-1", "Alice")]
    assert not router.has_pending_tool_requests()


def test_force_chat_bypasses_pending_tool_request() -> None:
    """force_chat flag should bypass pending tool input requests."""

    submit_called = False

    def _submit_tool_input(_request_id: str, _value: str) -> bool:
        nonlocal submit_called
        submit_called = True
        return True

    router = AgentInputRouter(submit_tool_input=_submit_tool_input)
    router.register_tool_request("call-1", "req-1", "Name: ")

    result = router.route_input(UserInputEvent(text="next turn", force_chat=True))

    assert result.route == "chat"
    assert result.chat_text == "next turn"
    assert submit_called is False
    assert router.has_pending_tool_requests()


def test_expired_tool_request_returns_rejected() -> None:
    """If submission fails, stale tool request should be rejected and dropped."""

    router = AgentInputRouter(submit_tool_input=lambda _req, _text: False)
    router.register_tool_request("call-1", "req-1", "Name: ")

    result = router.route_input(UserInputEvent(text="Alice"))

    assert result.route == "rejected"
    assert "expired" in result.reason.lower()
    assert not router.has_pending_tool_requests()


def test_clear_tool_requests_only_removes_target_tool() -> None:
    """Clearing one tool should not remove other pending tool requests."""

    router = AgentInputRouter(submit_tool_input=lambda _req, _text: True)
    router.register_tool_request("call-1", "req-1", "A: ")
    router.register_tool_request("call-2", "req-2", "B: ")

    router.clear_tool_requests("call-1")

    pending = router.peek_pending_tool_request()
    assert pending is not None
    assert pending.request_id == "req-2"
