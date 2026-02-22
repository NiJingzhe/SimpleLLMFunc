"""Tests for TUI decorator wiring and signature handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from SimpleLLMFunc.utils.tui.decorator import (
    _resolve_chat_parameters,
    tui,
)


def test_resolve_chat_parameters_prefers_history_name() -> None:
    """Resolver should pick first non-history arg as input."""

    async def fake_agent(message: str, history=None, language: str = "zh"):
        if False:
            yield None

    input_param, history_param = _resolve_chat_parameters(fake_agent)
    assert input_param == "message"
    assert history_param == "history"


def test_resolve_chat_parameters_with_chat_history() -> None:
    """Resolver should support chat_history alias."""

    async def fake_agent(query: str, chat_history=None):
        if False:
            yield None

    input_param, history_param = _resolve_chat_parameters(fake_agent)
    assert input_param == "query"
    assert history_param == "chat_history"


def test_tui_decorator_builds_app_and_runs() -> None:
    """Decorator should create TUI app with resolved runtime config."""

    async def fake_agent(message: str, history=None, language: str = "zh"):
        if False:
            yield None

    decorated = tui()(fake_agent)

    with patch("SimpleLLMFunc.utils.tui.decorator.AgentTUIApp") as mock_app_cls:
        app_instance = MagicMock()
        mock_app_cls.return_value = app_instance

        decorated(language="en")

        assert mock_app_cls.call_count == 1
        kwargs = mock_app_cls.call_args.kwargs
        assert kwargs["agent_func"] is fake_agent
        assert kwargs["input_param"] == "message"
        assert kwargs["history_param"] == "history"
        assert kwargs["static_kwargs"] == {"language": "en"}
        app_instance.run.assert_called_once()


def test_tui_decorator_raises_when_no_input_parameter() -> None:
    """Decorator should fail fast when no user-input parameter exists."""

    async def fake_agent(history=None):
        if False:
            yield None

    with pytest.raises(ValueError, match="at least one non-history parameter"):
        tui()(fake_agent)
