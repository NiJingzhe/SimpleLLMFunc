"""Tests for llm_function decorator defaults and orchestration."""

from __future__ import annotations

import inspect
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

from SimpleLLMFunc import llm_function
from SimpleLLMFunc.hooks.stream import ReactOutput, ResponseYield


class _DummyObservation:
    """Simple context manager used to stub Langfuse observations."""

    def __enter__(self) -> "_DummyObservation":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        return None


def test_llm_function_default_max_tool_calls_is_none() -> None:
    """llm_function should not impose a default tool-call limit."""

    signature = inspect.signature(llm_function)

    assert signature.parameters["max_tool_calls"].default is None


@pytest.mark.asyncio
async def test_llm_function_passes_none_max_tool_calls_to_execute_react_loop() -> None:
    """llm_function should forward the unbounded default into ReAct orchestration."""

    captured: dict[str, Any] = {}
    raw_response = object()

    async def fake_execute_react_loop(*args: Any, **kwargs: Any):
        _ = args
        captured["max_tool_calls"] = kwargs.get("max_tool_calls")

        async def _stream() -> AsyncGenerator[ReactOutput, None]:
            yield ResponseYield(type="response", response=raw_response, messages=[])

        return _stream()

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.execute_react_loop",
            new=fake_execute_react_loop,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.parse_and_validate_response",
            return_value="parsed-result",
        ) as mock_parse,
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_function(llm_interface=mock_llm)
        async def summarize(text: str) -> str:
            """Summarize input text."""

        result = await summarize("hello")

    assert captured["max_tool_calls"] is None
    assert result == "parsed-result"
    mock_parse.assert_called_once_with(
        response=raw_response,
        return_type=str,
        func_name="summarize",
    )
