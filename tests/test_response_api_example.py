"""Tests for the Responses API TUI example."""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncGenerator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_example_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parent.parent / "examples" / "response_api_example.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_response_api_example_module",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_module() -> ModuleType:
    return _load_example_module()


def test_response_api_example_uses_responses_interface(
    example_module: ModuleType,
) -> None:
    assert example_module.llm.__class__.__name__ == "OpenAIResponsesCompatible"
    assert example_module.llm.model_name == "gpt-5.4"


def test_response_api_example_decorator_sets_reasoning_kwargs(
    example_module: ModuleType,
) -> None:
    wrapped = example_module.core_agent
    assert getattr(wrapped, "__name__", "") == "core_agent"

    closure_cells = wrapped.__closure__ or ()
    llm_kwargs = None
    for cell in closure_cells:
        value = cell.cell_contents
        if isinstance(value, dict) and isinstance(value.get("reasoning"), dict):
            llm_kwargs = value
            break

    assert llm_kwargs is not None
    assert llm_kwargs["reasoning"] == {
        "effort": "xhigh",
        "summary": "detailed",
    }


def test_prepare_user_message_appends_compaction_instruction_when_threshold_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    example_module: ModuleType,
) -> None:
    monkeypatch.setattr(
        example_module,
        "llm",
        SimpleNamespace(
            input_token_count=30000,
            output_token_count=12000,
            context_window=200000,
        ),
    )

    prepared_message = example_module._prepare_user_message("Please finish this task.")

    assert prepared_message.startswith("Please finish this task.")
    assert example_module.CONTEXT_WINDOW_COMPACTION_INSTRUCTION in prepared_message


def test_tui_example_prompt_uses_explicit_runtime_primitive_language(
    example_module: ModuleType,
) -> None:
    docstring = example_module.core_agent.__doc__ or ""
    compaction_instruction = example_module.CONTEXT_WINDOW_COMPACTION_INSTRUCTION

    assert "runtime primitive" in docstring.lower()
    assert "call `runtime.selfref.fork.spawn(...)`" in docstring
    assert "call `runtime.selfref.fork.gather_all(...)`" in docstring
    assert "runtime primitive inside `execute_code`" in docstring
    assert "runtime primitive inside `execute_code`" in compaction_instruction


@pytest.mark.asyncio
async def test_agent_passes_runtime_toolkit_override_in_template_params(
    monkeypatch: pytest.MonkeyPatch,
    example_module: ModuleType,
) -> None:
    captured_kwargs: dict[str, object] = {}

    async def _fake_core_agent(**kwargs: object) -> AsyncGenerator[object, None]:
        captured_kwargs.update(kwargs)
        if False:
            yield None

    monkeypatch.setattr(example_module, "core_agent", _fake_core_agent)
    monkeypatch.setattr(
        example_module,
        "PROMPT_TEMPLATE_PARAMS",
        {"environment_block": "# Environment\n- Primary working directory: /tmp/ws"},
    )
    monkeypatch.setattr(
        example_module,
        "_build_runtime_toolkit",
        lambda: ["tool-a", "tool-b"],
    )

    outputs = []
    async for item in example_module.agent.__wrapped__("hello", history=[]):
        outputs.append(item)

    assert outputs == []
    template_params = captured_kwargs["_template_params"]
    assert isinstance(template_params, dict)
    assert template_params["environment_block"] == (
        "# Environment\n- Primary working directory: /tmp/ws"
    )
    assert template_params[
        example_module.SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM
    ] == ["tool-a", "tool-b"]
