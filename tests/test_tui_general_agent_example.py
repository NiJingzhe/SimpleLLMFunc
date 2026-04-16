"""Tests for context-window compaction prompt injection in the TUI example."""

from __future__ import annotations

import json
import importlib.util
from collections.abc import AsyncGenerator
from pathlib import Path
import shutil
from types import ModuleType, SimpleNamespace

import pytest


def _write_example_provider_json(example_dir: Path) -> None:
    (example_dir / "provider.json").write_text(
        json.dumps(
            {
                "openrouter": [
                    {
                        "model_name": "gpt-5.4",
                        "api_keys": ["sk-test-key"],
                        "base_url": "https://openrouter.ai/api/v1",
                        "max_retries": 1,
                        "retry_delay": 0.0,
                        "rate_limit_capacity": 10,
                        "rate_limit_refill_rate": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _load_example_module(tmp_path: Path) -> ModuleType:
    source_path = (
        Path(__file__).resolve().parent.parent
        / "examples"
        / "tui_general_agent_example.py"
    )
    module_path = tmp_path / "tui_general_agent_example.py"
    shutil.copy2(source_path, module_path)
    _write_example_provider_json(tmp_path)

    spec = importlib.util.spec_from_file_location(
        "test_tui_general_agent_example_module",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_module(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    tmp_path = tmp_path_factory.mktemp("tui_general_agent_example")
    return _load_example_module(tmp_path)


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


def test_prepare_user_message_skips_compaction_instruction_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
    example_module: ModuleType,
) -> None:
    monkeypatch.setattr(
        example_module,
        "llm",
        SimpleNamespace(
            input_token_count=25000,
            output_token_count=14999,
            context_window=200000,
        ),
    )

    prepared_message = example_module._prepare_user_message("Keep going.")

    assert prepared_message == "Keep going."


def test_prepare_user_message_does_not_duplicate_compaction_instruction(
    monkeypatch: pytest.MonkeyPatch,
    example_module: ModuleType,
) -> None:
    monkeypatch.setattr(
        example_module,
        "llm",
        SimpleNamespace(
            input_token_count=50000,
            output_token_count=10000,
            context_window=200000,
        ),
    )
    original_message = (
        "Please wrap up this task.\n\n"
        + example_module.CONTEXT_WINDOW_COMPACTION_INSTRUCTION
    )

    prepared_message = example_module._prepare_user_message(original_message)

    assert prepared_message == original_message


def test_resolve_workspace_dir_returns_default_workspace(
    example_module: ModuleType,
) -> None:
    workspace_dir = example_module._resolve_workspace_dir()

    assert workspace_dir == example_module.DEFAULT_WORKSPACE_DIR
    assert workspace_dir.is_dir()


def test_build_environment_block_uses_provided_workspace(
    tmp_path: Path,
    example_module: ModuleType,
) -> None:
    environment_block = example_module._build_environment_block(tmp_path)

    assert f"- Primary working directory: {tmp_path}" in environment_block


def test_tui_example_prompt_uses_explicit_runtime_primitive_language(
    example_module: ModuleType,
) -> None:
    docstring = example_module.core_agent.__doc__ or ""
    compaction_instruction = example_module.CONTEXT_WINDOW_COMPACTION_INSTRUCTION

    assert "runtime primitive" in docstring.lower()
    assert "call `runtime.selfref.fork.spawn(...)`" in docstring
    assert "call `runtime.selfref.fork.gather_all(...)`" in docstring
    assert "`fork.spawn`" not in docstring
    assert "`fork.gather_all`" not in docstring
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
