from __future__ import annotations

import importlib

import pytest

from SimpleLLMFunc.tool import TOO_LONG_TO_FILE_MAX_TOKENS, Tool


tool_module = importlib.import_module("SimpleLLMFunc.tool.tool")


@pytest.mark.asyncio
async def test_too_long_to_file_uses_20000_token_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    full_result = "full tool output"
    seen_max_tokens: list[int] = []

    def fake_estimate_token_count(text: str) -> int:
        if text == full_result:
            return TOO_LONG_TO_FILE_MAX_TOKENS + 1
        return 1

    def fake_truncate_text_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
        seen_max_tokens.append(max_tokens)
        assert text == full_result
        return "TRUNCATED", max_tokens

    monkeypatch.setattr(tool_module, "_estimate_token_count", fake_estimate_token_count)
    monkeypatch.setattr(
        tool_module,
        "_truncate_text_to_tokens",
        fake_truncate_text_to_tokens,
    )
    monkeypatch.setattr(tool_module.tempfile, "gettempdir", lambda: str(tmp_path))

    async def tool_func() -> str:
        return full_result

    tool = Tool(
        name="test_tool",
        description="Test tool",
        func=tool_func,
        too_long_to_file=True,
    )

    result = await tool.run()

    assert seen_max_tokens == [TOO_LONG_TO_FILE_MAX_TOKENS]
    assert result.startswith("TRUNCATED")
    assert "tool return was too long" in result

    temp_files = list(tmp_path.iterdir())
    assert len(temp_files) == 1
    assert temp_files[0].read_text(encoding="utf-8") == full_result
