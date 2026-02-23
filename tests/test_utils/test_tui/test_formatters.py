"""Tests for TUI argument/result formatter helpers."""

from __future__ import annotations

from SimpleLLMFunc.utils.tui.formatters import format_tool_arguments_markdown


def test_execute_code_arguments_render_as_python_block() -> None:
    """PyRepl execute_code should render code argument with fenced block."""

    markdown = format_tool_arguments_markdown(
        {"code": "print('hello')", "timeout": 30},
        tool_name="execute_code",
    )

    assert "- `code`:" in markdown
    assert "```python" in markdown
    assert "print('hello')" in markdown
    assert "- `timeout`: `30`" in markdown
    assert "`print('hello')`" not in markdown


def test_non_execute_code_arguments_keep_inline_style() -> None:
    """Regular tool arguments should keep existing inline rendering."""

    markdown = format_tool_arguments_markdown({"query": "hello"})

    assert markdown == "- `query`: `hello`"


def test_execute_code_arguments_with_backticks_use_safe_fence() -> None:
    """Code containing triple backticks should use a longer fence."""

    markdown = format_tool_arguments_markdown(
        {"code": "print('```')"},
        tool_name="execute_code",
    )

    assert "````python" in markdown
    assert markdown.strip().endswith("````")
