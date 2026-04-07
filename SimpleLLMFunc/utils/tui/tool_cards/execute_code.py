"""Specialized ToolCallCard for the PyRepl execute_code tool."""

from __future__ import annotations

from typing import Any

from SimpleLLMFunc.utils.tui.tool_cards.base import ToolCallCard


def _render_fenced_block(content: str, language: str = "") -> str:
    normalized = content.rstrip("\n")
    fence = "```"
    if "```" in normalized:
        fence = "````"
    header = f"{fence}{language}" if language else fence
    return f"{header}\n{normalized}\n{fence}"


class ExecuteCodeToolCallCard(ToolCallCard):
    """Dedicated card for execute_code with code/output/result sections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_class("execute-code-tool-call")

    def refresh_arguments_markdown(self) -> None:
        if not self.arguments:
            self.arguments_markdown = "_No arguments_"
            self.last_argument_changed = None
            return

        lines: list[str] = []
        code = self.arguments.get("code")
        if isinstance(code, str):
            lines.append("## Code")
            lines.append(_render_fenced_block(code, language="python"))

        control_keys = [key for key in ("timeout_seconds",) if key in self.arguments]
        if control_keys:
            lines.append("## Runtime Controls")
            for key in control_keys:
                lines.append(
                    self.render_single_argument_markdown(key, self.arguments[key])
                )

        remaining_keys = [
            key for key in self.arguments if key not in {"code", *control_keys}
        ]
        if remaining_keys:
            lines.append("## Arguments")
            for key in remaining_keys:
                lines.append(
                    self.render_single_argument_markdown(key, self.arguments[key])
                )

        self.arguments_markdown = "\n\n".join(line for line in lines if line)
        if not self.arguments_markdown.strip():
            self.arguments_markdown = "_No arguments_"
        self.last_argument_changed = None

    def build_output_markdown(self) -> str:
        if not self.output:
            return ""
        return "## Live Output\n\n" + _render_fenced_block(self.output, language="text")

    def build_result_markdown(self) -> str:
        if not self.result_markdown:
            return ""
        return f"## Execution Summary\n\n{self.result_markdown}"


__all__ = ["ExecuteCodeToolCallCard"]
