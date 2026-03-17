# Examples Guide

This directory is for runnable user-facing examples.

## Quick Play (No API key required)

- `runtime_primitives_basic_example.py`
  - Run: `poetry run python examples/runtime_primitives_basic_example.py`
  - Shows runtime backend/primitive registration and `runtime.selfref.history.*` usage.

## Interactive TUI Demos (Requires `examples/provider.json`)

- `tui_general_agent_example.py` (recommended)
  - Run: `poetry run python examples/tui_general_agent_example.py`
  - General TUI agent demo with runtime selfref primitives + FileToolset.
  - Workspace: `./sandbox` (FileToolset is scoped here).
- `tui_chat_example.py`
  - Run: `poetry run python examples/tui_chat_example.py`
  - Basic TUI chat + persistent `PyRepl`.

## Other Feature Examples

- `pyrepl_example.py`: standalone PyRepl patterns and event streaming.
- `event_stream_chatbot.py`: full event-stream chatbot UI.
- `custom_tool_event_example.py`: custom tool event emission.
- `parallel_toolcall_example.py`: parallel tool-calling behavior.
- `multi_modality_toolcall.py`: multimodal tool call flow.
- `dynamic_template_demo.py`: `_template_params` dynamic template usage.
- `llm_function_pydantic_example.py`: structured Pydantic output.
- `llm_function_event_pydantic.py`: events + Pydantic output.
- `llm_function_token_usage.py`: token usage collection via events.

## Notes

- `provider_template.json` is a starter config template.
- Maintenance scripts for docs localization were moved to `scripts/` to keep `examples/` focused.
