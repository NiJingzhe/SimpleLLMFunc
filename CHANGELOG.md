# Change log for SimpleLLMFunc

## 0.3.0 Release Notes (Latest)

1. Added dynamic template parameter support: The `llm_function` decorator now supports passing `_template_params` to dynamically set DocString template parameters. This allows developers to create a single function that can adapt to various use cases, changing its behavior by passing different template parameters at call time.

2. Integrated Langfuse support: You can now configure `LANGFUSE_BASE_URL`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_PUBLIC_KEY` to send logs to Langfuse for tracing and analysis.

3. Added multilingual support: The English README has been updated, now supporting both Chinese and English.

4. Added parallel tool calling support.

5. Fully native async implementation: All decorators are now implemented with native async support, completely dropping any sync fallback.

## 0.2.13 Release Notes

1. Added the `return_mode` parameter (`Literal["text", "raw"]`) to the `llm_chat` decorator, allowing you to specify the return mode. You can now return either the raw response or text. This is designed to better display tool call information when developing Agents.

2. Improved code type annotations.

-----

## 0.2.12.2 Release Notes

1. Added a `py.typed` file to the framework package to support type checking.
