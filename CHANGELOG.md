# Change log for SimpleLLMFunc

## 0.4.0 Release Notes

### Major Refactoring

1. **Modular Architecture Restructuring**: Completely refactored the base module, splitting messages, tool_call, and type_resolve into dedicated sub-modules for better code organization and maintainability.

2. **Decorator Logic Step-based Implementation**: Refactored decorator logic into a steps-based architecture within the `llm_decorator` module, improving code clarity and extensibility.

3. **Type System Enhancement**: Introduced new type support modules including decorator types and multimodal type support, expanding framework capabilities.

4. **Type Resolution System Refactoring**: Comprehensive refactoring of the type resolution system to enhance functionality support and improve type inference accuracy.

### Features

1. **Enhanced Tool Call Execution**: Improved tool call execution mechanism with extended support for multimodal interactions, enabling richer LLM interactions.

2. **Multimodal Type Support**: Added comprehensive multimodal type support throughout the framework for better handling of diverse content types.

### Bug Fixes

1. Fixed system prompt nesting issues when building multi-model content.

### Testing

Added extensive test coverage for refactored modules to ensure stability and reliability.

---

## 0.3.2.beta2 Release Notes

1. Remove dependence: `nest-asyncio`

2. Fix document error about `provider.json`

## 0.3.2.beta1 Release Notes

1. Better tool call tips in system prompt.

2. Better compound type annotations in prompt.

## 0.3.1 Release Notes

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
