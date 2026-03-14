# Change log for SimpleLLMFunc

## 0.7.1 (2026-03-14) - Fork API Simplification

### ⚠️ Breaking Changes

1. **SelfReference Forking**:
   - Removed `selfref.fork.run`, `selfref.fork.wait`, and `selfref.fork.wait_all` (plus chat variants).
   - Use `selfref.fork.spawn` + `selfref.fork.gather_all` for async fork collection; `gather_all` accepts a single fork id/handle or a list.

### 🔧 Improvements

1. **Docs & Tests**:
   - Updated runtime docs, examples, and tests to match the simplified fork API.


## 0.7.0 (2026-03-11) - Runtime Primitives, Forking, and Docs Refresh

### 🎉 Major Features

1. **Runtime Primitive Registry**:
   - Added host-side primitive registry with `runtime.list_primitives`, `runtime.get_primitive_spec`, and `runtime.list_primitive_specs` for in-REPL discovery.
   - Added worker proxy plumbing so primitives are callable in PyRepl without imports.
   - Bundled SelfReference primitive pack under `runtime.selfref.history.*` and `runtime.selfref.fork.*` namespaces.

2. **SelfReference Forking**:
   - Added fork lifecycle primitives: `run`, `spawn`, `wait`, `wait_all` (plus chat variants) for parallel agent work.
   - Fork results expose status, response, memory key, history count, and structured error details.

3. **Event Origin Metadata & TUI Routing**:
   - Normalized event origin metadata (session/fork context + tool linkage) for deterministic routing.
   - Added fork-aware routing and visualization in the Textual TUI.

### 🔧 Improvements

1. **Tool Prompt Injection**:
   - Added best-practice prompts and tool-specific prompt injection hooks for safer tool usage.


## 0.6.0 (2026-02-24) - PyRepl, Textual TUI, and Durable Agent Memory

### 🎉 Major Features

1. **Builtin PyRepl Toolchain**: Added a production-ready Python REPL builtin based on a subprocess IPython runtime.
   - Introduced `SimpleLLMFunc.builtin.PyRepl` for persistent code execution across tool calls.
   - Added startup, active, and idle timeout controls for long-running agent workflows.
   - Improved execution reliability with worker supervision and richer runtime diagnostics.

2. **Textual TUI for `llm_chat`**: Added an out-of-the-box terminal UI powered by event stream updates.
   - New `@tui` integration with streaming markdown conversation rendering.
   - Added tool call arguments/results panels and model/tool usage statistics.
   - Added built-in quit controls and improved multi-turn interaction stability.

3. **Durable `SelfReference` Memory Contract**: Added self-reference memory controls for stateful agents.
   - Supports local durable memory semantics shared by chat loops and tools.
   - Enables safer prompt-level memory ownership and lifecycle control.

### ✨ New Features

1. **Tool Event Emission Pipeline**:
   - Added custom tool event emission support via event emitter hooks.
   - Improved event injection behavior in tool execution and ReAct orchestration.

2. **Input Stream Routing**:
   - Added tool input stream hooks to route pending tool input before normal chat turns.
   - Improved agent interactivity for tools that require follow-up user input.

3. **Examples and Developer Experience**:
   - Added dedicated examples for PyRepl, Textual TUI, custom tool events, and SelfReference.
   - Added translation and locale workflow scripts for documentation maintenance.

### 🔧 Improvements

1. **ReAct and Interface Reliability**:
   - Improved OpenAI-compatible interface behavior and response handling.
   - Refined ReAct message and tool-call execution flows to better support streaming and tool-event scenarios.

2. **Documentation Refresh**:
   - Added a dedicated PyRepl guide and expanded examples documentation.
   - Updated both English and Chinese docs for new runtime, hooks, and TUI capabilities.

### 🧪 Testing

- Added comprehensive test coverage for PyRepl runtime, TUI modules, event emitter/input stream hooks, self-reference behaviors, and OpenAI-compatible execution paths.

### ⚠️ Compatibility Notes

- The previous builtin `Kernel` workflow is superseded by `PyRepl`. If you referenced legacy kernel APIs, migrate imports and usage to `SimpleLLMFunc.builtin.PyRepl`.
- For full TUI observability, enable event streaming in `@llm_chat` (for example, `enable_event=True`).

## 0.5.0.beta1 (2025-01-09) - Event Stream & Type System Refactoring

> ⚠️ **Beta Release Notice**: This is a beta release. Optional breaking changes may be introduced. Please review the migration guide below if you encounter any issues.

### 🎉 Major Features

1. **Event Stream System**: A brand new observability system that supports real-time observation of ReAct execution cycles
   - New `enable_event` parameter (defaults to `False` for backward compatibility)
   - Supports 13 event types: ReAct start/end, LLM calls, tool calls, iterations, etc.
   - Tagged Union design, type-safe and flexible
   - Provides filter functions: `responses_only()`, `events_only()`, `filter_events()`
   - Provides decorator: `with_event_observer()` for event observation

2. **Type System Refactoring**: Unified type definitions, eliminated duplicates, improved type safety
   - New `type/tool_call.py`: Tool call related types
   - New `type/llm.py`: LLM response related types
   - New `type/hooks.py`: Hook system related types
   - Reuses OpenAI SDK types, reduces custom types
   - Unified export of all types to `type/__init__.py`

### ✨ New Features

1. **Event Type System**:
   - `ReactStartEvent`: ReAct cycle start
   - `LLMCallStartEvent` / `LLMCallEndEvent`: LLM call events
   - `LLMChunkArriveEvent`: Streaming chunk arrival (streaming mode only)
   - `ToolCallsBatchStartEvent` / `ToolCallsBatchEndEvent`: Tool call batch events
   - `ToolCallStartEvent` / `ToolCallEndEvent` / `ToolCallErrorEvent`: Individual tool call events
   - `ReactIterationStartEvent` / `ReactIterationEndEvent`: Iteration events
   - `ReactEndEvent`: ReAct cycle end

2. **Event Stream API**:
   ```python
   @llm_chat(llm_interface=llm, enable_event=True)
   async def my_chat(message: str):
       pass
   
   # Handle events and responses
   async for output in my_chat("Hello"):
       if output.type == "response":
           print(output.response)
       elif output.type == "event":
           print(output.event.event_type)
   ```

3. **Helper Utility Functions**:
   - `responses_only()`: Get only responses (backward compatible)
   - `events_only()`: Get only events
   - `filter_events()`: Filter specific event types
   - `with_event_observer()`: Add event observer decorator

### 🔧 Improvements

1. **Type System**:
   - Unified use of `MessageList` instead of `List[Dict[str, Any]]`
   - Unified use of `ToolDefinitionList` instead of `Optional[List[Dict[str, Any]]]`
   - Unified use of `ToolCall` type (directly reuses OpenAI SDK types)
   - Removed duplicate type definitions (`ReasoningDetail`, `ToolCallFunctionInfo`, `AccumulatedToolCall`)

2. **Code Organization**:
   - Removed `type/decorator.py`, migrated `HistoryList` to `type/hooks.py`
   - Updated all import paths to use unified type system

3. **Code Refactoring**:
   - Removed unnecessary dynamic imports in `ReAct.py`
   - Use module-level imports for better testability

### 📝 Documentation Updates

- Updated `llm_chat.md`: Added Event Stream usage instructions
- Updated `llm_function.md`: Added `enable_event` parameter documentation
- Updated `examples.md`: Added event stream example documentation
- Added new `event_stream.md`: Complete Event Stream guide

### ⚠️ Backward Compatibility & Breaking Changes

#### Fully Backward Compatible (Default Behavior)
- **Default behavior unchanged**: `enable_event=False` is the default, existing code requires no modifications
- All existing APIs remain unchanged
- Type system refactoring does not affect runtime behavior

#### Optional Breaking Changes (When Using New Features)

1. **Type Imports** (Optional):
   - If you were importing types from `SimpleLLMFunc.type.decorator`, you need to update imports:
     - `HistoryList` is now in `SimpleLLMFunc.type.hooks`
   - Most users are not affected as these are internal types

2. **Event Stream Return Type** (When `enable_event=True`):
   - When `enable_event=True`, the return type changes from `AsyncGenerator[Tuple[Any, MessageList], None]` to `AsyncGenerator[ReactOutput, None]`
   - Use `responses_only()` helper to maintain backward compatibility:
     ```python
     # Old way (still works with enable_event=False)
     async for response, messages in my_chat("Hello"):
         ...
     
     # New way (with enable_event=True)
     async for output in my_chat("Hello"):
         if output.type == "response":
             response, messages = output.response, output.messages
     ```

3. **Type Annotations** (For Type Checkers):
   - If you use type checkers (mypy, pyright), you may need to update type hints
   - The framework now uses more specific types from OpenAI SDK

### 🔮 Future Plans

- **v0.5.1**: `enable_event=True` will become the default
- **v0.5.2**: Remove `enable_event` parameter, always enable event stream

### Migration Guide

If you encounter any issues after upgrading:

1. **Check your imports**: If you import internal types, update them according to the breaking changes section
2. **Test with `enable_event=False`**: The default behavior is unchanged, so existing code should work
3. **Gradually adopt Event Stream**: Enable `enable_event=True` only when you need observability features
4. **Use helper functions**: `responses_only()` can help maintain compatibility when using event stream

---

## 0.4.2 Release Notes

### Refactoring

1. **ReAct Engine Return Type Enhancement**: Modified `execute_llm` function to return both response and message history in streaming mode.
   - Changed return type from `AsyncGenerator[Any, None]` to `AsyncGenerator[Tuple[Any, List[Dict[str, Any]]], None]`
   - Now yields `(response, current_messages.copy())` instead of just `response`
   - Creates a copy of `current_messages` to avoid modifying the original list
   - Updated related test files to adapt to the new return type

---

## 0.4.1 Release Notes

### Features

1. **Gemini 3 Pro Preview Support**: Added `reasoning_details` field support to enable compatibility with Google Gemini 3 Pro Preview model under OpenAI-compatible interface.

2. **Reasoning Details Extraction**: 
   - Added `ReasoningDetail` type definition in `extraction.py`
   - Implemented extraction functions for both streaming and non-streaming responses
   - Support for extracting reasoning details from message objects (both dict and object formats)

3. **Message Type Enhancement**: Extended message type definitions in `message.py` to include `reasoning_details` field support.

4. **ReAct Engine Integration**: Integrated reasoning details extraction and propagation in the ReAct engine for tool call workflows.

### Examples

- Updated example files (`llm_function_pydantic_example.py`, `parallel_toolcall_example.py`, `llm_chat_raw_tooluse_example.py`) to use `gemini-3-pro-preview` model.

---

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
