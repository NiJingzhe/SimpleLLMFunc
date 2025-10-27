# SimpleLLMFunc - AI Coding Agent Instructions

## 📋 Project Overview

SimpleLLMFunc is a lightweight LLM application framework enabling developers to write LLM-powered workflows and agents using Python decorators with DocString-as-Prompt philosophy. Core idea: **"Everything is Function, Prompt is Code"**.

## 🏗 Architecture & Key Components

### 1. **Core Design Pattern: Prompt as DocString**
- Function behavior is defined in the DocString, not the function body (which stays empty with `pass`)
- Decorators (`@llm_function`, `@llm_chat`, `@tool`) intercept function calls and delegate to LLM
- Type hints on parameters and return types ensure type safety and automatic validation
- Example from `/SimpleLLMFunc/llm_decorator/llm_function_decorator.py`:
  ```python
  @llm_function(llm_interface=my_llm)
  async def analyze_product_review(product_name: str, review_text: str) -> ProductReview:
      """You are a product review expert analyzing the following review...
      
      Args:
          product_name: Name of the product
          review_text: User review content
      
      Returns:
          Structured ProductReview object with rating, pros, cons, summary
      """
      pass  # Prompt as Code - LLM handles the actual execution
  ```

### 2. **Async-First Architecture**
- **All LLM decorators ONLY support `async def` functions** (non-negotiable requirement)
- Use `await` when calling decorated functions or `asyncio.run()` at top level
- Example from examples/llm_chat_example.py: `async for content, updated_history in async_simple_chat(...)`
- Enables high-concurrency API calls via native asyncio integration

### 3. **Three-Tier Component Stack**

#### **Tier 1: Decorators** (`llm_decorator/`)
- **`@llm_function`**: Stateless single-call transformations (request → response)
- **`@llm_chat`**: Multi-turn conversations with history management
- **`@tool`**: Tool definitions for use by agents (async functions only)
- All accept `toolkit` parameter for tool composition
- Support `_template_params` for dynamic DocString templating (v0.2.14+)

#### **Tier 2: LLM Interface** (`interface/`)
- **`LLM_Interface` (abstract)**: Base contract for all LLM implementations
- **`OpenAICompatible`**: Universal implementation for any OpenAI-compatible API
  - Loads config from JSON files via `load_from_json_file()` method
  - Auto-routes to correct provider/model from hierarchical config
- **`APIKeyPool`**: Min-heap based load balancing for multiple API keys
- **`TokenBucket`**: Rate limiting to prevent API throttling

#### **Tier 3: Base Modules** (`base/`)
- **`messages.py`**: Constructs system/user prompts + handles multimodal content
- **`post_process.py`**: Deserializes LLM responses to target types
- **`type_resolve.py`**: Analyzes function signatures for type information
- **`ReAct.py`**: Orchestrates LLM calls with tool execution loops (max 5 by default)
- **`tool_call.py`**: Tool invocation and result formatting

### 4. **Configuration & Providers** 
- `.env` file: `LOG_LEVEL=DEBUG` setting
- `provider.json`: Hierarchical structure with vendor → model → credentials
  ```json
  {
    "volc_engine": {
      "deepseek-v3-250324": {
        "api_keys": ["key1", "key2"],
        "base_url": "https://api.volc.example.com/v1",
        "model": "deepseek-chat",
        "rate_limit_capacity": 10
      }
    }
  }
  ```
- Load via: `OpenAICompatible.load_from_json_file("provider.json")["volc_engine"]["deepseek-v3-250324"]`

### 5. **Logging & Tracing System** (`logger/`)
- Console-only output (no file persistence in v0.2.13+)
- Auto-generated `trace_id` in format `{func_name}_{uuid}` for correlated logging
- Context manager: `async_log_context(trace_id=..., function_name=...)`
- Functions: `app_log()`, `push_error()`, `push_debug()`, `push_warning()`
- Structured context inheritance across async calls
- See `logger/core.py` for logger setup

## 🔄 Execution Data Flow

### Standard `llm_function` Flow:
1. **Call Capture** → Decorator intercepts `async def my_func(...)`
2. **Argument Binding** → Map args/kwargs to function signature
3. **Prompt Construction**:
   - System prompt = function's DocString + custom template (if provided)
   - User prompt = formatted arguments + type descriptions
4. **LLM Invocation** → Send to `llm_interface.chat(messages=[...])`
5. **Tool Loop** (if toolkit provided):
   - Check if LLM invoked tools via `tool_calls` field
   - Execute tools via `base/ReAct.py` (max 5 iterations)
   - Include tool results in next LLM message
6. **Response Deserialization** → Convert LLM text output to return type
   - If return type is Pydantic model: auto-parse JSON
   - If return type is primitive: direct conversion

### `llm_chat` Flow (Multi-turn):
1. Accepts `history: List[Dict[str, str]]` parameter
2. Maintains conversation state across calls
3. Yields tuples of `(content_chunk, updated_history)` in stream mode
4. Returns mode (`text` vs `raw`) affects tool call visibility in history
5. Final yield contains only `("", updated_history)` to signal completion

## 🛠 Tool System Patterns

### Function Decorator Way (Recommended):
```python
from SimpleLLMFunc.tool import tool
from SimpleLLMFunc.type import ImgPath, ImgUrl

@tool(name="get_weather", description="Get weather for location")
async def get_weather(location: str, days: int = 1) -> dict:
    """Get weather forecast
    
    Args:
        location: City name or coordinates
        days: Forecast days (default 1)
        
    Returns:
        Weather data dictionary
    """
    # Actual implementation
    return {"location": location, "forecast": [...]}

# Multimodal tool returns (v0.2.10+):
@tool(name="generate_chart", description="Generate chart image")
async def generate_chart(data: str) -> ImgPath:
    """Return local image path for tool use"""
    return ImgPath("/path/to/chart.png")

@tool(name="search_image", description="Search web images")
async def search_image(query: str) -> ImgUrl:
    """Return web image URL for tool use"""
    return ImgUrl("https://example.com/image.jpg")
```

### Usage in Decorators:
```python
@llm_function(
    llm_interface=llm,
    toolkit=[get_weather, generate_chart]  # Pass decorated functions directly
)
async def plan_trip(destination: str) -> str:
    """Plan a trip using tools to get weather and generate itinerary"""
    pass
```

## 🚀 Development Patterns

### 1. **Type Safety through Pydantic**
- Always use Pydantic `BaseModel` for complex return types
- Define `Field(..., description="...")` for LLM understanding
- Example from README.md `ProductReview`:
  ```python
  class ProductReview(BaseModel):
      rating: int = Field(..., description="1-5 star rating")
      pros: List[str] = Field(..., description="List of advantages")
      cons: List[str] = Field(..., description="List of disadvantages")
      summary: str = Field(..., description="Summary text")
  ```

### 2. **Multimodal Input Handling**
- Use `Text`, `ImgUrl`, `ImgPath` from `SimpleLLMFunc.type`
- Framework auto-converts to proper message format
- Example:
  ```python
  from SimpleLLMFunc.type import Text, ImgUrl, ImgPath
  
  @llm_function(llm_interface=llm)
  async def analyze_images(
      description: Text,
      web_img: ImgUrl,
      local_img: ImgPath
  ) -> str:
      """Analyze and compare images"""
      pass
  ```

### 3. **Custom Templates for Reusable Functions**
- Use `system_prompt_template` / `user_prompt_template` parameters
- Reference variables with `{variable_name}` syntax
- Combine with `_template_params` at call time (v0.2.14+):
  ```python
  @llm_function(
      llm_interface=llm,
      system_prompt_template="You are a {role} expert..."
  )
  async def expert_analysis(topic: str) -> str:
      """Analyze the topic"""
      pass
  
  # Call with different roles:
  result = await expert_analysis(
      topic="Python design patterns",
      _template_params={"role": "Software Architecture"}
  )
  ```

### 4. **Error Handling & Validation**
- LLM response validation happens automatically in `base/post_process.py`
- Weak models may fail JSON parsing (issue logged, user must handle)
- Wrap calls in try-except for production:
  ```python
  try:
      result = await analyze_product_review(name, review)
  except Exception as e:
      app_log(f"Analysis failed: {e}")
      # Fallback logic
  ```

### 5. **Logging Integration**
```python
from SimpleLLMFunc.logger import app_log, async_log_context

@llm_function(llm_interface=llm)
async def my_function(text: str) -> str:
    """Process text"""
    pass

# In calling code:
async with async_log_context(trace_id="custom_id", function_name="my_func"):
    result = await my_function("input")
    app_log("Processing complete")  # Auto-inherits trace_id
```

## 📁 Critical Files by Use Case

| Use Case | Key Files |
|----------|-----------|
| Add new LLM provider | `interface/openai_compatible.py`, update `provider.json` |
| Create LLM function | `llm_decorator/llm_function_decorator.py` (see flow 219-260) |
| Create agent with tools | `llm_decorator/llm_chat_decorator.py`, `tool/tool.py` |
| Debug tool invocation | `base/ReAct.py` (tool loop orchestration), `base/tool_call.py` |
| Extend return types | `base/post_process.py`, `base/type_resolve.py` |
| Customize prompts | `base/messages.py` (lines 1-80 for template logic) |

## ⚠️ Common Pitfalls

1. **Using sync functions with decorators** → Will fail at runtime. **Always use `async def`**
2. **Weak models + Pydantic return types** → JSON parsing may fail silently. Verify with strong models (gpt-4, deepseek-v3) first
3. **Forgetting to pass `_template_params`** → Template variables won't be substituted
4. **Not awaiting or using `asyncio.run()`** → Coroutine object returned instead of result
5. **Tool parameter descriptions** → Extract from DocString (`Args:` section) and Pydantic `Field(description=...)`

## 🔗 Import Patterns

```python
# Core decorators & interfaces
from SimpleLLMFunc import (
    llm_function, llm_chat,
    OpenAICompatible,
    app_log, async_log_context
)

# Types
from SimpleLLMFunc.type import Text, ImgUrl, ImgPath

# Tool system
from SimpleLLMFunc.tool import tool, Tool

# Logger functions
from SimpleLLMFunc.logger import push_error, push_debug
```

## 📊 Testing Strategy

- **Integration tests** in `/examples` folder demonstrate real-world flows
- Use small/cheap models first (e.g., gpt-3.5-turbo) before production
- Validate Pydantic models separately from LLM output processing
- Check trace logs in `trace_indices/` for debugging multi-tool workflows

## 🎯 For This Repository Branch: `refactor/recoding-all`

This branch is actively refactoring all components. Key areas under transformation:
- Logger system: migrated to console-only in v0.2.13
- Tool system: multimodal return support stabilized in v0.2.8+
- Type inference: improved in v0.2.12+

Always refer to CHANGELOG.md for the latest breaking changes.
