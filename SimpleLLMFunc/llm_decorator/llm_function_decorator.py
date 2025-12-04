"""
LLM Function Decorator Module

This module provides LLM function decorators that delegate the execution of ordinary Python
functions to large language models. Using this decorator, simply define the function signature
(parameters and return type), then describe the function's execution strategy in the docstring.

Data Flow:
1. User defines function signature and docstring
2. Decorator captures function calls, extracts parameters and type information
3. Constructs system and user prompts
4. Calls LLM for reasoning
5. Processes tool calls (if necessary)
6. Converts LLM response to specified return type
7. Returns result to caller

Example:
```python
@llm_function(llm_interface=my_llm)
async def generate_summary(text: str) -> str:
    \"\"\"Generate a concise summary from the input text, should contain main points.\"\"\"
    pass
```
"""

import inspect
from functools import wraps
from typing import (
    List,
    Callable,
    TypeVar,
    Dict,
    Any,
    cast,
    Optional,
    Union,
    Awaitable,
)

from SimpleLLMFunc.llm_decorator.steps.common import (
    parse_function_signature,
    setup_log_context,
)
from SimpleLLMFunc.llm_decorator.steps.function import (
    build_initial_prompts,
    execute_react_loop,
    parse_and_validate_response,
)
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import push_error
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.observability.langfuse_client import langfuse_client

T = TypeVar("T")


def llm_function(
    llm_interface: LLM_Interface,
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]] = None,
    max_tool_calls: int = 5,
    system_prompt_template: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    **llm_kwargs: Any,
) -> Callable[
    [Union[Callable[..., T], Callable[..., Awaitable[T]]]], Callable[..., Awaitable[T]]
]:
    """
    Async LLM function decorator that delegates function execution to a large language model.

    This decorator provides native async implementation, ensuring that LLM calls do not
    block the event loop during execution.

    ## Usage
    1. Define an async function with type annotations for parameters and return value
    2. Describe the goal, constraints, or execution strategy in the function's docstring
    3. Use `@llm_function` decorator and obtain results via `await`

    ## Async Features
    - LLM calls execute directly through `await`, seamlessly cooperating with other coroutines
    - Compatible with `asyncio.gather` and other concurrent primitives
    - Tool calls are likewise completed asynchronously

    ## Parameter Passing Flow
    1. Decorator captures all parameters at call time
    2. Parameters are formatted into user prompt and sent to LLM
    3. Function docstring serves as system prompt guiding the LLM
    4. Return value is parsed according to type annotation

    ## Tool Usage
    - Tools provided via `toolkit` can be invoked by LLM during reasoning
    - Supports `Tool` instances or async functions decorated with `@tool`

    ## Custom Prompt Templates
    - Override default prompt format via `system_prompt_template` and `user_prompt_template`

    ## Response Handling
    - Response result is automatically converted based on return type annotation
    - Supports basic types, dictionaries, and Pydantic models

    ## LLM Interface Parameters
    - Settings passed via `**llm_kwargs` are directly forwarded to the underlying LLM interface

    Example:
        ```python
        @llm_function(llm_interface=my_llm)
        async def summarize_text(text: str, max_words: int = 100) -> str:
            \"\"\"Generate a summary of the input text, not exceeding the specified word count.\"\"\"
            ...

        summary = await summarize_text(long_text, max_words=50)
        ```

    Concurrent Example:
        ```python
        texts = ["text1", "text2", "text3"]

        @llm_function(llm_interface=my_llm)
        async def analyze_sentiment(text: str) -> str:
            \"\"\"Analyze the sentiment tendency of the text.\"\"\"
            ...

        results = await asyncio.gather(
            *(analyze_sentiment(text) for text in texts)
        )
        ```
    """

    def decorator(
        func: Union[Callable[..., T], Callable[..., Awaitable[T]]],
    ) -> Callable[..., Awaitable[T]]:
        signature = inspect.signature(func)
        docstring = func.__doc__ or ""
        func_name = func.__name__

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Step 1: 解析函数签名
            signature, template_params = parse_function_signature(func, args, kwargs)

            # Step 2: 设置日志上下文
            async with setup_log_context(
                func_name=signature.func_name,
                trace_id=signature.trace_id,
                arguments=signature.bound_args.arguments,
            ):
                # 创建 Langfuse parent span
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"{signature.func_name}_function_call",
                    input=signature.bound_args.arguments,
                    metadata={
                        "function_name": signature.func_name,
                        "trace_id": signature.trace_id,
                        "tools_available": len(toolkit) if toolkit else 0,
                        "max_tool_calls": max_tool_calls,
                    },
                ) as function_span:
                    try:
                        # Step 3: 构建初始提示
                        messages = build_initial_prompts(
                            signature=signature,
                            system_prompt_template=system_prompt_template,
                            user_prompt_template=user_prompt_template,
                            template_params=template_params,
                        )

                        # Step 4: 执行 ReAct 循环
                        final_response = await execute_react_loop(
                            llm_interface=llm_interface,
                            messages=messages,
                            toolkit=toolkit,
                            max_tool_calls=max_tool_calls,
                            llm_kwargs=llm_kwargs,
                            func_name=signature.func_name,
                        )

                        # Step 5: 解析和验证响应
                        result = parse_and_validate_response(
                            response=final_response,
                            return_type=signature.return_type,
                            func_name=signature.func_name,
                        )

                        # 更新 Langfuse span
                        function_span.update(
                            output={
                                "result": result,
                                "return_type": str(signature.return_type),
                            },
                        )

                        return result
                    except Exception as exc:
                        # 更新 span 错误信息
                        function_span.update(
                            output={"error": str(exc)},
                        )
                        push_error(
                            f"Async LLM function '{signature.func_name}' execution failed: {str(exc)}",
                            location=get_location(),
                        )
                        raise

        # Preserve original function metadata
        async_wrapper.__name__ = func_name
        async_wrapper.__doc__ = docstring
        async_wrapper.__annotations__ = func.__annotations__
        setattr(async_wrapper, "__signature__", signature)

        return cast(Callable[..., Awaitable[T]], async_wrapper)

    return decorator


async_llm_function = llm_function
