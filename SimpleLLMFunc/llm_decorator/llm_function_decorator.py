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
import json
from typing import (
    List,
    Callable,
    TypeVar,
    Dict,
    Any,
    cast,
    get_type_hints,
    Optional,
    Union,
    Tuple,
    NamedTuple,
    Awaitable,
)

import uuid

from pydantic import BaseModel

from SimpleLLMFunc.base.ReAct import execute_llm
from SimpleLLMFunc.base.messages import build_multimodal_content
from SimpleLLMFunc.base.post_process import (
    extract_content_from_response,
    process_response,
)
from SimpleLLMFunc.base.type_resolve import (
    get_detailed_type_description,
    has_multimodal_content,
    build_type_description_json,
    generate_example_object,
)
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import (
    app_log,
    async_log_context,
    get_current_trace_id,
    get_location,
    push_debug,
    push_error,
    push_warning,
)
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.utils import get_last_item_of_async_generator
from SimpleLLMFunc.llm_decorator.utils import process_tools
from SimpleLLMFunc.observability.langfuse_client import langfuse_client

T = TypeVar("T")


class FunctionCallContext(NamedTuple):
    """Context information for a function call."""

    func_name: str
    trace_id: str
    bound_args: Any  # inspect.BoundArguments
    signature: Any  # inspect.Signature
    type_hints: Dict[str, Any]
    return_type: Any
    docstring: str


class LLMCallParams(NamedTuple):
    """Encapsulates parameters required for LLM API calls."""

    messages: List[Dict[str, Any]]
    tool_param: Optional[List[Dict[str, Any]]]
    tool_map: Dict[str, Callable[..., Awaitable[Any]]]
    llm_kwargs: Dict[str, Any]


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
            # Prepare function call context and extract template parameters
            context, call_time_template_params = _prepare_function_call(
                func, args, kwargs
            )

            async with async_log_context(
                trace_id=context.trace_id,
                function_name=context.func_name,
                input_tokens=0,
                output_tokens=0,
            ):
                # Log function invocation with arguments
                args_str = json.dumps(
                    context.bound_args.arguments,
                    default=str,
                    ensure_ascii=False,
                    indent=4,
                )

                app_log(
                    f"Async LLM function '{context.func_name}' called with arguments: {args_str}",
                    location=get_location(),
                )

                # Build message list (system prompt + user prompt)
                messages = _build_messages(
                    context=context,
                    system_prompt_template=system_prompt_template,
                    user_prompt_template=user_prompt_template,
                    template_params=call_time_template_params,
                )

                # Prepare tools for LLM
                tool_param, tool_map = _prepare_tools_for_llm(
                    toolkit, context.func_name
                )

                # Package LLM call parameters
                llm_params = LLMCallParams(
                    messages=messages,
                    tool_param=tool_param,
                    tool_map=tool_map,
                    llm_kwargs=llm_kwargs,
                )

                # 创建 Langfuse parent span 用于追踪整个函数调用
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"{context.func_name}_function_call",
                    input=context.bound_args.arguments,
                    metadata={
                        "function_name": context.func_name,
                        "trace_id": context.trace_id,
                        "tools_available": len(toolkit) if toolkit else 0,
                        "max_tool_calls": max_tool_calls,
                    },
                ) as function_span:
                    try:
                        # Execute LLM call with retry logic
                        final_response = await _execute_llm_with_retry_async(
                            llm_interface=llm_interface,
                            context=context,
                            llm_params=llm_params,
                            max_tool_calls=max_tool_calls,
                        )
                        # Convert response to specified return type
                        result = _process_final_response(
                            final_response, context.return_type
                        )

                        # 更新 span 输出信息
                        function_span.update(
                            output={
                                "result": result,
                                "return_type": str(context.return_type),
                            },
                        )

                        return result
                    except Exception as exc:
                        # 更新 span 错误信息
                        function_span.update(
                            output={"error": str(exc)},
                        )
                        push_error(
                            f"Async LLM function '{context.func_name}' execution failed: {str(exc)}",
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


# ===== Default Prompt Templates =====

# Default system prompt template
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
Your task is to provide results that meet the requirements based on the **function description** 
and the user's request.

- Function Description:
    {function_description}

- You will receive the following parameters:
    {parameters_description}

- The type of content you need to return:
    {return_type_description}

Execution Requirements:
1. Use available tools to assist in completing the task if needed
2. Do not wrap results in markdown format or code blocks; directly output the expected content or JSON representation
"""

# Default user prompt template
DEFAULT_USER_PROMPT_TEMPLATE = """
The parameters provided are:
    {parameters}

Return the result directly without any explanation or formatting.
"""


# ===== Internal Helper Functions =====


def _prepare_function_call(
    func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[FunctionCallContext, Optional[Dict[str, Any]]]:
    """
    Prepare function call by processing parameter binding and creating execution context.

    Also extracts and returns call-time template parameters (if provided).

    Args:
        func: The decorated function
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function

    Returns:
        Tuple of (FunctionCallContext, call_time_template_params):
            - FunctionCallContext: Complete context for function execution
            - call_time_template_params: Optional template params for docstring substitution
    """
    # Extract call-time template parameters (if present)
    call_time_template_params = kwargs.pop("_template_params", None)

    # Extract function metadata
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return")
    docstring = func.__doc__ or ""
    func_name = func.__name__

    # Create unique trace ID for logging and request tracking
    context_current_trace_id = get_current_trace_id()
    current_trace_id = f"{func_name}_{uuid.uuid4()}" + (
        f"_{context_current_trace_id}" if context_current_trace_id else ""
    )

    # Bind arguments to function signature with defaults applied
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    context = FunctionCallContext(
        func_name=func_name,
        trace_id=current_trace_id,
        bound_args=bound_args,
        signature=signature,
        type_hints=type_hints,
        return_type=return_type,
        docstring=docstring,
    )

    return context, call_time_template_params


def _build_messages(
    context: FunctionCallContext,
    system_prompt_template: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    template_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build message list for LLM API call, with support for multimodal content.

    Constructs both system and user prompts, supporting text and multimodal formats.

    Args:
        context: Function call context containing function metadata and arguments
        system_prompt_template: Custom system prompt template (overrides default)
        user_prompt_template: Custom user prompt template (overrides default)
        template_params: DocString template parameters for substitution

    Returns:
        Message list ready for LLM API
    """
    # Check for multimodal content in arguments
    has_multimodal = _has_multimodal_content(
        context.bound_args.arguments, context.type_hints
    )

    if has_multimodal:
        # Build multimodal message list
        messages = _build_multimodal_messages(
            context, system_prompt_template, user_prompt_template, template_params
        )
    else:
        # Build traditional text message list
        messages = _build_prompts(
            docstring=context.docstring,
            arguments=context.bound_args.arguments,
            type_hints=context.type_hints,
            custom_system_template=system_prompt_template,
            custom_user_template=user_prompt_template,
            template_params=template_params,
        )

    return messages


def _prepare_tools_for_llm(
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]], func_name: str
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Callable[..., Awaitable[Any]]]]:
    """
    Prepare tools for LLM usage, returning tool parameters and tool mapping.

    Wrapper around the process_tools utility function.

    Args:
        toolkit: List of Tool objects or async callable functions
        func_name: Function name for logging

    Returns:
        Tuple of (tool_param, tool_map):
            - tool_param: Tool definitions formatted for LLM API
            - tool_map: Mapping from tool names to callable implementations
    """
    return process_tools(toolkit, func_name)


def _process_final_response(response: Any, return_type: Any) -> Any:
    """
    Process final LLM response and convert to specified return type.

    Args:
        response: Raw LLM response
        return_type: Expected return type

    Returns:
        Response converted to specified return type
    """
    return process_response(response, return_type)


def _build_prompts(
    docstring: str,
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    custom_system_template: Optional[str] = None,
    custom_user_template: Optional[str] = None,
    template_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Build system and user prompts for LLM from function metadata.

    Process Flow:
    1. Substitute template_params into docstring
    2. Extract parameter types from type hints
    3. Build parameter type descriptions
    4. Get detailed description of return type
    5. Format system and user prompts using templates

    Args:
        docstring: Function docstring
        arguments: Function argument values
        type_hints: Type hints for function
        custom_system_template: Custom system prompt template (overrides default)
        custom_user_template: Custom user prompt template (overrides default)
        template_params: DocString template parameters

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Step 1: Process DocString template parameter substitution
    processed_docstring = docstring
    if template_params:
        try:
            processed_docstring = docstring.format(**template_params)
        except KeyError as e:
            push_warning(
                f"DocString template parameter substitution failed: missing parameter {e}. Using original DocString.",
                location=get_location(),
            )
        except Exception as e:
            push_warning(
                f"Error during DocString template parameter substitution: {str(e)}. Using original DocString.",
                location=get_location(),
            )

    # Remove return type hint, keeping only parameter types
    param_type_hints = {k: v for k, v in type_hints.items() if k != "return"}

    # Step 2: Build parameter type descriptions (for system prompt)
    param_type_descriptions = []
    for param_name, param_type in param_type_hints.items():
        type_str = (
            get_detailed_type_description(param_type) if param_type else "Unknown Type"
        )
        param_type_descriptions.append(f"  - {param_name}: {type_str}")

    # Step 3: Get return type detailed description
    return_type = type_hints.get("return", None)

    # For primitive types, use simple text description
    # For complex types (BaseModel, List, Dict, Union), use structured JSON with example
    try:
        if return_type is None:
            return_type_description = "未知类型"
        elif return_type in (str, int, float, bool, type(None)):
            # Primitive types: simple description
            return_type_description = get_detailed_type_description(return_type)
        else:
            # Complex types: structured JSON description with example
            from typing import get_origin, Union as TypingUnion

            is_complex = False
            if isinstance(return_type, type) and issubclass(return_type, BaseModel):
                is_complex = True
            else:
                origin = getattr(return_type, "__origin__", None) or get_origin(
                    return_type
                )
                if origin in (list, List, dict, Dict, TypingUnion):
                    is_complex = True

            if is_complex:
                type_json_obj = build_type_description_json(return_type)
                example_obj = generate_example_object(return_type)
                import json as _json

                return_type_description = (
                    "Type Description (JSON):\n"
                    + _json.dumps(type_json_obj, ensure_ascii=False, indent=2)
                    + "\n\nExample JSON:\n"
                    + _json.dumps(example_obj, ensure_ascii=False, indent=2)
                )
            else:
                # Fallback to simple description for other types
                return_type_description = get_detailed_type_description(return_type)
    except Exception as e:
        push_warning(
            f"Failed to generate structured JSON type description, falling back to text format: {str(e)}",
            location=get_location(),
        )
        return_type_description = get_detailed_type_description(return_type)

    # Step 4: Use custom or default templates
    system_template = custom_system_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE
    user_template = custom_user_template or DEFAULT_USER_PROMPT_TEMPLATE

    # Step 5: Build system prompt
    system_prompt = system_template.format(
        function_description=processed_docstring,
        parameters_description="\n".join(param_type_descriptions),
        return_type_description=return_type_description,
    )

    # Step 6: Build user prompt (with parameter values)
    user_param_values = []
    for param_name, param_value in arguments.items():
        user_param_values.append(f"  - {param_name}: {param_value}")

    user_prompt = user_template.format(
        parameters="\n".join(user_param_values),
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt.strip(),
        },
        {
            "role": "user",
            "content": user_prompt.strip(),
        },
    ]

    return messages


# ===== Async Decorator and Helper Functions =====


async def _execute_llm_with_retry_async(
    llm_interface: LLM_Interface,
    context: FunctionCallContext,
    llm_params: LLMCallParams,
    max_tool_calls: int,
) -> Any:
    """
    Execute LLM call asynchronously with retry logic.

    Handles the core async LLM execution, including automatic retry if response
    content is empty after the call completes.

    Args:
        llm_interface: LLM interface instance for API calls
        context: Function call context containing function metadata
        llm_params: LLM call parameters (messages, tools, kwargs)
        max_tool_calls: Maximum number of tool calls

    Returns:
        Final LLM response

    Raises:
        ValueError: If response content remains empty after all retries
    """
    push_debug("Starting async LLM call...", location=get_location())

    # Call async LLM and get final response
    response_generator = execute_llm(
        llm_interface=llm_interface,
        messages=llm_params.messages,
        tools=llm_params.tool_param,
        tool_map=llm_params.tool_map,
        max_tool_calls=max_tool_calls,
        **llm_params.llm_kwargs,  # Pass additional LLM parameters
    )

    # Get last response as final result
    final_response = await get_last_item_of_async_generator(response_generator)

    # Check if content field in final response is empty
    retry_times = llm_params.llm_kwargs.get("retry_times", 2)
    content = ""
    if hasattr(final_response, "choices") and len(final_response.choices) > 0:  # type: ignore
        message = final_response.choices[0].message  # type: ignore
        content = message.content if message and hasattr(message, "content") else ""

    if content == "":
        # If response content is empty, log warning and retry
        push_warning(
            f"Async LLM function '{context.func_name}' returned empty response content, will retry automatically.",
            location=get_location(),
        )
        # Retry LLM call
        while (
            retry_times > 0
            and hasattr(final_response.choices[0].message, "content")  # type: ignore
            and final_response.choices[0].message.content == ""  # type: ignore
        ):
            retry_times -= 1
            push_debug(
                f"Async LLM function '{context.func_name}' retry attempt {llm_params.llm_kwargs.get('retry_times', 2) - retry_times + 1}...",
                location=get_location(),
            )
            response_generator = execute_llm(
                llm_interface=llm_interface,
                messages=llm_params.messages,
                tools=llm_params.tool_param,
                tool_map=llm_params.tool_map,
                max_tool_calls=max_tool_calls,
                **llm_params.llm_kwargs,  # Pass additional LLM parameters
            )
            final_response = await get_last_item_of_async_generator(response_generator)

            content = extract_content_from_response(final_response, context.func_name)
            if content != "":  # type: ignore
                break

    content = extract_content_from_response(final_response, context.func_name)

    if content == "":
        push_error(
            f"Async LLM function '{context.func_name}' response content still empty, retry attempts exhausted.",
            location=get_location(),
        )
        raise ValueError("LLM response content is empty after retries.")

    # Log final response
    push_debug(
        f"Async LLM function '{context.func_name}' received response {json.dumps(final_response, default=str, ensure_ascii=False, indent=2)}",
        location=get_location(),
    )

    return final_response


def _has_multimodal_content(
    arguments: Dict[str, Any], type_hints: Dict[str, Any]
) -> bool:
    """
    Check if arguments contain multimodal content (images, etc.).

    Args:
        arguments: Function argument values
        type_hints: Type hints for function

    Returns:
        True if multimodal content is present, False otherwise
    """
    return has_multimodal_content(arguments, type_hints)


def _build_multimodal_messages(
    context: FunctionCallContext,
    system_prompt_template: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    template_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build multimodal message list for LLM API call.

    Constructs messages with multimodal content (text + images).

    Args:
        context: Function call context
        system_prompt_template: Custom system prompt template (overrides default)
        user_prompt_template: Custom user prompt template (overrides default)
        template_params: DocString template parameters

    Returns:
        Message list with multimodal content
    """
    # Build system prompt (still plain text)
    system_prompt, _ = _build_prompts(
        docstring=context.docstring,
        arguments=context.bound_args.arguments,
        type_hints=context.type_hints,
        custom_system_template=system_prompt_template,
        custom_user_template=user_prompt_template,
        template_params=template_params,
    )

    # Build multimodal user message content
    user_content = build_multimodal_content(
        context.bound_args.arguments, context.type_hints
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    push_debug(f"System prompt: {system_prompt}", location=get_location())
    push_debug(
        f"Multimodal user message contains {len(user_content)} content blocks",
        location=get_location(),
    )

    return messages
