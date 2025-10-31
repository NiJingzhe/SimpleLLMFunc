import inspect
import json
import uuid
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
    cast,
    get_type_hints,
    Literal,
)

from SimpleLLMFunc.base.ReAct import execute_llm
from SimpleLLMFunc.base.messages import build_multimodal_content
from SimpleLLMFunc.base.post_process import (
    extract_content_from_response,
    extract_content_from_stream_response,
)
from SimpleLLMFunc.base.type_resolve import has_multimodal_content
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
from SimpleLLMFunc.llm_decorator.utils import process_tools
from SimpleLLMFunc.observability.langfuse_client import langfuse_client

# Type aliases
MessageDict = Dict[str, Any]  # Dictionary representing a message
HistoryList = List[
    MessageDict
]  # List of message dictionaries representing conversation history
ToolkitList = List[
    Union[Tool, Callable[..., Awaitable[Any]]]
]  # List of Tool objects or async functions

# Type variables
T = TypeVar("T")
P = ParamSpec("P")

# Constants
HISTORY_PARAM_NAMES: List[str] = [
    "history",
    "chat_history",
]  # Valid parameter names for conversation history
DEFAULT_MAX_TOOL_CALLS: int = (
    5  # Default maximum number of tool calls to prevent infinite loops
)


def llm_chat(
    llm_interface: LLM_Interface,
    toolkit: Optional[ToolkitList] = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    stream: bool = False,
    return_mode: Literal["text", "raw"] = "text",
    **llm_kwargs: Any,
) -> Callable[
    [Union[Callable[P, Any], Callable[P, Awaitable[Any]]]],
    Callable[P, AsyncGenerator[Tuple[Any, HistoryList], None]],
]:
    """
    Async LLM chat decorator for implementing asynchronous conversational interactions with
    large language models, with support for tool calling and conversation history management.

    This decorator provides native async support and returns an AsyncGenerator.

    ## Features
    - Automatic conversation history management
    - Tool calling and function execution support
    - Multimodal content support (text, image URLs, local images)
    - Streaming response support
    - Automatic history filtering and cleanup
    - Native async support with non-blocking execution

    ## Parameter Passing Rules
    - Decorator passes function parameters as `param_name: param_value` format to the LLM as user messages
    - `history`/`chat_history` parameters are treated specially and excluded from user messages
    - Function docstring is passed to the LLM as system prompt

    ## Conversation History Format
    ```python
    [
        {"role": "user", "content": "user message"},
        {"role": "assistant", "content": "assistant response"},
        {"role": "system", "content": "system message"}
    ]
    ```

    ## Return Value Format
    ```python
    AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]
    ```
    - `str`: Assistant's response content
    - `List[Dict[str, str]]`: Filtered conversation history (excluding tool call information)

    Args:
        llm_interface: LLM interface instance for communicating with the language model
        toolkit: Optional list of tools, can be Tool objects or functions decorated with @tool
        max_tool_calls: Maximum number of tool calls to prevent infinite loops
        stream: Whether to use streaming responses
        return_mode: Return mode, either "text" or "raw" (default: "text")
            - "text" mode: returns response as string, history as List[Dict[str, str]]
            - "raw" mode: returns raw OAI API response, history as List[Dict[str, str]]
        **llm_kwargs: Additional keyword arguments passed directly to the LLM interface

    Returns:
        Decorated async generator function that yields (response_content, updated_history) tuples

    Example:
        ```python
        @llm_chat(llm_interface=my_llm)
        async def chat_with_llm(message: str, history: List[Dict[str, str]] = []):
            '''System prompt information'''
            pass

        async for response, updated_history in chat_with_llm("Hello", history=[]):
            print(response)
        ```
    """

    def decorator(
        func: Union[Callable[P, Any], Callable[P, Awaitable[Any]]],
    ) -> Callable[P, AsyncGenerator[Tuple[Any, HistoryList], None]]:
        # Extract function metadata
        signature: inspect.Signature = inspect.signature(func)
        type_hints: Dict[str, Any] = get_type_hints(func)
        docstring: str = func.__doc__ or ""
        func_name: str = func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate unique trace ID for request tracking
            context_trace_id: Optional[str] = get_current_trace_id()
            current_trace_id: str = f"{func_name}_{uuid.uuid4()}"
            if context_trace_id:
                current_trace_id += f"_{context_trace_id}"

            # Bind arguments to function signature with defaults applied
            bound_args: inspect.BoundArguments = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Use async logging context
            async with async_log_context(
                trace_id=current_trace_id,
                function_name=func_name,
                input_tokens=0,
                output_tokens=0,
            ):
                # 创建 Langfuse parent span 用于追踪整个聊天函数调用
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"{func_name}_chat_call",
                    input=bound_args.arguments,
                    metadata={
                        "function_name": func_name,
                        "trace_id": current_trace_id,
                        "tools_available": len(toolkit) if toolkit else 0,
                        "max_tool_calls": max_tool_calls,
                        "stream": stream,
                        "return_mode": return_mode,
                    },
                ) as chat_span:
                    try:
                        # 收集所有响应内容用于 span 输出更新
                        collected_responses = []
                        final_history = None

                        async for result in _async_llm_chat_impl(
                            func_name=func_name,
                            signature=signature,
                            type_hints=type_hints,
                            docstring=docstring,
                            args=args,
                            kwargs=kwargs,
                            llm_interface=llm_interface,
                            toolkit=toolkit,
                            max_tool_calls=max_tool_calls,
                            stream=stream,
                            return_mode=return_mode,
                            use_log_context=False,  # Log context already set in wrapper
                            **llm_kwargs,
                        ):
                            response_content, history = result
                            collected_responses.append(response_content)
                            final_history = history
                            yield result

                        # 更新 span 输出信息
                        chat_span.update(
                            output={
                                "responses": collected_responses,
                                "final_history": final_history,
                                "total_responses": len(collected_responses),
                            },
                        )
                    except Exception as exc:
                        # 更新 span 错误信息
                        chat_span.update(
                            output={"error": str(exc)},
                        )
                        raise

        # Preserve original function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__
        wrapper.__signature__ = signature  # type: ignore

        return cast(
            Callable[P, AsyncGenerator[Tuple[Any, HistoryList], None]],
            wrapper,
        )

    return decorator


async def _async_llm_chat_impl(
    func_name: str,
    signature: inspect.Signature,
    type_hints: Dict[str, Any],
    docstring: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    llm_interface: LLM_Interface,
    toolkit: Optional[ToolkitList],
    max_tool_calls: int,
    stream: bool,
    return_mode: Literal["text", "raw"] = "text",
    use_log_context: bool = True,
    **llm_kwargs: Any,
) -> AsyncGenerator[Tuple[Any, HistoryList], None]:
    """
    Shared async LLM chat implementation logic.

    Handles the core workflow of extracting arguments, building messages,
    processing tools, and streaming responses from the LLM.

    Args:
        func_name: Name of the decorated function
        signature: Function signature for parameter binding
        type_hints: Type hints for the function parameters
        docstring: Function docstring to be used as system prompt
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function
        llm_interface: LLM interface instance for API calls
        toolkit: List of available tools for the LLM to call
        max_tool_calls: Maximum number of tool calls to prevent infinite loops
        stream: Whether to stream responses from the LLM
        return_mode: Response format ("text" or "raw")
        use_log_context: Whether to wrap execution in async logging context
        **llm_kwargs: Additional LLM interface parameters

    Yields:
        Tuples of (response_content, updated_conversation_history)
    """
    # Bind arguments to function signature with defaults applied
    bound_args: inspect.BoundArguments = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    async def _execute_impl() -> AsyncGenerator[Tuple[Any, HistoryList], None]:
        # Step 1: Process and validate tools
        tool_param_for_api: Optional[List[Dict[str, Any]]]
        tool_map: Dict[str, Callable[..., Awaitable[Any]]]
        tool_param_for_api, tool_map = _process_tools(toolkit, func_name)

        # Step 2: Check for multimodal content in arguments
        has_multimodal: bool = has_multimodal_content(
            bound_args.arguments, type_hints, exclude_params=HISTORY_PARAM_NAMES
        )

        # Step 3: Build user message content (text or multimodal)
        user_message_content: Union[str, List[Dict[str, Any]]] = (
            _build_user_message_content(
                bound_args.arguments, type_hints, has_multimodal
            )
        )

        # Step 4: Extract and validate conversation history from arguments
        custom_history: Optional[HistoryList] = _extract_history_from_args(
            bound_args.arguments, func_name
        )

        # Step 5: Construct complete message list (system + history + user message)
        current_messages: HistoryList = _build_messages(
            docstring,
            custom_history,
            user_message_content,
            tool_param_for_api,
            has_multimodal,
        )

        # Step 6: Log debug information about prepared messages
        push_debug(
            f"LLM Chat '{func_name}' will execute with messages:"
            f"\n{json.dumps(current_messages, ensure_ascii=False, indent=2)}",
            location=get_location(),
        )

        # Step 7: Execute LLM call and begin response streaming
        complete_content: str = ""
        response_flow = execute_llm(
            llm_interface=llm_interface,
            messages=current_messages,
            tools=tool_param_for_api,
            tool_map=tool_map,
            max_tool_calls=max_tool_calls,
            stream=stream,
            **llm_kwargs,
        )

        # Step 8: Process response stream (async iteration over LLM responses)
        async for response in response_flow:
            app_log(
                f"LLM Chat '{func_name}' received response:"
                f"\n{json.dumps(response, default=str, ensure_ascii=False, indent=2)}",
                location=get_location(),
            )

            if return_mode == "raw":
                # Return raw API response
                yield response, current_messages
            else:
                # Extract content based on stream mode
                if stream:
                    content = extract_content_from_stream_response(response, func_name)
                else:
                    content = extract_content_from_response(response, func_name) or ""
                complete_content += content
                yield content, current_messages

        # Step 9: Emit final empty content to signal stream completion
        if return_mode == "text":
            # Emit empty string to indicate end of stream in text mode
            yield "", current_messages

    if use_log_context:
        # Generate unique trace ID for request tracking
        context_trace_id = get_current_trace_id()
        current_trace_id = f"{func_name}_{uuid.uuid4()}"
        if context_trace_id:
            current_trace_id += f"_{context_trace_id}"

        async with async_log_context(
            trace_id=current_trace_id,
            function_name=func_name,
            input_tokens=0,
            output_tokens=0,
        ):
            try:
                async for result in _execute_impl():
                    yield result
            except Exception as e:
                push_error(
                    f"LLM Chat '{func_name}' execution failed: {str(e)}",
                    location=get_location(),
                )
                raise
    else:
        # Execute without additional logging context
        try:
            async for result in _execute_impl():
                yield result
        except Exception as e:
            push_error(
                f"LLM Chat '{func_name}' execution failed: {str(e)}",
                location=get_location(),
            )
            raise


async_llm_chat = llm_chat


# ===== Core Helper Functions =====


def _process_tools(
    toolkit: Optional[ToolkitList], func_name: str
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Callable[..., Awaitable[Any]]]]:
    """
    Process and validate tool list, returning API-ready tool parameters and tool mapping.

    Wrapper around the process_tools utility function.

    Args:
        toolkit: List of Tool objects or async callable functions
        func_name: Function name for logging and error messages

    Returns:
        Tuple of (tool_param_for_api, tool_map):
            - tool_param_for_api: Tool definitions formatted for LLM API
            - tool_map: Mapping from tool names to callable implementations
    """
    return process_tools(toolkit, func_name)


def _extract_history_from_args(
    arguments: Dict[str, Any], func_name: str
) -> Optional[HistoryList]:
    """
    Extract and validate conversation history from function arguments.

    Looks for parameters named 'history' or 'chat_history' and validates
    that they conform to the expected format.

    Args:
        arguments: Dictionary of function arguments and their values
        func_name: Function name for logging and error messages

    Returns:
        Conversation history list if found and valid, None otherwise
    """
    # Find history parameter by name
    history_param_name: Optional[str] = None
    for param_name in HISTORY_PARAM_NAMES:
        if param_name in arguments:
            history_param_name = param_name
            break

    if not history_param_name:
        push_warning(
            f"LLM Chat '{func_name}' missing history parameter "
            f"(parameter name should be one of {HISTORY_PARAM_NAMES}). History will not be passed.",
            location=get_location(),
        )
        return None

    custom_history: Any = arguments[history_param_name]

    # Validate history format
    if not (
        isinstance(custom_history, list)
        and all(isinstance(item, dict) for item in custom_history)
    ):
        push_warning(
            f"LLM Chat '{func_name}' history parameter should be List[Dict[str, str]] type. "
            "History will not be passed.",
            location=get_location(),
        )
        return None

    return custom_history


def _build_user_message_content(
    arguments: Dict[str, Any], type_hints: Dict[str, Any], has_multimodal: bool
) -> Union[str, List[Dict[str, Any]]]:
    """
    Build user message content from function arguments.

    Creates either plain text message or multimodal content list depending
    on the presence of multimodal elements (images, etc.).

    Args:
        arguments: Dictionary of function arguments and their values
        type_hints: Type hints for the function parameters
        has_multimodal: Whether multimodal content is present in arguments

    Returns:
        User message content as string (text mode) or list of dicts (multimodal mode)
    """
    if has_multimodal:
        return build_multimodal_content(
            arguments, type_hints, exclude_params=HISTORY_PARAM_NAMES
        )
    else:
        # Build traditional text message, excluding history parameters
        message_parts: List[str] = []
        for param_name, param_value in arguments.items():
            if param_name not in HISTORY_PARAM_NAMES:
                message_parts.append(f"{param_name}: {param_value}")
        return "\n\t".join(message_parts)


def _build_messages(
    docstring: str,
    custom_history: Optional[HistoryList],
    user_message_content: Union[str, List[Dict[str, Any]]],
    tool_objects: Optional[List[Dict[str, Any]]],
    has_multimodal: bool,
) -> HistoryList:
    """
    Build complete message list for LLM API call.

    Constructs the full message list including system message (from docstring),
    conversation history, and the current user message.

    Args:
        docstring: Function docstring to be used as system prompt
        custom_history: Conversation history provided by the user
        user_message_content: Current user message content (text or multimodal)
        tool_objects: Available tools for the LLM to call
        has_multimodal: Whether the message contains multimodal content

    Returns:
        Complete message list ready for LLM API
    """
    messages: HistoryList = []

    # Step 1: Add system message from docstring
    if docstring:
        system_content: str = docstring
        if tool_objects:
            system_content = "\n\nYou can use the following tools flexibly according to the real case and tool description:\n\t" + "\n\t".join(
                (
                    f"- {tool['function']['name']}: {tool['function']['description']}"
                    for tool in tool_objects
                )
            ) + "\n\n" + system_content.strip()

        messages.append({"role": "system", "content": system_content})

    # Step 2: Add conversation history (excluding system messages)
    if custom_history:
        for msg in custom_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["role"] not in ["system"]:
                    messages.append(msg)
            else:
                push_warning(
                    f"Skipping malformed history item: {msg}",
                    location=get_location(),
                )

    # Step 3: Add current user message
    if user_message_content:
        user_msg: MessageDict = {"role": "user", "content": user_message_content}
        messages.append(user_msg)

    return messages
