import inspect
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
    cast,
    Literal,
)

from SimpleLLMFunc.llm_decorator.steps.common import (
    parse_function_signature,
    setup_log_context,
)
from SimpleLLMFunc.llm_decorator.steps.chat import (
    build_chat_messages,
    execute_react_loop_streaming,
    process_chat_response_stream,
)
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import push_debug
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.type.decorator import HistoryList
from SimpleLLMFunc.observability.langfuse_client import langfuse_client

# Type aliases
ToolkitList = List[Union[Tool, Callable[..., Awaitable[Any]]]]  # List of Tool objects or async functions

# Type variables
T = TypeVar("T")
P = ParamSpec("P")

# Constants
HISTORY_PARAM_NAMES: List[str] = ["history", "chat_history"]  # Valid parameter names for conversation history
DEFAULT_MAX_TOOL_CALLS: int = 5  # Default maximum number of tool calls to prevent infinite loops


def llm_chat(
    llm_interface: LLM_Interface,
    toolkit: Optional[ToolkitList] = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    stream: bool = False,
    return_mode: Literal["text", "raw"] = "text",
    **llm_kwargs: Any,
) -> Callable[
    [Union[Callable[P, Any], Callable[P, Awaitable[Any]]]],
    Callable[P, AsyncGenerator[Tuple[Any, HistoryList], None]]
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
        signature_meta = inspect.signature(func)
        docstring = func.__doc__ or ""
        func_name = func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Step 1: 解析函数签名
            function_signature, _ = parse_function_signature(func, args, kwargs)

            # Step 2: 设置日志上下文
            async with setup_log_context(
                func_name=function_signature.func_name,
                trace_id=function_signature.trace_id,
                arguments=function_signature.bound_args.arguments,
            ):
                # 创建 Langfuse parent span
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"{function_signature.func_name}_chat_call",
                    input=function_signature.bound_args.arguments,
                    metadata={
                        "function_name": function_signature.func_name,
                        "trace_id": function_signature.trace_id,
                        "tools_available": len(toolkit) if toolkit else 0,
                        "max_tool_calls": max_tool_calls,
                        "stream": stream,
                        "return_mode": return_mode,
                    },
                ) as chat_span:
                    try:
                        # Step 3: 构建聊天消息
                        messages = build_chat_messages(
                            signature=function_signature,
                            toolkit=toolkit,
                            exclude_params=HISTORY_PARAM_NAMES,
                        )

                        # Step 4: 执行 ReAct 循环（流式）
                        response_stream = execute_react_loop_streaming(
                            llm_interface=llm_interface,
                            messages=messages,
                            toolkit=toolkit,
                            max_tool_calls=max_tool_calls,
                            stream=stream,
                            llm_kwargs=llm_kwargs,
                            func_name=function_signature.func_name,
                        )

                        # Step 5: 处理响应流
                        collected_responses = []
                        final_history = None

                        async for content, history in process_chat_response_stream(
                            response_stream=response_stream,
                            return_mode=return_mode,
                            messages=messages,
                            func_name=function_signature.func_name,
                            stream=stream,
                        ):
                            collected_responses.append(content)
                            final_history = history
                            yield content, history

                        # 更新 Langfuse span
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
        wrapper.__signature__ = signature_meta  # type: ignore

        return cast(
            Callable[P, AsyncGenerator[Tuple[Any, HistoryList], None]],
            wrapper,
        )

    return decorator


async_llm_chat = llm_chat
