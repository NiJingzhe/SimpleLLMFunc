import inspect
import json
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
from SimpleLLMFunc.llm_decorator.steps.chat.message import HISTORY_PARAM_NAMES
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.self_reference import (
    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM,
    SelfReference,
)
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.type import HistoryList, MessageList
from SimpleLLMFunc.hooks.events import ReactEndEvent
from SimpleLLMFunc.hooks.stream import ReactOutput, is_event_yield
from SimpleLLMFunc.observability.langfuse_client import langfuse_client

# Type aliases
ToolkitList = List[Union[Tool, Callable[..., Awaitable[Any]]]]

# Type variables
T = TypeVar("T")
P = ParamSpec("P")

# Constants
DEFAULT_MAX_TOOL_CALLS: int = 5  # Default maximum number of tool calls
_SELF_REFERENCE_PROMPT_BLOCK_START = "[SelfReference Memory Contract]"
_SELF_REFERENCE_PROMPT_BLOCK_END = "[/SelfReference Memory Contract]"
_AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR = "__simplellmfunc_accepts_template_params__"


def _extract_raw_history_reference(arguments: dict[str, Any]) -> Optional[HistoryList]:
    """Extract the original history list object from bound call arguments."""

    for history_param_name in HISTORY_PARAM_NAMES:
        if history_param_name not in arguments:
            continue

        history = arguments[history_param_name]
        if history is None:
            return None

        if isinstance(history, list) and all(
            isinstance(item, dict) for item in history
        ):
            return cast(HistoryList, history)

        return None

    return None


def _set_history_argument(
    arguments: dict[str, Any],
    history: HistoryList,
) -> bool:
    """Inject a history snapshot into bound arguments when history param exists."""

    for history_param_name in HISTORY_PARAM_NAMES:
        if history_param_name in arguments:
            arguments[history_param_name] = history
            return True
    return False


def _resolve_self_reference_key(
    explicit_key: Optional[str],
    func_name: str,
) -> str:
    if explicit_key is None:
        return func_name

    normalized = explicit_key.strip()
    if not normalized:
        raise ValueError("self_reference_key must be a non-empty string")
    return normalized


def _build_self_reference_prompt_block(memory_key: str) -> str:
    return "\n".join(
        [
            _SELF_REFERENCE_PROMPT_BLOCK_START,
            "Self-reference memory is enabled for this agent.",
            f'- Use memory handle: self_reference.memory["{memory_key}"]',
            "- Method purposes:",
            "  count(): number of messages currently stored.",
            "  all(): deep-copy snapshot of all messages.",
            "  get(index): read one message by index.",
            "  append(message): add one message at the end.",
            "  insert(index, message): insert one message at index.",
            "  update(index, message): replace one message at index.",
            "  delete(index): remove one message at index.",
            "  replace(messages): replace entire history with validated messages.",
            "  clear(): remove all messages for this key.",
            "  get_system_prompt(): read latest system prompt text.",
            "  set_system_prompt(text): overwrite system prompt text.",
            (
                "  append_system_prompt(text): append text to current system "
                "prompt with a newline."
            ),
            "- Self instance methods:",
            "  self_reference.instance.is_bound(): check if recursive fork is available.",
            (
                "  self_reference.instance.fork(message, "
                f'source_memory_key="{memory_key}"): '
                "fork this agent with inherited memory snapshot."
            ),
            "- Forgetting memory:",
            "  reset_repl only clears Python variables in REPL.",
            "  It does NOT delete conversation memory in self_reference.",
            "  To forget memory, delete message records via memory methods:",
            "  delete(index), replace(messages), or clear().",
            "- Durable preference example:",
            (f'  self_reference.memory["{memory_key}"].append_system_prompt("...")'),
            _SELF_REFERENCE_PROMPT_BLOCK_END,
        ]
    )


def _remove_self_reference_prompt_block(system_prompt: str) -> str:
    cleaned_prompt = system_prompt

    while True:
        start_index = cleaned_prompt.find(_SELF_REFERENCE_PROMPT_BLOCK_START)
        if start_index < 0:
            break

        end_index = cleaned_prompt.find(_SELF_REFERENCE_PROMPT_BLOCK_END, start_index)
        if end_index < 0:
            cleaned_prompt = cleaned_prompt[:start_index]
            break

        cleaned_prompt = (
            cleaned_prompt[:start_index]
            + cleaned_prompt[end_index + len(_SELF_REFERENCE_PROMPT_BLOCK_END) :]
        )

    return cleaned_prompt.strip()


def _append_self_reference_prompt_to_messages(
    messages: MessageList,
    memory_key: str,
) -> None:
    prompt_block = _build_self_reference_prompt_block(memory_key)

    for index, message in enumerate(messages):
        if message.get("role") != "system":
            continue

        content = message.get("content")
        base_prompt = ""
        if isinstance(content, str):
            base_prompt = _remove_self_reference_prompt_block(content)

        if base_prompt:
            merged_prompt = f"{base_prompt}\n\n{prompt_block}"
        else:
            merged_prompt = prompt_block

        messages[index] = {**message, "content": merged_prompt}
        return

    messages.insert(0, {"role": "system", "content": prompt_block})


def _extract_first_system_prompt_from_messages(messages: MessageList) -> Optional[str]:
    for message in messages:
        if message.get("role") != "system":
            continue

        content = message.get("content")
        if isinstance(content, str):
            return content

    return None


def _seed_self_reference_system_prompt_if_missing(
    self_reference: SelfReference,
    memory_key: str,
    messages: MessageList,
) -> None:
    if self_reference.get_system_prompt(memory_key) is not None:
        return

    system_prompt = _extract_first_system_prompt_from_messages(messages)
    if system_prompt is None:
        return

    # Keep memory store focused on durable prompt content. If the system prompt
    # already contains the auto-injected SelfReference contract block, store a
    # cleaned version and let llm_chat append the contract again per turn.
    cleaned_system_prompt = _remove_self_reference_prompt_block(system_prompt)
    system_prompt_for_store = cleaned_system_prompt or system_prompt

    current_history = self_reference.snapshot_history(memory_key)
    seeded_history: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt_for_store},
        *cast(List[Dict[str, Any]], current_history),
    ]
    self_reference.replace_history(
        key=memory_key,
        messages=seeded_history,
        strict=False,
    )


def llm_chat(
    llm_interface: LLM_Interface,
    toolkit: Optional[ToolkitList] = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    stream: bool = False,
    return_mode: Literal["text", "raw"] = "text",
    enable_event: bool = False,
    self_reference: Optional[SelfReference] = None,
    self_reference_key: Optional[str] = None,
    **llm_kwargs: Any,
) -> Callable[
    [Union[Callable[P, Any], Callable[P, Awaitable[Any]]]],
    Callable[P, AsyncGenerator[Union[Tuple[Any, HistoryList], ReactOutput], None]],
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
        enable_event: Whether to enable event stream (default: False)
            - False: yields (response, messages) tuples (backward compatible)
            - True: yields ReactOutput (ResponseYield or EventYield)
        self_reference: Optional SelfReference shared object for agent self-referential
            memory operations. When provided, llm_chat appends a
            SelfReference memory contract block to the system prompt.
        self_reference_key: Memory key used with self_reference for this decorator.
            Defaults to function name when self_reference is provided.
        **llm_kwargs: Additional keyword arguments passed directly to the LLM interface

    Returns:
        Decorated async generator function that yields (response_content, updated_history) tuples
        or ReactOutput when enable_event=True

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
    ) -> Callable[P, AsyncGenerator[Union[Tuple[Any, HistoryList], ReactOutput], None]]:
        signature_meta = inspect.signature(func)
        docstring = func.__doc__ or ""
        func_name = func.__name__

        resolved_default_self_reference_key: Optional[str] = None
        if self_reference is not None:
            resolved_default_self_reference_key = _resolve_self_reference_key(
                self_reference_key,
                func_name,
            )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Step 1: 解析函数签名
            function_signature, template_params = parse_function_signature(
                func,
                args,
                kwargs,
            )

            # 构建用户任务提示（用于事件）
            user_task_prompt = json.dumps(
                function_signature.bound_args.arguments,
                default=str,
                ensure_ascii=False,
            )

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
                        "enable_event": enable_event,
                        "self_reference_enabled": self_reference is not None,
                        "self_reference_key": self_reference_key,
                    },
                ) as chat_span:
                    try:
                        raw_history_reference = _extract_raw_history_reference(
                            function_signature.bound_args.arguments
                        )

                        resolved_self_reference_key: Optional[str] = None
                        baseline_history_count = 0

                        if self_reference is not None:
                            runtime_self_reference_key = self_reference_key
                            if template_params is not None:
                                override_key = template_params.get(
                                    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM
                                )
                                if override_key is not None:
                                    if not isinstance(override_key, str):
                                        raise ValueError(
                                            "self_reference key override must be a non-empty string"
                                        )
                                    runtime_self_reference_key = override_key

                            resolved_self_reference_key = _resolve_self_reference_key(
                                runtime_self_reference_key,
                                function_signature.func_name,
                            )

                            if raw_history_reference is not None:
                                self_reference.bind_history(
                                    resolved_self_reference_key,
                                    cast(List[Dict[str, Any]], raw_history_reference),
                                )
                            elif not self_reference.has_history(
                                resolved_self_reference_key
                            ):
                                self_reference.bind_history(
                                    resolved_self_reference_key,
                                    [],
                                )

                            history_snapshot = self_reference.snapshot_history(
                                resolved_self_reference_key
                            )
                            _set_history_argument(
                                function_signature.bound_args.arguments,
                                history_snapshot,
                            )
                            baseline_history_count = (
                                self_reference.filtered_history_count(
                                    resolved_self_reference_key
                                )
                            )

                        # Step 3: 构建聊天消息
                        messages = build_chat_messages(
                            signature=function_signature,
                            toolkit=toolkit,
                            exclude_params=HISTORY_PARAM_NAMES,
                        )

                        if (
                            self_reference is not None
                            and resolved_self_reference_key is not None
                        ):
                            _append_self_reference_prompt_to_messages(
                                messages,
                                resolved_self_reference_key,
                            )
                            _seed_self_reference_system_prompt_if_missing(
                                self_reference,
                                resolved_self_reference_key,
                                messages,
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
                            enable_event=enable_event,
                            trace_id=function_signature.trace_id,
                            user_task_prompt=user_task_prompt,
                        )

                        collected_responses = []
                        final_history = None

                        if enable_event:
                            # 事件模式：直接 yield ReactOutput
                            typed_event_stream = cast(
                                AsyncGenerator[ReactOutput, None],
                                response_stream,
                            )
                            async for output in typed_event_stream:
                                if (
                                    self_reference is not None
                                    and resolved_self_reference_key is not None
                                    and is_event_yield(output)
                                    and isinstance(output.event, ReactEndEvent)
                                ):
                                    merged_history = self_reference.merge_turn_history(
                                        key=resolved_self_reference_key,
                                        baseline_history_count=baseline_history_count,
                                        updated_history=cast(
                                            List[Dict[str, Any]],
                                            output.event.final_messages,
                                        ),
                                        commit=True,
                                    )
                                    output.event.final_messages = merged_history
                                    if raw_history_reference is not None:
                                        raw_history_reference[:] = merged_history

                                yield output
                        else:
                            # 向后兼容模式：处理响应流
                            # 类型断言：当 enable_event=False 时，response_stream 只包含 Tuple[Any, MessageList]
                            typed_response_stream = cast(
                                AsyncGenerator[Tuple[Any, MessageList], None],
                                response_stream,
                            )
                            latest_merged_history: Optional[HistoryList] = None
                            async for content, history in process_chat_response_stream(
                                response_stream=typed_response_stream,
                                return_mode=return_mode,
                                messages=messages,
                                func_name=function_signature.func_name,
                                stream=stream,
                            ):
                                history_to_yield: MessageList = history
                                if (
                                    self_reference is not None
                                    and resolved_self_reference_key is not None
                                ):
                                    merged_history = self_reference.merge_turn_history(
                                        key=resolved_self_reference_key,
                                        baseline_history_count=baseline_history_count,
                                        updated_history=cast(
                                            List[Dict[str, Any]],
                                            history,
                                        ),
                                        commit=False,
                                    )
                                    history_to_yield = merged_history
                                    latest_merged_history = merged_history

                                collected_responses.append(content)
                                final_history = history_to_yield
                                yield content, history_to_yield

                            if (
                                self_reference is not None
                                and resolved_self_reference_key is not None
                                and latest_merged_history is not None
                            ):
                                self_reference.replace_history(
                                    key=resolved_self_reference_key,
                                    messages=cast(
                                        List[Dict[str, Any]],
                                        latest_merged_history,
                                    ),
                                    strict=False,
                                )
                                if raw_history_reference is not None:
                                    raw_history_reference[:] = latest_merged_history

                        # 更新 Langfuse span（仅在非事件模式或收集到响应时）
                        if not enable_event or collected_responses:
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
        setattr(wrapper, _AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR, True)

        if self_reference is not None:
            self_reference.bind_agent_instance(
                wrapper,
                default_memory_key=resolved_default_self_reference_key,
            )

        return cast(
            Callable[
                P, AsyncGenerator[Union[Tuple[Any, HistoryList], ReactOutput], None]
            ],
            wrapper,
        )

    return decorator


async_llm_chat = llm_chat
