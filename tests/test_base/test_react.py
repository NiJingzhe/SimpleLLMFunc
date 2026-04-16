"""Tests for base.ReAct module."""

from __future__ import annotations

from typing import Any
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.base.ReAct import execute_llm, execute_single_llm_call
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.hooks.events import (
    LLMCallEndEvent,
    ReactEndEvent,
    ReActEventType,
)
from SimpleLLMFunc.hooks.stream import EventYield


class TestExecuteLLM:
    """Tests for execute_llm function."""

    def _make_chunk(self, content: str) -> ChatCompletionChunk:
        delta = ChoiceDelta(content=content, role="assistant")
        choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
        return ChatCompletionChunk(
            id="chunk-id",
            choices=[choice],
            created=123,
            model="test-model",
            object="chat.completion.chunk",
        )

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_non_streaming_no_tools(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
    ) -> None:
        """Test executing non-streaming LLM call without tools."""
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        responses = []
        async for response, _ in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
        ):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0] == mock_chat_completion
        mock_llm_interface.chat.assert_called_once()

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_non_streaming_with_tools_no_calls(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
    ) -> None:
        """Test executing non-streaming LLM call with tools but no tool calls."""
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        responses = []
        async for response, _ in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map={},
            max_tool_calls=5,
            stream=False,
        ):
            responses.append(response)

        assert len(responses) == 1

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    @patch("SimpleLLMFunc.base.ReAct.process_tool_calls")
    async def test_execute_with_tool_calls(
        self,
        mock_process_tools: AsyncMock,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion_with_tool_calls: Any,
    ) -> None:
        """Test executing LLM call with tool calls."""
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(
            return_value=mock_chat_completion_with_tool_calls
        )

        # Mock tool processing to return updated messages
        updated_messages = sample_messages + [
            {"role": "tool", "tool_call_id": "call_123", "content": "result"}
        ]
        mock_process_tools.return_value = updated_messages

        # Mock second call to return final response without tool calls
        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message = MagicMock()
        final_response.choices[0].message.content = "Final response"
        final_response.choices[0].message.tool_calls = None

        mock_llm_interface.chat.side_effect = [
            mock_chat_completion_with_tool_calls,
            final_response,
        ]

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        tool_map = {"test_tool": AsyncMock(return_value="result")}

        responses = []
        async for response, _ in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=5,
            stream=False,
        ):
            responses.append(response)

        assert len(responses) >= 1
        mock_process_tools.assert_called()

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_streaming_no_tools(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion_chunk: Any,
    ) -> None:
        """Test executing streaming LLM call without tools."""
        mock_get_context.return_value = "test_func"

        async def stream_generator(**kwargs):
            yield mock_chat_completion_chunk

        mock_llm_interface.chat_stream = stream_generator
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        responses = []
        async for response, _ in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=True,
        ):
            responses.append(response)

        assert len(responses) >= 1

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_streaming_abort_appends_partial_history(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        """Abort should close stream and persist partial assistant content."""
        mock_get_context.return_value = "test_func"
        abort_signal = AbortSignal()

        async def stream_generator(**kwargs):
            yield self._make_chunk("Hello ")
            abort_signal.abort("user_interrupt")
            await asyncio.sleep(0)
            yield self._make_chunk("world")

        mock_llm_interface.chat_stream = stream_generator
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        outputs = []
        async for output in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=True,
            enable_event=True,
            abort_signal=abort_signal,
        ):
            outputs.append(output)

        end_events = [
            item.event
            for item in outputs
            if isinstance(item, EventYield) and isinstance(item.event, ReactEndEvent)
        ]
        assert end_events
        final_messages = end_events[-1].final_messages
        assert final_messages[-1]["role"] == "assistant"
        assert "Hello" in final_messages[-1]["content"]
        assert "world" not in final_messages[-1]["content"]
        assert end_events[-1].extra.get("aborted") is True

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_streaming_emits_tool_argument_delta_event(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
    ) -> None:
        """Streaming tool argument chunks should emit argument delta events."""

        mock_get_context.return_value = "test_func"

        def _chunk(arguments_delta: str, *, include_id: bool, include_name: bool):
            tool_call = ChoiceDeltaToolCall(
                index=0,
                id="call_123" if include_id else None,
                type="function" if include_id else None,
                function=ChoiceDeltaToolCallFunction(
                    name="execute_code" if include_name else None,
                    arguments=arguments_delta,
                ),
            )
            delta = ChoiceDelta(
                content=None,
                role="assistant",
                tool_calls=[tool_call],
            )
            choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
            return ChatCompletionChunk(
                id="chunk-id",
                choices=[choice],
                created=123,
                model="test-model",
                object="chat.completion.chunk",
            )

        async def stream_generator(**kwargs):
            yield _chunk('{{"code":"print(', include_id=True, include_name=True)
            yield _chunk('1)"}', include_id=False, include_name=False)

        mock_llm_interface.chat_stream = stream_generator
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        tool_map = {"execute_code": AsyncMock(return_value="ok")}
        tools = [{"type": "function", "function": {"name": "execute_code"}}]

        delta_events = []
        async for output in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=1,
            stream=True,
            enable_event=True,
        ):
            if (
                isinstance(output, EventYield)
                and output.event.event_type == ReActEventType.TOOL_CALL_ARGUMENTS_DELTA
            ):
                delta_events.append(output.event)

        assert delta_events
        merged_delta = "".join(
            event.argcontent_delta for event in delta_events if event.argname == "code"
        )
        assert merged_delta.endswith("print(1)")
        assert all(event.tool_call_id == "call_123" for event in delta_events)

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_llm_hook_order_with_tool_iterations(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
        mock_chat_completion_with_tool_calls: Any,
    ) -> None:
        """Hook order should stay stable across initial call, tool batch, iteration, and finalize."""

        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(
            side_effect=[
                mock_chat_completion_with_tool_calls,
                mock_chat_completion,
            ]
        )

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        tool_map = {"test_tool": AsyncMock(return_value="result")}
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        calls: list[tuple[str, int, int]] = []

        class Hooks:
            async def on_run_start(self, state: Any) -> None:
                calls.append(("on_run_start", state.iteration, len(state.messages)))

            async def before_llm_call(self, state: Any) -> None:
                calls.append(("before_llm_call", state.iteration, len(state.messages)))

            async def after_llm_call(self, state: Any) -> None:
                calls.append(("after_llm_call", state.iteration, len(state.messages)))

            async def before_tool_batch(self, state: Any) -> None:
                calls.append(
                    ("before_tool_batch", state.iteration, len(state.messages))
                )

            async def after_tool_batch(self, state: Any) -> None:
                calls.append(("after_tool_batch", state.iteration, len(state.messages)))

            async def before_finalize(self, state: Any) -> None:
                calls.append(("before_finalize", state.iteration, len(state.messages)))

        async for _response, _messages in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=5,
            stream=False,
            hooks=Hooks(),
        ):
            pass

        assert [name for name, _, _ in calls] == [
            "on_run_start",
            "before_llm_call",
            "after_llm_call",
            "before_tool_batch",
            "after_tool_batch",
            "before_llm_call",
            "after_llm_call",
            "before_finalize",
        ]

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    @patch("SimpleLLMFunc.base.ReAct.process_tool_calls")
    async def test_execute_max_tool_calls_reached(
        self,
        mock_process_tools: AsyncMock,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion_with_tool_calls: Any,
    ) -> None:
        """Test executing when max_tool_calls is reached."""
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(
            return_value=mock_chat_completion_with_tool_calls
        )

        updated_messages = sample_messages + [
            {"role": "tool", "tool_call_id": "call_123", "content": "result"}
        ]
        mock_process_tools.return_value = updated_messages

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        tool_map = {"test_tool": AsyncMock(return_value="result")}

        responses = []
        async for response, _ in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=1,  # Low limit to trigger max_tool_calls
            stream=False,
        ):
            responses.append(response)

        # Should have responses including final call
        assert len(responses) >= 1

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    @patch("SimpleLLMFunc.base.ReAct.process_tool_calls")
    async def test_execute_max_tool_calls_none_keeps_iterating_until_no_tools(
        self,
        mock_process_tools: AsyncMock,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
        mock_chat_completion_with_tool_calls: Any,
    ) -> None:
        """None max_tool_calls should not force a final cap-based LLM call."""

        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(
            side_effect=[
                mock_chat_completion_with_tool_calls,
                mock_chat_completion_with_tool_calls,
                mock_chat_completion,
            ]
        )

        updated_messages_first = sample_messages + [
            {"role": "tool", "tool_call_id": "call_123", "content": "result-1"}
        ]
        updated_messages_second = updated_messages_first + [
            {"role": "tool", "tool_call_id": "call_123", "content": "result-2"}
        ]
        mock_process_tools.side_effect = [
            updated_messages_first,
            updated_messages_second,
        ]

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        tool_map = {"test_tool": AsyncMock(return_value="result")}

        responses = []
        async for response, _ in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=None,
            stream=False,
        ):
            responses.append(response)

        assert responses == [
            mock_chat_completion_with_tool_calls,
            mock_chat_completion_with_tool_calls,
            mock_chat_completion,
        ]
        assert mock_llm_interface.chat.await_count == 3
        assert mock_process_tools.await_count == 2

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_empty_messages(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        mock_chat_completion: Any,
    ) -> None:
        """Test executing with empty messages."""
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        responses = []
        async for response, _ in execute_llm(
            llm_interface=mock_llm_interface,
            messages=[],
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
        ):
            responses.append(response)

        assert len(responses) == 1

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct._usage_from_context_delta")
    @patch("SimpleLLMFunc.base.ReAct.extract_usage_from_response", return_value=None)
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_streaming_usage_fallback_without_tools(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        _mock_extract_usage: MagicMock,
        mock_usage_from_context_delta: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion_chunk: Any,
    ) -> None:
        """Streaming LLMCallEndEvent should fallback to context delta usage."""

        mock_get_context.return_value = "test_func"

        async def stream_generator(**kwargs):
            yield mock_chat_completion_chunk

        mock_llm_interface.chat_stream = stream_generator
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        mock_usage_from_context_delta.return_value = CompletionUsage(
            prompt_tokens=12,
            completion_tokens=5,
            total_tokens=17,
        )

        llm_end_usage = None
        async for output in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=True,
            enable_event=True,
        ):
            if isinstance(output, EventYield) and isinstance(
                output.event, LLMCallEndEvent
            ):
                llm_end_usage = output.event.usage

        assert llm_end_usage is not None
        assert llm_end_usage.total_tokens == 17
        assert mock_usage_from_context_delta.called

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_streaming_react_end_uses_accumulated_content_when_tail_chunk_has_no_choices(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion_chunk: Any,
    ) -> None:
        """ReactEndEvent should keep accumulated stream content from prior chunks."""

        mock_get_context.return_value = "test_func"

        usage_tail_chunk = ChatCompletionChunk(
            id="test-id-usage",
            choices=[],
            created=1234567891,
            model="test-model",
            object="chat.completion.chunk",
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
        )

        async def stream_generator(**kwargs):
            yield mock_chat_completion_chunk
            yield usage_tail_chunk

        mock_llm_interface.chat_stream = stream_generator
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        final_response = None
        async for output in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=True,
            enable_event=True,
        ):
            if isinstance(output, EventYield) and isinstance(
                output.event, ReactEndEvent
            ):
                final_response = output.event.final_response

        assert final_response == "chunk"

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_event_mode_attaches_origin_metadata(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
    ) -> None:
        """Event mode should provide monotonic origin metadata."""
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        origins = []
        async for output in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            enable_event=True,
        ):
            if isinstance(output, EventYield):
                origins.append(output.origin)

        assert origins
        session_ids = {origin.session_id for origin in origins}
        assert len(session_ids) == 1
        assert "" not in session_ids
        event_seqs = [origin.event_seq for origin in origins]
        assert event_seqs == sorted(event_seqs)
        assert event_seqs[0] == 1

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_execute_event_mode_tool_phase_uses_event_bus_origin(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion_with_tool_calls: Any,
    ) -> None:
        """Tool-phase events should carry monotonic origin and tool context."""
        mock_get_context.return_value = "test_func"

        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message = MagicMock()
        final_response.choices[0].message.content = "Final response"
        final_response.choices[0].message.tool_calls = None

        mock_llm_interface.chat = AsyncMock(
            side_effect=[mock_chat_completion_with_tool_calls, final_response]
        )

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        async def _tool_impl(arg1: str, event_emitter: Any = None) -> str:
            return "tool-result"

        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        tool_map = {"test_tool": _tool_impl}

        outputs: list[EventYield] = []
        async for output in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=5,
            stream=False,
            enable_event=True,
        ):
            if isinstance(output, EventYield):
                outputs.append(output)

        assert outputs
        session_ids = {output.origin.session_id for output in outputs}
        assert len(session_ids) == 1
        assert "legacy" not in session_ids

        event_seqs = [output.origin.event_seq for output in outputs]
        assert event_seqs == sorted(event_seqs)

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_before_llm_call_hook_can_mutate_active_messages(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
    ) -> None:
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        class _Hooks:
            async def before_llm_call(self, state: Any) -> None:
                state.messages = [
                    *state.messages,
                    {"role": "user", "content": "[hook] extra context"},
                ]

        async for _response, _messages in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            hooks=_Hooks(),
        ):
            pass

        all_called_messages = [
            call.kwargs["messages"] for call in mock_llm_interface.chat.await_args_list
        ]
        assert any(
            {"role": "user", "content": "[hook] extra context"} in message_list
            for message_list in all_called_messages
        )

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_before_finalize_hook_can_override_final_messages_and_response(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
    ) -> None:
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        class _Hooks:
            async def before_finalize(self, state: Any) -> None:
                state.messages = [
                    {"role": "system", "content": "compacted"},
                    {"role": "assistant", "content": "summary"},
                ]
                state.final_response = "compacted-response"

        react_end = None
        async for output in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            enable_event=True,
            hooks=_Hooks(),
        ):
            if isinstance(output, EventYield) and isinstance(
                output.event, ReactEndEvent
            ):
                react_end = output.event

        assert react_end is not None
        assert react_end.final_response == "compacted-response"
        assert react_end.final_messages == [
            {"role": "system", "content": "compacted"},
            {"role": "assistant", "content": "summary"},
        ]

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_before_finalize_hook_runs_for_non_event_terminal_response(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
    ) -> None:
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        finalized_messages = [
            {"role": "system", "content": "compacted"},
            {"role": "assistant", "content": "summary"},
        ]
        finalize_calls: list[tuple[int, int]] = []
        yielded_messages: list[list[dict[str, Any]]] = []

        class _Hooks:
            async def before_finalize(self, state: Any) -> None:
                finalize_calls.append((state.iteration, len(state.messages)))
                state.messages = list(finalized_messages)

        async for _response, messages in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            hooks=_Hooks(),
        ):
            yielded_messages.append(messages)

        assert finalize_calls == [(0, 3)]
        assert yielded_messages
        assert yielded_messages[-1] == finalized_messages

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    @patch("SimpleLLMFunc.base.ReAct.process_tool_calls")
    async def test_before_finalize_hook_runs_for_non_event_after_tool_iterations(
        self,
        mock_process_tools: AsyncMock,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
        mock_chat_completion_with_tool_calls: Any,
    ) -> None:
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(
            side_effect=[mock_chat_completion_with_tool_calls, mock_chat_completion]
        )

        mock_process_tools.return_value = sample_messages + [
            {"role": "tool", "tool_call_id": "call_123", "content": "result"}
        ]

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        finalized_messages = [
            {"role": "system", "content": "post-tools compacted"},
            {"role": "assistant", "content": "summary"},
        ]
        finalize_calls: list[tuple[int, int]] = []
        yielded_messages: list[list[dict[str, Any]]] = []
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        tool_map = {"test_tool": AsyncMock(return_value="result")}

        class _Hooks:
            async def before_finalize(self, state: Any) -> None:
                finalize_calls.append((state.iteration, len(state.messages)))
                state.messages = list(finalized_messages)

        async for _response, messages in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=5,
            stream=False,
            hooks=_Hooks(),
        ):
            yielded_messages.append(messages)

        assert finalize_calls == [(2, 4)]
        assert yielded_messages
        assert yielded_messages[-1] == finalized_messages

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_before_finalize_hook_runs_for_abort_after_tool_batch(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion_with_tool_calls: Any,
    ) -> None:
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(
            return_value=mock_chat_completion_with_tool_calls
        )

        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_observation.update = MagicMock()
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        abort_signal = AbortSignal()
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        react_end = None
        finalize_calls: list[tuple[int, int]] = []

        async def _tool_impl(arg1: str, event_emitter: Any = None) -> str:
            _ = event_emitter
            abort_signal.abort("stop-after-tool")
            return "tool-result"

        class _Hooks:
            async def before_finalize(self, state: Any) -> None:
                finalize_calls.append((state.iteration, len(state.messages)))
                state.messages = [
                    {"role": "system", "content": "aborted compacted"},
                    {"role": "assistant", "content": "summary"},
                ]
                state.final_response = "aborted-response"

        async for output in execute_llm(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map={"test_tool": _tool_impl},
            max_tool_calls=5,
            stream=False,
            enable_event=True,
            abort_signal=abort_signal,
            hooks=_Hooks(),
        ):
            if isinstance(output, EventYield) and isinstance(
                output.event, ReactEndEvent
            ):
                react_end = output.event

        assert finalize_calls == [(1, 3)]
        assert react_end is not None
        assert react_end.final_response == "aborted-response"
        assert react_end.final_messages == [
            {"role": "system", "content": "aborted compacted"},
            {"role": "assistant", "content": "summary"},
        ]


class TestExecuteSingleLLMCall:
    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_non_streaming_returns_final_llm_call_end_event(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
    ) -> None:
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        events = []
        async for event in execute_single_llm_call(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            stream=False,
        ):
            events.append(event)

        assert events
        assert isinstance(events[-1], LLMCallEndEvent)
        assert events[-1].content == "Test response"
        assert events[-1].tool_calls == []

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.ReAct.langfuse_client")
    @patch("SimpleLLMFunc.base.ReAct.get_current_context_attribute")
    async def test_streaming_accumulates_content_and_tool_calls(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_get_context.return_value = "test_func"

        def _chunk(
            arguments_delta: str,
            *,
            include_id: bool,
            include_name: bool,
            content: str | None = None,
        ):
            tool_call = ChoiceDeltaToolCall(
                index=0,
                id="call_123" if include_id else None,
                type="function" if include_id else None,
                function=ChoiceDeltaToolCallFunction(
                    name="execute_code" if include_name else None,
                    arguments=arguments_delta,
                ),
            )
            delta = ChoiceDelta(
                content=content,
                role="assistant",
                tool_calls=[tool_call],
            )
            choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
            return ChatCompletionChunk(
                id="chunk-id",
                choices=[choice],
                created=123,
                model="test-model",
                object="chat.completion.chunk",
            )

        async def stream_generator(**kwargs):
            _ = kwargs
            yield _chunk(
                '{{"code":"print(', include_id=True, include_name=True, content="hel"
            )
            yield _chunk('1)"}', include_id=False, include_name=False, content="lo")

        mock_llm_interface.chat_stream = stream_generator
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        events = []
        async for event in execute_single_llm_call(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=[{"type": "function", "function": {"name": "execute_code"}}],
            stream=True,
        ):
            events.append(event)

        assert events
        assert isinstance(events[-1], LLMCallEndEvent)
        assert events[-1].content == "hello"
        assert len(events[-1].tool_calls) == 1
        assert events[-1].tool_calls[0].id == "call_123"
        assert events[-1].tool_calls[0].function.name == "execute_code"
