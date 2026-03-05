"""Tests for base.ReAct module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.base.ReAct import execute_llm
from SimpleLLMFunc.hooks.events import LLMCallEndEvent, ReactEndEvent
from SimpleLLMFunc.hooks.stream import EventYield


class TestExecuteLLM:
    """Tests for execute_llm function."""

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
