"""Tests for base.ReAct module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from SimpleLLMFunc.base.ReAct import execute_llm


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
        async for response in execute_llm(
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
        async for response in execute_llm(
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
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion_with_tool_calls)
        
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
        async for response in execute_llm(
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
        async for response in execute_llm(
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
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion_with_tool_calls)
        
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
        async for response in execute_llm(
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
        async for response in execute_llm(
            llm_interface=mock_llm_interface,
            messages=[],
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
        ):
            responses.append(response)

        assert len(responses) == 1

