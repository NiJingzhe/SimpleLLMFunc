"""Tests for tool call types."""

from __future__ import annotations

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as ChatCompletionMessageToolCall,
    Function as OpenAIFunction,
)

from SimpleLLMFunc.type.tool_call import (
    AccumulatedToolCall,
    ToolCall,
    ToolCallArguments,
    ToolCallFunction,
    ToolCallFunctionInfo,
    ToolDefinition,
    ToolDefinitionList,
    ToolFunctionDefinition,
    dict_to_tool_call,
    tool_call_to_dict,
)


class TestToolCallTypes:
    """Test tool call type aliases."""

    def test_tool_call_type_alias(self):
        """Test that ToolCall is an alias for ChatCompletionMessageToolCall."""
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=OpenAIFunction(
                name="test_tool",
                arguments='{"arg1": "value1"}',
            ),
        )
        # ToolCall should be the same type
        assert isinstance(tool_call, ChatCompletionMessageToolCall)
        # Type check: ToolCall should accept ChatCompletionMessageToolCall
        tool_call_typed: ToolCall = tool_call
        assert tool_call_typed.id == "call_123"
        assert tool_call_typed.function.name == "test_tool"

    def test_tool_call_function_type_alias(self):
        """Test that ToolCallFunction is an alias for OpenAIFunction."""
        func = OpenAIFunction(
            name="test_tool",
            arguments='{"arg1": "value1"}',
        )
        # Type check: ToolCallFunction should accept OpenAIFunction
        func_typed: ToolCallFunction = func
        assert func_typed.name == "test_tool"
        assert func_typed.arguments == '{"arg1": "value1"}'

    def test_tool_call_arguments_type(self):
        """Test ToolCallArguments type."""
        args: ToolCallArguments = {"arg1": "value1", "arg2": 123}
        assert args["arg1"] == "value1"
        assert args["arg2"] == 123


class TestToolDefinitionTypes:
    """Test tool definition types."""

    def test_tool_function_definition(self):
        """Test ToolFunctionDefinition TypedDict."""
        func_def: ToolFunctionDefinition = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"},
                },
            },
        }
        assert func_def["name"] == "test_tool"
        assert func_def["description"] == "A test tool"

    def test_tool_definition(self):
        """Test ToolDefinition TypedDict."""
        tool_def: ToolDefinition = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string"},
                    },
                },
            },
        }
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "test_tool"

    def test_tool_definition_list(self):
        """Test ToolDefinitionList type."""
        # None case
        tools_none: ToolDefinitionList = None
        assert tools_none is None

        # List case
        tools_list: ToolDefinitionList = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {},
                },
            },
        ]
        assert len(tools_list) == 1
        assert tools_list[0]["function"]["name"] == "test_tool"


class TestInternalTypes:
    """Test internal types for streaming accumulation."""

    def test_tool_call_function_info(self):
        """Test ToolCallFunctionInfo TypedDict."""
        func_info: ToolCallFunctionInfo = {
            "name": "test_tool",
            "arguments": '{"arg1": "value1"}',
        }
        assert func_info["name"] == "test_tool"
        assert func_info["arguments"] == '{"arg1": "value1"}'

        # Optional name case
        func_info_optional: ToolCallFunctionInfo = {
            "name": None,
            "arguments": "",
        }
        assert func_info_optional["name"] is None

    def test_accumulated_tool_call(self):
        """Test AccumulatedToolCall TypedDict."""
        accumulated: AccumulatedToolCall = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": '{"arg1": "value1"}',
            },
        }
        assert accumulated["id"] == "call_123"
        assert accumulated["type"] == "function"
        assert accumulated["function"]["name"] == "test_tool"

        # Optional fields case
        accumulated_optional: AccumulatedToolCall = {
            "id": None,
            "type": None,
            "function": {
                "name": None,
                "arguments": "",
            },
        }
        assert accumulated_optional["id"] is None


class TestToolCallConversion:
    """Test tool call conversion functions."""

    def test_dict_to_tool_call(self):
        """Test converting dict to ToolCall."""
        data = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": '{"arg1": "value1"}',
            },
        }
        tool_call = dict_to_tool_call(data)
        assert isinstance(tool_call, ChatCompletionMessageToolCall)
        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "test_tool"
        assert tool_call.function.arguments == '{"arg1": "value1"}'

    def test_dict_to_tool_call_default_type(self):
        """Test dict_to_tool_call with default type."""
        data = {
            "id": "call_123",
            "function": {
                "name": "test_tool",
                "arguments": '{"arg1": "value1"}',
            },
        }
        tool_call = dict_to_tool_call(data)
        assert tool_call.type == "function"  # Default value

    def test_tool_call_to_dict(self):
        """Test converting ToolCall to dict."""
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=OpenAIFunction(
                name="test_tool",
                arguments='{"arg1": "value1"}',
            ),
        )
        data = tool_call_to_dict(tool_call)
        assert data["id"] == "call_123"
        assert data["type"] == "function"
        assert data["function"]["name"] == "test_tool"
        assert data["function"]["arguments"] == '{"arg1": "value1"}'

    def test_round_trip_conversion(self):
        """Test round-trip conversion: dict -> ToolCall -> dict."""
        original_data = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": '{"arg1": "value1"}',
            },
        }
        tool_call = dict_to_tool_call(original_data)
        converted_data = tool_call_to_dict(tool_call)
        assert converted_data == original_data
