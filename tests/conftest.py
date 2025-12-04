"""Shared pytest fixtures for SimpleLLMFunc tests."""

from __future__ import annotations

import inspect
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import BaseModel

from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text
from SimpleLLMFunc.tool import Tool


@pytest.fixture
def mock_llm_interface() -> LLM_Interface:
    """Mock LLM interface for testing."""
    mock = MagicMock(spec=LLM_Interface)
    mock.model_name = "test-model"
    mock.input_token_count = 0
    mock.output_token_count = 0
    return mock


@pytest.fixture
def mock_api_key_pool() -> APIKeyPool:
    """Mock API key pool for testing."""
    mock = MagicMock(spec=APIKeyPool)
    mock.get_key.return_value = "test-api-key"
    return mock


@pytest.fixture
def mock_chat_completion() -> ChatCompletion:
    """Create a mock ChatCompletion response."""
    message = ChatCompletionMessage(
        role="assistant",
        content="Test response",
    )
    choice = Choice(
        finish_reason="stop",
        index=0,
        message=message,
    )
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )


@pytest.fixture
def mock_chat_completion_with_tool_calls() -> ChatCompletion:
    """Create a mock ChatCompletion response with tool calls."""
    tool_call = ChatCompletionMessageToolCall(
        id="call_123",
        function=Function(
            name="test_tool",
            arguments='{"arg1": "value1"}',
        ),
        type="function",
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )
    choice = Choice(
        finish_reason="tool_calls",
        index=0,
        message=message,
    )
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )


@pytest.fixture
def mock_chat_completion_chunk() -> ChatCompletionChunk:
    """Create a mock ChatCompletionChunk for streaming."""
    delta = ChoiceDelta(
        content="chunk",
        role="assistant",
    )
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

    choice = ChunkChoice(
        delta=delta,
        finish_reason=None,
        index=0,
    )
    return ChatCompletionChunk(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )


@pytest.fixture
async def mock_stream_response(
    mock_chat_completion_chunk: ChatCompletionChunk,
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Create a mock streaming response generator."""

    async def _generator() -> AsyncGenerator[ChatCompletionChunk, None]:
        yield mock_chat_completion_chunk

    return _generator()


@pytest.fixture
def mock_tool() -> Tool:
    """Create a mock Tool instance."""

    async def tool_func(arg1: str) -> str:
        return f"Result: {arg1}"

    return Tool(
        name="test_tool",
        description="A test tool",
        func=tool_func,
    )


@pytest.fixture
def mock_tool_map(mock_tool: Tool) -> Dict[str, Any]:
    """Create a mock tool map."""
    return {"test_tool": mock_tool.func}


@pytest.fixture
def sample_messages() -> List[Dict[str, Any]]:
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]


@pytest.fixture
def sample_function_signature() -> Dict[str, Any]:
    """Sample function signature data for testing."""
    return {
        "func_name": "test_function",
        "docstring": "A test function",
        "return_type": str,
        "type_hints": {"param1": str, "param2": int, "return": str},
    }


@pytest.fixture
def mock_langfuse_client(mocker: Any) -> Any:
    """Mock Langfuse client."""
    mock_client = MagicMock()
    mock_observation = MagicMock()
    mock_observation.__enter__ = Mock(return_value=mock_observation)
    mock_observation.__exit__ = Mock(return_value=None)
    mock_observation.update = Mock()
    mock_client.start_as_current_observation = Mock(return_value=mock_observation)
    return mock_client


@pytest.fixture
def mock_log_context(mocker: Any) -> Any:
    """Mock log context manager."""
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=None)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    return mock_context


@pytest.fixture
def sample_pydantic_model() -> type[BaseModel]:
    """Sample Pydantic model for testing."""

    class SampleModel(BaseModel):
        name: str
        age: int
        email: Optional[str] = None

    return SampleModel


@pytest.fixture
def img_path() -> ImgPath:
    """Create a test ImgPath instance."""
    import tempfile
    from pathlib import Path

    # Create a temporary file for testing
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.write(b"fake image data")
    temp_file.close()
    return ImgPath(Path(temp_file.name))


@pytest.fixture
def img_url() -> ImgUrl:
    """Create a test ImgUrl instance."""
    return ImgUrl("https://example.com/image.png")


@pytest.fixture
def text_content() -> Text:
    """Create a test Text instance."""
    return Text("Test text content")


@pytest.fixture
def sample_bound_args() -> inspect.BoundArguments:
    """Create sample BoundArguments for testing."""

    def sample_func(param1: str, param2: int = 10) -> str:
        """Sample function."""
        return "result"

    sig = inspect.signature(sample_func)
    bound = sig.bind("test", param2=20)
    bound.apply_defaults()
    return bound

