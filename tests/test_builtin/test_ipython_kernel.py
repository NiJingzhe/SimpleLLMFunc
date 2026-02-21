"""Tests for iPython Kernel builtin tool."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestKernelCreation:
    """Test Kernel class creation."""

    def test_kernel_creation(self):
        """Test creating a Kernel instance."""
        from SimpleLLMFunc.builtin import Kernel

        kernel = Kernel()
        assert kernel is not None
        assert kernel.kernel_name == "python3"
        assert kernel.timeout == 30

    def test_kernel_creation_with_custom_params(self):
        """Test creating a Kernel with custom parameters."""
        from SimpleLLMFunc.builtin import Kernel

        kernel = Kernel(kernel_name="python3", timeout=60)
        assert kernel.kernel_name == "python3"
        assert kernel.timeout == 60


class TestKernelToolset:
    """Test Kernel.toolset property."""

    def test_toolset_returns_list(self):
        """Test that toolset returns a list."""
        from SimpleLLMFunc.builtin import Kernel

        kernel = Kernel()
        toolset = kernel.toolset
        assert isinstance(toolset, list)

    def test_toolset_contains_expected_tools(self):
        """Test that toolset contains expected tool names."""
        from SimpleLLMFunc.builtin import Kernel

        kernel = Kernel()
        toolset = kernel.toolset

        tool_names = [tool.name for tool in toolset]
        assert "execute_code" in tool_names
        assert "reset_kernel" in tool_names
        assert "list_variables" in tool_names
        assert "close_kernel" in tool_names

    def test_different_kernel_instances_have_same_tool_names(self):
        """Test that different kernel instances have toolset with same names."""
        from SimpleLLMFunc.builtin import Kernel

        kernel1 = Kernel()
        kernel2 = Kernel()

        names1 = [tool.name for tool in kernel1.toolset]
        names2 = [tool.name for tool in kernel2.toolset]

        assert names1 == names2


class TestKernelToolFunctions:
    """Test Kernel tool functions."""

    @pytest.mark.asyncio
    async def test_execute_code_basic(self):
        """Test basic code execution."""
        from SimpleLLMFunc.builtin import Kernel

        kernel = Kernel()
        with patch.object(kernel, "_ensure_started", new_callable=AsyncMock):
            result = await kernel.execute(code="print('hello')")

        assert result is not None
        assert "success" in result

    @pytest.mark.asyncio
    async def test_reset_kernel(self):
        """Test kernel reset."""
        from SimpleLLMFunc.builtin import Kernel

        kernel = Kernel()
        with patch.object(kernel, "_ensure_started", new_callable=AsyncMock):
            result = await kernel.reset()

        assert result is not None

    @pytest.mark.asyncio
    async def test_list_variables(self):
        """Test listing variables."""
        from SimpleLLMFunc.builtin import Kernel

        kernel = Kernel()
        with patch.object(kernel, "_ensure_started", new_callable=AsyncMock):
            result = await kernel.list_variables()

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_close_kernel(self):
        """Test closing kernel."""
        from SimpleLLMFunc.builtin import Kernel

        kernel = Kernel()
        result = await kernel.close()
        assert result is not None


class TestKernelStreaming:
    """Test Kernel streaming with event_emitter."""

    @pytest.mark.asyncio
    async def test_execute_code_with_event_emitter(self):
        """Test code execution with event_emitter."""
        from SimpleLLMFunc.builtin import Kernel
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        kernel = Kernel()
        emitter = ToolEventEmitter()

        with patch.object(kernel, "_ensure_started", new_callable=AsyncMock):
            with patch.object(
                kernel, "execute", new_callable=AsyncMock
            ) as mock_execute:
                mock_execute.return_value = {"success": True, "stdout": "hello"}
                result = await kernel.execute(
                    code="print('hello')", event_emitter=emitter
                )

        assert result["success"] is True


class TestKernelSession:
    """Test Kernel session management."""

    def test_each_kernel_has_unique_session_id(self):
        """Test that each kernel instance has unique session ID."""
        from SimpleLLMFunc.builtin import Kernel

        kernel1 = Kernel()
        kernel2 = Kernel()

        assert kernel1.session_id != kernel2.session_id
