"""
Observability module for SimpleLLMFunc framework.

This module provides integration with observability platforms like Langfuse
to track LLM generations, tool calls, and overall function execution.
"""

from .langfuse_client import (
    get_langfuse_client,
    langfuse_client,
    flush_all_observations,
)

from .langfuse_config import (
    get_langfuse_config,
    langfuse_config,
)


__all__ = [
    "get_langfuse_client",
    "langfuse_client",
    "get_langfuse_config",
    "langfuse_config",
    "flush_all_observations",
]   