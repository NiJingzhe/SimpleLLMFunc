"""Console-only logger facade for SimpleLLMFunc.

This module re-exports the public logging API while keeping compatibility
with historical import paths.
"""

from __future__ import annotations

from .core import (
    setup_logger,
    get_logger,
    push_debug,
    push_info,
    push_warning,
    push_error,
    push_critical,
    app_log,
)
from .context_manager import (
    log_context,
    async_log_context,
    get_current_trace_id,
    get_current_context_attribute,
    set_current_context_attribute,
)
from .types import LogLevel
from .utils import get_location
from .formatters import ConsoleFormatter

__all__ = [
    "setup_logger",
    "get_logger",
    "push_debug",
    "push_info",
    "push_warning",
    "push_error",
    "push_critical",
    "app_log",
    "log_context",
    "async_log_context",
    "get_current_trace_id",
    "get_current_context_attribute",
    "set_current_context_attribute",
    "LogLevel",
    "get_location",
    "ConsoleFormatter",
]
