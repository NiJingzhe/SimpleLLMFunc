"""
日志系统核心功能模块

本模块包含了日志系统的核心功能，包括日志器设置、日志记录函数等。
这是日志系统的核心部分，提供了所有主要的日志操作接口。
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from .context_manager import _merge_context, get_current_trace_id
from .formatters import ConsoleFormatter, JsonFormatter
from .types import LogLevel
from .utils import get_location
from .logger_config import logger_config


# 全局日志器对象和处理器
_logger: Optional[logging.Logger] = None


def setup_logger(
    log_dir: Optional[str] = None,
    log_file: str = "application.log",
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.DEBUG,
    use_json: bool = True,
    use_color: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    logger_name: str = "SimpleLLMFunc",
) -> logging.Logger:
    """
    设置日志系统

    此函数是日志系统的核心设置函数，配置控制台和文件的日志处理器，
    并初始化全局日志器。

    Args:
        log_dir: 日志文件目录，默认为"logs"
        log_file: 日志文件名，默认为"application.log"
        console_level: 控制台日志级别，默认为INFO
        file_level: 文件日志级别，默认为DEBUG
        use_json: 是否使用JSON格式记录文件日志，默认为True
        use_color: 控制台日志是否使用彩色输出，默认为True
        max_file_size: 单个日志文件最大大小（字节），默认为10MB
        backup_count: 保留的日志文件备份数量，默认为5
        logger_name: 日志器名称，默认为"SimpleLLMFunc"

    Returns:
        配置好的Logger对象

    Example:
        >>> logger = setup_logger(
        ...     log_dir="custom_logs",
        ...     console_level=LogLevel.DEBUG
        ... )

    Note:
        - 如果日志器已存在，返回现有日志器
        - 自动创建日志目录
        - 设置独立的控制台和文件处理器
    """
    global _logger

    # 如果日志器已存在，返回现有日志器
    if _logger is not None:
        return _logger

    # 使用配置中的LOG_DIR作为默认值
    if log_dir is None:
        log_dir = logger_config.LOG_DIR

    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 创建logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # 设置为最低级别，让handlers决定过滤
    logger.propagate = False  # 不传播到父logger

    # 清除任何现有的处理器
    if logger.handlers:
        logger.handlers.clear()

    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.name))
    console_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    console_formatter = ConsoleFormatter(
        use_color=use_color, format_string=console_format
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 配置文件处理器
    log_path = os.path.join(log_dir, log_file)

    # 使用不同的格式化器，取决于是否使用JSON
    if use_json:
        file_formatter = JsonFormatter()
    else:
        file_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        file_formatter = logging.Formatter(file_format)  # type: ignore[assignment]

    # 使用标准的旋转文件处理器
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, file_level.name))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 缓存对象
    _logger = logger

    # 记录初始化日志
    location = get_location()
    logger.info(
        f"Logger initialized (dir={log_dir}, file={log_file})",
        extra={"trace_id": "init", "location": location},
    )

    return logger


def get_logger() -> logging.Logger:
    """
    获取已配置的logger，如果未配置则自动配置一个默认的

    Returns:
        配置好的Logger对象

    Example:
        >>> logger = get_logger()
        >>> logger.info("This is a test message")

    Note:
        如果全局logger未初始化，会自动调用setup_logger()进行初始化
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def _log_message(
    level: int,
    message: str,
    trace_id: str = "",
    location: Optional[str] = None,
    exc_info: bool = False,
    **kwargs: Any,
) -> None:
    """
    内部日志记录函数

    Args:
        level: 日志级别（logging.DEBUG, logging.INFO等）
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志
        location: 代码位置，如不提供则自动获取
        exc_info: 是否包含异常信息
        **kwargs: 额外的键值对，将作为字段添加到日志中

    Note:
        - 自动处理trace_id的合并
        - 自动获取代码位置
        - 支持额外字段的传递
    """
    logger = get_logger()
    location = location or get_location()

    # 获取上下文中的trace_id
    context_trace_id = get_current_trace_id()

    # 处理trace_id：如果同时有上下文trace_id和显式传递的trace_id，则通过下划线连接它们
    if context_trace_id and trace_id:
        trace_id = f"{context_trace_id}_{trace_id}"
    elif not trace_id and context_trace_id:
        trace_id = context_trace_id

    # 合并上下文和额外参数
    extra = _merge_context({"trace_id": trace_id, "location": location, **kwargs})

    logger.log(level, message, exc_info=exc_info, extra=extra)


def push_debug(
    message: str, trace_id: str = "", location: Optional[str] = None, **kwargs: Any
) -> None:
    """
    记录调试信息

    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志
        location: 代码位置，如不提供则自动获取
        **kwargs: 额外的键值对，将作为字段添加到日志中

    Example:
        >>> push_debug("This is a debug message", user_id="12345")
    """
    _log_message(logging.DEBUG, message, trace_id, location, **kwargs)


def push_info(
    message: str, trace_id: str = "", location: Optional[str] = None, **kwargs: Any
) -> None:
    """
    记录信息

    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志
        location: 代码位置，如不提供则自动获取
        **kwargs: 额外的键值对，将作为字段添加到日志中

    Example:
        >>> push_info("User login successful", user_id="12345", action="login")
    """
    _log_message(logging.INFO, message, trace_id, location, **kwargs)


def push_warning(
    message: str, trace_id: str = "", location: Optional[str] = None, **kwargs: Any
) -> None:
    """
    记录警告信息

    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志
        location: 代码位置，如不提供则自动获取
        **kwargs: 额外的键值对，将作为字段添加到日志中

    Example:
        >>> push_warning("Configuration file not found, using defaults")
    """
    _log_message(logging.WARNING, message, trace_id, location, **kwargs)


def push_error(
    message: str,
    trace_id: str = "",
    location: Optional[str] = None,
    exc_info: bool = False,
    **kwargs: Any,
) -> None:
    """
    记录错误信息

    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志
        location: 代码位置，如不提供则自动获取
        exc_info: 是否包含异常信息，默认为False
        **kwargs: 额外的键值对，将作为字段添加到日志中

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     push_error("Operation failed", error=str(e), exc_info=True)
    """
    _log_message(logging.ERROR, message, trace_id, location, exc_info, **kwargs)


def push_critical(
    message: str,
    trace_id: str = "",
    location: Optional[str] = None,
    exc_info: bool = True,
    **kwargs: Any,
) -> None:
    """
    记录严重错误信息

    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志
        location: 代码位置，如不提供则自动获取
        exc_info: 是否包含异常信息，默认为True
        **kwargs: 额外的键值对，将作为字段添加到日志中

    Example:
        >>> push_critical("System is shutting down due to critical error")
    """
    _log_message(logging.CRITICAL, message, trace_id, location, exc_info, **kwargs)


def app_log(
    message: str, trace_id: str = "", location: Optional[str] = None, **kwargs: Any
) -> None:
    """
    记录应用信息日志（等同于push_info）

    此函数提供了一个别名，用于记录应用级别的信息日志，
    与push_info功能完全相同。

    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志
        location: 代码位置，如不提供则自动获取
        **kwargs: 额外的键值对，将作为字段添加到日志中

    Example:
        >>> app_log("Application started successfully", version="1.0.0")
    """
    push_info(message, trace_id, location, **kwargs)
