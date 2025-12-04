"""
日志格式化器模块

本模块包含了用于格式化日志输出的各种格式化器类。
支持JSON格式化和控制台彩色输出格式化。
"""

import json
import logging
from logging import LogRecord
import sys
from typing import Optional


class JsonFormatter(logging.Formatter):
    """
    JSON格式化器，将日志记录转换为结构化JSON格式

    此格式化器将日志记录转换为JSON字符串，便于机器解析和日志分析。
    包含了完整的日志信息，包括时间戳、级别、消息、代码位置等。

    Example:
        >>> formatter = JsonFormatter()
        >>> record = logging.LogRecord(...)
        >>> json_output = formatter.format(record)
    """

    def __init__(self) -> None:
        """
        初始化JSON格式化器

        设置基本的日志格式化器配置。
        """
        super().__init__()

    def format(self, record: LogRecord) -> str:
        """
        将日志记录格式化为JSON字符串

        Args:
            record: 日志记录对象，包含日志的所有信息

        Returns:
            格式化后的JSON字符串

        Note:
            - 自动处理异常信息和堆栈跟踪
            - 对不可序列化的值进行字符串转换
            - 包含所有extra字段和标准日志字段
        """
        # 基本日志字段
        log_data = {
            "timestamp": record.created,  # 使用时间戳而不是格式化的时间字符串
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
            "process": record.process,
        }

        # 添加异常信息（如果有）
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,  # type: ignore
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "traceback": self.formatException(record.exc_info),
            }

        # 添加extra字段
        for key, value in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "id",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "trace_id",
                "location",
            } and not key.startswith("_"):
                try:
                    # 尝试JSON序列化，确保值可序列化
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, OverflowError):
                    # 如果不可序列化，转换为字符串
                    log_data[key] = str(value)

        return json.dumps(log_data, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    控制台日志格式化器，支持彩色输出

    此格式化器为控制台输出提供美观的格式化，包括颜色支持和额外信息显示。
    支持不同日志级别的颜色区分。

    Attributes:
        COLORS: ANSI颜色代码字典
        SUPPORTTED_EXTRA_INFO: 支持显示的额外信息字段

    Example:
        >>> formatter = ConsoleFormatter(use_color=True)
        >>> record = logging.LogRecord(...)
        >>> formatted_output = formatter.format(record)
    """

    # ANSI颜色代码
    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
        "RESET": "\033[0m",  # 重置
    }

    SUPPORTTED_EXTRA_INFO = ["trace_id", "location", "input_tokens", "output_tokens"]

    def __init__(
        self, use_color: bool = True, format_string: Optional[str] = None
    ) -> None:
        """
        初始化控制台格式化器

        Args:
            use_color: 是否使用彩色输出，默认True
            format_string: 自定义格式字符串，默认使用标准格式

        Note:
            颜色支持会自动检测终端是否支持彩色输出
        """
        if format_string is None:
            format_string = (
                "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
            )
        super().__init__(format_string)
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record: LogRecord) -> str:
        """
        格式化日志记录为控制台输出格式

        Args:
            record: 日志记录对象

        Returns:
            格式化后的字符串，包含颜色和额外信息

        Features:
            - 彩色日志级别显示
            - 额外的上下文信息显示
            - 边框装饰
        """
        # 使用标准格式器格式化
        formatted = super().format(record)

        # 应用颜色（如果启用）
        if self.use_color:
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.COLORS["RESET"])
            formatted = f"{color}{formatted}{self.COLORS['RESET']}"

        # 添加各类extra info（如果存在）
        extra_info = []
        for attr in self.SUPPORTTED_EXTRA_INFO:
            attr_value = getattr(record, attr, "")
            if hasattr(record, attr) and attr_value:
                extra_info.append(f"{attr}={attr_value}")

        if extra_info:
            formatted += "\n" + "\n".join(extra_info)

        formatted = "=" * 30 + "\n" + formatted + "\n" + "=" * 30

        return formatted
