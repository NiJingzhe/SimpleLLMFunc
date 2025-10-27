"""
日志系统类型定义

本模块定义了日志系统使用到的所有类型和枚举。
"""

from enum import Enum, auto


class LogLevel(Enum):
    """
    日志级别枚举

    定义了标准的日志级别，用于控制日志的详细程度。

    Attributes:
        DEBUG: 调试级别，用于开发和调试时的详细信息
        INFO: 信息级别，用于记录正常运行时的关键信息
        WARNING: 警告级别，用于记录可能出现问题的情况
        ERROR: 错误级别，用于记录运行时错误
        CRITICAL: 严重错误级别，用于记录严重影响系统运行的错误
    """

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
