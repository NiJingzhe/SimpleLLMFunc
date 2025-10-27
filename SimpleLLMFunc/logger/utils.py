"""
日志系统工具函数模块

本模块包含了日志系统使用的各种工具函数，包括位置获取、时间转换等。
"""

import inspect
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


def convert_float_to_datetime_with_tz(
    time_float: float, tz=timezone(timedelta(hours=8))
) -> datetime:
    """
    将浮点时间戳转换为带时区的datetime对象

    Args:
        time_float: 浮点时间戳（从1970-01-01 00:00:00 UTC开始的秒数）
        tz: 时区信息，默认使用东八区（北京时间）

    Returns:
        转换完成后的datetime对象，带有时区信息

    Example:
        >>> timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
        >>> dt = convert_float_to_datetime_with_tz(timestamp)
        >>> print(dt)  # 2022-01-01 08:00:00+08:00
    """
    return datetime.fromtimestamp(time_float, tz=tz)


def get_location(depth: int = 2) -> str:
    """
    获取调用者的代码位置信息

    此函数通过检查调用栈来获取调用者的位置信息，用于在日志中标识代码位置。

    Args:
        depth: 调用栈深度，默认为2（调用者的调用者）

    Returns:
        格式化的位置字符串，如 "module.py:function:42"

    Example:
        >>> location = get_location()
        >>> print(location)  # "main.py:main:10"

    Note:
        - depth=1: 当前函数
        - depth=2: 调用当前函数的函数（默认）
        - 如果无法获取位置信息，返回"unknown"
    """
    frame = inspect.currentframe()
    try:
        # 向上追溯调用栈
        for _ in range(depth):
            if frame is None:
                break
            frame = frame.f_back

        if frame:
            frame_info = inspect.getframeinfo(frame)
            filename = os.path.basename(frame_info.filename)
            return f"{filename}:{frame_info.function}:{frame_info.lineno}"
        else:
            return "unknown"
    finally:
        # 删除引用，避免循环引用
        del frame


def format_extra_fields(record_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化日志记录的额外字段

    将日志记录字典中的额外字段进行处理，确保所有值都可以序列化。
    对不可序列化的值进行字符串转换。

    Args:
        record_dict: 日志记录的字典表示

    Returns:
        处理后的字典，所有值都可以进行JSON序列化

    Note:
        - 排除标准日志字段和私有字段
        - 对不可序列化的值进行字符串转换
    """
    import json

    # 需要排除的标准字段
    excluded_fields = {
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
    }

    result = {}
    for key, value in record_dict.items():
        if not key.startswith("_") and key not in excluded_fields:
            try:
                # 尝试JSON序列化，确保值可序列化
                json.dumps(value)
                result[key] = value
            except (TypeError, OverflowError):
                # 如果不可序列化，转换为字符串
                result[key] = str(value)

    return result


def safe_dict_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    安全地合并多个字典

    按顺序合并字典，后面的字典覆盖前面的字典中的相同键。

    Args:
        *dicts: 要合并的字典列表

    Returns:
        合并后的新字典

    Example:
        >>> dict1 = {"a": 1, "b": 2}
        >>> dict2 = {"b": 3, "c": 4}
        >>> merged = safe_dict_merge(dict1, dict2)
        >>> print(merged)  # {"a": 1, "b": 3, "c": 4}
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result
