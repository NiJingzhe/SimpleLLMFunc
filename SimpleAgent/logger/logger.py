import os
import sys
import time
import json
import logging
from logging import LogRecord
import inspect
import traceback
from enum import Enum, auto
from typing import Dict, Optional, Any, List, Union, Callable, Type, TypeVar, cast
from datetime import datetime
from logging.handlers import RotatingFileHandler
import threading
from pathlib import Path


# 扩展 LogRecord 类型，增加我们的自定义属性
class ExtendedLogRecord(LogRecord):
    """扩展标准 LogRecord 类型，添加 trace_id 和 location 属性"""
    trace_id: str
    location: str


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class CustomLogRecord(logging.LogRecord):
    """自定义日志记录类，添加了trace_id和location属性"""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.trace_id: str = kwargs.pop('trace_id', '')
        self.location: str = kwargs.pop('location', '')
        super().__init__(*args, **kwargs)


class CustomFormatter(logging.Formatter):
    """自定义日志格式化器，支持颜色和JSON格式"""
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }

    def __init__(self, use_color: bool = True, json_format: bool = False) -> None:
        super().__init__()
        self.use_color = use_color and sys.stdout.isatty()
        self.json_format = json_format

    def format(self, record: LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        if self.json_format:
            log_data: Dict[str, Any] = {
                'timestamp': timestamp,
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'line': record.lineno,
                'thread': record.threadName
            }
            
            # 添加trace_id和location（如果存在）
            if hasattr(record, 'trace_id') and getattr(record, 'trace_id', ''):
                log_data['trace_id'] = getattr(record, 'trace_id', '')
            if hasattr(record, 'location') and getattr(record, 'location', ''):
                log_data['location'] = getattr(record, 'location', '')
                
            return json.dumps(log_data, ensure_ascii=False)
        
        else:
            level_name = record.levelname
            
            level_str = ""
            # 为控制台输出添加颜色
            if self.use_color:
                level_color = self.COLORS.get(level_name, self.COLORS['RESET'])
                level_str = f"{level_color}{level_name:8}{self.COLORS['RESET']}"
            else:
                level_str = f"{level_name:8}"
                
            # 构建基本日志格式
            log_msg = (
                "========================================================"
                "\n"
                f"[{timestamp}] {level_str}"
                f"\n\t[{record.threadName}] {record.module}:{record.lineno}"
                f"\n\t{record.getMessage()}"
            )            
            # 添加trace_id和location信息（如果存在）
            if hasattr(record, 'trace_id') and getattr(record, 'trace_id', ''):
                log_msg += f"\n\t[trace_id:{getattr(record, 'trace_id', '')}]"
            if hasattr(record, 'location') and getattr(record, 'location', ''):
                log_msg += f"\n\t[location:{getattr(record, 'location', '')}]"
            
            log_msg += "\n" + "========================================================"
            
            return log_msg


class SearchableLogHandler(RotatingFileHandler):
    """可搜索的日志处理器，支持按trace_id快速检索日志"""
    
    def __init__(
        self, 
        filename: str, 
        mode: str = 'a', 
        maxBytes: int = 10*1024*1024, 
        backupCount: int = 5, 
        encoding: Optional[str] = None, 
        delay: bool = False, 
        index_dir: Optional[str] = None
    ) -> None:
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.index_dir = index_dir or os.path.join(os.path.dirname(filename), 'log_indices')
        self.trace_indices: Dict[str, List[Dict[str, Any]]] = {}
        self.trace_file_lock = threading.Lock()
        
        # 确保索引目录存在
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载已有的索引
        self._load_indices()
        
    def _load_indices(self) -> None:
        """加载已有的trace索引"""
        try:
            index_file = os.path.join(self.index_dir, 'trace_index.json')
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.trace_indices = json.load(f)
        except Exception as e:
            sys.stderr.write(f"Error loading trace indices: {str(e)}\n")
            
    def _save_indices(self) -> None:
        """保存trace索引"""
        try:
            with self.trace_file_lock:
                index_file = os.path.join(self.index_dir, 'trace_index.json')
                with open(index_file, 'w', encoding='utf-8') as f:
                    json.dump(self.trace_indices, f, ensure_ascii=False, indent=2)
        except Exception as e:
            sys.stderr.write(f"Error saving trace indices: {str(e)}\n")
            
    def emit(self, record: LogRecord) -> None:
        """发送日志记录，同时更新索引"""
        super().emit(record)
        
        # # 如果有trace_id，则更新索引
        # if hasattr(record, 'trace_id') and getattr(record, 'trace_id', ''):
        #     trace_id = getattr(record, 'trace_id', '')
        #     log_entry = {
        #         'timestamp': record.created,
        #         'file': self.baseFilename,
        #         'position': self.stream.tell(),
        #         'level': record.levelname,
        #         'message_preview': record.getMessage()[:100]  # 存储消息预览
        #     }
            
        #     with self.trace_file_lock:
        #         if trace_id not in self.trace_indices:
        #             self.trace_indices[trace_id] = []
        #         self.trace_indices[trace_id].append(log_entry)
                
        #         # 定期保存索引（每100条记录）
        #         if sum(len(entries) for entries in self.trace_indices.values()) % 100 == 0:
        #             self._save_indices()
                    
    def search_by_trace_id(self, trace_id: str) -> List[Dict[str, Any]]:
        """按trace_id搜索日志"""
        results: List[Dict[str, Any]] = []
        
        if trace_id in self.trace_indices:
            entries = self.trace_indices[trace_id]
            
            for entry in entries:
                try:
                    # 打开日志文件并定位到特定位置
                    with open(entry['file'], 'r', encoding='utf-8') as f:
                        f.seek(entry['position'])
                        line = f.readline().strip()
                        
                        # 如果行为空（可能是因为日志轮换），使用预览
                        if not line:
                            line = f"[Preview] {entry['message_preview']}"
                            
                        results.append({
                            'timestamp': datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                            'level': entry['level'],
                            'content': line
                        })
                except Exception as e:
                    results.append({
                        'timestamp': datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        'level': entry['level'],
                        'content': f"[Error reading log: {str(e)}] {entry['message_preview']}"
                    })
                    
        return results
    
    def close(self) -> None:
        """关闭处理器，保存索引"""
        self._save_indices()
        super().close()


# 全局日志器对象
_logger: Optional[logging.Logger] = None
_searchable_handler: Optional[SearchableLogHandler] = None


def get_location(depth: int = 2) -> str:
    """获取调用者的代码位置信息
    
    Args:
        depth: 调用栈深度，默认为2（调用者的调用者）
        
    Returns:
        str: 格式化的位置字符串，如 "module.py:function:42"
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


def setup_logger(
    log_dir: str = "logs",
    log_file: str = "application.log",
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.DEBUG,
    use_json: bool = False,
    use_color: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """设置日志系统
    
    Args:
        log_dir: 日志文件目录
        log_file: 日志文件名
        console_level: 控制台日志级别
        file_level: 文件日志级别
        use_json: 是否使用JSON格式记录日志
        use_color: 控制台日志是否使用彩色输出
        max_file_size: 单个日志文件最大大小（字节）
        backup_count: 保留的日志文件备份数量
        
    Returns:
        配置好的Logger对象
    """
    global _logger, _searchable_handler
    
    if _logger is not None:
        return _logger
    
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger("SimpleAgent")
    logger.setLevel(logging.DEBUG)  # 设置为最低级别，让handlers决定过滤
    logger.propagate = False  # 不传播到父logger
    
    # 清除任何现有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 使用自定义LogRecord工厂
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args: Any, **kwargs: Any) -> LogRecord:
        record = old_factory(*args, **kwargs)
        # 我们不会在这里设置trace_id和location属性，而是依赖extra参数
        return record
    logging.setLogRecordFactory(record_factory)
    
    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.name))
    console_formatter = CustomFormatter(use_color=use_color, json_format=False)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 配置文件处理器
    log_path = os.path.join(log_dir, log_file)
    file_formatter = CustomFormatter(use_color=False, json_format=use_json)
    
    # 使用可搜索的日志处理器
    searchable_handler = SearchableLogHandler(
        filename=log_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    searchable_handler.setLevel(getattr(logging, file_level.name))
    searchable_handler.setFormatter(file_formatter)
    logger.addHandler(searchable_handler)
    
    # 缓存对象
    _logger = logger
    _searchable_handler = searchable_handler
    
    # 使用正确的方式记录初始化日志
    location = get_location()
    logger.info("Logger initialized", extra={"trace_id": "init", "location": location})
    return logger


def get_logger() -> logging.Logger:
    """获取已配置的logger，如果未配置则自动配置一个默认的"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def app_log(message: str, trace_id: str = "", location: Optional[str] = None) -> None:
    """记录应用信息日志
    
    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志
        location: 代码位置，如不提供则自动获取
    """
    logger = get_logger()
    location = location or get_location()
    logger.info(message, extra={"trace_id": trace_id, "location": location})


def push_debug(message: str, location: Optional[str] = None, trace_id: str = "") -> None:
    """记录调试信息
    
    Args:
        message: 日志消息
        location: 代码位置，如不提供则自动获取
        trace_id: 跟踪ID，用于关联相关日志
    """
    logger = get_logger()
    location = location or get_location()
    logger.debug(message, extra={"trace_id": trace_id, "location": location})


def push_info(message: str, location: Optional[str] = None, trace_id: str = "") -> None:
    """记录信息
    
    Args:
        message: 日志消息
        location: 代码位置，如不提供则自动获取
        trace_id: 跟踪ID，用于关联相关日志
    """
    logger = get_logger()
    location = location or get_location()
    logger.info(message, extra={"trace_id": trace_id, "location": location})


def push_warning(message: str, location: Optional[str] = None, trace_id: str = "") -> None:
    """记录警告信息
    
    Args:
        message: 日志消息
        location: 代码位置，如不提供则自动获取
        trace_id: 跟踪ID，用于关联相关日志
    """
    logger = get_logger()
    location = location or get_location()
    logger.warning(message, extra={"trace_id": trace_id, "location": location})


def push_error(message: str, location: Optional[str] = None, trace_id: str = "", exc_info: bool = False) -> None:
    """记录错误信息
    
    Args:
        message: 日志消息
        location: 代码位置，如不提供则自动获取
        trace_id: 跟踪ID，用于关联相关日志
        exc_info: 是否包含异常信息
    """
    logger = get_logger()
    location = location or get_location()
    logger.error(message, exc_info=exc_info, extra={"trace_id": trace_id, "location": location})


def push_critical(message: str, location: Optional[str] = None, trace_id: str = "", exc_info: bool = True) -> None:
    """记录严重错误信息
    
    Args:
        message: 日志消息
        location: 代码位置，如不提供则自动获取
        trace_id: 跟踪ID，用于关联相关日志
        exc_info: 是否包含异常信息，默认为True
    """
    logger = get_logger()
    location = location or get_location()
    logger.critical(message, exc_info=exc_info, extra={"trace_id": trace_id, "location": location})


def search_logs_by_trace_id(trace_id: str) -> List[Dict[str, Any]]:
    """按trace_id搜索日志
    
    Args:
        trace_id: 要搜索的跟踪ID
        
    Returns:
        匹配的日志条目列表
    """
    global _searchable_handler
    if _searchable_handler is None:
        get_logger()  # 确保logger已初始化
        
    if _searchable_handler:
        return _searchable_handler.search_by_trace_id(trace_id)
    return []