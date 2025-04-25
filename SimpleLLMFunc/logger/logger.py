import os
import sys
import time
import json
import logging
from logging import LogRecord
import inspect
import traceback
from enum import Enum, auto
from typing import Dict, Optional, Any, List, Union, Callable, Type, TypeVar, cast, ContextManager
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Generator


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class JsonFormatter(logging.Formatter):
    """JSON格式化器，将日志记录转换为结构化JSON格式"""
    
    def __init__(self, include_extra_fields: bool = True) -> None:
        """
        初始化JSON格式化器
        
        Args:
            include_extra_fields: 是否包含额外字段（如trace_id, location等）
        """
        super().__init__()
        self.include_extra_fields = include_extra_fields
        
    def format(self, record: LogRecord) -> str:
        """
        将日志记录格式化为JSON字符串
        
        Args:
            record: 日志记录对象
            
        Returns:
            格式化后的JSON字符串
        """
        # 基本日志字段
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
            "process": record.process
        }
        
        # 添加异常信息（如果有）
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # 添加自定义字段
        if self.include_extra_fields:
            # 添加trace_id
            if hasattr(record, "trace_id") and getattr(record, "trace_id", ""):
                log_data["trace_id"] = getattr(record, "trace_id", "")
            
            # 添加位置信息
            if hasattr(record, "location") and getattr(record, "location", ""):
                log_data["location"] = getattr(record, "location", "")
            
            # 添加其他extra字段
            for key, value in record.__dict__.items():
                if key not in {"args", "asctime", "created", "exc_info", "exc_text", 
                              "filename", "funcName", "id", "levelname", "levelno", 
                              "lineno", "module", "msecs", "message", "msg", 
                              "name", "pathname", "process", "processName", 
                              "relativeCreated", "stack_info", "thread", "threadName",
                              "trace_id", "location"} and not key.startswith("_"):
                    try:
                        # 尝试JSON序列化，确保值可序列化
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, OverflowError):
                        # 如果不可序列化，转换为字符串
                        log_data[key] = str(value)
        
        return json.dumps(log_data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """控制台日志格式化器，支持彩色输出"""
    
    # ANSI颜色代码
    COLORS = {
        "DEBUG": "\033[36m",      # 青色
        "INFO": "\033[32m",       # 绿色
        "WARNING": "\033[33m",    # 黄色
        "ERROR": "\033[31m",      # 红色
        "CRITICAL": "\033[35m",   # 紫色
        "RESET": "\033[0m"        # 重置
    }
    
    def __init__(self, use_color: bool = True, format_string: Optional[str] = None) -> None:
        """
        初始化控制台格式化器
        
        Args:
            use_color: 是否使用彩色输出
            format_string: 自定义格式字符串
        """
        if format_string is None:
            format_string = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        super().__init__(format_string)
        self.use_color = use_color and sys.stdout.isatty()
    
    def format(self, record: LogRecord) -> str:
        """格式化日志记录"""
        # 使用标准格式器格式化
        formatted = super().format(record)
        
        # 应用颜色（如果启用）
        if self.use_color:
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.COLORS["RESET"])
            formatted = f"{color}{formatted}{self.COLORS['RESET']}"
            
        # 添加trace_id和location（如果存在）
        extra_info = []
        if hasattr(record, "trace_id") and getattr(record, "trace_id", ""):
            extra_info.append(f"trace_id={getattr(record, 'trace_id', '')}")
        if hasattr(record, "location") and getattr(record, "location", ""):
            extra_info.append(f"location={getattr(record, 'location', '')}")
        
        if extra_info:
            formatted += f"\n{'\n'.join(extra_info)}"
            
        formatted = "="*30 + "\n" + formatted + "\n" + "="*30
            
        return formatted


class IndexedRotatingFileHandler(RotatingFileHandler):
    """支持索引的日志文件处理器，增强版RotatingFileHandler"""
    
    def __init__(
        self, 
        filename: str, 
        mode: str = "a", 
        maxBytes: int = 10 * 1024 * 1024, 
        backupCount: int = 5, 
        encoding: Optional[str] = None, 
        delay: bool = False, 
        index_dir: Optional[str] = None
    ) -> None:
        """
        初始化索引日志处理器
        
        Args:
            filename: 日志文件名
            mode: 文件打开模式
            maxBytes: 单个日志文件最大大小
            backupCount: 保留的备份文件数量
            encoding: 文件编码
            delay: 是否延迟打开文件
            index_dir: 索引文件存储目录
        """
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.index_dir = index_dir or os.path.join(os.path.dirname(filename), "log_indices")
        self.trace_indices: Dict[str, List[Dict[str, Any]]] = {}
        self.index_lock = threading.RLock()  # 使用可重入锁
        
        # 确保索引目录存在
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载现有索引
        self._load_indices()
    
    def _get_index_file_path(self) -> str:
        """获取索引文件路径"""
        return os.path.join(self.index_dir, "trace_index.json")
    
    def _load_indices(self) -> None:
        """加载现有索引"""
        try:
            index_file = self._get_index_file_path()
            if os.path.exists(index_file):
                with open(index_file, "r", encoding="utf-8") as f:
                    self.trace_indices = json.load(f)
        except Exception as e:
            sys.stderr.write(f"Error loading trace indices: {str(e)}\n")
    
    def _save_indices(self) -> None:
        """保存索引"""
        with self.index_lock:
            try:
                index_file = self._get_index_file_path()
                # 先写入临时文件，然后重命名，避免文件损坏
                temp_file = f"{index_file}.tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(self.trace_indices, f, ensure_ascii=False, indent=2)
                os.replace(temp_file, index_file)
            except Exception as e:
                sys.stderr.write(f"Error saving trace indices: {str(e)}\n")
    
    def emit(self, record: LogRecord) -> None:
        """发送日志记录并更新索引"""
        # 调用父类方法记录日志
        super().emit(record)
        
        # 如果有trace_id，则更新索引
        if hasattr(record, "trace_id") and getattr(record, "trace_id", ""):
            trace_id = getattr(record, "trace_id", "")
            with self.index_lock:
                try:
                    # 创建索引条目
                    log_entry = {
                        "timestamp": record.created,
                        "file": self.baseFilename,
                        "position": self.stream.tell() if self.stream else 0,
                        "level": record.levelname,
                        "message": record.getMessage()
                    }
                    
                    # 更新索引
                    if trace_id not in self.trace_indices:
                        self.trace_indices[trace_id] = []
                    self.trace_indices[trace_id].append(log_entry)
                    
                    # 定期保存索引（每100条记录）
                    total_entries = sum(len(entries) for entries in self.trace_indices.values())
                    if total_entries % 100 == 0:
                        self._save_indices()
                except Exception as e:
                    sys.stderr.write(f"Error updating trace index: {str(e)}\n")
    
    def search_by_trace_id(self, trace_id: str) -> List[Dict[str, Any]]:
        """按trace_id搜索日志
        
        Args:
            trace_id: 跟踪ID
            
        Returns:
            匹配的日志条目列表
        """
        results: List[Dict[str, Any]] = []
        
        if trace_id in self.trace_indices:
            entries = self.trace_indices[trace_id]
            
            for entry in entries:
                try:
                    # 打开日志文件并定位到特定位置
                    with open(entry["file"], "r", encoding="utf-8") as f:
                        f.seek(entry["position"])
                        line = f.readline().strip()
                        
                        # 如果行为空（可能是因为日志轮换），使用预览
                        if not line:
                            line = f"[Preview] {entry['message_preview']}"
                        
                        # 尝试解析JSON
                        try:
                            log_data = json.loads(line)
                            # 添加原始日志对象到结果
                            results.append({
                                "timestamp": datetime.fromtimestamp(entry["timestamp"]).isoformat(),
                                "level": entry["level"],
                                "content": log_data
                            })
                        except json.JSONDecodeError:
                            # 如果不是JSON，添加原始文本
                            results.append({
                                "timestamp": datetime.fromtimestamp(entry["timestamp"]).isoformat(),
                                "level": entry["level"],
                                "content": line
                            })
                except Exception as e:
                    # 添加错误信息
                    results.append({
                        "timestamp": datetime.fromtimestamp(entry["timestamp"]).isoformat(),
                        "level": entry["level"],
                        "content": f"[Error reading log: {str(e)}] {entry['message_preview']}"
                    })
        
        # 按时间戳排序
        results.sort(key=lambda x: x["timestamp"])
        return results
    
    def doRollover(self) -> None:
        """执行日志文件轮换"""
        # 在轮换前保存索引
        self._save_indices()
        # 调用父类方法轮换文件
        super().doRollover()
    
    def close(self) -> None:
        """关闭处理器并保存索引"""
        self._save_indices()
        super().close()


# 全局日志器对象和处理器
_logger: Optional[logging.Logger] = None
_indexed_handler: Optional[IndexedRotatingFileHandler] = None
_log_context: Dict[str, Any] = {}
_context_lock = threading.RLock()

# 用于表示默认trace_id的常量
DEFAULT_TRACE_ID = ""


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


def _merge_context(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合并当前上下文和额外参数
    
    Args:
        extra: 额外参数字典
        
    Returns:
        合并后的字典
    """
    result = {}
    with _context_lock:
        # 复制当前上下文
        result.update(_log_context)
    
    # 添加额外参数（如果有）
    if extra:
        result.update(extra)
    
    return result


def get_current_trace_id() -> str:
    """获取当前上下文中的trace_id
    
    如果上下文中没有trace_id，则返回空字符串
    
    Returns:
        当前上下文中的trace_id
    """
    with _context_lock:
        return _log_context.get("trace_id", DEFAULT_TRACE_ID)


def setup_logger(
    log_dir: str = "logs",
    log_file: str = "application.log",
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.DEBUG,
    use_json: bool = True,
    use_color: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    logger_name: str = "SimpleLLMFunc"
) -> logging.Logger:
    """设置日志系统
    
    Args:
        log_dir: 日志文件目录
        log_file: 日志文件名
        console_level: 控制台日志级别
        file_level: 文件日志级别
        use_json: 是否使用JSON格式记录文件日志
        use_color: 控制台日志是否使用彩色输出
        max_file_size: 单个日志文件最大大小（字节）
        backup_count: 保留的日志文件备份数量
        logger_name: 日志器名称
        
    Returns:
        配置好的Logger对象
    """
    global _logger, _indexed_handler
    
    # 如果日志器已存在，返回现有日志器
    if _logger is not None:
        return _logger
    
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
    console_formatter = ConsoleFormatter(use_color=use_color, format_string=console_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 配置文件处理器
    log_path = os.path.join(log_dir, log_file)
    
    # 使用不同的格式化器，取决于是否使用JSON
    if use_json:
        file_formatter = JsonFormatter(include_extra_fields=True)
    else:
        file_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        file_formatter = logging.Formatter(file_format)
    
    # 使用可索引的日志处理器
    indexed_handler = IndexedRotatingFileHandler(
        filename=log_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding="utf-8"
    )
    indexed_handler.setLevel(getattr(logging, file_level.name))
    indexed_handler.setFormatter(file_formatter)
    logger.addHandler(indexed_handler)
    
    # 缓存对象
    _logger = logger
    _indexed_handler = indexed_handler
    
    # 记录初始化日志
    location = get_location()
    logger.info(
        f"Logger initialized (dir={log_dir}, file={log_file})", 
        extra={"trace_id": "init", "location": location}
    )
    
    return logger


def get_logger() -> logging.Logger:
    """获取已配置的logger，如果未配置则自动配置一个默认的
    
    Returns:
        配置好的Logger对象
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger

@contextmanager
def log_context(**kwargs: Any) -> Generator[None, None, None]:
    """创建日志上下文，在上下文中的所有日志都会包含指定的字段
    
    可以通过提供trace_id参数来设置当前上下文的跟踪ID
    
    Args:
        **kwargs: 要添加到上下文的键值对
        
    Example:
        with log_context(trace_id="my_function_123", user_id="456"):
            push_info("处理用户请求")  # 日志会自动包含trace_id和user_id
    """
    global _log_context
    # 保存原始上下文
    with _context_lock:
        old_context = _log_context.copy()
        # 更新上下文
        _log_context.update(kwargs)
    
    try:
        # 执行上下文中的代码
        yield
    finally:
        # 恢复原始上下文
        with _context_lock:
            _log_context = old_context


def app_log(message: str, trace_id: str = "", location: Optional[str] = None, **kwargs: Any) -> None:
    """记录应用信息日志
    
    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志。如果为空则使用上下文中的trace_id，如果上下文中也有trace_id，则拼接为"上下文trace_id_传递的trace_id"
        location: 代码位置，如不提供则自动获取
        **kwargs: 额外的键值对，将作为字段添加到日志中
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
    
    logger.info(message, extra=extra)


def push_debug(message: str, trace_id: str = "", location: Optional[str] = None, **kwargs: Any) -> None:
    """记录调试信息
    
    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志。如果为空则使用上下文中的trace_id，如果上下文中也有trace_id，则拼接为"上下文trace_id_传递的trace_id"
        location: 代码位置，如不提供则自动获取
        **kwargs: 额外的键值对，将作为字段添加到日志中
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
    
    logger.debug(message, extra=extra)


def push_info(message: str, trace_id: str = "", location: Optional[str] = None, **kwargs: Any) -> None:
    """记录信息
    
    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志。如果为空则使用上下文中的trace_id，如果上下文中也有trace_id，则拼接为"上下文trace_id_传递的trace_id"
        location: 代码位置，如不提供则自动获取
        **kwargs: 额外的键值对，将作为字段添加到日志中
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
    
    logger.info(message, extra=extra)


def push_warning(message: str, trace_id: str = "", location: Optional[str] = None, **kwargs: Any) -> None:
    """记录警告信息
    
    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志。如果为空则使用上下文中的trace_id，如果上下文中也有trace_id，则拼接为"上下文trace_id_传递的trace_id"
        location: 代码位置，如不提供则自动获取
        **kwargs: 额外的键值对，将作为字段添加到日志中
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
    
    logger.warning(message, extra=extra)


def push_error(message: str, trace_id: str = "", location: Optional[str] = None, exc_info: bool = False, **kwargs: Any) -> None:
    """记录错误信息
    
    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志。如果为空则使用上下文中的trace_id，如果上下文中也有trace_id，则拼接为"上下文trace_id_传递的trace_id"
        location: 代码位置，如不提供则自动获取
        exc_info: 是否包含异常信息
        **kwargs: 额外的键值对，将作为字段添加到日志中
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
    
    logger.error(message, exc_info=exc_info, extra=extra)


def push_critical(message: str, trace_id: str = "", location: Optional[str] = None, exc_info: bool = True, **kwargs: Any) -> None:
    """记录严重错误信息
    
    Args:
        message: 日志消息
        trace_id: 跟踪ID，用于关联相关日志。如果为空则使用上下文中的trace_id，如果上下文中也有trace_id，则拼接为"上下文trace_id_传递的trace_id"
        location: 代码位置，如不提供则自动获取
        exc_info: 是否包含异常信息，默认为True
        **kwargs: 额外的键值对，将作为字段添加到日志中
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
    
    logger.critical(message, exc_info=exc_info, extra=extra)


def search_logs_by_trace_id(trace_id: str) -> List[Dict[str, Any]]:
    """按trace_id搜索日志
    
    Args:
        trace_id: 要搜索的跟踪ID
        
    Returns:
        匹配的日志条目列表
    """
    global _indexed_handler
    if _indexed_handler is None:
        get_logger()  # 确保logger已初始化
        
    if _indexed_handler:
        return _indexed_handler.search_by_trace_id(trace_id)
    return []