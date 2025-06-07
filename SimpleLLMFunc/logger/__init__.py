from .logger import (
    setup_logger,
    app_log,
    push_warning,
    push_error,
    push_critical,
    push_info,
    push_debug,
    get_location,
    LogLevel,
    get_logger,
    search_logs_by_trace_id,
    log_context,
    async_log_context,
    JsonFormatter,
    ConsoleFormatter,
    IndexedRotatingFileHandler,
    get_current_trace_id,
    get_current_context_attribute,
    set_current_context_attribute
)

"""
初始化全局日志系统单例并导出日志函数
"""
from .logger_config import logger_config

_log_dir = logger_config.LOG_DIR
_log_file = logger_config.LOG_FILE
_log_level = logger_config.LOG_LEVEL.upper()

# 日志级别映射
_log_level_map = {
    "DEBUG": LogLevel.DEBUG,
    "INFO": LogLevel.INFO,
    "WARNING": LogLevel.WARNING,
    "ERROR": LogLevel.ERROR,
    "CRITICAL": LogLevel.CRITICAL
}

# 将字符串级别转换为枚举
console_level = _log_level_map.get(_log_level, LogLevel.INFO)

# 初始化全局单例日志器
GLOBAL_LOGGER = setup_logger(
    log_dir=_log_dir,
    log_file=_log_file,
    console_level=console_level,     # 控制台显示配置文件中的级别
    file_level=LogLevel.DEBUG,       # 文件记录DEBUG及以上级别，确保所有日志都被记录到文件
    use_json=True,                   # 文件中使用JSON格式便于解析
    use_color=True,                  # 控制台使用彩色输出
    max_file_size=50 * 1024 * 1024,  # 50MB
    backup_count=10                  # 保留10个备份
)

# 记录日志系统初始化完成
push_info(f"全局日志系统初始化完成, 日志文件路径: {_log_dir}/{_log_file}, 日志级别: {_log_level}")

# 确保启动时打印一条测试日志
push_debug("测试DEBUG级别日志")
app_log("测试INFO级别日志(app_log)")
push_info("测试INFO级别日志(push_info)")

__all__ = [
    'app_log',
    'push_warning',
    'push_error',
    'push_critical',
    'push_info',
    'push_debug',
    'get_location',
    'search_logs_by_trace_id',
    'log_context',
    'async_log_context',
    'LogLevel',
    'get_logger',
    'setup_logger',
    'JsonFormatter',
    'ConsoleFormatter',
    'IndexedRotatingFileHandler',
    'get_current_trace_id',
    'get_current_context_attribute',
    'set_current_context_attribute'
]

