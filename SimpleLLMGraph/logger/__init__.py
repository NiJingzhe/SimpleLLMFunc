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
    SearchableLogHandler,
    search_logs_by_trace_id
)

"""
初始化全局日志系统单例并导出日志函数
"""
from .logger_config import logger_config

_log_dir = logger_config.LOG_DIR
_log_level_map = {
    "DEBUG": LogLevel.DEBUG,
    "INFO": LogLevel.INFO,
    "WARNING": LogLevel.WARNING,
    "ERROR": LogLevel.ERROR,
    "CRITICAL": LogLevel.CRITICAL
}

# 初始化全局单例日志器
GLOBAL_LOGGER = setup_logger(
    log_dir=_log_dir,
    log_file=logger_config.LOG_FILE,
    console_level=LogLevel.INFO,     # 控制台显示INFO及以上级别
    file_level=_log_level_map[logger_config.LOG_LEVEL],       # 文件记录DEBUG及以上级别
    use_json=True,                   # 文件中使用JSON格式便于解析
    use_color=True,                  # 控制台使用彩色输出
    max_file_size=50 * 1024 * 1024,  # 50MB
    backup_count=10                  # 保留10个备份
)

# 记录日志系统初始化完成
push_info("全局日志系统初始化完成")

__all__ = [
    'app_log',
    'push_warning',
    'push_error',
    'push_critical',
    'push_info',
    'push_debug',
    'get_location',
    'search_logs_by_trace_id',
    'SearchableLogHandler',
    'GLOBAL_LOGGER',
    'LogLevel'
]

