from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class LoggerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "agent.log"
    LOG_DIR: str = "./"  # 默认为当前目录，避免使用相对路径


@lru_cache
def get_logger_config() -> LoggerConfig:
    return LoggerConfig()


# 全局配置实例
logger_config = get_logger_config()
