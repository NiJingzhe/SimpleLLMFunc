from logging import getLogRecordFactory, getLogger
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
import os
from pathlib import Path


class LoggerConfig(BaseSettings):


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "agent.log"
    LOG_DIR: str = Field(
        default=str(Path(__file__).parent.parent), description="日志文件存放目录"
    )


@lru_cache
def get_logger_config() -> LoggerConfig:
    return LoggerConfig()

logger_config = get_logger_config()
