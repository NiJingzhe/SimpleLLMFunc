from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
import os
from pathlib import Path


class LoggerConfig(BaseSettings):

    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "agent.log"
    LOG_DIR: str = Field(
        default=str(Path(__file__).parent.parent), description="日志文件存放目录"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


logger_config = LoggerConfig()
