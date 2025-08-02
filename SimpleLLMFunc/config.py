"""
@File    :   config.py
@Time    :   2025/08/03 02:19:19
@Author  :   Jingzhe Ni 
@Contact :   nijingzhe@zju.edu.cn
@License :   (C)Copyright 2025, Jingzhe Ni
@Desc    :   Config for SimpleLLMFunc
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Config class for SimpleLLMFunc
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get settings from .env file
    """
    return Settings()


global_settings = get_settings()

__all__ = [
    "global_settings",
]
