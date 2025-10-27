from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class LangfuseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LANGFUSE_PUBLIC_KEY: str = ""  
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    LANGFUSE_ENABLED: bool = True


@lru_cache
def get_langfuse_config() -> LangfuseConfig:
    return LangfuseConfig()


# 全局配置实例
langfuse_config = get_langfuse_config()
