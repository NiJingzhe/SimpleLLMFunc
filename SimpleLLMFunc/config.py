from functools import lru_cache
from pydantic_settings import BaseSettings, ForceDecode, SettingsConfigDict
from typing import Annotated, List


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
   
    ENABLE_ZHIPU: bool = False 
    ZHIPU_API_KEYS: Annotated[List[str], ForceDecode] = []
    
    ENABLE_VOLCENGINE: bool = False
    VOLCENGINE_API_KEYS: Annotated[List[str], ForceDecode] = []

@lru_cache
def get_settings() -> Settings:
    return Settings()

global_settings = get_settings()

__all__ = [
    "global_settings",
]