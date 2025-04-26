from functools import lru_cache
from json import load
from pydantic import field_validator
from pydantic_settings import BaseSettings, ForceDecode, SettingsConfigDict
from typing import Annotated, List
from dotenv import load_dotenv


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    ZHIPU_API_KEYS: Annotated[List[str], ForceDecode] = []
    VOLCENGINE_API_KEYS: Annotated[List[str], ForceDecode] = []

@lru_cache
def get_settings() -> Settings:
    return Settings()

global_settings = get_settings()
