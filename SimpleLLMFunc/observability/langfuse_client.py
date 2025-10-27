from langfuse import Langfuse
from SimpleLLMFunc.observability.langfuse_config import langfuse_config
from functools import lru_cache


@lru_cache
def get_langfuse_client() -> Langfuse:

    return Langfuse(
        public_key=langfuse_config.LANGFUSE_PUBLIC_KEY,
        secret_key=langfuse_config.LANGFUSE_SECRET_KEY,
        host=langfuse_config.LANGFUSE_BASE_URL,
    )

# 全局配置实例
langfuse_client = get_langfuse_client()

def flush_all_observations() -> None:
    langfuse_client.flush()


__all__ = [
    "langfuse_client",
    "flush_all_observations",
]