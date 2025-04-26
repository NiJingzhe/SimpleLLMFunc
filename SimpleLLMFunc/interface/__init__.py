from SimpleLLMFunc.interface.zhipu import Zhipu
from SimpleLLMFunc.interface.volcengine import VolcEngine
from SimpleLLMFunc.config import global_settings
from SimpleLLMFunc.interface.key_pool import APIKeyPool
from typing import List

# 从 .env 文件读取 API KEY 列表
ZHIPUAI_API_KEY_LIST: List[str] = global_settings.ZHIPU_API_KEYS
VOLCENGINE_API_KEY_LIST: List[str] = global_settings.VOLCENGINE_API_KEYS

# API KEY POOL is a singleton object
ZHIPUAI_API_KEY_POOL = APIKeyPool(ZHIPUAI_API_KEY_LIST, "zhipu")
VOLCENGINE_API_KEY_POOL = APIKeyPool(VOLCENGINE_API_KEY_LIST, "volcengine")


ZhipuAI_glm_4_flash_Interface = Zhipu(ZHIPUAI_API_KEY_POOL, "glm-4-flash")

VolcEngine_deepseek_v3_Interface = VolcEngine(VOLCENGINE_API_KEY_POOL, "deepseek-v3-250324")


__all__ = [
    "ZhipuAI_glm_4_flash_Interface",
    "VolcEngine_deepseek_v3_Interface",
]