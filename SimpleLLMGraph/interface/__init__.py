from SimpleLLMGraph.interface.zhipu import Zhipu
from SimpleLLMGraph.config import global_settings
from SimpleLLMGraph.interface.key_pool import APIKeyPool
from typing import List

# 从 .env 文件读取 API KEY 列表
ZHIPUAI_API_KEY_LIST: List[str] = global_settings.ZHIPU_API_KEYS

# API KEY POOL is a singleton object
ZHIPUAI_API_KEY_POOL = APIKeyPool(ZHIPUAI_API_KEY_LIST, "zhipu")

ZhipuAI_glm_4_flash_Interface = Zhipu(ZHIPUAI_API_KEY_POOL, "glm-4-flash")