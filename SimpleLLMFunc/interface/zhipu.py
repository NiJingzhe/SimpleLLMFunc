import json
import time
from typing import Generator, Optional, Dict, List, Union, Literal, Iterable, Any

# 修改导入路径
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible
from SimpleLLMFunc.interface.key_pool import APIKeyPool
# 修复全局日志器函数导入
from SimpleLLMFunc.logger import app_log, push_warning, get_location, get_current_trace_id

# 定义智谱AI的基础URL
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# 修改Zhipu类继承自OpenAICompatible
class Zhipu(OpenAICompatible):
    """智谱AI的LLM接口实现，基于OpenAICompatible"""

    def __init__(
        self,
        api_key_pool: APIKeyPool,
        model_name: Literal[
            "glm-4-flash",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-long",
            "glm-4-plus",
            "glm-4-0520",
        ],
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ):
        """初始化智谱AI接口

        Args:
            api_key_pool: API密钥池，用于管理和分配API密钥
            model_name: 要使用的智谱模型名称
            max_retries: 最大重试次数
            retry_delay: 重试间隔时间（秒）
        """
        # 智谱AI支持的模型列表
        allowed_models = [
            "glm-4-flash",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-long",
            "glm-4-plus",
            "glm-4-0520",
        ]
        
        # 调用OpenAICompatible的初始化方法，传入智谱AI特定的参数
        super().__init__(
            api_key_pool=api_key_pool,
            model_name=model_name,
            base_url=ZHIPU_BASE_URL,
            max_retries=max_retries,
            retry_delay=retry_delay,
            allowed_models=allowed_models
        )


if __name__ == "__main__":
    # 测试interface
    
    from SimpleLLMFunc.interface.key_pool import APIKeyPool
    from typing import List
    from SimpleLLMFunc.config import global_settings
    import re
    
    # 修改后的正则表达式模式，保留减号
    pattern = re.compile(r'[\s\n]+')
    
    # 去除多余字符的函数
    def clean_api_keys(api_key_list: List[str]) -> List[str]:
        return [pattern.sub('', key.strip()) for key in api_key_list]
    
    # 直接使用 global_config 中的 API KEY 列表，不需要 split
    app_log(
        f"ZHIPUAI_API_KEY_LIST: {global_settings.ZHIPU_API_KEYS}",
        trace_id="test_trace_id"
    )
    
    ZHIPUAI_API_KEY_POOL = APIKeyPool(global_settings.ZHIPU_API_KEYS, "zhipu")
    
    ZhipuAI_glm_4_flash_Interface = Zhipu(ZHIPUAI_API_KEY_POOL, "glm-4-flash")
    
    # 测试 chat 方法
    trace_id = "test_trace_id"
    messages = [{"role": "user", "content": "你好"}]
    response = ZhipuAI_glm_4_flash_Interface.chat(trace_id, messages=messages)
    print("Chat response:", response)
    
    # 测试 chat_stream 方法
    trace_id = "test_trace_id"
    messages = [{"role": "user", "content": "你好"}]
    response = ZhipuAI_glm_4_flash_Interface.chat_stream(trace_id, messages=messages)
    print("Chat stream response:")
    for chunk in response:
        print(chunk)
