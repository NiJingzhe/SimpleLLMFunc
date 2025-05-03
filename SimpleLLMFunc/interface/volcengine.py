import json
import time
from typing import Generator, Optional, Dict, List, Union, Literal, Iterable, Any

# 修改导入路径
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible
from SimpleLLMFunc.interface.key_pool import APIKeyPool
# 修复全局日志器函数导入
from SimpleLLMFunc.logger import app_log, push_warning, push_error, get_location, get_current_trace_id

# 定义火山引擎的基础URL
VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# 修改VolcEngine类继承自OpenAICompatible
class VolcEngine(OpenAICompatible):
    """火山引擎的LLM接口实现，基于OpenAICompatible"""

    def __init__(
        self,
        api_key_pool: APIKeyPool,
        # 目前只做针对DSV3的支持，用于测试，后面慢慢补，然后准备迁移到VolcEngine自己的SDK上
        model_name: Literal[
            "deepseek-v3-250324"
        ],
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ):
        """初始化火山引擎接口

        Args:
            api_key_pool: API密钥池，用于管理和分配API密钥
            model_name: 要使用的火山引擎模型名称
            max_retries: 最大重试次数
            retry_delay: 重试间隔时间（秒）
        """
        # 火山引擎支持的模型列表
        allowed_models = [
            "deepseek-v3-250324",
        ]
        
        # 调用OpenAICompatible的初始化方法，传入火山引擎特定的参数
        super().__init__(
            api_key_pool=api_key_pool,
            model_name=model_name,
            base_url=VOLCENGINE_BASE_URL,
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
    
    # 直接使用 global_config 中的 API KEY 列表，不需要 split
    app_log(
        f"VOLCENGINE_API_KEY_LIST: {global_settings.VOLCENGINE_API_KEYS}",
        trace_id="test_trace_id"
    )
    
    VOLCENGINE_API_KEY_POOL = APIKeyPool(global_settings.VOLCENGINE_API_KEYS, "volcengine")
    
    VolcEngine_deepseek_v3_Interface = VolcEngine(VOLCENGINE_API_KEY_POOL, "deepseek-v3-250324")
    
    # 测试 chat 方法
    trace_id = "test_trace_id"
    messages = [{"role": "user", "content": "你好"}]
    response = VolcEngine_deepseek_v3_Interface.chat(trace_id, messages=messages)
    print("Chat response:", response)
    
    # 测试 chat_stream 方法
    trace_id = "test_trace_id"
    messages = [{"role": "user", "content": "你好"}]
    response = VolcEngine_deepseek_v3_Interface.chat_stream(trace_id, messages=messages)
    print("Chat stream response:")
    for chunk in response:
        print(chunk)

