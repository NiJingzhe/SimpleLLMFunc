import json
import time
from typing import Generator, Optional, Dict, List, Union, Literal, Iterable, Any
from openai import OpenAI
# 修复导入路径
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.interface.key_pool import APIKeyPool
# 修复全局日志器函数导入
from SimpleLLMFunc.logger import app_log, push_warning, push_error, get_location, get_current_trace_id

VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

class VolcEngine(LLM_Interface):

    def __init__(
        self,
        api_key_pool: APIKeyPool,
        # TODO:
        # 目前只做针对DSV3的支持，用于测试，后面慢慢补，然后准备迁移到VolcEngine自己的SDK上
        model_name: Literal[
            "deepseek-v3-250324"
        ],
        max_retries: int = 5,  # 新增最大重试次数参数
        retry_delay: float = 1.0,  # 新增重试延迟时间
    ):
        super().__init__(api_key_pool, model_name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = VOLCENGINE_BASE_URL

        self.model_list = [
            "deepseek-v3-250324",
        ]

        if model_name not in self.model_list:
            location = get_location()
            push_warning(
                f"model_name should be one of {self.model_list}", location=get_location()
            )
        self.model_name = model_name

        self.key_pool = api_key_pool
        self.client = OpenAI(api_key=api_key_pool.get_least_loaded_key(), base_url=self.base_url)

    def chat(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[False] = False,
        messages: Iterable[Dict[str, str]] = [{"role": "system", "content": "你是一位乐于助人的助手，可以帮助用户解决各种问题。"}],
        timeout: Optional[int] = 120,
        *args,
        **kwargs,
    ) -> Dict[Any, Any]:
        key = self.key_pool.get_least_loaded_key()
        self.client = OpenAI(api_key=key, base_url=self.base_url)

        attempt = 0
        while attempt < self.max_retries:
            try:

                self.key_pool.increment_task_count(key)
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                # data = data[: min(512, len(data))]
                app_log(
                    f"VolcEngine::chat: {self.model_name} request with API key: {key}, and message: {data}",
                    location=get_location()
                )
                response: Dict[Any, Any] = self.client.chat.completions.create(  # type: ignore
                    messages=messages,   # type: ignore
                    model=self.model_name,
                    stream=stream,
                    timeout=timeout,
                    *args,
                    **kwargs,
                )

                self.key_pool.decrement_task_count(key)
                return response  # 请求成功，返回结果
            except Exception as e:

                self.key_pool.decrement_task_count(key)
                attempt += 1
                location = get_location()
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                # data = data[: min(512, len(data))]
                push_warning(
                    f"{self.model_name} Interface attempt {attempt} failed: With message : {data} send, \n but exception : {str(e)} was caught",
                    location=get_location(),
                )

                key = self.key_pool.get_least_loaded_key()
                self.client = OpenAI(api_key=key, base_url=self.base_url)

                if attempt >= self.max_retries:
                    push_error(
                        f"Max retries reached. {self.model_name} Failed to get a response for {data}",
                        location=location,
                    )
                    raise e  # 达到最大重试次数后抛出异常
                time.sleep(self.retry_delay)  # 重试前等待一段时间
        return {}  # 添加默认返回以满足类型检查，实际上这行代码永远不会执行

    def chat_stream(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[True] = True,
        messages: Iterable[Dict[str, str]] = [{"role": "system", "content": "你是一位乐于助人的助手，可以帮助用户解决各种问题。"}],
        timeout: Optional[int] = 50,
        *args,
        **kwargs,
    ) -> Generator[Dict[Any, Any], None, None]:
        key = self.key_pool.get_least_loaded_key()
        self.client = OpenAI(api_key=key, base_url=self.base_url)

        attempt = 0
        while attempt < self.max_retries:
            try:
                self.key_pool.increment_task_count(key)
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                # data = data[: min(512, len(data))]
                app_log(
                    f"VolcEngine::chat_stream: {self.model_name} request with API key: {key}, and message: {data}",
                    location=get_location()
                )
                response: Generator[Dict[Any, Any], None, None] = self.client.chat.completions.create(  # type: ignore
                    messages=messages,  # type: ignore
                    model=self.model_name,
                    stream=stream,
                    timeout=timeout,
                    *args,
                    **kwargs,
                )

                for chunk in response:
                    yield chunk  # 按块返回生成器中的数据

                self.key_pool.decrement_task_count(key)
                break  # 如果成功，跳出重试循环
            except Exception as e:

                self.key_pool.decrement_task_count(key)
                attempt += 1
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                # data = data[: min(512, len(data))]
                push_warning(
                    f"{self.model_name} Interface attempt {attempt} failed: With message : {data} send, \n but exception : {str(e)} was caught",
                    location=get_location()
                )

                key = self.key_pool.get_least_loaded_key()
                self.client = OpenAI(api_key=key, base_url=self.base_url)

                if attempt >= self.max_retries:
                    push_error(
                        f"Max retries reached. {self.model_name} Failed to get a response for {data}",
                        location=get_location()
                    )
                    raise e
                time.sleep(self.retry_delay)

        # 下面是一个空生成器，用于满足类型检查，实际上永远不会执行到这里
        if False:
            yield {}


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

