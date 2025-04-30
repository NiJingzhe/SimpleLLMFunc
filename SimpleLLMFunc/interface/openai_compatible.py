import json
import time
from typing import Generator, Optional, Dict, List, Literal, Iterable, Any

from openai import OpenAI
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.logger import app_log, push_warning, push_error, get_location, get_current_trace_id

class OpenAICompatible(LLM_Interface):
    """与OpenAI API兼容的LLM接口实现，支持任何符合OpenAI格式的API接口。

    这个类提供了一个通用的接口，可以连接任何兼容OpenAI API格式的大语言模型服务，
    而不需要为每个供应商创建特定的实现。只需要提供正确的base_url和模型名称即可。
    """

    def __init__(
        self,
        api_key_pool: APIKeyPool,
        model_name: str,
        base_url: str,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        allowed_models: Optional[List[str]] = None,
    ):
        """初始化OpenAI兼容的LLM接口

        Args:
            api_key_pool: API密钥池，用于管理和分配API密钥
            model_name: 要使用的模型名称
            base_url: API基础URL，例如"https://api.openai.com/v1"或"https://open.bigmodel.cn/api/paas/v4/"
            max_retries: 最大重试次数
            retry_delay: 重试间隔时间（秒）
            allowed_models: 允许使用的模型列表，如果为None则不检查
        """
        super().__init__(api_key_pool, model_name, base_url)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        self.allowed_models = allowed_models

        # 如果提供了允许的模型列表，检查当前模型是否在列表中
        if self.allowed_models and model_name not in self.allowed_models:
            location = get_location()
            push_warning(
                f"model_name should be one of {self.allowed_models}", location=location
            )
        self.model_name = model_name

        self.key_pool = api_key_pool
        self.client = OpenAI(api_key=api_key_pool.get_least_loaded_key(), base_url=self.base_url)

    def chat(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[False] = False,
        messages: Iterable[Dict[str, str]] = [{"role": "system", "content": "你是一位乐于助人的助手，可以帮助用户解决各种问题。"}],
        timeout: Optional[int] = 30,
        *args,
        **kwargs,
    ) -> Dict[Any, Any]:
        """执行非流式LLM对话请求

        Args:
            trace_id: 跟踪ID，用于日志记录
            stream: 是否使用流式响应，这里必须为False
            messages: 消息历史，包含角色和内容的字典列表
            timeout: 请求超时时间（秒）
            *args, **kwargs: 传递给OpenAI API的其他参数

        Returns:
            LLM的响应内容
        """
        key = self.key_pool.get_least_loaded_key()
        self.client = OpenAI(api_key=key, base_url=self.base_url)
        
        attempt = 0
        while attempt < self.max_retries:
            try:
                self.key_pool.increment_task_count(key)
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                app_log(
                    f"OpenAICompatible::chat: {self.model_name} request with API key: {key}, and message: {data}",
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
        timeout: Optional[int] = 30,
        *args,
        **kwargs,
    ) -> Generator[Dict[Any, Any], None, None]:
        """执行流式LLM对话请求

        Args:
            trace_id: 跟踪ID，用于日志记录
            stream: 是否使用流式响应，这里必须为True
            messages: 消息历史，包含角色和内容的字典列表
            timeout: 请求超时时间（秒）
            *args, **kwargs: 传递给OpenAI API的其他参数

        Yields:
            LLM的响应块
        """
        key = self.key_pool.get_least_loaded_key()
        self.client = OpenAI(api_key=key, base_url=self.base_url)

        attempt = 0
        while attempt < self.max_retries:
            try:
                self.key_pool.increment_task_count(key)
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                app_log(
                    f"OpenAICompatible::chat_stream: {self.model_name} request with API key: {key}, and message: {data}",
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
                location = get_location()
                data = json.dumps(messages, ensure_ascii=False, indent=4)
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