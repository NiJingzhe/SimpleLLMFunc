from __future__ import annotations
import json
import os
import time
from typing import Generator, Optional, Dict, List, Literal, Iterable, Any

from openai import OpenAI, AsyncOpenAI
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.logger import app_log, push_warning, push_error, get_location, get_current_trace_id
from SimpleLLMFunc.logger.logger import push_critical, get_current_context_attribute, set_current_context_attribute

class OpenAICompatible(LLM_Interface):
    """与OpenAI API兼容的LLM接口实现，支持任何符合OpenAI格式的API接口。

    这个类提供了一个通用的接口，可以连接任何兼容OpenAI API格式的大语言模型服务，
    而不需要为每个供应商创建特定的实现。只需要提供正确的base_url和模型名称即可。
    """

    def _count_tokens(self, response: Any) -> tuple[int, int]:
        """计算响应中的token数量
        
        Args:
            response: OpenAI API的响应对象
            
        Returns:
            (输入token数, 输出token数)的元组
        """
        try:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            return prompt_tokens, completion_tokens
        except (AttributeError, TypeError):
            # 如果无法获取token计数,返回0
            return 0, 0

    @classmethod
    def load_from_json_file(cls, json_path: str) -> Dict[str, Dict[str, OpenAICompatible]]:
        """从JSON字符串加载OpenAICompatible实例

        Args:
            json_str: JSON字符串，包含API密钥和模型名称
            
            例如:
            ```
            {
                "openai": [
                    {
                        "model_name": "gpt-3.5-turbo",
                        "api_keys": [key1, key2, key3],
                        "base_url": "https://api.openai.com/v1"
                        "max_retries": 5,
                        "retry_delay": 1.0
                    },
                    {
                        "model_name": "gpt-4",
                        "api_keys": [key1, key2, key3],
                        "base_url": "https://api.openai.com/v1"
                        "max_retries": 5,
                        "retry_delay": 1.0
                    }
                ],
                "zhipu": [
                    {
                        "model_name": "gpt-3.5-turbo",
                        "api_keys": [key1, key2, key3],
                        "base_url": "https://open.bigmodel.cn/api/paas/v4/"
                        "max_retries": 5,
                        "retry_delay": 1.0
                    },
                    {
                        "model_name": "gpt-4",
                        "api_keys": [key1, key2, key3],
                        "base_url": "https://open.bigmodel.cn/api/paas/v4/"
                        "max_retries": 5,
                        "retry_delay": 1.0
                    }
                ]    
            }
            ```

        Returns:
            OpenAICompatible实例的字典, 可以这样访问： 
            ```python
            from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible

            all_models = OpenAICompatible.load_from_json(json_str)
            gpt_3_5 = all_models["openai"]["gpt-3.5-turbo"]
            gpt_4 = all_models["openai"]["gpt-4"]
            ``` 
        """
        
        if not os.path.exists(json_path):
            push_critical(f"JSON 文件 {json_path} 不存在。请检查您的配置。", location=get_location())
            raise FileNotFoundError(f"JSON 文件 {json_path} 不存在。")
        
        with open(json_path, "r", encoding="utf-8") as f:
            json_str = f.read()
        # 解析JSON字符串
        try:
            all_providers_info = json.loads(json_str)
        except json.JSONDecodeError as e:
            push_critical(f"解析 JSON 字符串失败：{e}", location=get_location())
            raise ValueError(f"解析 JSON 字符串失败：{e}")
        # 检查JSON格式
        
        try: 
            all_providers_dict: Dict[str, Dict[str, OpenAICompatible]] = {}
            for provider_id, models in all_providers_info.items():
                all_providers_dict[provider_id] = {}
                app_log(
                    f"正在为提供商加载 OpenAICompatible 实例：{provider_id}",
                    location=get_location()
                )
                
                if not isinstance(models, list):
                    push_critical(f"提供商 {provider_id} 下的模型格式无效。应为列表。", location=get_location())
                    raise TypeError(f"提供商 {provider_id} 下的模型格式无效。应为列表。")
                
                for model_info in models:
                    model_name = model_info["model_name"]
                    api_keys = model_info["api_keys"]
                    base_url = model_info["base_url"]
                    max_retries = model_info.get("max_retries", 5)
                    retry_delay = model_info.get("retry_delay", 1.0)

                    # 创建APIKeyPool实例
                    key_pool = APIKeyPool(api_keys, f"{provider_id}-{model_name}")

                    # 创建OpenAICompatible实例
                    instance = OpenAICompatible(
                        api_key_pool=key_pool,
                        model_name=model_name,
                        base_url=base_url,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                    )

                    all_providers_dict[provider_id][model_name] = instance
                    
                    app_log(
                        f"已为提供商 {provider_id} 加载模型 {model_name} 的 OpenAICompatible 实例",
                        location=get_location()
                    )
        except ValueError as e:
            push_critical(f"加载 OpenAICompatible 实例时出错：{e}", location=get_location())
            raise ValueError(f"加载 OpenAICompatible 实例时出错：{e}")
        except TypeError as e:
            push_critical(f"JSON 中的类型无效：{e}", location=get_location())
            raise ValueError(f"JSON 中的类型无效：{e}")
        except KeyError as e:
            push_critical(f"JSON 中缺少必需的密钥：{e}", location=get_location())
            raise ValueError(f"JSON 中缺少必需的密钥：{e}")
        except Exception as e:
            push_critical(f"加载 OpenAICompatible 实例时发生未知错误：{e}", location=get_location())
            raise ValueError(f"加载 OpenAICompatible 实例时发生未知错误：{e}")
        
        return all_providers_dict

    def __repr__(self) -> str:
        """返回OpenAICompatible实例的字符串表示"""
        return f"OpenAICompatible(model_name={self.model_name}, base_url={self.base_url})"

    def __init__(
        self,
        api_key_pool: APIKeyPool,
        model_name: str,
        base_url: str,
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ):
        """初始化OpenAI兼容的LLM接口

        Args:
            api_key_pool: API密钥池，用于管理和分配API密钥
            model_name: 要使用的模型名称
            base_url: API基础URL，例如"https://api.openai.com/v1"或"https://open.bigmodel.cn/api/paas/v4/"
            max_retries: 最大重试次数
            retry_delay: 重试间隔时间（秒）
        """
        super().__init__(api_key_pool, model_name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url

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

                # 统计token
                if not (response.choices and response.choices[0].message and response.choices[0].message.tool_calls):  # type: ignore
                    prompt_tokens, completion_tokens = self._count_tokens(response)
                    
                    # 更新上下文中的token计数
                    input_tokens = get_current_context_attribute("input_tokens") or 0
                    output_tokens = get_current_context_attribute("output_tokens") or 0
                    
                    set_current_context_attribute("input_tokens", input_tokens + prompt_tokens)
                    set_current_context_attribute("output_tokens", output_tokens + completion_tokens)
                
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

                total_prompt_tokens = 0
                total_completion_tokens = 0

                for chunk in response:
                    yield chunk  # 按块返回生成器中的数据
                    if chunk.choices and chunk.choices[0].delta:  # type: ignore
                        if not chunk.choices[0].delta.tool_calls:  # type: ignore
                            prompt_tokens, completion_tokens = self._count_tokens(chunk)
                            total_prompt_tokens += prompt_tokens
                            total_completion_tokens += completion_tokens

                # 在整个流结束后统计token
                input_tokens = get_current_context_attribute("input_tokens") or 0
                output_tokens = get_current_context_attribute("output_tokens") or 0

                set_current_context_attribute("input_tokens", input_tokens + total_prompt_tokens)
                set_current_context_attribute("output_tokens", output_tokens + total_completion_tokens)

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