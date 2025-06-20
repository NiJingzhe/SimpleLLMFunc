from abc import ABC, abstractmethod
from typing import Optional, Dict, Iterable, Literal, Any, AsyncGenerator

from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.logger import get_current_trace_id


class LLM_Interface(ABC):

    @abstractmethod
    def __init__(
        self, api_key_pool: APIKeyPool, model_name: str, base_url: Optional[str] = None
    ):
        self.input_token_count = 0
        self.output_token_count = 0

    @abstractmethod
    async def chat(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[False] = False,
        messages: Iterable[Dict[str, str]] = [{"role": "user", "content": ""}],
        timeout: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Dict[Any, Any]:
        pass

    @abstractmethod
    async def chat_stream(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[True] = True,
        messages: Iterable[Dict[str, str]] = [{"role": "user", "content": ""}],
        timeout: Optional[int] = None,
        *args,
        **kwargs,
    ) -> AsyncGenerator[Dict[Any, Any], None]:
        # 空的异步生成器，永远不会产生任何值
        if False:
            yield {}
