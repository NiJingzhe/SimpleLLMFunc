from abc import ABC, abstractmethod
from typing import Generator, Union, Optional, Dict, List, Iterable, Literal, Any

from SimpleLLMGraph.interface.key_pool import APIKeyPool

class LLM_Interface(ABC):
    
    @abstractmethod
    def __init__(self, api_key_pool: APIKeyPool, model_name: str):
        self.input_token_count = 0
        self.output_token_count = 0
    
    @abstractmethod
    def chat(
        self, 
        trace_id: str,
        stream: Literal[False] = False,
        messages: Iterable[Dict[str, str]] = [{"role": "user", "content": ""}],
        timeout: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Dict[Any, Any]:
        pass
    
    @abstractmethod
    def chat_stream(
        self, 
        trace_id: str,
        stream: Literal[True] = True,
        messages: Iterable[Dict[str, str]] = [{"role": "user", "content": ""}],
        timeout: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Generator[Dict[Any, Any], None, None]:
        pass
