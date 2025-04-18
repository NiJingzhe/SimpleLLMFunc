from abc import ABC, abstractmethod
from SimpleAgent.tool import Tool, ToolParameters
from typing import List, Dict, Any, override

from SimpleAgent.agent.schemas import AgentMemoryItem

class Agent(ABC):
    
    def __init__(
        self, 
        name: str,
        description: str,
        toolkit: List[Tool],
        one_sentence_target: str,
    ):
        
        self.name = name                                 # Agent名称
        self.description = description                   # Agent描述
        self.toolkit = toolkit                           # Agent工具列表
        self.one_sentence_target = one_sentence_target   # 一句话描述Agent目标
        
        self.memory: List[AgentMemoryItem]