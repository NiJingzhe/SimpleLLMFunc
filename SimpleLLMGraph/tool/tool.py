from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import re

from .schemas import ToolParameters, ParameterType

class Tool(ABC):
    """
    Abstract base class for a tool.
    """

    def __init__(
        self, 
        name: str, 
        description: str,
        parameters: List[ToolParameters]
    ):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the tool with the given arguments.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _parse_parameter_type(self, type_str: str) -> Dict[str, Any]:
        """
        解析参数类型字符串，支持嵌套类型
        
        Args:
            type_str: 参数类型字符串，如 "string", "list<string>", "dict<string,integer>"
            
        Returns:
            对应的 JSON Schema 类型定义
        """
        # 简单类型直接映射
        simple_types = {
            ParameterType.STRING: {"type": "string"},
            ParameterType.INTEGER: {"type": "integer"},
            ParameterType.FLOAT: {"type": "number"},
            ParameterType.BOOLEAN: {"type": "boolean"}
        }
        
        if type_str in [str(param_type) for param_type in simple_types.keys()]:
            # Convert the string to the corresponding ParameterType enum value
            for param_type in simple_types.keys():
                if str(param_type) == type_str:
                    return simple_types[param_type]
        
        # 处理嵌套类型
        list_match = re.match(r"list<(.+)>", type_str)
        if list_match:
            item_type = list_match.group(1)
            return {
                "type": "array",
                "items": self._parse_parameter_type(item_type)
            }
        
        dict_match = re.match(r"dict<(.+),(.+)>", type_str)
        if dict_match:
            # OpenAI 工具格式不直接支持字典类型的键类型规范
            # 所以我们只使用值类型，并在描述中注明键类型
            value_type = dict_match.group(2)
            return {
                "type": "object",
                "additionalProperties": self._parse_parameter_type(value_type)
            }
        
        # 如果无法识别类型，默认为字符串
        return {"type": "string"}
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """
        序列化工具为 OpenAI 工具格式
        
        Returns:
            符合 OpenAI Function Calling API 格式的工具描述字典
        """
        properties = {}
        required_params = []
        
        for param in self.parameters:
            # 解析参数类型
            param_type = str(param.type)  # 确保获取枚举的字符串值
            type_schema = self._parse_parameter_type(param_type)
            
            param_schema = {
                **type_schema,
                "description": param.description
            }
            
            # 添加示例值
            if param.example is not None:
                param_schema["example"] = param.example
                
            # 处理默认值
            if param.default is not None:
                param_schema["default"] = param.default
                
            properties[param.name] = param_schema
            
            # 如果参数是必需的，添加到 required 列表
            if param.required:
                required_params.append(param.name)
        
        # 构建符合 OpenAI 格式的工具描述
        tool_spec = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties
                }
            }
        }
        
        # 只有在有必需参数时才添加 required 字段
        if required_params:
            tool_spec["function"]["parameters"]["required"] = required_params
            
        return tool_spec
    
    @staticmethod
    def serialize_tools(tools: List["Tool"]) -> List[Dict[str, Any]]:
        """
        将多个工具序列化为 OpenAI 工具列表
        
        Args:
            tools: 要序列化的工具列表
            
        Returns:
            符合 OpenAI Function Calling API 格式的工具描述列表
        """
        return [tool.to_openai_tool() for tool in tools]
    
    
# main function for test
if __name__ == "__main__":
    # 测试工具类
    class ExampleTool(Tool):
        def run(self, param1, param2=None):
            """
            Implement the abstract run method
            """
            return f"Running with param1={param1}, param2={param2}"
    
    tool = ExampleTool(
        name="example_tool",
        description="This is an example tool.",
        parameters=[
            ToolParameters(
                name="param1",
                description="An integer parameter.",
                type=ParameterType.INTEGER,
                required=True,
                default=0,
                example=42
            ),
            ToolParameters(
                name="param2",
                description="A string parameter.",
                type=ParameterType.STRING,
                required=False,
                default="default_value",
                example="example_value"
            )
        ]
    )
    
    print(tool.to_openai_tool())