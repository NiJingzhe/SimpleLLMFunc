from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, get_type_hints, Type, TypeVar, get_origin, get_args
import re
import inspect
import json
from functools import wraps
from pydantic import BaseModel

class Parameter:
    """
    工具参数的简单包装类，仅用于存储信息，不作为主要API
    """
    def __init__(
        self, 
        name: str, 
        description: str, 
        type_annotation: Type,
        required: bool, 
        default: Any = None, 
        example: Any = None
    ):
        self.name = name
        self.description = description
        self.type_annotation = type_annotation  # 存储原生Python类型
        self.required = required
        self.default = default
        self.example = example


class Tool(ABC):
    """
    抽象工具基类，可以通过两种方式创建：
    1. 通过子类继承并实现run方法（支持向后兼容）
    2. 通过@tool装饰器装饰一个函数（推荐方式）
    """

    def __init__(
        self, 
        name: str, 
        description: str,
        func: Optional[Callable] = None
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = self._extract_parameters() if func else []

    def _extract_parameters(self) -> List[Parameter]:
        """
        从函数签名中提取参数信息
        
        Returns:
            参数列表
        """
        if not self.func:
            return []
            
        signature = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        docstring = inspect.getdoc(self.func) or ""
        
        # 尝试解析函数文档字符串中的参数描述
        param_descriptions = self._parse_docstring_params(docstring)
        
        parameters = []
        
        for param_name, param in signature.parameters.items():
            # 跳过self参数
            if param_name == "self":
                continue
                
            # 获取参数类型
            param_type = type_hints.get(param_name, Any)
            
            # 确定参数是否必需
            required = param.default == inspect.Parameter.empty
            
            # 获取默认值
            default = None if required else param.default
            
            # 获取参数描述
            description = param_descriptions.get(param_name, f"Parameter {param_name}")
            
            # 创建参数对象
            param_obj = Parameter(
                name=param_name,
                description=description,
                type_annotation=param_type,
                required=required,
                default=default,
                example=None  # 示例值需要单独设置
            )
            
            parameters.append(param_obj)
            
        return parameters

    def _parse_docstring_params(self, docstring: str) -> Dict[str, str]:
        """
        解析docstring中的参数描述
        
        Args:
            docstring: 函数的文档字符串
            
        Returns:
            参数名到描述的映射
        """
        param_descriptions = {}
        
        # 查找Args部分
        args_pattern = re.compile(r"(?:Args|Parameters):(.*?)(?:\n\n|\n[A-Z]|\Z)", re.DOTALL)
        args_match = args_pattern.search(docstring)
        
        if args_match:
            args_section = args_match.group(1).strip()
            # 匹配参数名和描述
            param_pattern = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:(.+?)(?=\n\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:|$)", re.MULTILINE | re.DOTALL)
            for match in param_pattern.finditer(args_section):
                param_name, description = match.groups()
                param_descriptions[param_name.strip()] = description.strip()
                
        return param_descriptions

    def run(self, *args, **kwargs):
        """
        运行工具。如果提供了函数，则调用该函数；否则调用子类实现的run方法。
        """
        if self.func is not None:
            return self.func(*args, **kwargs)
        
        # 默认实现，子类应该重写这个方法
        raise NotImplementedError("Subclasses must implement this method or provide a function.")

    def _get_type_schema(self, type_annotation: Type) -> Dict[str, Any]:
        """
        从Python类型生成JSON Schema类型定义
        
        Args:
            type_annotation: Python类型标注
            
        Returns:
            对应的JSON Schema类型定义
        """
        # 基本类型映射
        if type_annotation == str:
            return {"type": "string"}
        elif type_annotation == int:
            return {"type": "integer"}
        elif type_annotation == float:
            return {"type": "number"}
        elif type_annotation == bool:
            return {"type": "boolean"}
        
        # 处理列表类型
        origin = get_origin(type_annotation)
        args = get_args(type_annotation)
        
        if origin == list or origin == List:
            if args:
                return {
                    "type": "array",
                    "items": self._get_type_schema(args[0])
                }
            return {"type": "array"}
        
        # 处理字典类型
        if origin == dict or origin == Dict:
            if len(args) >= 2:
                return {
                    "type": "object",
                    "additionalProperties": self._get_type_schema(args[1])
                }
            return {"type": "object"}
        
        # 处理Pydantic模型
        if isinstance(type_annotation, type) and issubclass(type_annotation, BaseModel):
            # 使用Pydantic的model_json_schema方法获取模型的JSON Schema
            schema = type_annotation.model_json_schema()
            return {"type": "object", "properties": schema.get("properties", {})}
        
        # 默认为字符串类型
        return {"type": "string"}
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """
        序列化工具为OpenAI工具格式
        
        Returns:
            符合OpenAI Function Calling API格式的工具描述字典
        """
        properties = {}
        required_params = []
        
        for param in self.parameters:
            # 获取类型的JSON Schema表示
            type_schema = self._get_type_schema(param.type_annotation)
            
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
            
            # 如果参数是必需的，添加到required列表
            if param.required:
                required_params.append(param.name)
        
        # 构建符合OpenAI格式的工具描述
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
        
        # 只有在有必需参数时才添加required字段
        if required_params:
            tool_spec["function"]["parameters"]["required"] = required_params
            
        return tool_spec
    
    @staticmethod
    def serialize_tools(tools: List["Tool"]) -> List[Dict[str, Any]]:
        """
        将多个工具序列化为OpenAI工具列表
        
        Args:
            tools: 要序列化的工具列表
            
        Returns:
            符合OpenAI Function Calling API格式的工具描述列表
        """
        return [tool.to_openai_tool() for tool in tools]


# 工具装饰器函数
T = TypeVar('T')

def tool(name: str, description: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    工具装饰器，用于将函数转换为Tool对象
    
    Args:
        name: 工具名称
        description: 工具描述
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # 创建工具对象
        tool_obj = Tool(name=name, description=description, func=func)
        
        # 保留原始函数的功能，同时附加工具对象
        func._tool = tool_obj
        
        return func
    
    return decorator


# 测试代码
if __name__ == "__main__":
    from pydantic import BaseModel, Field
    
    # 示例Pydantic模型
    class Location(BaseModel):
        latitude: float = Field(..., description="纬度")
        longitude: float = Field(..., description="经度")
    
    # 使用装饰器创建工具
    @tool(name="get_weather", description="获取指定位置的天气信息")
    def get_weather(location: Location, days: int = 1) -> Dict[str, Any]:
        """
        获取指定位置的天气预报
        
        Args:
            location: 位置信息，包含经纬度
            days: 预报天数，默认为1天
            
        Returns:
            天气预报信息
        """
        # 实际实现会调用天气API
        return {
            "location": f"{location.latitude},{location.longitude}",
            "forecast": [{"day": i, "temp": 25, "condition": "晴朗"} for i in range(days)]
        }
    
    # 测试工具
    print("\n装饰器方式:")
    print(json.dumps(get_weather._tool.to_openai_tool(), indent=2, ensure_ascii=False))
    
    # 也可以直接调用函数
    location = Location(latitude=39.9, longitude=116.3)
    result = get_weather(location, days=3)
    print("\n函数调用结果:")
    print(result)