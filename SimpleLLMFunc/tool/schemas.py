from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Union, List, Dict, Any, Type, get_origin, get_args

class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"

    @classmethod
    def nested_type(cls, base_type: str, sub_type: Optional[Union[str, Dict[str, str]]] = None) -> str:
        """
        创建嵌套类型表达式
        
        Args:
            base_type: 基础类型 (如 "list", "dict")
            sub_type: 子类型，可以是单一类型或字典类型 (用于dict的键值类型)
            
        Returns:
            嵌套类型的字符串表示
        """
        if sub_type is None:
            return base_type
        
        if base_type == cls.LIST:
            return f"{base_type}<{sub_type}>"
        elif base_type == cls.DICT:
            if isinstance(sub_type, dict):
                key_type = sub_type.get("key", cls.STRING)
                value_type = sub_type.get("value", cls.STRING)
                return f"{base_type}<{key_type},{value_type}>"
            else:
                # 默认键类型为字符串
                return f"{base_type}<string,{sub_type}>"
        return base_type

class ToolParameters(BaseModel):
    name: str = Field(..., description="The name of the tool.")
    description: str = Field(..., description="The description of the tool.")
    type: ParameterType = Field(..., description="The type of the tool.")
    required: bool = Field(..., description="Whether the parameter is required.")
    default: Optional[Any] = Field(None, description="The default value of the parameter.")
    example: Optional[Any] = Field(None, description="An example value for the parameter.")


def get_parameter_type_from_annotation(annotation: Type) -> ParameterType:
    """
    从类型标注中获取对应的ParameterType
    
    Args:
        annotation: Python类型标注
        
    Returns:
        对应的ParameterType枚举值
    """
    # 处理基本类型
    if annotation == str:
        return ParameterType.STRING
    elif annotation == int:
        return ParameterType.INTEGER
    elif annotation == float:
        return ParameterType.FLOAT
    elif annotation == bool:
        return ParameterType.BOOLEAN
    
    # 处理泛型类型
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    if origin == list or origin == List:
        # 列表类型
        if args:
            # 提取子类型，但目前返回的是嵌套类型的字符串表示
            item_type = get_parameter_type_from_annotation(args[0])
            return ParameterType.nested_type(ParameterType.LIST, str(item_type))
        return ParameterType.LIST
    
    elif origin == dict or origin == Dict:
        # 字典类型
        if len(args) >= 2:
            key_type = get_parameter_type_from_annotation(args[0])
            value_type = get_parameter_type_from_annotation(args[1])
            return ParameterType.nested_type(
                ParameterType.DICT, 
                {"key": str(key_type), "value": str(value_type)}
            )
        return ParameterType.DICT
    
    # 默认返回字符串类型
    return ParameterType.STRING


