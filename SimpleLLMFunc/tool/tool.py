from __future__ import annotations
from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    get_type_hints,
    Type,
    TypeVar,
    get_origin,
    get_args,
)
import re
import inspect
import json
from pydantic import BaseModel

from SimpleLLMFunc.logger.logger import push_error


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
        example: Any = None,
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

    def __init__(self, name: str, description: str, func: Optional[Callable] = None):
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
                example=None,  # 示例值需要单独设置
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
        args_pattern = re.compile(
            r"(?:Args|Parameters):(.*?)(?:\n\n|\n[A-Z]|\Z)", re.DOTALL
        )
        args_match = args_pattern.search(docstring)

        if args_match:
            args_section = args_match.group(1).strip()
            # 匹配参数名和描述
            param_pattern = re.compile(
                r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:(.+?)(?=\n\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:|$)",
                re.MULTILINE | re.DOTALL,
            )
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
        raise NotImplementedError(
            "Subclasses must implement this method or provide a function."
        )

    def _is_optional_type(self, type_annotation: Type) -> bool:
        """
        判断类型是否为Optional[X]或Union[X, None]
        """
        origin = get_origin(type_annotation)
        if origin is Union:
            args = get_args(type_annotation)
            return type(None) in args
        return False

    def _get_inner_type(self, type_annotation: Type) -> Type:
        """
        提取Optional[X]或Union[X, None]中的X
        """
        if self._is_optional_type(type_annotation):
            args = get_args(type_annotation)
            return next(arg for arg in args if arg is not type(None))
        return type_annotation

    def _get_type_schema(self, type_annotation: Type) -> Dict[str, Any]:
        """
        从Python类型生成JSON Schema类型定义

        Args:
            type_annotation: Python类型标注

        Returns:
            对应的JSON Schema类型定义
        """
        #
        # 处理Optional类型
        if self._is_optional_type(type_annotation):
            inner_type = self._get_inner_type(type_annotation)
            schema = self._get_type_schema(inner_type)
            # 允许为null
            if "type" in schema:
                if isinstance(schema["type"], list):
                    if "null" not in schema["type"]:
                        schema["type"].append("null")
                else:
                    schema["type"] = [schema["type"], "null"]
            return schema

        # 基本类型映射
        if type_annotation is str:
            return {"type": "string"}
        elif type_annotation is int:
            return {"type": "integer"}
        elif type_annotation is float:
            return {"type": "number"}
        elif type_annotation is bool:
            return {"type": "boolean"}

        # 处理列表类型
        origin = get_origin(type_annotation)
        args = get_args(type_annotation)

        if origin is list:
            if args:
                return {"type": "array", "items": self._get_type_schema(args[0])}
            return {"type": "array"}

        # 处理字典类型
        if origin is dict:
            if len(args) >= 2:
                return {
                    "type": "object",
                    "additionalProperties": self._get_type_schema(args[1]),
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

            param_schema = {**type_schema, "description": param.description}

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
                "parameters": {"type": "object", "properties": properties},
            },
        }

        # 只有在有必需参数时才添加required字段
        if required_params:
            tool_spec["function"]["parameters"]["required"] = required_params

        return tool_spec

    @staticmethod
    def serialize_tools(tools: List[Tool | Callable]) -> List[Dict[str, Any]]:
        """
        将多个工具序列化为OpenAI工具列表

        Args:
            tools: 要序列化的工具列表

        Returns:
            符合OpenAI Function Calling API格式的工具描述列表
        """
        try:
            result = [
                (
                    tool.to_openai_tool()
                    if isinstance(tool, Tool)
                    else getattr(tool, "_tool").to_openai_tool()
                )
                for tool in tools
            ]
        except AttributeError as e:
            push_error(
                f"传入的工具列表中可能存在非 Tool 类型对象或者没有被 @tool 装饰的函数，序列化发生错误: {e}"
            )
            raise AttributeError(e)
        except Exception as e:
            push_error(f"序列化过程中发生未知错误: {e}")
            raise Exception(e)

        return result


# 工具装饰器函数
T = TypeVar("T")


def tool(name: str, description: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    工具装饰器，用于将函数转换为Tool对象。

    请务必好好写tool函数的DocString，因为这会被作为工具描述信息的一部分。

    工具的描述信息是: `description + "\\n" + docstring`

    ## 一个工具函数支持的传入参数类型：
    - **基本类型**: str, int, float, bool
    - **复合类型**: List[T], Dict[K, V]  
    - **可选类型**: Optional[T], Union[T, None]
    - **Pydantic模型**: 继承自BaseModel的类，会自动解析为JSON Schema
    - **多模态类型**: ImgPath（本地图片）, ImgUrl（网络图片）, Text（文本）

    ## 一个工具函数支持的返回值类型：
    - **基本类型**: str, int, float, bool, dict, list等可序列化类型
    - **多模态返回**: 
      - ImgPath: 返回本地图片路径，用于生成图表、处理后的图片等
      - ImgUrl: 返回网络图片URL，用于搜索到的图片、在线资源等
      - Tuple[str, ImgPath]: 返回说明文本和图片的组合
      - Tuple[str, ImgUrl]: 返回说明文本和网络图片的组合
    - **注意**: 返回值必须是可序列化的，不支持复杂对象

    ## 参数描述解析：
    装饰器会自动解析函数docstring中的Args/Parameters部分，格式为：
    ```
    Args:
        param_name: 参数描述信息
        another_param: 另一个参数的描述
    ```

    Args:
        name: 工具名称，在LLM工具调用中使用
        description: 工具简短描述，更详细的内容可以在被装饰函数的docstring中给出

    Returns:
        装饰器函数，保持原函数功能的同时添加_tool属性

    Example:
        ```python
        from SimpleLLMFunc.tool import tool
        from SimpleLLMFunc.type import ImgPath
        from pydantic import BaseModel, Field

        class Location(BaseModel):
            lat: float = Field(..., description="纬度")
            lng: float = Field(..., description="经度")

        @tool(name="generate_map", description="生成位置地图")
        def generate_map(location: Location, zoom: int = 10) -> ImgPath:
            '''
            根据位置信息生成地图图片
            
            Args:
                location: 位置坐标信息
                zoom: 地图缩放级别，默认为10
                
            Returns:
                生成的地图图片路径
            '''
            # 实现地图生成逻辑
            map_path = "./generated_map.png"
            return ImgPath(map_path)
        ```
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # 创建工具对象
        tool_obj = Tool(name=name, description=description, func=func)

        # 保留原始函数的功能，同时附加工具对象
        setattr(func, "_tool", tool_obj)

        return func

    return decorator


# 测试代码
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
            "forecast": [
                {"day": i, "temp": 25, "condition": "晴朗"} for i in range(days)
            ],
        }

    # 使用装饰器创建带有Optional参数的工具
    @tool(name="search_restaurants", description="搜索附近的餐厅")
    def search_restaurants(
        location: Location,
        cuisine: Optional[str] = None,
        max_distance: Optional[float] = None,
        price_range: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        搜索指定位置附近的餐厅

        Args:
            location: 位置信息，包含经纬度
            cuisine: 菜系类型，如中式、日式等，可选
            max_distance: 最大搜索距离（公里），可选
            price_range: 价格范围（1-4星级），可选

        Returns:
            餐厅搜索结果
        """
        # 实际实现会调用餐厅搜索API
        filters = {}
        if cuisine:
            filters["cuisine"] = cuisine
        if max_distance:
            filters["max_distance"] = max_distance
        if price_range:
            filters["price_range"] = price_range

        return {
            "location": f"{location.latitude},{location.longitude}",
            "filters": filters,
            "restaurants": [{"name": "示例餐厅", "rating": 4.5, "distance": 0.5}],
        }

    # 测试工具
    print("\n装饰器方式 - 基础工具:")
    print(
        json.dumps(
            getattr(get_weather, "_tool").to_openai_tool(), indent=2, ensure_ascii=False
        )
    )

    print("\n装饰器方式 - 带Optional参数的工具:")
    optional_tool_schema = getattr(search_restaurants, "_tool").to_openai_tool()
    print(json.dumps(optional_tool_schema, indent=2, ensure_ascii=False))

    # 验证Optional参数的序列化结果
    print("\n=== Optional参数序列化验证 ===")
    function_params = optional_tool_schema["function"]["parameters"]
    required_params = function_params.get("required", [])
    properties = function_params["properties"]

    print(f"必需参数: {required_params}")
    print("参数详情:")
    for param_name, param_schema in properties.items():
        is_required = param_name in required_params
        param_type = param_schema.get("type")
        print(f"  {param_name}: 类型={param_type}, 必需={is_required}")
        if isinstance(param_type, list) and "null" in param_type:
            print("    -> 支持null值")

    # 也可以直接调用函数
    location = Location(latitude=39.9, longitude=116.3)
    result = get_weather(location, days=3)
    print("\n基础函数调用结果:")
    print(result)

    # 测试Optional参数函数调用
    result_with_optional = search_restaurants(
        location=location, cuisine="中式", max_distance=2.0
    )
    print("\n带Optional参数函数调用结果:")
    print(result_with_optional)

    # 测试完全不传Optional参数
    result_without_optional = search_restaurants(location=location)
    print("\n不传Optional参数函数调用结果:")
    print(result_without_optional)

