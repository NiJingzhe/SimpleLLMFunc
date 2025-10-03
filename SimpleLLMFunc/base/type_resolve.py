"""Type resolution helpers for LLM decorators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from SimpleLLMFunc.logger import push_error
from SimpleLLMFunc.logger.logger import get_location


def get_detailed_type_description(type_hint: Any) -> str:
    """Generate a human-readable description for a type hint."""

    if type_hint is None:
        return "未知类型"

    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        return describe_pydantic_model(type_hint)

    origin = getattr(type_hint, "__origin__", None)
    if origin is list or origin is List:
        args = getattr(type_hint, "__args__", [])
        if args:
            item_type_desc = get_detailed_type_description(args[0])
            return f"List[{item_type_desc}]"
        return "List"

    if origin is dict or origin is Dict:
        args = getattr(type_hint, "__args__", [])
        if len(args) >= 2:
            key_type_desc = get_detailed_type_description(args[0])
            value_type_desc = get_detailed_type_description(args[1])
            return f"Dict[{key_type_desc}, {value_type_desc}]"
        return "Dict"

    return str(type_hint)


def has_multimodal_content(
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    exclude_params: Optional[List[str]] = None,
) -> bool:
    """Check whether arguments contain multimodal payloads."""

    exclude_params = exclude_params or []

    for param_name, param_value in arguments.items():
        if param_name in exclude_params:
            continue

        if param_name in type_hints:
            annotation = type_hints[param_name]
            if is_multimodal_type(param_value, annotation):
                return True
    return False


def is_multimodal_type(value: Any, annotation: Any) -> bool:
    """Determine whether a value/annotation pair represents multimodal content."""

    from typing import List as TypingList, Union, get_args, get_origin

    from SimpleLLMFunc.llm_decorator.multimodal_types import ImgPath, ImgUrl, Text

    if isinstance(value, (Text, ImgUrl, ImgPath)):
        return True

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        for arg_type in non_none_args:
            if is_multimodal_type(value, arg_type):
                return True
        return False

    if origin in (list, TypingList):
        if not args:
            return False
        element_type = args[0]
        if element_type in (Text, ImgUrl, ImgPath):
            return True
        if isinstance(value, (list, tuple)):
            return any(isinstance(item, (Text, ImgUrl, ImgPath)) for item in value)
        return False

    if annotation in (Text, ImgUrl, ImgPath):
        return True

    return False


def handle_union_type(value: Any, args: tuple, param_name: str) -> List[Dict[str, Any]]:
    """Handle Union annotations containing multimodal payload combinations."""

    from SimpleLLMFunc.llm_decorator.multimodal_types import ImgPath, ImgUrl, Text

    content: List[Dict[str, Any]] = []

    if isinstance(value, (Text, ImgUrl, ImgPath, str)):
        from SimpleLLMFunc.base.messages import (
            create_image_path_content,
            create_image_url_content,
            create_text_content,
        )

        if isinstance(value, (Text, str)):
            content.append(create_text_content(value, param_name))
        elif isinstance(value, ImgUrl):
            content.append(create_image_url_content(value, param_name))
        elif isinstance(value, ImgPath):
            content.append(create_image_path_content(value, param_name))
        return content

    if isinstance(value, (list, tuple)):
        from SimpleLLMFunc.base.messages import (
            create_image_path_content,
            create_image_url_content,
            create_text_content,
        )

        for i, item in enumerate(value):
            if isinstance(item, (Text, ImgUrl, ImgPath, str)):
                if isinstance(item, (Text, str)):
                    content.append(create_text_content(item, f"{param_name}[{i}]"))
                elif isinstance(item, ImgUrl):
                    content.append(create_image_url_content(item, f"{param_name}[{i}]"))
                elif isinstance(item, ImgPath):
                    content.append(create_image_path_content(item, f"{param_name}[{i}]"))
            else:
                push_error(
                    "多模态参数只能被标注为Optional[List[Text/ImgUrl/ImgPath]] 或 Optional[Text/ImgUrl/ImgPath] 或 List[Text/ImgUrl/ImgPath] 或 Text/ImgUrl/ImgPath",
                    location=get_location(),
                )
                from SimpleLLMFunc.base.messages import create_text_content

                content.append(create_text_content(item, f"{param_name}[{i}]"))
        return content

    from SimpleLLMFunc.base.messages import create_text_content

    return [create_text_content(value, param_name)]


def describe_pydantic_model(model_class: Type[BaseModel]) -> str:
    """Expand a Pydantic model to a descriptive summary."""

    model_name = model_class.__name__
    schema = model_class.model_json_schema()

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields_desc = []
    for field_name, field_info in properties.items():
        field_type = field_info.get("type", "unknown")
        field_desc = field_info.get("description", "")
        is_required = field_name in required

        req_marker = "必填" if is_required else "可选"

        extra_info = ""
        if "minimum" in field_info:
            extra_info += f", 最小值: {field_info['minimum']}"
        if "maximum" in field_info:
            extra_info += f", 最大值: {field_info['maximum']}"
        if "default" in field_info:
            extra_info += f", 默认值: {field_info['default']}"

        fields_desc.append(
            f"  - {field_name} ({field_type}, {req_marker}): {field_desc}{extra_info}"
        )

    model_desc = f"{model_name} (Pydantic模型) 包含以下字段:\n" + "\n".join(fields_desc)
    return model_desc


__all__ = [
    "get_detailed_type_description",
    "has_multimodal_content",
    "is_multimodal_type",
    "handle_union_type",
]
