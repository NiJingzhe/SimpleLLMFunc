"""Type description generation helpers."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel


def get_detailed_type_description(type_hint: Any) -> str:
    """Generate a human-readable description for a type hint."""

    if type_hint is None:
        return "未知类型"

    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        return describe_pydantic_model(type_hint)

    origin = getattr(type_hint, "__origin__", None)
    if origin is list or origin is List:
        from typing import List

        args = getattr(type_hint, "__args__", [])
        if args:
            item_type_desc = get_detailed_type_description(args[0])
            return f"List[{item_type_desc}]"
        return "List"

    if origin is dict or origin is Dict:
        from typing import Dict

        args = getattr(type_hint, "__args__", [])
        if len(args) >= 2:
            key_type_desc = get_detailed_type_description(args[0])
            value_type_desc = get_detailed_type_description(args[1])
            return f"Dict[{key_type_desc}, {value_type_desc}]"
        return "Dict"

    return str(type_hint)


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

