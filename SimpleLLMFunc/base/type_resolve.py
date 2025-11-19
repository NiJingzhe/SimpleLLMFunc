"""Type resolution helpers for LLM decorators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from SimpleLLMFunc.logger import push_error
from SimpleLLMFunc.logger.logger import get_location


def _is_primitive_type(type_hint: Any) -> bool:
    """Check if a type hint is a primitive type (str, int, float, bool, None)."""
    if type_hint is None or type_hint is type(None):
        return True
    return type_hint in (str, int, float, bool, type(None))


def get_detailed_type_description(type_hint: Any) -> str:
    """Generate a human-readable description for a type hint."""

    if type_hint is None:
        return "未知类型"

    # For complex types (BaseModel, List, Dict, Union), prefer structured JSON description
    try:
        from typing import get_origin, get_args, Union as TypingUnion
        import json as _json

        # Check if it's a BaseModel
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            desc_obj = build_type_description_json(type_hint)
            return _json.dumps(desc_obj, ensure_ascii=False, indent=2)

        # Check for List, Dict, Union
        origin = getattr(type_hint, "__origin__", None) or get_origin(type_hint)
        if origin in (list, List, dict, Dict, TypingUnion):
            desc_obj = build_type_description_json(type_hint)
            return _json.dumps(desc_obj, ensure_ascii=False, indent=2)
    except Exception:
        # Fallback to old format on error
        pass

    # Fallback: simple type descriptions
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
                    content.append(
                        create_image_path_content(item, f"{param_name}[{i}]")
                    )
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


# ===== New: Structured JSON description and example generation =====


def build_type_description_json(
    type_hint: Any,
    depth: int = 0,
    max_depth: int = 5,
    seen: Optional[set] = None,
) -> Dict[str, Any]:
    """Build a structured JSON-like description for a type hint (recursive).

    - Fully expands nested BaseModel, List, Dict, and Union (excluding NoneType)
    - Guards against cycles and excessive depth
    """
    from typing import get_origin, get_args, Union as TypingUnion

    if seen is None:
        seen = set()

    def _json_type_name(py_type: Any) -> str:
        mapping: Dict[Any, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            type(None): "null",
        }
        return mapping.get(py_type, getattr(py_type, "__name__", str(py_type)))

    if depth > max_depth:
        name = getattr(type_hint, "__name__", str(type_hint))
        return {"type": "object", "title": name, "note": "depth_limit"}

    # Pydantic model - check first before get_origin
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        type_id = ("model", type_hint)
        if type_id in seen:
            return {
                "type": "object",
                "title": type_hint.__name__,
                "note": "circular_ref",
            }
        seen.add(type_id)

        schema = type_hint.model_json_schema()
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        result: Dict[str, Any] = {
            "type": "object",
            "title": type_hint.__name__,
            "required": required,
            "properties": {},
        }

        # Use model_fields to get accurate annotations for recursion
        model_fields = getattr(type_hint, "model_fields", {})
        for field_name, field_info in properties.items():
            # Recurse by field annotation when available
            field_ann = None
            if field_name in model_fields:
                field_ann = getattr(model_fields[field_name], "annotation", None)
            child_desc = (
                build_type_description_json(field_ann, depth + 1, max_depth, seen)
                if field_ann is not None
                else {
                    k: v
                    for k, v in field_info.items()
                    if k in {"type", "description", "minimum", "maximum", "default"}
                }
            )
            # Enrich with simple constraints from schema
            if "type" not in child_desc and "type" in field_info:
                child_desc["type"] = field_info["type"]
            if "description" in field_info and "description" not in child_desc:
                child_desc["description"] = field_info["description"]
            if "minimum" in field_info:
                child_desc.setdefault("minimum", field_info["minimum"])
            if "maximum" in field_info:
                child_desc.setdefault("maximum", field_info["maximum"])
            if "default" in field_info:
                child_desc.setdefault("default", field_info["default"])

            result["properties"][field_name] = child_desc

        return result

    # Get origin after BaseModel check
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # List / sequence
    if origin in (list, List):
        items_type = args[0] if args else Any
        return {
            "type": "array",
            "items": build_type_description_json(
                items_type, depth + 1, max_depth, seen
            ),
        }

    # Dict mapping
    if origin in (dict, Dict):
        value_type = args[1] if len(args) >= 2 else Any
        return {
            "type": "object",
            "additionalProperties": build_type_description_json(
                value_type, depth + 1, max_depth, seen
            ),
        }

    # Union / Optional
    if origin is TypingUnion:
        non_none = [t for t in args if t is not type(None)]
        if len(non_none) == 1:
            return build_type_description_json(non_none[0], depth + 1, max_depth, seen)
        return {
            "anyOf": [
                build_type_description_json(t, depth + 1, max_depth, seen)
                for t in non_none
            ]
        }

    # Simple / builtin types
    return {"type": _json_type_name(type_hint)}


def _generate_primitive_example(type_hint: Any) -> Any:
    """Generate example value for primitive types directly.

    Also handles Optional[T] by extracting the inner type.
    """
    from typing import get_origin, get_args, Union as TypingUnion

    # Handle Optional[T] / Union[T, None]
    origin = get_origin(type_hint)
    if origin is TypingUnion:
        args = get_args(type_hint)
        # Extract first non-None type from Union
        for t in args:
            if t is not type(None):
                return _generate_primitive_example(t)  # Recursively check inner type

    # Primitive types
    if type_hint is str:
        return "example"
    if type_hint is int:
        return 123
    if type_hint is float:
        return 1.23
    if type_hint is bool:
        return True
    if type_hint is type(None):
        return None

    return None  # Not a primitive, need recursive handling


def generate_example_object(
    type_hint: Any,
    depth: int = 0,
    max_depth: int = 5,
    seen: Optional[set] = None,
) -> Any:
    """Generate an example object for the given type hint (recursive)."""
    from typing import get_origin, get_args, Union as TypingUnion

    if seen is None:
        seen = set()

    if depth > max_depth:
        return {}

    # BaseModel - check first before get_origin
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        type_id = ("model", type_hint)
        if type_id in seen:
            return {}
        seen.add(type_id)

        example: Dict[str, Any] = {}
        model_fields = getattr(type_hint, "model_fields", {})

        # Check for PydanticUndefined to avoid JSON serialization issues
        try:
            from pydantic import PydanticUndefined
        except ImportError:
            # Pydantic v1 or older versions don't have PydanticUndefined
            PydanticUndefined = type("PydanticUndefined", (), {})

        for field_name, field in model_fields.items():
            ann = getattr(field, "annotation", Any)
            default = getattr(field, "default", ...)

            # Check if default is valid (not ... and not PydanticUndefined)
            has_default = (
                default is not ...
                and default is not PydanticUndefined
                and not (
                    hasattr(type(default), "__name__")
                    and "PydanticUndefined" in str(type(default))
                )
            )

            if has_default:
                # Try to serialize default to ensure it's JSON-compatible
                try:
                    import json as _json_test

                    _json_test.dumps(default)  # Test serialization
                    example[field_name] = default
                except (TypeError, ValueError):
                    # If default is not JSON-serializable, generate example based on type
                    primitive_example = _generate_primitive_example(ann)
                    if primitive_example is not None:
                        example[field_name] = primitive_example
                    else:
                        # For complex types, use recursive generation
                        example[field_name] = generate_example_object(
                            ann, depth + 1, max_depth, seen
                        )
            else:
                # No default value: generate example based on type annotation
                # Try primitive types first for better performance
                primitive_example = _generate_primitive_example(ann)
                if primitive_example is not None:
                    example[field_name] = primitive_example
                else:
                    # For complex types (List, Dict, Union, nested BaseModel), use recursive generation
                    example[field_name] = generate_example_object(
                        ann, depth + 1, max_depth, seen
                    )
        return example

    # Get origin after BaseModel check
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # List
    if origin in (list, List):
        item_t = args[0] if args else Any
        return [generate_example_object(item_t, depth + 1, max_depth, seen)]

    # Dict
    if origin in (dict, Dict):
        key_t = args[0] if args else str
        val_t = args[1] if len(args) >= 2 else Any
        key_example = "key" if key_t in (str, Any) else (1 if key_t is int else "key")
        return {key_example: generate_example_object(val_t, depth + 1, max_depth, seen)}

    # Union / Optional: pick first non-None
    if origin is TypingUnion:
        for t in args:
            if t is not type(None):
                return generate_example_object(t, depth + 1, max_depth, seen)
        return None

    # Scalars
    if type_hint in (str, Any):
        return "example"
    if type_hint is int:
        return 123
    if type_hint is float:
        return 1.23
    if type_hint is bool:
        return True
    if type_hint is type(None):
        return None

    # Unknown types → string representation as placeholder
    return "example"


__all__ = [
    "get_detailed_type_description",
    "has_multimodal_content",
    "is_multimodal_type",
    "handle_union_type",
    "build_type_description_json",
    "generate_example_object",
]
