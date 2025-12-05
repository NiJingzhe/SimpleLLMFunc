# SimpleLLMFunc 支持的输入和返回类型

本文档详细列出了 `llm_function`、`llm_chat` 和 `@tool` 装饰器支持的输入参数类型和返回参数类型，并附有具体的代码片段作为证据。

---

## 1. llm_function 支持的输入参数类型

### 1.1 基本类型
- `str` - 字符串
- `int` - 整数
- `float` - 浮点数
- `bool` - 布尔值

**代码证据：**
```python
# SimpleLLMFunc/base/type_resolve/description.py:107-119
if type_hint is str:
    return "example"
if type_hint is int:
    return 123
if type_hint is float:
    return 1.23
if type_hint is bool:
    return True
```

### 1.2 复合类型
- `List[T]` - 列表，支持嵌套类型
- `Dict[K, V]` - 字典，支持嵌套类型
- `Optional[T]` / `Union[T, None]` - 可选类型

**代码证据：**
```python
# SimpleLLMFunc/base/type_resolve/description.py:19-33
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
```

### 1.3 Pydantic 模型
- `BaseModel` 子类 - 自动解析为 JSON Schema

**代码证据：**
```python
# SimpleLLMFunc/base/type_resolve/description.py:16-17
if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
    return describe_pydantic_model(type_hint)

# SimpleLLMFunc/base/type_resolve/description.py:38-68
def describe_pydantic_model(model_class: Type[BaseModel]) -> str:
    """Expand a Pydantic model to a descriptive summary."""
    model_name = model_class.__name__
    schema = model_class.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    # ... 构建字段描述
```

### 1.4 多模态类型
- `Text` - 文本内容
- `ImgUrl` - 网络图片URL
- `ImgPath` - 本地图片路径
- `List[Text]` / `List[ImgUrl]` / `List[ImgPath]` - 多模态列表
- `Union[Text, ImgUrl, ImgPath]` - 多模态联合类型

**代码证据：**
```python
# SimpleLLMFunc/type/multimodal.py:27-106
class Text:
    """文本内容类型"""
    def __init__(self, content: str):
        self.content = content

class ImgUrl:
    """图片URL类型"""
    def __init__(self, url: str, detail: str = "auto"):
        # 验证URL格式
        if not (url.startswith("http://") or url.startswith("https://") or url.startswith("data:")):
            raise ValueError("Image URL must start with http://, https://, or data:")

class ImgPath:
    """本地图片路径类型"""
    def __init__(self, path: Union[str, Path], detail: str = "auto"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
```

**多模态检测代码：**
```python
# SimpleLLMFunc/base/type_resolve/multimodal.py:30-61
def is_multimodal_type(value: Any, annotation: Any) -> bool:
    """Determine whether a value/annotation pair represents multimodal content."""
    if isinstance(value, (Text, ImgUrl, ImgPath)):
        return True
    
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        for arg_type in non_none_args:
            if is_multimodal_type(value, arg_type):
                return True
    
    if origin in (list, TypingList):
        if not args:
            return False
        element_type = args[0]
        if element_type in (Text, ImgUrl, ImgPath):
            return True
    
    if annotation in (Text, ImgUrl, ImgPath):
        return True
    
    return False
```

---

## 2. llm_function 支持的返回参数类型

### 2.1 基本类型
- `str` - 字符串（默认）
- `int` - 整数
- `float` - 浮点数
- `bool` - 布尔值
- `None` - 无返回值

**代码证据：**
```python
# SimpleLLMFunc/base/post_process.py:23-27
if return_type is None or return_type is str:
    return cast(T, content)

if return_type in (int, float, bool):
    return cast(T, _convert_to_primitive_type(content, return_type))
```

### 2.2 复合类型
- `List[T]` - 列表，从XML解析
- `Dict[str, Any]` - 字典，从XML解析

**代码证据：**
```python
# SimpleLLMFunc/base/post_process.py:29-35
# 检查是否为 List 类型
origin = getattr(return_type, "__origin__", None) or get_origin(return_type)
if origin is list or origin is List:
    return cast(T, _convert_xml_to_list(content, return_type, func_name))

if return_type is dict or getattr(return_type, "__origin__", None) is dict:
    return cast(T, _convert_from_xml(content, func_name))
```

### 2.3 Pydantic 模型
- `BaseModel` 子类 - 从XML解析并验证

**代码证据：**
```python
# SimpleLLMFunc/base/post_process.py:37-38
if return_type and hasattr(return_type, "model_validate"):
    return cast(T, _convert_xml_to_pydantic(content, return_type, func_name))
```

---

## 3. llm_chat 支持的输入参数类型

### 3.1 基本类型（与 llm_function 相同）
- `str`, `int`, `float`, `bool`
- `List[T]`, `Dict[K, V]`
- `Optional[T]`
- Pydantic `BaseModel` 子类

### 3.2 多模态类型（与 llm_function 相同）
- `Text`, `ImgUrl`, `ImgPath`
- `List[Text]` / `List[ImgUrl]` / `List[ImgPath]`
- `Union[Text, ImgUrl, ImgPath]`

### 3.3 特殊参数
- `history: List[Dict[str, str]]` - 对话历史（特殊处理）
- `chat_history: List[Dict[str, str]]` - 对话历史（别名）

**代码证据：**
```python
# SimpleLLMFunc/llm_decorator/llm_chat_decorator.py:42-45
HISTORY_PARAM_NAMES: List[str] = [
    "history",
    "chat_history",
]  # Valid parameter names for conversation history

# SimpleLLMFunc/llm_decorator/steps/chat/message.py:20-59
def extract_conversation_history(
    arguments: Dict[str, Any],
    func_name: str,
    history_param_names: Optional[List[str]] = None,
) -> Optional[HistoryList]:
    """提取并验证对话历史"""
    # ... 查找并验证历史参数
```

**消息构建代码：**
```python
# SimpleLLMFunc/llm_decorator/steps/chat/message.py:62-82
def build_chat_user_message_content(
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    has_multimodal: bool,
    exclude_params: List[str],
) -> Union[str, List[Dict[str, Any]]]:
    """构建用户消息内容"""
    if has_multimodal:
        return build_multimodal_content(
            arguments,
            type_hints,
            exclude_params=exclude_params,
        )
    else:
        # 构建文本消息，排除历史参数
        message_parts = [
            f"{param_name}: {param_value}"
            for param_name, param_value in arguments.items()
            if param_name not in exclude_params
        ]
        return "\n\t".join(message_parts)
```

---

## 4. llm_chat 支持的返回参数类型

### 4.1 返回格式
`llm_chat` 返回 `AsyncGenerator[Tuple[Any, HistoryList], None]`

- **第一个元素** (`Any`): 
  - `return_mode="text"`: 返回 `str`（响应文本）
  - `return_mode="raw"`: 返回原始 LLM API 响应对象

- **第二个元素** (`HistoryList`): `List[Dict[str, str]]` - 过滤后的对话历史

**代码证据：**
```python
# SimpleLLMFunc/llm_decorator/llm_chat_decorator.py:58-61
) -> Callable[
    [Union[Callable[P, Any], Callable[P, Awaitable[Any]]]],
    Callable[P, AsyncGenerator[Tuple[Any, HistoryList], None]],
]:

# SimpleLLMFunc/llm_decorator/llm_chat_decorator.py:90-104
## Return Value Format
```python
AsyncGenerator[Tuple[str, List[Dict[str, str]]], None]
```
- `str`: Assistant's response content
- `List[Dict[str, str]]`: Filtered conversation history (excluding tool call information)
```

---

## 5. @tool 装饰器支持的输入参数类型

### 5.1 基本类型
- `str`, `int`, `float`, `bool`

**代码证据：**
```python
# SimpleLLMFunc/tool/tool.py:204-212
# 基本类型映射
if type_annotation is str:
    return {"type": "string"}
elif type_annotation is int:
    return {"type": "integer"}
elif type_annotation is float:
    return {"type": "number"}
elif type_annotation is bool:
    return {"type": "boolean"}
```

### 5.2 复合类型
- `List[T]` - 列表类型
- `Dict[K, V]` - 字典类型
- `Optional[T]` / `Union[T, None]` - 可选类型

**代码证据：**
```python
# SimpleLLMFunc/tool/tool.py:214-230
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
```

### 5.3 Pydantic 模型
- `BaseModel` 子类 - 自动转换为 JSON Schema

**代码证据：**
```python
# SimpleLLMFunc/tool/tool.py:232-236
# 处理Pydantic模型
if isinstance(type_annotation, type) and issubclass(type_annotation, BaseModel):
    # 使用Pydantic的model_json_schema方法获取模型的JSON Schema
    schema = type_annotation.model_json_schema()
    return {"type": "object", "properties": schema.get("properties", {})}
```

### 5.4 多模态类型
- `Text` - 文本内容
- `ImgUrl` - 网络图片URL
- `ImgPath` - 本地图片路径
- `List[Text]` - 文本内容列表
- `List[ImgUrl]` - 网络图片URL列表
- `List[ImgPath]` - 本地图片路径列表
- `Optional[List[Text/ImgUrl/ImgPath]]` - 可选的多模态列表

**代码证据：**
```python
# SimpleLLMFunc/tool/tool.py:337-342
## 一个工具函数支持的传入参数类型：
- **基本类型**: str, int, float, bool
- **复合类型**: List[T], Dict[K, V]  
- **可选类型**: Optional[T], Union[T, None]
- **Pydantic模型**: 继承自BaseModel的类，会自动解析为JSON Schema
- **多模态类型**: ImgPath（本地图片）, ImgUrl（网络图片）, Text（文本）
- **多模态列表**: List[ImgPath], List[ImgUrl], List[Text]

# SimpleLLMFunc/tool/tool.py:204-212
# 处理多模态类型
if type_annotation is ImgPath:
    return {"type": "string", "description": "本地图片路径"}
elif type_annotation is ImgUrl:
    return {"type": "string", "description": "网络图片URL"}
elif type_annotation is Text:
    return {"type": "string", "description": "文本内容"}
```

**参数自动转换：**
```python
# SimpleLLMFunc/base/tool_call/execution.py:16-150
def _convert_tool_arguments(
    arguments: Dict[str, Any],
    tool_func: Callable[..., Awaitable[Any]],
) -> Dict[str, Any]:
    """转换工具参数，将字符串列表转换为多模态对象列表。
    
    根据工具函数的类型注解，自动将 LLM 传递的字符串数组转换为对应的多模态对象数组。
    支持的类型：
    - List[ImgPath] -> List[ImgPath对象]
    - List[ImgUrl] -> List[ImgUrl对象]
    - List[Text] -> List[Text对象]
    """
    # ... 实现参数转换逻辑
```

**使用示例：**
```python
from SimpleLLMFunc.tool import tool
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl
from typing import List

@tool(name="process_images", description="处理图片列表")
async def process_images(image_paths: List[ImgPath], image_urls: List[ImgUrl]) -> str:
    """
    处理多个图片
    
    Args:
        image_paths: 本地图片路径列表
        image_urls: 网络图片URL列表
    """
    # LLM 传递的字符串列表会自动转换为 ImgPath/ImgUrl 对象列表
    for img_path in image_paths:
        print(f"处理本地图片: {img_path.path}")
    for img_url in image_urls:
        print(f"处理网络图片: {img_url.url}")
    return "处理完成"
```

---

## 6. @tool 装饰器支持的返回参数类型

### 6.1 基本可序列化类型
- `str` - 字符串
- `int`, `float`, `bool` - 数值和布尔值
- `dict` - 字典（JSON可序列化）
- `list` - 列表（JSON可序列化）

**代码证据：**
```python
# SimpleLLMFunc/base/tool_call/validation.py:54-73
def is_valid_tool_result(result: Any) -> bool:
    """Validate whether a tool return value is supported."""
    if isinstance(result, (ImgPath, ImgUrl)):
        return True
    
    if isinstance(result, str):
        return True
    
    if isinstance(result, tuple) and len(result) == 2:
        text_part, img_part = result
        if isinstance(text_part, str) and isinstance(img_part, (ImgPath, ImgUrl)):
            return True
        return False
    
    try:
        json.dumps(result)
        return True
    except (TypeError, ValueError):
        return False:54-73
def is_valid_tool_result(result: Any) -> bool:
    """Validate whether a tool return value is supported."""
    if isinstance(result, str):
        return True
    
    try:
        json.dumps(result)
        return True
    except (TypeError, ValueError):
        return False
```

### 6.2 多模态返回类型
- `ImgPath` - 返回本地图片路径
- `ImgUrl` - 返回网络图片URL
- `Tuple[str, ImgPath]` - 返回说明文本和本地图片的组合
- `Tuple[str, ImgUrl]` - 返回说明文本和网络图片的组合
- `Text` - 返回文本内容

**代码证据：**
```python
# SimpleLLMFunc/tool/tool.py:344-351
## 一个工具函数支持的返回值类型：
- **基本类型**: str, int, float, bool, dict, list等可序列化类型
- **多模态返回**: 
  - ImgPath: 返回本地图片路径，用于生成图表、处理后的图片等
  - ImgUrl: 返回网络图片URL，用于搜索到的图片、在线资源等
  - Tuple[str, ImgPath]: 返回说明文本和图片的组合
  - Tuple[str, ImgUrl]: 返回说明文本和网络图片的组合
```

**多模态返回处理代码：**
```python
# SimpleLLMFunc/base/tool_call/execution.py:86-182
if isinstance(tool_result, ImgUrl):
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": tool_result.url,
            "detail": tool_result.detail,
        },
    }
    user_multimodal_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"这是工具 '{tool_name}' 返回的图像：",
            },
            image_content,
        ],
    }
    messages_to_append.append(user_multimodal_message)
    return (tool_call, messages_to_append, True)

if isinstance(tool_result, ImgPath):
    base64_img = tool_result.to_base64()
    mime_type = tool_result.get_mime_type()
    data_url = f"data:{mime_type};base64,{base64_img}"
    # ... 构建多模态消息

if isinstance(tool_result, tuple) and len(tool_result) == 2:
    text_part, img_part = tool_result
    if isinstance(text_part, str) and isinstance(img_part, ImgUrl):
        # ... 处理文本+图片URL组合
    if isinstance(text_part, str) and isinstance(img_part, ImgPath):
        # ... 处理文本+图片路径组合
```

---

## 总结

### llm_function
- **输入**: 基本类型、复合类型、Pydantic模型、多模态类型
- **返回**: 基本类型、List、Dict、Pydantic模型（从XML解析）

### llm_chat
- **输入**: 与 llm_function 相同，额外支持 `history`/`chat_history` 参数
- **返回**: `AsyncGenerator[Tuple[Any, HistoryList], None]`，其中 `Any` 可以是文本或原始响应

### @tool
- **输入**: 基本类型、复合类型、Pydantic模型、多模态类型（转换为JSON Schema）
- **返回**: JSON可序列化类型、多模态类型（ImgPath、ImgUrl、Text）、多模态组合（Tuple[str, ImgPath/ImgUrl]）

