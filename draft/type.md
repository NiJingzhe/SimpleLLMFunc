# Type 类型定义重构说明文档

## 概述

本文档规划 SimpleLLMFunc 框架中需要暴露给用户的类型定义，并说明这些类型的声明位置和组织方式。

## 设计原则

1. **最小暴露原则**：只暴露用户真正需要的类型，内部实现细节不暴露
2. **类型安全**：所有暴露的类型都应该有完整的类型注解
3. **集中管理**：所有公共类型定义集中在 `SimpleLLMFunc/type/` 模块中
4. **向后兼容**：保持现有 API 的兼容性

---

## 需要暴露的类型分类

### 1. 多模态类型（Multimodal Types）

**用途**：用于函数参数和返回值的类型标注，支持多模态输入输出

**类型列表**：
- `Text` - 文本内容类型
- `ImgUrl` - 图片 URL 类型
- `ImgPath` - 本地图片路径类型

**当前状态**：
- ✅ 已在 `SimpleLLMFunc/llm_decorator/multimodal_types.py` 中定义
- ✅ 已在 `SimpleLLMFunc/type/__init__.py` 中重新导出

**重构后位置**：
- **定义位置**：`SimpleLLMFunc/type/multimodal.py`（从 `llm_decorator/multimodal_types.py` 移动）
- **导出位置**：`SimpleLLMFunc/type/__init__.py`

**理由**：
- 多模态类型是用户直接使用的类型，应该属于 `type` 模块的核心功能
- 从 `llm_decorator` 移动到 `type` 更符合模块职责划分

---

### 2. 接口类型（Interface Types）

**用途**：定义 LLM 接口的抽象基类，供用户实现自定义 LLM 接口

**类型列表**：
- `LLM_Interface` - LLM 接口抽象基类

**当前状态**：
- ✅ 已在 `SimpleLLMFunc/interface/llm_interface.py` 中定义
- ✅ 已在 `SimpleLLMFunc/type/__init__.py` 中重新导出

**重构后位置**：
- **定义位置**：保持 `SimpleLLMFunc/interface/llm_interface.py`（不变）
- **导出位置**：`SimpleLLMFunc/type/__init__.py`（保持重新导出）

**理由**：
- `LLM_Interface` 是接口定义，属于 `interface` 模块的核心
- 通过 `type` 模块重新导出，方便用户统一导入

---

### 3. 工具类型（Tool Types）

**用途**：定义工具系统的基类和装饰器

**类型列表**：
- `Tool` - 工具抽象基类
- `@tool` - 工具装饰器函数

**当前状态**：
- ✅ 已在 `SimpleLLMFunc/tool/tool.py` 中定义
- ✅ 已在 `SimpleLLMFunc/tool/__init__.py` 中导出
- ❌ 未在 `SimpleLLMFunc/type/__init__.py` 中导出

**重构后位置**：
- **定义位置**：保持 `SimpleLLMFunc/tool/tool.py`（不变）
- **导出位置**：
  - `SimpleLLMFunc/tool/__init__.py`（主要导出位置）
  - `SimpleLLMFunc/type/__init__.py`（可选，重新导出 `Tool` 类型）

**理由**：
- `Tool` 是工具系统的核心，应该从 `tool` 模块导出
- 如果用户需要统一从 `type` 导入，可以考虑重新导出，但不是必须的

---

### 4. 装饰器相关类型（Decorator Types）

**用途**：用于装饰器内部的数据结构，部分可能需要暴露给高级用户

#### 4.1 FunctionSignature（新增）

**用途**：替代当前的 `FunctionCallContext`，用于存储函数签名信息

**定义**：
```python
from typing import NamedTuple, Any, Dict
import inspect

class FunctionSignature(NamedTuple):
    """函数签名信息"""
    func_name: str
    trace_id: str
    bound_args: inspect.BoundArguments
    signature: inspect.Signature
    type_hints: Dict[str, Any]
    return_type: Any
    docstring: str
```

**当前状态**：
- ❌ 尚未定义（重构后新增）
- 当前使用 `FunctionCallContext`（内部类型）

**重构后位置**：
- **定义位置**：`SimpleLLMFunc/type/decorator.py`（新建文件）
- **导出位置**：`SimpleLLMFunc/type/__init__.py`

**暴露策略**：
- ⚠️ **内部类型，不暴露给用户**
- 仅在 `llm_decorator/steps/common/types.py` 中使用
- 如果未来需要暴露给高级用户（如自定义步骤函数），再考虑导出

#### 4.2 HistoryList（新增）

**用途**：用于 `llm_chat` 装饰器的历史记录类型别名

**定义**：
```python
from typing import List, Dict, Any

HistoryList = List[Dict[str, Any]]
```

**当前状态**：
- ❌ 尚未定义（重构后新增）

**重构后位置**：
- **定义位置**：`SimpleLLMFunc/type/decorator.py`
- **导出位置**：`SimpleLLMFunc/type/__init__.py`

**暴露策略**：
- ✅ **暴露给用户**：`llm_chat` 的返回类型是 `AsyncGenerator[Tuple[Any, HistoryList], None]`，用户需要知道 `HistoryList` 的类型

---

### 5. 内部类型（Internal Types）

**用途**：仅用于框架内部实现，不暴露给用户

**类型列表**：
- `FunctionCallContext` - 将被 `FunctionSignature` 替代
- `LLMCallParams` - LLM 调用参数封装（内部使用）
- `Parameter` - 工具参数包装类（内部使用）

**暴露策略**：
- ❌ **不暴露**：这些类型是内部实现细节，用户不需要知道

---

## 类型定义位置规划

### 目录结构

```
SimpleLLMFunc/
├── type/                              # 类型定义模块（公共 API）
│   ├── __init__.py                   # 统一导出所有公共类型
│   ├── multimodal.py                  # 多模态类型定义（从 llm_decorator 移动）
│   └── decorator.py                  # 装饰器相关类型（新建）
│
├── interface/                         # 接口定义模块
│   ├── __init__.py
│   └── llm_interface.py              # LLM_Interface 定义（保持不变）
│
├── tool/                              # 工具系统模块
│   ├── __init__.py
│   └── tool.py                        # Tool 基类定义（保持不变）
│
└── llm_decorator/
    └── steps/
        └── common/
            └── types.py               # 装饰器内部类型（FunctionSignature 等）
```

---

## 重构步骤

### Phase 1: 创建类型定义文件

1. **创建 `SimpleLLMFunc/type/multimodal.py`**
   - 从 `SimpleLLMFunc/llm_decorator/multimodal_types.py` 移动代码
   - 保持 API 不变

2. **创建 `SimpleLLMFunc/type/decorator.py`**
   - 定义 `HistoryList` 类型别名
   - 预留 `FunctionSignature` 的定义位置（如果未来需要暴露）

3. **创建 `SimpleLLMFunc/llm_decorator/steps/common/types.py`**
   - 定义 `FunctionSignature`（内部使用）
   - 定义其他装饰器内部类型

### Phase 2: 更新导出

1. **更新 `SimpleLLMFunc/type/__init__.py`**
   ```python
   # 多模态类型
   from SimpleLLMFunc.type.multimodal import Text, ImgUrl, ImgPath
   
   # 接口类型
   from SimpleLLMFunc.interface.llm_interface import LLM_Interface
   
   # 装饰器相关类型
   from SimpleLLMFunc.type.decorator import HistoryList
   
   __all__ = [
       # 多模态类型
       "Text",
       "ImgUrl",
       "ImgPath",
       # 接口类型
       "LLM_Interface",
       # 装饰器类型
       "HistoryList",
   ]
   ```

2. **更新 `SimpleLLMFunc/llm_decorator/__init__.py`**
   - 确保装饰器函数正确导出
   - 移除对 `multimodal_types` 的直接导入（改为从 `type` 导入）

### Phase 3: 更新导入引用

1. **更新所有使用多模态类型的文件**
   - 将 `from SimpleLLMFunc.llm_decorator.multimodal_types import ...` 
   - 改为 `from SimpleLLMFunc.type import ...` 或 `from SimpleLLMFunc.type.multimodal import ...`

2. **更新装饰器实现**
   - 使用 `FunctionSignature` 替代 `FunctionCallContext`
   - 更新所有相关导入

### Phase 4: 清理旧文件

1. **删除 `SimpleLLMFunc/llm_decorator/multimodal_types.py`**
   - 代码已移动到 `SimpleLLMFunc/type/multimodal.py`

2. **更新文档**
   - 更新导入示例
   - 更新类型说明

---

## 类型导出策略

### 主要导出路径

**用户应该使用的导入方式**：

```python
# 方式 1: 从 type 模块统一导入（推荐）
from SimpleLLMFunc.type import Text, ImgUrl, ImgPath, LLM_Interface, HistoryList

# 方式 2: 从顶层包导入（如果 __init__.py 重新导出）
from SimpleLLMFunc import Text, ImgUrl, ImgPath, LLM_Interface

# 方式 3: 从具体模块导入（高级用户）
from SimpleLLMFunc.type.multimodal import Text, ImgUrl, ImgPath
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
```

### 向后兼容性

**保持现有导入方式有效**：

```python
# 旧方式（仍然有效）
from SimpleLLMFunc.type import Text, ImgUrl, ImgPath, LLM_Interface

# 新方式（推荐）
from SimpleLLMFunc.type import Text, ImgUrl, ImgPath, LLM_Interface, HistoryList
```

---

## 类型使用场景

### 1. 多模态类型使用场景

```python
from SimpleLLMFunc.type import Text, ImgUrl, ImgPath
from SimpleLLMFunc import llm_function

@llm_function
async def analyze_image(
    description: Text,
    image_url: ImgUrl,
    reference: ImgPath
) -> str:
    """分析图像"""
    pass
```

### 2. 接口类型使用场景

```python
from SimpleLLMFunc.type import LLM_Interface
from SimpleLLMFunc.interface.key_pool import APIKeyPool

class CustomLLMInterface(LLM_Interface):
    async def chat(self, ...):
        # 实现自定义 LLM 接口
        pass
```

### 3. 装饰器类型使用场景

```python
from SimpleLLMFunc.type import HistoryList
from SimpleLLMFunc import llm_chat
from typing import AsyncGenerator, Tuple

@llm_chat
async def chat_function(...) -> AsyncGenerator[Tuple[str, HistoryList], None]:
    """聊天函数"""
    pass
```

---

## 类型定义详细说明

### Text

**定义位置**：`SimpleLLMFunc/type/multimodal.py`

**用途**：表示文本内容

**API**：
```python
class Text:
    def __init__(self, content: str)
    def __str__(self) -> str
    def __repr__(self) -> str
```

**使用示例**：
```python
text = Text("Hello, world!")
```

---

### ImgUrl

**定义位置**：`SimpleLLMFunc/type/multimodal.py`

**用途**：表示网络图片 URL

**API**：
```python
class ImgUrl:
    def __init__(self, url: str, detail: str = "auto")
    def __str__(self) -> str
    def __repr__(self) -> str
    
    # 属性
    url: str
    detail: str  # "low" | "high" | "auto"
```

**使用示例**：
```python
img = ImgUrl("https://example.com/image.jpg", detail="high")
```

---

### ImgPath

**定义位置**：`SimpleLLMFunc/type/multimodal.py`

**用途**：表示本地图片路径

**API**：
```python
class ImgPath:
    def __init__(self, path: Union[str, Path], detail: str = "auto")
    def __str__(self) -> str
    def __repr__(self) -> str
    def to_base64(self) -> str
    def get_mime_type(self) -> str
    
    # 属性
    path: Path
    detail: str  # "low" | "high" | "auto"
```

**使用示例**：
```python
img = ImgPath("./local_image.png", detail="high")
base64_data = img.to_base64()
```

---

### LLM_Interface

**定义位置**：`SimpleLLMFunc/interface/llm_interface.py`

**用途**：LLM 接口抽象基类

**API**：
```python
class LLM_Interface(ABC):
    @abstractmethod
    def __init__(self, api_key_pool: APIKeyPool, model_name: str, base_url: Optional[str] = None)
    
    @abstractmethod
    async def chat(self, ...) -> ChatCompletion
    
    @abstractmethod
    async def chat_stream(self, ...) -> AsyncGenerator[ChatCompletionChunk, None]
```

---

### HistoryList

**定义位置**：`SimpleLLMFunc/type/decorator.py`

**用途**：聊天历史记录类型别名

**定义**：
```python
HistoryList = List[Dict[str, Any]]
```

**使用示例**：
```python
from SimpleLLMFunc.type import HistoryList
from typing import AsyncGenerator, Tuple

@llm_chat
async def chat(...) -> AsyncGenerator[Tuple[str, HistoryList], None]:
    """聊天函数"""
    pass
```

---

## 不暴露的类型

以下类型是内部实现细节，不应该暴露给用户：

1. **FunctionCallContext** - 将被 `FunctionSignature` 替代（内部使用）
2. **LLMCallParams** - LLM 调用参数封装（内部使用）
3. **Parameter** - 工具参数包装类（内部使用）
4. **FunctionSignature** - 装饰器内部数据结构（当前不暴露，未来可能暴露）

---

## 迁移检查清单

### 代码迁移

- [ ] 创建 `SimpleLLMFunc/type/multimodal.py`
- [ ] 创建 `SimpleLLMFunc/type/decorator.py`
- [ ] 创建 `SimpleLLMFunc/llm_decorator/steps/common/types.py`
- [ ] 移动多模态类型代码到新位置
- [ ] 更新 `SimpleLLMFunc/type/__init__.py` 导出
- [ ] 更新所有导入引用
- [ ] 删除旧文件 `multimodal_types.py`

### 测试验证

- [ ] 验证所有类型可以正确导入
- [ ] 验证多模态功能正常工作
- [ ] 验证装饰器功能正常工作
- [ ] 验证向后兼容性

### 文档更新

- [ ] 更新 README 中的导入示例
- [ ] 更新文档中的类型说明
- [ ] 更新 CHANGELOG

---

## 总结

### 暴露的类型列表

| 类型 | 定义位置 | 导出位置 | 暴露策略 |
|------|---------|---------|---------|
| `Text` | `type/multimodal.py` | `type/__init__.py` | ✅ 暴露 |
| `ImgUrl` | `type/multimodal.py` | `type/__init__.py` | ✅ 暴露 |
| `ImgPath` | `type/multimodal.py` | `type/__init__.py` | ✅ 暴露 |
| `LLM_Interface` | `interface/llm_interface.py` | `type/__init__.py` | ✅ 暴露 |
| `HistoryList` | `type/decorator.py` | `type/__init__.py` | ✅ 暴露 |
| `Tool` | `tool/tool.py` | `tool/__init__.py` | ✅ 暴露（从 tool 模块） |
| `FunctionSignature` | `llm_decorator/steps/common/types.py` | - | ❌ 不暴露（内部使用） |

### 关键决策

1. **多模态类型移动**：从 `llm_decorator` 移动到 `type` 模块，更符合职责划分
2. **FunctionSignature 不暴露**：当前是内部类型，如果未来需要暴露给高级用户，再考虑
3. **HistoryList 暴露**：因为 `llm_chat` 的返回类型需要它，用户需要知道这个类型
4. **集中导出**：所有公共类型通过 `type/__init__.py` 统一导出，方便用户使用

### 预期效果

- ✅ 类型定义更清晰，职责划分明确
- ✅ 用户导入更方便，统一从 `type` 模块导入
- ✅ 向后兼容，现有代码无需修改
- ✅ 为未来扩展预留空间

