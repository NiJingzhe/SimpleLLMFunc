"""
多模态内容类型定义（向后兼容性重新导出）

本模块已移动到 SimpleLLMFunc.type.multimodal，此处仅为向后兼容性重新导出。
建议使用新的导入路径：
    from SimpleLLMFunc.type import Text, ImgUrl, ImgPath
"""

# 向后兼容性重新导出
from SimpleLLMFunc.type.multimodal import (
    ImgPath,
    ImgUrl,
    MultimodalContent,
    MultimodalList,
    Text,
)

__all__ = [
    "Text",
    "ImgUrl",
    "ImgPath",
    "MultimodalContent",
    "MultimodalList",
]
