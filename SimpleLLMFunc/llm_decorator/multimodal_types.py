"""
多模态内容类型定义

本模块定义了用于多模态LLM函数的类型，支持文本、图片URL和图片路径。
通过类型提示，框架可以自动识别参数类型并构建适当的消息格式。

示例:
```python
from multimodal_types import Text, ImgUrl, ImgPath

@llm_function
def analyze_image(
    description: Text,
    image_url: ImgUrl,
    reference_image: ImgPath
) -> str:
    \"\"\"分析图像并提供描述\"\"\"
    pass
```
"""

from typing import Union, List
from pathlib import Path
import base64


class Text:
    """文本内容类型"""

    def __init__(self, content: str):
        self.content = content

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"Text({self.content!r})"


class ImgUrl:
    """图片URL类型"""

    def __init__(self, url: str, detail: str = "auto"):
        if not (
            url.startswith("http://")
            or url.startswith("https://")
            or url.startswith("data:")
        ):
            raise ValueError("Image URL must start with http://, https://, or data:")
        if detail not in ("low", "high", "auto"):
            raise ValueError("detail must be 'low', 'high', or 'auto'")

        self.url = url
        self.detail = detail

    def __str__(self) -> str:
        return self.url

    def __repr__(self) -> str:
        return f"ImgUrl({self.url!r}, detail={self.detail!r})"


class ImgPath:
    """本地图片路径类型"""

    def __init__(self, path: Union[str, Path], detail: str = "auto"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if not self.path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if detail not in ("low", "high", "auto"):
            raise ValueError("detail must be 'low', 'high', or 'auto'")

        # 检查是否为图片文件
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        if self.path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported image format: {self.path.suffix}")

        self.detail = detail

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"ImgPath({self.path!r}, detail={self.detail!r})"

    def to_base64(self) -> str:
        """将图片转换为base64编码"""
        with open(self.path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_mime_type(self) -> str:
        """获取图片的MIME类型"""
        extension = self.path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        return mime_types.get(extension, "image/jpeg")


# 类型别名，方便使用
MultimodalContent = Union[Text, ImgUrl, ImgPath]
MultimodalList = List[MultimodalContent]
