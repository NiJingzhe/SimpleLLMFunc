import asyncio
import os

from SimpleLLMFunc import OpenAICompatible, llm_function, tool
from SimpleLLMFunc.type import ImgPath

# 当前脚本文件所在的文件夹下的provider.json文件
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")
llm_model = OpenAICompatible.load_from_json_file(provider_json_path)["openrouter"][
    "minimax/minimax-m2.5"
]


@tool(
    name="get_image",
    description="Get an image from local path",
)
async def get_image(image_path: str) -> tuple[str, ImgPath]:
    """Get an image from the local file system.

    Args:
        image_path: The path to the image file.

    Returns:
        A tuple containing the analysis instruction text and the image payload.
    """

    return "仔细分析这张图的几何结构", ImgPath(image_path, detail="low")


@llm_function(  # type: ignore
    llm_interface=llm_model,
    toolkit=[get_image],
    timeout=600,
)
async def analyze_image(
    focus: str,
    image_path: str,
) -> str:  # type: ignore
    """Analyze an image and provide a description.

    Args:
        focus: The focus of the image analysis.
        image_path: The path to the local image file.

    Returns:
        str: A description of the image analysis result.
    """

    return ""


async def main() -> None:
    path = input("Enter the path to the image: ")
    result: str = await analyze_image(
        "Analyze the image for objects, provide the simplest description possible",
        path,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
