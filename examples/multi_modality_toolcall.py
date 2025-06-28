from SimpleLLMFunc import llm_function, tool
from SimpleLLMFunc.type import ImgPath
from SimpleLLMFunc import OpenAICompatible
import os
from typing import Optional

# 当前脚本文件所在的文件夹下的provider.json文件
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")
gpt_4o = OpenAICompatible.load_from_json_file(provider_json_path)["dreamcatcher"]["gpt-4o"]


@tool(
    name="get_image",
    description="Get an image from local path",
)
def get_image(image_path: str) -> tuple[str, ImgPath]:

    """Get an image from the local file system.

    Args:
        image_path: The path to the image file.
        detail: The detail level for the image retrieval.
            Can be 'low', 'high', or 'auto'.
            
    Returns:
        ImgPath: An object representing the image file with its path and detail level.
    """

    return "仔细分析这张图的几何结构", ImgPath(image_path, detail='low')

@llm_function(
   llm_interface=gpt_4o,
   toolkit=[get_image], 
   timeout=600
)
def analyze_image(
    focus: str,
    image_path: str
) -> str:  # type: ignore
    """Analyze an image and provide a description.

    Args:
        focus: The focus of the image analysis.
        image_path: The path to the local image file.

    Returns:
        str: A description of the image analysis result.
    """

if __name__ == "__main__":
    # Example usage
    path = input("Enter the path to the image: ")
    result = analyze_image("Analyze the image for objects, provide the simplest description possible", path)  # type: ignore
    print(result)  # This will not execute as the function body is empty