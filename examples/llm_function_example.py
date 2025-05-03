"""
使用LLM函数装饰器的示例
"""

from typing import Dict, List
from pydantic import BaseModel, Field

from SimpleLLMFunc import llm_function
from SimpleLLMFunc import app_log
from SimpleLLMFunc import tool
from SimpleLLMFunc.interface import APIKeyPool, Zhipu
from SimpleLLMFunc.config import global_settings
from SimpleLLMFunc.interface.zhipu import ZhipuAI_glm_4_flash_Interface


zhipu_key_pool = APIKeyPool(
    api_keys=global_settings.ZHIPU_API_KEYS,
    provider_id="zhipu"
)
ZhipuAI_glm_4_flash_Interface = Zhipu(
    api_key_pool=zhipu_key_pool,
    model_name='glm-4-flash'
)

# 定义一个Pydantic模型作为返回类型
class ProductReview(BaseModel):
    rating: int = Field(..., description="产品评分，1-5分")
    pros: List[str] = Field(..., description="产品优点列表")
    cons: List[str] = Field(..., description="产品缺点列表")
    summary: str = Field(..., description="评价总结")


# 使用装饰器创建一个LLM函数
@llm_function(
    llm_interface=ZhipuAI_glm_4_flash_Interface,
)
def analyze_product_review(product_name: str, review_text: str) -> ProductReview:  # type: ignore
    """
    分析产品评论，提取关键信息并生成结构化评测报告

    Args:
        product_name: 产品名称
        review_text: 用户评论文本

    Returns:
        包含评分、优缺点和总结的产品评测报告
    """
    pass  # 函数体为空，实际执行由LLM完成


@tool(name="get_weather", description="获取指定城市的天气信息")
def get_weather(city: str) -> Dict[str, str]:
    """
    获取指定城市的天气信息

    Args:
        city: 城市名称

    Returns:
        包含温度、湿度和天气状况的字典
    """
    return {"temperature": "32°C", "humidity": "80%", "condition": "Raining"}


class WeatherInfo(BaseModel):
    city: str = Field(..., description="城市名称")
    temperature: str = Field(..., description="当前温度")
    humidity: str = Field(..., description="当前湿度")
    condition: str = Field(..., description="天气状况")
    recommendation: str = Field(..., description="推荐的活动")


@llm_function(llm_interface=ZhipuAI_glm_4_flash_Interface, toolkit=[get_weather])
def get_daily_recommendation(city: str) -> WeatherInfo:  # type: ignore
    """
    通过get_weather工具获取天气信息，并给出推荐的活动

    Args:
        city: 城市名称

    Returns:
        WeatherInfo对象，包含温度、湿度和天气状况
    """
    pass


def main():

    app_log("开始运行示例代码")
    # 测试产品评测分析
    product_name = "XYZ无线耳机"
    review_text = """
    我买了这款XYZ无线耳机已经使用了一个月。音质非常不错，尤其是低音部分表现出色，
    佩戴也很舒适，可以长时间使用不感到疲劳。电池续航能力也很强，充满电后可以使用约8小时。
    不过连接偶尔会有些不稳定，有时候会突然断开。另外，触控操作不够灵敏，经常需要点击多次才能响应。
    总的来说，这款耳机性价比很高，适合日常使用，但如果你需要用于专业音频工作可能还不够。
    """

    try:
        print("\n===== 产品评测分析 =====")
        result = analyze_product_review(product_name, review_text)
        print(f"评分: {result.rating}/5")
        print("优点:")
        for pro in result.pros:
            print(f"- {pro}")
        print("缺点:")
        for con in result.cons:
            print(f"- {con}")
        print(f"总结: {result.summary}")
    except Exception as e:
        print(f"产品评测分析失败: {e}")

    # 测试天气查询
    city = "Hangzhou"
    try:
        print("\n===== 天气查询 =====")
        weather_info = get_daily_recommendation(city)
        print(f"推荐活动: {weather_info.recommendation}")
        print(f"城市: {city}")
        print(f"温度: {weather_info.temperature}")
        print(f"湿度: {weather_info.humidity}")
        print(f"天气状况: {weather_info.condition}")
    except Exception as e:
        print(f"天气查询失败: {e}")


if __name__ == "__main__":
    main()
