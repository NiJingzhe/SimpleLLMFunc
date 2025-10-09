"""
使用LLM函数装饰器的示例
"""

import asyncio
from pydantic import BaseModel, Field

from SimpleLLMFunc import llm_chat, llm_function, app_log
from SimpleLLMFunc import tool
from SimpleLLMFunc import APIKeyPool
from SimpleLLMFunc.config import global_settings
import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Union

from SimpleLLMFunc.type import *

from SimpleLLMFunc import OpenAICompatible
import os

# 当前脚本文件所在的文件夹下的provider.json文件
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")
VolcEngine_deepseek_v3_Interface = OpenAICompatible.load_from_json_file(provider_json_path)["dreamcatcher"]["gpt-4o"]

@tool(name="get_weather", description="获取指定城市的天气信息")
async def get_weather(city: str) -> Dict[str, str]:
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


@llm_function(llm_interface=VolcEngine_deepseek_v3_Interface, toolkit=[get_weather])
async def get_daily_recommendation(city: str) -> WeatherInfo:  # type: ignore
    """
    通过get_weather工具获取天气信息，并给出推荐的活动

    Args:
        city: 城市名称

    Returns:
        WeatherInfo对象，包含温度、湿度和天气状况
    """
    pass

@llm_function(llm_interface=VolcEngine_deepseek_v3_Interface, toolkit=[])
async def analyze_image(
    focus: Text,
    image_path: ImgPath 
) -> str:  # type: ignore
    """
    分析图像并提供描述

    Args:
        focus: 图像分析的重点描述
        image_path: 本地图片路径

    Returns:
        图像分析结果的字符串描述
    """
    return ""


async def main():

    app_log("开始运行示例代码")

    try:
        # 测试天气查询
        city = "Hangzhou"
        try:
            print("\n===== 天气查询 =====")
            weather_info = await get_daily_recommendation(city)
            print(f"推荐活动: {weather_info.recommendation}")
            print(f"城市: {city}")
            print(f"温度: {weather_info.temperature}")
            print(f"湿度: {weather_info.humidity}")
            print(f"天气状况: {weather_info.condition}")
        except Exception as e:
            print(f"天气查询失败: {e}")

            
        # 测试图像分析
        focus = Text("分析图像中的主要元素")
        image_path = ImgPath("./repocover_new.png", detail="high") 

        try:
            print("\n===== 图像分析 =====")
            analysis_result = await analyze_image(focus, image_path)
            print(f"图像分析结果: {analysis_result}")
        except Exception as e:
            print(f"图像分析失败: {e}")

    finally:
        # 确保异步资源得到清理
        await asyncio.sleep(0.1)  # 给异步清理任务一些时间完成
        
    app_log("示例代码运行结束")
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
