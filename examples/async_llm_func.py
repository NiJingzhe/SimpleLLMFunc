"""
使用异步LLM函数装饰器的示例

本示例展示了如何使用async_llm_function装饰器创建异步的LLM函数。
异步装饰器现在使用原生异步实现，无需线程池，提供更好的性能和并发能力。

主要特性:
1. 原生异步实现，不阻塞事件循环
2. 支持真正的并发调用，提高性能
3. 与其他异步操作无缝配合
4. 自动处理异步LLM接口调用
"""

import asyncio
import time
from typing import Dict, List
from pydantic import BaseModel, Field

from SimpleLLMFunc import llm_chat, app_log
from SimpleLLMFunc.llm_decorator import async_llm_function
from SimpleLLMFunc import tool
from SimpleLLMFunc import APIKeyPool
from SimpleLLMFunc.config import global_settings
import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Union

from SimpleLLMFunc import OpenAICompatible
import os

# 当前脚本文件所在的文件夹下的provider.json文件
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")
VolcEngine_deepseek_v3_Interface = OpenAICompatible.load_from_json_file(provider_json_path)["volc_engine"]["deepseek-v3-250324"]

# 定义一个Pydantic模型作为返回类型
class ProductReview(BaseModel):
    rating: int = Field(..., description="产品评分，1-5分")
    pros: List[str] = Field(..., description="产品优点列表")
    cons: List[str] = Field(..., description="产品缺点列表")
    summary: str = Field(..., description="评价总结")


# 使用异步装饰器创建一个异步LLM函数
@async_llm_function(
    llm_interface=VolcEngine_deepseek_v3_Interface,
)
async def analyze_product_review(product_name: str, review_text: str) -> ProductReview:  # type: ignore
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


@async_llm_function(
    llm_interface=VolcEngine_deepseek_v3_Interface, 
    toolkit=[get_weather],
)
async def get_daily_recommendation(city: str) -> WeatherInfo:  # type: ignore
    """
    通过get_weather工具获取天气信息，并给出推荐的活动

    Args:
        city: 城市名称

    Returns:
        WeatherInfo对象，包含温度、湿度和天气状况
    """
    pass


@async_llm_function(
    llm_interface=VolcEngine_deepseek_v3_Interface,
)
async def translate_text(text: str, target_language: str = "English") -> str:  # type: ignore
    """
    将输入文本翻译成指定语言

    Args:
        text: 要翻译的文本
        target_language: 目标语言，默认为英语

    Returns:
        翻译后的文本
    """
    pass


@async_llm_function(
    llm_interface=VolcEngine_deepseek_v3_Interface,
)
async def generate_summary(text: str, max_words: int = 100) -> str:  # type: ignore
    """
    生成文本摘要

    Args:
        text: 要总结的文本
        max_words: 摘要的最大字数

    Returns:
        文本摘要
    """
    pass


async def test_single_function():
    """测试单个异步LLM函数"""
    print("\n===== 测试单个异步函数 =====")
    
    app_log("开始运行异步示例代码")
    
    # 测试产品评测分析
    product_name = "XYZ无线耳机"
    review_text = """
    我买了这款XYZ无线耳机已经使用了一个月。音质非常不错，尤其是低音部分表现出色，
    佩戴也很舒适，可以长时间使用不感到疲劳。电池续航能力也很强，充满电后可以使用约8小时。
    不过连接偶尔会有些不稳定，有时候会突然断开。另外，触控操作不够灵敏，经常需要点击多次才能响应。
    总的来说，这款耳机性价比很高，适合日常使用，但如果你需要用于专业音频工作可能还不够。
    """

    print("===== 产品评测分析 =====")
    start_time = time.time()
    result = await analyze_product_review(product_name, review_text)
    end_time = time.time()
    
    print(f"评分: {result.rating}/5")
    print("优点:")
    for pro in result.pros:
        print(f"- {pro}")
    print("缺点:")
    for con in result.cons:
        print(f"- {con}")
    print(f"总结: {result.summary}")
    print(f"耗时: {end_time - start_time:.2f}秒")


async def test_concurrent_functions():
    """测试并发调用多个异步LLM函数"""
    print("\n===== 测试并发调用 =====")
    
    # 测试数据
    texts = [
        "今天是个好天气，适合外出游玩。",
        "最新的人工智能技术正在快速发展。", 
        "这家餐厅的菜品质量非常不错，服务也很好。"
    ]
    
    cities = ["Beijing", "Shanghai", "Hangzhou"]
    
    print("开始并发处理（使用原生异步实现）...")
    start_time = time.time()
    
    # 创建所有任务
    translation_tasks = [translate_text(text, "English") for text in texts]
    summary_tasks = [generate_summary(text, 50) for text in texts]
    weather_tasks = [get_daily_recommendation(city) for city in cities]
    
    # 使用 asyncio.gather 并发执行所有任务
    translations, summaries, weather_infos = await asyncio.gather(
        asyncio.gather(*translation_tasks),
        asyncio.gather(*summary_tasks),
        asyncio.gather(*weather_tasks)
    )
    
    end_time = time.time()
    
    print(f"并发处理完成，总耗时: {end_time - start_time:.2f}秒")
    
    # 显示结果
    print("\n===== 翻译结果 =====")
    for i, (text, translation) in enumerate(zip(texts, translations)):
        print(f"原文 {i+1}: {text}")
        print(f"译文 {i+1}: {translation}")
        print()
    
    print("===== 摘要结果 =====")
    for i, (text, summary) in enumerate(zip(texts, summaries)):
        print(f"原文 {i+1}: {text}")
        print(f"摘要 {i+1}: {summary}")
        print()
    
    print("===== 天气推荐结果 =====")
    for i, (city, weather_info) in enumerate(zip(cities, weather_infos)):
        print(f"城市: {city}")
        print(f"温度: {weather_info.temperature}")
        print(f"湿度: {weather_info.humidity}")
        print(f"天气状况: {weather_info.condition}")
        print(f"推荐活动: {weather_info.recommendation}")
        print()


async def test_sequential_vs_concurrent():
    """测试串行执行 vs 并发执行的性能差异（原生异步实现）"""
    print("\n===== 性能对比测试 =====")
    
    texts = [
        "人工智能技术正在改变我们的生活。",
        "气候变化是全球面临的重大挑战。",
        "远程工作已成为一种新趋势。"
    ]
    
    # 测试串行执行
    print("测试串行执行...")
    start_time = time.time()
    
    serial_results = []
    for text in texts:
        result = await translate_text(text, "English")
        serial_results.append(result)
    
    serial_time = time.time() - start_time
    
    # 测试并发执行（原生异步）
    print("测试并发执行（原生异步）...")
    start_time = time.time()
    
    concurrent_tasks = [translate_text(text, "English") for text in texts]
    concurrent_results = await asyncio.gather(*concurrent_tasks)
    
    concurrent_time = time.time() - start_time
    
    # 显示结果
    print(f"串行执行耗时: {serial_time:.2f}秒")
    print(f"并发执行耗时（原生异步）: {concurrent_time:.2f}秒")
    print(f"性能提升: {serial_time / concurrent_time:.2f}x")
    
    # 验证结果一致性
    print("\n结果验证:")
    for i, (serial, concurrent) in enumerate(zip(serial_results, concurrent_results)):
        print(f"文本 {i+1} 结果一致: {serial == concurrent}")


async def test_error_handling():
    """测试异步函数的错误处理"""
    print("\n===== 错误处理测试 =====")
    
    # 模拟一些可能导致错误的输入
    test_cases = [
        {"text": "正常文本", "should_work": True},
        {"text": "", "should_work": False},  # 空文本可能导致错误
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {'正常情况' if case['should_work'] else '可能出错'}")
        
        try:
            result = await translate_text(case["text"], "English")
            print(f"  成功: {result[:50]}{'...' if len(result) > 50 else ''}")
        except Exception as e:
            print(f"  错误: {str(e)}")


async def test_with_real_weather():
    """测试带有工具调用的异步函数"""
    print("\n===== 带工具调用的异步函数测试 =====")
    
    cities = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen"]
    
    print("开始并发查询天气信息（原生异步）...")
    start_time = time.time()
    
    # 并发查询多个城市的天气
    weather_tasks = [get_daily_recommendation(city) for city in cities]
    weather_results = await asyncio.gather(*weather_tasks)
    
    end_time = time.time()
    
    print(f"并发查询完成，耗时: {end_time - start_time:.2f}秒")
    
    for city, weather_info in zip(cities, weather_results):
        print(f"\n{city}:")
        print(f"  温度: {weather_info.temperature}")
        print(f"  湿度: {weather_info.humidity}")
        print(f"  天气: {weather_info.condition}")
        print(f"  建议: {weather_info.recommendation}")


async def test_massive_concurrency():
    """测试大规模并发能力（展示原生异步的优势）"""
    print("\n===== 大规模并发测试 =====")
    
    # 创建大量的并发任务
    test_texts = [
        f"这是第{i}条测试文本，用于验证大规模并发处理能力。"
        for i in range(1, 21)  # 20个并发任务
    ]
    
    print(f"开始处理 {len(test_texts)} 个并发翻译任务...")
    start_time = time.time()
    
    # 创建所有翻译任务
    translation_tasks = [
        translate_text(text, "English") 
        for text in test_texts
    ]
    
    # 并发执行所有任务
    results = await asyncio.gather(*translation_tasks)
    
    end_time = time.time()
    
    print(f"大规模并发处理完成!")
    print(f"处理了 {len(results)} 个任务")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"平均每个任务: {(end_time - start_time) / len(results):.2f}秒")
    
    # 显示部分结果
    print("\n前3个翻译结果:")
    for i, (original, translated) in enumerate(zip(test_texts[:3], results[:3])):
        print(f"{i+1}. 原文: {original}")
        print(f"   译文: {translated}\n")


async def main():
    """主函数，运行所有测试"""
    print("异步 LLM 函数装饰器示例")
    print("=" * 50)
    print("使用原生异步实现，无需线程池，性能更优！")
    
    try:
        # 运行各种测试
        await test_single_function()
        await test_concurrent_functions()
        await test_sequential_vs_concurrent()
        await test_error_handling()
        await test_with_real_weather()
        await test_massive_concurrency()  # 新增的大规模并发测试
        
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("所有异步示例运行完成！")
    print("原生异步实现提供了更好的性能和并发能力。")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())