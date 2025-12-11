"""
展示在 return_mode="raw" 下，@llm_chat 透传底层原始响应，
从而实时解析到工具调用（tool_calls / delta.tool_calls）。

运行前准备：请在 examples/provider.json 中配置兼容的提供方与模型。
建议选择支持函数调用/工具调用的 OpenAI 兼容模型。
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from SimpleLLMFunc import tool
from SimpleLLMFunc.llm_decorator import llm_chat
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible


# ============ Provider & Interface ============
current_dir: str = os.path.dirname(os.path.abspath(__file__))
provider_json_path: str = os.path.join(current_dir, "provider.json")

# 选用与现有示例一致的 key（可按需修改到你可用的 provider/model）
VolcEngine_deepseek_v3_Interface = OpenAICompatible.load_from_json_file(
    provider_json_path
)["openrouter"]["google/gemini-3-pro-preview"]


# ============ 定义一个简单工具 ============
@tool(name="get_weather", description="获取指定城市的天气信息")
async def get_weather(city: str) -> Dict[str, str]:
    """
    获取城市天气。

    Args:
        city: 城市名

    Returns:
        包含温度/湿度/天气状况的字典
    """
    return {"temperature": "28°C", "humidity": "65%", "condition": "Sunny"}


# ============ 原始响应透传：流式示例 ============
@llm_chat(
    llm_interface=VolcEngine_deepseek_v3_Interface,
    toolkit=[get_weather],
    stream=True,
    return_mode="raw",
)
async def chat_stream_raw(history: Optional[List[Dict[str, Any]]] = None, query: str = ""):
    """
    流式原始响应透传示例。
    要求：若用户询问天气，请调用 get_weather 工具。
    """
    pass


# ============ 原始响应透传：非流式示例 ============
@llm_chat(
    llm_interface=VolcEngine_deepseek_v3_Interface,
    toolkit=[get_weather],
    stream=False,
    return_mode="raw",
)
async def chat_nonstream_raw(history: Optional[List[Dict[str, Any]]] = None, query: str = ""):
    """
    非流式原始响应透传示例。
    要求：若用户询问天气，请调用 get_weather 工具。
    """
    pass


def _print_stream_chunk(chunk: Any) -> None:
    """打印流式 chunk 中的文本增量与工具调用增量（若有）。"""
    if hasattr(chunk, "choices") and chunk.choices:
        choice = chunk.choices[0]
        # 文本增量
        if hasattr(choice, "delta") and choice.delta:
            delta = choice.delta
            if getattr(delta, "content", None):
                print(delta.content, end="")
            # 工具调用增量
            if getattr(delta, "tool_calls", None):
                try:
                    print("\n[tool_calls(delta)]:", json.dumps(delta.tool_calls, default=str, ensure_ascii=False))
                except Exception:
                    print("\n[tool_calls(delta) detected]")


def _print_nonstream_message(raw: Any) -> None:
    """打印非流式响应中的工具调用（若有）与文本。"""
    if not hasattr(raw, "choices") or not raw.choices:
        return
    msg = raw.choices[0].message
    # 文本
    content = getattr(msg, "content", None)
    if content:
        print(content)
    # 工具调用
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        try:
            print("[tool_calls]:", json.dumps(tool_calls, default=str, ensure_ascii=False))
        except Exception:
            print("[tool_calls detected]")


async def demo_stream() -> None:
    print("\n=== 流式原始响应透传（可观测 delta.tool_calls）===")
    history: List[Dict[str, Any]] = []
    async for raw, messages in chat_stream_raw(history=history, query="请查询北京今天的天气，并给出建议"):
        # 流式阶段：多次接收 chunk（含 delta 与可能的 delta.tool_calls）
        _print_stream_chunk(raw)
    # 工具调用结束与最终应答后，messages 将包含 tool 轨迹
    print("\n--- 最终 messages 中的 tool 轨迹（节选） ---")
    for m in messages:
        if m.get("role") in ("assistant", "tool"):
            print(json.dumps(m, ensure_ascii=False))


async def demo_nonstream() -> None:
    print("\n=== 非流式原始响应透传（可观测 message.tool_calls）===")
    history: List[Dict[str, Any]] = []
    async for raw, messages in chat_nonstream_raw(history=history, query="请查询上海今天的天气，并给出建议"):
        # 非流式阶段：首帧为 initial_response，含 message.tool_calls（如果模型触发工具）
        _print_nonstream_message(raw)
    print("\n--- 最终 messages 中的 tool 轨迹（节选） ---")
    for m in messages:
        if m.get("role") in ("assistant", "tool"):
            print(json.dumps(m, ensure_ascii=False))


async def main() -> None:
    await demo_stream()
    await demo_nonstream()


if __name__ == "__main__":
    asyncio.run(main())


