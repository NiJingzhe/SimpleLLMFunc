"""
并行工具调用 - 简化示例

这是一个简化的并行工具调用示例，展示了最基础的用法。
适合快速理解并行工具调用的核心概念。

关键点：
1. 定义多个工具函数，使用 @tool 装饰器
2. 将工具列表传给 @llm_chat 装饰器
3. LLM 可以在单个回复中请求多个工具调用
4. 框架自动并发执行所有工具
5. 工具结果自动反馈给 LLM

运行前准备：请在 examples/provider.json 中配置兼容的提供方与模型。
建议选择支持函数调用/工具调用的 OpenAI 兼容模型。
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from SimpleLLMFunc import tool
from SimpleLLMFunc.llm_decorator import llm_chat
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible


# ============ 初始化 LLM 接口 ============
current_dir: str = os.path.dirname(os.path.abspath(__file__))
provider_json_path: str = os.path.join(current_dir, "provider.json")
llm_interface = OpenAICompatible.load_from_json_file(provider_json_path)[
    "volc_engine"
]["deepseek-v3-250324"]


# ============ 定义工具 ============
@tool(name="add", description="两个数字相加")
async def add(a: int, b: int) -> int:
    """
    计算两个数字的和。
    
    Args:
        a: 第一个数字
        b: 第二个数字
    
    Returns:
        两数之和
    """
    await asyncio.sleep(0.3)  # 模拟耗时操作
    result = a + b
    print(f"  [执行 add] {a} + {b} = {result}")
    return result


@tool(name="multiply", description="两个数字相乘")
async def multiply(a: int, b: int) -> int:
    """
    计算两个数字的乘积。
    
    Args:
        a: 第一个数字
        b: 第二个数字
    
    Returns:
        两数之积
    """
    await asyncio.sleep(0.4)  # 模拟耗时操作
    result = a * b
    print(f"  [执行 multiply] {a} * {b} = {result}")
    return result


@tool(name="subtract", description="第一个数字减去第二个数字")
async def subtract(a: int, b: int) -> int:
    """
    计算两个数字的差。
    
    Args:
        a: 被减数
        b: 减数
    
    Returns:
        差值
    """
    await asyncio.sleep(0.35)  # 模拟耗时操作
    result = a - b
    print(f"  [执行 subtract] {a} - {b} = {result}")
    return result


# ============ 定义 LLM 函数 ============
@llm_chat(
    llm_interface=llm_interface,
    toolkit=[add, multiply, subtract],
    stream=False,
    return_mode="raw",
)
async def math_calculator(
    history: Optional[List[Dict[str, Any]]] = None,
    query: str = "",
):
    """
    数学计算器 - 可以并行执行多个数学运算。
    
    例如查询时可以要求：
    - "请同时计算 10+5、8*3 和 20-7"
    - "计算 15+25、100-45 和 12*8"
    """
    pass


# ============ 演示函数 ============
def print_response(raw_response: Any) -> None:
    """打印 LLM 响应和工具调用信息"""
    if not hasattr(raw_response, "choices") or not raw_response.choices:
        return
    
    msg = raw_response.choices[0].message
    
    # 打印文本回复
    if hasattr(msg, "content") and msg.content:
        print(f"\n[LLM 回复]\n{msg.content}")
    
    # 打印工具调用
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"\n[检测到 {len(msg.tool_calls)} 个并行工具调用]")
        for i, tc in enumerate(msg.tool_calls, 1):
            try:
                args = json.loads(tc.function.arguments)
                print(f"  {i}. 工具: {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
            except:
                print(f"  {i}. 工具: {tc.function.name}({tc.function.arguments})")


async def demo_1():
    """演示 1: 简单的并行计算"""
    print("\n" + "="*60)
    print("  演示 1: 简单的并行计算")
    print("="*60)
    
    query = "请同时计算：10加5、8乘3、20减7"
    print(f"\n查询: {query}\n")
    
    history: List[Dict[str, Any]] = []
    async for raw, messages in math_calculator(history=history, query=query):
        print_response(raw)


async def demo_2():
    """演示 2: 更复杂的并行计算"""
    print("\n" + "="*60)
    print("  演示 2: 更复杂的并行计算")
    print("="*60)
    
    query = "请并行计算以下所有表达式的结果，然后告诉我最大的那个：100+50、60*2、150-30、25+75"
    print(f"\n查询: {query}\n")
    
    history: List[Dict[str, Any]] = []
    async for raw, messages in math_calculator(history=history, query=query):
        print_response(raw)


async def demo_3():
    """演示 3: 链式计算（工具结果作为后续查询的输入）"""
    print("\n" + "="*60)
    print("  演示 3: 链式计算（多轮对话）")
    print("="*60)
    
    print("\n第一步: 计算基础值")
    query1 = "请同时计算 100+50 和 20*3"
    print(f"查询: {query1}\n")
    
    history: List[Dict[str, Any]] = []
    async for raw, messages in math_calculator(history=history, query=query1):
        print_response(raw)
        # 保存消息历史
        if hasattr(raw, "choices") and raw.choices:
            msg = raw.choices[0].message
            if hasattr(msg, "content") and msg.content:
                history.append({
                    "role": "assistant",
                    "content": msg.content
                })
    
    print("\n\n第二步: 基于前面的结果进行计算")
    query2 = "现在请用第一步的结果来计算：第一个结果加上100，第二个结果减去20"
    print(f"查询: {query2}\n")
    
    async for raw, messages in math_calculator(history=history, query=query2):
        print_response(raw)


async def demo_performance():
    """演示：性能对比"""
    print("\n" + "="*60)
    print("  演示 4: 性能分析 - 并行 vs 顺序")
    print("="*60)
    
    print("\n场景: 并行执行 3 个计算")
    print("预期:")
    print("  - 顺序执行: 0.3 + 0.4 + 0.35 = 1.05 秒")
    print("  - 并行执行: max(0.3, 0.4, 0.35) = 0.4 秒")
    print("  - 性能提升: 1.05 / 0.4 = 2.6 倍\n")
    
    query = "请同时计算：5+10、15*2、30-8"
    print(f"执行查询: {query}\n")
    
    start = asyncio.get_event_loop().time()
    
    history: List[Dict[str, Any]] = []
    async for raw, messages in math_calculator(history=history, query=query):
        print_response(raw)
    
    elapsed = asyncio.get_event_loop().time() - start
    print(f"\n✓ 实际执行时间: {elapsed:.2f} 秒")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("  SimpleLLMFunc 并行工具调用 - 简化示例")
    print("="*60)
    print("\n本示例展示 SimpleLLMFunc 框架如何并发执行多个工具调用")
    print("所有工具通过 asyncio.gather() 并发执行，大幅提高性能\n")
    
    try:
        await demo_1()
        await demo_2()
        await demo_3()
        await demo_performance()
        
        print("\n" + "="*60)
        print("  所有演示完成")
        print("="*60)
        print("\n✓ 并行工具调用示例执行成功！")
        print("\n关键要点:")
        print("  1. LLM 可以在单个回复中请求多个工具调用")
        print("  2. 框架自动并发执行这些工具")
        print("  3. 工具结果按原始顺序追加到消息历史")
        print("  4. 支持多轮对话，后续查询可以利用前面的结果\n")
        
    except KeyboardInterrupt:
        print("\n[⚠] 用户中断执行")
    except Exception as e:
        print(f"\n[✗] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
