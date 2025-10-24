"""
同时支持 Sync 和 Async 被装饰函数的示例

这个示例展示了装饰器如何以低成本支持同时装饰 sync 和 async 函数。
虽然被装饰函数是 sync 还是 async 无关紧要（因为函数体从不执行），
但支持两种方式提供了更灵活的 API，让用户可以按照自己的习惯定义被装饰函数。

关键要点：
1. 装饰器装饰完的函数一定是 async 的
2. 但被装饰的原始函数可以是 sync 或 async
3. 原始函数的函数体根本不会被执行
4. Prompt 构建逻辑完全是 sync 的
5. 只有 LLM 调用是 async 的
"""

import asyncio
import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from SimpleLLMFunc import llm_chat, llm_function, tool, app_log
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible
from SimpleLLMFunc.type import Text

# 当前脚本文件所在的文件夹下的provider.json文件
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")
llm_interface = OpenAICompatible.load_from_json_file(provider_json_path)[
    "volc_engine"
]["deepseek-v3-250324"]


# ===== llm_function 装饰器：同时支持 sync 和 async =====


class SentimentResult(BaseModel):
    """情感分析结果"""

    text: str = Field(..., description="输入的文本")
    sentiment: str = Field(..., description="情感：positive, neutral, negative")
    confidence: float = Field(..., description="置信度 0-1")
    reason: str = Field(..., description="分析理由")


# 方式 1: 使用 sync 原始函数（新方式！）
@llm_function(llm_interface=llm_interface)
def analyze_sentiment_sync(text: str) -> SentimentResult:
    """
    分析输入文本的情感倾向。

    返回包含情感标签、置信度和分析理由的结构化数据。
    """
    ...  # 函数体不会被执行，使用 ... 作为占位符


# 方式 2: 使用 async 原始函数（传统方式）
@llm_function(llm_interface=llm_interface)
async def analyze_sentiment_async(text: str) -> SentimentResult:
    """
    分析输入文本的情感倾向。

    返回包含情感标签、置信度和分析理由的结构化数据。
    """
    ...  # 函数体不会被执行，使用 ... 作为占位符


# ===== llm_chat 装饰器：同时支持 sync 和 async =====


# 方式 1: 使用 sync 原始函数（新方式！）
@llm_chat(llm_interface=llm_interface, stream=False)
def simple_qa_sync(history: Optional[List[Dict[str, str]]] = None, question: str = ""):
    """
    一个简单的 QA 助手。我可以回答关于各种主题的问题。
    
    请注意：这是一个 sync 原始函数，但装饰后返回的是 async 函数。
    """
    ...


# 方式 2: 使用 async 原始函数（传统方式）
@llm_chat(llm_interface=llm_interface, stream=True)
async def streaming_chat_async(
    history: Optional[List[Dict[str, str]]] = None, message: str = ""
):
    """
    一个支持流式输出的聊天助手。我可以帮助您进行实时对话。
    
    注意：这是一个 async 原始函数，装饰后返回的也是 async 函数。
    """
    ...


# ===== 测试函数 =====


async def test_llm_function_sync_and_async():
    """测试 llm_function 装饰器同时支持 sync 和 async"""
    console = Console()
    
    console.print(
        Panel(
            "[bold cyan]测试 llm_function 装饰器：sync vs async[/bold cyan]",
            style="bold blue",
        )
    )

    test_texts = [
        "这个产品真的太棒了，我非常喜欢！",
        "这个服务还可以，没什么特别的。",
        "我感到非常失望，完全不符合期望。",
    ]

    for i, text in enumerate(test_texts, 1):
        console.print(f"\n[bold yellow]测试 {i}:[/bold yellow] {text}")

        try:
            # 虽然 analyze_sentiment_sync 是 sync 函数，但装饰后必须 await
            result_sync = await analyze_sentiment_sync(text)
            console.print(f"[green]Sync 方式结果:[/green]")
            console.print(f"  情感: {result_sync.sentiment}")
            console.print(f"  置信度: {result_sync.confidence}")
            console.print(f"  理由: {result_sync.reason}")

            # 传统的 async 方式
            result_async = await analyze_sentiment_async(text)
            console.print(f"[green]Async 方式结果:[/green]")
            console.print(f"  情感: {result_async.sentiment}")
            console.print(f"  置信度: {result_async.confidence}")
            console.print(f"  理由: {result_async.reason}")

        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")


async def test_llm_chat_sync_and_async():
    """测试 llm_chat 装饰器同时支持 sync 和 async"""
    console = Console()
    
    console.print(
        Panel(
            "[bold cyan]测试 llm_chat 装饰器：sync vs async[/bold cyan]",
            style="bold blue",
        )
    )

    # 第一个问题 - 使用 sync 原始函数
    console.print("\n[bold]问题 1 (使用 sync 原始函数):[/bold]")
    console.print("Q: Python 中的 async/await 是什么？")
    console.print("[bold cyan]A:[/bold cyan] ", end="")

    try:
        full_response = ""
        async for content, history in simple_qa_sync(
            history=[], question="Python 中的 async/await 是什么？"
        ):
            if content:
                console.print(content, end="", highlight=False)
                full_response += content
        console.print()  # 换行
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")

    # 第二个问题 - 使用 async 原始函数（流式输出）
    console.print("\n[bold]问题 2 (使用 async 原始函数，流式输出):[/bold]")
    console.print("Q: 什么是装饰器？")
    console.print("[bold cyan]A:[/bold cyan] ", end="")

    try:
        async for content, history in streaming_chat_async(
            history=[], message="什么是装饰器？"
        ):
            if content:
                console.print(content, end="", highlight=False)
        console.print()  # 换行
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")


async def comparison_demo():
    """对比演示 - 显示两种方式的等价性"""
    console = Console()
    
    console.print(
        Panel(
            "[bold cyan]对比演示：Sync vs Async 被装饰函数[/bold cyan]",
            style="bold blue",
        )
    )

    # 创建对比表格
    table = Table(title="Sync vs Async 被装饰函数对比")
    table.add_column("特性", style="cyan")
    table.add_column("Sync 原始函数", style="green")
    table.add_column("Async 原始函数", style="yellow")

    table.add_row(
        "原始函数定义",
        "def func(text: str) -> str:",
        "async def func(text: str) -> str:",
    )
    table.add_row("装饰后的返回类型", "Coroutine", "Coroutine")
    table.add_row("调用方式", "await func(...)", "await func(...")
    table.add_row("原始函数体执行", "❌ 不执行", "❌ 不执行")
    table.add_row("Prompt 构建", "✅ 同步", "✅ 同步")
    table.add_row("LLM 调用", "✅ 异步", "✅ 异步")
    table.add_row("实际成本差异", "❌ 无", "❌ 无")

    console.print(table)

    console.print("\n[bold green]关键结论:[/bold green]")
    console.print(
        "1. ✅ 两种方式装饰后的函数都是 async 的，都需要 await"
    )
    console.print("2. ✅ 原始函数的函数体根本不会执行")
    console.print("3. ✅ Prompt 构建完全是同步的，与被装饰函数类型无关")
    console.print("4. ✅ 只有 LLM 调用是异步的")
    console.print("5. ✅ 支持两种方式的代价极低（仅需类型签名调整）")
    console.print("6. ✅ 用户可以自由选择 sync 或 async 方式定义被装饰函数")


async def concurrent_examples():
    """并发示例 - 同时执行多个被装饰函数"""
    console = Console()
    
    console.print(
        Panel(
            "[bold cyan]并发执行示例[/bold cyan]",
            style="bold blue",
        )
    )

    async def process_sentiment(text: str, use_sync: bool) -> str:
        """处理情感分析"""
        try:
            if use_sync:
                result = await analyze_sentiment_sync(text)
                return f"[sync] {text[:20]}... → {result.sentiment}"
            else:
                result = await analyze_sentiment_async(text)
                return f"[async] {text[:20]}... → {result.sentiment}"
        except Exception as e:
            return f"Error: {str(e)[:50]}"

    # 并发执行多个任务
    tasks = [
        process_sentiment("这真是太好了！", use_sync=True),
        process_sentiment("一般般吧", use_sync=False),
        process_sentiment("非常失望", use_sync=True),
        process_sentiment("还不错", use_sync=False),
    ]

    console.print("\n[yellow]并发执行 4 个情感分析任务...[/yellow]")
    results = await asyncio.gather(*tasks)

    for result in results:
        console.print(f"  {result}")

    console.print("[green]✅ 所有任务完成[/green]")


async def main():
    """主函数"""
    console = Console()

    # 标题
    title_panel = Panel(
        "[bold magenta]Sync/Async 被装饰函数支持示例[/bold magenta]\n"
        "[dim]展示装饰器如何以低成本同时支持 sync 和 async 被装饰函数[/dim]",
        style="bright_blue",
    )
    console.print(title_panel)

    # 运行所有示例
    try:
        await comparison_demo()
        await test_llm_function_sync_and_async()
        await test_llm_chat_sync_and_async()
        await concurrent_examples()

        console.print(
            "\n[bold green]🎉 所有示例运行完成！[/bold green]"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]发生错误: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())
