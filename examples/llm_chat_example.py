"""
异步 LLM Chat装饰器使用示例

这个示例展示了如何使用 @async_llm_chat 装饰器来创建异步聊天功能。
重点展示异步聊天的并发能力和流式响应。
"""

import asyncio
import os
from typing import Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from SimpleLLMFunc import async_llm_chat
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible


# 当前脚本文件所在的文件夹下的provider.json文件
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")

# 使用与async_llm_func.py相同的方式创建LLM接口
VolcEngine_deepseek_v3_Interface = OpenAICompatible.load_from_json_file(
    provider_json_path
)["volc_engine"]["deepseek-v3-250324"]


@async_llm_chat(llm_interface=VolcEngine_deepseek_v3_Interface, stream=True)
async def async_simple_chat(
    history: Optional[List[Dict[str, str]]] = None, message: str = ""
):
    """
    一个异步的简单聊天助手。我可以帮助您解答问题、进行对话或协助完成各种任务。
    """
    pass


@async_llm_chat(llm_interface=VolcEngine_deepseek_v3_Interface, stream=False)
async def async_programming_assistant(
    history: Optional[List[Dict[str, str]]] = None, code: str = "", question: str = ""
):
    """
    异步编程助手：我专门帮助解决编程问题。
    我可以：
    - 解释代码逻辑
    - 发现代码中的问题
    - 建议改进方案
    - 提供最佳实践建议
    """
    pass


async def async_chat_example():
    """异步聊天示例"""
    console = Console()
    console.print("[bold cyan]=== 异步聊天示例 ===[/bold cyan]")

    # 初始化历史记录
    history = []

    # 第一轮对话
    console.print("\n[bold]用户:[/bold] 你好！介绍一下你自己")
    console.print("[bold]助手:[/bold] ", end="")

    full_response = ""
    async for content, updated_history in async_simple_chat(
        history=history, message="你好！介绍一下你自己"
    ):
        if content:  # 忽略空内容（最后的历史记录更新）
            console.print(content, end="")
            full_response += content
        else:
            # 更新历史记录
            history = updated_history
    console.print()  # 换行

    # 第二轮对话
    console.print("\n[bold]用户:[/bold] 编写一个Python函数来计算斐波那契数列")
    console.print("[bold]助手:[/bold] ", end="")

    full_response = ""
    async for content, updated_history in async_simple_chat(
        history=history, message="编写一个Python函数来计算斐波那契数列"
    ):
        if content:
            console.print(content, end="")
            full_response += content
        else:
            history = updated_history
    console.print()


async def concurrent_chat_example():
    """并发聊天示例 - 使用Rich库展示分区域实时输出"""
    console = Console()

    console.print("\n[bold cyan]=== 并发聊天示例（Rich TUI界面） ===[/bold cyan]")
    console.print("同时发起3个异步聊天请求，每个问题在独立区域显示...\n")

    questions = [
        "请介绍一下人工智能的发展历史（请详细说明）",
        "请解释一下什么是深度学习及其应用",
        "请谈谈机器学习在现实生活中的具体应用案例",
    ]

    # 创建布局
    layout = Layout()
    layout.split_column(
        Layout(name="top"), Layout(name="middle"), Layout(name="bottom")
    )

    # 状态追踪
    status = {
        "top": {"content": "", "status": "准备中...", "question": questions[0]},
        "middle": {"content": "", "status": "准备中...", "question": questions[1]},
        "bottom": {"content": "", "status": "准备中...", "question": questions[2]},
    }

    def update_layout():
        """更新布局显示"""
        for i, (area_name, data) in enumerate(status.items(), 1):
            # 构建显示内容
            content_text = f"[bold blue]问题 {i}:[/bold blue] {data['question']}\n\n"
            content_text += f"[bold green]状态:[/bold green] {data['status']}\n\n"

            if data["content"]:
                content_text += f"[bold yellow]回答:[/bold yellow]\n{data['content']}"
            else:
                content_text += "[dim]等待回答...[/dim]"

            # 选择边框颜色
            if data["status"] == "完成":
                border_style = "green"
            elif "进行中" in data["status"]:
                border_style = "yellow"
            elif "错误" in data["status"]:
                border_style = "red"
            else:
                border_style = "blue"

            # 更新对应区域
            layout[area_name].update(
                Panel(
                    content_text,
                    title=f"[bold]聊天区域 {i}[/bold]",
                    border_style=border_style,
                    padding=(1, 2),
                )
            )

    async def process_question_with_ui(area_name: str, question: str):
        """处理单个问题并更新UI"""
        try:
            status[area_name]["status"] = "连接中..."

            # 开始聊天
            async for content, _ in async_simple_chat(history=[], message=question):
                if content:
                    status[area_name]["content"] += content
                    char_count = len(status[area_name]["content"])
                    status[area_name]["status"] = f"进行中... ({char_count} 字符)"

            status[area_name]["status"] = "完成"

        except Exception as e:
            status[area_name]["status"] = f"错误: {str(e)}"
            status[area_name]["content"] = f"处理问题时发生错误: {str(e)}"

    # 使用 Live 显示，实时更新
    with Live(layout, refresh_per_second=8, screen=False) as _:
        # 初始显示
        update_layout()

        # 创建并发任务
        tasks = [
            process_question_with_ui("top", questions[0]),
            process_question_with_ui("middle", questions[1]),
            process_question_with_ui("bottom", questions[2]),
        ]

        # 启动一个定时器来更新显示
        async def update_display():
            while any(
                data["status"] not in ["完成", "错误"]
                or not data["status"].startswith("错误")
                for data in status.values()
            ):
                update_layout()
                await asyncio.sleep(0.125)  # 每125ms更新一次
            update_layout()  # 最后更新一次

        # 并发执行任务和显示更新
        await asyncio.gather(*tasks, update_display())

        # 最终显示
        update_layout()

    console.print("\n[bold green]所有问题处理完成！[/bold green]")


async def main():
    """主函数"""
    console = Console()

    # 创建标题
    title = Text("异步 LLM Chat装饰器使用示例", style="bold magenta")
    console.print(Panel(title, style="bright_blue"))

    # 运行所有异步示例
    await async_chat_example()
    await concurrent_chat_example()

    console.print("\n[bold green]🎉 所有示例运行完成！[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
