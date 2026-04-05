"""
复杂多 Agent 辩论流示例（研究员A/B -> 仲裁 -> 再辩）

运行：
    conda run -n simple_llm_func_demo python examples/multi_agent_research_debate.py
"""

from __future__ import annotations

import asyncio
import os
import importlib.util
import sys
from typing import Any
from pathlib import Path

# ------------------------------------------------------------
# Manual flow-graph support (safe import)
# ------------------------------------------------------------
# IMPORTANT:
# - Graph generator may run in an environment where the full SimpleLLMFunc
#   package dependencies are not installed (e.g. missing `pydantic_settings`).
# - To keep this file importable for "graph generation", we load only the
#   lightweight agent_graph_manual utilities by file path.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_manual_path = (PROJECT_ROOT / "SimpleLLMFunc" / "utils" / "agent_graph_manual.py").resolve()
_spec = importlib.util.spec_from_file_location("agent_graph_manual", str(_manual_path))
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load agent_graph_manual from: {_manual_path}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

AgentFlowGraph = _mod.AgentFlowGraph
manual_agent_flow_graph = _mod.manual_agent_flow_graph


# ------------------------------------------------------------
# LLM runtime imports (may fail if deps are missing)
# ------------------------------------------------------------
try:
    from SimpleLLMFunc import OpenAICompatible, llm_chat
    from SimpleLLMFunc.type import HistoryList
except Exception:
    OpenAICompatible = None  # type: ignore
    llm_chat = None  # type: ignore
    HistoryList = Any  # type: ignore


def load_llm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    provider_json_path = os.path.join(current_dir, "provider.json")
    models = OpenAICompatible.load_from_json_file(provider_json_path)
    return models["deepseek"]["deepseek/deepseek-v3.2"]


if OpenAICompatible is not None and llm_chat is not None:
    llm = load_llm()
else:
    llm = None


@manual_agent_flow_graph(
    title="Multi Agent Research Debate",
    description="研究员A/B辩论 -> 仲裁 -> 平局交叉质询 -> 再裁决，最多 rounds=2 轮。",
)
def build_multi_agent_research_debate_flow() -> AgentFlowGraph:
    """
    Build a manual flow graph for this example controller.

    Notes:
      - 图的构建是“声明式”的：只返回 AgentFlowGraph，不执行任何 LLM 调用。
      - 节点粒度以 Agent 为主；控制流用 if/for 节点表达。
    """

    g = AgentFlowGraph(title="Multi Agent Research Debate", direction="TD")

    entry = g.add_entry("entry_node", "Start: topic 入参进入控制器")

    # for i in range(rounds=2)
    rounds_for = g.add_for_node(
        "round_for",
        init="i=0",
        condition="i < rounds(=2)",
        update="i += 1",
    )

    ask_a = g.add_agent_node("ask_a", label="researcher_a (@llm_chat) 生成 A观点")
    ask_b = g.add_agent_node("ask_b", label="researcher_b (@llm_chat) 生成 B观点")

    # judge_prompt 是 controller 里的普通逻辑（把 A/B 拼成仲裁输入）
    make_judge_prompt = g.add_logic_node(
        "make_judge_prompt",
        "构建 judge_prompt：把 A观点与 B观点拼接成仲裁输入",
    )

    judge_1 = g.add_agent_node(
        "judge_1",
        label="arbiter_agent (@llm_chat) 第一次裁决（WINNER: A/B/TIE）",
    )

    decide = g.add_if_node(
        "decide_winner",
        condition="upper_judge 含 WINNER: A 或 WINNER: B？",
        then_label="非平局：更新比分后进入下一轮",
        else_label="平局：进入交叉质询并再裁决",
    )

    update_normal = g.add_logic_node(
        "update_normal",
        "更新 winner_score：如果 WINNER: A 则 A+=1；如果 WINNER: B 则 B+=1",
    )
    merge_round = g.add_merge("merge_round", "Merge / 汇总本轮产物")

    # TIE path
    cross_a = g.add_agent_node(
        "cross_a",
        label="researcher_a (@llm_chat) 交叉质询：指出对方最大漏洞并给修正",
    )
    cross_b = g.add_agent_node(
        "cross_b",
        label="researcher_b (@llm_chat) 交叉质询：指出对方最大漏洞并给修正",
    )
    make_second_judge_prompt = g.add_logic_node(
        "make_second_judge_prompt",
        "构建 second_judge_prompt：把交叉质询结果拼接后交给仲裁",
    )
    judge_2 = g.add_agent_node(
        "judge_2",
        label="arbiter_agent (@llm_chat) 再裁决（WINNER: A/B）",
    )
    update_tie = g.add_logic_node(
        "update_tie",
        "更新 winner_score：根据第二次裁决结果对 A/B 计分",
    )

    finalize = g.add_logic_node(
        "finalize",
        "循环结束后：计算 final_winner（A/B/TIE），拼接 transcript 并返回",
    )
    exit_node = g.add_exit("exit_node", "End: 输出最终辩论记录与 Winner")

    # ----- Edges (manual control-flow) -----
    g.add_edge(entry, rounds_for, "进入 for 循环控制")

    # for true -> body start
    g.add_edge(rounds_for, ask_a, "进入第 i 轮")

    # body sequence
    g.add_edge(ask_a, ask_b, "获得 A观点后生成 B观点")
    g.add_edge(ask_b, make_judge_prompt, "生成仲裁输入")
    g.add_edge(make_judge_prompt, judge_1, "调用仲裁 Agent（第1次裁决）")
    g.add_edge(judge_1, decide, "解析 WINNER 字段")

    # decide branches
    g.add_edge(decide, update_normal, "WINNER: A/B -> 更新比分")
    g.add_edge(update_normal, merge_round, "结束本轮（非平局路径）")

    g.add_edge(decide, cross_a, "WINNER: TIE/其他 -> 进入交叉质询")
    g.add_edge(cross_a, cross_b, "获得交叉质询A补充后生成交叉质询B")
    g.add_edge(cross_b, make_second_judge_prompt, "生成第二次仲裁输入")
    g.add_edge(make_second_judge_prompt, judge_2, "调用仲裁 Agent（第2次裁决）")
    g.add_edge(judge_2, update_tie, "解析第二次 WINNER")
    g.add_edge(update_tie, merge_round, "结束本轮（平局路径）")

    # back edge
    g.add_edge(merge_round, rounds_for, "i += 1，进入下一轮")

    # for false -> exit
    g.add_edge(rounds_for, finalize, "循环结束（i >= rounds）")
    g.add_edge(finalize, exit_node, "返回最终报告")

    return g


if llm_chat is not None and llm is not None:

    @llm_chat(llm_interface=llm, stream=True, temperature=0.9)
    async def researcher_a(message: str, history: HistoryList):
        """
        你是研究员A，偏向激进创新方案。
        输出：观点、证据、风险、适用边界。
        """

    @llm_chat(llm_interface=llm, stream=True, temperature=0.2)
    async def researcher_b(message: str, history: HistoryList):
        """
        你是研究员B，偏向稳健保守方案。
        输出：观点、证据、风险、适用边界。
        """

    @llm_chat(llm_interface=llm, stream=True, temperature=0.3)
    async def arbiter_agent(message: str, history: HistoryList):
        """
        你是仲裁 Agent。必须明确输出 WINNER: A 或 WINNER: B 或 WINNER: TIE。
        """

else:

    async def researcher_a(message: str, history: HistoryList):  # type: ignore
        raise RuntimeError(
            "SimpleLLMFunc runtime dependencies are missing; researcher_a is unavailable."
        )

    async def researcher_b(message: str, history: HistoryList):  # type: ignore
        raise RuntimeError(
            "SimpleLLMFunc runtime dependencies are missing; researcher_b is unavailable."
        )

    async def arbiter_agent(message: str, history: HistoryList):  # type: ignore
        raise RuntimeError(
            "SimpleLLMFunc runtime dependencies are missing; arbiter_agent is unavailable."
        )


async def _ask(
    agent: Any,
    message: str,
    history: HistoryList,
    label: str,
) -> tuple[str, HistoryList]:
    """
    将某个子 Agent 的 stream 输出实时打印到控制台，并返回最终文本与更新后的历史。
    """
    output = ""
    new_hist = history

    async for content, h in agent(message, history):
        content_str = str(content)
        new_hist = h

        # 注意：llm_chat(stream=True) 在结束时还会再 yield 一次 ("", history)
        # 这次不要覆盖 output，否则上层拼 judge_prompt 时会出现 A观点/B观点 为空。
        if not content_str:
            continue

        prev = output

        # 兼容两种 stream 形态：
        # 1) 增量 token：content_str 不是 output 的前缀 => output 需要累积拼接
        # 2) 累计全文：content_str 是 output 的前缀扩展 => 只打印新增片段
        if prev and content_str.startswith(prev):
            new_part = content_str[len(prev) :]
            output = content_str
        else:
            new_part = content_str
            output = prev + content_str

        if new_part:
            print(new_part, end="", flush=True)

    # 一轮输出结束补换行，避免与下一个 Agent 的 header 撞在一起
    if output:
        print()

    return output, new_hist


async def research_debate_controller(topic: str) -> str:
    history_a: HistoryList = []
    history_b: HistoryList = []
    history_judge: HistoryList = []

    rounds = 2
    winner_score = {"A": 0, "B": 0}
    transcript: list[str] = []

    for i in range(rounds):
        context = f"第 {i + 1} 轮辩论，主题: {topic}"
        print(f"\n================ Round {i + 1} ================")
        print("[Researcher A] input:")
        print(context)
        print("[Researcher A] output:")
        a_text, history_a = await _ask(
            researcher_a, context, history_a, label="A"
        )

        print("\n[Researcher B] input:")
        print(context)
        print("[Researcher B] output:")
        b_text, history_b = await _ask(
            researcher_b, context, history_b, label="B"
        )

        judge_prompt = (
            f"{context}\n\nA观点:\n{a_text}\n\nB观点:\n{b_text}\n"
            "请给出仲裁，并明确 WINNER 字段。"
        )
        print("\n[Arbiter] input:")
        print(judge_prompt)
        print("[Arbiter] output:")
        judge_text, history_judge = await _ask(
            arbiter_agent, judge_prompt, history_judge, label="Judge"
        )
        transcript.append(
            f"[Round {i + 1}] A:\n{a_text}\n\nB:\n{b_text}\n\nJudge:\n{judge_text}"
        )

        upper_judge = judge_text.upper()
        if "WINNER: A" in upper_judge:
            winner_score["A"] += 1
        elif "WINNER: B" in upper_judge:
            winner_score["B"] += 1
        else:
            # 若平局，追加一轮“交叉质询”
            cross_prompt_a = "请指出对方论证的最大漏洞，并给出可验证修正。"
            cross_prompt_b = "请指出对方论证的最大漏洞，并给出可验证修正。"
            print("\n[Round 平局] 进入交叉质询：")

            print("\n[Researcher A - Cross] input:")
            print(cross_prompt_a)
            print("[Researcher A - Cross] output:")
            cross_a, history_a = await _ask(
                researcher_a, cross_prompt_a, history_a, label="A-Cross"
            )

            print("\n[Researcher B - Cross] input:")
            print(cross_prompt_b)
            print("[Researcher B - Cross] output:")
            cross_b, history_b = await _ask(
                researcher_b, cross_prompt_b, history_b, label="B-Cross"
            )
            second_judge_prompt = (
                "以下是交叉质询结果，请重新裁决：\n"
                f"A补充:\n{cross_a}\n\nB补充:\n{cross_b}"
            )
            print("\n[Arbiter - ReJudge] input:")
            print(second_judge_prompt)
            print("[Arbiter - ReJudge] output:")
            second_judge, history_judge = await _ask(
                arbiter_agent, second_judge_prompt, history_judge, label="Judge-ReJudge"
            )
            transcript.append(f"[Round {i + 1} - ReJudge]\n{second_judge}")
            second_upper = second_judge.upper()
            if "WINNER: A" in second_upper:
                winner_score["A"] += 1
            elif "WINNER: B" in second_upper:
                winner_score["B"] += 1

        print(
            f"\n[Score] After Round {i + 1}: A={winner_score['A']}, B={winner_score['B']}"
        )

    final_winner = "TIE"
    if winner_score["A"] > winner_score["B"]:
        final_winner = "A"
    elif winner_score["B"] > winner_score["A"]:
        final_winner = "B"

    transcript.append(f"[FINAL] Winner={final_winner}, Score={winner_score}")
    return "\n\n".join(transcript)


async def main():
    topic = "是否应在高并发检索系统中默认启用激进缓存淘汰策略？"
    result = await research_debate_controller(topic)
    print("\n===== DEBATE RESULT =====\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

