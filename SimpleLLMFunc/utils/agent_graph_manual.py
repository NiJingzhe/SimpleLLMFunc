"""
Manual Agent Flow Visualization (no AST/static parsing)

Goal:
  - Let users manually declare a directed agent/control-flow graph.
  - Nodes can represent @llm_chat/@llm_function agents, and also "logic" blocks.
  - Control-flow nodes (if/for/while) are represented as dedicated nodes whose
    condition/init/update are provided as natural-language strings.

Output:
  - Mermaid flowchart text (.mmd)
  - Graphviz DOT text (.dot)

Design notes:
  - This module is "manual": 图由用户显式声明节点与有向边，不做 AST/静态解析推断。
  - 节点 label 会被直接嵌入到 Mermaid/DOT 输出中，因此建议用自然语言做清晰描述。
  - 对于 if/for/while：用专门的控制流节点承载 condition/init/update，然后通过有向边与边标签表达分支/循环回边。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence


class NodeKind(str, Enum):
    """
    Node kind：节点在图中的语义角色。

    该枚举影响两件事：
      1) 语义：这个节点到底代表什么（Agent/Tool/普通逻辑/控制流）。
      2) 渲染：导出到 Mermaid/DOT 时采用的形状（diamond/oval/box/note）。

    你的目标场景（可视化粒度到 Agent 级别）通常映射为：
      - `@llm_chat/@llm_function`：用 `AGENT` 节点表示“调用点”（黑盒）。
      - `@tool`：用 `TOOL` 节点表示“工具执行点”（模型 function call 的对应结果）。
      - 非 Agent 的通用 Python 逻辑：用 `LOGIC` 节点表示（用自然语言描述代码块做了什么）。
      - 控制流：用 `IF/FOR/WHILE` 专门节点表达条件/初始化/更新，然后用有向边把 then/else 或循环回边连接起来。
      - `MERGE`：把分支汇合到单一执行流。
      - `ENTRY/EXIT`：入口/出口。
    """

    ENTRY = "entry"  # 入口节点：开始执行/接收输入/进入主流程，推荐填写主Agent名称
    AGENT = "agent"  # Agent 节点：一个 `@llm_chat/@llm_function` 的调用点（黑盒）
    TOOL = "tool"    # 工具节点：一个 `@tool` 工具的执行点
    LOGIC = "logic"  # 普通逻辑节点：非 Agent/非 Tool 的代码块逻辑（自然语言描述）

    IF = "if"        # 分支节点：条件（then/else）由边标签与有向边共同表达
    FOR = "for"      # for 循环节点：init/condition/update 用自然语言承载
    WHILE = "while"  # while 循环节点：init/condition/update 用自然语言承载

    MERGE = "merge"  # 汇合节点：把多分支收敛到同一路径（便于理解与布局）
    EXIT = "exit"    # 出口节点：结束执行/返回最终结果


def _escape_mermaid_label(text: str) -> str:
    """
    Escape/normalize node/edge labels for Mermaid.

    Why:
      - Some Mermaid renderers/compilers do not like raw HTML tags like `<br/>`
        inside quoted labels, which can lead to parse errors.
      - We want a robust "line break" representation that compiles everywhere.

    Strategy:
      - Convert Mermaid-style `<br/>` / `<br />` / `<br>` to literal `\\n`.
      - Convert actual newline characters to `\\n`.
      - Convert double quotes to single quotes to keep the label within quotes safe.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = (
        normalized.replace("<br/>", "\\n")
        .replace("<br />", "\\n")
        .replace("<br>", "\\n")
    )
    # Also convert real newline into \n to keep the label on a single line.
    normalized = normalized.replace("\n", "\\n")
    # Keep quotes safe inside: ... "label" ...
    normalized = normalized.replace('"', "'")
    return normalized


def _sanitize_node_id(node_id: str) -> str:
    # Mermaid and DOT both require stable identifiers. Keep it simple.
    out = []
    for ch in node_id:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    v = "".join(out).strip("_")
    return v or "node"


@dataclass
class Node:
    id: str
    kind: NodeKind
    label: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class AgentFlowGraph:
    """
    手动声明的有向图（directed graph），用于可视化 Agent/control-flow。

    你需要显式完成两件事：
      - 节点声明：通过 `add_entry/add_agent_node/add_tool_node/add_logic_node/...`
      - 有向边声明：通过 `add_edge(src, dst, label=...)`

    导出的 Mermaid/DOT 图只是“展示”，不会参与执行；因此你可以按自己的理解粒度组织节点与边。
    """

    def __init__(self, title: str = "", direction: Literal["TD", "LR"] = "TD") -> None:
        self.title = title
        self.direction = direction
        self._nodes: Dict[str, Node] = {}
        self._node_order: List[str] = []
        self._edges: List[Edge] = []

    def _ensure_id(self, node_id: str) -> str:
        node_id = _sanitize_node_id(node_id)
        if node_id in self._nodes:
            raise ValueError(f"Duplicate node_id: {node_id}")
        return node_id

    def _add_node(self, node_id: str, kind: NodeKind, label: str, **meta: Any) -> str:
        nid = self._ensure_id(node_id)
        self._nodes[nid] = Node(id=nid, kind=kind, label=label, meta=meta)
        self._node_order.append(nid)
        return nid

    # ---------- Node helpers ----------

    def add_entry(self, node_id: str, label: str = "Start") -> str:
        """添加 ENTRY 节点。

        ENTRY 通常位于图的最上游，用来表示：
          - 开始执行
          - 用户输入进入主流程
          - 或进入某个子流程（如果你把图拆成多个子图）
        """
        return self._add_node(node_id, NodeKind.ENTRY, label)

    def add_exit(self, node_id: str, label: str = "End") -> str:
        """添加 EXIT 节点。

        EXIT 通常位于图的最下游，用来表示：
          - 已生成最终输出（FinalReport）
          - 或循环/流程终止
        """
        return self._add_node(node_id, NodeKind.EXIT, label)

    def add_merge(self, node_id: str, label: str = "Merge") -> str:
        """添加 MERGE 节点。

        MERGE 节点常用于：
          - IF 的 then/else 两条路径汇合
          - 循环退出后与其他逻辑合流
        """
        return self._add_node(node_id, NodeKind.MERGE, label)

    def add_agent_node(
        self,
        node_id: str,
        agent: Any = None,
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """添加 AGENT 节点（Agent 级别调用点）。

        语义：
          - 代表一个被 `@llm_chat` 或 `@llm_function` 装饰的函数“被调用”的位置（黑盒）。
        实践建议：
          - label 建议包含：agent 名称 + 装饰器信息（例如 "@llm_chat"）
          - description 可用于补充该 Agent 的输出/契约（例如需要 PASS/FAIL）
        """
        if label is None:
            if agent is None:
                label = "Agent"
            else:
                label = getattr(agent, "__name__", None) or agent.__class__.__name__
        if description:
            label = f"{label}<br/>---<br/>{description}"
        return self._add_node(node_id, NodeKind.AGENT, label, agent=agent)

    def add_tool_node(
        self,
        node_id: str,
        tool: Any = None,
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """添加 TOOL 节点（工具执行点）。

        语义：
          - 代表一个被 `@tool` 注册的工具“被执行”的位置。
        你可以选择展示的粒度：
          - 展示：让图更直观地体现工具调用链路
          - 不展示：把工具调用当作 Agent 节点内部细节，直接写进 label/description
        """
        if label is None:
            if tool is None:
                label = "Tool"
            else:
                label = getattr(tool, "__name__", None) or tool.__class__.__name__
        if description:
            label = f"{label}<br/>---<br/>{description}"
        return self._add_node(node_id, NodeKind.TOOL, label, tool=tool)

    def add_logic_node(self, node_id: str, description: str) -> str:
        """添加 LOGIC 节点（普通逻辑块）。

        语义：
          - 代表非 Agent/非 Tool 的通用逻辑代码块。
          - 你可以用自然语言描述它“做了什么”，让图把关键步骤纳入可视化。
        常见用途：
          - 构建 prompt（拼接 plan/execution_result）
          - 解析审核结果（PASS/FAIL）并生成下一轮反馈
          - 格式化最终报告
        """
        return self._add_node(node_id, NodeKind.LOGIC, description)

    def add_if_node(
        self,
        node_id: str,
        condition: str,
        then_label: str = "Yes",
        else_label: str = "No",
    ) -> str:
        """添加 IF 节点（分支）。

        该节点本身不直接决定 then/else 的路径，关键在你如何连边：
          - 通常会有两条边：IF -> then_branch 和 IF -> else_branch
          - 边的 label 可以写成 "PASS" / "FAIL" / "True" / "False" 等
        """
        label = (
            f"if：{condition}<br/>"
            f"then：{then_label}<br/>"
            f"else：{else_label}"
        )
        return self._add_node(node_id, NodeKind.IF, label, condition=condition)

    def add_while_node(
        self,
        node_id: str,
        init: str,
        condition: str,
        update: str,
    ) -> str:
        """添加 WHILE 节点（while 循环控制流）。

        init/condition/update 都用自然语言提供在节点 label 中。
        你需要通过有向边来体现循环结构，例如：
          - entry -> while
          - while -> body (满足条件的路径)
          - body -> while (回边)
          - while -> exit/merge (条件不满足的退出路径)
        """
        label = f"while<br/>初始化：{init}<br/>条件：{condition}<br/>更新：{update}"
        return self._add_node(node_id, NodeKind.WHILE, label, init=init, condition=condition, update=update)

    def add_for_node(
        self,
        node_id: str,
        init: str,
        condition: str,
        update: str,
    ) -> str:
        """添加 FOR 节点（for 循环控制流）。

        与 WHILE 类似：init/condition/update 用自然语言承载，然后由你手动连边来体现迭代与退出。
        """
        label = f"for<br/>初始化：{init}<br/>条件：{condition}<br/>更新：{update}"
        return self._add_node(node_id, NodeKind.FOR, label, init=init, condition=condition, update=update)

    # ---------- Edge helpers ----------

    def add_edge(self, src: str, dst: str, label: str = "", **meta: Any) -> None:
        """添加有向边 src -> dst。

        参数说明：
          - src/dst：必须在图中已存在（否则会抛错，避免“画出来但其实断链”的图）
          - label：边标签（强烈建议用于表达控制流含义，比如 PASS/FAIL/next-round）
          - meta：预留给未来扩展（例如你可以存“边的触发原因/变量变化”等元数据）
        """
        src = _sanitize_node_id(src)
        dst = _sanitize_node_id(dst)
        if src not in self._nodes:
            raise KeyError(f"src node not found: {src}")
        if dst not in self._nodes:
            raise KeyError(f"dst node not found: {dst}")
        self._edges.append(Edge(src=src, dst=dst, label=label, meta=meta))

    # ---------- Export ----------

    def to_mermaid(self) -> str:
        """
        Mermaid flowchart output.

        Notes:
          - Labels with newlines are normalized to <br/>.
          - Node shapes:
              ENTRY/EXIT/MERGE: oval
              AGENT/TOOL/LOGIC: rectangle / note-like
              IF/FOR/WHILE: diamond
        """
        lines: List[str] = []
        if self.title:
            lines.append(f"%% {self.title}")
        lines.append(f"flowchart {self.direction}")

        for nid in self._node_order:
            node = self._nodes[nid]
            label = _escape_mermaid_label(node.label)

            if node.kind in (NodeKind.IF, NodeKind.WHILE, NodeKind.FOR):
                # Diamond
                lines.append(f'{nid}{{"{label}"}}')
            elif node.kind in (NodeKind.ENTRY, NodeKind.EXIT, NodeKind.MERGE):
                # Rounded / circle-ish
                lines.append(f'{nid}(["{label}"])')
            elif node.kind == NodeKind.LOGIC:
                # Subtle differentiation: use note shape (still rendered as rectangle in many Mermaid engines)
                lines.append(f'{nid}["{label}"]')
            else:
                # AGENT/TOOL default rectangle
                lines.append(f'{nid}["{label}"]')

        for e in self._edges:
            if e.label:
                elabel = _escape_mermaid_label(e.label)
                lines.append(f'{e.src} -->|"{elabel}"| {e.dst}')
            else:
                lines.append(f"{e.src} --> {e.dst}")

        return "\n".join(lines) + "\n"

    def save_mermaid(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_mermaid(), encoding="utf-8")
        return p

    def to_dot(self) -> str:
        """
        Graphviz DOT output.
        """
        lines: List[str] = []
        lines.append("digraph AgentFlow {")
        lines.append('  rankdir="TB";')
        lines.append('  node [fontname="Helvetica"];')

        # Nodes
        for nid in self._node_order:
            node = self._nodes[nid]
            # Node label 里可能包含 Mermaid 风格的 <br/>，Graphviz 不认识它，
            # 需要转成 DOT 可识别的换行符。
            dot_label = (
                node.label.replace("<br/>", "\n")
                .replace("<br />", "\n")
                .replace("<br>", "\n")
            )
            label = dot_label.replace('"', '\\"').replace("\n", "\\n")

            if node.kind in (NodeKind.IF, NodeKind.WHILE, NodeKind.FOR):
                shape = "diamond"
            elif node.kind in (NodeKind.ENTRY, NodeKind.EXIT, NodeKind.MERGE):
                shape = "oval"
            elif node.kind == NodeKind.LOGIC:
                shape = "note"
            else:
                shape = "box"

            lines.append(f'  {nid} [label="{label}", shape="{shape}"];')

        # Edges
        for e in self._edges:
            if e.label:
                dot_edge_label = (
                    e.label.replace("<br/>", "\n")
                    .replace("<br />", "\n")
                    .replace("<br>", "\n")
                )
                label = dot_edge_label.replace('"', '\\"').replace("\n", "\\n")
                lines.append(f'  {e.src} -> {e.dst} [label="{label}"];')
            else:
                lines.append(f"  {e.src} -> {e.dst};")

        lines.append("}")
        return "\n".join(lines) + "\n"

    def save_dot(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_dot(), encoding="utf-8")
        return p


__all__ = [
    "AgentFlowGraph",
    "NodeKind",
]


# ============================================================================
# Manual graph declaration (annotation)
# ============================================================================

# Function attribute key for discovery by the unified generator script.
MANUAL_AGENT_FLOW_GRAPH_ATTR = "__manual_agent_flow_graph__"


def manual_agent_flow_graph(
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """
    Decorator for declaring a manual agent flow graph.

    Usage pattern:
      1) Define a function that returns an `AgentFlowGraph` instance.
      2) Annotate it with `@manual_agent_flow_graph(title=..., description=...)`.
      3) Run the unified generator script, which scans for this attribute and
         generates files into: <output_dir>/<function_name>/.

    Why this approach is "non-invasive":
      - The graph builder code remains a pure function returning data.
      - The export/generation concern is centralized in one script.
    """

    def _decorator(fn: Any) -> Any:
        setattr(
            fn,
            MANUAL_AGENT_FLOW_GRAPH_ATTR,
            {
                "title": title,
                "description": description,
            },
        )
        return fn

    return _decorator


__all__ += [
    "manual_agent_flow_graph",
    "MANUAL_AGENT_FLOW_GRAPH_ATTR",
]

