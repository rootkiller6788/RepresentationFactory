"""血缘图。

与同工作区其它项目共用同一套图抽象（见 `factory-everything-项目总表.md` 的
「跨栈同构族：图引擎」），但**这张图记录的东西又不一样**：

| 项目 | 图记录什么 | 边的含义 |
| --- | --- | --- |
| DatasetFactory | 数据流过哪些 stage | 上游产物 → 下游产物 |
| SyntheticFactory | 进化链（基因组如何被迭代） | 上一轮 evolve → 下一轮 generate |
| DataMixtureFactory | 搜索谱系（配方如何被提出） | 父试验 → 子试验 |
| TokenizerFactory | 设计空间的展开与收敛 | 语料 → 规则 → 训练 → 评测 → 入选 |
| EmbeddingFactory | 两个空间的展开与收敛 | 文档 → 分块 → 嵌入 → 建索引 → 评测 → 入选 |
| **RepresentationFactory** | **一次观测是怎么做出来的** | **模型 → 抓取 → 三类探针 → 表征图 → 跨模型对比** |

本项目**没有设计空间**，也没有前沿 —— 所以这张图里没有 ``sel:`` 节点。
它记录的是**观测的构造过程**：

    corpus
      └── model:constructive
            └── capture:constructive        ← 一次抓取，被该模型下全部 spec 共用
                  ├── probe:similarity:constructive-tok-cos-bs-raw
                  ├── probe:cluster:constructive-tok-cos-bs-raw
                  └── probe:geometry:constructive-tok-cos-bs-raw
                        └── graph:constructive-tok-cos-bs-raw   ← 收拢点
      └── model:random
            └── …

**``capture`` 节点的出度就是「一次抓取被多少个观测配置复用」** ——
本项目 2 个模型各扇出 3 × 24 = 72 条探针分支，而 transformer 前向只跑了 2 遍。
这正是「同一份中间结果必须缓存并共享给所有兄弟分支」这条纪律的图形化表达：
如果实现层忘了缓存，图会长得一模一样（节点 id 相同），但运行时间会翻 24 倍 ——
所以这条纪律不能只靠看图，还得靠一条对象同一性断言（见 ``test_capture_shared``）。

节点粒度
--------
一个 spec 拆成四个节点：

    capture:{model}          ← 输入是该模型的 model 节点
    probe:{探针}:{spec_id}    ← 输入是 capture，三个探针平行
    graph:{spec_id}          ← **输入恰好是三类探针**（收拢点）
    compare:main             ← 输入是全部 graph 节点

不变量
------
**收拢点的输入必须恰好是三类探针** —— 一个不多、一个不少：

    dag.parents("graph:{spec_id}") == sorted(
        f"probe:{kind}:{spec_id}" for kind in PROBE_KINDS
    )

这条不变量是本项目**结构上最要紧的一条**：它保证「表征图确实由三类探针合成」，
而不是只用了其中一类（少一类图会静默缺一块），也不是偷偷多接了一个来源。
README 第六节明确要求三类探针**缺一不可**，这条断言就是把那句话变成可执行的检查。
"""

from __future__ import annotations

import heapq
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schema import PROBE_KINDS, utcnow

#: 语料 / 真值根节点 —— 所有链路的起点
CORPUS_NODE_ID = "corpus"

#: 跨模型对比节点
COMPARE_NODE_ID = "compare:main"

#: 可视化节点
VIZ_NODE_ID = "viz:main"

#: 版本节点
VERSION_NODE_ID = "version"


def model_node_id(model: str) -> str:
    """被观测模型节点。"""
    return f"model:{model}"


def capture_node_id(model: str) -> str:
    """抓取节点。

    **同一个模型只建一个节点** —— 它被该模型下全部 ``观测配置`` 共用。
    「一个 capture 有 72 个子节点」这件事在图上一眼可见，
    而它正是本项目「抓取一次、观测多次」这条设计纪律的证据。
    """
    return f"capture:{model}"


def probe_node_id(probe: str, spec_id: str) -> str:
    return f"probe:{probe}:{spec_id}"


def graph_node_id(spec_id: str) -> str:
    return f"graph:{spec_id}"


def project_node_id(model: str) -> str:
    """降维节点。

    降维发生在**抓取之后、探针之前**，且与「用哪个相似度度量」无关，
    所以它按 ``model`` 建节点、被该模型下的探针共用 —— 与 ``capture`` 同理。
    本项目里降维只影响展示与相似度探针，不影响线性探针的判定结果，
    这一点由 ``project: none`` 那一行作为对照来证明。
    """
    return f"project:{model}"


@dataclass
class LineageNode:
    """DAG 节点。

    kind 取值：``corpus`` | ``model`` | ``capture`` | ``project`` | ``probe`` |
    ``graph`` | ``compare`` | ``viz`` | ``version``
    """

    node_id: str
    kind: str
    label: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # ``outputs`` 排序后导出。它的追加顺序取决于「子节点是何时被加进来的」，
        # 而 ``from_dict`` 重建时子节点是按拓扑序加进来的 —— 不排序的话，
        # 同一个图的两条构建路径会导出两份不同的 lineage.json，
        # 于是「产物是否可复现」这个检查永远过不了。
        payload["outputs"] = sorted(self.outputs)
        return payload


class LineageDAG:
    """不可变追加式血缘图。"""

    def __init__(self) -> None:
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def add_node(self, node: LineageNode) -> str:
        if node.node_id in self.nodes:
            raise ValueError(f"node already exists: {node.node_id}")
        for parent in node.inputs:
            if parent not in self.nodes:
                raise ValueError(
                    f"unknown parent '{parent}' for node '{node.node_id}'"
                )
        self.nodes[node.node_id] = node
        for parent in node.inputs:
            self.nodes[parent].outputs.append(node.node_id)
            self.edges.append((parent, node.node_id))
        return node.node_id

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def parents(self, node_id: str) -> list[str]:
        return list(self.nodes[node_id].inputs)

    def children(self, node_id: str) -> list[str]:
        return list(self.nodes[node_id].outputs)

    def ancestors(self, node_id: str) -> list[str]:
        seen: set[str] = set()
        stack = list(self.parents(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.parents(cur))
        return sorted(seen)

    def descendants(self, node_id: str) -> list[str]:
        seen: set[str] = set()
        stack = list(self.children(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.children(cur))
        return sorted(seen)

    def path_to_root(self, node_id: str) -> list[str]:
        """沿父节点走到根的一条路径（多父时取名字最小的，保证确定）。"""
        path = [node_id]
        seen = {node_id}
        cur = node_id
        while True:
            nxt = next(
                (p for p in sorted(self.parents(cur)) if p not in seen), None
            )
            if nxt is None:
                return path
            path.append(nxt)
            seen.add(nxt)
            cur = nxt

    def topological(self) -> list[str]:
        """拓扑序；存在环则抛错。

        用最小堆而不是 FIFO 队列：同一批「已就绪」的节点里总是先取名字最小的，
        得到**字典序最小的拓扑序** —— 既确定，又符合阅读直觉。
        """
        indeg = {n: 0 for n in self.nodes}
        for _, dst in self.edges:
            indeg[dst] += 1
        queue = [n for n, d in indeg.items() if d == 0]
        heapq.heapify(queue)
        order: list[str] = []
        while queue:
            cur = heapq.heappop(queue)
            order.append(cur)
            for child in sorted(self.children(cur)):
                indeg[child] -= 1
                if indeg[child] == 0:
                    heapq.heappush(queue, child)
        if len(order) != len(self.nodes):
            raise ValueError("cycle detected in lineage DAG")
        return order

    def by_kind(self, kind: str) -> list[str]:
        return sorted(n for n, node in self.nodes.items() if node.kind == kind)

    def merge_points(self) -> list[str]:
        """多输入节点 —— 本项目里是每个 spec 的 ``graph`` 与最终的 ``compare``。

        注意**不能**用「多输入」当成「收拢点」的定义去写断言：
        ``compare:main`` 有 48 个输入，但它的语义是「横向对比」，
        不是「把一次观测的三类探针合成一个图」。要断言具体节点，就写具体节点。
        """
        return sorted(n for n in self.topological() if len(self.nodes[n].inputs) > 1)

    def leaves(self) -> list[str]:
        return sorted(n for n, node in self.nodes.items() if not node.outputs)

    def fan_out_ratio(self) -> float:
        """平均每个非叶节点的出度 —— 「扇出有多宽」的一个可读数字。"""
        internal = [n for n, node in self.nodes.items() if node.outputs]
        if not internal:
            return 0.0
        return len(self.edges) / len(internal)

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for node in self.nodes.values():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "by_kind": kinds,
            "merge_points": len(self.merge_points()),
            "fan_out_ratio": round(self.fan_out_ratio(), 3),
        }

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[n].to_dict() for n in self.topological()],
            # 边**排序后**导出。``self.edges`` 的插入顺序取决于「节点是何时被加进来的」，
            # 而 ``from_dict`` 重建时节点是按拓扑序加进来的 —— 不排序的话，
            # 同一个图的两条构建路径会导出两份不同的 lineage.json。
            "edges": [{"from": a, "to": b} for a, b in sorted(self.edges)],
            "summary": self.summary(),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LineageDAG":
        """从 ``to_dict()`` 的产物重建。

        存在的理由很实际：``dag`` 是只读子命令，它不能靠「同一次 run 的内存对象」，
        但 ``--dot`` / ``--selected-only`` / ``--limit`` 又都需要图对象。
        于是就地重建一份 —— 不重建的话 ``--dot`` 在 CLI 里会静默退化成
        「打印一段摘要加一句提示」，用户以为拿到了 DOT 却拿到别的东西。

        两个细节：
        - ``nodes`` 按拓扑序排列，所以顺序迭代时父节点一定先于子节点；
        - 节点的 ``outputs`` 一律**清空**后交给 ``add_node`` 重新推出来。
          直接照抄会让父节点的 ``outputs`` 被追加两次。
        """
        dag = cls()
        for raw in payload.get("nodes", []):
            dag.add_node(
                LineageNode(
                    node_id=str(raw["node_id"]),
                    kind=str(raw["kind"]),
                    label=str(raw["label"]),
                    inputs=[str(p) for p in raw.get("inputs", [])],
                    params=dict(raw.get("params", {})),
                    created_at=str(raw.get("created_at", utcnow())),
                    stats=dict(raw.get("stats", {})),
                )
            )
        return dag

    def to_dot(self, only_selected_ancestry: bool = False) -> str:
        """Graphviz DOT。

        ``only_selected_ancestry=True`` 时只画入选 spec 的祖先闭包 ——
        本项目有 48 个 spec × 3 类探针 = 144 个探针节点，全画出来是一张
        无法阅读的毛球。真正有信息量的是「一次入选的观测是怎么被构造出来的」。
        """
        if only_selected_ancestry:
            keep: set[str] = set()
            for nid in self.by_kind("compare") + self.by_kind("viz"):
                keep.add(nid)
                keep |= set(self.ancestors(nid))
            # 没有 compare/viz 时退化成「只画根与模型层」，避免导出空图
            if not keep:
                keep = set(self.nodes)
        else:
            keep = set(self.nodes)

        colors = {
            "corpus": "lightgrey",
            "model": "lightcyan",
            "capture": "lightyellow",
            "project": "lightyellow3",
            "probe": "white",
            "graph": "lightgreen",
            "compare": "lightpink",
            "viz": "lavender",
            "version": "lightblue",
        }
        lines = [
            "digraph lineage {",
            "  rankdir=LR;",
            '  node [shape=box, fontname="Helvetica"];',
        ]
        # 按**拓扑序**遍历节点，不按插入顺序：插入顺序取决于「节点是何时被
        # 加进来的」，重建（``from_dict``）时会不一样，于是同一张图会导出
        # 两份不同的 DOT。拓扑序是图自身的性质，与构建路径无关。
        for node in (self.nodes[n] for n in self.topological()):
            if node.node_id not in keep:
                continue
            color = colors.get(node.kind, "white")
            lines.append(
                f'  "{node.node_id}" [label="{node.label}\\n({node.kind})", '
                f"style=filled, fillcolor={color}];"
            )
        for a, b in sorted(self.edges):
            if a in keep and b in keep:
                lines.append(f'  "{a}" -> "{b}";')
        lines.append("}")
        return "\n".join(lines)

    def to_ascii(self, limit: int | None = None) -> str:
        """按拓扑序渲染。收拢点（多输入节点）单独列出且不受 ``limit`` 影响。"""
        order = self.topological()
        if not order:
            return "(empty DAG)"

        shown = order
        elided = 0
        if limit is not None and len(order) > limit:
            head = limit // 2
            tail = limit - head
            shown = order[:head] + order[len(order) - tail :]
            elided = len(order) - len(shown)

        id_w = max(len(n) for n in order)
        label_w = min(48, max(len(self.nodes[n].label) for n in order))
        kind_w = max(len(self.nodes[n].kind) for n in order)

        lines: list[str] = ["Lineage DAG（拓扑序）", ""]
        for idx, nid in enumerate(shown):
            if elided and idx == limit // 2:
                lines.append(f"  … 省略 {elided} 个节点（用 --dot 看全图）…")
            node = self.nodes[nid]
            stats = "  ".join(f"{k}={v}" for k, v in (node.stats or {}).items())
            lines.append(
                f"  {nid:<{id_w}}  {node.label:<{label_w}}  "
                f"[{node.kind:<{kind_w}}]  {stats}"
            )

        merges = self.merge_points()
        if merges:
            lines.append("")
            lines.append(f"收拢点（多输入节点，{len(merges)} 个）：")
            for nid in merges[:12]:
                node = self.nodes[nid]
                lines.append(f"  {nid}  ← {', '.join(node.inputs)}")
            if len(merges) > 12:
                lines.append(f"  … 另有 {len(merges) - 12} 个")

        return "\n".join(lines)


def probe_children_of_capture(model: str) -> list[str]:
    """给定模型，返回它的 capture 节点理论上应有的子节点种类（供测试核对）。"""
    return list(PROBE_KINDS)
