"""表征图（Representation Graph）。

README 第八节 M5「构造图」的产物。它与 DAG 里那个 ``graph:{spec_id}`` 节点
是**两件事**，但二者必须对齐 —— 本文件负责前者（图本身），
后者（收拢点在血缘图里的位置）由 ``pipeline.py`` 按固定的「输入恰好是三类探针」
接好，并在测试里用 ``dag.parents("graph:{spec_id}")`` 钉死。

表征图回答的是「结构与表征如何耦合」：
节点 = ``layer × head``（本项目 4 层 × 4 头 = 16 个节点）；
边   = 两个头的**画像相似度** ≥ 阈值（画像来自 Geometry Probe）；
聚类 = 连通分量（与 Geometry Probe 用的是同一张相似度矩阵，不另算）。

三个刻意的选择：

1. **节点是「头」不是「层」。**
   层只有 4 个，办不成一个有意义的图；头有 16 个，而且「哪些头在做同一件事」
   这个问题只能靠头之间的距离来答 —— 看注意力矩阵是答不出来的
   （见 probes.py 的 Geometry Probe 一节）。

2. **边用的是「画像相似度」而不是「注意力模式直接相似度」。**
   直接比两个头的注意力矩阵会得到一堆近邻（因为它们都 attend 到自身），
   这种相似度与「在做什么」无关。画像把每个头压成 5 维可读向量
   （熵 / 自权重 / 局部性 / sink 权重 / 平均偏移），比的是「在做的事」
   而不是「都 attend 自己」这个 trivial 共性。

3. **聚类是连通分量，不是 k-means。**
   k 要事先定，而「有几个功能组」正是要观测的结论，不能先假设。
   连通分量把阈值直接变成「耦合强度」的可读旋钮。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import ProbeResult


@dataclass
class GraphNode:
    """图里的一个节点 = 一个注意力头。"""

    node_id: str
    layer: int
    head: int
    #: 画像（来自 HeadStats.to_dict），用于展示与可解释性
    profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "layer": self.layer,
            "head": self.head,
            "profile": dict(self.profile),
        }


@dataclass
class GraphEdge:
    """图里的一条边 = 两个头的画像相似度 ≥ 阈值。"""

    a: str
    b: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "weight": round(self.weight, 6)}


@dataclass
class RepresentationGraph:
    """一次观测的表征图。

    ``spec_id`` 让它和血缘图里的 ``graph:{spec_id}`` 节点一一对应 ——
    可视化节点 ``viz:main`` 收拢的正是所有 spec 的表征图。
    """

    spec_id: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    clusters: list[list[str]] = field(default_factory=list)
    threshold: float = 0.0

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "threshold": round(self.threshold, 6),
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_clusters": self.n_clusters,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "clusters": [sorted(c) for c in self.clusters],
        }


def build_representation_graph(geometry: ProbeResult) -> RepresentationGraph:
    """从 Geometry Probe 的结果构造表征图。

    Geometry Probe 已经算好了每个头的画像、头间相似度矩阵、以及按阈值的
    连通分量聚类 —— 这里只把它们**重新打包成一个独立、可序列化的图对象**，
    不重复任何计算（重复计算会让「同一张图两处算出不同结构」变成一个长期
    难查的幽灵，与 ``gram`` 只实现一份是同一个纪律）。
    """
    detail = geometry.detail or {}
    heads = detail.get("heads", [])
    sim = detail.get("similarity", [])
    clusters = detail.get("clusters", [])
    threshold = float(detail.get("threshold", geometry.metrics.get("threshold", 0.0)))

    nodes = []
    for h in heads:
        nodes.append(
            GraphNode(
                node_id=str(h.get("node_id", f"h{h['layer']}.{h['head']}")),
                layer=int(h["layer"]),
                head=int(h["head"]),
                profile={
                    k: h[k]
                    for k in ("entropy", "self_weight", "locality", "sink_weight", "mean_offset")
                    if k in h
                },
            )
        )

    # 边：相似度矩阵上三角、权重 ≥ 阈值、且不是自己。
    n = len(nodes)
    edges: list[GraphEdge] = []
    for i in range(n):
        for j in range(i + 1, n):
            w = sim[i][j] if i < len(sim) and j < len(sim[i]) else 0.0
            if w >= threshold:
                edges.append(GraphEdge(a=nodes[i].node_id, b=nodes[j].node_id, weight=w))

    return RepresentationGraph(
        spec_id=geometry.spec_id,
        nodes=nodes,
        edges=edges,
        clusters=[list(c) for c in clusters],
        threshold=threshold,
    )


def graph_digest(graph: RepresentationGraph) -> str:
    """内容寻址：图的「结构指纹」。

    只取节点数与边集（节点对 + 权重），**不算** ``created_at`` 之类的墙上时间。
    两个 spec 的图若结构相同（同样的头、同样的耦合关系），指纹必相同 ——
    这正是「随机对照模型与构造式标本的图是否不同」这条结论可核对的前提。
    """
    from .schema import stable_id

    edge_str = ",".join(
        sorted(f"{e.a}|{e.b}|{round(e.weight, 6)}" for e in graph.edges)
    )
    cluster_str = ";".join(
        "/".join(c) for c in (sorted(tuple(c) for c in graph.clusters))
    )
    return stable_id(
        graph.spec_id, graph.n_nodes, edge_str, cluster_str, n=16
    )
