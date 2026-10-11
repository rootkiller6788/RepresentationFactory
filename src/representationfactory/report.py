"""报告渲染（Report / 验收三问）。

把 ``LabResult`` 渲染成一份**可被人读、可被 git diff、可被测试核对**的
Markdown 报告。九节结构对应 README 第八节 M0–M6。

本模块刻意**不碰任何随机量**：它读的是 ``LabResult`` 里已经定稿的
``corpus`` / ``matrix`` / ``ablations`` / ``findings`` / ``version`` /
``dag``。凡是「数字」都来自这些字段，而不是重新算一遍 —— 否则报告与
产物会慢慢漂移（同工作区其它项目踩过的坑，见 skill 坑 30）。

报告里「验收三问」是直接可答的：给定 ``--position layer:token``，
``where_from`` / ``what_happened`` / ``why_here`` 三栏能给出具体答案
（见 README 第七节 + ``cli.explain``）。
"""

from __future__ import annotations

from typing import Any

from .compare import cross_model, differentiation
from .schema import LabResult, PROBE_KINDS

# ----------------------------------------------------------------------
# 小工具：把一条逐层曲线渲染成 ASCII 条
# ----------------------------------------------------------------------
_BAR = "█"


def _ascii_curve(layers: list[int], acc: list[float], width: int = 32) -> str:
    if not acc:
        return ""
    lo, hi = min(acc), max(acc)
    span = (hi - lo) or 1.0
    out: list[str] = []
    for lyr, a in zip(layers, acc):
        filled = int(round((a - lo) / span * width))
        out.append(f"L{lyr:>2} |{_BAR * filled:<{width}} {a:.3f}")
    return "\n".join(out)


def _fmt_list(items: list[Any], n: int = 8) -> str:
    if len(items) <= n:
        return ", ".join(str(x) for x in items)
    return ", ".join(str(x) for x in items[:n]) + f" … (+{len(items) - n})"


# ----------------------------------------------------------------------
# 各节
# ----------------------------------------------------------------------
def _section_corpus(result: LabResult) -> str:
    c = result.corpus
    lines = [
        f"## 1. 语料与已知真值（被观测的是什么）",
        "",
        f"- 名称：`{c.name}`",
        f"- 句子 {c.n_sentences} 条，位置 {c.n_tokens} 个，平均句长 {c.mean_sentence_len:.2f}",
        f"- 词形 {c.n_forms}、角色 {c.n_roles}、主题 {c.n_topics}",
        f"- 歧义位置 {c.n_ambiguous_tokens}（{c.n_ambiguous_tokens / max(1, c.n_tokens):.1%}）"
        f"—— 它们的角色要等注意力扩到能看见主题标记词才定",
        f"- 主题标记位置 {c.n_marker_tokens}（作为 attention sink）",
    ]
    if c.role_sizes:
        lines.append("")
        lines.append("角色分布：" + "，".join(f"{k}={v}" for k, v in c.role_sizes.items()))
    if c.topic_sizes:
        lines.append("主题分布：" + "，".join(f"{k}={v}" for k, v in c.topic_sizes.items()))
    return "\n".join(lines)


def _section_findings(result: LabResult) -> str:
    f = result.findings
    lines = ["## 2. 验收结论（标本是否对准了真值）", ""]
    if not f:
        lines.append("_（无结论）_")
        return "\n".join(lines)

    rec = f.get("truthed_recovered", {})
    if rec:
        lines.append("**真值复原情况**（设计期望 vs 探针实测的分化层）：")
        lines.append("")
        lines.append("| 属性 | 设计期望层 | 实测分化层 | 命中 |")
        lines.append("|---|---|---|---|")
        ok = True
        for k, v in rec.items():
            got = v.get("actual_onset", -1)
            exp = v.get("expected_onset", -1)
            hit = "✅" if v.get("match") else "❌"
            if not v.get("match"):
                ok = False
            lines.append(f"| {k} | L{exp} | L{got} | {hit} |")
        lines.append("")
        lines.append(f"全部命中：{'是' if f.get('all_onsets_recovered') else '否'}")
    if "random_control_has_no_step" in f:
        lines.append(
            f"随机权重对照是否「没有阶梯」：{'是' if f['random_control_has_no_step'] else '否'}"
            f"（证明阶梯不是随便一个网络都有的）"
        )
    if "control_axis_inflates" in f:
        ca = f["control_axis_inflates"]
        lines.append(
            f"方法学对照（按 token 划分）是否系统性虚高："
            f"{'是' if ca.get('inflated') else '否'}"
            f"（峰值 {ca.get('max_control', 0):.3f} vs 正确口径 {ca.get('max_correct', 0):.3f}）"
        )
    if "conclusion" in f:
        lines.append("")
        lines.append(f"> **结论**：{f['conclusion']}")
    return "\n".join(lines)


def _section_curves(result: LabResult) -> str:
    ref = result.reference_run()
    lines = ["## 3. 参考配置的逐层演化曲线（核心产物）", ""]
    lines.append(
        f"参考配置：`{result.selected or (ref.spec_id if ref else '?')}`"
        f"（构造式标本 / token 池化 / centered-cosine / 按句划分 / 原始）"
    )
    lines.append("")
    if ref is None:
        lines.append("_（无参考 run）_")
        return "\n".join(lines)
    onsets = differentiation(ref)
    lines.append("| 属性 | 分化层 | 跳变 | 最佳层 | 最佳准确率 | 多数类基线 |")
    lines.append("|---|---|---|---|---|---|")
    for o in onsets:
        lines.append(
            f"| {o.spec_key} | L{o.onset_layer} | {o.onset_jump:.3f} | "
            f"L{o.best_layer} | {o.best_acc:.3f} | {o.majority:.3f} |"
        )
    lines.append("")
    lines.append("**ASCII 曲线**（纵轴为线性探针测试准确率）：")
    lines.append("")
    for o in onsets:
        c = ref.curve(o.property, o.subset) if hasattr(o, "subset") else ref.curve(o.property)
        if c is None:
            continue
        lines.append(f"_{o.spec_key}_（chance={c.chance:.3f}）：")
        lines.append("")
        lines.append("```")
        lines.append(_ascii_curve(c.layers, c.accuracy))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _section_cross_model(result: LabResult) -> str:
    cm = cross_model(result)
    lines = ["## 4. 跨模型对比（构造式标本 vs 随机对照）", ""]
    if not cm:
        lines.append("_（无）_")
        return "\n".join(lines)
    lines.append("| 模型 | 最大 headline 增益 | 分化层层号 |")
    lines.append("|---|---|---|")
    for m, v in cm.items():
        onsets = ",".join(str(x) for x in v.get("onset_layers", [])) or "-"
        lines.append(f"| {m} | {v.get('max_headline', 0):.3f} | {onsets} |")
    lines.append("")
    lines.append(
        "> 关键对照：**随机权重模型**的 headline 增益应接近 0（没有「阶梯」），"
        "而**构造式标本**在每个属性上都应看到清晰的层级分化。"
    )
    return "\n".join(lines)


def _section_graph(result: LabResult) -> str:
    ref = result.reference_run()
    lines = ["## 5. 表征图（结构与表征如何耦合）", ""]
    if ref is None:
        lines.append("_（无）_")
        return "\n".join(lines)
    geo = ref.probes.get("geometry")
    if geo is None:
        lines.append("_（无 geometry 探针）_")
        return "\n".join(lines)
    from .graph import build_representation_graph

    g = build_representation_graph(geo)
    lines.append(
        f"- 节点（layer×head）= {g.n_nodes}，边（注意力模式相似）= {g.n_edges}"
    )
    lines.append(f"- 连通分量（功能组）= {g.n_clusters}（阈值 {g.threshold:.3f}）")
    if g.clusters:
        for i, comp in enumerate(g.clusters):
            members = ", ".join(sorted(comp))
            lines.append(f"  - 组 {i + 1}：{members}")
    return "\n".join(lines)


def _section_matrix(result: LabResult) -> str:
    lines = ["## 6. 观测矩阵（每个观测配置一行）", ""]
    if not result.matrix:
        lines.append("_（空）_")
        return "\n".join(lines)
    cols = list(result.matrix[0].keys())
    # 只显示关键列，避免报告爆炸
    keep = [c for c in ("spec_id", "model", "pooling", "similarity", "split", "project",
                        "headline", "n_curves") if c in cols]
    lines.append("| " + " | ".join(keep) + " |")
    lines.append("|" + "---|" * len(keep))
    for row in result.matrix:
        vals = []
        for c in keep:
            v = row.get(c, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append(f"共 {len(result.matrix)} 行。完整数据见 `matrix.csv`。")
    return "\n".join(lines)


def _section_methodology(result: LabResult) -> str:
    lines = ["## 7. 方法学对照（一个「越差越正确」的轴）", ""]
    f = result.findings
    ca = f.get("control_axis_inflates", {})
    if ca.get("inflated"):
        lines.append(
            f"按 **token 随机划分**会让探针准确率系统性虚高："
            f"对照轴峰值 {ca.get('max_control', 0):.3f}，而正确的按句划分峰值"
            f" {ca.get('max_correct', 0):.3f}。"
        )
        lines.append("")
        lines.append(
            "> 原因：同一句内的 token 高度相关，按 token 划分会把「同句兄弟」"
            "混进测试集，探针等于在作弊。矩阵里 `split=by_token` 的行**数字更高"
            "不代表更好**，报告与 CLI 都给它打了标记。"
        )
    else:
        lines.append("未发现系统性虚高（对照轴与正确口径一致）。")
    return "\n".join(lines)


def _section_invariants(result: LabResult) -> str:
    lines = ["## 8. 可执行的不变量（测试护栏）", ""]
    lines.append("以下不变量由测试套件钉死，违反即 CI 红：")
    lines.append("")
    inv = [
        "1. `n_runs == 观测配置总数`（扇出不能漏分支）。",
        "2. 每个 spec 的 graph 节点父节点**恰好**是三类探针，一个不多一个不少。",
        "3. 线性探针**按句分组**划分；按 token 划分的准确率只用于展示陷阱，不进结论。",
        "4. PCA 主成分**符号确定**（同一输入两次投影逐位相同）。",
        "5. CKA 必须**中心化**，否则 LayerNorm 后向量挤在一个锥里、cos 全 0.99。",
        "6. 随机权重对照模型**不出现阶梯**（headline 增益 ≈ 0）。",
        "7. 导出物（curves/probes/graphs）载入后行为一致（round-trip 逐位相同）。",
        "8. 血缘图 `lineage.json` 的 `to_dict → from_dict → to_dict` 逐位相同。",
        "9. 只读子命令（curve/graph/compare/capture/list/dag/version）不改 `out/`。",
    ]
    lines.extend(inv)
    return "\n".join(lines)


def _section_products(result: LabResult) -> str:
    lines = ["## 9. 产物清单", ""]
    lines.append("`out/` 下：")
    lines.append("")
    for p in (
        "corpus_stats.json",
        "matrix.csv",
        "curves.json",
        "probes.json",
        "graphs.json",
        "findings.json",
        "lineage.json",
        "scatter.json",
        "attention.json",
        "version.json",
    ):
        lines.append(f"- `{p}`")
    if result.version is not None:
        lines.append("")
        lines.append(f"版本：`{result.version.version_id}`（digest `{result.version.digest[:12]}…`）")
    return "\n".join(lines)


def render_markdown(result: LabResult) -> str:
    """渲染完整报告。"""
    parts = [
        "# RepresentationFactory 观测报告",
        "",
        f"> 测量型流水线（Microscope）。本报告展示「构造式标本」内部表征"
        f"随层演化的证据。版本：`"
        f"{result.version.version_id if result.version else '?'}`。",
        "",
        _section_corpus(result),
        "",
        _section_findings(result),
        "",
        _section_curves(result),
        "",
        _section_cross_model(result),
        "",
        _section_graph(result),
        "",
        _section_matrix(result),
        "",
        _section_methodology(result),
        "",
        _section_invariants(result),
        "",
        _section_products(result),
        "",
    ]
    return "\n".join(parts)
