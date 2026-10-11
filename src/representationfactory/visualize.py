"""零依赖可视化（HTML + 内联 SVG）。

`cli visualize` 调用 :func:`render_html` 把 ``LabResult`` 渲染成一份
**单文件 HTML**（无外部依赖、无 JS 框架、可离线打开），包含四类图：

1. 逐层演化曲线（SVG 折线图，构造式标本 vs 随机对照）
2. 表征图（SVG 节点-连线，节点 = layer×head，边 = 注意力模式相似）
3. 注意力热图（HTML 表格，按 cell 着色）
4. PCA 散点（SVG，按主题着色）

全部 SVG 手画，**不引入 matplotlib / plotly**：本项目铁律是零第三方依赖
（见 skill 铁律章）。图里的数字都直接来自 ``LabResult`` 的字段，
与报告 / 矩阵严格一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .compare import cross_model, differentiation
from .graph import build_representation_graph
from .schema import CaptureRun, LabResult

# 主题调色板（浅色底，深色字，按 IDE 主题约定）
_TOPIC_COLORS = [
    "#c0392b", "#2980b9", "#27ae60", "#8e44ad",
    "#d35400", "#16a085", "#2c3e50", "#c0399b",
]
_COLOR_BY_TOPIC: dict[str, str] = {}


def _color_for_topic(topic: str, topics: list[str]) -> str:
    if topic not in _COLOR_BY_TOPIC:
        idx = topics.index(topic) if topic in topics else len(_COLOR_BY_TOPIC)
        _COLOR_BY_TOPIC[topic] = _TOPIC_COLORS[idx % len(_TOPIC_COLORS)]
    return _COLOR_BY_TOPIC[topic]


def _svg_line_chart(
    series: list[dict[str, Any]],
    width: int = 640,
    height: int = 320,
    ylabel: str = "线性探针准确率",
) -> str:
    """series: [{"label", "layers":[int], "acc":[float], "color"}]"""
    if not series:
        return "<p>（无曲线）</p>"
    pad = 48
    x0, y0 = pad, height - pad
    x1, y1 = width - pad, pad
    # x 轴 = 层 0..max_layer；y 轴 = 0..1
    max_layer = max(max(s["layers"]) for s in series)
    min_layer = min(min(s["layers"]) for s in series)

    def px(layer: int) -> float:
        if max_layer == min_layer:
            return (x0 + x1) / 2
        return x0 + (layer - min_layer) / (max_layer - min_layer) * (x1 - x0)

    def py(acc: float) -> float:
        a = max(0.0, min(1.0, acc))
        return y0 + (0.0 - a) / (0.0 - 1.0) * (y1 - y0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="layer curves">'
    ]
    # 轴
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#888"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#888"/>')
    # y 网格
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = py(g)
        parts.append(
            f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#eee"/>'
        )
        parts.append(
            f'<text x="{x0 - 6}" y="{yy + 3:.1f}" font-size="10" fill="#666" '
            f'text-anchor="end">{g:.2f}</text>'
        )
    # x 刻度
    for layer in range(min_layer, max_layer + 1):
        xx = px(layer)
        parts.append(
            f'<text x="{xx:.1f}" y="{y0 + 16}" font-size="10" fill="#666" '
            f'text-anchor="middle">L{layer}</text>'
        )
    # 线
    for s in series:
        pts = " ".join(f"{px(l):.1f},{py(a):.1f}" for l, a in zip(s["layers"], s["acc"]))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" '
            f'stroke-width="2"/>'
        )
        for l, a in zip(s["layers"], s["acc"]):
            parts.append(
                f'<circle cx="{px(l):.1f}" cy="{py(a):.1f}" r="3" '
                f'fill="{s["color"]}"/>'
            )
    parts.append(
        f'<text x="{x0}" y="{y1 - 8}" font-size="11" fill="#333">{ylabel}</text>'
    )
    # 图例
    lx = x1 - 160
    for i, s in enumerate(series):
        ly = y1 + 4 + i * 14
        parts.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{s["color"]}"/>')
        parts.append(
            f'<text x="{lx + 14}" y="{ly + 9}" font-size="10" fill="#333">{s["label"]}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_graph(g: Any, width: int = 520, height: int = 420) -> str:
    """SVG 节点-连线。节点按 (layer, head) 网格排布。"""
    if g.n_nodes == 0:
        return "<p>（无图）</p>"
    coords: dict[str, tuple[int, int]] = {}
    layers_seen: set[int] = set()
    for n in g.nodes:
        layers_seen.add(n.layer)
        coords[n.node_id] = (n.layer, n.head)
    max_layer = max(layers_seen)
    n_heads = max(h for (_, h) in coords.values()) + 1

    pad = 40
    x0, y0 = pad, pad
    x1, y1 = width - pad, height - pad

    def px(layer_i: int) -> float:
        return x0 + (layer_i / max(max_layer, 1)) * (x1 - x0)

    def py(head_i: int) -> float:
        return y0 + (head_i / max(n_heads - 1, 1)) * (y1 - y0)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    # 边
    for e in g.edges:
        a, b = e.a, e.b
        if a in coords and b in coords:
            la, ha = coords[a]
            lb, hb = coords[b]
            parts.append(
                f'<line x1="{px(la):.1f}" y1="{py(ha):.1f}" x2="{px(lb):.1f}" '
                f'y2="{py(hb):.1f}" stroke="#bbb" stroke-width="1"/>'
            )
    # 节点
    for nid, (la, ha) in coords.items():
        parts.append(
            f'<circle cx="{px(la):.1f}" cy="{py(ha):.1f}" r="9" fill="#3498db" '
            f'stroke="#21618c"/>'
        )
        parts.append(
            f'<text x="{px(la):.1f}" y="{py(ha) + 3:.1f}" font-size="8" '
            f'fill="#fff" text-anchor="middle">{ha}</text>'
        )
    # 层标签
    for layer_i in sorted(layers_seen):
        parts.append(
            f'<text x="{px(layer_i):.1f}" y="{y1 + 16}" font-size="10" fill="#666" '
            f'text-anchor="middle">L{layer_i}</text>'
        )
    parts.append("</svg>")
    # 聚类说明
    if g.clusters:
        parts.append("<p>功能组（连通分量）：</p><ul>")
        for i, comp in enumerate(g.clusters):
            parts.append(f"<li>组 {i + 1}：{', '.join(sorted(comp))}</li>")
        parts.append("</ul>")
    return "\n".join(parts)


def _html_attention(attn: dict[str, Any]) -> str:
    if not attn or not attn.get("samples"):
        return "<p>（无注意力样本）</p>"
    parts = []
    for samp in attn["samples"][:3]:
        n = samp["n"]
        parts.append(f"<h4>句 {samp['sentence_id']}（主题 {samp['topic']}）</h4>")
        parts.append(f"<p>{samp['text']}</p>")
        # 只画第 0 个头、每层
        for l, per_head in enumerate(samp["attention"]):
            head0 = per_head[0] if per_head else []
            parts.append(f"<p>层 L{l} / 头 0：</p>")
            parts.append('<table class="heat">')
            for i, row in enumerate(head0):
                cells = []
                for j, v in enumerate(row):
                    # 着色：v 越大越红
                    red = int(min(1.0, v) * 200)
                    cells.append(
                        f'<td style="background:rgb({red+55},{255-red-55},{255-red-55})">'
                        f"{v:.2f}</td>"
                    )
                marker = " ★" if j == samp.get("marker_index") else ""
                parts.append(f"<tr><th>{i}{marker}</th>{''.join(cells)}</tr>")
            parts.append("</table>")
    return "\n".join(parts)


def _svg_scatter(scatter: dict[str, Any], width: int = 560, height: int = 400) -> str:
    if not scatter or not scatter.get("points"):
        return "<p>（无散点）</p>"
    pts = scatter["points"]
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    if not xs or not ys:
        return "<p>（无散点）</p>"
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 40
    x0, y0, x1, y1 = pad, height - pad, width - pad, pad

    def px(x: float) -> float:
        if xmax == xmin:
            return (x0 + x1) / 2
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def py(y: float) -> float:
        if ymax == ymin:
            return (y0 + y1) / 2
        return y0 - (y - ymin) / (ymax - ymin) * (y1 - y0)

    topics = sorted({p["topic"] for p in pts})
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    for p in pts:
        c = _color_for_topic(p["topic"], topics)
        parts.append(
            f'<circle cx="{px(p["x"]):.1f}" cy="{py(p["y"]):.1f}" r="4" '
            f'fill="{c}" fill-opacity="0.7"/>'
        )
    parts.append(
        f'<text x="{x0}" y="{y0 + 18}" font-size="10" fill="#666">'
        f'PCA 散点（层 L{scatter.get("layer", "?")}，'
        f'{scatter.get("n_points", len(pts))} 点，按主题着色）</text>'
    )
    # 图例
    ly = y0 - 10
    lx = x1 - 200
    for i, t in enumerate(topics):
        yy = ly - i * 14
        parts.append(f'<rect x="{lx}" y="{yy}" width="10" height="10" fill="{_color_for_topic(t, topics)}"/>')
        parts.append(f'<text x="{lx + 14}" y="{yy + 9}" font-size="10" fill="#333">{t}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_html(result: LabResult, out_html: str | Path) -> Path:
    """渲染单文件 HTML。返回写出路径。"""
    ref = result.reference_run()
    cm = cross_model(result)
    scatter = result.counters.get("scatter") or {}
    topics = sorted({p["topic"] for p in scatter.get("points", [])})

    # 1. 曲线：参考 run 的逐属性
    curve_svg = ""
    if ref is not None:
        series = []
        for o in differentiation(ref):
            c = ref.curve(o.property, o.subset)
            if c is None:
                continue
            series.append(
                {
                    "label": o.spec_key,
                    "layers": list(c.layers),
                    "acc": list(c.accuracy),
                    "color": "#2980b9" if o.property != "topic" else "#c0392b",
                }
            )
        curve_svg = _svg_line_chart(series)

    # 2. 图
    graph_svg = ""
    if ref is not None and ref.probes.get("geometry") is not None:
        g = build_representation_graph(ref.probes["geometry"])
        graph_svg = _svg_graph(g)

    # 3. 注意力
    attn = result.counters.get("attention_sample") or getattr(result, "_attn", None)
    # write_outputs 把注意力写进 attention.json；可视化时若 result 带 _captures 可取
    attn_html = ""
    captures = getattr(result, "_captures", {}) or {}
    if "constructive" in captures and ref is not None:
        from .pipeline import _attention_samples

        attn = _attention_samples(captures["constructive"], ref.spec_id)
    if attn:
        attn_html = _html_attention(attn)

    # 4. 散点
    scatter = result.counters.get("scatter") or {}
    scatter_svg = _svg_scatter(scatter)

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>RepresentationFactory 可视化</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
       color:#222; background:#fff; margin:24px; line-height:1.5; }}
h1 {{ font-size:22px; }} h2 {{ font-size:18px; border-bottom:1px solid #eee;
       padding-bottom:4px; }} h4 {{ margin:16px 0 4px; }}
table.heat {{ border-collapse:collapse; font-size:10px; }}
table.heat td, table.heat th {{ border:1px solid #ddd; padding:2px 4px;
       text-align:center; min-width:30px; }}
.legend {{ color:#666; font-size:12px; }}
</style></head><body>
<h1>RepresentationFactory 可视化</h1>
<p class="legend">版本：{result.version.version_id if result.version else '?'} ·
模型：{', '.join(sorted(cm.keys()))}</p>

<h2>1. 逐层演化曲线（参考配置）</h2>
{curve_svg}

<h2>2. 表征图（layer×head，边=注意力模式相似）</h2>
{graph_svg}

<h2>3. 注意力热图（参考样本）</h2>
{attn_html}

<h2>4. PCA 散点（按主题着色）</h2>
{scatter_svg}

<p class="legend">零依赖 SVG 渲染。数字与 report.md / matrix.csv 一致。</p>
</body></html>
"""
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
