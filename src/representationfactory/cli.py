"""命令行入口。

子命令分两类：

- **写命令**：``run``（跑全流程并落盘）、``visualize``（生成
  ``out/visualize.html`` 单文件 HTML）。
- **只读命令**：``capture / project / similarity / probe / curve / graph /
  compare / explain / list / dag / version``。它们**从不写 ``out/``**——
  只在内存里复算并打印。``test_read_only_commands`` 用 md5sum 实测这一点
  （见 skill 坑 30）。

只读命令为什么能「复算却不写」：

- 若 ``out/`` 已存在，它们走 :func:`load_result` **直接从产物重建
  ``LabResult``**（秒级、不重跑 75s 流水线、不改任何文件）；
- 若 ``out/`` 缺失，才退回 ``run_pipeline`` 在内存里算；而 ``run_pipeline``
  本身不落盘（落盘是 ``write_outputs`` 的事）。

两种路径都不会往 ``out/`` 写一个字节 —— 磁盘上的产物一个字节都不变。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .corpus import load_corpus
from .model import specimen_data
from .pipeline import load_result, run_and_write, run_pipeline, write_outputs
from .schema import LabResult


# ----------------------------------------------------------------------
# 公共：取结果（内存，不写盘）
# ----------------------------------------------------------------------
def _cfg(args: argparse.Namespace) -> dict[str, Any]:
    path = getattr(args, "config", None) or "configs/default.json"
    return load_config(path)


def _result(args: argparse.Namespace) -> LabResult:
    """尽量从 ``out/`` 加载（快、且不重跑）; 缺产物时再内存里跑一遍。"""
    out = getattr(args, "out", None) or "out"
    res = load_result(out)
    if res is not None:
        return res
    return run_pipeline(_cfg(args))


# ----------------------------------------------------------------------
# 各子命令
# ----------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    out = Path(args.out)
    result = run_and_write(cfg, out)
    print(f"run 完成：{len(result.runs)} 个 run，{len(result.matrix)} 行矩阵")
    print(f"产物已写入 {out}")
    print(f"版本：{result.version.version_id if result.version else '?'}")
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    result = _result(args)
    print(f"模型：{', '.join(sorted({r.model for r in result.runs}))}")
    print(f"观测配置数：{len(result.runs)}")
    for r in result.runs[:1]:
        print(f"示例 spec：{r.spec_id}（headline={r.headline():.3f}）")
        print(f"  曲线数：{len(r.curves)}，头画像数：{len(r.heads)}")
    return 0


def _cmd_project(args: argparse.Namespace) -> int:
    result = _result(args)
    ref = result.reference_run()
    if ref is None:
        print("无参考 run", file=sys.stderr)
        return 1
    sc = result.counters.get("scatter") or {}
    if not sc:
        print("无散点数据（先 run 生成）", file=sys.stderr)
        return 1
    print(f"参考配置：{sc.get('spec_id')}")
    print(f"降维方法：{sc.get('method')}（层 L{sc.get('layer')}）")
    ev = sc.get("explained_variance", [])
    if ev:
        print(f"方差解释率：[{(', '.join(f'{v:.3f}' for v in ev[:4]))}]")
    print(f"点数：{sc.get('n_points')}（按主题着色）")
    return 0


def _cmd_similarity(args: argparse.Namespace) -> int:
    result = _result(args)
    spec_id = args.spec
    run = result.find(spec_id)
    if run is None:
        print(f"未知 spec：{spec_id}", file=sys.stderr)
        return 1
    pr = run.probes.get("similarity")
    if pr is None:
        print(f"{spec_id} 无 similarity 探针", file=sys.stderr)
        return 1
    matrix = pr.detail.get("matrix", [])
    layers = pr.detail.get("layers", [])
    print(f"spec：{spec_id}  层间相似度（{pr.detail.get('metric')}）：")
    for i, row in enumerate(matrix):
        print(f"  L{layers[i]}: " + " ".join(f"{v:.2f}" for v in row))
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    result = _result(args)
    run = result.find(args.spec)
    if run is None:
        print(f"未知 spec：{args.spec}", file=sys.stderr)
        return 1
    kind = args.kind
    pr = run.probes.get(kind)
    if pr is None:
        print(f"{args.spec} 无 {kind} 探针", file=sys.stderr)
        return 1
    print(f"spec：{args.spec}  探针：{kind}")
    print(f"指标：{json.dumps(pr.metrics, ensure_ascii=False)}")
    if kind == "cluster":
        curves = pr.detail.get("curves", {})
        for key, c in sorted(curves.items()):
            # detail 里的曲线是 LayerCurve.to_dict()，没有 best_acc 字段，自己算
            lyrs = c.get("layers", [])
            accs = c.get("accuracy", [])
            best_layer = c.get("best_layer")
            best_acc = 0.0
            if best_layer in lyrs and accs:
                best_acc = accs[lyrs.index(best_layer)]
            print(
                f"  {key}: onset=L{c['onset_layer']} best=L{best_layer} "
                f"acc={best_acc:.3f} majority={c['majority']:.3f}"
            )
    elif kind == "geometry":
        m = pr.metrics
        print(
            f"  头数(nodes)={int(m.get('n_heads', 0))} "
            f"聚类={int(m.get('n_clusters', 0))} "
            f"阈值={pr.detail.get('threshold')} "
            f"sink头={int(m.get('n_sink_heads', 0))} "
            f"局部头={int(m.get('n_local_heads', 0))}"
        )
    return 0


def _cmd_curve(args: argparse.Namespace) -> int:
    result = _result(args)
    run = result.find(args.spec)
    if run is None:
        print(f"未知 spec：{args.spec}", file=sys.stderr)
        return 1
    c = run.curve(args.property, args.subset)
    if c is None:
        print(f"{args.spec} 无曲线 {args.property}/{args.subset}", file=sys.stderr)
        return 1
    print(f"spec：{args.spec}  属性：{args.property}/{args.subset}")
    print(f"chance={c.chance:.3f} majority={c.majority:.3f} "
          f"onset=L{c.onset_layer} best=L{c.best_layer} deep_gain={c.deep_gain:.3f}")
    for lyr, a in zip(c.layers, c.accuracy):
        print(f"  L{lyr}: {a:.3f}")
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    from .graph import build_representation_graph

    result = _result(args)
    run = result.find(args.spec)
    if run is None:
        print(f"未知 spec：{args.spec}", file=sys.stderr)
        return 1
    geo = run.probes.get("geometry")
    if geo is None:
        print(f"{args.spec} 无 geometry 探针", file=sys.stderr)
        return 1
    g = build_representation_graph(geo)
    print(f"spec：{args.spec}")
    print(f"节点={g.n_nodes} 边={g.n_edges} 聚类={g.n_clusters} "
          f"阈值={g.threshold:.3f}")
    for i, comp in enumerate(g.clusters):
        print(f"  组 {i + 1}：{', '.join(sorted(comp))}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import cross_model, differentiation

    result = _result(args)
    ref = result.reference_run()
    print("参考配置：", result.selected or (ref.spec_id if ref else "?"))
    if ref is not None:
        print("逐层分化：")
        for o in differentiation(ref):
            print(f"  {o.spec_key}: onset=L{o.onset_layer} jump={o.onset_jump:.3f} "
                  f"best=L{o.best_layer} acc={o.best_acc:.3f}")
    print("\n跨模型：")
    for m, v in cross_model(result).items():
        onsets = ",".join(str(x) for x in v.get("onset_layers", [])) or "-"
        print(f"  {m}: headline={v.get('max_headline', 0):.3f} onsets=[{onsets}]")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    """验收三问：where_from / what_happened / why_here。

    曲线来自 ``_result``（通常从 ``out/`` 加载，不重跑流水线）；
    真值来自语料（解析很快），所以这条只读命令基本是即时的。
    """
    cfg = _cfg(args)
    corpus = load_corpus(cfg)
    data = specimen_data(corpus)
    result = _result(args)

    try:
        layer_s, tok_s = args.position.split(":")
        layer, tok = int(layer_s), int(tok_s)
    except ValueError:
        print("position 格式应为 layer:token，如 3:12", file=sys.stderr)
        return 1
    if not (0 <= tok < data.n_tokens):
        print(f"token 越界：0..{data.n_tokens - 1}", file=sys.stderr)
        return 1

    form = data.forms[tok]
    role = data.roles[tok]
    topic = data.position_topics[tok]
    is_amb = bool(data.is_ambiguous[tok])
    is_marker = bool(data.is_marker[tok])

    ref = result.reference_run()
    print(f"位置 L{layer}:token#{tok}")
    print(f"\n【已知真值】")
    print(f"  词形={form}  角色={role}  主题={topic}")
    print(f"  歧义位置={is_amb}（角色取决于主题）  标记词={is_marker}")

    if ref is not None:
        print(f"\n【what_happened · 该层表征编码了什么】")
        for prop, lab in (("form", form), ("role", role), ("topic", topic)):
            c = ref.curve(prop, "ambiguous" if (prop == "role" and is_amb) else "all")
            if c is None:
                continue
            acc = c.at(layer)
            if acc is None:
                continue
            onset = c.onset_layer
            formed = "已分化" if layer >= onset else "尚未分化"
            print(f"  {prop}: 在 L{layer} 的线性可分准确率={acc:.3f}（{formed}，"
                  f"分化层=L{onset}）")

        print(f"\n【why_here · 为什么是这个表示】")
        if is_amb:
            print("  该位置是歧义词：角色要等注意力扩到能看见主题标记词(L3/L4)才定，"
                  "所以 role 曲线在深层才跳变。")
        elif is_marker:
            print("  该位置是主题标记词：作为 attention sink，把主题信息广播到全句，"
                  "是 topic 属性在 L4 物化的来源。")
        else:
            print("  该位置是普通词：词形在 embedding 层即可读，角色经局部注意在 "
                  "L2 物化，主题随标记词广播在 L4 物化。")

    print(f"\n【where_from · 表示来自哪里】")
    print("  词形 → embedding；局部注意(L1/L2)写句法；主题标记词 sink(L3/L4)"
          "写语义；全部经残差流累加。")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    result = _result(args)
    print(f"模型：{', '.join(sorted({r.model for r in result.runs}))}")
    print(f"观测配置（{len(result.runs)}）：")
    for r in result.runs:
        ctrl = " [对照]" if r.spec.is_control else ""
        print(f"  {r.spec_id}{ctrl}")
    return 0


def _cmd_dag(args: argparse.Namespace) -> int:
    result = _result(args)
    if result.dag is None:
        print("无血缘图", file=sys.stderr)
        return 1
    if args.dot:
        print(result.dag.to_dot())
    else:
        print(result.dag.to_ascii())
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    result = _result(args)
    if result.version is None:
        print("无版本", file=sys.stderr)
        return 1
    print(json.dumps(result.version.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    from .visualize import render_html

    result = _result(args)
    out = Path(args.out)
    html = render_html(result, out / "visualize.html")
    print(f"已生成 {html}")
    return 0


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
#: 项目根：`src/{pkg}/cli.py` → 上三级。**所有相对路径默认值都锚到这里。**
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _anchor_args(args: argparse.Namespace) -> None:
    """把 `--config` / `--out` 的**相对路径**锚定到项目根。

    契约是 `cd <项目目录> && python main.py ...`，从项目目录里跑时锚不锚一个样
    —— 缺陷因此**长期隐形**。一旦从别处调用（例如跨项目的批量核对
    `python L5-xxx/main.py check`），`"configs/default.json"` 就指向了 cwd 下一个
    不存在的位置：严格 `load_config` 直接崩，宽松的会**静默回退 DEFAULTS**
    （更糟 —— 不报错，只是悄悄换了一套口径）。

    **在 `main()` 里紧跟 `parse_args` 锚一次**，比在每个调用点各锚一次可靠：
    漏掉一个调用点，就是一条只在某种调用方式下才现形的路径。

    绝对路径原样使用。语义与 `pytest` 的 rootdir / `cargo` / `npm --prefix` 一致。
    """
    # 只列举 parser 里真实存在的参数：`--config`（顶层）与 `--out`（run / visualize）。
    # 曾经还在这里写了个 "root"，但没有任何子命令定义过 `--root` ——
    # getattr 每次拿到 None，什么也不做，只让人误以为存在一个 --root 开关。
    for name in ("config", "out"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value:
            p = Path(value)
            if not p.is_absolute():
                setattr(args, name, str(_PKG_ROOT / p))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="representationfactory", description="测量型流水线")
    p.add_argument("--config", default="configs/default.json", help="配置文件路径；相对路径按项目根解析")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("run", help="跑全流程并落盘").add_argument(
        "--out", default="out", help="产物目录；相对路径按项目根解析"
    )

    sub.add_parser("capture", help="抓取概况")
    sub.add_parser("project", help="降维投影概况")
    sub.add_parser("compare", help="逐层分化 + 跨模型对比")
    sub.add_parser("list", help="列出全部观测配置")
    sub.add_parser("visualize", help="生成 HTML 可视化").add_argument(
        "--out", default="out", help="产物目录；相对路径按项目根解析"
    )

    sp = sub.add_parser("similarity", help="层间相似度矩阵")
    sp.add_argument("--spec", required=True)

    sp = sub.add_parser("probe", help="某 spec 的某类探针")
    sp.add_argument("--spec", required=True)
    sp.add_argument("--kind", required=True, choices=("similarity", "cluster", "geometry"))

    sp = sub.add_parser("curve", help="某 spec 的某条曲线")
    sp.add_argument("--spec", required=True)
    sp.add_argument("--property", required=True, choices=("form", "role", "topic"))
    sp.add_argument("--subset", default="all")

    sp = sub.add_parser("graph", help="某 spec 的表征图")
    sp.add_argument("--spec", required=True)

    sp = sub.add_parser("explain", help="验收三问")
    sp.add_argument("--position", required=True, help="layer:token")

    sp = sub.add_parser("dag", help="血缘图")
    sp.add_argument("--dot", action="store_true", help="输出 DOT 而非 ASCII")

    sub.add_parser("version", help="内容寻址版本")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _anchor_args(args)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    handlers = {
        "run": _cmd_run,
        "capture": _cmd_capture,
        "project": _cmd_project,
        "similarity": _cmd_similarity,
        "probe": _cmd_probe,
        "curve": _cmd_curve,
        "graph": _cmd_graph,
        "compare": _cmd_compare,
        "explain": _cmd_explain,
        "list": _cmd_list,
        "dag": _cmd_dag,
        "version": _cmd_version,
        "visualize": _cmd_visualize,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)
