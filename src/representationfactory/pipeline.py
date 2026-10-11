"""流水线编排（Pipeline / Assemble）。

README 第八节 M0–M6 的总装车间。一个 ``run`` 做的事：

    corpus → specimen → 每个模型 capture 一次 →
    每个观测配置跑三类探针 → 收拢成 graph → 跨模型对比 →
    生成表征图 + 逐层曲线 + 结论 + 血缘 + 版本 → 落盘。

三个本项目特有的纪律（与同工作区其它项目的 pipeline 不同）：

1. **capture 一次、观测多次。** 同一模型的 24 个 token 观测配置看的是
   **同一份**内部状态。``CaptureCache`` 保证这一点，且 ``probes`` 模块里
   跨 spec 的探针缓存进一步保证「同口径的配置只算一遍」。
   违反它不会报错，只是慢 24 倍 —— 所以 ``test_capture_shared`` 用对象同一性把它钉死。

2. **收拢点输入恰好是三类探针。** 每个 spec 的 ``graph`` 节点的父节点
   严格是 ``probe:{similarity|cluster|geometry}:{spec_id}`` 三个，一个不多、
   一个不少。这是「三类探针缺一不可」这句话在图上的可执行检查。

3. **版本号由「被观测对象的定义」决定，不由观测配置决定。**
   换一个相似度度量只该产生新的一行，不该产生新的版本号。
   所以 digest 只喂语料真值 + 模型结构 + 标本种子，不喂 ``observation``。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .capture import Capture, CaptureCache
from .compare import ablation_summary, assemble_findings, build_matrix
from .config import expand_specs, load_config, resolve_data_path
from .corpus import load_corpus, write_corpus
from .graph import build_representation_graph
from .lineage import (
    CORPUS_NODE_ID,
    COMPARE_NODE_ID,
    VERSION_NODE_ID,
    VIZ_NODE_ID,
    LineageDAG,
    LineageNode,
    capture_node_id,
    graph_node_id,
    model_node_id,
    project_node_id,
)
from .model import Model, build_model, specimen_data
from .project import Projection, reduce_states
from .probes import clear_probe_cache, run_probes
from .schema import (
    CODE_VERSION,
    CaptureRun,
    LabResult,
    PROBE_KINDS,
    Version,
    sha256_text,
    utcnow,
)

_PROBE_KINDS = PROBE_KINDS


def _version_digest(cfg: dict[str, Any], corpus: Any, model_names: list[str]) -> str:
    """版本 digest 只取决于「被观测对象的定义」。

    语料真值（lexicon.digest）+ 标本结构（d_model / n_layers / subspaces /
    code_dim / code_sparsity）+ 标本种子。换观测配置（``observation``）不该改它。
    """
    m = cfg["model"]
    structure = "|".join(
        [
            f"d{m['d_model']}",
            f"L{m['n_layers']}",
            f"H{m['n_heads']}",
            "-".join(f"{k}{v}" for k, v in sorted(m["subspaces"].items())),
            f"cd{m['code_dim']}",
            f"cs{m['code_sparsity']}",
            f"ns{cfg['corpus']['n_sentences']}",
            f"seed{cfg['seed']}",
        ]
    )
    lex_digest = corpus.lexicon.digest() if hasattr(corpus, "lexicon") else ""
    payload = "\x1f".join(
        [CODE_VERSION, lex_digest, structure, ",".join(sorted(model_names))]
    )
    return sha256_text(payload)[:12]


def _scatter_for_reference(
    cap: Capture, ref_run: CaptureRun, cfg: dict[str, Any], data: Any
) -> dict[str, Any]:
    """给参考配置算一份 PCA 散点（带真值标签），供可视化用。

    取 ``topic`` 属性分化最强的那一层（参考 run 里 ``topic`` 的最佳层），
    把该层 token 表征降到 2 维，最多取 ``max_scatter_points`` 个点，
    每个点带 ``form / role / topic / is_ambiguous / is_marker`` 标签。
    """
    curve = ref_run.curve("topic", "all")
    layer = curve.best_layer if curve else int(cfg["observation"]["layers"][-1])
    states = cap.hidden_at(layer)
    proj: Projection = reduce_states(
        states,
        "pca",
        cfg,
        seed_parts=("pca", int(cfg["seed"]), ref_run.spec_id, layer),
    )
    max_pts = int(cfg["observation"].get("max_scatter_points", 160))
    idx = list(range(len(states)))
    if len(idx) > max_pts:
        step = len(idx) / max_pts
        idx = [int(i * step) for i in range(max_pts)]
    points = []
    for i in idx:
        coord = proj.coords[i] if proj.coords else []
        points.append(
            {
                "x": round(coord[0], 4) if len(coord) > 0 else 0.0,
                "y": round(coord[1], 4) if len(coord) > 1 else 0.0,
                "form": data.forms[i],
                "role": data.roles_of(i) if hasattr(data, "roles_of") else data.roles[i],
                "topic": data.position_topics[i],
                "is_ambiguous": bool(data.is_ambiguous[i]),
                "is_marker": bool(data.is_marker[i]),
            }
        )
    return {
        "spec_id": ref_run.spec_id,
        "layer": layer,
        "method": "pca",
        "explained_variance": [round(v, 6) for v in proj.explained_variance],
        "n_points": len(points),
        "points": points,
    }


def _attention_samples(cap: Capture, spec_id: str) -> dict[str, Any]:
    """取参考配置的注意力样本，写成热图可读的结构。

    只保留 ``observation.attention_samples`` 句、每个 (层, 句, 头) 的
    ``n × n`` 矩阵 + 该句文本 + 标记词位置。完整矩阵很大，
    但「注意力长什么样」看几条就够，所以默认截断。
    """
    data = cap.data
    samples = []
    for si in cap.attention_sample_sentences:
        n = len(data.sequences[si])
        # 句内位置是连续的：start = 前 si 句的长度和
        start = sum(len(data.sequences[s]) for s in range(si))
        words = [data.forms[i] for i in range(start, start + n)]
        text = "".join(words)
        per_layer = []
        for l in range(len(cap.trace.attention)):
            per_head = []
            for h in range(len(cap.trace.attention[l][si])):
                per_head.append(
                    [[round(p, 4) for p in row] for row in cap.trace.attention[l][si][h]]
                )
            per_layer.append(per_head)
        samples.append(
            {
                "sentence_id": data.sentence_ids[si],
                "text": text,
                "topic": data.topics[si],
                "marker_index": data.marker_index[si],
                "n": n,
                "attention": per_layer,
            }
        )
    return {"spec_id": spec_id, "samples": samples}


def _build_lineage(result: LabResult) -> LineageDAG:
    """构造血缘图。节点粒度见 ``lineage.py`` 模块文档。"""
    dag = LineageDAG()
    dag.add_node(LineageNode(node_id=CORPUS_NODE_ID, kind="corpus", label="语料 + 已知真值"))

    models = sorted({r.model for r in result.runs})
    graph_nodes: list[str] = []
    for model_name in models:
        m_id = model_node_id(model_name)
        dag.add_node(
            LineageNode(node_id=m_id, kind="model", label=f"模型 {model_name}", inputs=[CORPUS_NODE_ID])
        )
        cap_id = capture_node_id(model_name)
        dag.add_node(
            LineageNode(node_id=cap_id, kind="capture", label=f"抓取 {model_name}", inputs=[m_id])
        )
        proj_id = project_node_id(model_name)
        dag.add_node(
            LineageNode(node_id=proj_id, kind="project", label=f"降维桶 {model_name}", inputs=[cap_id])
        )

        for r in result.runs:
            if r.model != model_name:
                continue
            probe_inputs = []
            for kind in _PROBE_KINDS:
                pid = f"probe:{kind}:{r.spec_id}"
                dag.add_node(
                    LineageNode(node_id=pid, kind="probe", label=f"{kind} {r.spec_id}", inputs=[proj_id])
                )
                probe_inputs.append(pid)
            # 收拢点：输入恰好是三类探针（排序后与 PROBE_KINDS 一致）
            g_id = graph_node_id(r.spec_id)
            dag.add_node(
                LineageNode(
                    node_id=g_id,
                    kind="graph",
                    label=f"表征图 {r.spec_id}",
                    inputs=sorted(probe_inputs),
                )
            )
            graph_nodes.append(g_id)

    dag.add_node(
        LineageNode(
            node_id=COMPARE_NODE_ID,
            kind="compare",
            label="跨模型 / 跨层对比",
            inputs=sorted(graph_nodes),
        )
    )
    dag.add_node(
        LineageNode(
            node_id=VIZ_NODE_ID,
            kind="viz",
            label="可视化主节点",
            inputs=sorted([COMPARE_NODE_ID] + graph_nodes),
        )
    )
    dag.add_node(
        LineageNode(
            node_id=VERSION_NODE_ID,
            kind="version",
            label="内容寻址版本",
            inputs=[COMPARE_NODE_ID],
        )
    )
    return dag


def _assemble(result: LabResult, captures: dict[str, Capture], data: Any) -> None:
    result.matrix = build_matrix(result)
    result.ablations = ablation_summary(result)
    result.findings = assemble_findings(result)
    ref = result.reference_run()
    result.selected = ref.spec_id if ref else ""
    if ref is not None and "constructive" in captures:
        result.counters["scatter"] = _scatter_for_reference(
            captures["constructive"], ref, result.cfg, data
        )
    result.dag = _build_lineage(result)
    model_names = list(result.cfg["model"]["models"])
    corpus = load_corpus(result.cfg)
    digest = _version_digest(result.cfg, corpus, model_names)
    result.version = Version(
        version_id=f"{result.cfg.get('version_tag', 'v1')}-{digest}",
        created_at=utcnow(),
        digest=digest,
        tag=str(result.cfg.get("version_tag", "v1")),
        counts={
            "n_runs": len(result.runs),
            "n_models": len(model_names),
            "n_sentences": data.n_sentences,
        },
    )


def run_pipeline(cfg: dict[str, Any]) -> LabResult:
    """跑完整条流水线，返回 ``LabResult``（不落盘）。"""
    clear_probe_cache()
    corpus = load_corpus(cfg)
    data = specimen_data(corpus)
    model_names = list(cfg["model"]["models"])

    cache = CaptureCache()
    captures: dict[str, Capture] = {}
    for model_name in model_names:
        model = build_model(model_name, cfg, corpus.lexicon, data)
        captures[model_name] = cache.get(model, data, cfg)

    runs: list[CaptureRun] = []
    for spec in expand_specs(cfg):
        if spec.model not in captures:
            continue
        curves, probes = run_probes(captures[spec.model], spec, cfg, corpus.lexicon)
        runs.append(
            CaptureRun(spec=spec, curves=curves, probes=probes, heads=captures[spec.model].heads)
        )

    result = LabResult(corpus=corpus.stats(str(cfg.get("corpus", {}).get("name", "corpus"))), runs=runs, cfg=cfg)
    _assemble(result, captures, data)
    result._captures = captures  # 供 write_outputs 取注意力样本
    result._corpus = corpus  # 供 run_and_write 把生成的语料落盘
    return result


def write_outputs(result: LabResult, out_dir: str | Path) -> list[Path]:
    """把 11 个产物落盘。返回写出的文件路径列表。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    (out / "corpus_stats.json").write_text(
        json.dumps(result.corpus.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(out / "corpus_stats.json")

    if result.matrix:
        cols = list(result.matrix[0].keys())
        with (out / "matrix.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in result.matrix:
                w.writerow(row)
    written.append(out / "matrix.csv")

    curves_blob = {
        r.spec_id: {k: c.to_dict() for k, c in sorted(r.curves.items())} for r in result.runs
    }
    (out / "curves.json").write_text(
        json.dumps(curves_blob, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(out / "curves.json")

    probes_blob = {
        r.spec_id: {k: p.to_dict() for k, p in sorted(r.probes.items())} for r in result.runs
    }
    (out / "probes.json").write_text(
        json.dumps(probes_blob, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(out / "probes.json")

    graphs_blob = {}
    for r in result.runs:
        geo = r.probes.get("geometry")
        if geo is not None:
            graphs_blob[r.spec_id] = build_representation_graph(geo).to_dict()
    (out / "graphs.json").write_text(
        json.dumps(graphs_blob, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(out / "graphs.json")

    (out / "findings.json").write_text(
        json.dumps(result.findings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(out / "findings.json")

    if result.dag is not None:
        result.dag.save(out / "lineage.json")
    written.append(out / "lineage.json")

    scatter = result.counters.get("scatter", {})
    (out / "scatter.json").write_text(
        json.dumps(scatter, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(out / "scatter.json")

    captures = getattr(result, "_captures", {}) or {}
    ref = result.reference_run()
    if ref is not None and "constructive" in captures:
        attn = _attention_samples(captures["constructive"], ref.spec_id)
    else:
        attn = {}
    (out / "attention.json").write_text(
        json.dumps(attn, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(out / "attention.json")

    if result.version is not None:
        (out / "version.json").write_text(
            json.dumps(result.version.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    written.append(out / "version.json")

    # 报告（Markdown，可 git diff / 可测试核对）
    from .report import render_markdown

    (out / "report.md").write_text(render_markdown(result), encoding="utf-8")
    written.append(out / "report.md")

    return written


def run_and_write(cfg: dict[str, Any], out_dir: str | Path) -> LabResult:
    """跑流水线并把全部产物落盘；返回 ``LabResult``。

    除 ``out/`` 的产物外，还把生成的语料写回 ``cfg.corpus.sentences``。
    配置文件把 ``sentences`` 声明成一个路径，``load_corpus`` 也按「文件存在
    且词典 digest 一致就直接读、否则重新生成」来用它 —— 所以 ``run`` 必须
    真的把语料落盘，否则那个缓存路径永远是空的、``load_corpus`` 每次都重建
    （并让 ``write_corpus`` 变成死代码）。
    """
    result = run_pipeline(cfg)
    write_outputs(result, out_dir)
    corpus = getattr(result, "_corpus", None)
    if corpus is not None:
        write_corpus(corpus, resolve_data_path(cfg, "sentences"))
    return result


def load_result(out_dir: str | Path) -> LabResult | None:
    """从 ``out/`` 的产物重建 ``LabResult``（只读命令用它，**不重跑流水线**）。

    重建需要 6 个产物：``curves.json`` / ``probes.json`` / ``findings.json`` /
    ``version.json`` / ``lineage.json`` / ``corpus_stats.json``；
    ``matrix.csv`` 可选，缺失时矩阵留空、其余字段照常。
    任一**必需**产物缺失即返回 ``None``（调用方此时改为 ``run_pipeline`` 计算）。
    **本函数只读不写** —— 这是「只读命令不改产物」约束的底层保证。
    """
    out = Path(out_dir)
    needed = [
        out / "curves.json",
        out / "probes.json",
        out / "findings.json",
        out / "version.json",
        out / "lineage.json",
        out / "corpus_stats.json",
    ]
    if not all(p.exists() for p in needed):
        return None

    import csv as _csv

    from .lineage import LineageDAG
    from .schema import CaptureRun, CorpusStats, LabResult, Version

    curves_blob = json.loads((out / "curves.json").read_text(encoding="utf-8"))
    probes_blob = json.loads((out / "probes.json").read_text(encoding="utf-8"))
    findings = json.loads((out / "findings.json").read_text(encoding="utf-8"))
    version = Version.from_dict(
        json.loads((out / "version.json").read_text(encoding="utf-8"))
    )
    dag = LineageDAG.from_dict(
        json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    )
    corpus = CorpusStats.from_dict(
        json.loads((out / "corpus_stats.json").read_text(encoding="utf-8"))
    )

    runs: list[CaptureRun] = []
    for spec_id in sorted(curves_blob):
        cdict = curves_blob[spec_id]
        pdict = probes_blob.get(spec_id, {})
        d = {"spec_id": spec_id, "curves": cdict, "probes": pdict, "heads": []}
        runs.append(CaptureRun.from_dict(d))

    # matrix.csv
    matrix: list[dict[str, Any]] = []
    mpath = out / "matrix.csv"
    if mpath.exists():
        with mpath.open(encoding="utf-8", newline="") as fh:
            for row in _csv.DictReader(fh):
                matrix.append({k: (v if k == "spec_id" else _coerce(v)) for k, v in row.items()})

    ref_spec = findings.get("reference_spec")
    result = LabResult(
        corpus=corpus,
        runs=runs,
        matrix=matrix,
        ablations=findings.get("ablation", {}),
        findings=findings,
        selected=ref_spec or "",
        dag=dag,
        version=version,
        cfg={},
    )
    # 把散点 / 注意力也读进 counters，让 visualize 从磁盘也能渲染
    scatter_path = out / "scatter.json"
    if scatter_path.exists():
        result.counters["scatter"] = json.loads(scatter_path.read_text(encoding="utf-8"))
    attn_path = out / "attention.json"
    if attn_path.exists():
        result.counters["attention_sample"] = json.loads(
            attn_path.read_text(encoding="utf-8")
        )
    return result


def _coerce(v: str) -> Any:
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except ValueError:
        return v
