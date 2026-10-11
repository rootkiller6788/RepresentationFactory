"""三类探针（Probe Engine）。

README 第六节给了三类，缺一不可 —— 它们回答三个**不同的**问题，
而且互相验证：

| 探针 | 回答什么 | 本项目怎么实现它 |
| --- | --- | --- |
| **Similarity** | 哪些 token / 样本在表征空间里靠近？ | 层 × 层相似度矩阵 + 逐位置近邻 |
| **Cluster**    | 表征里是否已**线性可分**地编码了某属性？ | 闭式岭回归线性探针（**不训练、无种子**） |
| **Geometry**   | 结构与表征如何耦合？哪些头在做同一件事？ | 头画像聚类 + 表征图 |

Cluster Probe 是本项目的主证据
------------------------------
逐层演化曲线就是它。实现上三个刻意的选择：

1. **用闭式岭回归，不用 SGD。**
   ``W = (XᵀX + λI)⁻¹ XᵀY`` 完全确定：没有初始化、没有学习率、没有早停，
   于是「探针准确率」只是一个**关于数据的函数**，两次运行必然一致。
   用 SGD 的话，「曲线上的抖动」里会混进优化噪声，
   而本项目的结论恰恰是「曲线在哪一层跳」—— 抖一下就会指认错层。

2. **必须按句子划分训练 / 测试。**
   这是本项目最容易犯也最贵的一个方法学错误：按 token 随机划分时，
   同一句内的相邻位置会被分到两侧，而它们高度相关
   （本项目的构造式标本里，注意力会做局部混合，相邻位置的表征天然相似）。
   于是测试集里混进了训练集的「同句兄弟」，准确率**系统性虚高** ——
   而且是安静地虚高，不报错、不警告。
   本项目把它做成一个**显式的对照轴**（``split``），
   矩阵里同时给出两种划分，让读者自己看到差距。

3. **必须同时报多数类基线与 chance level。**
   0.42 这个数字到底是「学到了」还是「数据本来就偏」，没有基线根本说不清。
   本项目的角色分布很不均匀（NOUN 占 33%），多数类基线就有 0.33。
"""

from __future__ import annotations

import math
from typing import Any

from .capture import Capture
from .corpus import Lexicon
from .linalg import add_ridge, gram, solve_multi
from .project import reduce_states
from .rng import make_rng
from .schema import HeadStats, LayerCurve, ProbeResult, PROBE_KINDS, StateSpec

#: 三种划分/评测口径的角色子集。``unambiguous`` 是 ``role`` 曲线的第三档 ——
#: 它与 ``ambiguous`` 的分离正是「句法分化发生在第几层」这件事的直接证据。
ROLE_SUBSETS: tuple[str, ...] = ("all", "ambiguous", "unambiguous")

#: 跨 spec 的探针结果缓存。**它的存在不是加速，是正确性前提的次生收益**：
#: 同一模型的 24 个 token 观测配置里，``cluster`` 探针的结果只取决于
#: ``(pooling, project, split)`` 三个量，与 ``similarity`` 无关 —— 不缓存的话
#: 同样的闭式计算会被重复 3 遍；``geometry`` 探针的 ``heads`` 来自同一个
#: ``Capture``，更不该每 spec 重算。缓存按「被观测对象身份 + 相关轴」键控，
#: 既去重又不串味（见下方 ``_cache_key_*``）。
_PROBE_CACHE: dict[tuple, Any] = {}


def _capture_key(capture: "Capture") -> tuple:
    """被观测对象的身份指纹。

    用模型名 + 语料规模 + 注意力样本数，而不用 ``id(capture)`` ——
    ``id`` 在对象回收后会被 Python 复用到新对象上，跨多次 ``pipeline.run``
    时可能撞键（测试里反复建模型就会触发）。规模指纹在「同一份标本」上
    必然相同、不同标本必然不同。
    """
    d = capture.data
    return (
        capture.model.name,
        d.n_tokens,
        d.n_sentences,
        int(capture.meta.get("attention_samples", -1)),
    )


def _cluster_cache_key(capture: "Capture", spec: "StateSpec") -> tuple:
    return (_capture_key(capture), "cluster", spec.pooling, spec.project, spec.split)


def _similarity_cache_key(capture: "Capture", spec: "StateSpec") -> tuple:
    return (_capture_key(capture), "sim", spec.pooling, spec.similarity, spec.project)


def _geometry_cache_key(capture: "Capture") -> tuple:
    return (_capture_key(capture), "geo")


def clear_probe_cache() -> None:
    """清空跨 spec 探针缓存。

    缓存按「被观测对象身份」键控，所以正常情况下跨模型/跨 run 不会串味；
    但测试里会反复构造**规模相同**的不同标本，若不清缓存，上一次的
    探针结果会被当成这一次的返回（缓存键撞了）。``pipeline.run`` 与
    每个涉及探针的测试 setUp 都应调一次。
    """
    _PROBE_CACHE.clear()



# ----------------------------------------------------------------------
# 池化与标签
# ----------------------------------------------------------------------
def states_for(capture: Capture, layer: int, pooling: str) -> list[list[float]]:
    """取某一层的表征。

    ``token``    = 每个位置一行（本项目的主要视图）；
    ``sentence`` = 每句一行（句内均值），**只用于句级属性**。
    """
    H = capture.hidden_at(layer)
    if pooling == "token":
        return [list(v) for v in H]
    if pooling == "sentence":
        out: list[list[float]] = []
        pos = 0
        for seq in capture.data.sequences:
            n = len(seq)
            if n == 0:
                continue
            block = H[pos : pos + n]
            d = len(block[0])
            means = [sum(row[j] for row in block) / n for j in range(d)]
            out.append(means)
            pos += n
        return out
    raise ValueError(f"未知的池化方式 {pooling!r}")


def labels_for(
    capture: Capture,
    prop: str,
    pooling: str,
    subset: str = "all",
) -> tuple[list[str], list[int]]:
    """取标签 + 参与评测的位置下标。

    返回 ``(labels, row_indices)``：``row_indices`` 是 ``states_for`` 结果里的行号。
    分开返回而不是直接过滤，是为了让「过滤掉哪些位置」这件事显式可见 ——
    静默少算会让曲线看起来很平，而读者会以为是「这一层没编码这个属性」。
    """
    data = capture.data
    if pooling == "sentence":
        return [t for t in data.topics], list(range(len(data.sequences)))
    if prop == "form":
        labels = list(data.forms)
    elif prop == "role":
        labels = list(data.roles)
    elif prop == "topic":
        labels = list(data.position_topics)
    else:
        raise ValueError(f"未知属性 {prop!r}")
    if subset == "all":
        rows = list(range(len(labels)))
    elif subset == "ambiguous":
        rows = [i for i, a in enumerate(data.is_ambiguous) if a]
    elif subset == "unambiguous":
        rows = [i for i, a in enumerate(data.is_ambiguous) if not a]
    else:
        raise ValueError(f"未知子集 {subset!r}")
    return labels, rows


# ----------------------------------------------------------------------
# 划分
# ----------------------------------------------------------------------
def train_test_split(
    capture: Capture,
    spec: StateSpec,
    cfg: dict[str, Any],
    rows: list[int],
) -> tuple[list[int], list[int]]:
    """划分训练 / 测试。

    ``by_sentence``：先按句分组，把句子随机分成两堆，再把位置映射到各自那堆。
    **同一句的所有位置必然同侧** —— 这是无泄漏的口径。

    ``by_token``：直接按位置随机分。这是**对照**，不是可选实现：
    它的数字会更高，而那正是它要展示的陷阱。
    """
    data = capture.data
    frac = float(cfg["observation"]["test_fraction"])
    rng = make_rng("split", int(cfg["seed"]), spec.spec_id)
    if spec.split == "by_sentence":
        n_sentences = len(data.sequences)
        order = list(range(n_sentences))
        rng.shuffle(order)
        n_test = max(1, int(round(n_sentences * frac)))
        test_sentences = set(order[:n_test])
        sent_of = data.sentence_of
        train, test = [], []
        for r in rows:
            (test if sent_of[r] in test_sentences else train).append(r)
        return train, test
    if spec.split == "by_token":
        order = list(rows)
        rng.shuffle(order)
        n_test = max(1, int(round(len(order) * frac)))
        return order[n_test:], order[:n_test]
    raise ValueError(f"未知的划分方式 {spec.split!r}")


# ----------------------------------------------------------------------
# 闭式线性探针
# ----------------------------------------------------------------------
def fit_ridge(
    x: list[list[float]],
    labels: list[str],
    lam: float,
) -> tuple[list[str], list[list[float]]]:
    """闭式岭回归。返回 ``(classes, W)``，``W`` 是 ``d × C``。

    **没有随机性、没有迭代**：同样的输入必得同样的 W。
    这是「曲线可复现」的前提（见模块文档第 1 点）。
    """
    classes = sorted(set(labels))
    xtx = gram(x)
    d = len(xtx)
    # ``XᵀY`` 按类累加，而不是真的去乘 one-hot 矩阵。
    # 乘 one-hot 是 O(n·d·C)：217 个词形 × 1664 个训练样本 × 48 维 ≈ 1700 万次
    # Python 运算；按类累加是 O(n·d) ≈ 8 万次（Y 的每一列只有一个 1）。
    # **必须这么写**：曲线是 48 个观测配置 × 5 层都要跑的。
    index = {c: i for i, c in enumerate(classes)}
    xty = [[0.0] * len(classes) for _ in range(d)]
    for row, lab in zip(x, labels):
        ci = index[lab]
        for r in range(d):
            xty[r][ci] += row[r]
    w = solve_multi(add_ridge(xtx, lam), xty)
    return classes, w


def _predict(
    x: list[list[float]], classes: list[str], w: list[list[float]]
) -> list[str]:
    # ``w`` 是 d×C。**先取列**再用 ``math.sumprod``：直接按行写生成器是
    # O(n·C·d) 次 Python 运算，取列后是 n·C 次 C 层调用（实测快约 20 倍）。
    cols = [[w[r][c] for r in range(len(w))] for c in range(len(classes))]
    out: list[str] = []
    for row in x:
        best_i, best_v = 0, -1e30
        for c in range(len(classes)):
            v = math.sumprod(row, cols[c]) if hasattr(math, "sumprod") else sum(
                a * b for a, b in zip(row, cols[c])
            )
            if v > best_v:
                best_i, best_v = c, v
        out.append(classes[best_i])
    return out


def _majority(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    return max(counts.values()) / len(labels)


# ----------------------------------------------------------------------
# Cluster Probe
# ----------------------------------------------------------------------
def _projected_states(
    states: list[list[float]], spec: StateSpec, cfg: dict[str, Any]
) -> tuple[list[list[float]], Any]:
    """按 spec 决定是否降维后探针。

    ``project="pca"`` 时探针只在 2 维上做 —— 这会**显著降低**准确率，
    而那正是这一行要展示的事实：「降维丢掉了多少可分性」。
    """
    if spec.project == "none":
        return states, None
    proj = reduce_states(
        states,
        "pca" if spec.project == "pca" else spec.project,
        cfg,
        seed_parts=("pca", int(cfg["seed"]), spec.spec_id),
    )
    return proj.coords, proj


def run_cluster_probe(
    capture: Capture,
    spec: StateSpec,
    cfg: dict[str, Any],
    lexicon: Lexicon,
) -> tuple[dict[str, LayerCurve], ProbeResult]:
    """跑 Cluster Probe，返回逐属性的曲线 + 探针自己的指标。

    两层去重，否则 48 个配置跑不完：

    1. **跨 spec 缓存**：结果只取决于 ``(pooling, project, split)``，
       与 ``similarity`` 无关。同一模型的 24 个 token 配置因此只算 4 遍。
    2. **spec 内按层复用**：``states_for`` / 投影 / ``gram`` 在每个 layer 上
       对所有属性/子集是同一份，先算好缓存，避免每个 ``(属性, 子集, 层)``
       都重算一遍 ``XᵀX``（后者是纯 Python 热点）。
    """
    key = _cluster_cache_key(capture, spec)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    layers = [int(x) for x in cfg["observation"]["layers"]]
    lam = float(cfg["probe"]["ridge_lambda"])
    min_class = int(cfg["probe"]["min_class_count"])
    subsets = list(cfg["observation"]["role_subsets"])

    # 按层预计算投影后的状态（token 池化 = 每位置一行；sentence = 每句一行）。
    # 同一 layer 的所有属性/子集共用这一份。
    proj_by_layer: dict[int, list[list[float]]] = {}
    for layer in layers:
        X = states_for(capture, layer, spec.pooling)
        Xp, _ = _projected_states(X, spec, cfg)
        proj_by_layer[layer] = Xp

    curves: dict[str, LayerCurve] = {}
    skipped: list[str] = []
    for prop in spec.applicable_properties:
        wanted = subsets if prop == "role" else ["all"]
        # ``all`` 必须在列里，否则曲线集不完整（报告会静默少一档）
        for subset in wanted:
            all_labels, rows = labels_for(capture, prop, spec.pooling, subset)
            if len(rows) < 2:
                skipped.append(f"{prop}/{subset}: 位置太少")
                continue
            train_rows, test_rows = train_test_split(capture, spec, cfg, rows)
            if not train_rows or not test_rows:
                skipped.append(f"{prop}/{subset}: 划分后有一侧为空")
                continue
            train_labels = [all_labels[r] for r in train_rows]
            test_labels = [all_labels[r] for r in test_rows]
            if len(set(train_labels)) < 2:
                skipped.append(f"{prop}/{subset}: 训练集只有一类")
                continue
            # **丢弃稀有类**而不是跳过整条曲线：217 个词形里总有几十个
            # 出现不足 ``min_class`` 次的位置（这是 280 句语料的固有稀疏性，
            # 不是 bug）。把它们当 OOV 剔除，只在「被充分表示」的词形上衡量
            # 「词形身份是否线性可读」—— 这才是我们要的论断。否则 ``form``
            # 这条「浅层可读」的对照组会整条消失，报告不完整。
            global_counts: dict[str, int] = {}
            for lab in train_labels + test_labels:
                global_counts[lab] = global_counts.get(lab, 0) + 1
            keep = {lab for lab, c in global_counts.items() if c >= min_class}
            if len(keep) < 2:
                skipped.append(f"{prop}/{subset}: 保留类不足 2（稀有类过多）")
                continue
            tr2 = [r for r in train_rows if all_labels[r] in keep]
            te2 = [r for r in test_rows if all_labels[r] in keep]
            if not tr2 or not te2:
                skipped.append(f"{prop}/{subset}: 剔除稀有类后有一侧为空")
                continue
            train_rows, test_rows = tr2, te2
            train_labels = [all_labels[r] for r in train_rows]
            test_labels = [all_labels[r] for r in test_rows]
            accs: list[float] = []
            for layer in layers:
                Xp = proj_by_layer[layer]
                xtr = [Xp[r] for r in train_rows]
                xte = [Xp[r] for r in test_rows]
                classes, w = fit_ridge(xtr, train_labels, lam)
                pred = _predict(xte, classes, w)
                accs.append(sum(int(p == t) for p, t in zip(pred, test_labels)) / len(test_labels))

            curves[f"{prop}/{subset}"] = LayerCurve(
                spec_id=spec.spec_id,
                property=prop,
                layers=list(layers),
                accuracy=accs,
                majority=_majority(test_labels),
                n_classes=len(keep),
                n_train=len(train_rows),
                n_test=len(test_rows),
                subset=subset,
            )

    metrics = {
        "n_curves": float(len(curves)),
        "mean_deep_gain": (
            sum(c.deep_gain for c in curves.values()) / len(curves) if curves else 0.0
        ),
        "max_deep_gain": max((c.deep_gain for c in curves.values()), default=0.0),
        "n_skipped": float(len(skipped)),
    }
    detail = {
        "skipped": skipped,
        "curves": {k: c.to_dict() for k, c in sorted(curves.items())},
    }
    result = (curves, ProbeResult(
        probe="cluster", spec_id=spec.spec_id, metrics=metrics, detail=detail
    ))
    _PROBE_CACHE[key] = result
    return result


# ----------------------------------------------------------------------
# Similarity Probe
# ----------------------------------------------------------------------
def run_similarity_probe(
    capture: Capture,
    spec: StateSpec,
    cfg: dict[str, Any],
) -> ProbeResult:
    """层 × 层相似度矩阵 + 逐位置近邻。

    ``layer_similarity`` 矩阵是它最有用的产物：对角线 1、
    越远越小，而**下降的陡峭程度**就是「表征逐层重写的速度」。
    """
    key = _similarity_cache_key(capture, spec)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    from .similarity import layer_similarity_matrix

    layers = [int(x) for x in cfg["observation"]["layers"]]
    by_layer: dict[int, list[list[float]]] = {}
    for layer in layers:
        X = states_for(capture, layer, spec.pooling)
        Xp, _ = _projected_states(X, spec, cfg)
        by_layer[layer] = Xp
    matrix = layer_similarity_matrix(by_layer, layers, spec.similarity)

    # 「相邻层的相似度」与「跨全网的相似度」—— 两个标量，用来读曲线的形状
    adjacent = [matrix[i][i + 1] for i in range(len(layers) - 1)]
    far = matrix[0][-1] if len(layers) > 1 else 1.0
    metrics = {
        "adjacent_mean": sum(adjacent) / len(adjacent) if adjacent else 0.0,
        "adjacent_min": min(adjacent) if adjacent else 0.0,
        "far": far,
        # 「逐层重写的剧烈程度」：相邻相似度的下降幅度越大，说明层在做的事越不一样
        "rewrite_steepness": (sum(adjacent) / len(adjacent) - far) if adjacent else 0.0,
    }
    result = ProbeResult(
        probe="similarity",
        spec_id=spec.spec_id,
        metrics=metrics,
        detail={
            "layers": list(layers),
            "metric": spec.similarity,
            "matrix": [[round(v, 6) for v in row] for row in matrix],
        },
    )
    _PROBE_CACHE[key] = result
    return result


# ----------------------------------------------------------------------
# Geometry Probe
# ----------------------------------------------------------------------
def _head_profile(stats: HeadStats) -> list[float]:
    """把一个头压成一个 5 维画像向量（用于头之间的相似度）。"""
    return [
        stats.entropy,
        stats.self_weight,
        stats.locality,
        stats.sink_weight,
        stats.mean_offset,
    ]


def run_geometry_probe(
    capture: Capture,
    spec: StateSpec,
    cfg: dict[str, Any],
) -> ProbeResult:
    """头画像 + 头之间的相似度 + 粗聚类。

    「哪些头在做同一件事」这个问题，靠看注意力矩阵是答不出来的 ——
    它得靠**画像之间的相似度**。本项目用画像向量的余弦相似度，
    再按阈值取连通分量当聚类。

    聚类是**连通分量**而不是 k-means：k 要事先定，
    而「有几个功能组」正是我们要观测的结论，不能先假设。
    """
    key = _geometry_cache_key(capture)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    heads = capture.heads
    profiles = [_head_profile(h) for h in heads]
    n = len(profiles)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = _cos(profiles[i], profiles[j])
            sim[i][j] = v
            sim[j][i] = v

    threshold = float(cfg.get("probe", {}).get("head_cluster_threshold", 0.99))
    clusters = _connected(sim, threshold, [h.node_id for h in heads])

    # 「结构是否与表征耦合」的可读答案：sink 型头有几个、局部型头有几个
    sink_heads = [h for h in heads if h.sink_weight > 0.25]
    local_heads = [h for h in heads if h.locality > 0.25]
    metrics = {
        "n_heads": float(n),
        "n_clusters": float(len(clusters)),
        "n_sink_heads": float(len(sink_heads)),
        "n_local_heads": float(len(local_heads)),
        "max_head_similarity_offdiag": _max_offdiag(sim),
        "mean_head_similarity_offdiag": _mean_offdiag(sim),
    }
    result = ProbeResult(
        probe="geometry",
        spec_id=spec.spec_id,
        metrics=metrics,
        detail={
            "heads": [h.to_dict() for h in heads],
            "clusters": clusters,
            "threshold": threshold,
            "similarity": [[round(v, 6) for v in row] for row in sim],
        },
    )
    _PROBE_CACHE[key] = result
    return result


def _cos(u: list[float], v: list[float]) -> float:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


def _max_offdiag(sim: list[list[float]]) -> float:
    n = len(sim)
    best = -1.0
    for i in range(n):
        for j in range(n):
            if i != j:
                best = max(best, sim[i][j])
    return best if best >= -1.0 else 0.0


def _mean_offdiag(sim: list[list[float]]) -> float:
    n = len(sim)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                total += sim[i][j]
    return total / (n * (n - 1))


def _connected(
    sim: list[list[float]], threshold: float, node_ids: list[str]
) -> list[list[str]]:
    """按阈值取连通分量。

    每个分量内部按节点 id 排序、分量之间按最小节点 id 排序 ——
    都是为了让输出与遍历顺序无关（否则 ``lineage`` 之类依赖它排序的产物
    会在「同样的数据」上给出两种结果）。
    """
    n = len(sim)
    seen = [False] * n
    out: list[list[str]] = []
    for start in range(n):
        if seen[start]:
            continue
        comp: list[int] = []
        stack = [start]
        seen[start] = True
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in range(n):
                if not seen[nxt] and sim[cur][nxt] >= threshold:
                    seen[nxt] = True
                    stack.append(nxt)
        out.append(sorted(node_ids[c] for c in comp))
    return sorted(out, key=lambda c: c[0] if c else "")


def run_probes(
    capture: Capture,
    spec: StateSpec,
    cfg: dict[str, Any],
    lexicon: Lexicon,
) -> tuple[dict[str, LayerCurve], dict[str, ProbeResult]]:
    """跑完三类探针。**三类都要跑** ——
    缺任意一类，``graph`` 节点的输入就少一个，而那正是结构不变量要守的东西。"""
    curves, cluster = run_cluster_probe(capture, spec, cfg, lexicon)
    probes: dict[str, ProbeResult] = {"cluster": cluster}
    probes["similarity"] = run_similarity_probe(capture, spec, cfg)
    probes["geometry"] = run_geometry_probe(capture, spec, cfg)
    missing = [k for k in PROBE_KINDS if k not in probes]
    if missing:
        raise RuntimeError(f"探针缺失：{missing}")
    return curves, probes
