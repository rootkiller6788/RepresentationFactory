"""配置：全部阈值 / 权重 / 口径都在这里，不硬编码进代码。

三层结构（与同工作区其它项目一致）：

    corpus        被观测的语料与词典（真值来源）
    model         被观测的标本（微型 transformer 的结构与构造常量）
    observation   观测配置（五个观测轴的取值集合 + 探针/降维参数）
    ablation      单变量消融的参考配置
    select        最终入选规则

``DEFAULTS`` 里的每个值都必须与 ``configs/default.json`` **一致** ——
这一条有测试钉住。它踩过坑：默认值与被加载的配置不一致时，
「用默认配置跑」与「用仓库里的配置跑」会给出两份不同的结果，
而报告里只写了一份，读者无从判断看的是哪一份。

本项目没有「设计空间」这一层 —— ``observation`` 描述的是「怎么观测」，
五个轴交叉展开成 48 个观测配置（2 模型 × 2 池化 × 3 相似度 × 2 划分 × 2 降维）。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from .schema import MODELS, OBSERVATION_AXES, PROPERTIES, StateSpec

DEFAULTS: dict[str, Any] = {
    "pipeline_id": "representationfactory",
    "version_tag": "v1",
    "seed": 20260930,
    "corpus": {
        "lexicon": "data/lexicon/lexicon.jsonl",
        "sentences": "data/sentences/sentences.jsonl",
        "name": "representationfactory-demo",
        # 句子条数。**由「每个词形在训练集里至少出现 3 次」倒推出来**：
        # 词表 217 个词形、每句 8 个位置，280 句 = 2240 位置 ⇒ 每词形约 10 次、
        # 训练集（65%）约 6.5 次。少于这个数，``form`` 这条曲线会因为
        # 「某类训练样本不足 min_class_count」被整条跳过 —— 而它正是
        # 「浅层属性读得出来」的对照组，缺了它曲线集就不完整。
        #
        # 这条数**只影响运行时间，不影响结论**：标本的定性结论
        # （role 在 L2/L3、topic 在 L4 分化）来自「词表规模 vs code_dim」的
        # 叠加关系，与句子数无关。560 句时全流水线约 3 分钟，降到 280 后
        # 约 1 分钟，曲线数字不变（已核对）。
        # 踩过的坑：320 句时歧义槽概率 0.55，导致通用词形被挤到每词形仅 2-5 次，
        # 50 个词形训练样本 <3，form 曲线被跳过。
        "n_sentences": 280,
    },
    "model": {
        # 两个模型：构造式标本（真值已知）+ 随机权重对照（证明阶梯不是白来的）
        "models": ["constructive", "random"],
        "d_model": 48,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 96,
        # 残差流按子空间分段，每段承担一类信息。**和必须等于 d_model**（有校验）。
        "subspaces": {"lexical": 32, "syntax": 8, "semantic": 4, "marker": 4},
        # 词形码的**有效维数**。lexical 段有 32 维，码就占满这 32 维
        # （``code_dim`` ≤ lexical 段宽度，当前两者相等）。
        #
        # 它是整个标本最关键的一个旋钮，因为它同时被两个**相反**的约束夹着：
        #
        # - 要**大**：FFN 查表单元的信噪比是 ``sparsity / √(sparsity²/code_dim) = √code_dim``。
        #   code_dim=8 时只有 2.83，实测 L2 的 14 个单元**一个都标定不干净**
        #   （阈值落在重叠区 → 一半正样本漏检、一批负样本误点火 → 曲线是平的）。
        # - 要**小**：``code_dim`` 越大，「词形 → 角色」越可能线性可分。
        #   判据可以解析地算出来：类中心分类器 ``w_j = mean(code of role j)``
        #   对正样本的响应 ≈ 1/a、噪声 ≈ 1/√(a·D)（a = 该角色的词形数），
        #   信噪比 = √(D/a) ⇒ **a < D 时角色线性可读，a > D 时读不出来**。
        #
        # 于是**词表必须跟着变大**：当前每个角色有 40+ 个词形 > code_dim = 32，
        # 于是「角色」这个划分不可线性分 —— 这正是「叠加」（superposition）：
        # 特征数远多于维数，大部分属性无法被线性读出，必须由后面的层物化出来。
        #
        # 踩过的坑（三次，一次比一次隐蔽）：
        # 1. 24 个词形 + 24 维码：p ≈ 2d 时**任何**划分都线性可分，
        #    role 曲线在 L0 就是 1.000，整条曲线是平的，「中层句法」根本观察不到；
        # 2. 49 个词形 + 16 维码：a/D = 1.5，role 曲线 L0 仍有 0.76；
        # 3. 49 个词形 + 32 维码：单元标定干净了（33/39），但 L0 反而升到 0.94 ——
        #    **只调 D 不调词表，两个目标会互相打架**。
        # 现在的配比（每角色 ≥40 个词形 / 32 维码，a/D ≈ 1.25）是扫出来的：
        # L0 = 0.51（≈ 多数类基线）、L2 = 0.98，跳变 0.47。
        "code_dim": 32,
        # 每个词形码里非零分量的个数（稀疏 ±1）。
        # 稀疏是为了把「自点积 = sparsity」与「互点积 ~1」拉开，
        # 否则 FFN 的查表单元会大量误触发 —— 见 model.py 的标定一节。
        "code_sparsity": 6,
        # 标记词在 marker 段里的幅度。调大是为了让 (词形, 标记) 合取单元的
        # 两个信号量级可比，否则小量级那一项会被 gelu 阈值吃掉。
        "marker_scale": 6.0,
        # 输入噪声：加到 embedding 上的高斯扰动。**没有它，探针准确率会全是 1.000** ——
        # 一个全部满分的结果说明不了任何事（它只是把构造好的查找表读回来了）。
        # 噪声让「线性可分」变成一个有难度的判定，曲线才有信息量。
        "input_noise": 0.15,
        # 逐层位置偏置斜率（对称 ALiBi）。0 = 无位置偏好；越大越只看邻近位置。
        # 注意：**注意力的「局部性」主要不来自它** —— 词形码本身就会让
        # 自注意力的 logit 远高于其他位置（见 ``attn_code_scale``）。
        # 这个斜率只是把「邻域」和「远处」再拉开一点，让 L1/L2 的
        # 平均注意力距离明显小于 L3/L4（实测 0.08 / 0.39 / 2.9 / 2.9）。
        "alibi_slopes": [0.8, 0.3, 0.0, 0.0],
        # 逐层「主题标记词」吸引强度。只在第 3、4 层打开 —— 这是一个
        # register/sink 电路，它让标记词的信息被广播到所有位置。
        # **标定过**：3.5 时标记词权重达 0.987，注意力输出几乎全是标记词的状态，
        # 合取单元读不到位置自身的词形（24 个单元全部标定失败）。
        # 取 1.05 时自身权重 ≈ 0.337、标记词权重 ≈ 0.337，两项信号可比。
        "sink_gains": [0.0, 0.0, 1.05, 1.05],
        # 词形码投影到 Q/K 的幅度。**它决定了自注意力有多锐利**（logit ∝ 它的平方）。
        #
        # 这是第二个「标定出来的」旋钮：FFN 合取单元要能干净点火，前提是
        # 它读到的各项**幅度稳定** —— 而 FFN 读的是注意力输出 ``a_vec``，
        # 其中词形项 ≈ α_self × code(self)。α_self 一抖，词形项就抖，
        # 「词形 + 标记」这个 AND 门的阈值就压不住负样本。
        # 实测（217 词形 / 32 维码）：
        #   2.75 → α_self = 0.70/0.77/0.50/0.49，L2 干净率 184/228、L3 只有 4/21，
        #          role_amb 从 0.55 只涨到 0.72（跳变 0.17，几乎看不出来）；
        #   5.0  → α_self = 0.92/0.96/0.72/0.73，L3 干净率 15/21，role_amb 涨到 0.90；
        #   9.0  → α_self = 0.99/0.96/0.74/0.75，L3 干净率 18/21，role_amb 涨到 0.93。
        # 取 9.0：**浅层几乎是纯自注意力、深层仍留 25% 的全局混合** ——
        # 「注意力范围随深度扩大」这条结论正是靠这个差值观察的。
        "attn_code_scale": 9.0,
        # 合取单元里「词形项」与「标记项」的相对权重：1.0 = 两项贡献相等。
        # 由 ``calibrate()`` 自动换算成实际系数（见 model.py 的 balance 机制），
        # 这里只是「希望两者等权」这个意图，不是魔法数。
        "conj_gain": 1.0,
        # FFN 写入幅度（构造式标本用）
        "ffn_gain": 1.0,
        # 随机对照模型的权重尺度
        "random_scale": 0.30,
    },
    "observation": {
        "poolings": ["token", "sentence"],
        "similarities": ["cosine", "centered_cosine", "cka"],
        "splits": ["by_sentence", "by_token"],
        "projects": ["none", "pca"],
        # 抓哪些层。0 = embedding（未过任何 block），1..n_layers = 各 block 之后。
        "layers": [0, 1, 2, 3, 4],
        # 测试集比例。**按句子划分**（by_sentence）才是无泄漏的口径。
        "test_fraction": 0.35,
        "pca_components": 2,
        "power_iters": 60,
        # 散点图最多画多少个点（产物要小到能进 git）
        "max_scatter_points": 160,
        # 完整注意力矩阵留几句（其余只留画像统计）
        "attention_samples": 4,
        # role 属性要分「全体 / 歧义词 / 非歧义词」三条曲线 ——
        # 这三条曲线的分离正是本项目最核心的那个发现。
        "role_subsets": ["all", "ambiguous", "unambiguous"],
    },
    "probe": {
        # 闭式岭回归的正则强度。不设种子、不迭代 —— 探针本身必须完全确定。
        "ridge_lambda": 1.0,
        # 某一类样本数少于这个值时，该属性在这个 spec 上标记为「不适用」
        "min_class_count": 3,
    },
    "ablation": {
        "reference": "constructive-tok-cos-bs-raw",
    },
    "select": {
        "rule": "max_deep_gain",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """递归合并：``override`` 里出现的键覆盖 ``base``，嵌套字典逐层合并。"""
    out = dict(base)
    if not override:
        return out
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_path(root: str | Path, rel: str | Path) -> Path:
    """把配置里的相对路径解析成绝对路径。

    **签名是 ``(root, rel)`` 而不是 ``(cfg, rel)``** —— stage 拿到的往往是
    子配置（``cfg["corpus"]``），里面没有 ``_root``，按 cfg 取会 ``KeyError``。
    """
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def load_config(path: str | Path) -> dict[str, Any]:
    """读配置并与 ``DEFAULTS`` 合并；把项目根写进 ``cfg["_root"]``。"""
    p = Path(path).resolve()
    raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    cfg = deep_merge(DEFAULTS, raw)
    cfg["_root"] = str(p.parent.parent)
    cfg["_config_path"] = str(p)
    validate(cfg)
    return cfg


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读 JSONL。空行忽略；文件不存在返回空列表（由调用方决定是否报错）。"""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_data_path(cfg: dict[str, Any], key: str) -> Path:
    return resolve_path(cfg["_root"], cfg["corpus"][key])


# ----------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------
def _validate_corpus(cfg: dict[str, Any]) -> None:
    corpus = cfg["corpus"]
    for key in ("lexicon", "sentences", "name"):
        if key not in corpus:
            raise ValueError(f"corpus 缺少字段 {key!r}")
    # 只要求词典存在。**句子不要求** —— 它由词典 + 模板 + 种子确定性生成，
    # 文件缺失时会自动生成（``load_corpus`` 里按 lexical digest 判断是否过期）。
    # 把句子也当成必需文件会让人以为它是手写的，从而不敢改词典。
    if not resolve_data_path(cfg, "lexicon").exists():
        raise FileNotFoundError(
            f"corpus.lexicon 指向的文件不存在：{resolve_data_path(cfg, 'lexicon')}"
        )


def _validate_model(cfg: dict[str, Any]) -> None:
    m = cfg["model"]
    subs = m["subspaces"]
    total = sum(int(v) for v in subs.values())
    if total != int(m["d_model"]):
        raise ValueError(
            f"subspaces 各段之和 {total} 必须等于 d_model {m['d_model']}"
        )
    for name in ("lexical", "syntax", "semantic", "marker"):
        if name not in subs:
            raise ValueError(f"subspaces 缺少段 {name!r}")
    code_dim = int(m.get("code_dim", subs["lexical"]))
    if code_dim > int(subs["lexical"]):
        raise ValueError(
            f"code_dim {code_dim} 不能大于 lexical 段 {subs['lexical']}"
        )
    if int(m["code_sparsity"]) > code_dim:
        raise ValueError(
            f"code_sparsity {m['code_sparsity']} 不能大于 code_dim {code_dim}"
        )
    if int(m["d_model"]) % int(m["n_heads"]) != 0:
        raise ValueError("d_model 必须能被 n_heads 整除")
    for key in ("alibi_slopes", "sink_gains"):
        if len(m[key]) != int(m["n_layers"]):
            raise ValueError(
                f"model.{key} 的长度 {len(m[key])} 必须等于 n_layers {m['n_layers']}"
            )
    if not m["models"]:
        raise ValueError("model.models 不能为空")
    for name in m["models"]:
        if name not in MODELS:
            raise ValueError(
                f"未知的模型 {name!r}：只支持 {sorted(MODELS)} "
                "（constructive = 构造式标本，random = 随机权重对照）"
            )


def _validate_observation(cfg: dict[str, Any]) -> None:
    obs = cfg["observation"]
    allowed = {
        "poolings": {"token", "sentence"},
        "similarities": {"cosine", "centered_cosine", "cka"},
        "splits": {"by_sentence", "by_token"},
        "projects": {"none", "pca"},
    }
    for key, valid in allowed.items():
        values = obs.get(key)
        if not values:
            raise ValueError(f"observation.{key} 不能为空（否则矩阵会静默少一整列）")
        for value in values:
            if value not in valid:
                raise ValueError(
                    f"observation.{key} 里的 {value!r} 不是合法取值，合法值："
                    f"{sorted(valid)}"
                )
    layers = [int(x) for x in obs["layers"]]
    if not layers:
        raise ValueError(
            "observation.layers 不能为空 —— 一条可观测的层都没有，曲线与矩阵会静默退化成空表"
        )
    if sorted(set(layers)) != layers:
        raise ValueError(f"observation.layers 必须是严格递增且无重复：{layers}")
    if layers[0] != 0:
        raise ValueError("observation.layers 必须以 0 开头（0 = embedding 层）")
    if layers[-1] > int(cfg["model"]["n_layers"]):
        raise ValueError(
            f"observation.layers 里的 {layers[-1]} 超过 n_layers "
            f"{cfg['model']['n_layers']}"
        )
    frac = float(obs["test_fraction"])
    if not 0.0 < frac < 1.0:
        raise ValueError(f"test_fraction 必须在 (0,1) 内：{frac}")
    if int(obs["pca_components"]) < 1:
        raise ValueError("pca_components 至少为 1")
    subsets = obs.get("role_subsets")
    if not subsets:
        raise ValueError("observation.role_subsets 不能为空")
    for s in subsets:
        if s not in ("all", "ambiguous", "unambiguous"):
            raise ValueError(f"未知的 role subset {s!r}")


def _validate_probe(cfg: dict[str, Any]) -> None:
    probe = cfg["probe"]
    if float(probe["ridge_lambda"]) <= 0:
        # λ=0 时法方程可能奇异（d 维特征、类别数多于样本时必然奇异）。
        # 允许 0 会让「探针跑不出结果」表现为数值异常而不是配置错误。
        raise ValueError("probe.ridge_lambda 必须为正（λ=0 会让法方程奇异）")
    if int(probe["min_class_count"]) < 1:
        raise ValueError("probe.min_class_count 至少为 1")


def _validate_ablation(cfg: dict[str, Any]) -> None:
    ref = cfg["ablation"]["reference"]
    ids = {s.spec_id for s in expand_specs(cfg)}
    if ref not in ids:
        raise ValueError(
            f"ablation.reference {ref!r} 不在观测配置集合里。"
            f"合法示例：{sorted(ids)[:3]} …"
        )


def expand_specs(cfg: dict[str, Any]) -> list[StateSpec]:
    """把五个观测轴交叉展开成全部观测配置。

    展开顺序 = :data:`OBSERVATION_AXES` 的顺序 = 矩阵列顺序 = 消融分组顺序。
    **不做任何去重或剪枝** —— 每个轴上的每个取值都必须出现在结果里，
    否则矩阵会静默缺一整列（写一条测试钉住基数）。
    """
    obs = cfg["observation"]
    axes = {
        "model": list(cfg["model"]["models"]),
        "pooling": list(obs["poolings"]),
        "similarity": list(obs["similarities"]),
        "split": list(obs["splits"]),
        "project": list(obs["projects"]),
    }
    specs: list[StateSpec] = []
    for combo in itertools.product(*(axes[a] for a in OBSERVATION_AXES)):
        specs.append(StateSpec(**dict(zip(OBSERVATION_AXES, combo))))
    return specs


def validate(cfg: dict[str, Any]) -> None:
    """全部校验。任何一条不过就抛 —— 不静默降级。"""
    if not cfg.get("_root"):
        raise ValueError("配置缺少 _root（应通过 load_config 加载）")
    # 判「缺失」要用 is None，不能用真值：seed=0 是一个完全合法的种子，
    # 而 0 是 falsy —— 真值判断会把它当成「没填」直接拒绝。
    if cfg.get("seed") is None:
        raise ValueError("seed 不能为空 —— 本项目的全部结论都依赖可复现")
    if not PROPERTIES:
        raise ValueError("PROPERTIES 不能为空")
    _validate_corpus(cfg)
    _validate_model(cfg)
    _validate_observation(cfg)
    _validate_probe(cfg)
    _validate_ablation(cfg)
