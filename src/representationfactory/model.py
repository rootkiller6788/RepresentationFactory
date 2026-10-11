"""被观测的标本：一个微型 transformer（纯 Python，手写前向）。

**这是本项目的核心，也是最需要解释的一个文件。**

为什么要「构造」而不是「训练」
------------------------------
本项目的验收标准是「对同一输入给出逐层表征演化视图，**可指认**语义 / 语法分化的具体层」。
「可指认」这三个字要求我们知道**正确答案**。一个训练出来的模型给不出这个答案 ——
它只会给我们一些曲线，然后我们围着曲线编故事。

所以这里的做法是：**把答案先设计好，再让模型去实现它。**
每一层写哪个子空间、哪个 FFN 单元负责哪张查找表，都是写死在代码里的。
于是「探针在第几层读到角色」这个问题有了唯一的正确答案，
而 Lab 的验收就变成了一个**可核对的等式**：探针复原的分化层 == 设计的分化层。

这不是作弊，这是**标定**：一台仪器要能被验收，就得先对准一个结构已知的标本。
工程上完全一样 —— 你用已知频率的信号源校准频谱仪，不会拿一台未知的机器去校。

构造出来的电路（**这就是「真值」，报告里逐条核对**）
--------------------------------------------------
残差流按子空间分段（``d_model = 40``）：

    [0:24)  lexical   词形码（稀疏 ±1 随机码）
    [24:32) syntax    句法角色 one-hot
    [32:36) semantic  语义主题 one-hot
    [36:40) marker    主题标记词 one-hot

| 层 | 注意力范围 | 写入子空间 | FFN 干的活 | 于是哪个属性在这一层变得线性可分 |
| --- | --- | --- | --- | --- |
| 0 | —（embedding） | — | — | **form**（词形码本来就在里面） |
| 1 | 局部（±1） | lexical | 无（只做局部混合） | — |
| 2 | 局部（±2） | syntax | 通用词 `form → role` 查表 | **role**（非歧义部分） |
| 3 | **全局** + 标记词 sink | syntax | 歧义词 `(form, marker) → role` 合取 | **role（歧义部分）** |
| 4 | **全局** + 标记词 sink | semantic | `marker → topic` 查表并广播 | **topic** |

三个机制各自解释一条曲线形状：

1. **词形码是稀疏 ±1 随机码，不是 one-hot。** 于是「词形 → 角色」这个映射
   **线性不可分** —— 探针在第 0/1 层读不出角色，只有第 2 层把它写成 one-hot 之后才读得到。
   这解释了 role 曲线为什么在第 2 层跳。
2. **第 3、4 层的注意力有一个「标记词 sink」**（一个 register 电路）：
   所有位置都强烈注意主题标记词，于是标记词的信息被**广播到全句**。
   第 2 层的注意力只有 ±2，看不见远处的标记词 —— 这解释了
   ``role_amb`` 曲线为什么在第 3 层才跳。
3. **主题只在第 4 层被写成 one-hot。** 这解释了 topic 曲线为什么在第 4 层跳。

**没有 LayerNorm。** 真实模型需要它，因为权重是学出来的、尺度不可控；
本项目的每一段子空间幅度都是设计好的（词形码自点积 = 6、标记词 = 6、
角色 one-hot = 1），加归一化反而会把精心设定的相对尺度抹平，
让 FFN 的合取单元阈值变得不可预测。

阈值**自动标定**，不手拍
--------------------------
每个 FFN 单元要回答「这个位置的输入是不是我要的那张表的一行」。
手拍阈值会在改一个常量（比如词形码稀疏度）之后静默失效 ——
表现为「曲线整体塌掉」，而没人知道是阈值错了。
所以 ``calibrate()`` 用一次前向把每个单元的正样本与负样本激活都量出来，
把阈值定在两者中间。**并且逐层标定**：第 l 层的标定依赖第 1..l−1 层已经定好。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .corpus import Corpus, Lexicon
from .rng import make_rng

#: 数值下限。softmax / 除法里用它挡 0，不用 ``1e-12`` 这种魔法数散落各处。
EPS = 1e-12

#: 自动标定失败（正负样本激活区间重叠）时的兜底间距。
#: 重叠说明构造本身有问题，我们会把这件事记进 ``model.warnings`` 而不是静默继续。
_MIN_MARGIN = 1e-3


# ----------------------------------------------------------------------
# 数据视图
# ----------------------------------------------------------------------
@dataclass
class SpecimenData:
    """标本输入：词形 id 序列 + 每位置真值。**顺序即观测顺序**。"""

    #: 每句的词形 id 列表
    sequences: list[list[int]] = field(default_factory=list)
    sentence_ids: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    marker_index: list[int] = field(default_factory=list)
    # -- 展平后的逐位置真值（长度 = n_tokens）--
    sentence_of: list[int] = field(default_factory=list)
    index_in_sentence: list[int] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    position_topics: list[str] = field(default_factory=list)
    is_ambiguous: list[bool] = field(default_factory=list)
    is_marker: list[bool] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.forms)

    @property
    def n_sentences(self) -> int:
        return len(self.sequences)

    def position_ids(self) -> list[str]:
        return [
            f"{self.sentence_ids[self.sentence_of[i]]}#{self.index_in_sentence[i]}"
            for i in range(self.n_tokens)
        ]


def specimen_data(corpus: Corpus) -> SpecimenData:
    """把 :class:`~.corpus.Corpus` 转成模型需要的扁平视图。"""
    lex = corpus.lexicon
    data = SpecimenData()
    for s in sorted(corpus.sentences, key=lambda x: x.sentence_id):
        si = len(data.sequences)
        data.sequences.append([lex.form_ids[t.form] for t in s.tokens])
        data.sentence_ids.append(s.sentence_id)
        data.topics.append(s.topic)
        data.marker_index.append(s.marker_index)
        for t in s.tokens:
            data.sentence_of.append(si)
            data.index_in_sentence.append(t.index)
            data.forms.append(t.form)
            data.roles.append(t.role)
            data.position_topics.append(t.topic)
            data.is_ambiguous.append(t.is_ambiguous)
            data.is_marker.append(t.is_marker)
    return data


# ----------------------------------------------------------------------
# 权重
# ----------------------------------------------------------------------
@dataclass
class FFNUnit:
    """一个构造出来的 FFN 单元。

    ``expected`` 是「这个单元**应该**在哪些位置上点火」的位置下标集合 ——
    它由真值算出，是自动标定的依据。``threshold`` 就是我们要定的那个数。
    """

    kind: str
    w1: list[float]
    w2: list[float]
    expected: set[int]
    threshold: float = 0.0
    #: 标定后记下的正/负样本激活区间（报告里要用它说明标定是否干净）
    pos_range: tuple[float, float] = (0.0, 0.0)
    neg_range: tuple[float, float] = (0.0, 0.0)
    #: ``(主项所在子空间, 次项所在子空间, 次项的下标)``。
    #: 合取单元要同时匹配「词形」与「标记词」，而这两项在注意力输出里的量级
    #: 相差好几倍（词形码自点积 6 vs 标记词 1.8）—— 不等权的话小项会被
    #: gelu 阈值吃掉，表现为「只看词形也能点火」，分化层就没意义了。
    #: 所以 ``calibrate()`` 会按正样本上两项的实测均值把次项的系数算出来。
    balance: tuple[str, str, int] | None = None
    #: 平衡后记下的实测两项均值（报告要用它说明「两项确实等权了」）
    term_ranges: tuple[float, float] = (0.0, 0.0)

    @property
    def clean(self) -> bool:
        """正样本的最小激活是否严格大于负样本的最大激活。"""
        return self.pos_range[0] > self.neg_range[1]

    @property
    def balanced(self) -> bool:
        """两个信号项的量级是否可比（相差不超过 2 倍）。"""
        a, b = self.term_ranges
        if a <= 0.0 or b <= 0.0:
            return True
        return 0.5 <= a / b <= 2.0


@dataclass
class LayerWeights:
    """一层的全部权重。

    ``Wq/Wk/Wv/Wo`` 都是**逐头**的：``Wq[h]`` 是 ``d_head × d_model``。
    头之间靠「各占残差流的一段」来分工（``Wv[h]`` 选出第 h 段、
    ``Wo[h]`` 写回第 h 段），所以 ``Σ_h Wo[h]·(V[h]·x) = x`` ——
    注意力输出默认是「上下文加权平均」，而不是把信息压到某个低维子空间里。
    """

    Wq: list[list[list[float]]]
    Wk: list[list[list[float]]]
    Wv: list[list[list[float]]]
    Wo: list[list[list[float]]]
    bq: list[list[float]]
    bk: list[list[float]]
    alibi: float
    sink: float
    target_span: tuple[int, int]
    units: list[FFNUnit] = field(default_factory=list)
    #: 自动标定出来的阈值（units 的顺序）
    thresholds: list[float] = field(default_factory=list)

    def W1(self) -> list[list[float]]:
        return [list(u.w1) for u in self.units]

    def W2(self, d_model: int) -> list[list[float]]:
        """``d_model × n_units``。"""
        cols = [u.w2 for u in self.units]
        return [[cols[j][i] for j in range(len(cols))] for i in range(d_model)]


@dataclass
class Model:
    """被观测的模型。"""

    name: str
    cfg: dict[str, Any]
    lexicon: Lexicon
    #: n_forms × d_model
    E: list[list[float]]
    layers: list[LayerWeights]
    subspaces: dict[str, tuple[int, int]]
    d_model: int
    n_heads: int
    d_head: int
    warnings: list[str] = field(default_factory=list)

    def span(self, name: str) -> tuple[int, int]:
        return self.subspaces[name]

    def subspace_of(self, vec: list[float], name: str) -> list[float]:
        a, b = self.subspaces[name]
        return vec[a:b]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "d_model": self.d_model,
            "n_layers": len(self.layers),
            "n_heads": self.n_heads,
            "d_head": self.d_head,
            "n_forms": len(self.E),
            "subspaces": {k: list(v) for k, v in sorted(self.subspaces.items())},
            "alibi_slopes": [round(l.alibi, 4) for l in self.layers],
            "sink_gains": [round(l.sink, 4) for l in self.layers],
            "n_ffn_units": [len(l.units) for l in self.layers],
            "unit_kinds": sorted({u.kind for l in self.layers for u in l.units}),
            "warnings": list(self.warnings),
        }


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def softmax(logits: list[float]) -> list[float]:
    """数值稳定版 softmax。"""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    if total <= EPS:
        n = len(logits)
        return [1.0 / n] * n
    return [x / total for x in exps]


def gelu(x: float) -> float:
    """tanh 近似的 GELU（与主流实现一致）。

    **选它而不是 ReLU**，是因为它有一个「软阈值」的形状：
    ``gelu(x) ≈ 0`` for ``x < 0``，在 ``x > 0`` 附近平滑上升。
    构造式合取单元正好需要这个性质 —— 阈值以下不点火、以上按比例输出。
    ReLU 也能用，但它的折点会让「阈值稍微偏一点就完全没输出」，
    标定的容错小得多。
    """
    return 0.5 * x * (1.0 + math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _dot(u: list[float], v: list[float]) -> float:
    if hasattr(math, "sumprod"):
        return math.sumprod(u, v)
    return sum(a * b for a, b in zip(u, v))


def _matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [_dot(row, v) for row in m]


def _unrank_combination(rank: int, n: int, k: int) -> tuple[int, ...]:
    """把 ``rank`` 解成一个长度为 ``k`` 的递增下标组（字典序 unranking）。

    有了它就不用把「全部组合」真的列出来了 —— 见 ``assign_codes``。
    """
    out: list[int] = []
    start = 0
    for i in range(k):
        for c in range(start, n):
            cnt = math.comb(n - c - 1, k - i - 1)
            if rank < cnt:
                out.append(c)
                start = c + 1
                break
            rank -= cnt
    return tuple(out)


def assign_codes(
    forms: list[str], seed: int, code_dim: int, sparsity: int
) -> dict[str, list[float]]:
    """给每个词形分配一个**互不相同**的稀疏 ±1 码。

    两个刻意的选择：

    1. **稀疏**：自点积 = sparsity，互点积 ≈ 0（均值 0，标准差 ~1）。
       满 ±1 码时自点积 = code_dim、互点积标准差 = √code_dim，比值只有
       √code_dim；稀疏码把比值拉到 sparsity/1，于是 FFN 查表单元的
       误触发率从「每位置好几个」降到「整份语料几个」。
    2. **互不相同**：从「全部 (support, signs) 组合」里按种子抽取**不重复**的
       N 个，而不是每个词形独立随机抽。独立抽取会撞码（24 个词形、160 个
       可用码时撞码概率约 45%），撞了的两个词形**永远无法被探针区分** ——
       表现为 form 曲线整体掉一截，而原因根本看不出来。

    不建池子，只抽下标再解码
    ------------------------
    最早的实现是 ``itertools`` 把全部组合物化成列表再 ``shuffle``。
    code_dim=8 / sparsity=3 时池子只有 448 个向量，看不出问题；
    等到把词表从 24 扩到 96、code_dim 提到 32、sparsity 提到 6 时，
    池子变成 C(32,6)×2⁶ ≈ **5800 万个** 32 维向量 —— 直接 MemoryError。
    而它本来只需要 ``len(forms)`` 个：用「组合数 unranking」把随机下标
    直接解成 (support, signs)，内存就从 O(池子) 降到 O(词表)。
    """
    n_signs = 2**sparsity
    total = math.comb(code_dim, sparsity) * n_signs
    if total < len(forms):
        raise ValueError(
            f"code_dim={code_dim} / sparsity={sparsity} 只有 {total} 个可用码，"
            f"不够 {len(forms)} 个词形。调大 code_dim 或 sparsity。"
        )
    rng = make_rng("codes", seed, code_dim, sparsity)
    # 抽不重复的下标。total 远大于 len(forms) 时冲突概率可以忽略，
    # 但仍用集合兜住 —— 撞码的后果是「两个词形永远无法被区分」。
    picked: set[int] = set()
    while len(picked) < len(forms):
        picked.add(rng.randrange(total))
    out: dict[str, list[float]] = {}
    for form, idx in zip(sorted(forms), sorted(picked)):
        comb_rank, sign_bits = divmod(idx, n_signs)
        support = _unrank_combination(comb_rank, code_dim, sparsity)
        vec = [0.0] * code_dim
        for bit, d in enumerate(support):
            vec[d] = -1.0 if (sign_bits >> bit) & 1 else 1.0
        out[form] = vec
    return out


def _glorot(rng: Any, out_dim: int, in_dim: int, scale: float) -> list[list[float]]:
    """随机权重（对照模型用）。``scale`` 是整体幅度旋钮。"""
    limit = scale / math.sqrt(max(in_dim, 1))
    return [
        [rng.uniform(-limit, limit) for _ in range(in_dim)] for _ in range(out_dim)
    ]


# ----------------------------------------------------------------------
# 构造
# ----------------------------------------------------------------------
def _subspaces(cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    subs = cfg["model"]["subspaces"]
    out: dict[str, tuple[int, int]] = {}
    cur = 0
    for name in ("lexical", "syntax", "semantic", "marker"):
        w = int(subs[name])
        out[name] = (cur, cur + w)
        cur += w
    return out


def _select_span(span: tuple[int, int], d_model: int) -> list[list[float]]:
    """选出 ``span`` 那一段的单位矩阵（``(b−a) × d_model``）。"""
    a, b = span
    return [[1.0 if j == i else 0.0 for j in range(d_model)] for i in range(a, b)]


def _scatter_span(span: tuple[int, int], d_model: int) -> list[list[float]]:
    """把 ``b−a`` 维写回 ``span`` 那一段（``d_model × (b−a)``）。"""
    a, b = span
    return [[1.0 if i == a + j else 0.0 for j in range(b - a)] for i in range(d_model)]


def _mask_span(vec: list[float], span: tuple[int, int]) -> list[float]:
    a, b = span
    return [v if a <= i < b else 0.0 for i, v in enumerate(vec)]


def _build_attention(
    cfg: dict[str, Any],
    seed: int,
    layer_index: int,
    d_model: int,
    n_heads: int,
    subspaces: dict[str, tuple[int, int]],
    rng: Any,
    constructive: bool,
) -> tuple[list, list, list, list, list, list, float, float]:
    """构造一层的注意力权重。

    构造式：
    - ``Wv[h]`` / ``Wo[h]`` = 第 h 段的「取出 / 写回」—— 于是注意力输出的默认语义是
      「上下文的加权平均」，这是本项目全部「广播」机制的基础；
    - ``Wq[h]`` / ``Wk[h]`` = 把 **lexical 段**随机投到 d_head 维。
      两侧同投影 ⇒ ``<q_i, k_j> = c_iᵀ M c_j``（M 半正定）⇒ 自身位置天然高、
      其他位置接近 0 —— **自注意力不是加出来的偏置，而是词形码结构的自然结果**；
    - 第 3、4 层额外给 head 0 一个 **sink**：query 加一个常数偏置（``bq[0]=1``），
      key 把 marker 段投到同一维（幅度 = sink_gain）。于是所有位置都强烈注意标记词。
    """
    lex_span = subspaces["lexical"]
    mk_span = subspaces["marker"]
    d_head = d_model // n_heads
    m = cfg["model"]
    # **sink 必须放在「其 V 覆盖 marker 子空间」的那个头上。**
    # 每个头的 V 只搬运残差流里自己那一段（这是本项目多头分工的定义），
    # 所以放在别的头上，被广播出去的是标记词的**词形码**而不是它的
    # **身份子空间** —— 于是「标记词是谁」这件事根本没有传到全句，
    # 而 topic 单元读的正是一段全零。放在 head 0 时实测就是这个结果
    # （marker 段幅度恒为 0，topic 单元全部标定失败）。
    sink_head = mk_span[0] // d_head

    Wq, Wk, Wv, Wo, bq, bk = [], [], [], [], [], []
    for h in range(n_heads):
        span = (h * d_head, (h + 1) * d_head)
        # V/O：取出与写回第 h 段
        Wv.append(_select_span(span, d_model))
        Wo.append(_scatter_span(span, d_model))

        if constructive:
            # 词形码 → d_head 的随机投影（同一投影给 Q 与 K）
            proj = _glorot(
                rng, d_head, d_model, scale=float(m.get("attn_code_scale", 1.0))
            )
            # 只让 lexical 段参与：其余列清零
            for row in proj:
                for j in range(d_model):
                    if not (lex_span[0] <= j < lex_span[1]):
                        row[j] = 0.0
            q, k = [list(r) for r in proj], [list(r) for r in proj]
            bqv = [0.0] * d_head
            bkv = [0.0] * d_head
            if h == sink_head and float(m["sink_gains"][layer_index]) > 0.0:
                # sink 头：**整头只做一件事**——把标记词广播给所有位置。
                # Q/K 的其余行全部清零，只留「常数 query（纯偏置）× 只认标记词的 key」。
                #
                # 为什么必须这么干净？这里踩过三次，三次都不报错：
                # 1. 整块清零（含 lexical）→ 这头完全没自注意力，α_marker 达 0.99，
                #    注意力输出几乎全是标记词的状态，合取单元读不到位置自身词形；
                # 2. 只清 lexical 列 → q[0] 仍随机投影到 syntax / semantic 上，
                #    而深层这些子空间幅度很大，q[0] 在 ±几十乱跳，logit 尺度被搅乱；
                # 3. 不清内容项 → 内容项本身有 ±1.2 的 logit 标准差（词形码之间的
                #    随机交叉项），与 sink 的 1.99 同量级，于是 α_marker 在
                #    0 ~ 0.97 之间乱跳 —— 表现为 topic 单元的正样本激活出现负值。
                # 清零之后：logit_ij = sink·marker_scale/√d_head 当且仅当 j 是标记词，
                # 与 i 无关 ⇒ **α_marker 对所有位置都是一个常数**（实测 0.477）。
                # 「广播强度不随位置变化」正是本项目想观察的那个机制。
                for row in q:
                    for j in range(d_model):
                        row[j] = 0.0
                for row in k:
                    for j in range(d_model):
                        row[j] = 0.0
                bqv[0] = 1.0
                gain = float(m["sink_gains"][layer_index])
                for j in range(mk_span[0], mk_span[1]):
                    k[0][j] = gain
            Wq.append(q)
            Wk.append(k)
            bq.append(bqv)
            bk.append(bkv)
        else:
            Wq.append(_glorot(rng, d_head, d_model, float(m["random_scale"])))
            Wk.append(_glorot(rng, d_head, d_model, float(m["random_scale"])))
            bq.append([0.0] * d_head)
            bk.append([0.0] * d_head)

    alibi = float(m["alibi_slopes"][layer_index]) if constructive else 0.0
    sink = float(m["sink_gains"][layer_index]) if constructive else 0.0
    return Wq, Wk, Wv, Wo, bq, bk, alibi, sink


def _ffn_units(
    cfg: dict[str, Any],
    lexicon: Lexicon,
    data: SpecimenData,
    codes: dict[str, list[float]],
    layer_index: int,
    subspaces: dict[str, tuple[int, int]],
    d_model: int,
) -> list[FFNUnit]:
    """构造第 ``layer_index`` 层（1 起）的 FFN 单元。

    单元的种类与 ``expected``（应该点火的位置）：
    - ``generic``     : 层 2，``form → role``。expected = 该词形的全部位置。
    - ``conjunction`` : 层 3，``(form, marker) → role``。expected = 该词形**且**该主题的位置。
    - ``topic``       : 层 4，``marker → topic``。expected = 该主题的**全部**位置
      （不是标记词本身！主题要广播到全句 —— 这一点写错的话，
      曲线会变成「只有标记词位置可读」，而报告里的 topic 曲线会整体塌到 chance）。
    """
    m = cfg["model"]
    gain = float(m["ffn_gain"])
    lex_span = subspaces["lexical"]
    syn_span = subspaces["syntax"]
    sem_span = subspaces["semantic"]
    mk_span = subspaces["marker"]
    roles = list(lexicon.roles)
    topics = list(lexicon.topics)
    units: list[FFNUnit] = []

    def put_code(target: list[float], code: list[float]) -> None:
        """把词形码写进 lexical 段的前 ``len(code)`` 维。

        码只占 ``code_dim`` 维（不是整个 lexical 段），所以这里不能写成
        ``target[a:b] = code`` —— 维度对不上会直接 IndexError；
        而写成「先切片再赋值」又会在 code_dim 变化时静默错位。
        """
        a = lex_span[0]
        for i, v in enumerate(code):
            target[a + i] = v

    def unit(
        kind: str,
        w1: list[float],
        w2: list[float],
        expected: set[int],
        balance: tuple[str, str, int] | None = None,
    ) -> None:
        units.append(
            FFNUnit(kind=kind, w1=w1, w2=w2, expected=expected, balance=balance)
        )

    if layer_index == 2:
        for form, role in sorted(lexicon.generic.items()):
            if lexicon.is_marker(form):
                continue
            w1 = [0.0] * d_model
            put_code(w1, codes[form])
            w2 = [0.0] * d_model
            w2[syn_span[0] + roles.index(role)] = gain
            expected = {i for i, f in enumerate(data.forms) if f == form}
            if expected:
                unit("generic", w1, w2, expected)

    elif layer_index == 3:
        for form, mapping in sorted(lexicon.ambiguous.items()):
            for topic, role in sorted(mapping.items()):
                w1 = [0.0] * d_model
                put_code(w1, codes[form])
                mk_idx = mk_span[0] + topics.index(topic)
                # 系数先放 0，由 calibrate() 按实测把「标记项」配到与「词形项」等权
                w1[mk_idx] = 0.0
                w2 = [0.0] * d_model
                w2[syn_span[0] + roles.index(role)] = gain
                expected = {
                    i
                    for i, (f, t) in enumerate(zip(data.forms, data.position_topics))
                    if f == form and t == topic
                }
                if expected:
                    unit(
                        "conjunction", w1, w2, expected,
                        balance=("lexical", "marker", mk_idx),
                    )

    elif layer_index == 4:
        for topic in topics:
            w1 = [0.0] * d_model
            w1[mk_span[0] + topics.index(topic)] = float(m["marker_scale"])
            w2 = [0.0] * d_model
            w2[sem_span[0] + topics.index(topic)] = gain
            expected = {
                i for i, t in enumerate(data.position_topics) if t == topic
            }
            if expected:
                unit("topic", w1, w2, expected)

    # 层 1 不建查表单元：它只做局部混合（写入 lexical 段由 masked write 负责）。
    return units


def build_model(
    name: str,
    cfg: dict[str, Any],
    lexicon: Lexicon,
    data: SpecimenData,
) -> Model:
    """建一个标本。``name`` ∈ {``constructive``, ``random``}。"""
    if name not in ("constructive", "random"):
        raise ValueError(f"未知模型 {name!r}")
    constructive = name == "constructive"
    seed = int(cfg["seed"])
    m = cfg["model"]
    d_model = int(m["d_model"])
    n_layers = int(m["n_layers"])
    n_heads = int(m["n_heads"])
    subspaces = _subspaces(cfg)

    forms = lexicon.forms
    code_dim = int(m.get("code_dim", subspaces["lexical"][1] - subspaces["lexical"][0]))
    codes = assign_codes(forms, seed, code_dim, int(m["code_sparsity"]))

    # ---- embedding ----
    E: list[list[float]] = []
    for form in forms:
        row = [0.0] * d_model
        if constructive:
            a, _ = subspaces["lexical"]
            # 码只占 lexical 段的前 code_dim 维 —— 见 config 里 code_dim 的说明：
            # 这是「角色在浅层读不出来」的机制所在。
            row[a : a + code_dim] = codes[form]
            if lexicon.is_marker(form):
                t = lexicon.topics.index(lexicon.markers[form])
                row[subspaces["marker"][0] + t] = float(m["marker_scale"])
        else:
            rng = make_rng("emb", seed, form)
            row = [
                rng.uniform(-1.0, 1.0) * float(m["random_scale"]) for _ in range(d_model)
            ]
        E.append(row)

    # ---- 逐层 ----
    # **只有第 1 层做 masked write**（把注意力输出的目标子空间搬进残差流）。
    # 第 2/3/4 层一律「只由 FFN 写」—— 这条是标定出来的，不是一开始就想到的：
    # 原设计让第 3 层把 syntax 子空间也 masked write 一遍，而第 3 层的注意力
    # 已经是全局的，于是它做的是「把全句所有位置的角色取平均」，
    # 把第 2 层刚写好的逐位置角色**抹平**了（syntax 段幅度 3.07 → 全是噪声）。
    # 现在这个分工反而更干净：**注意力决定 FFN 能看到什么，FFN 负责写答案，
    # 残差负责把答案留住**。这也正是本项目想观察到的那个机制。
    empty = (0, 0)
    targets = {
        1: subspaces["lexical"],
        2: empty,
        3: empty,
        4: empty,
    }
    layers: list[LayerWeights] = []
    for l in range(1, n_layers + 1):
        rng = make_rng("layer", seed, l)
        Wq, Wk, Wv, Wo, bq, bk, alibi, sink = _build_attention(
            cfg, seed, l - 1, d_model, n_heads, subspaces, rng, constructive
        )
        if constructive:
            units = _ffn_units(cfg, lexicon, data, codes, l, subspaces, d_model)
        else:
            d_ff = int(m["d_ff"])
            scale = float(m["random_scale"])
            W1r = _glorot(rng, d_ff, d_model, scale)
            W2r = _glorot(rng, d_model, d_ff, scale)
            units = [
                FFNUnit(
                    kind="random",
                    w1=W1r[i],
                    w2=[W2r[j][i] for j in range(d_model)],
                    expected=set(),
                    threshold=0.0,
                )
                for i in range(d_ff)
            ]
        layers.append(
            LayerWeights(
                Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, bq=bq, bk=bk,
                alibi=alibi, sink=sink, target_span=targets.get(l, (0, d_model)),
                units=units,
            )
        )

    model = Model(
        name=name, cfg=cfg, lexicon=lexicon, E=E, layers=layers,
        subspaces=subspaces, d_model=d_model, n_heads=n_heads,
        d_head=d_model // n_heads,
    )
    if constructive:
        calibrate(model, data, cfg)
    return model


# ----------------------------------------------------------------------
# 前向
# ----------------------------------------------------------------------
@dataclass
class ForwardTrace:
    """一次前向的全部中间结果。

    ``hidden[l]`` 是**展平后**的逐位置向量（长度 = n_tokens），
    ``attention[l][h]`` 是逐句的 n×n 矩阵（列表的列表，按句顺序）。
    """

    hidden: list[list[list[float]]] = field(default_factory=list)
    attention: list[list[list[list[list[float]]]]] = field(default_factory=list)
    #: 逐层逐位置的实际 FFN 输入（自动标定要用；默认不保留以省内存）
    ffn_inputs: list[list[list[float]]] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return max(0, len(self.hidden) - 1)


def _layer_forward(
    model: Model,
    h_prev: list[list[float]],
    data: SpecimenData,
    layer_index: int,
    want_attention: bool,
    want_ffn_input: bool,
) -> tuple[list[list[float]], list[list[list[float]]] | None, list[list[float]] | None]:
    """跑一层。``h_prev`` 是展平的逐位置向量。"""
    layer = model.layers[layer_index]
    d_model = model.d_model
    n_heads = model.n_heads
    d_head = model.d_head
    scale = 1.0 / math.sqrt(d_head)

    h_new = [list(v) for v in h_prev]
    attn_out: list[list[list[list[float]]]] | None = [] if want_attention else None
    ffn_in: list[list[float]] | None = [] if want_ffn_input else None

    # 句起点只算一次。原先在循环里 ``sum(len(s) for s in sequences[:si])``
    # 是 O(n_sentences²)，80 句时看不出来，但这个函数的调用次数是 O(层数×标定轮数)。
    offsets: list[int] = []
    acc_off = 0
    for seq in data.sequences:
        offsets.append(acc_off)
        acc_off += len(seq)

    for si, seq in enumerate(data.sequences):
        n = len(seq)
        if n == 0:
            if attn_out is not None:
                attn_out.append([[] for _ in range(n_heads)])
            continue
        start = offsets[si]
        hs = h_prev[start : start + n]

        # 逐头的 q / k（先算全，避免在内层重复矩阵乘）
        kw = [layer.Wk[h] for h in range(n_heads)]
        vw = [layer.Wv[h] for h in range(n_heads)]
        qs = [
            [
                [_dot(layer.Wq[h][r], hs[i]) + layer.bq[h][r] for r in range(d_head)]
                for i in range(n)
            ]
            for h in range(n_heads)
        ]
        ks = [
            [[_dot(kw[h][r], hs[j]) + layer.bk[h][r] for r in range(d_head)] for j in range(n)]
            for h in range(n_heads)
        ]
        vs = [[_matvec(vw[h], hs[j]) for j in range(n)] for h in range(n_heads)]

        head_out = [[0.0] * d_model for _ in range(n)]
        per_head_alpha: list[list[list[float]]] = []
        for h in range(n_heads):
            alpha_rows: list[list[float]] = []
            for i in range(n):
                qi = qs[h][i]
                logits = [
                    _dot(qi, ks[h][j]) * scale - layer.alibi * abs(i - j)
                    for j in range(n)
                ]
                alpha = softmax(logits)
                alpha_rows.append(alpha)
                acc = [0.0] * d_head
                for j in range(n):
                    a = alpha[j]
                    if a == 0.0:
                        continue
                    vj = vs[h][j]
                    for r in range(d_head):
                        acc[r] += a * vj[r]
                # 写回残差流的第 h 段（``Wo[h]`` 是 scatter 矩阵，等价于直接切片写回，
                # 走矩阵乘是为了让「构造式」与「随机式」走同一条代码路径）
                a0 = h * d_head
                out = head_out[i]
                for r in range(d_head):
                    out[a0 + r] += acc[r]
            per_head_alpha.append(alpha_rows)

        if attn_out is not None:
            attn_out.append(per_head_alpha)

        # masked write + FFN
        units = layer.units
        w1s = [u.w1 for u in units]
        thr = [u.threshold for u in units]
        for i in range(n):
            a_vec = head_out[i]
            delta = _mask_span(a_vec, layer.target_span)
            if want_ffn_input:
                ffn_in.append(list(a_vec))
            if units:
                # 只有点火的单元才参与累加：gelu 会输出 0，跳过它们能省掉大部分乘法
                for j, u in enumerate(units):
                    act = gelu(_dot(w1s[j], a_vec) - thr[j])
                    if act == 0.0:
                        continue
                    w2 = u.w2
                    for r in range(d_model):
                        if w2[r]:
                            delta[r] += act * w2[r]
            h_new[start + i] = [hv + dv for hv, dv in zip(h_prev[start + i], delta)]

    return h_new, attn_out, ffn_in


def forward(
    model: Model,
    data: SpecimenData,
    want_attention: bool = False,
    want_ffn_input: bool = False,
    stop_after: int | None = None,
) -> ForwardTrace:
    """跑一遍完整前向，返回逐层隐藏状态（含第 0 层 embedding）。

    ``stop_after=k`` 只跑前 k 层 —— ``calibrate()`` 标定第 l 层时只需要
    第 1..l 层的输出，跑满 4 层会白算 4−l 层（合计多算 3+2+1 = 6 层，
    占标定时光的 40%）。
    """
    n_tokens = data.n_tokens
    # 输入噪声：让「线性可分」变成一个有难度的判定（否则全是 1.000，什么也说明不了）
    noise = float(model.cfg["model"].get("input_noise", 0.0))
    h: list[list[float]] = []
    for i, form in enumerate(data.forms):
        base = list(model.E[model.lexicon.form_ids[form]])
        if noise > 0.0:
            rng = make_rng("noise", int(model.cfg["seed"]), data.sentence_of[i], data.index_in_sentence[i])
            base = [v + rng.gauss(0.0, noise) for v in base]
        h.append(base)

    trace = ForwardTrace(hidden=[h])
    for l in range(len(model.layers)):
        h, attn, ffn_in = _layer_forward(
            model, h, data, l, want_attention, want_ffn_input
        )
        trace.hidden.append(h)
        if want_attention and attn is not None:
            trace.attention.append(attn)
        if want_ffn_input and ffn_in is not None:
            trace.ffn_inputs.append(ffn_in)
        if stop_after is not None and l + 1 >= stop_after:
            break
    _ = n_tokens
    return trace


# ----------------------------------------------------------------------
# 自动标定
# ----------------------------------------------------------------------
def calibrate(model: Model, data: SpecimenData, cfg: dict[str, Any]) -> None:
    """逐层标定 FFN 阈值。

    **为什么必须逐层**：第 l 层的输入是第 1..l−1 层输出的函数。
    如果把所有层的阈值一次性定完，第 2 层用的是「第 1 层还没标定」时的激活，
    定出来的阈值在第 1 层标定完之后就不对了 —— 而曲线只会整体塌掉，
    没人知道是标定顺序错了。

    重叠（正样本最小激活 ≤ 负样本最大激活）说明构造本身有问题
    （比如词形码太密、marker 幅度太小）。这时**记进 warnings 并继续**，
    不抛错 —— 因为「探针读不出来」本身就是一条要观测的现象，
    我们应该把它如实报出来，而不是让流水线崩掉。
    """
    conj_gain = float(cfg["model"].get("conj_gain", 1.0))

    for l, layer in enumerate(model.layers):
        if not layer.units:
            continue
        # 跑到第 l 层 FFN 的输入为止（前 l 层已标定，第 l 层的阈值暂用 0）。
        # ``stop_after=l+1``：第 l 层（0 起）跑完就够了，后面的层还没标定、
        # 算了也没用 —— 白算的是 4−l−1 层。
        trace = forward(model, data, want_ffn_input=True, stop_after=l + 1)
        if l >= len(trace.ffn_inputs):
            break
        inputs = trace.ffn_inputs[l]

        # ---- 第一遍：把合取单元的「标记项」配到与「词形项」等权 ----
        # 这一遍不改前向（w1 的标记系数只影响 FFN 输出，不影响本层的注意力输入），
        # 所以可以就地量、就地改，不需要再跑一次前向。
        for u in layer.units:
            if not u.balance:
                continue
            _, _, idx = u.balance
            prim = [_dot(u.w1, inputs[i]) for i in u.expected]
            sec = [inputs[i][idx] for i in u.expected]
            p_mean = sum(prim) / len(prim) if prim else 0.0
            s_mean = sum(sec) / len(sec) if sec else 0.0
            u.term_ranges = (p_mean, s_mean)
            if abs(s_mean) < EPS:
                model.warnings.append(
                    f"层 {l + 1} 的 {u.kind} 单元配平失败：次项在正样本上的均值 "
                    f"{s_mean:.4f} 接近 0 —— 标记词的信号没被广播到全句"
                    "（检查 sink_gains 是否为 0）"
                )
                continue
            u.w1[idx] = conj_gain * (p_mean / s_mean)

        # ---- 第二遍：定阈值 ----
        for u in layer.units:
            pos: list[float] = []
            neg: list[float] = []
            for i in range(data.n_tokens):
                act = _dot(u.w1, inputs[i])
                (pos if i in u.expected else neg).append(act)
            if not pos:
                u.threshold = 0.0
                continue
            pmin, pmax = min(pos), max(pos)
            nmin, nmax = (min(neg), max(neg)) if neg else (-math.inf, -math.inf)
            u.pos_range = (pmin, pmax)
            u.neg_range = (nmin, nmax) if neg else (0.0, 0.0)
            if pmin > nmax:
                u.threshold = 0.5 * (pmin + nmax)
            else:
                # 重叠：取正样本分位数，宁可漏检也不误触发（漏检只让准确率低一点，
                # 误触发会往残差流里写错角色，污染的不止一个位置）
                pos_sorted = sorted(pos)
                u.threshold = pos_sorted[len(pos_sorted) // 2]
                model.warnings.append(
                    f"层 {l + 1} 的 {u.kind} 单元标定重叠："
                    f"正样本最小激活 {pmin:.4f} ≤ 负样本最大激活 {nmax:.4f}"
                )
        layer.thresholds = [u.threshold for u in layer.units]
