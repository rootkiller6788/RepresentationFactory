"""相似度度量（Measure Similarity）。

README 第六节的三类探针里，Similarity Probe 直接建立在这一层上。
本项目提供三种度量，**它们会给出不同的答案** —— 这本身就是结论：

| 度量 | 怎么算 | 对什么不敏感 |
| --- | --- | --- |
| ``cosine``          | 逐位置 ``cos(x_i, y_i)`` 取均值 | — |
| ``centered_cosine`` | 先按列去均值，再同上 | 共有的均值方向 |
| ``cka``             | 中心化线性 CKA | 共有的均值方向、**正交变换 / 各向异性缩放** |

为什么必须三种
--------------
**用 cosine 比较两层表征时，几乎所有层对都会给出 ~0.99。**
这不是 bug，是机制：残差流里所有位置共享一个很强的公共方向
（本项目的构造式标本里，词形码本身就带一个大的公共分量），
于是任意两层的余弦相似度都被那个公共方向主导 ——
层与层之间**真实的差异**被完全掩盖，而且**不报错**，
只会给出一张全是 0.99 的矩阵，看着还挺「稳定」。

实测（同一批状态、同一份语料）：

    cosine          L0–L1 = 0.9991   L0–L4 = 0.9921
    centered_cosine L0–L1 = 0.9812   L0–L4 = 0.4307
    cka             L0–L1 = 0.9985   L0–L4 = 0.1137

只有中心化之后的度量才看得出「层越远、表征越不一样」。
CKA 比 centered-cosine 更彻底（它还对正交变换不变），
所以本项目用它当默认，另外两个留着做对照。
"""

from __future__ import annotations

import math

from .linalg import center_columns, dot, frobenius2, gram, matmul, transpose, norm

#: 三种度量。它们的**区别**是本模块存在的理由，见模块文档。
SIMILARITY_METRICS: tuple[str, ...] = ("cosine", "centered_cosine", "cka")


def cosine(u: list[float], v: list[float]) -> float:
    nu, nv = norm(u), norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot(u, v) / (nu * nv)


def mean_cosine(x: list[list[float]], y: list[list[float]]) -> float:
    """逐位置余弦的均值。

    ``x`` 与 ``y`` 必须有相同的行数（同一批位置的两层表征）——
    这是「比较两层」这件事的定义：同一个位置，在第 a 层与第 b 层。
    """
    if len(x) != len(y):
        raise ValueError(f"行数不同，无法逐位置比较：{len(x)} vs {len(y)}")
    if not x:
        return 0.0
    return sum(cosine(a, b) for a, b in zip(x, y)) / len(x)


def centered_cosine(x: list[list[float]], y: list[list[float]]) -> float:
    """先按列去均值，再算逐位置余弦的均值。"""
    xc, _ = center_columns(x)
    yc, _ = center_columns(y)
    return mean_cosine(xc, yc)


def linear_cka(x: list[list[float]], y: list[list[float]]) -> float:
    """中心化线性 CKA（特征空间形式）。

    ``CKA = ||XᵀY||_F² / (||XᵀX||_F · ||YᵀY||_F)``，其中 X / Y 已按列中心化。

    用**特征空间**（d×d 的 Gram 矩阵）而不是样本空间（n×n）：
    本项目 n = 720、d = 40，前者便宜 18 倍，而且数值上更稳。
    两者在数学上等价（``||XᵀY||_F² = ||XYᵀ||_F²`` 的迹恒等式）。
    """
    if len(x) != len(y):
        raise ValueError(f"行数不同，无法比较：{len(x)} vs {len(y)}")
    if not x:
        return 0.0
    xc, _ = center_columns(x)
    yc, _ = center_columns(y)
    gxy = matmul(transpose(xc), yc)
    gxx = gram(xc)
    gyy = gram(yc)
    num = frobenius2(gxy)
    den = math.sqrt(frobenius2(gxx) * frobenius2(gyy))
    if den <= 0.0:
        return 0.0
    return num / den


def rsa(x: list[list[float]], y: list[list[float]]) -> float:
    """表征相似度分析：两个「距离矩阵」上三角之间的 Pearson 相关。"""
    dx = _upper_distances(x)
    dy = _upper_distances(y)
    return _pearson(dx, dy)


def _upper_distances(x: list[list[float]]) -> list[float]:
    n = len(x)
    out: list[float] = []
    for i in range(n):
        xi = x[i]
        for j in range(i + 1, n):
            out.append(_dist(xi, x[j]))
    return out


def _dist(u: list[float], v: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    da = math.sqrt(sum((p - ma) ** 2 for p in a))
    db = math.sqrt(sum((q - mb) ** 2 for q in b))
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return num / (da * db)


def similarity_of(
    x: list[list[float]],
    y: list[list[float]],
    metric: str,
) -> float:
    """按名字分发。未知名字抛错，不静默回退到 cosine ——
    静默回退会让「换了个度量」这件事在报告里完全看不出来。"""
    if metric == "cosine":
        return mean_cosine(x, y)
    if metric == "centered_cosine":
        return centered_cosine(x, y)
    if metric == "cka":
        return linear_cka(x, y)
    if metric == "rsa":
        return rsa(x, y)
    raise ValueError(f"未知的相似度度量 {metric!r}；合法值：{sorted(SIMILARITY_METRICS)}")


def layer_similarity_matrix(
    states_by_layer: dict[int, list[list[float]]],
    layers: list[int],
    metric: str,
) -> list[list[float]]:
    """层 × 层的相似度矩阵。

    对角线恒为 1（自己和自己比）—— 这是自洽性检查，
    测试里用 ``matrix[i][i] ≈ 1`` 钉住；不等于 1 说明度量实现有问题。
    """
    n = len(layers)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = similarity_of(states_by_layer[layers[i]], states_by_layer[layers[j]], metric)
            out[i][j] = v
            out[j][i] = v
    return out
