"""降维（Project / Reduce）。

README 第四节的 ``Project / Reduce`` 就是这一步。它**不是可选的装饰** ——
``d_model = 40`` 的表征没法画出来，而本项目的产物之一就是「可交互散点」。

三个必须钉死的细节（每一个都会让产物不可复现或不可读）
--------------------------------------------------------

1. **PCA 的特征向量符号**。``v`` 与 ``−v`` 是同一个方向，数值上却完全不同，
   于是散点图会整体镜像。LAPACK / 不同实现 / 甚至同一实现的不同运行
   都可能给出不同的符号。本项目用 ``canonical_sign``（绝对值最大的分量取正）
   把符号**自己**定下来 —— 这是「两次运行产物一致」的唯一依据。

2. **必须按列标准化**。本项目各子空间的量级并不相同：lexical 段自点积 6、
   marker 段 6、semantic 段 4…… 不标准化的话 PCA 的第一主成分会被
   方差最大的那一段独占，散点图上只看得到「词形码」，
   而本项目要看的是**层与层之间的差异**。

3. **要有随机投影对照**。没有对照，「PCA 之后层间分化更清楚了」这句话
   是不可证伪的 —— 随便投一下也许也一样清楚。``random_projection`` 给的就是
   这个对照：它与 PCA 用同一份数据、同一个种子、同样的维数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .linalg import power_iteration, project, random_projection, standardize_columns
from .rng import make_rng

#: 降维方法。``random`` 是**对照**，不是可选实现 —— 见模块文档第 3 点。
PROJECT_METHODS: tuple[str, ...] = ("none", "pca", "random")


@dataclass
class Projection:
    """一次降维的结果。"""

    method: str
    n_components: int
    #: 降维后的坐标（n × k）
    coords: list[list[float]] = field(default_factory=list)
    #: PCA 的特征值（降序）；``none`` / ``random`` 时为空
    eigenvalues: list[float] = field(default_factory=list)
    #: 方差解释率（降序，和为 1）
    explained_variance: list[float] = field(default_factory=list)
    #: 标准化用的均值 / 标准差 —— **必须存下来**，否则无法把新点投到同一坐标系
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_explained(self) -> float:
        return sum(self.explained_variance)

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_components": self.n_components,
            "eigenvalues": [round(v, 6) for v in self.eigenvalues],
            "explained_variance": [round(v, 6) for v in self.explained_variance],
            "total_explained": round(self.total_explained, 6),
            "n_points": len(self.coords),
        }


def reduce_states(
    states: list[list[float]],
    method: str,
    cfg: dict[str, Any],
    n_components: int | None = None,
    seed_parts: tuple[Any, ...] = ("pca",),
) -> Projection:
    """把一组表征降到低维。

    ``method == "none"`` 时仍然做标准化并返回原维度 ——
    这是刻意的：**对照行要能落在同一条代码路径上**，
    否则「降维有用」这个结论混进了「有没有标准化」这个变量。
    """
    k = (
        int(cfg["observation"]["pca_components"])
        if n_components is None
        else n_components
    )
    xs, means, stds = standardize_columns(states)
    if method == "none":
        return Projection(
            method="none",
            n_components=len(xs[0]) if xs else 0,
            coords=xs,
            means=means,
            stds=stds,
            meta={"note": "未降维，仅标准化（对照行）"},
        )

    iters = int(cfg["observation"]["power_iters"])
    d = len(xs[0]) if xs else 0
    if method == "pca":
        cov = _covariance(xs)
        basis, values = power_iteration(cov, min(k, d), iters, make_rng(*seed_parts))
        total = sum(values) if values else 0.0
        evr = [v / total for v in values] if total > 0 else [0.0] * len(values)
        return Projection(
            method="pca",
            n_components=len(basis),
            coords=project(xs, basis),
            eigenvalues=values,
            explained_variance=evr,
            means=means,
            stds=stds,
            meta={"iters": iters},
        )
    if method == "random":
        basis = random_projection(d, min(k, d), make_rng(*seed_parts))
        return Projection(
            method="random",
            n_components=len(basis),
            coords=project(xs, basis),
            means=means,
            stds=stds,
            meta={"note": "随机正交投影对照（证明 PCA 的效果不是白来的）"},
        )
    raise ValueError(f"未知的降维方法 {method!r}；合法值：{sorted(PROJECT_METHODS)}")


def _covariance(xs: list[list[float]]) -> list[list[float]]:
    """标准化之后协方差就是 ``XᵀX / n``。

    单独写出来（而不是在调用处内联）是为了让「协方差」只有一份定义 ——
    PCA 与「方差解释率」必须用同一个矩阵算，否则两者会各说各话。
    """
    if not xs:
        return []
    d = len(xs[0])
    n = len(xs)
    out = [[0.0] * d for _ in range(d)]
    for row in xs:
        for i in range(d):
            ri = row[i]
            if ri == 0.0:
                continue
            oi = out[i]
            for j in range(i, d):
                oi[j] += ri * row[j]
    for i in range(d):
        for j in range(i):
            out[i][j] = out[j][i]
    return [[v / n for v in row] for row in out]
