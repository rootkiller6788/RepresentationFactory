"""手写线性代数 —— 零依赖，且**完全确定**。

为什么手写而不用 numpy？两个理由，都不是「不能装」：

1. **可复核**：本项目的结论是「某个属性在第几层变得线性可分」。
   要让人能核对，从 ``XᵀX + λI`` 到解出权重、到 argmax 打分，每一步都得看得见。
   调一次 ``numpy.linalg.lstsq`` 拿到一个数字，没人能复核它是不是被 LAPACK
   的某个分支悄悄改过（不同 BLAS 后端在病态问题上确实会给不同的最小二乘解）。
2. **确定性**：``numpy.linalg.eigh`` 的特征向量**符号是任意的**（由 LAPACK
   实现决定，跨版本可能翻转）。本项目要把降维结果写进产物、并断言「两次运行
   一致」，所以特征向量的符号必须由**我们自己**定下来（见 :func:`canonical_sign`）。

规模说明：本项目 d ≤ 40、n ≤ 600、类别数 ≤ 24，所以 O(d³) 的高斯消元
与 O(iters·d²) 的幂迭代都绰绰有余 —— 全流水线 48 个观测配置秒级跑完。
真上大模型时这套实现会立刻成为瓶颈，那时该换 numpy / torch（见 pyproject 的 extra）。
"""

from __future__ import annotations

import math
from typing import Any

Vector = list[float]
Matrix = list[list[float]]


# ----------------------------------------------------------------------
# 基本运算
# ----------------------------------------------------------------------
def zeros(n: int, m: int | None = None) -> Matrix:
    """``m is None`` 时返回 n×n 方阵。"""
    m = n if m is None else m
    return [[0.0] * m for _ in range(n)]


def eye(n: int) -> Matrix:
    out = zeros(n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def transpose(a: Matrix) -> Matrix:
    if not a:
        return []
    n, m = len(a), len(a[0])
    return [[a[i][j] for i in range(n)] for j in range(m)]


def dot(u: Vector, v: Vector) -> float:
    """点积。用 ``math.sumprod``（Python 3.12+）—— 比 ``sum(map(mul, ...))``
    快约 1.8 倍，且同样精确。低版本回退到生成器写法。"""
    if hasattr(math, "sumprod"):
        return math.sumprod(u, v)
    return sum(x * y for x, y in zip(u, v))


def norm(v: Vector) -> float:
    return math.sqrt(dot(v, v))


def scale(v: Vector, k: float) -> Vector:
    return [x * k for x in v]


def axpy(a: float, x: Vector, y: Vector) -> Vector:
    """``a*x + y``。"""
    return [a * xi + yi for xi, yi in zip(x, y)]


def matvec(a: Matrix, v: Vector) -> Vector:
    return [dot(row, v) for row in a]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """``a @ b``。按 ``b`` 的行取，避免转置。"""
    if not a or not b:
        return []
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]


def gram(x: Matrix) -> Matrix:
    """``XᵀX``（d×d）。d ≤ 40，直接算；n 远大于 d 时这是更便宜的那一侧。

    CKA 与 PCA 都建立在这个矩阵上，所以它只有一份实现 ——
    两份实现会让「同一个量在两处算出不同的值」变成一个长期难查的幽灵。

    **按列算，不要按行累加。** 第一版是 ``for row in x: for i, j: out[i][j] += ...``，
    那是 O(n·d²) 次 Python 层运算 —— 2560 个位置 × 48 维时要跑 300 万次，
    而它是**每个观测配置、每一层**都要算一次的（48 个配置 × 5 层）。
    取列之后变成 d²/2 次 ``math.sumprod``（C 层实现），实测快两个数量级。
    """
    if not x:
        return []
    d = len(x[0])
    cols = [[row[j] for row in x] for j in range(d)]
    out = zeros(d)
    for i in range(d):
        oi = out[i]
        ci = cols[i]
        for j in range(i, d):
            oi[j] = dot(ci, cols[j])
    for i in range(d):
        for j in range(i):
            out[i][j] = out[j][i]
    return out


def frobenius2(a: Matrix) -> float:
    return sum(dot(row, row) for row in a)


# ----------------------------------------------------------------------
# 线性方程组
# ----------------------------------------------------------------------
def solve_multi(a: Matrix, b: Matrix) -> Matrix:
    """解 ``A X = B``（``A``: d×d，``B``: d×k）。

    高斯消元 + 部分主元。主元接近 0 时抛错而不是静默给出发散的结果 ——
    「解不出来」必须是一个显式的失败，否则它会被伪装成「探针准确率 0.02」
    这种看起来像结论的数字。
    """
    d = len(a)
    if d == 0:
        return []
    k = len(b[0]) if b else 0
    # 增广矩阵 [A | B]，原地消元
    m = [list(a[i]) + list(b[i]) for i in range(d)]
    for col in range(d):
        # 部分主元
        piv = max(range(col, d), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError(
                f"矩阵在列 {col} 上奇异（主元 {m[piv][col]:.3e}）——"
                "岭回归的 λ 是否太小？特征是否退化？"
            )
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        row_c = m[col]
        for j in range(col, d + k):
            row_c[j] *= inv
        for r in range(d):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            row_r = m[r]
            for j in range(col, d + k):
                row_r[j] -= factor * row_c[j]
    return [m[i][d:] for i in range(d)]


def solve(a: Matrix, v: Vector) -> Vector:
    return [row[0] for row in solve_multi(a, [[x] for x in v])]


def invert(a: Matrix) -> Matrix:
    d = len(a)
    return solve_multi(a, eye(d))


def add_ridge(a: Matrix, lam: float) -> Matrix:
    """``A + λI``（返回新矩阵，不改入参）。"""
    d = len(a)
    out = [list(row) for row in a]
    for i in range(d):
        out[i][i] += lam
    return out


# ----------------------------------------------------------------------
# 中心化 / 标准化
# ----------------------------------------------------------------------
def column_means(x: Matrix) -> Vector:
    if not x:
        return []
    n, d = len(x), len(x[0])
    out = [0.0] * d
    for row in x:
        for j in range(d):
            out[j] += row[j]
    return [v / n for v in out]


def center_columns(x: Matrix) -> tuple[Matrix, Vector]:
    """按列去均值。返回 ``(Xc, means)``。

    **这一步是 CKA 与 centered-cosine 的全部区别所在**，也是本项目的
    一个关键机制：LayerNorm 之后（或本项目的构造式标本里）所有向量都带一个
    共同的均值方向，不做中心化时任意两层的余弦相似度都会趋近 1，
    于是「层间差异」被那个共同方向掩盖掉 —— 而且**不报错**，
    只会给出一张全是 0.99 的矩阵，看着还挺「稳定」。
    """
    means = column_means(x)
    out = [[xi - mj for xi, mj in zip(row, means)] for row in x]
    return out, means


def standardize_columns(x: Matrix) -> tuple[Matrix, Vector, Vector]:
    """按列标准化到零均值、单位方差。返回 ``(Xs, means, stds)``。

    降维前必须做：本项目里 lexical 段的自点积是 6，而 marker 段是 6、
    semantic 段是 4 —— 量级相近但不同，不标准化的话 PCA 的主方向
    会被方差最大的那一段独占，散点图上只看得到「词形码」，看不到任何层间演化。
    """
    xc, means = center_columns(x)
    if not xc:
        return [], means, []
    d = len(xc[0])
    var = [0.0] * d
    for row in xc:
        for j in range(d):
            var[j] += row[j] * row[j]
    n = len(xc)
    stds = [math.sqrt(v / n) if v > 0 else 1.0 for v in var]
    out = [[xi / sj for xi, sj in zip(row, stds)] for row in xc]
    return out, means, stds


# ----------------------------------------------------------------------
# PCA（幂迭代 + 收缩）
# ----------------------------------------------------------------------
def canonical_sign(v: Vector) -> Vector:
    """把特征向量的符号钉死：**绝对值最大的那个分量取正**。

    这条约定看着琐碎，但它是「降维产物可复现」的**唯一**依据：
    特征向量 ``v`` 与 ``-v`` 描述同一个方向，数值上却完全不同，
    于是散点图会整体镜像，而 PCA 的符号约定在不同实现、不同 BLAS、
    甚至同一实现的不同运行里都可能翻转。**不钉死的话，
    「两次运行产物一致」这个检查会变成随机失败。**
    """
    if not v:
        return v
    best = 0
    for i in range(1, len(v)):
        if abs(v[i]) > abs(v[best]) + 1e-15:
            best = i
    if v[best] < 0:
        return [-x for x in v]
    return list(v)


def power_iteration(
    cov: Matrix,
    n_components: int,
    iters: int = 60,
    rng: Any = None,
) -> tuple[list[Vector], list[float]]:
    """对称矩阵的前 ``n_components`` 个特征向量（降序）+ 特征值。

    做法：幂迭代 + **收缩**（每求出一个方向就从矩阵里减掉它的贡献，
    相当于把该方向上的特征值归零）。收缩而不是「正交化后重跑」，
    是因为后者在特征值接近时会反复收敛到同一个方向。

    返回的特征向量都过了 :func:`canonical_sign`，所以**逐位可复现**。
    """
    d = len(cov)
    if d == 0 or n_components <= 0:
        return [], []
    work = [list(row) for row in cov]
    vecs: list[Vector] = []
    vals: list[float] = []
    for _ in range(min(n_components, d)):
        # 初始向量：有 rng 就用它（同一 seed 必得同一序列），否则用确定性构造
        if rng is not None:
            v = [rng.uniform(-1.0, 1.0) for _ in range(d)]
        else:
            v = [1.0 if i == (len(vecs) % d) else 0.0 for i in range(d)]
        nv = norm(v)
        if nv == 0:
            v = [1.0] + [0.0] * (d - 1)
            nv = 1.0
        v = [x / nv for x in v]
        lam = 0.0
        for _ in range(iters):
            w = matvec(work, v)
            nw = norm(w)
            if nw < 1e-14:
                lam = 0.0
                break
            v = [x / nw for x in w]
            lam = nw
        v = canonical_sign(v)
        # 用 Rayleigh 商把特征值算准（幂迭代的 ||w|| 收敛略慢）
        av = matvec(cov, v)
        lam = dot(v, av)
        if lam <= 1e-12:
            break
        vecs.append(v)
        vals.append(lam)
        # 收缩：work ← work − λ v vᵀ
        for i in range(d):
            vi = v[i]
            if vi == 0.0:
                continue
            wi = work[i]
            for j in range(d):
                wi[j] -= lam * vi * v[j]
    return vecs, vals


def project(x: Matrix, basis: list[Vector]) -> Matrix:
    """把每一行投到 ``basis`` 上，返回 ``n × len(basis)``。"""
    if not basis:
        return [[] for _ in x]
    return [[dot(row, b) for b in basis] for row in x]


def random_projection(d: int, k: int, rng: Any) -> list[Vector]:
    """随机正交化投影 —— 用来证明「PCA 的效果不是随便投影一下就能拿到」。

    做法：随机高斯矩阵 → 逐列 Gram–Schmidt 正交化 → 归一化。
    没有这个对照，报告里「PCA 后层间分化更清楚」这句话是不可证伪的。
    """
    basis: list[Vector] = []
    for _ in range(k):
        v = [rng.gauss(0.0, 1.0) for _ in range(d)]
        for b in basis:
            p = dot(v, b)
            v = [vi - p * bi for vi, bi in zip(v, b)]
        nv = norm(v)
        if nv < 1e-12:
            continue
        basis.append(canonical_sign([x / nv for x in v]))
    return basis
