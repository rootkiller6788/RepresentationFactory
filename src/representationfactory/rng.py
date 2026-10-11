"""确定性随机数 —— 可复现性的地基。

本项目的验收标准之一是「输出**可复现**的 Representation Graph，支持跨模型结构对比」，
所以任何不确定性都必须被钉死。两个必须绕开的坑（与同工作区其它项目共用同一条纪律）：

1. **内置 ``hash()`` 对 str 加随机盐**（``PYTHONHASHSEED``），跨进程不稳定。
   用它派生种子会让「重跑结果一致」直接失效，因此这里一律用 sha256。
2. **全局 ``random`` 的状态会被任何一次调用推进**，导致结果依赖调用顺序。
   这里按 ``(seed, 用途, 对象 id)`` 派生**独立实例**。

本项目真正需要随机性的地方有四处，**前两处会直接改变被观测的表征**，
后两处只影响「怎么展示」—— 但四者都必须可复现，否则同一份配置在两次运行里
会给出不同的曲线，而曲线正是本项目的结论本身：

- **构造式标本的权重**：``model.py`` 里的 Q/K/V/O 与 FFN 权重全部由种子派生。
  种子一变，被观测的对象就变了 —— 这是最根本的一处。
- **位置编码与「自由子空间」的填充**：embedding 里预留了一小段无结构维度，
  用来给探针制造一点真实噪声；不钉死的话曲线会在两次运行间抖动。
- **PCA 的初始向量**：幂迭代从一个随机向量起步。它不影响最终特征向量
  （除符号，见 ``project.py`` 的符号约定），但会改变迭代轮数，从而改变
  浮点尾数 —— 而尾数会被 ``round()`` 之后写进产物。
- **随机投影对照**：用来证明「PCA 的效果不是随便投影一下就能拿到」，
  必须与 PCA 用同一份数据、同一个种子。
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


def derive_seed(*parts: Any) -> int:
    """由若干片段派生一个 64bit 整数种子（跨平台、跨进程稳定）。"""
    joined = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(joined.encode("utf-8")).digest()[:8], "big")


def make_rng(*parts: Any) -> random.Random:
    """派生一个独立的 ``Random`` 实例。同样的 ``parts`` 必得同样的序列。"""
    return random.Random(derive_seed(*parts))


def stable_hash64(*parts: Any) -> int:
    """把若干片段映射到 64bit 无符号整数。

    与 :func:`derive_seed` 的区别：那个是「取前 8 字节」，这个是**取全部**
    再折叠，用于需要把任意字符串均匀撒到 N 个桶里的场景（如把「词形」
    映射到 ±1 词形码的一个分量）。同样不用内置 ``hash()``。
    """
    joined = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
