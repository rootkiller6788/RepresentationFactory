"""RepresentationFactory —— 把模型内部表征看清楚。

**它不是 Factory，是 Microscope**（README 第二节）。不做生成、不做优化，
只做一件事：把隐藏状态、注意力、激活看清楚 —— 它长什么样、
怎么随层数演化、不同输入之间如何分群。

流水线：

    Model → Forward → Capture(Hidden State | Attention Map | Activation)
        → Project/Reduce → Measure Similarity
        → Probe(Similarity | Cluster | Geometry)
        → Construct Graph → Compare Layer/Head/Model → Visualize

核心产物：**逐层演化曲线**（哪个属性在第几层变得线性可分）+ **Representation Graph**。

零第三方依赖：微型 transformer 与全部线性代数都是手写的。
理由不是「不能装」，而是「要能复核」—— 见 pyproject.toml 里的说明。

快速开始：

    python main.py run                       # 端到端，产物落 out/
    python main.py curve                     # 逐层演化曲线（核心）
    python main.py probe --kind cluster       # 线性可分性探针
    python main.py graph --dot                # Representation Graph
    python main.py compare                    # 跨模型 / 跨头对比
    python main.py explain --position s01#3   # 血缘三问
"""

from __future__ import annotations

from .schema import CODE_VERSION

__version__ = "0.1.0"
__all__ = ["CODE_VERSION", "__version__"]
