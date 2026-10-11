"""抓取内部状态。

README 第七节的 M0 就是这一层：「定义 Hook 点与 State Spec 的 schema，
跑通 Forward → Capture → 保存一次」。

Hook 点
-------
本项目把 hook 点写成**常量**，而不是「在哪儿插个回调」这种运行时机制：

    HOOKS = ("embedding", "attn_out", "block_out", "ffn_out")

理由：回调式 hook 会让「抓了什么」依赖调用顺序，而本项目要断言
「两次运行抓到的是同一个东西」。写死成常量后，hook 点是**配置**而不是**行为**，
于是「抓哪些层」这件事可以被测试钉住（``observation.layers`` 与之一一对应）。

``block_out`` 是本项目真正用的那一个（每层 block 之后的残差流）。
``attn_out`` 与 ``ffn_out`` 也留着 —— 想看「注意力搬了什么 / FFN 写了什么」
可以分别取它们，这正是本 Lab 存在的意义（把两者分开看）。

一次抓取，多次观测
------------------
``Capture`` 被**同一个模型下的全部观测配置共用**（48 个配置里有一半用它）。
这不是省时间的优化，而是**正确性前提**：如果每个配置各抓一次，
「所有配置看的是同一份内部状态」这个前提在实现层就失效了，
而本报告的全部结论都建立在「同一批状态、换一种看法」之上。
所以缓存要用对象同一性断言守住（``assert first is again``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .model import ForwardTrace, Model, SpecimenData, forward
from .schema import HeadStats

#: Hook 点。写死成常量而不是回调，见模块文档。
HOOKS: tuple[str, ...] = ("embedding", "attn_out", "block_out", "ffn_out")


@dataclass
class Capture:
    """一次抓取的全部产物。

    ``hidden[l]`` 是第 ``l`` 层的 ``block_out``（``l = 0`` 是 embedding），
    展平成长度 ``n_tokens`` 的逐位置向量表 —— 与 ``data`` 的位置顺序一一对应。

    ``attention[l][si][h]`` 是第 ``l`` 层第 ``h`` 个头在第 ``si`` 句上的
    ``n × n`` 注意力矩阵。它只在需要时保留（完整矩阵很大，见
    ``attention_samples``）：默认**句数被截断**，因为「注意力长什么样」
    只需要看几条就够，而头画像（``heads``）是逐句算的、不需要留矩阵。
    """

    model: Model
    data: SpecimenData
    trace: ForwardTrace
    #: 逐层逐头的画像（全部句子）
    heads: list[HeadStats] = field(default_factory=list)
    #: 保留了完整注意力矩阵的那些句子下标
    attention_sample_sentences: list[int] = field(default_factory=list)
    #: {spec_id: 该 spec 专属的观测结果} 由上层填，这里只放抓取本身
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def layers(self) -> list[int]:
        return list(range(len(self.trace.hidden)))

    def hidden_at(self, layer: int) -> list[list[float]]:
        return self.trace.hidden[layer]

    def attention_at(self, layer: int, sentence_index: int, head: int) -> list[list[float]]:
        return self.trace.attention[layer][sentence_index][head]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.name,
            "n_tokens": self.data.n_tokens,
            "n_sentences": self.data.n_sentences,
            "layers": self.layers,
            "heads": [h.to_dict() for h in self.heads],
            "attention_sample_sentences": list(self.attention_sample_sentences),
            "model_info": self.model.to_dict(),
        }


def head_statistics(trace: ForwardTrace, data: SpecimenData) -> list[HeadStats]:
    """逐层逐头算注意力画像。

    五个量各自回答一个问题，缺一个都会让「结构与表征如何耦合」
    变成一句空话：

    - ``entropy``      注意力有多集中（归一化到 [0,1]，1 = 完全均匀）
    - ``self_weight``  有多少注意力留在自己身上
    - ``locality``     有多少注意力落在 ±1 邻域内
    - ``sink_weight``  有多少注意力落在主题标记词上（**本项目最关键的量**）
    - ``mean_offset``  平均看多远（「注意力的有效半径」）
    """
    out: list[HeadStats] = []
    for l in range(len(trace.attention)):
        n_heads = len(trace.attention[l][0])
        for h in range(n_heads):
            ent_sum = self_sum = loc_sum = sink_sum = off_sum = 0.0
            cnt = 0
            for si, per_head in enumerate(trace.attention[l]):
                alpha = per_head[h]
                n = len(alpha)
                if n == 0:
                    continue
                marker = data.marker_index[si]
                log_n = math.log(n) if n > 1 else 1.0
                for i in range(n):
                    row = alpha[i]
                    ent_sum += -sum(p * math.log(p + 1e-15) for p in row) / log_n
                    self_sum += row[i]
                    loc_sum += row[max(0, i - 1)] + row[min(n - 1, i + 1)]
                    if marker >= 0:
                        sink_sum += row[marker]
                    off_sum += sum(abs(i - j) * p for j, p in enumerate(row))
                    cnt += 1
            cnt = max(cnt, 1)
            out.append(
                HeadStats(
                    layer=l + 1,  # 层号 1 起（0 是 embedding，没有注意力）
                    head=h,
                    entropy=ent_sum / cnt,
                    self_weight=self_sum / cnt,
                    locality=loc_sum / cnt / 2.0,
                    sink_weight=sink_sum / cnt,
                    mean_offset=off_sum / cnt,
                )
            )
    return out


def capture_model(
    model: Model,
    data: SpecimenData,
    cfg: dict[str, Any],
    attention_samples: int | None = None,
) -> Capture:
    """抓一次。

    ``attention_samples`` 控制完整注意力矩阵留几句；
    头画像是**全部句子**上算的（不受它影响）—— 两者别混，
    否则「头画像」会变成「只看了几句的头画像」，而它要回答的是
    「这个头在做什么」，那必须用全部句子。
    """
    n_samples = (
        int(cfg["observation"]["attention_samples"])
        if attention_samples is None
        else attention_samples
    )
    trace = forward(model, data, want_attention=True)
    heads = head_statistics(trace, data)
    n_sentences = data.n_sentences
    sample_ids = (
        sorted(range(min(n_samples, n_sentences)))
        if n_samples < n_sentences
        else list(range(n_sentences))
    )
    if len(sample_ids) < n_sentences:
        # 只保留前几句的完整矩阵，其余换成 None（**不是**删掉元素 ——
        # 删掉会让 sentence_index 与真实句号错位）
        for l in range(len(trace.attention)):
            for si in range(n_sentences):
                if si not in sample_ids:
                    trace.attention[l][si] = [[] for _ in trace.attention[l][si]]
    return Capture(
        model=model,
        data=data,
        trace=trace,
        heads=heads,
        attention_sample_sentences=sample_ids,
        meta={
            "attention_samples": len(sample_ids),
            "n_sentences": n_sentences,
        },
    )


class CaptureCache:
    """按模型缓存抓取结果。

    **它存在的理由不是加速，是正确性**：48 个观测配置里，同一个模型的
    24 个配置必须看**同一份**内部状态。缓存保证这一点，
    而 ``test_capture_is_shared`` 用对象同一性断言把它钉住。
    """

    def __init__(self) -> None:
        self._by_model: dict[str, Capture] = {}

    def get(
        self,
        model: Model,
        data: SpecimenData,
        cfg: dict[str, Any],
    ) -> Capture:
        key = model.name
        cached = self._by_model.get(key)
        if cached is not None:
            return cached
        cap = capture_model(model, data, cfg)
        self._by_model[key] = cap
        return cap

    def models(self) -> list[str]:
        return sorted(self._by_model)

    def __len__(self) -> int:
        return len(self._by_model)
