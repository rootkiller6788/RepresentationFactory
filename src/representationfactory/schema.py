"""核心对象。

本项目的形状是**「测量型流水线」**（Instrument）—— 与同工作区的
`DatasetFactory` / `TokenizerFactory` / `EmbeddingFactory` **都不一样**：

    Model → Forward → Capture(Hidden State | Attention Map | Activation)
        → Project/Reduce → Measure Similarity → Probe(Similarity | Cluster | Geometry)
        → Construct Graph → Compare Layer/Head/Model → Visualize

**它不是 Factory，是 Microscope**（README 第二节的原话）。三点结构性差异：

1. **没有候选，也没有前沿。** 前三类项目都在「挑一个更好的东西」；
   本项目不挑 —— 它把内部状态**看清楚**。所以没有 Pareto 支配、没有超体积、
   没有设计空间的笛卡尔积。有的是一条**逐层演化曲线**。
2. **被观测的对象是模型内部，不是数据。** 记录不再是「样本」，而是
   「位置」（``layer × head × token``）。于是「漏斗差额」这类证据不存在了，
   主证据变成：**某个属性在第几层变得线性可分**。
3. **必须有已知真值的标定标本。** 一台仪器要能被验收，就得先对准一个
   「结构已知」的标本。本项目因此自带一个**构造式标本**（``model.py``）：
   它的每一层负责写哪一类特征，是我们**设计**出来的、写在文档里的。
   验收标准 = **探针能否复原这个已知真值**；并且必须有一个**随机权重对照**，
   证明「阶梯不是随便一个模型都有的」。没有对照组，曲线只是漂亮的折线图。

三层对象
--------
    CorpusStats    语料画像 + 已知真值统计（「被观测的是什么」）
    StateSpec      一次观测的配置（模型 × 池化 × 相似度 × 划分 × 降维）
    CaptureRun     一个 spec 的完整观测结果（曲线 + 探针 + 头画像 + 图）
    LabResult      整条流水线的产物：全部 run + 矩阵 + 消融 + 结论 + 版本
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: 参与 digest 的代码版本。口径变了就改这个值，否则新旧结果会被当成同一批。
CODE_VERSION = "representationfactory/0.1.0"

#: 被观测的三类属性。它们有**已知真值**（来自 `corpus.py` 的词典），
#: 所以「第几层变得线性可分」是一个可核对的数字，而不是主观印象。
#: ``form``  = 词形（浅层：词表级）
#: ``role``  = 句法角色（中层：语法）
#: ``topic`` = 语义主题（深层：语义）
PROPERTIES: tuple[str, ...] = ("form", "role", "topic")

#: README 第六节的三类探针。**三类都要有，缺一不可** ——
#: 它们回答三个不同的问题，且互相验证：
#:   similarity — 哪些 token 在表征空间里靠近？（度量层）
#:   cluster    — 表征里是否已线性可分地编码了某属性？（判定层）
#:   geometry   — 结构与表征如何耦合？哪些头在做同一件事？（结构层）
PROBE_KINDS: tuple[str, ...] = ("similarity", "cluster", "geometry")

#: 观测轴。注意：本项目**没有设计空间** —— 这五个轴描述的是「怎么观测」，
#: 而不是「生成什么」。展开顺序 = 报告里的列顺序 = 消融的分组顺序。
OBSERVATION_AXES: tuple[str, ...] = (
    "model",
    "pooling",
    "similarity",
    "split",
    "project",
)

#: **方法学对照轴**：``split`` 不改变被观测对象，只改变「怎么划分训练/测试」。
#: 保留它是为了在矩阵里**如实展示**一个陷阱：按 token 随机划分会让探针准确率
#: 系统性虚高（同一句内的 token 高度相关，测试集里混进了训练集的「同句兄弟」）。
#: 它是本项目里唯一一个「越差越正确」的轴 —— 数字更高说明方法更错。
CONTROL_AXES: tuple[str, ...] = ("split",)

#: 短标签。它们会进 spec_id（进而进表格与 DAG 节点 id），所以必须短。
_AXIS_ABBR: dict[str, dict[str, str]] = {
    "pooling": {"token": "tok", "sentence": "sent"},
    "similarity": {"cosine": "cos", "centered_cosine": "ccos", "cka": "cka"},
    "split": {"by_sentence": "bs", "by_token": "bt"},
    "project": {"none": "raw", "pca": "pca"},
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, n: int = 12) -> str:
    """稳定短 id。

    **不用内置 ``hash()``** —— 它按 ``PYTHONHASHSEED`` 加盐，跨进程不稳定。
    """
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:n]


# ----------------------------------------------------------------------
# 被观测对象：语料 + 已知真值
# ----------------------------------------------------------------------
@dataclass
class TokenRecord:
    """一个 token 位置的**已知真值**。

    这是本项目的「标尺」：词典在 ``corpus.py`` 里写死，所以每个位置
    属于哪个词形、什么句法角色、什么语义主题，都是**先验已知**的。
    探针要复原的正是这三个标签。

    ``is_ambiguous`` 是本项目最关键的一个布尔量：歧义词的**角色不由词形决定**，
    而由句子的语义主题决定。于是「角色」这个属性对它们来说需要
    **全局上下文**才能确定 —— 逐层曲线上那一次跳变就发生在
    「注意力的有效范围扩大到能看见主题标记词」的那一层。
    """

    sentence_id: str
    index: int
    form: str
    role: str
    topic: str
    #: 角色是否不由词形唯一决定（需要看上下文）
    is_ambiguous: bool = False
    #: 该位置是不是本句的「主题标记词」（topic marker）
    is_marker: bool = False

    @property
    def position_id(self) -> str:
        return f"{self.sentence_id}#{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "sentence_id": self.sentence_id,
            "index": self.index,
            "form": self.form,
            "role": self.role,
            "topic": self.topic,
            "is_ambiguous": self.is_ambiguous,
            "is_marker": self.is_marker,
        }


@dataclass
class Sentence:
    """一句话 + 它的全部位置真值。"""

    sentence_id: str
    text: str
    topic: str
    tokens: list[TokenRecord] = field(default_factory=list)
    #: 主题标记词在句中的位置（-1 表示这句没有标记词）
    marker_index: int = -1

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "topic": self.topic,
            "n_tokens": self.n_tokens,
            "marker_index": self.marker_index,
            "tokens": [t.to_dict() for t in self.tokens],
        }


@dataclass
class CorpusStats:
    """语料画像 + 真值分布。它决定「该关注哪个属性」。"""

    name: str
    n_sentences: int
    n_tokens: int
    n_forms: int
    n_roles: int
    n_topics: int
    n_ambiguous_tokens: int
    n_marker_tokens: int
    mean_sentence_len: float
    #: 主题 → 句子数
    topic_sizes: dict[str, int] = field(default_factory=dict)
    #: 角色 → 位置数
    role_sizes: dict[str, int] = field(default_factory=dict)
    #: 词形 → 出现次数（只列出现最多的若干个，报告里用）
    top_forms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_sentences": self.n_sentences,
            "n_tokens": self.n_tokens,
            "n_forms": self.n_forms,
            "n_roles": self.n_roles,
            "n_topics": self.n_topics,
            "n_ambiguous_tokens": self.n_ambiguous_tokens,
            "n_marker_tokens": self.n_marker_tokens,
            "mean_sentence_len": round(self.mean_sentence_len, 3),
            "topic_sizes": {k: v for k, v in sorted(self.topic_sizes.items())},
            "role_sizes": {k: v for k, v in sorted(self.role_sizes.items())},
            "top_forms": {k: v for k, v in sorted(self.top_forms.items())},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorpusStats":
        return cls(
            name=d["name"],
            n_sentences=d["n_sentences"],
            n_tokens=d["n_tokens"],
            n_forms=d["n_forms"],
            n_roles=d["n_roles"],
            n_topics=d["n_topics"],
            n_ambiguous_tokens=d["n_ambiguous_tokens"],
            n_marker_tokens=d["n_marker_tokens"],
            mean_sentence_len=d["mean_sentence_len"],
            topic_sizes=d.get("topic_sizes", {}),
            role_sizes=d.get("role_sizes", {}),
            top_forms=d.get("top_forms", {}),
        )


# ----------------------------------------------------------------------
# 观测配置
# ----------------------------------------------------------------------
@dataclass
class StateSpec:
    """一次观测的配置。

    字段与 :data:`OBSERVATION_AXES` 一一对应，顺序也一致 ——
    报告里的矩阵列顺序直接来自这里，不在报告层重新排。
    """

    #: 被观测的模型：``constructive``（构造式标本）| ``random``（随机权重对照）
    model: str
    #: 池化：``token``（逐位置）| ``sentence``（句均值）
    pooling: str
    #: 相似度度量：``cosine`` | ``centered_cosine`` | ``cka``
    similarity: str
    #: 训练/测试划分：``by_sentence``（按句分组）| ``by_token``（按位置随机）
    split: str
    #: 降维：``none`` | ``pca``
    project: str

    @property
    def spec_id(self) -> str:
        """稳定且唯一的短标识。DAG 节点 id / 矩阵行名都用它。

        形如 ``constructive-tok-ccos-bs-pca``。五段固定，顺序 = 观测轴顺序。
        """
        parts = [self.model]
        for axis in OBSERVATION_AXES[1:]:
            value = str(getattr(self, axis))
            parts.append(_AXIS_ABBR.get(axis, {}).get(value, value))
        return "-".join(parts)

    @classmethod
    def from_spec_id(cls, spec_id: str) -> "StateSpec":
        """``spec_id`` 的反函数（见 :attr:`spec_id`）。

        只读命令从 ``out/curves.json`` 等产物重建 ``LabResult`` 时要用它，
        避免为了打印一条曲线而重跑整条 75s 的流水线。
        """
        parts = spec_id.split("-")
        if len(parts) != 1 + len(OBSERVATION_AXES) - 1:
            raise ValueError(f"spec_id 格式不符：{spec_id!r}")
        model = parts[0]
        # 反查缩写表
        rev: dict[str, dict[str, str]] = {}
        for axis, abbr in _AXIS_ABBR.items():
            rev[axis] = {v: k for k, v in abbr.items()}
        kwargs: dict[str, str] = {}
        for axis, abbr in zip(OBSERVATION_AXES[1:], parts[1:]):
            kwargs[axis] = rev.get(axis, {}).get(abbr, abbr)
        return cls(model=model, **kwargs)

    def axis(self, name: str) -> str:
        return str(getattr(self, name))

    @property
    def is_control(self) -> bool:
        """这一行是不是「方法学对照」行（``split=by_token``）。

        对照行的数字**更高不代表更好**，报告里必须单独标出来，
        否则读者会把它当成一个更优的配置。
        """
        return self.split in ("by_token",)

    @property
    def applicable_properties(self) -> tuple[str, ...]:
        """这个 spec 能测哪些属性。

        ``sentence`` 池化把每个位置压成一个句向量，于是 ``form`` / ``role``
        这两个**逐位置**的属性不再有定义 —— 它们是「不适用」，不是「不满足」
        （见 skill 的坑 10）。报告里对它们留空并标注，绝不当成 0 分算进均值。
        """
        if self.pooling == "sentence":
            return ("topic",)
        return PROPERTIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            **{axis: self.axis(axis) for axis in OBSERVATION_AXES},
            "is_control": self.is_control,
            "applicable_properties": list(self.applicable_properties),
        }

    @property
    def digest(self) -> str:
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))


# ----------------------------------------------------------------------
# 曲线与探针
# ----------------------------------------------------------------------
@dataclass
class LayerCurve:
    """**本项目的核心产物**：一个属性在逐层上的线性可分性。

    ``accuracy[i]`` 是「用第 ``layers[i]`` 层的表征做线性探针」的测试准确率。
    ``majority`` 是同一划分下的多数类基线 —— 没有它，0.42 这个数字
    到底是「学到了」还是「数据本来就偏」根本说不清。

    ``onset_layer`` = 相邻层增幅最大的那一层，也就是「这个属性
    **开始变得可分**的层」。README 要求「可指认语义 / 语法分化的具体层」，
    这个字段就是那个「指认」。
    """

    spec_id: str
    property: str
    layers: list[int] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    #: 多数类基线（逐层相同，但存下来便于核对）
    majority: float = 0.0
    #: 类别数 —— chance level = 1 / n_classes
    n_classes: int = 0
    n_train: int = 0
    n_test: int = 0
    #: 只在这些位置上评测（如 ``role`` 会额外给出 ``role_amb`` 这一条）
    subset: str = "all"

    @property
    def chance(self) -> float:
        return 1.0 / self.n_classes if self.n_classes else 0.0

    @property
    def deep_gain(self) -> float:
        """最深层 − 最浅层。**「表征逐层形成」的量化指标**。

        本项目的头条数字就是它：如果一条曲线从头到尾是平的，
        说明这一层什么都没「形成」；有增益才说明层在做功。
        """
        if len(self.accuracy) < 2:
            return 0.0
        return self.accuracy[-1] - self.accuracy[0]

    def _max_jump(self) -> tuple[int, float]:
        """相邻层增幅最大的一段，返回 ``(accuracy 下标, 增幅)``。

        用严格大于比较，所以并列时取**更浅**的那一段。
        ``accuracy`` 不足两层时返回 ``(-1, 0.0)``。
        """
        if len(self.accuracy) < 2:
            return -1, 0.0
        best_i = 1
        best_d = self.accuracy[1] - self.accuracy[0]
        for i in range(2, len(self.accuracy)):
            d = self.accuracy[i] - self.accuracy[i - 1]
            if d > best_d:
                best_i, best_d = i, d
        return best_i, best_d

    @property
    def onset_layer(self) -> int:
        """增幅最大的层（从上一层到该层的跳变最大）。

        整条曲线是平的（所有相邻增幅相同）时没有「跳变」可言，
        此时返回 ``layers[1]``；不足两层则返回 ``layers[-1]``。
        """
        i, _ = self._max_jump()
        if i < 0:
            return self.layers[-1] if self.layers else -1
        return self.layers[i]

    @property
    def onset_jump(self) -> float:
        """分化层的跳变幅度（``onset_layer`` 相对上一层的增量）。"""
        return self._max_jump()[1]

    @property
    def best_layer(self) -> int:
        """准确率最高的层（并列取最浅 —— 越浅越省算力）。"""
        if not self.accuracy:
            return -1
        best = 0
        for i in range(1, len(self.accuracy)):
            if self.accuracy[i] > self.accuracy[best] + 1e-12:
                best = i
        return self.layers[best]

    def at(self, layer: int) -> float | None:
        for i, lyr in enumerate(self.layers):
            if lyr == layer:
                return self.accuracy[i]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "property": self.property,
            "subset": self.subset,
            "layers": list(self.layers),
            "accuracy": [round(a, 6) for a in self.accuracy],
            "majority": round(self.majority, 6),
            "chance": round(self.chance, 6),
            "n_classes": self.n_classes,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "deep_gain": round(self.deep_gain, 6),
            "onset_layer": self.onset_layer,
            "onset_jump": round(self.onset_jump, 6),
            "best_layer": self.best_layer,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayerCurve":
        return cls(
            spec_id=d["spec_id"],
            property=d["property"],
            layers=list(d["layers"]),
            accuracy=list(d["accuracy"]),
            majority=d["majority"],
            n_classes=d["n_classes"],
            n_train=d["n_train"],
            n_test=d["n_test"],
            subset=d.get("subset", "all"),
        )


@dataclass
class HeadStats:
    """一个注意力头的画像（Geometry Probe 的原料）。

    ``max_offset`` 是「这个头平均看多远」，``sink_weight`` 是「多少注意力
    落在主题标记词上」。这两个量是**可读的**，也是本项目
    「结构与表征如何耦合」这个问题的答案载体 ——
    若没有它们，``Geometry Probe`` 只能给出一堆相似度矩阵。
    """

    layer: int
    head: int
    entropy: float
    self_weight: float
    locality: float
    sink_weight: float
    mean_offset: float

    @property
    def node_id(self) -> str:
        return f"h{self.layer}.{self.head}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "layer": self.layer,
            "head": self.head,
            "entropy": round(self.entropy, 6),
            "self_weight": round(self.self_weight, 6),
            "locality": round(self.locality, 6),
            "sink_weight": round(self.sink_weight, 6),
            "mean_offset": round(self.mean_offset, 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HeadStats":
        """``to_dict`` 的反函数。

        注意 ``to_dict`` 多写了一个展示用的 ``node_id``，它不是构造参数 ——
        这里**显式按字段取值**而不是 ``HeadStats(**d)``，否则会因多出的
        ``node_id`` 抛 ``TypeError``。
        """
        return cls(
            layer=int(d["layer"]),
            head=int(d["head"]),
            entropy=float(d["entropy"]),
            self_weight=float(d["self_weight"]),
            locality=float(d["locality"]),
            sink_weight=float(d["sink_weight"]),
            mean_offset=float(d["mean_offset"]),
        )


@dataclass
class ProbeResult:
    """一类探针的结果。"""

    probe: str
    spec_id: str
    #: 标量指标（相似度探针的平均层间 CKA / 几何探针的图模块度 …）
    metrics: dict[str, float] = field(default_factory=dict)
    #: 结构化结果（逐层相似度矩阵 / 头聚类 / 图的节点边）
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "spec_id": self.spec_id,
            "metrics": {k: round(v, 6) for k, v in sorted(self.metrics.items())},
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProbeResult":
        return cls(
            probe=d["probe"],
            spec_id=d["spec_id"],
            metrics=dict(d.get("metrics", {})),
            detail=dict(d.get("detail", {})),
        )


@dataclass
class CaptureRun:
    """一个 spec 的完整观测结果。"""

    spec: StateSpec
    #: 属性 → 曲线（``role`` 会额外带一条 ``role_amb``，见 ``subset``）
    curves: dict[str, LayerCurve] = field(default_factory=dict)
    #: 探针名 → 结果
    probes: dict[str, ProbeResult] = field(default_factory=dict)
    #: 逐层逐头的注意力画像
    heads: list[HeadStats] = field(default_factory=list)
    #: 层 × 层的表征相似度矩阵（相似度探针的产物）
    layer_similarity: list[list[float]] = field(default_factory=list)
    #: 降维后的点（前若干个，供散点图）
    projection: list[dict[str, Any]] = field(default_factory=list)
    #: 方差解释率（PCA 才有）
    explained_variance: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def spec_id(self) -> str:
        return self.spec.spec_id

    @property
    def model(self) -> str:
        return self.spec.model

    def axis(self, name: str) -> str:
        return self.spec.axis(name)

    def curve(self, prop: str, subset: str = "all") -> LayerCurve | None:
        """按属性取曲线；``subset`` 用于区分 ``role`` 与 ``role_amb``。"""
        for c in self.curves.values():
            if c.property == prop and c.subset == subset:
                return c
        return None

    def headline(self) -> float:
        """头条数字 = 全部**适用**属性上「深层 − 浅层」的最大值。

        取 ``max`` 而不是 ``mean``：``form`` 这类属性在 embedding 层就已经
        完全可分（它本来就在词表里），增益恒为 0，用均值会把它当成分母，
        把真正在「逐层形成」的 ``role`` / ``topic`` 稀释掉。
        """
        gains = [c.deep_gain for c in self.curves.values()]
        return max(gains) if gains else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "spec_id": self.spec_id,
            "headline": round(self.headline(), 6),
            "curves": {k: c.to_dict() for k, c in sorted(self.curves.items())},
            "probes": {k: p.to_dict() for k, p in sorted(self.probes.items())},
            "heads": [h.to_dict() for h in self.heads],
            "meta": dict(sorted(self.meta.items())),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaptureRun":
        spec = StateSpec.from_spec_id(d["spec_id"])
        # curves.json 的键已经是 property/subset，直接用
        curves = {k: LayerCurve.from_dict(v) for k, v in d.get("curves", {}).items()}
        probes = {k: ProbeResult.from_dict(v) for k, v in d.get("probes", {}).items()}
        return cls(
            spec=spec,
            curves=curves,
            probes=probes,
            heads=[HeadStats.from_dict(h) for h in d.get("heads", [])],
        )


# ----------------------------------------------------------------------
# 版本
# ----------------------------------------------------------------------
@dataclass
class Version:
    """内容寻址版本：digest 由**被观测对象的定义**决定。

    口径与同工作区其它项目一致 —— 这里「内容」是标本本身
    （模型结构 + 语料 + 真值口径），而不是观测配置。观测配置换了
    只该产生新的行，不该产生新的版本号。
    """

    version_id: str
    created_at: str
    digest: str
    tag: str
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "digest": self.digest,
            "tag": self.tag,
            "counts": dict(sorted(self.counts.items())),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Version":
        return cls(
            version_id=d["version_id"],
            created_at=d["created_at"],
            digest=d["digest"],
            tag=d["tag"],
            counts=dict(d.get("counts", {})),
        )


# ----------------------------------------------------------------------
# 流水线产物
# ----------------------------------------------------------------------
@dataclass
class LabResult:
    """整条流水线的产物。"""

    corpus: CorpusStats
    runs: list[CaptureRun] = field(default_factory=list)
    #: 对比矩阵的行（每个 spec 一行），由 compare 模块生成
    matrix: list[dict[str, Any]] = field(default_factory=list)
    #: 单变量消融表 + 归因，由 compare 模块生成
    ablations: dict[str, Any] = field(default_factory=dict)
    #: 结论：分化层定位 + 随机对照 + 方法学对照
    findings: dict[str, Any] = field(default_factory=dict)
    #: 最终入选的 spec_id
    selected: str = ""
    dag: Any = None
    version: Version | None = None
    counters: dict[str, Any] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)

    # -- 便捷查询 ----------------------------------------------------
    def find(self, spec_id: str) -> CaptureRun | None:
        for r in self.runs:
            if r.spec_id == spec_id:
                return r
        return None

    def axis_values(self, axis: str) -> list[str]:
        return sorted({r.axis(axis) for r in self.runs})

    def axes(self) -> list[str]:
        return [a for a in OBSERVATION_AXES if self.axis_values(a)]

    def by_model(self, model: str) -> list[CaptureRun]:
        return [r for r in self.runs if r.model == model]

    def curve_of(self, spec_id: str, prop: str, subset: str = "all") -> LayerCurve | None:
        run = self.find(spec_id)
        return run.curve(prop, subset) if run else None

    def known_spec_ids(self) -> list[str]:
        return sorted(r.spec_id for r in self.runs)

    def best_by(self, field_name: str, direction: str = "max") -> CaptureRun | None:
        if not self.runs:
            return None
        key = lambda r: r.headline() if field_name == "headline" else float(r.meta.get(field_name, 0.0))  # noqa: E731
        return (min if direction == "min" else max)(self.runs, key=key)

    def reference_run(self) -> CaptureRun | None:
        """参考配置对应的 run（消融表的基准行）。

        解析顺序：``selected``（落盘/加载时由 ``findings.reference_spec``
        回填）→ ``cfg.ablation.reference``（内存跑流水线时）→
        ``findings.reference_spec``。三者任选其一能定位到 run 即可，
        这样「从磁盘加载的只读命令」也能正确拿到参考配置。
        """
        if self.selected:
            r = self.find(self.selected)
            if r is not None:
                return r
        ref = (self.cfg or {}).get("ablation", {}).get("reference")
        if ref:
            r = self.find(str(ref))
            if r is not None:
                return r
        ref_spec = (self.findings or {}).get("reference_spec")
        if ref_spec:
            return self.find(str(ref_spec))
        return None
