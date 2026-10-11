"""语料与**已知真值**。

本项目是显微镜，所以第一件事是**标定标本**：被观测的东西必须有一份
「结构已知」的真值，否则曲线只是漂亮的折线图。

三层真值（README 第五节的「Representation 是怎么逐层形成的」就建立在这三层上）：

| 属性 | 含义 | 设计上应该在第几层可分 |
| --- | --- | --- |
| ``form``  | 词形（词表里的哪个词） | **浅层**：它本来就在 embedding 里 |
| ``role``  | 句法角色（DET / ADJ / NOUN / VERB / PREP / PUNCT） | **中层**：要由 FFN 查表写出来 |
| ``topic`` | 语义主题（tech / nature / finance / food） | **深层**：要先把标记词广播到全句 |

三个关键设计（都不是随手写的，每一个都对应曲线上一个可核对的形状）
------------------------------------------------------------------------

1. **词形码是稀疏 ±1 随机码，不是 one-hot。**
   于是「词形 → 角色」这个映射**不是线性可分的** —— 线性探针在 embedding 层
   读不出角色。只有当某一层的 FFN 把角色**写成 one-hot** 之后，探针才读得到。
   这就是「中层句法」这条曲线的机制：它量的不是「模型有没有角色信息」，
   而是「角色有没有被**物化**成一个线性可读的方向」。

2. **主题只由「标记词」决定，而内容词跨主题共享。**
   如果每个主题有自己的内容词，那么主题从 embedding 层就可分了
   （词形码直接泄漏主题），深层语义这条曲线就永远不会出现。
   所以：内容词 4 个主题共用；每个主题有一个专属标记词（``芯片`` / ``森林`` /
   ``利率`` / ``香料``），标记词位置在句内**随机**（否则会被位置而非内容识别）。
   于是「主题」只能靠**把标记词的信息广播到全句**才能得到 —— 而那需要全局注意力。

3. **歧义词的角色由 (词形, 主题) 共同决定。**
   ``设计`` 在技术句里是动词、在饮食句里是名词。所以它的角色需要
   **全局上下文**才能确定，而普通词只需要自己的词形。
   把这两类分开统计（``role`` 的 ``all`` / ``unambiguous`` / ``ambiguous``
   三条子曲线），就能直接看到「句法分化的那一层」到底是哪一层。

   注意：歧义词在**全部 4 个主题**里都出现，且各自都有角色 ——
   如果它只出现在两个主题里，「看到词形就能猜一半主题」会污染 topic 曲线。

句子由「词典 + 模板 + 种子」**确定性生成**，不手写。
手写 80 句只会让真值更容易写错，而且改一个词就要重抄一遍。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import load_jsonl, resolve_data_path
from .rng import make_rng
from .schema import CorpusStats, Sentence, TokenRecord, sha256_text

#: 句子的角色模板。**这是「句子」这个概念在本项目里的全部含义** ——
#: 一个角色序列。本项目不做句法分析，模板只是为了让角色分布像一句话。
#:
#: **没有 PUNCT**：词表里一个角色必须有足够多的词形，而「角色在浅层线性不可分」
#: 的前提是「该角色的词形数 > 词形码维数」（见 config 里 code_dim 的说明）。
#: 标点只有三四个词形，永远满足不了 —— 它会在 embedding 层就被探针读出来，
#: 把 role 曲线的基线抬高一截，反而让「中层才分化」这个结论变模糊。
#: 七个槽位也让多数类基线降到 0.25（NOUN / ADJ / VERB 各 2 个槽）。
ROLE_TEMPLATE: tuple[str, ...] = ("DET", "ADJ", "NOUN", "VERB", "PREP", "ADJ", "VERB")

#: 哪些槽位允许被歧义词填充（概率见配置外的 ``AMBIGUOUS_SLOT_RATE``）
AMBIGUOUS_SLOT_ROLES: tuple[str, ...] = ("ADJ", "NOUN", "VERB")

#: 每个可用槽位被歧义词占据的概率。
#:
#: 这个概率直接决定「歧义词 vs 通用词」的槽位分配，而槽位分配又决定
#: ``form`` 曲线会不会因为「某类训练样本 < min_class_count」被整条跳过。
#: 实测（320 句 / 217 词形）：
#:   0.55 → 歧义位置 879（34%），通用 ADJ/NOUN/VERB 每词形仅 2-5 次，
#:          50 个词形训练样本 <3，form 曲线被跳过；
#:   0.30 → 歧义位置 480（19%），通用词每词形 7-10 次，
#:          训练样本 <3 的词形 <2%，form 曲线保留。
#: ``role_amb`` 测试集在 0.30 下仍有 168 个位置，足够稳定。
AMBIGUOUS_SLOT_RATE = 0.30


@dataclass
class Lexicon:
    """词典 —— 全部真值的来源。

    ``form_ids`` 是**词表序**（按词形排序后的下标），它决定 embedding 的行号。
    排序而不是按文件顺序：文件顺序变了词表就变了，而词表是 digest 的一部分。
    """

    topics: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    #: 通用词形 → 唯一角色
    generic: dict[str, str] = field(default_factory=dict)
    #: 标记词形 → 它代表的主题
    markers: dict[str, str] = field(default_factory=dict)
    #: 标记词形 → 它自己的角色（标记词也是句中的一个词，也有角色）
    marker_roles: dict[str, str] = field(default_factory=dict)
    #: 歧义词形 → {主题: 角色}（**必须覆盖全部主题**，见模块文档第 3 点）
    ambiguous: dict[str, dict[str, str]] = field(default_factory=dict)
    form_ids: dict[str, int] = field(default_factory=dict)

    @property
    def forms(self) -> list[str]:
        return sorted(self.form_ids, key=lambda f: self.form_ids[f])

    @property
    def n_forms(self) -> int:
        return len(self.form_ids)

    @property
    def ambiguous_forms(self) -> list[str]:
        return sorted(self.ambiguous)

    @property
    def marker_forms(self) -> list[str]:
        return sorted(self.markers)

    def is_ambiguous(self, form: str) -> bool:
        return form in self.ambiguous

    def is_marker(self, form: str) -> bool:
        return form in self.markers

    def role_of(self, form: str, topic: str) -> str:
        """在给定主题下，这个词形的句法角色。**这就是真值的定义**。"""
        if form in self.ambiguous:
            try:
                return self.ambiguous[form][topic]
            except KeyError as exc:  # pragma: no cover - 由 load_lexicon 挡住
                raise KeyError(
                    f"歧义词 {form!r} 没有为主题 {topic!r} 定义角色"
                ) from exc
        if form in self.markers:
            return self.marker_roles[form]
        if form in self.generic:
            return self.generic[form]
        raise KeyError(f"词典里没有词形 {form!r}")

    def digest(self) -> str:
        payload = "\n".join(
            [
                "topics:" + ",".join(sorted(self.topics)),
                "roles:" + ",".join(sorted(self.roles)),
                *[f"g:{f}={r}" for f, r in sorted(self.generic.items())],
                *[
                    f"m:{f}={t}/{self.marker_roles[f]}"
                    for f, t in sorted(self.markers.items())
                ],
                *[
                    f"a:{f}=" + ",".join(f"{t}:{r}" for t, r in sorted(m.items()))
                    for f, m in sorted(self.ambiguous.items())
                ],
            ]
        )
        return sha256_text(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_forms": self.n_forms,
            "n_generic": len(self.generic),
            "n_markers": len(self.markers),
            "n_ambiguous": len(self.ambiguous),
            "topics": sorted(self.topics),
            "roles": sorted(self.roles),
            "ambiguous_forms": self.ambiguous_forms,
            "marker_forms": self.marker_forms,
            "digest": self.digest(),
        }


@dataclass
class Corpus:
    """语料 + 词典。"""

    sentences: list[Sentence]
    lexicon: Lexicon
    #: 是不是从词典重新生成的（而不是读的落盘句子文件）
    rebuilt: bool = False

    def positions(self) -> list[TokenRecord]:
        """全部位置，按 (句子 id, 句内下标) 排序。

        这个顺序是**观测的顺序**：``capture`` 抓出来的向量表与它一一对应，
        探针的标签向量也从它来。顺序不稳定会让「第 37 个位置」指向别的东西，
        而所有依赖下标的东西（散点图、逐位置样本）都会静默错位。
        """
        out: list[TokenRecord] = []
        for s in sorted(self.sentences, key=lambda x: x.sentence_id):
            out.extend(s.tokens)
        return out

    def topic_of(self, sentence_id: str) -> str:
        for s in self.sentences:
            if s.sentence_id == sentence_id:
                return s.topic
        raise KeyError(sentence_id)

    def stats(self, name: str) -> CorpusStats:
        positions = self.positions()
        topics: dict[str, int] = {}
        roles: dict[str, int] = {}
        forms: dict[str, int] = {}
        n_amb = 0
        n_marker = 0
        for p in positions:
            roles[p.role] = roles.get(p.role, 0) + 1
            forms[p.form] = forms.get(p.form, 0) + 1
            n_amb += int(p.is_ambiguous)
            n_marker += int(p.is_marker)
        for s in self.sentences:
            topics[s.topic] = topics.get(s.topic, 0) + 1
        top = dict(sorted(forms.items(), key=lambda kv: (-kv[1], kv[0]))[:10])
        mean_len = (
            sum(s.n_tokens for s in self.sentences) / len(self.sentences)
            if self.sentences
            else 0.0
        )
        return CorpusStats(
            name=name,
            n_sentences=len(self.sentences),
            n_tokens=len(positions),
            n_forms=self.lexicon.n_forms,
            n_roles=len(self.lexicon.roles),
            n_topics=len(self.lexicon.topics),
            n_ambiguous_tokens=n_amb,
            n_marker_tokens=n_marker,
            mean_sentence_len=mean_len,
            topic_sizes=topics,
            role_sizes=roles,
            top_forms=top,
        )


# ----------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------
def load_lexicon(cfg: dict[str, Any]) -> Lexicon:
    """读词典并**把全部结构性问题挡在入口**。

    词典是唯一的手写真值来源，写错了整条流水线的结论都错 ——
    所以这里校验得比别处狠：
    - 每个主题必须有且只有一个标记词；
    - 歧义词必须为**全部主题**定义角色（否则「看词形猜主题」会污染 topic 曲线）；
    - 通用词的词形不能与标记词/歧义词重名（重名会让 role_of 的结果依赖分支顺序）。
    """
    rows = load_jsonl(resolve_data_path(cfg, "lexicon"))
    if not rows:
        raise FileNotFoundError(
            f"词典为空：{resolve_data_path(cfg, 'lexicon')}"
        )
    meta = next((r for r in rows if r.get("kind") == "meta"), None)
    if meta is None:
        raise ValueError("词典缺少 kind=meta 的首行（它声明 topics / roles）")
    topics = [str(t) for t in meta["topics"]]
    roles = [str(r) for r in meta["roles"]]
    if len(set(topics)) != len(topics):
        raise ValueError(f"topics 有重复：{topics}")
    if len(set(roles)) != len(roles):
        raise ValueError(f"roles 有重复：{roles}")

    generic: dict[str, str] = {}
    markers: dict[str, str] = {}
    marker_roles: dict[str, str] = {}
    ambiguous: dict[str, dict[str, str]] = {}
    for row in rows:
        kind = row.get("kind")
        if kind == "meta":
            continue
        form = str(row["form"])
        if kind == "generic":
            role = str(row["role"])
            if role not in roles:
                raise ValueError(f"通用词 {form!r} 的角色 {role!r} 不在 roles 里")
            generic[form] = role
        elif kind == "marker":
            topic = str(row["topic"])
            role = str(row["role"])
            if topic not in topics:
                raise ValueError(f"标记词 {form!r} 的主题 {topic!r} 不在 topics 里")
            if role not in roles:
                raise ValueError(f"标记词 {form!r} 的角色 {role!r} 不在 roles 里")
            markers[form] = topic
            marker_roles[form] = role
        elif kind == "ambiguous":
            mapping = {str(k): str(v) for k, v in row["roles_by_topic"].items()}
            missing = [t for t in topics if t not in mapping]
            if missing:
                raise ValueError(
                    f"歧义词 {form!r} 缺少主题 {missing} 的角色定义 ——"
                    "缺了会让「看到词形就能猜主题」，污染 topic 曲线"
                )
            for role in mapping.values():
                if role not in roles:
                    raise ValueError(f"歧义词 {form!r} 的角色 {role!r} 不在 roles 里")
            ambiguous[form] = mapping
        else:
            raise ValueError(f"未知的词典条目类型 {kind!r}（form={form!r}）")

    overlap = (
        set(generic) & set(markers)
        | set(generic) & set(ambiguous)
        | set(markers) & set(ambiguous)
    )
    if overlap:
        raise ValueError(f"词形在多个类别里重名：{sorted(overlap)}")
    covered = {t for t in markers.values()}
    if covered != set(topics):
        raise ValueError(
            f"每个主题必须有且只有一个标记词；已覆盖 {sorted(covered)}，"
            f"缺 {sorted(set(topics) - covered)}"
        )
    if not generic:
        raise ValueError("词典里没有通用词")
    if not ambiguous:
        raise ValueError("词典里没有歧义词 —— 没有它就观察不到「句法分化」")

    all_forms = sorted(set(generic) | set(markers) | set(ambiguous))
    lexicon = Lexicon(
        topics=sorted(topics),
        roles=sorted(roles),
        generic=generic,
        markers=markers,
        marker_roles=marker_roles,
        ambiguous=ambiguous,
        form_ids={f: i for i, f in enumerate(all_forms)},
    )
    # 模板里出现的角色必须都在词典里有词可用
    for role in ROLE_TEMPLATE:
        if role not in roles:
            raise ValueError(f"模板要求角色 {role!r}，但词典的 roles 里没有")
    return lexicon


# ----------------------------------------------------------------------
# 生成
# ----------------------------------------------------------------------
def _forms_with_role(lexicon: Lexicon, role: str, topic: str) -> list[str]:
    """在给定主题下角色为 ``role`` 的全部词形（通用词 + 歧义词）。"""
    out = [f for f, r in lexicon.generic.items() if r == role]
    out += [f for f, m in lexicon.ambiguous.items() if m[topic] == role]
    return sorted(out)


def build_sentences(lexicon: Lexicon, cfg: dict[str, Any]) -> list[Sentence]:
    """由词典 + 模板 + 种子确定性生成句子。

    确定性来自 ``make_rng("sentences", seed, sentence_index)`` ——
    **每句一个独立 rng**，而不是共用一个流。共用一个流时，
    改一句的生成逻辑会让后面所有句子全部变化，diff 里看不出改了什么。
    """
    n_sentences = int(cfg["corpus"].get("n_sentences", 80))
    seed = int(cfg["seed"])
    topics = lexicon.topics
    sentences: list[Sentence] = []
    for si in range(n_sentences):
        rng = make_rng("sentences", seed, si)
        topic = topics[si % len(topics)]  # 轮流，保证各主题句数均衡
        marker_form = next(f for f, t in sorted(lexicon.markers.items()) if t == topic)

        # 1) 按模板逐槽填词
        slots: list[str] = []
        for role in ROLE_TEMPLATE:
            candidates = _forms_with_role(lexicon, role, topic)
            ambiguous_here = [
                f for f in candidates if lexicon.is_ambiguous(f)
            ]
            generic_here = [f for f in candidates if not lexicon.is_ambiguous(f)]
            use_ambiguous = (
                role in AMBIGUOUS_SLOT_ROLES
                and ambiguous_here
                and rng.random() < AMBIGUOUS_SLOT_RATE
            )
            pool = ambiguous_here if use_ambiguous else (generic_here or candidates)
            slots.append(pool[rng.randrange(len(pool))])

        # 2) 把标记词插到一个**随机**位置。
        #    固定位置会让「主题」被位置而不是内容识别 —— 那这条曲线就没意义了。
        marker_index = rng.randrange(len(slots) + 1)
        slots.insert(marker_index, marker_form)

        # 3) 落真值
        sid = f"s{si:03d}"
        tokens: list[TokenRecord] = []
        for i, form in enumerate(slots):
            is_marker = form == marker_form
            is_amb = lexicon.is_ambiguous(form)
            tokens.append(
                TokenRecord(
                    sentence_id=sid,
                    index=i,
                    form=form,
                    role=lexicon.role_of(form, topic),
                    topic=topic,
                    is_ambiguous=is_amb,
                    is_marker=is_marker,
                )
            )
        sentences.append(
            Sentence(
                sentence_id=sid,
                text=" ".join(slots),
                topic=topic,
                tokens=tokens,
                marker_index=marker_index,
            )
        )
    return sentences


def corpus_to_rows(corpus: Corpus) -> list[dict[str, Any]]:
    """落盘格式：第一行是 meta（含词典 digest），其余每行一句。"""
    rows: list[dict[str, Any]] = [
        {
            "kind": "meta",
            "lexicon_digest": corpus.lexicon.digest(),
            "n_sentences": len(corpus.sentences),
            "topics": sorted(corpus.lexicon.topics),
            "roles": sorted(corpus.lexicon.roles),
        }
    ]
    rows.extend(s.to_dict() for s in corpus.sentences)
    return rows


def corpus_from_rows(rows: list[dict[str, Any]], lexicon: Lexicon) -> Corpus:
    sentences: list[Sentence] = []
    for row in rows:
        if row.get("kind") == "meta":
            continue
        tokens = [
            TokenRecord(
                sentence_id=str(t["sentence_id"]),
                index=int(t["index"]),
                form=str(t["form"]),
                role=str(t["role"]),
                topic=str(t["topic"]),
                is_ambiguous=bool(t.get("is_ambiguous", False)),
                is_marker=bool(t.get("is_marker", False)),
            )
            for t in row["tokens"]
        ]
        sentences.append(
            Sentence(
                sentence_id=str(row["sentence_id"]),
                text=str(row["text"]),
                topic=str(row["topic"]),
                tokens=tokens,
                marker_index=int(row.get("marker_index", -1)),
            )
        )
    return Corpus(sentences=sentences, lexicon=lexicon, rebuilt=False)


def load_corpus(cfg: dict[str, Any]) -> Corpus:
    """读语料。

    落盘文件存在**且词典 digest 一致**时直接读；否则用词典重新生成。

    为什么要存 digest 而不是「文件在就用」：改一个词形、换一个主题，
    落盘句子就**静默过期**了 —— 而它带着旧的 ``role`` 真值，
    于是「探针能不能复原真值」这个核心检查会在错误的真值上通过。
    这类错误不会报错，只会让结论悄悄变成假的。
    """
    lexicon = load_lexicon(cfg)
    path = resolve_data_path(cfg, "sentences")
    rows = load_jsonl(path)
    if rows:
        meta = next((r for r in rows if r.get("kind") == "meta"), None)
        if meta and meta.get("lexicon_digest") == lexicon.digest():
            return corpus_from_rows(rows, lexicon)
    return Corpus(sentences=build_sentences(lexicon, cfg), lexicon=lexicon, rebuilt=True)


def write_corpus(corpus: Corpus, path: str | Path) -> None:
    """把生成的语料落盘（供人查阅与复现）。

    用 ``json.dumps(..., ensure_ascii=False)`` —— 中文词形必须原样写出来，
    否则产物不可读，而「可读」正是这个项目存在的理由。
    """
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in corpus_to_rows(corpus):
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
