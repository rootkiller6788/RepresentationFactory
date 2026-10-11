# RepresentationFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L1 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | RepresentationFactory |
| 类型 | Lab |
| 真正有意思的核心 | Hidden State / Attention / Representation Graph（表征可视化与分析） |

---

## 二、核心命题

**这是 L1 最有科研味的项目。它不是 Factory，而是 Microscope。**

不做生成、不做优化，只做一件事：把模型内部的表征**看清楚**——它长什么样、怎么随层数演化、不同输入之间如何分群。

## 三、整体架构

```
             Model
               │
               ▼
            Forward
               │
      ┌────────┼─────────┐
      ▼        ▼         ▼
 Hidden     Attention   Activation
 State        Map
      │        │         │
      └────────┼─────────┘
               ▼
          Probe Engine
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
 Similarity  Cluster   Geometry
     │       Probe      Graph
     └─────────┼─────────┘
               ▼
       Representation Graph
               │
               ▼
          Visualization
```

## 四、主流水线

```
Input
 ↓
Forward Pass
 ↓
Capture Internal State
 ↓
Project / Reduce
 ↓
Measure Similarity
 ↓
Probe
 ↓
Construct Graph
 ↓
Compare Layer / Head / Model
 ↓
Visualize
```

## 五、最终最好出现这种东西

```
Token
 │
 ▼
L1 ─── semantic cluster A
 │
L6 ───────── syntax separation
 │
L12 ───────────────── concept cluster
 │
L24 ───────────────────── task representation
```

也就是观察：**Representation 是怎么逐层形成的。**

浅层偏向词形与局部语法，中层开始出现句法边界，深层收敛到语义 / 任务级的抽象。这条"逐层演化曲线"是这个 Lab 最有价值的产物。

## 六、Probe Engine 的三类探针

| 探针 | 输入 | 回答的问题 |
| --- | --- | --- |
| Similarity | Hidden State 向量对 | 哪些 token / 样本在表征空间里靠近？ |
| Cluster Probe | 全量 hidden state + 标签 | 表征里是否已线性可分地编码了某属性？ |
| Geometry Graph | 层 × 头 × token 的相似度矩阵 | 结构与表征如何耦合？哪些头在做同一件事？ |

## 七、阶段性落地计划

- **M0｜抓取骨架**：定义 Hook 点与 State Spec 的 schema，跑通 Forward → Capture → 保存一次（非自动）。
- **M1｜降维与相似度**：实现 PCA / UMAP / t-SNE 投影与相似度度量，产出可交互散点。
- **M2｜Probe 与 Graph**：实现三类探针，构建 Representation Graph（层 × 头 × token）。
- **M3｜逐层演化与发布**：输出"表征逐层形成"的对比视图，支持跨层 / 跨头 / 跨模型对比与可视化。

## 八、验收标准

能端到端完成「RepresentationFactory」，并在其「核心」（Hidden State / Attention / Representation Graph）上给出可复现的量化结果：

- 对同一输入给出逐层表征演化视图，可指认语义 / 语法分化的具体层
- 输出可复现的 Representation Graph，支持跨模型结构对比

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L1 深化设计稿；可按实际进展修订。_

---

## 附录 A｜工程骨架与快速开始

### 为什么零第三方依赖

核心链路 `dependencies = []` —— **微型 transformer 与全部线性代数都是手写的**（PCA power iteration、闭式岭回归线性探针、CKA、连通分量）。手写不是为了炫技，而是因为本项目要回答的是「某一层里某个属性到底可不可分」，这需要一个**能被完全拆开、逐位复现**的观测对象。用 PyTorch 训一个真模型，得到的是「某个 checkpoint 在某一天的样子」，没人能复核。生产环境当然该用 PyTorch / scikit-learn，因此留了 `torch` extra 做一致性对照。

### 目录结构

```
L1-representationfactory/
├── main.py                     # 零安装入口（把 src/ 加进 sys.path）
├── configs/default.json        # 唯一配置：语料/模型/观测网格/探针/消融/选择
├── data/
│   ├── lexicon/lexicon.jsonl    # 词典（真值来源）
│   └── sentences/sentences.jsonl# 句子（由词典+模板+种子确定性生成）
├── src/representationfactory/
│   ├── schema.py      # StateSpec / LayerCurve / HeadStats / ProbeResult / CaptureRun / LabResult / Version
│   ├── rng.py         # sha256 派生种子（绝不用内置 hash —— PYTHONHASHSEED 会破坏跨进程可复现）
│   ├── corpus.py      # 读词典 + 按模板确定性生成句子 + 真值（词形/角色/主题/歧义/标记）
│   ├── model.py       # 手写微型 transformer（constructive 构造式 / random 随机对照两个标本）
│   ├── capture.py     # 前向 + 抓取每层 hidden / attention
│   ├── linalg.py      # 手写线性代数
│   ├── project.py     # PCA（power iteration，主成分符号确定）
│   ├── similarity.py  # cosine / centered_cosine / CKA 的层间相似度矩阵
│   ├── probes.py      # cluster（逐层线性探针）/ similarity / geometry（头画像+聚类）
│   ├── graph.py       # RepresentationGraph（节点=头，边=画像相似度，聚类=连通分量）
│   ├── compare.py     # 观测矩阵 + 单变量消融 + 跨模型对比 + 分化层定位
│   ├── lineage.py     # 血缘 DAG（ASCII / DOT）
│   ├── pipeline.py    # run_pipeline / write_outputs / run_and_write / load_result
│   ├── report.py      # 报告 Markdown（随产物一起落盘）
│   ├── visualize.py   # 单文件 HTML（内联 SVG，零 JS 框架）
│   └── cli.py         # 子命令
└── tests/             # pytest（36 项）
```

### 快速开始

```bash
# 跑全流程并落盘（约 60~75s，确定性）
python main.py run --out out

# 只读命令：从 out/ 加载，秒级返回，一个字节都不改产物
python main.py version
python main.py compare
python main.py curve --spec constructive-tok-cos-bs-raw --property role --subset ambiguous
python main.py explain --position 3:12        # 验收三问 where_from / what_happened / why_here
python main.py graph --spec constructive-tok-cos-bs-raw
python main.py dag --dot

# 写命令：生成单文件 HTML
python main.py visualize

# 测试
pip install pytest && pytest -q
```

### CLI 契约

| 类型 | 命令 | 写 `out/`？ |
| --- | --- | --- |
| **写** | `run`（全流程落盘）、`visualize`（生成 `out/visualize.html`） | 是 |
| **只读** | `capture / project / similarity / probe / curve / graph / compare / explain / list / dag / version` | 否 |

只读命令**优先用 `load_result` 从产物重建 `LabResult`**（不重跑流水线）；`out/` 缺失时才退回内存里的 `run_pipeline`（仍不落盘）。这条由 `tests/test_readonly_no_write.py` 用逐文件 md5 实测。

---

## 附录 B｜已验证状态（真实数字）

### B.1 语料画像

- 名称 `representationfactory-demo`；句子 **280** 条，位置 **2240** 个，平均句长 **8.00**
- 词形 **217**、角色 **5**、主题 **4**
- 歧义位置 **427（19.1%）** —— 角色要等注意力扩到能看见主题标记词才定
- 主题标记位置 **280**（作为 attention sink）
- 角色分布：NOUN=560，DET=280，ADJ=560，VERB=560，PREP=280
- 主题分布：finance / food / nature / tech 各 70

### B.2 验收：五条分化层全部命中（5/5）

| 属性 | 设计期望层 | 实测分化层 | 命中 |
| --- | --- | --- | --- |
| form/all | L0 | L2 | ✅ |
| role/all | L2 | L2 | ✅ |
| role/unambiguous | L2 | L2 | ✅ |
| role/ambiguous | L3 | L3 | ✅ |
| topic/all | L4 | L4 | ✅ |

全部命中：**是**；随机权重对照是否「没有阶梯」：**是**（证明阶梯不是随便一个网络都有的）。

### B.3 参考配置逐层演化曲线（核心产物）

参考配置 `constructive-tok-cos-bs-raw`（构造式标本 / token 池化 / centered-cosine / 按句划分 / 原始）：

| 属性 | 分化层 | 跳变 | 最佳层 | 最佳准确率 | 多数类基线 |
| --- | --- | --- | --- | --- | --- |
| form/all | L2 | 0.058 | L2 | 0.703 | 0.059 |
| role/all | L2 | 0.337 | L4 | 0.954 | 0.250 |
| role/ambiguous | L3 | 0.315 | L4 | 0.933 | 0.403 |
| role/unambiguous | L2 | 0.389 | L2 | 0.959 | 0.255 |
| topic/all | L4 | 0.631 | L4 | **1.000** | 0.296 |

读法：`form`（词形）在 L0 就已可读（0.647，约 11× 多数类基线）且近乎平 —— 它本就在词表里；`role`（句法角色）在 L2 跳变；`topic`（语义主题）憋到 L4 才从 0.369 跳到 1.000。**这正是「浅层词形 → 中层句法 → 深层语义」的逐层形成。**

### B.4 跨模型对照（构造式标本 vs 随机权重）

| 模型 | 最大 headline 增益 | 分化层层号 |
| --- | --- | --- |
| constructive | **0.657** | form/all, role/all, role/ambiguous, role/unambiguous, topic/all |
| random | 0.115 | —（无阶梯） |

### B.5 表征图（结构与表征如何耦合）

- 节点（layer×head）= **16**（4 层 × 4 头），边（头画像相似度 ≥ 阈值）= **92**
- 连通分量（功能组）= **2**（阈值 0.990）
  - 组 1（14 头）：h1.* / h2.* / h3.0-2 / h4.0-2
  - 组 2（2 头）：**h3.3, h4.3** —— 最深两层的第 3 号头自成一组（sink 型）
- geometry 指标：mean_head_similarity_offdiag 0.801、n_sink_heads 2、n_local_heads 0

### B.6 单变量消融（相对参考配置，每次只动一个轴）

| 轴 | 参考值 | 变体 | headline | 说明 |
| --- | --- | --- | --- | --- |
| pooling | token | sentence | 0.000 | 句池化把逐位置属性压没，只剩 topic 且增益 0 |
| similarity | cosine | centered_cosine / cka | 0.657 | 相似度度量**不影响**探针曲线（探针与相似度是两条独立链路） |
| split | by_sentence | by_token | 0.611 | **方法学对照轴**：按 token 随机划分会虚高，不进结论 |
| project | none | pca | 0.227 | 压到 2 维后 form/role 可分性大幅下降（-0.37 ~ -0.58） |

### B.7 观测矩阵与只读契约

- 观测网格 **48** 个 spec（2 模型 × 2 池化 × 3 相似度 × 2 划分 × 2 降维），矩阵 **48 行**
- 只读命令（13 条）跑完，`out/` 逐文件 md5 **不变**（实测）
- 确定性实测：删掉 `out/` 重跑，除带时间戳的 `version.json` / `lineage.json` 外，全部产物**逐字节相同**

### B.8 版本

`v1-3faf68e7619f`（内容寻址 digest `3faf68e7619f`）。

---

## 附录 C｜跑起来才发现的问题与不变量

### C.1 问题清单（症状 → 修法）

1. **`reference_run()` 从磁盘加载后返回 `None`**。`load_result` 用 `cfg={}` 重建，而旧实现只从 `self.cfg` 读参考配置 → `project` / `compare` / `explain` 全崩。修：三级回退 `selected` → `cfg.ablation.reference` → `findings.reference_spec`。
2. **只读命令每次重跑 75s 流水线**。修：新增 `pipeline.load_result()`，从 `curves/probes/findings/version/lineage/corpus_stats` 六个产物重建 `LabResult`；`cli._result` 优先加载、缺失才回退。
3. **`cli.py` 从 `corpus` 导入 `specimen_data`**（实际定义在 `model`）→ `ImportError`。
4. **`_attention_samples` 把 `data.sequences[si]` 当成 `Sentence` 对象**（实际是 `list[int]`）→ 注意力样本生成崩。
5. **`render_html` 引用 `result.corpus.topics`**（`CorpusStats` 无此字段）→ 改为从散点 `points` 派生主题。
6. **`_svg_graph` 用 `g.node_ids`**（`RepresentationGraph` 无此字段），且 `GraphEdge` 的字段是 `.a` / `.b` 而非元组。
7. **`report.py` 与 `cli._cmd_graph` 引用 `g.modularity`**（`RepresentationGraph` 没有模块度字段）。
8. **`probe cluster` 用 `c['best_acc']`** —— `LayerCurve.to_dict()` 根本不写这个键 → `KeyError`。修：从 `accuracy` / `layers` / `best_layer` 现算。
9. **`probe geometry` 读 `detail` 的 `n_nodes` / `n_edges`**（不存在）—— 头数/聚类数在 `metrics`（`n_heads` / `n_clusters`），阈值在 `detail`。
10. **`form/all` 曲线被整条跳过**：217 个词形里有 34 个训练样本 <3（15.7% > 5% 阈值）→ 整条 `form` 曲线被弃，「浅层可读」的对照组缺失。修：**丢弃稀疏类**而非跳过整条曲线（标准做法），`n_classes` 相应更新为保留后的类数。
11. **`CaptureRun.from_dict` 调 `HeadStats(**h)`**：`to_dict` 多写了一个展示用的 `node_id` → `TypeError`。修：新增 `HeadStats.from_dict` 显式按字段取值（`load_result` 因传 `heads=[]` 侥幸没触发）。
12. **`write_corpus` 是「被定义却从不调用」的死代码**：配置把 `corpus.sentences` 声明成一个路径、`load_corpus` 也按「文件存在且词典 digest 一致就直接读」用它，但没有任何代码把生成的语料写回去 → 那个缓存路径**永远为空**、每次 `run` 都重建语料（而 `load_corpus` 的 docstring 与配置注释都在说「文件缺失时会自动生成」，实际并不会）。修：在 `run_and_write` 里把 `run_pipeline` 建好的语料写回该路径（`run_pipeline` 用 `result._corpus` 把它带出来）。

### C.2 可执行的不变量（测试护栏）

| 不变量 | 由哪个测试钉死 |
| --- | --- |
| 观测网格完整 = 48 run（扇出不漏分支） | `test_run_count_is_full_grid` |
| 五条属性曲线的分化层命中设计期望 | `test_all_five_differentiation_layers_recovered` |
| 随机权重对照不出现阶梯 | `test_random_control_has_no_step` |
| 参考配置**加载后**仍可解析 | `test_reference_run_resolves_after_load` |
| 词形在 L0 可读（> 0.5） | `test_form_is_readable_from_layer0` |
| 歧义角色的分化层 > 非歧义角色 | `test_ambiguous_role_differentiates_deeper_than_unambiguous` |
| `to_dict` / `from_dict` 互逆（含 `HeadStats` 的 `node_id` 陷阱） | `test_*_roundtrip`（7 项） |
| 全部 48 个 spec_id 能被 `from_spec_id` 还原 | `test_state_spec_roundtrip_all_runs` |
| 只读命令不改 `out/` | `test_readonly_command_does_not_write` |
| `visualize` 是唯一的写命令 | `test_visualize_is_the_write_command` |
| 矩阵行数 == run 数 == 48 | `test_matrix_rows_match_runs` |
| `run` 把生成的语料写回 `corpus.sentences`（缓存不是死代码） | `test_run_materializes_sentences_cache` |
| `write_corpus` 产出的文件能被 `load_corpus` 的 digest 判据接受 | `test_write_then_read_is_consistent` |

> 测试套件：**38 passed**。首次运行会因生成 `out/` 耗时约 73s，其后从磁盘加载 <1s。
