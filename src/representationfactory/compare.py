"""对比与结论（Compare / Assemble）。

README 第八节 M6「对比」的产物。本项目**不是 Factory**，没有候选、没有前沿、
没有帕累托支配 —— 所以这里的「对比」与同工作区其它项目**语义完全不同**：

- 别的项目对比的是「哪个候选更好」；
- 本项目对比的是「**标本的已知真值，被探针复原到了哪一层**」，以及
  「**随机对照模型**是否也出现了同样的阶梯」（没出现，才说明阶梯是构造出来的、
  不是随便一个模型都有）。

三类对比，缺一不可：

1. **分化层定位（differentiation）**
   对每个属性取「相邻层增幅最大的层」= ``onset_layer``。
   这是 README 要求「可指认语义 / 语法分化的具体层」的那个数字。
   构造式标本的预期：``form`` 在 L0、``role`` 在 L2（非歧义）/ L3（歧义）、
   ``topic`` 在 L4。

2. **跨模型对比（cross_model）**
   构造式标本 vs 随机权重对照。**只有构造式标本出现阶梯** ——
   随机模型的所有曲线都应该是平的（或噪声级），否则「阶梯」就只是
   「随便一个 transformer 都有」的平凡现象，结论就垮了。

3. **单变量方法消融（ablation）**
   不改被观测对象，只改「怎么观测」：
   - ``by_token`` vs ``by_sentence`` 划分 → 展示按 token 划分如何**系统性虚高**
     （本项目唯一一个「数字更高 = 方法更错」的轴）；
   - ``pca`` vs ``none`` 降维 → 展示「降维丢了多少可分性」。
   这两项是方法学对照，必须在矩阵里如实并列，绝不能只报「最优」那一组。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .schema import CaptureRun, LabResult, PROPERTIES


# ----------------------------------------------------------------------
# 分化层定位
# ----------------------------------------------------------------------
@dataclass
class PropertyOnset:
    """一个属性的分化层结论。"""

    property: str
    subset: str
    onset_layer: int
    onset_jump: float
    best_layer: int
    best_acc: float
    majority: float
    chance: float

    @property
    def spec_key(self) -> str:
        return f"{self.property}/{self.subset}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "property": self.property,
            "subset": self.subset,
            "onset_layer": self.onset_layer,
            "onset_jump": round(self.onset_jump, 6),
            "best_layer": self.best_layer,
            "best_acc": round(self.best_acc, 6),
            "majority": round(self.majority, 6),
            "chance": round(self.chance, 6),
        }


def differentiation(run: CaptureRun) -> list[PropertyOnset]:
    """对一个 run 的逐属性分化层定位。"""
    out: list[PropertyOnset] = []
    for key, curve in sorted(run.curves.items()):
        if not curve.accuracy:
            continue
        best = max(curve.accuracy)
        out.append(
            PropertyOnset(
                property=curve.property,
                subset=curve.subset,
                onset_layer=curve.onset_layer,
                onset_jump=curve.onset_jump,
                best_layer=curve.best_layer,
                best_acc=best,
                majority=curve.majority,
                chance=curve.chance,
            )
        )
    return out


def differentiation_map(run: CaptureRun) -> dict[str, PropertyOnset]:
    """``property/subset`` → onset，便于报告里按键取。"""
    return {o.spec_key: o for o in differentiation(run)}


# ----------------------------------------------------------------------
# 跨模型对比
# ----------------------------------------------------------------------
def cross_model(result: LabResult) -> dict[str, Any]:
    """构造式标本 vs 随机对照。

    返回每个模型的「头条增益」（max deep_gain 跨曲线）与逐属性分化层。
    **断言性结论**：构造式标本在 ``role`` / ``topic`` 上有正分化层，
    随机对照的同一组属性分化层应明显更浅、增益更接近 0。
    """
    by_model: dict[str, list[CaptureRun]] = {}
    for r in result.runs:
        by_model.setdefault(r.model, []).append(r)

    rows: dict[str, Any] = {}
    for model, runs in sorted(by_model.items()):
        headlines = [r.headline() for r in runs if r.curves]
        ref_run = result.reference_run() if model == "constructive" else None
        # 取该模型下与参考同口径的那一行（若它存在），否则取头条增益最大的
        if ref_run is not None and ref_run.model == model:
            onset = differentiation_map(ref_run)
        else:
            onset = {}
        rows[model] = {
            "n_runs": len(runs),
            "mean_headline": round(statistics.mean(headlines), 6) if headlines else 0.0,
            "max_headline": round(max(headlines), 6) if headlines else 0.0,
            "onset_layers": {
                k: o.onset_layer for k, o in sorted(onset.items())
            },
        }
    return rows


# ----------------------------------------------------------------------
# 单变量方法消融
# ----------------------------------------------------------------------
def _matching_runs(result: LabResult, base: CaptureRun, axis: str) -> list[CaptureRun]:
    """返回与 ``base`` 仅在第 ``axis`` 个观测轴上取值不同的所有 run。"""
    out: list[CaptureRun] = []
    for r in result.runs:
        if r.model != base.model:
            continue
        diff = [
            a
            for a in ("pooling", "similarity", "split", "project")
            if r.axis(a) != base.axis(a)
        ]
        if diff == [axis]:
            out.append(r)
    return out


def ablation_summary(result: LabResult) -> dict[str, Any]:
    """单变量方法消融表。

    对每个观测轴，把「参考配置」那一行与「只改这一个轴」的那几行并排，
    报出该轴带来的**相对变化**。这把 README 第八节的「方法学对照轴」
    落成了可计算的量，而不是一句口号。
    """
    ref = result.reference_run()
    if ref is None:
        return {"reference": None, "axes": {}}
    ref_map = differentiation_map(ref)
    axes: dict[str, Any] = {}
    for axis in ("pooling", "similarity", "split", "project"):
        variants = _matching_runs(result, ref, axis)
        if not variants:
            continue
        recs = []
        for v in variants:
            v_map = differentiation_map(v)
            # 对每个属性，比较 best_acc 的变化
            delta: dict[str, float] = {}
            for key, o in ref_map.items():
                vo = v_map.get(key)
                if vo is not None:
                    delta[key] = round(vo.best_acc - o.best_acc, 6)
            recs.append(
                {
                    "axis_value": v.axis(axis),
                    "headline": round(v.headline(), 6),
                    "delta_best_acc": delta,
                    "is_control": v.spec.is_control,
                }
            )
        axes[axis] = {
            "reference_value": ref.axis(axis),
            "variants": sorted(recs, key=lambda x: str(x["axis_value"])),
        }
    return {"reference": ref.spec_id, "axes": axes}


# ----------------------------------------------------------------------
# 对比矩阵（每个观测配置一行）
# ----------------------------------------------------------------------
def build_matrix(result: LabResult) -> list[dict[str, Any]]:
    """一行一个观测配置。

    列 = 五个观测轴 + 控制标记 + 头条指标 + 逐属性分化层。
    这是报告里那张大表，也是 ``--dot`` 之外最主要的「全部观测怎么排」视图。
    """
    ref = result.reference_run()
    rows: list[dict[str, Any]] = []
    for r in result.runs:
        onset = differentiation_map(r)
        row: dict[str, Any] = {
            "spec_id": r.spec_id,
            "model": r.model,
            "pooling": r.axis("pooling"),
            "similarity": r.axis("similarity"),
            "split": r.axis("split"),
            "project": r.axis("project"),
            "is_control": r.spec.is_control,
            "headline": round(r.headline(), 6),
            "n_curves": len(r.curves),
        }
        # 逐属性的最佳层与最佳准确率（报告里一眼能看出「阶梯在哪」）
        for prop in PROPERTIES:
            best = max(
                (o for o in onset.values() if o.property == prop),
                key=lambda o: o.best_acc,
                default=None,
            )
            row[f"{prop}_best_layer"] = best.best_layer if best else -1
            row[f"{prop}_best_acc"] = round(best.best_acc, 6) if best else 0.0
            # 分化层（onset），用全部子集里最早的那个
            onsets = [o.onset_layer for o in onset.values() if o.property == prop]
            row[f"{prop}_onset"] = min(onsets) if onsets else -1
        rows.append(row)
    rows.sort(key=lambda x: x["spec_id"])
    return rows


# ----------------------------------------------------------------------
# 收尾：把三类对比合成结论
# ----------------------------------------------------------------------
def assemble_findings(result: LabResult) -> dict[str, Any]:
    """把分化层定位、跨模型、方法消融合成一段可入库的 ``findings``。"""
    ref = result.reference_run()
    ref_onset = differentiation_map(ref) if ref else {}
    cross = cross_model(result)

    # 构造式标本的预期分化层（与 model.py 的设计真值对照）。
    # 键必须是 ``property/subset``（与 spec_key 对齐）。
    expected = {
        "form/all": 0,           # 词形在 embedding 层即可读（无需「分化」）
        "role/all": 2,
        "role/unambiguous": 2,
        "role/ambiguous": 3,
        "topic/all": 4,
    }
    recovered = {}
    for key, exp in expected.items():
        prop, _, subset = key.partition("/")
        got = ref_onset.get(key)
        curve = ref.curve(prop, subset) if ref else None
        if prop == "form":
            # ``form`` 在 embedding 层就线性可分：用「首层是否远超多数类基线」
            # 判定，而不是 onset_layer（平曲线会把 onset 推到最后一层）。
            readable = curve is not None and (curve.at(0) or 0.0) - curve.majority > 0.1
            recovered[key] = {
                "expected_onset": exp,
                "actual_onset": (got.onset_layer if got else 0),
                "match": bool(readable),
                "note": "词形在 embedding 层即可读（无需分化步骤）",
            }
        else:
            recovered[key] = {
                "expected_onset": exp,
                "actual_onset": got.onset_layer if got else -1,
                "match": (got is not None and got.onset_layer == exp),
            }

    # 随机对照是否「没有阶梯」：它的 role/topic 头条增益应远小于构造式标本
    constructive_headline = cross.get("constructive", {}).get("max_headline", 0.0)
    random_headline = cross.get("random", {}).get("max_headline", 0.0)
    control_clean = random_headline < 0.5 * constructive_headline

    findings = {
        "reference_spec": ref.spec_id if ref else None,
        "differentiation": {
            k: o.to_dict() for k, o in sorted(ref_onset.items())
        },
        "truthed_recovered": recovered,
        "all_onsets_recovered": all(v["match"] for v in recovered.values()),
        "cross_model": cross,
        "random_control_has_no_step": control_clean,
        "ablation": ablation_summary(result),
        "conclusion": _conclusion(recovered, control_clean, cross),
    }
    return findings


def _conclusion(
    recovered: dict[str, dict[str, Any]],
    control_clean: bool,
    cross: dict[str, Any],
) -> str:
    parts: list[str] = []
    if recovered:
        ok = [k for k, v in recovered.items() if v["match"]]
        parts.append(
            f"逐层线性探针在参考配置下复原了 {len(ok)}/{len(recovered)} 个已知分化层"
        )
    parts.append(
        "随机权重对照模型" + ("没有出现可比的阶梯" if control_clean else "仍出现了阶梯（对照失败）")
    )
    c = cross.get("constructive", {}).get("max_headline", 0.0)
    parts.append(f"构造式标本的最大逐层增益为 {c:.3f}")
    return "；".join(parts) + "。"
