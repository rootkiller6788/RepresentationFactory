"""验收不变量：真值复原、随机对照、参考配置可解析。

这些断言直接对应用户可见的验收标准 ——
「可指认语义 / 语法分化的具体层」与「随机权重对照没有出现可比的阶梯」。
"""

from __future__ import annotations


def test_run_count_is_full_grid(result):
    """观测网格 = 2 模型 × 2 池化 × 3 相似度 × 2 划分 × 2 降维 = 48。"""
    assert len(result.runs) == 48
    assert len({r.spec_id for r in result.runs}) == 48


def test_all_five_differentiation_layers_recovered(result):
    """五条属性曲线的实测分化层全部命中设计期望层。"""
    rec = result.findings["truthed_recovered"]
    assert len(rec) == 5
    assert all(v["match"] for v in rec.values())
    assert result.findings["all_onsets_recovered"] is True


def test_random_control_has_no_step(result):
    """随机权重对照模型没有出现可比的逐层阶梯。"""
    assert result.findings["random_control_has_no_step"] is True


def test_reference_run_resolves_after_load(result, ref_id):
    """从磁盘加载的 LabResult 也能解析出参考配置（回归：曾因 cfg={} 返回 None）。"""
    ref = result.reference_run()
    assert ref is not None
    assert ref.spec_id == ref_id


def test_reference_curves_beat_majority(result, ref_id):
    """三条主属性的最佳层准确率都显著高于多数类基线。"""
    ref = result.find(ref_id)
    assert ref is not None
    for prop in ("form", "role", "topic"):
        c = ref.curve(prop, "all")
        assert c is not None, f"缺少曲线 {prop}/all"
        assert max(c.accuracy) > c.majority + 0.2, f"{prop} 未显著超过基线"


def test_form_is_readable_from_layer0(result, ref_id):
    """词形在 embedding 层即可读（浅层可读的对照）。"""
    c = result.find(ref_id).curve("form", "all")
    assert c.at(0) is not None
    assert c.at(0) > 0.5


def test_onset_layers_match_design(result, ref_id):
    """逐属性分化层：role 在 L2、topic 在 L4。"""
    ref = result.find(ref_id)
    assert ref.curve("role", "all").onset_layer == 2
    assert ref.curve("topic", "all").onset_layer == 4


def test_ambiguous_role_differentiates_deeper_than_unambiguous(result, ref_id):
    """歧义位置的角色要等注意力扩到主题标记词，故分化层比非歧义位置更深。"""
    ref = result.find(ref_id)
    amb = ref.curve("role", "ambiguous").onset_layer
    unamb = ref.curve("role", "unambiguous").onset_layer
    assert amb > unamb
