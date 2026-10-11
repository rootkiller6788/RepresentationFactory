"""schema 序列化往返：``to_dict`` / ``from_dict`` 必须互逆。

只读命令依赖 ``load_result`` 从产物重建 ``LabResult``，
所以每个被重建的 dataclass 都必须能无损往返。
"""

from __future__ import annotations

from representationfactory.schema import (
    CaptureRun,
    CorpusStats,
    HeadStats,
    LayerCurve,
    ProbeResult,
    StateSpec,
    Version,
)


def test_state_spec_roundtrip_all_runs(result):
    """全部 48 个 spec_id 都能被 from_spec_id 解析并还原回同一 id。"""
    for run in result.runs:
        assert StateSpec.from_spec_id(run.spec_id).spec_id == run.spec_id


def test_layercurve_roundtrip():
    c = LayerCurve(
        spec_id="x",
        property="form",
        layers=[0, 1, 2],
        accuracy=[0.1, 0.5, 0.9],
        majority=0.2,
        n_classes=5,
        n_train=10,
        n_test=4,
        subset="all",
    )
    c2 = LayerCurve.from_dict(c.to_dict())
    assert c2.layers == c.layers
    assert c2.accuracy == c.accuracy
    assert c2.majority == c.majority
    assert c2.n_classes == c.n_classes
    # 计算属性必须由还原后的字段重新推出
    assert c2.onset_layer == c.onset_layer
    assert c2.best_layer == c.best_layer
    assert c2.deep_gain == c.deep_gain


def test_headstats_roundtrip():
    """回归：``to_dict`` 多写 ``node_id``，``from_dict`` 必须忽略它。"""
    h = HeadStats(
        layer=2,
        head=1,
        entropy=1.5,
        self_weight=0.3,
        locality=0.4,
        sink_weight=0.2,
        mean_offset=1.1,
    )
    h2 = HeadStats.from_dict(h.to_dict())
    assert h2.to_dict() == h.to_dict()


def test_proberesult_roundtrip():
    p = ProbeResult(
        probe="similarity",
        spec_id="x",
        metrics={"a": 1.234567},
        detail={"k": [1, 2, 3], "nested": {"v": 0.5}},
    )
    p2 = ProbeResult.from_dict(p.to_dict())
    assert p2.probe == p.probe
    assert p2.spec_id == p.spec_id
    assert p2.metrics == p.metrics
    assert p2.detail == p.detail


def test_version_roundtrip(result):
    v = result.version
    assert Version.from_dict(v.to_dict()).to_dict() == v.to_dict()


def test_corpusstats_roundtrip(result):
    cs = result.corpus
    assert CorpusStats.from_dict(cs.to_dict()).to_dict() == cs.to_dict()


def test_capturerun_roundtrip(result):
    """整条 run（含曲线 / 探针 / 头画像）往返无损。"""
    run = result.reference_run()
    r2 = CaptureRun.from_dict(run.to_dict())
    assert r2.spec_id == run.spec_id
    assert set(r2.curves) == set(run.curves)
    for key, curve in run.curves.items():
        assert r2.curves[key].to_dict() == curve.to_dict()
    assert set(r2.probes) == set(run.probes)
    assert [h.to_dict() for h in r2.heads] == [h.to_dict() for h in run.heads]
