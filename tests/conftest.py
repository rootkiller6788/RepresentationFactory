"""测试夹具。

两条设计要点：

1. **会话级只跑一次流水线**：``lab_out`` 只在 ``out/`` 缺失时调用一次
   ``run_and_write``（确定性，约 75s），之后所有测试都走 ``load_result``
   从磁盘快速重建 —— 于是「跑一次 75s + 几十个秒级断言」，而不是每个
   测试各跑一遍流水线。
2. **自动 chdir 到项目根**：只读子命令内部用相对路径 ``out`` /
   ``configs/default.json``，必须从项目根运行才能命中同一份产物。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from representationfactory.config import load_config  # noqa: E402
from representationfactory.pipeline import load_result, run_and_write  # noqa: E402

OUT = ROOT / "out"
CONFIG = ROOT / "configs" / "default.json"

#: 参考配置（``configs/default.json`` 的 ``ablation.reference``）
REF_ID = "constructive-tok-cos-bs-raw"


@pytest.fixture(scope="session", autouse=True)
def _chdir_root():
    """把 cwd 固定到项目根，让相对路径 ``out`` / ``configs`` 可预测。"""
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(old)


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config(CONFIG)


@pytest.fixture(scope="session")
def lab_out(cfg) -> Path:
    """确保 ``out/`` 存在；缺失时才生成（确定性）。"""
    if not (OUT / "findings.json").exists():
        run_and_write(cfg, OUT)
    return OUT


@pytest.fixture(scope="session")
def result(lab_out):
    r = load_result(lab_out)
    assert r is not None, "load_result 应能从 out/ 重建 LabResult"
    return r


@pytest.fixture(scope="session")
def ref_id() -> str:
    return REF_ID
