"""只读契约：只读子命令**一个字节都不写** ``out/``。

这是「跑一次流水线、其余全部从产物加载」这条设计的关键保证：
若某条只读命令偷偷重算并落盘，它就不再只读，且会与 ``run`` 的产物竞争。
用逐文件 md5 实测（见 skill 坑 30）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from representationfactory.cli import main as cli_main

REF = "constructive-tok-cos-bs-raw"

#: 全部只读子命令（``run`` / ``visualize`` 是写命令，不在其列）
READONLY: list[list[str]] = [
    ["version"],
    ["compare"],
    ["list"],
    ["dag"],
    ["capture"],
    ["project"],
    ["explain", "--position", "3:12"],
    ["curve", "--spec", REF, "--property", "form"],
    ["curve", "--spec", REF, "--property", "role", "--subset", "ambiguous"],
    ["probe", "--spec", REF, "--kind", "cluster"],
    ["probe", "--spec", REF, "--kind", "geometry"],
    ["similarity", "--spec", REF],
    ["graph", "--spec", REF],
]


def _fingerprint(directory: Path) -> dict[str, str]:
    return {
        p.name: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(directory.glob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize("argv", READONLY, ids=lambda a: "-".join(a))
def test_readonly_command_does_not_write(lab_out, argv):
    before = _fingerprint(lab_out)
    rc = cli_main(argv)
    after = _fingerprint(lab_out)
    assert rc == 0, f"{argv} 退出码非 0"
    assert after == before, f"{argv} 改动了 out/ 产物"


def test_visualize_is_the_write_command(lab_out):
    """``visualize`` 是显式写命令：它生成 ``out/visualize.html``。"""
    rc = cli_main(["visualize"])
    assert rc == 0
    assert (lab_out / "visualize.html").exists()


def test_readonly_commands_leave_data_dir_untouched(lab_out):
    """只读命令也不该改写 ``data/``（句子由种子确定性生成）。"""
    data = Path("data")
    if not data.exists():
        pytest.skip("无 data/ 目录")
    before = {
        p.name: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(data.rglob("*"))
        if p.is_file()
    }
    for argv in READONLY:
        cli_main(argv)
    after = {
        p.name: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(data.rglob("*"))
        if p.is_file()
    }
    assert after == before
