"""CLI 冒烟与产物一致性。

覆盖：无子命令 / 未知 spec / 非法 position 的退出码，
以及矩阵行数、曲线条数与 run 数的一致。
"""

from __future__ import annotations

from representationfactory.cli import build_parser, main as cli_main


def test_no_command_returns_1(capsys):
    assert cli_main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_unknown_spec_returns_1():
    assert cli_main(["curve", "--spec", "does-not-exist", "--property", "form"]) == 1


def test_bad_explain_position_returns_1():
    assert cli_main(["explain", "--position", "not-a-position"]) == 1


def test_parser_parses_core_commands():
    parser = build_parser()
    assert parser.parse_args(["version"]).command == "version"
    assert parser.parse_args(["graph", "--spec", "x"]).command == "graph"
    assert parser.parse_args(["explain", "--position", "3:12"]).command == "explain"
    assert parser.parse_args(["probe", "--spec", "x", "--kind", "geometry"]).kind == "geometry"


def test_matrix_rows_match_runs(result):
    assert len(result.matrix) == len(result.runs) == 48


def test_every_run_has_cluster_probe(result):
    for run in result.runs:
        assert "cluster" in run.probes, f"{run.spec_id} 缺 cluster 探针"
