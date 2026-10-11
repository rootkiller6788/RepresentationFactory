"""语料缓存：``run`` 把生成的语料落盘，``load_corpus`` 按词典 digest 复用。

回归：``write_corpus`` 曾是被定义、却**从不被调用**的死代码 ——
配置声明的 ``corpus.sentences`` 路径永远是空的，``load_corpus`` 每次都重建。
"""

from __future__ import annotations

from representationfactory.config import load_jsonl, resolve_data_path
from representationfactory.corpus import load_corpus, write_corpus


def test_write_then_read_is_consistent(cfg, tmp_path):
    """``write_corpus`` 产出的文件必须能被 ``load_corpus`` 的 digest 判据接受。"""
    corpus = load_corpus(cfg)
    p = tmp_path / "sentences.jsonl"
    write_corpus(corpus, p)
    rows = load_jsonl(p)
    meta = next((r for r in rows if r.get("kind") == "meta"), None)
    assert meta is not None, "落盘语料必须带 meta 行"
    assert meta["lexicon_digest"] == corpus.lexicon.digest()


def test_run_materializes_sentences_cache(lab_out, cfg):
    """``run`` 必须把语料写回配置声明的路径（否则该缓存路径永远是空的）。"""
    assert resolve_data_path(cfg, "sentences").exists()
