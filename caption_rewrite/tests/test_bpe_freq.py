import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from caption_rewrite import bpe_freq


def test_encode_ids_pure_bpe():
    ids = bpe_freq.encode_ids("a red zebra")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert 49406 not in ids and 49407 not in ids
    assert len(ids) >= 3


def test_rare_ids_token_mode():
    freq = {10: 100, 20: 5, 30: 1}
    rs = bpe_freq.rare_ids(freq, n=10, mode='token')
    assert rs == {20, 30}
    ids = bpe_freq.encode_ids("zebra")
    freq2 = {i: 1 for i in ids}
    rs2 = bpe_freq.rare_ids(freq2, n=5, mode='token')
    assert bpe_freq.count_rare("zebra", rs2) == len(ids)
    assert bpe_freq.count_rare("zebra", set()) == 0


def test_rare_ids_word_mode_merges_variants(monkeypatch):
    # 模拟 CLIP BPE: 同一词面有带尾空格/不带两个 token-id (如 'second '/'second')。
    fake_decode = {1: 'second ', 2: 'second', 3: 'artichoke '}
    class _Fake:
        def decode(self, ids): return fake_decode[ids[0]]
    monkeypatch.setattr(bpe_freq, 'get_tokenizer', lambda: _Fake())
    # 'second' 两变体: 带空格 90 + 句末 5 = 词面 95
    freq = {1: 90, 2: 5, 3: 1}
    # word 模式: second 合并 95 >= 50 不算稀有; artichoke 1 < 50 算稀有
    assert bpe_freq.rare_ids(freq, n=50, mode='word') == {3}
    # token 模式(旧): 句末低频变体 2 被误判稀有
    assert bpe_freq.rare_ids(freq, n=50, mode='token') == {2, 3}
    # 阈值 >95 时 second 两变体也算稀有
    assert bpe_freq.rare_ids(freq, n=200, mode='word') == {1, 2, 3}
