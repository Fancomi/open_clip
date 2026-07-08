import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from caption_rewrite import bpe_freq


def test_encode_ids_pure_bpe():
    ids = bpe_freq.encode_ids("a red zebra")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert 49406 not in ids and 49407 not in ids
    assert len(ids) >= 3


def test_rare_ids_and_count_rare():
    freq = {10: 100, 20: 5, 30: 1}
    rs = bpe_freq.rare_ids(freq, n=10)
    assert rs == {20, 30}
    ids = bpe_freq.encode_ids("zebra")
    freq2 = {i: 1 for i in ids}
    rs2 = bpe_freq.rare_ids(freq2, n=5)
    assert bpe_freq.count_rare("zebra", rs2) == len(ids)
    assert bpe_freq.count_rare("zebra", set()) == 0
