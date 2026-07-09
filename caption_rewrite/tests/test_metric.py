import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import metric


def test_norm_levenshtein():
    assert metric.norm_levenshtein("abc", "abc") == 0.0
    assert abs(metric.norm_levenshtein("abc", "abd") - 1 / 3) < 1e-9
    assert metric.norm_levenshtein("", "") == 0.0


def test_rare_reduction_rate():
    assert abs(metric.rare_reduction_rate(3, 1) - 2 / 3) < 1e-9
    assert metric.rare_reduction_rate(1, 3) == 0.0
    assert metric.rare_reduction_rate(0, 0) == 1.0


def test_score_faithful_path(monkeypatch):
    monkeypatch.setattr(metric, "judge_faithful", lambda t, o, n: (True, "ok"))
    monkeypatch.setattr(metric, "_count_rare", lambda cap: 2 if cap == "orig" else 0)
    gold = type("G", (), {"caption": "orig"})()
    pred = type("P", (), {"rewritten_caption": "new"})()
    out = metric.make_metric(teacher=None, rare_set=set(), lam=0.3)(gold, pred)
    assert 0.5 < out.score <= 1.0


def test_score_unfaithful_path(monkeypatch):
    monkeypatch.setattr(metric, "judge_faithful", lambda t, o, n: (False, "改变了颜色"))
    monkeypatch.setattr(metric, "_count_rare", lambda cap: 2 if cap == "orig" else 0)
    gold = type("G", (), {"caption": "orig"})()
    pred = type("P", (), {"rewritten_caption": "new"})()
    out = metric.make_metric(teacher=None, rare_set=set(), lam=0.3)(gold, pred)
    assert out.score == 0.0                       # 硬乘子: 歪曲改写零收益, 不可交易
    assert "改变了颜色" in out.feedback
