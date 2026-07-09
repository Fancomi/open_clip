"""逐句评分: 保真优先。teacher(Opus) 判"图文仍匹配"作硬闸,
达标后 score = max(0, 稀有词降幅率 − λ·归一化编辑距离)。

数据集是闭集, 优化目标是数据间互连而非绝对真实度, 故允许"向上抽象":
把具体词换成图片仍满足的更常见上位类别 (某种鸟名→水生鸟, currant→berry,
poodle→dog) 算保真; 换成图里没有的东西/改属性数量/删整个物体=不保真(零分)。
"""
import logging

import dspy

log = logging.getLogger(__name__)

_RARE_SET = set()   # 由 make_metric 注入


def _count_rare(caption):
    import bpe_freq
    return bpe_freq.count_rare(caption, _RARE_SET)


def norm_levenshtein(a, b):
    """归一化编辑距离 ∈ [0,1]。"""
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[lb] / max(la, lb)


def rare_reduction_rate(orig_rare, new_rare):
    """(orig-new)/max(orig,1) 夹到 [0,1]; orig==0 记 1.0 (无需改)。"""
    if orig_rare == 0:
        return 1.0
    return max(0.0, (orig_rare - new_rare) / orig_rare)


class _Faithful(dspy.Signature):
    """Judge whether the rewritten caption is still TRUE of the same image as
    the original. This is for a closed dataset: broadening a word to a more
    general category is allowed, only falsehood is not.

    FAITHFUL (yes): exact synonym (purchase->buy); OR generalizing a specific
    term UP to a broader category the image still satisfies (currant->berry,
    poodle->dog, 'pied kingfisher'->'water bird', ramen->noodles); OR keeping a
    rare word unchanged. The rewrite may be less specific than the original.

    UNFAITHFUL (no) ONLY IF it states something FALSE of the image: a changed
    object/attribute/count/action/relation (cat->dog, red->blue, two->three,
    'on'->'under'), an invented detail not in the original, or a whole object/
    entity dropped from the caption. Going more general is NOT unfaithful;
    going to something wrong or absent IS."""
    original = dspy.InputField()
    rewritten = dspy.InputField()
    faithful = dspy.OutputField(desc="yes or no")
    reason = dspy.OutputField(desc="short reason, name any falsehood/dropped entity")


def judge_faithful(teacher, original, rewritten):
    """返回 (是否保真, 理由)。teacher 失败降级 (True,'judge-failed') 不阻塞。"""
    try:
        with dspy.context(lm=teacher):
            r = dspy.Predict(_Faithful)(original=original, rewritten=rewritten)
        ok = str(r.faithful).strip().lower().startswith("y")
        return ok, str(r.reason)
    except Exception as e:
        log.warning(f"[metric] teacher judge failed: {e}")
        return True, "judge-failed"


def make_metric(teacher, rare_set, lam=0.3, unfaithful_score=0.0):
    """构造 GEPA metric。rare_set 冻结注入; teacher 做保真硬闸。

    保真优先: 语义歪曲 → score=0 (硬乘子, 不可交易)。达标后才算 降词率 − λ·编辑距离。
    unfaithful_score 默认 0.0: 歪曲改写零收益, 逼优化器只走安全替换。
    """
    global _RARE_SET
    _RARE_SET = set(rare_set)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        orig = gold.caption
        new = getattr(pred, "rewritten_caption", "") or ""
        ok, reason = judge_faithful(teacher, orig, new)
        if not ok:
            return dspy.Prediction(
                score=unfaithful_score,
                feedback=f"改成了图里没有的东西(零分, 硬闸): {reason}. 可以把具体词抽象成"
                         f"图片仍满足的更常见上位类别(某种鸟→水生鸟, currant→berry), "
                         f"但绝不能改成错误/不存在的物体属性数量, 也不能删掉整个物体。")
        o_rare, n_rare = _count_rare(orig), _count_rare(new)
        red = rare_reduction_rate(o_rare, n_rare)
        edit = norm_levenshtein(orig, new)
        score = max(0.0, red - lam * edit)
        fb = f"保真通过。稀有token {o_rare}->{n_rare} (降幅率{red:.2f}), 编辑距离{edit:.2f}。"
        if n_rare > 0:
            fb += (f" 仍有 {n_rare} 个稀有token: 换成更常见的同义词或图片仍满足的上位类别"
                   f"(某种鸟→水生鸟), 别改成错误的东西。")
        if edit > 0.5:
            fb += " 改动过大, 尽量少改词。"
        return dspy.Prediction(score=score, feedback=fb)

    return metric
