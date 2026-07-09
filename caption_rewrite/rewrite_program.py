"""dspy 改写程序: caption -> rewritten_caption。

策略: 仅把不常用词替换为常用同义词; 严格保原意, 不加不删信息; 无稀有词则原样返回。
GEPA 会在此 instructions 基础上反思进化。
"""
import dspy


class RewriteCaption(dspy.Signature):
    """Rewrite an image caption to use more common, everyday vocabulary, so a
    dataset of captions shares more common words and connects better.

    Replace a rare or unusual word with a more common word that is still TRUE
    of the same image. Two safe moves: (1) an exact common synonym
    (purchase->buy); (2) generalizing a specific term UP to a correct, more
    common category that the image still satisfies (currant->berry,
    poodle->dog, 'pied kingfisher'->'water bird', ramen->noodles). Prefer the
    most common word that stays true. NEVER change a word to something FALSE
    or not shown (cat->dog, red->blue, two->three), never invent details, and
    never drop a whole object or entity from the caption. If a rare word has
    no truer common replacement, keep it. Output lowercase, no quotes, no
    explanation."""

    caption = dspy.InputField(desc="original image caption")
    rewritten_caption = dspy.OutputField(desc="caption with rare words replaced by common words or true broader categories, image still described correctly")


class Rewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RewriteCaption)

    def forward(self, caption):
        return self.predict(caption=caption)
