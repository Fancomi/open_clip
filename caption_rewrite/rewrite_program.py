"""dspy 改写程序: caption -> rewritten_caption。

策略: 仅把不常用词替换为常用同义词; 严格保原意, 不加不删信息; 无稀有词则原样返回。
GEPA 会在此 instructions 基础上反思进化。
"""
import dspy


class RewriteCaption(dspy.Signature):
    """Rewrite an image caption to use more common, everyday vocabulary.

    Replace a rare or unusual word ONLY when a common word means EXACTLY the
    same thing. Preserving meaning always beats replacing a word: if no common
    word carries the same specific meaning (e.g. specific foods, materials,
    breeds, tools, place types like 'ramen', 'currant', 'motel'), KEEP the
    original word unchanged. Never generalize, drop, add, or distort any
    object, attribute, count, action, or spatial relation just to remove a
    rare word. Change as few words as possible; keeping a rare word is fine
    when no exact synonym exists. Output lowercase, no quotes, no explanation."""

    caption = dspy.InputField(desc="original image caption")
    rewritten_caption = dspy.OutputField(desc="caption with only safely-replaceable rare words swapped for exact common synonyms, meaning fully preserved")


class Rewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RewriteCaption)

    def forward(self, caption):
        return self.predict(caption=caption)
