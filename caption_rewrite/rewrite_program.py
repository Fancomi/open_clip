"""dspy 改写程序: caption -> rewritten_caption。

策略: 仅把不常用词替换为常用同义词; 严格保原意, 不加不删信息; 无稀有词则原样返回。
GEPA 会在此 instructions 基础上反思进化。
"""
import dspy


class RewriteCaption(dspy.Signature):
    """Rewrite an image caption to use only common, everyday vocabulary.

    Replace rare or unusual words with their most common synonyms.
    Keep the original meaning exactly: do not add, remove, or invent any
    information, objects, attributes, or actions. Change as few words as
    possible. If every word is already common, return the caption unchanged.
    Output lowercase, no quotes, no explanation."""

    caption = dspy.InputField(desc="original image caption")
    rewritten_caption = dspy.OutputField(desc="caption with rare words replaced by common ones, meaning preserved")


class Rewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RewriteCaption)

    def forward(self, caption):
        return self.predict(caption=caption)
