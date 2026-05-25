"""Prompt templates for caption slot extraction."""
from .slots import SLOT_TYPES

SLOT_SCHEMA = {slot: [] for slot in SLOT_TYPES}

SLOT_EXTRACTION_PROMPT = """You are extracting lexical slots from short image-text captions for CLIP data analysis.

Task
- Read one caption.
- Extract only words or short phrases that explicitly appear in the caption.
- Put each extracted item into the correct slot category.
- Do not rewrite the caption.
- Do not improve, enrich, optimize, or paraphrase the caption.
- Do not infer visual details that are not present in the text.
- Do not add synonyms, hypernyms, object categories, attributes, or actions that do not appear in the caption.
- Preserve the original meaning and use the shortest useful word or phrase from the caption.

Slot categories
1. nouns: common concrete or abstract nouns, including objects, people, animals, places, materials, and scene entities.
2. verbs: actions, events, states, or verb phrases that appear in the caption.
3. adjectives: descriptive modifiers such as color, size, age, material, quality, condition, or appearance.
4. adverbs: adverbial modifiers such as manner, degree, time, or frequency.
5. numbers: explicit numerals or number words.
6. spatial_relations: prepositions or short phrases describing spatial layout or relative position, such as on, under, next to, in front of, behind, near.
7. proper_nouns: named entities or capitalized proper names that appear as names rather than sentence-initial common words.
8. others: useful caption words or short phrases that do not fit the categories above.

Rules
- Return valid JSON only. Do not output markdown, explanations, comments, or extra text.
- JSON keys must be exactly: nouns, verbs, adjectives, adverbs, numbers, spatial_relations, proper_nouns, others.
- Each value must be a JSON array of strings.
- If a category has no items, return an empty array.
- Remove duplicate items within each category.
- Use lowercase strings.
- Keep short multi-word phrases only when the phrase is necessary, for example "traffic light", "fire hydrant", "in front of".
- Do not include punctuation-only strings.
- Do not extract articles, determiners, conjunctions, pronouns, or generic function words, such as a, an, the, this, that, its, of, with, and, or, to, be.
- Use the others category sparingly. Only use others for content-bearing caption words that truly do not fit any other slot. If unsure, leave others empty.
- Do not extract auxiliary or copula verbs such as is, are, be, been, being, has, have, had, do, does, did, unless they are part of a meaningful verb phrase from the caption.
- Prefer content verbs over light verbs. For example, extract "sitting", "riding", "standing", "flying", but avoid standalone "is" or "are".
- Avoid duplicating the same semantic content across noun phrases and adjectives when a shorter noun is enough. For example, for "panoramic view", put "view" in nouns and "panoramic" in adjectives.

Input caption:
{{caption}}

Output JSON schema:
{
  "nouns": [],
  "verbs": [],
  "adjectives": [],
  "adverbs": [],
  "numbers": [],
  "spatial_relations": [],
  "proper_nouns": [],
  "others": []
}
"""


def format_slot_prompt(caption):
    return SLOT_EXTRACTION_PROMPT.replace('{{caption}}', str(caption))
