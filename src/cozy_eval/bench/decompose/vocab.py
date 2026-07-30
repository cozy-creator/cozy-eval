"""The generation vocabulary. OUR OWN word lists (MIT) — GenEval2's are CC-BY-NC.

Coverage follows its published axes: 20 COCO + 20 non-COCO objects split
animate/inanimate, colours + materials + patterns, geometric + 3D + verb
relations, counts 2..7. COCO names double as detector vocabulary; non-COCO
names are open-vocab words Grounding DINO handles zero-shot.
"""

from __future__ import annotations

# 10 + 10 COCO classes (detector-friendly), 10 + 10 beyond COCO.
COCO_ANIMATE = ("dog", "cat", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "bird")
COCO_INANIMATE = ("car", "bicycle", "motorcycle", "airplane", "bus",
                  "train", "boat", "bench", "chair", "couch")
EXTRA_ANIMATE = ("rabbit", "squirrel", "fox", "owl", "penguin",
                 "dolphin", "butterfly", "hedgehog", "frog", "duck")
EXTRA_INANIMATE = ("lantern", "teapot", "typewriter", "telescope", "cactus",
                   "windmill", "anchor", "drum", "violin", "wheelbarrow")

ANIMATE = COCO_ANIMATE + EXTRA_ANIMATE
INANIMATE = COCO_INANIMATE + EXTRA_INANIMATE
OBJECTS = ANIMATE + INANIMATE

COLORS = ("red", "orange", "yellow", "green", "blue",
          "purple", "pink", "brown", "black", "white")
MATERIALS = ("wooden", "metal", "glass", "plastic", "stone", "wicker")
PATTERNS = ("striped", "checkered", "polka-dot", "spotted")
ATTRIBUTES = COLORS + MATERIALS + PATTERNS

# Detector lane: closed-form from bounding boxes.
GEOMETRIC_RELATIONS = ("left of", "right of", "above", "below")
# Judge lane only: unanswerable from boxes (3D or interaction).
SOFT_RELATIONS = ("on top of", "under", "inside", "behind",
                  "in front of", "next to", "facing", "chasing")

COUNTS = (2, 3, 4, 5, 6, 7)
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}

_PLURAL_EXCEPTIONS = {"sheep": "sheep"}


def article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def plural(word: str) -> str:
    if word in _PLURAL_EXCEPTIONS:
        return _PLURAL_EXCEPTIONS[word]
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


__all__ = [
    "ANIMATE",
    "ATTRIBUTES",
    "COCO_ANIMATE",
    "COCO_INANIMATE",
    "COLORS",
    "COUNTS",
    "EXTRA_ANIMATE",
    "EXTRA_INANIMATE",
    "GEOMETRIC_RELATIONS",
    "INANIMATE",
    "MATERIALS",
    "NUMBER_WORDS",
    "OBJECTS",
    "PATTERNS",
    "SOFT_RELATIONS",
    "article",
    "plural",
]
