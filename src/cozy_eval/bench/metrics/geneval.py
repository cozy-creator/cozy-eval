"""Compositional adherence: detector-scored object presence, count, colour, position.

STABILITY: experimental (v0.x). Metric NAMES (``composition_correct``,
``composition_recall``) are locked by the registry; the typed spec and the
backend classes may still move.

Method: GenEval (Ghosh et al., arXiv:2310.11513 — MIT). The scoring semantics
below follow that paper exactly (thresholds, the dead-zone position rule,
exact counts via exclude clauses) so scores stay comparable with published
GenEval numbers. The implementation is original; backends differ deliberately:

  detector   Grounding DINO (Apache-2.0 weights+code, transformers-native,
             open-vocabulary) instead of mmdet Mask2Former — no mmcv stack,
             and non-COCO classes work.
  colours    SigLIP2 (Apache-2.0, explicit on the HF card) instead of OpenAI
             CLIP, whose HF cards carry no licence tag. Same three prompt
             templates as GenEval; crops are plain bbox crops (no instance
             mask compositing — boxes are all an open-vocab detector gives).

Confidence CALIBRATION is detector-specific: GenEval's 0.3/0.9 thresholds are
Mask2Former posteriors; Grounding DINO similarity scores run lower, so the
backend carries its own calibrated defaults and ``score_spec`` takes thresholds
explicitly. See PROVENANCE.md for the full model-licence table.

Everything through :func:`score_spec` is closed-form over typed detections —
no model, no torch — so the scoring logic is testable without weights.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import msgspec

from ..device import AUTO, resolve_device
from ..errors import ConfigError, DataError

# GenEval constants (Mask2Former-calibrated; kept for parity and as the
# closed-form defaults).
CONF_THRESHOLD = 0.3
COUNTING_CONF_THRESHOLD = 0.9
POSITION_DEAD_ZONE = 0.1     # fraction of (dim_a + dim_b) ignored per axis
MAX_OBJECTS_PER_CLASS = 16
NMS_MAX_OVERLAP = 1.0        # GenEval default: only exact duplicates suppressed

COLORS = ("red", "orange", "yellow", "green", "blue",
          "purple", "pink", "brown", "black", "white")
RELATIONS = ("left of", "right of", "above", "below")

# GenEval's three zero-shot templates, verbatim from the paper's method.
COLOR_TEMPLATES = (
    "a photo of a {color} {name}",
    "a photo of a {color}-colored {name}",
    "a photo of a {color} object",
)

DEFAULT_DETECTOR = "IDEA-Research/grounding-dino-tiny"
DEFAULT_COLOR_MODEL = "google/siglip2-base-patch16-224"

_MODEL_CACHE: dict[str, Any] = {}


def free_models() -> None:
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# typed spec + detections
# ---------------------------------------------------------------------------

class Detection(msgspec.Struct, frozen=True, kw_only=True):
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    color: str = ""                          # filled by a colour classifier


class ObjectTerm(msgspec.Struct, frozen=True, kw_only=True):
    """One required object clause: N instances of ``name``, optionally of a
    colour, optionally positioned relative to another include term."""

    name: str
    count: int = 1
    color: str = ""
    relation: str = ""       # one of RELATIONS
    relative_to: int = -1    # index into CompositionalSpec.include


class ExcludeTerm(msgspec.Struct, frozen=True, kw_only=True):
    """Forbidden: ``count`` or more detections of ``name`` fail the spec.
    GenEval encodes exact counts as include(N) + exclude(N+1)."""

    name: str
    count: int = 1


class CompositionalSpec(msgspec.Struct, frozen=True, kw_only=True):
    spec_id: str
    prompt: str
    include: tuple[ObjectTerm, ...]
    exclude: tuple[ExcludeTerm, ...] = ()
    tags: tuple[str, ...] = ()


class TermVerdict(msgspec.Struct, kw_only=True):
    term: str
    satisfied: bool
    detail: str = ""


class CompositionalScore(msgspec.Struct, kw_only=True):
    """One image against one spec. ``correct`` is the GenEval-style boolean
    (ALL include AND no exclude); ``term_recall`` is the graded fraction that
    feeds the adherence dimension."""

    spec_id: str
    correct: bool
    term_recall: float
    terms: list[TermVerdict] = msgspec.field(default_factory=list)


def validate_spec(spec: CompositionalSpec) -> None:
    """Author errors fail at load time, not as silent zeros at eval time."""
    for i, term in enumerate(spec.include):
        where = f"{spec.spec_id}.include[{i}]"
        if not term.name.strip():
            raise DataError(f"{where}: empty object name")
        if term.count < 1:
            raise DataError(f"{where}: count must be >= 1, got {term.count}")
        if term.relation:
            if term.relation not in RELATIONS:
                raise DataError(
                    f"{where}: unknown relation {term.relation!r}; known: {list(RELATIONS)}"
                )
            if not 0 <= term.relative_to < len(spec.include):
                raise DataError(
                    f"{where}: relative_to={term.relative_to} is out of range for "
                    f"{len(spec.include)} include terms"
                )
            if term.relative_to == i:
                raise DataError(f"{where}: term is relative to itself")
    for i, term in enumerate(spec.exclude):
        if not term.name.strip():
            raise DataError(f"{spec.spec_id}.exclude[{i}]: empty object name")
        if term.count < 1:
            raise DataError(f"{spec.spec_id}.exclude[{i}]: count must be >= 1")
    if not spec.include and not spec.exclude:
        raise DataError(f"{spec.spec_id}: empty spec (no include and no exclude terms)")


def vocabulary(spec: CompositionalSpec) -> tuple[str, ...]:
    """The object names a detector must be queried for, deduped, order kept."""
    seen: dict[str, None] = {}
    for term in spec.include:
        seen.setdefault(term.name, None)
    for term in spec.exclude:
        seen.setdefault(term.name, None)
    return tuple(seen)


def needs_counting(spec: CompositionalSpec) -> bool:
    """GenEval scores counting prompts with the strict threshold for the whole
    image, exclude clauses included — the N+1 exclude must see the same set."""
    return any(t.count >= 2 for t in spec.include)


# ---------------------------------------------------------------------------
# geometry — GenEval's relative-position rule, exactly
# ---------------------------------------------------------------------------

def relative_position(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    *,
    dead_zone: float = POSITION_DEAD_ZONE,
) -> frozenset[str]:
    """Relations of A with respect to B (image coords, y grows downward).

    GenEval's rule: centre offset per axis, minus a dead zone of
    ``dead_zone * (dim_a + dim_b)``, zero-clamped and re-signed; if the revised
    offset survives, normalize by the ORIGINAL offset norm and call an axis
    when its component exceeds 0.5. Overlapping boxes yield no relation rather
    than a wrong one.
    """
    ax = (box_a[0] + box_a[2]) / 2.0 - (box_b[0] + box_b[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0 - (box_b[1] + box_b[3]) / 2.0
    aw, ah = box_a[2] - box_a[0], box_a[3] - box_a[1]
    bw, bh = box_b[2] - box_b[0], box_b[3] - box_b[1]
    rx = max(abs(ax) - dead_zone * (aw + bw), 0.0) * math.copysign(1.0, ax)
    ry = max(abs(ay) - dead_zone * (ah + bh), 0.0) * math.copysign(1.0, ay)
    if abs(rx) < 1e-3 and abs(ry) < 1e-3:
        return frozenset()
    norm = math.hypot(ax, ay)
    dx, dy = rx / norm, ry / norm
    out = set()
    if dx < -0.5:
        out.add("left of")
    if dx > 0.5:
        out.add("right of")
    if dy < -0.5:
        out.add("above")
    if dy > 0.5:
        out.add("below")
    return frozenset(out)


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _dedup(dets: list[Detection], max_overlap: float) -> list[Detection]:
    """Score-descending NMS within one class; GenEval's default of 1.0 only
    drops exact duplicates. Capped at MAX_OBJECTS_PER_CLASS either way."""
    kept: list[Detection] = []
    for det in sorted(dets, key=lambda d: -d.score):
        if all(_iou(det.box, k.box) < max_overlap for k in kept):
            kept.append(det)
        if len(kept) >= MAX_OBJECTS_PER_CLASS:
            break
    return kept


# ---------------------------------------------------------------------------
# closed-form scoring
# ---------------------------------------------------------------------------

def score_spec(
    spec: CompositionalSpec,
    detections: list[Detection],
    *,
    conf: float = CONF_THRESHOLD,
    counting_conf: float = COUNTING_CONF_THRESHOLD,
    max_overlap: float = NMS_MAX_OVERLAP,
) -> CompositionalScore:
    """Score one image's detections against one spec. Pure function.

    Detections must already carry ``color`` for classes any include term
    constrains by colour (``score_image`` arranges this); a colour-constrained
    term treats an unclassified detection as a non-match.
    """
    threshold = counting_conf if needs_counting(spec) else conf
    by_class: dict[str, list[Detection]] = {}
    for det in detections:
        if det.score >= threshold:
            by_class.setdefault(det.label, []).append(det)
    by_class = {k: _dedup(v, max_overlap) for k, v in by_class.items()}

    verdicts: list[TermVerdict] = []
    # Candidate sets per include term, for position checks against the group.
    candidates: list[list[Detection]] = []
    for term in spec.include:
        found = by_class.get(term.name, [])
        if term.color:
            found = [d for d in found if d.color == term.color]
        candidates.append(found)

    for i, term in enumerate(spec.include):
        name = f"{term.color} {term.name}".strip()
        label = f"include: {term.count}x {name}"
        found = candidates[i]
        if len(found) < term.count:
            verdicts.append(TermVerdict(
                term=label, satisfied=False,
                detail=f"found {len(found)} of {term.count}",
            ))
            continue
        if term.relation:
            anchors = candidates[term.relative_to]
            hit = any(
                a is not b and term.relation in relative_position(a.box, b.box)
                for a in found for b in anchors
            )
            if not hit:
                verdicts.append(TermVerdict(
                    term=f"{label} {term.relation} {spec.include[term.relative_to].name}",
                    satisfied=False, detail="relation not observed",
                ))
                continue
            label = f"{label} {term.relation} {spec.include[term.relative_to].name}"
        verdicts.append(TermVerdict(term=label, satisfied=True))

    for term in spec.exclude:
        n = len(by_class.get(term.name, []))
        verdicts.append(TermVerdict(
            term=f"exclude: <{term.count}x {term.name}", satisfied=n < term.count,
            detail="" if n < term.count else f"found {n}",
        ))

    hits = sum(1 for v in verdicts if v.satisfied)
    return CompositionalScore(
        spec_id=spec.spec_id,
        correct=hits == len(verdicts),
        term_recall=hits / len(verdicts) if verdicts else 0.0,
        terms=verdicts,
    )


# ---------------------------------------------------------------------------
# backends (lazy imports; all weights Apache-2.0 — see PROVENANCE.md)
# ---------------------------------------------------------------------------

class Detector(Protocol):
    conf: float
    counting_conf: float

    def detect(self, image: Any, labels: tuple[str, ...]) -> list[Detection]: ...


class ColorClassifier(Protocol):
    def classify(self, image: Any, det: Detection, *, palette: tuple[str, ...]) -> str: ...


class GroundingDino:
    """Open-vocabulary detection via transformers Grounding DINO.

    Similarity scores run LOWER than Mask2Former posteriors, so the GenEval
    0.3/0.9 thresholds do not transfer; the defaults here are this backend's
    calibrated equivalents (box_threshold at detect time, then presence/count
    thresholds at scoring time). Calibration evidence: parity fixtures in
    ``tests/`` — re-measure before changing.
    """

    def __init__(
        self, model_ref: str = DEFAULT_DETECTOR, *, device: str = AUTO,
        box_threshold: float = 0.25, text_threshold: float = 0.25,
        conf: float = 0.30, counting_conf: float = 0.45,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_ref = model_ref
        self.device = resolve_device(device)
        device = self.device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.conf = conf
        self.counting_conf = counting_conf
        key = f"gdino:{model_ref}:{device}"
        if key not in _MODEL_CACHE:
            processor = AutoProcessor.from_pretrained(model_ref)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_ref)
            _MODEL_CACHE[key] = (processor, model.to(device).eval())
        self.processor, self.model = _MODEL_CACHE[key]

    def detect(self, image: Any, labels: tuple[str, ...]) -> list[Detection]:
        import torch

        image = image.convert("RGB") if hasattr(image, "convert") else image
        # Grounding DINO's query format: lowercase phrases joined by ". ".
        queries = [lbl.strip().lower() for lbl in labels]
        text = ". ".join(queries) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        out: list[Detection] = []
        for score, box, phrase in zip(
            result["scores"], result["boxes"], result["text_labels"], strict=True,
        ):
            label = _match_label(str(phrase), queries)
            if label is None:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            out.append(Detection(label=label, score=float(score), box=(x0, y0, x1, y1)))
        return out


def _match_label(phrase: str, queries: list[str]) -> str | None:
    """Map a predicted phrase back to the requested vocabulary. Grounding DINO
    may return merged or partial phrases; exact match first, then containment,
    longest query wins so 'traffic light' beats 'light'."""
    phrase = phrase.strip().lower()
    if phrase in queries:
        return phrase
    best = None
    for q in queries:
        if q in phrase and (best is None or len(q) > len(best)):
            best = q
    return best


class SiglipColors:
    """Zero-shot colour naming on bbox crops, GenEval's three-template method."""

    def __init__(self, model_ref: str = DEFAULT_COLOR_MODEL, *, device: str = AUTO):
        from transformers import AutoModel, AutoProcessor

        self.model_ref = model_ref
        self.device = resolve_device(device)
        device = self.device
        key = f"siglip:{model_ref}:{device}"
        if key not in _MODEL_CACHE:
            processor = AutoProcessor.from_pretrained(model_ref)
            model = AutoModel.from_pretrained(model_ref)
            _MODEL_CACHE[key] = (processor, model.to(device).eval())
        self.processor, self.model = _MODEL_CACHE[key]

    def classify(
        self, image: Any, det: Detection, *, palette: tuple[str, ...] = COLORS,
    ) -> str:
        import torch

        image = image.convert("RGB") if hasattr(image, "convert") else image
        x0, y0, x1, y1 = det.box
        crop = image.crop((int(x0), int(y0), max(int(x1), int(x0) + 1), max(int(y1), int(y0) + 1)))
        texts = [
            tpl.format(color=c, name=det.label) for c in palette for tpl in COLOR_TEMPLATES
        ]
        inputs = self.processor(
            text=texts, images=crop, padding="max_length", return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits_per_image[0]
        per_color = logits.view(len(palette), len(COLOR_TEMPLATES)).mean(dim=1)
        return palette[int(per_color.argmax())]


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def score_image(
    spec: CompositionalSpec,
    image: Any,
    detector: Detector,
    *,
    colors: ColorClassifier | None = None,
    palette: tuple[str, ...] = COLORS,
) -> CompositionalScore:
    """Detect once with the spec's whole vocabulary, classify colours only on
    the classes that need them, then score closed-form."""
    validate_spec(spec)
    detections = detector.detect(image, vocabulary(spec))
    color_classes = {t.name for t in spec.include if t.color}
    if color_classes:
        if colors is None:
            raise ConfigError(
                f"{spec.spec_id}: spec constrains the colour of {sorted(color_classes)} "
                "but no colour classifier was given; pass colors=SiglipColors()"
            )
        detections = [
            msgspec.structs.replace(d, color=colors.classify(image, d, palette=palette))
            if d.label in color_classes else d
            for d in detections
        ]
    return score_spec(
        spec, detections,
        conf=detector.conf, counting_conf=detector.counting_conf,
    )


__all__ = [
    "COLORS",
    "COLOR_TEMPLATES",
    "CONF_THRESHOLD",
    "COUNTING_CONF_THRESHOLD",
    "DEFAULT_COLOR_MODEL",
    "DEFAULT_DETECTOR",
    "MAX_OBJECTS_PER_CLASS",
    "NMS_MAX_OVERLAP",
    "POSITION_DEAD_ZONE",
    "RELATIONS",
    "ColorClassifier",
    "CompositionalScore",
    "CompositionalSpec",
    "Detection",
    "Detector",
    "ExcludeTerm",
    "GroundingDino",
    "ObjectTerm",
    "SiglipColors",
    "TermVerdict",
    "free_models",
    "needs_counting",
    "relative_position",
    "score_image",
    "score_spec",
    "validate_spec",
    "vocabulary",
]
