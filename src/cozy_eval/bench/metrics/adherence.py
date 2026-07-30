"""Dimension 2: prompt adherence — did the render contain what was asked for.

STABILITY: locked core (v0.x) for the CHECKLIST FORMAT and the loader
(``ChecklistItem`` / ``Checklist`` / ``EditChecklist`` / ``ChecklistSet`` /
:func:`load_checklists`) and for the metric names. The judge helpers around them
are experimental.

*"if a user asks for a set of specific elements in an image,
we want to make sure that all of those elements are present, exactly as
specified. That's probably the most important single metric when evaluating a
model honestly."*

Two modes, one dimension:

**t2i** — score one render against its prompt's AUTHORED checklist.
``element_recall`` = weighted fraction of items verified. Reference-free, so it
scores any model's output, not only a quant candidate.

**edit** — a duality, because an edit can fail in two opposite directions:
``edit_compliance``   did the instructed change happen (UNDER-edit fails here);
``edit_preservation`` did everything else stay put (OVER-edit fails here).
Both are judged on the BEFORE+AFTER pair — "did X change" is unanswerable from
the after image alone. ``element_recall`` in edit mode is their weighted union,
so one number still names the dimension.

Cost discipline (*"fast and easy to evaluate... intensive within
reason"*): ONE structured multi-item query per image, not one forward pass per
item — a 10-item checklist costs one judge call, not ten. The judge stays
resident for the whole pass.

Checklist items are AUTHORED and versioned with the prompt set rather than
LLM-decomposed per run (the TIFA/DSG shape): a generated question set makes the
score irreproducible and unreviewable, and the decomposition becomes an
unversioned dependency on whatever model happened to run that day.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import msgspec

from ..device import AUTO, resolve_device
from ..errors import DataError

# Judge: Apache-2.0, ungated, ~18 GB, stock transformers class, and 89.6% on
# OCRBench v1 — so it also backstops the OCR pass on stylized lettering. The
# purpose-built alternative, Qwen's own checklist judge (Qwen/Qwen-Image-Bench,
# Apache-2.0), is 54.7 GB and runs thinking-mode at 4096 tokens x5 dimensions
# per image: too slow for a per-candidate gate, but the right OFFLINE calibration
# reference for checking this judge's agreement. Its checklist prompt/parse
# structure is what this module follows.
DEFAULT_JUDGE = "Qwen/Qwen3-VL-8B-Instruct"
# Detector+recognizer OCR, NOT a VLM (see bench.ocr for why). ~35 MB all-in.
DEFAULT_OCR = "rapidocr:PP-OCRv5-latin"

OCR_FUZZY_PASS = 0.85  # normalized similarity at which an OCR item counts as present

KIND_OCR = "ocr"
KIND_VQA = "vqa"
KINDS = (KIND_OCR, KIND_VQA)


class ChecklistItem(msgspec.Struct, frozen=True, kw_only=True):
    id: str
    kind: str
    text: str = ""        # kind=ocr: the literal string that must be readable
    question: str = ""    # kind=vqa: a yes/no question; YES means "prompt obeyed"
    absent: bool = False  # invert: the string/fact must NOT be present
    weight: float = 1.0


class Checklist(msgspec.Struct, frozen=True, kw_only=True):
    prompt_id: str
    items: tuple[ChecklistItem, ...]


class EditChecklist(msgspec.Struct, frozen=True, kw_only=True):
    edit_id: str
    # The t2i prompt whose RENDER is the before image. "" when the before is an
    # external image (a real photo from an edit corpus), which is not a defect —
    # the magicbrush rows are exactly that.
    before_from: str = ""
    instruction: str = ""
    change: tuple[ChecklistItem, ...]
    preserve: tuple[ChecklistItem, ...]


class VideoChecklist(msgspec.Struct, frozen=True, kw_only=True):
    """A t2v prompt's checklist — the motion/hold duality, mirroring edit's
    change/preserve: a video can fail by NOT MOVING (the asked-for dynamics
    never happen; a still image scores perfectly on static items) or by NOT
    HOLDING (identity drifts, the background quietly reshuffles). Judged on an
    ordered strip of sampled frames, because neither is answerable from one."""

    prompt_id: str
    motion: tuple[ChecklistItem, ...]    # what must HAPPEN across frames
    hold: tuple[ChecklistItem, ...]      # what must STAY the same across frames


class ChecklistSet(msgspec.Struct, frozen=True, kw_only=True):
    checklist_set_id: str
    version: int
    prompt_set: str
    prompt_set_version: int
    t2i: dict[str, Checklist]
    edit: dict[str, EditChecklist]
    t2v: dict[str, VideoChecklist] = {}


class ItemVerdict(msgspec.Struct, kw_only=True):
    id: str
    kind: str
    verified: bool
    score: float          # 1.0/0.0 for vqa; fuzzy similarity for ocr
    detail: str = ""


class AdherenceScore(msgspec.Struct, kw_only=True):
    """One image (t2i) or one before/after pair (edit)."""

    measured: bool = False
    element_recall: float = 0.0
    text_exact: float = 0.0
    text_fuzzy: float = 0.0
    compliance: float = 0.0     # edit mode only
    preservation: float = 0.0   # edit mode only
    items: list[ItemVerdict] = msgspec.field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# checklist loading + validation
# ---------------------------------------------------------------------------

def _items(rows: Any, *, where: str, seen: set[str]) -> tuple[ChecklistItem, ...]:
    out: list[ChecklistItem] = []
    for row in rows or ():
        item = msgspec.convert(row, ChecklistItem)
        if item.kind not in KINDS:
            raise DataError(
                f"{where}: item {item.id!r} has unknown kind {item.kind!r}; "
                f"known: {list(KINDS)}"
            )
        if item.kind == KIND_OCR and not item.text.strip():
            raise DataError(
                f"{where}: ocr item {item.id!r} has no 'text' — an ocr item needs the "
                "literal string that must be readable in the image"
            )
        if item.kind == KIND_VQA and not item.question.strip():
            raise DataError(
                f"{where}: vqa item {item.id!r} has no 'question' — a vqa item needs a "
                "yes/no question phrased so that YES means the prompt was obeyed"
            )
        if item.weight <= 0.0:
            raise DataError(
                f"{where}: item {item.id!r} has weight {item.weight}; weights must be > 0"
            )
        if item.id in seen:
            raise DataError(
                f"{where}: duplicate item id {item.id!r} — item ids must be unique "
                "across the whole checklist set, because a report names failed items by id"
            )
        seen.add(item.id)
        out.append(item)
    if not out:
        raise DataError(f"{where}: empty checklist (no items)")
    return tuple(out)


def load_checklists(path: str | Path) -> ChecklistSet:
    """Parse and VALIDATE a checklist file. A malformed checklist must fail
    loudly here rather than silently score zero at eval time."""
    path = Path(path)
    try:
        spec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: not valid JSON: {exc}") from exc
    except OSError as exc:
        raise DataError(f"{path}: cannot be read: {exc}") from exc
    for key in ("checklist_set_id", "version", "prompt_set", "prompt_set_version"):
        if key not in spec:
            raise DataError(
                f"{path}: checklist set is missing required key {key!r}; a checklist set "
                "must name itself and the prompt set + version it was authored against"
            )
    seen: set[str] = set()
    t2i = {}
    for row in spec.get("t2i", ()):
        pid = str(row["prompt_id"])
        t2i[pid] = Checklist(
            prompt_id=pid, items=_items(row.get("items"), where=pid, seen=seen),
        )
    edit = {}
    for row in spec.get("edit", ()):
        eid = str(row["edit_id"])
        before = str(row.get("before_from") or "")
        if before and before not in t2i:
            raise DataError(
                f"{eid}: before_from={before!r} names no t2i checklist in this set; "
                f"known: {sorted(t2i)}"
            )
        edit[eid] = EditChecklist(
            edit_id=eid, before_from=before,
            instruction=str(row.get("instruction") or ""),
            change=_items(row.get("change"), where=f"{eid}.change", seen=seen),
            preserve=_items(row.get("preserve"), where=f"{eid}.preserve", seen=seen),
        )
    t2v = {}
    for row in spec.get("t2v", ()):
        pid = str(row["prompt_id"])
        t2v[pid] = VideoChecklist(
            prompt_id=pid,
            motion=_items(row.get("motion"), where=f"{pid}.motion", seen=seen),
            hold=_items(row.get("hold"), where=f"{pid}.hold", seen=seen),
        )
    if not (t2i or edit or t2v):
        raise DataError(
            f"{path}: checklist set is empty (no t2i, edit or t2v entries)"
        )
    return ChecklistSet(
        checklist_set_id=str(spec["checklist_set_id"]),
        version=int(spec["version"]),
        prompt_set=str(spec["prompt_set"]),
        prompt_set_version=int(spec["prompt_set_version"]),
        t2i=t2i, edit=edit, t2v=t2v,
    )


# ---------------------------------------------------------------------------
# OCR scoring
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w一-鿿぀-ヿ]+", re.UNICODE)


def normalize_text(value: str) -> str:
    """Case/width/punctuation-insensitive form for comparing rendered glyphs.

    Generators legitimately vary spacing and punctuation shape (an em dash for
    a hyphen); those are not spelling errors and must not read as ones.
    """
    folded = unicodedata.normalize("NFKC", str(value)).upper()
    return " ".join(_PUNCT.sub(" ", folded).split())


def text_match(target: str, page: str) -> tuple[bool, float]:
    """(exact, fuzzy) for one target string against all OCR text of an image."""
    t, p = normalize_text(target), normalize_text(page)
    if not t:
        return False, 0.0
    if t in p:
        return True, 1.0
    # Best window of the page the same length as the target.
    best = difflib.SequenceMatcher(None, t, p).ratio() if p else 0.0
    words = p.split()
    span = len(t.split())
    for i in range(max(0, len(words) - span + 1)):
        window = " ".join(words[i:i + span])
        best = max(best, difflib.SequenceMatcher(None, t, window).ratio())
    return False, float(best)


def score_ocr_items(
    items: tuple[ChecklistItem, ...], page_text: str,
) -> list[ItemVerdict]:
    out = []
    for item in items:
        exact, fuzzy = text_match(item.text, page_text)
        present = exact or fuzzy >= OCR_FUZZY_PASS
        verified = (not present) if item.absent else present
        out.append(ItemVerdict(
            id=item.id, kind=KIND_OCR, verified=verified,
            score=(1.0 - fuzzy) if item.absent else fuzzy,
            detail=f"{'absent-required ' if item.absent else ''}exact={exact} fuzzy={fuzzy:.2f}",
        ))
    return out


# ---------------------------------------------------------------------------
# judge VQA — ONE structured query per image
# ---------------------------------------------------------------------------

_JUDGE_PREAMBLE = (
    "You are grading whether an image generator followed its instructions. "
    "Answer each numbered question about the image(s) with yes or no. "
    "Judge only what is visible; if an element is absent or wrong, answer no. "
    'Reply with ONLY a JSON array like [{"n": 1, "answer": "yes"}, '
    '{"n": 2, "answer": "no"}] — one object per question, no other text.'
)


def build_judge_prompt(
    items: tuple[ChecklistItem, ...], *, instruction: str = "", paired: bool = False,
) -> str:
    lines = [_JUDGE_PREAMBLE]
    if paired:
        lines.append(
            "You are shown TWO images: the FIRST is the BEFORE image, the SECOND "
            "is the AFTER image produced by an edit."
        )
    if instruction:
        lines.append(f"The edit instruction given to the generator was: {instruction!r}")
    lines.append("Questions:")
    lines += [f"{i}. {item.question}" for i, item in enumerate(items, start=1)]
    return "\n".join(lines)


def parse_judge_answers(raw: str, count: int) -> dict[int, bool]:
    """Tolerant parse of the judge's reply; unanswered questions are absent
    (they become unverified, never silently true)."""
    out: dict[int, bool] = {}
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            for row in json.loads(match.group(0)):
                n = int(row.get("n", 0))
                ans = str(row.get("answer", "")).strip().lower()
                if 1 <= n <= count and ans in ("yes", "no", "true", "false"):
                    out[n] = ans in ("yes", "true")
        except (ValueError, TypeError, AttributeError):
            pass
    if out:
        return out
    for n, ans in re.findall(r"(\d+)\s*[.:)-]\s*(yes|no)", raw, re.IGNORECASE):
        idx = int(n)
        if 1 <= idx <= count:
            out[idx] = ans.lower() == "yes"
    return out


def resident_vram() -> int:
    """Bytes currently allocated on the GPU, 0 on CPU — so a report can answer
    "does this need a big model in VRAM" with a measurement."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_allocated())
    except Exception:
        return 0


class VlmJudge:
    """A resident local judge VLM — the reference :class:`cozy_eval.bench.judge.Judge`.

    Loaded once per pass, on the machine that rendered. Anything else with an
    ``ask(images, prompt) -> str`` method is equally a judge; nothing here is
    privileged.
    """

    def __init__(self, model_ref: str, device: str = AUTO, max_new_tokens: int = 512):
        import time

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_ref = model_ref
        self.device = resolve_device(device)
        device = self.device
        self.max_new_tokens = max_new_tokens
        # The judge is the expensive part of the pass, so it measures
        # itself — load vs inference, call count, and resident VRAM.
        self.load_seconds = 0.0
        self.infer_seconds = 0.0
        self.call_count = 0
        self.vram_bytes = 0
        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(model_ref)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_ref, dtype="auto", device_map=device,
        ).eval()
        self.load_seconds = time.monotonic() - started
        self.vram_bytes = resident_vram()

    def ask(self, images: list[Any], prompt: str) -> str:
        import time

        import torch

        _started = time.monotonic()

        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False,
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        reply = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0]
        self.infer_seconds += time.monotonic() - _started
        self.call_count += 1
        return str(reply)


def score_vqa_items(
    judge: Any, images: list[Any], items: tuple[ChecklistItem, ...],
    *, instruction: str = "", paired: bool = False,
) -> list[ItemVerdict]:
    """One judge call for the whole item list."""
    if not items:
        return []
    prompt = build_judge_prompt(items, instruction=instruction, paired=paired)
    answers = parse_judge_answers(judge.ask(images, prompt), len(items))
    out = []
    for i, item in enumerate(items, start=1):
        answered = answers.get(i)
        yes = bool(answered)
        verified = (not yes) if item.absent else yes
        out.append(ItemVerdict(
            id=item.id, kind=KIND_VQA, verified=verified,
            score=1.0 if verified else 0.0,
            detail="unanswered" if answered is None else ("yes" if yes else "no"),
        ))
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _recall(items: tuple[ChecklistItem, ...], verdicts: list[ItemVerdict]) -> float:
    weights = {i.id: i.weight for i in items}
    total = sum(weights.get(v.id, 0.0) for v in verdicts)
    if total <= 0.0:
        return 0.0
    hit = sum(weights.get(v.id, 0.0) for v in verdicts if v.verified)
    return hit / total


def _text_rates(
    items: tuple[ChecklistItem, ...], verdicts: list[ItemVerdict],
) -> tuple[float, float]:
    ocr = [v for v in verdicts if v.kind == KIND_OCR]
    if not ocr:
        return 0.0, 0.0
    exact = sum(1.0 for v in ocr if v.verified) / len(ocr)
    fuzzy = sum(v.score for v in ocr) / len(ocr)
    return exact, fuzzy


def score_t2i(
    checklist: Checklist, image: Any, *, judge: Any = None, page_text: str | None = None,
) -> AdherenceScore:
    """Element fidelity for one t2i render. ``judge``/``page_text`` may be None
    on an image with no judge or no OCR — the missing half is reported as
    unmeasured rather than scored zero."""
    ocr_items = tuple(i for i in checklist.items if i.kind == KIND_OCR)
    vqa_items = tuple(i for i in checklist.items if i.kind == KIND_VQA)
    verdicts: list[ItemVerdict] = []
    missing = []
    if ocr_items:
        if page_text is None:
            missing.append(f"{len(ocr_items)} ocr items (no OCR reader)")
        else:
            verdicts += score_ocr_items(ocr_items, page_text)
    if vqa_items:
        if judge is None:
            missing.append(f"{len(vqa_items)} vqa items (no judge)")
        else:
            verdicts += score_vqa_items(judge, [image], vqa_items)
    if not verdicts:
        return AdherenceScore(measured=False, note="; ".join(missing) or "empty checklist")
    scored = tuple(i for i in checklist.items if any(v.id == i.id for v in verdicts))
    exact, fuzzy = _text_rates(scored, verdicts)
    return AdherenceScore(
        measured=True, element_recall=_recall(scored, verdicts),
        text_exact=exact, text_fuzzy=fuzzy, items=verdicts,
        note=("partial: " + "; ".join(missing)) if missing else "",
    )


def score_edit(
    checklist: EditChecklist, before: Any, after: Any, instruction: str,
    *, judge: Any = None, after_text: str | None = None,
) -> AdherenceScore:
    """Instruction compliance + preservation for one before/after pair.

    The judge sees BOTH images in one call per group; OCR items are read from
    the AFTER image (that is where the required text must or must not appear).
    """
    groups = (("change", checklist.change), ("preserve", checklist.preserve))
    per_group: dict[str, float] = {}
    verdicts: list[ItemVerdict] = []
    missing = []
    for name, items in groups:
        ocr_items = tuple(i for i in items if i.kind == KIND_OCR)
        vqa_items = tuple(i for i in items if i.kind == KIND_VQA)
        group_verdicts: list[ItemVerdict] = []
        if ocr_items:
            if after_text is None:
                missing.append(f"{len(ocr_items)} {name} ocr items (no OCR reader)")
            else:
                group_verdicts += score_ocr_items(ocr_items, after_text)
        if vqa_items:
            if judge is None:
                missing.append(f"{len(vqa_items)} {name} vqa items (no judge)")
            else:
                group_verdicts += score_vqa_items(
                    judge, [before, after], vqa_items,
                    instruction=instruction, paired=True,
                )
        if group_verdicts:
            scored = tuple(i for i in items if any(v.id == i.id for v in group_verdicts))
            per_group[name] = _recall(scored, group_verdicts)
        verdicts += group_verdicts
    if not verdicts:
        return AdherenceScore(measured=False, note="; ".join(missing) or "empty checklist")
    all_items = checklist.change + checklist.preserve
    scored = tuple(i for i in all_items if any(v.id == i.id for v in verdicts))
    exact, fuzzy = _text_rates(scored, verdicts)
    return AdherenceScore(
        measured=True,
        element_recall=_recall(scored, verdicts),
        compliance=per_group.get("change", 0.0),
        preservation=per_group.get("preserve", 0.0),
        text_exact=exact, text_fuzzy=fuzzy, items=verdicts,
        note=("partial: " + "; ".join(missing)) if missing else "",
    )


__all__ = [
    "DEFAULT_JUDGE",
    "DEFAULT_OCR",
    "KINDS",
    "KIND_OCR",
    "KIND_VQA",
    "OCR_FUZZY_PASS",
    "AdherenceScore",
    "Checklist",
    "ChecklistItem",
    "ChecklistSet",
    "EditChecklist",
    "ItemVerdict",
    "VideoChecklist",
    "VlmJudge",
    "build_judge_prompt",
    "load_checklists",
    "normalize_text",
    "parse_judge_answers",
    "resident_vram",
    "score_edit",
    "score_ocr_items",
    "score_t2i",
    "score_vqa_items",
    "text_match",
]
