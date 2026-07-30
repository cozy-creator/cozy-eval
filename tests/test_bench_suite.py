"""Benchmark suite: checklists, adherence scoring, the compliance/preservation
duality.

Integration style — the real checklist file, real PIL images, the real metric
models where they are cheap (SSIM/MS-SSIM on small images). The judge VLM is
the one thing stubbed: it is a 7B model that only exists pod-side, so these
tests drive the ORCHESTRATION around it (prompt shape, answer parsing, the
compliance/preservation split) rather than pretending to run it.

Registry wiring is NOT re-asserted here — `test_public_api.py` sweeps the whole
registry in one test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from cozy_eval.bench import errors, promptset
from cozy_eval.bench.metrics import adherence, similarity

SET = "hard-eval-v1"


def _noise_image(seed: int, size: int = 128) -> Image.Image:
    import numpy as np

    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    )


def _photoish(size: int = 128) -> Image.Image:
    img = Image.new("RGB", (size, size), (40, 60, 90))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 90, 90], fill=(220, 180, 90))
    draw.rectangle([0, size - 30, size, size], fill=(30, 30, 30))
    draw.text((10, size - 25), "KIRKWOOD", fill=(240, 240, 240))
    return img


# ---------------------------------------------------------------------------
# SSIM / MS-SSIM
# ---------------------------------------------------------------------------

def test_the_ssim_family_is_one_at_identity_and_falls_as_damage_increases() -> None:
    """torchmetrics computes in float32, so identity lands at ~0.9999 rather
    than exactly 1.0. That is the library's precision, not a defect — the
    tolerance records it so a real regression still shows."""
    import numpy as np

    img = _photoish()
    assert similarity.ssim(img, img) == pytest.approx(1.0, abs=1e-3)
    assert similarity.ms_ssim(img, img) == pytest.approx(1.0, abs=1e-3)
    assert similarity.psnr(img, img) == 99.0  # identical -> the finite sentinel

    base = np.asarray(img, dtype=np.float64)
    rng = np.random.default_rng(7)
    scores = []
    for sigma in (0.0, 5.0, 15.0, 40.0):
        noisy = np.clip(base + rng.normal(0, sigma, base.shape), 0, 255)
        scores.append(similarity.ssim(
            Image.fromarray(base.astype("uint8")),
            Image.fromarray(noisy.astype("uint8")),
        ))
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] > 0.99 and scores[-1] < 0.5

    # Too small for five scales: MS-SSIM degrades to fewer, it does not raise.
    assert 0.0 <= similarity.ms_ssim(_photoish(48), _noise_image(1, 48)) < 1.0


def test_ssim_rejects_mismatched_and_undersized_images() -> None:
    with pytest.raises(errors.ConfigError, match="differ in shape"):
        similarity.ssim(_photoish(64), _photoish(128))
    with pytest.raises(errors.ConfigError, match="smaller than"):
        similarity.ssim(_photoish(8), _photoish(8))


# ---------------------------------------------------------------------------
# checklists
# ---------------------------------------------------------------------------

def test_hard_eval_v1_is_internally_consistent() -> None:
    """One comprehensive validator for the shipped set. The authored checklists
    and the prompt set are versioned together, so a prompt with no checklist
    (or the reverse) is an authoring bug; every quoted literal is a
    readable-text axis that OCR must check rather than a VLM's impression; and
    every edit row carries BOTH halves of the duality against the before-image
    the prompt set names."""
    import re

    loaded = promptset.checklists_for(SET)
    ps = promptset.load(SET)

    assert loaded.prompt_set == ps.set_id
    assert set(loaded.t2i) == {p.id for p in ps.t2i}
    assert set(loaded.edit) == {e.id for e in ps.edit}
    assert all(c.items for c in loaded.t2i.values())

    for prompt in ps.t2i:
        for literal in set(re.findall(r'"([^"]+)"', prompt.prompt)):
            norm = adherence.normalize_text(literal)
            if len(norm) <= 1:  # single compass letters live in a vqa item
                continue
            ocr_texts = {
                adherence.normalize_text(i.text)
                for i in loaded.t2i[prompt.id].items if i.kind == adherence.KIND_OCR
            }
            assert any(norm in t or t in norm for t in ocr_texts), (
                f"{prompt.id}: quoted {literal!r} has no ocr item"
            )

    for row in ps.edit:
        entry = loaded.edit[row.id]
        assert entry.before_from == row.before_from
        assert entry.change and entry.preserve
        assert len(entry.change) >= 2 and len(entry.preserve) >= 3


def _t2i_doc(items: list[dict]) -> dict:
    return {
        "checklist_set_id": "x", "version": 1, "prompt_set": "p",
        "prompt_set_version": 1,
        "t2i": [{"prompt_id": "t01", "items": items}],
    }


def _item(**overrides: object) -> dict:
    return {"id": "a", "kind": "vqa", "question": "is it?", **overrides}


@pytest.mark.parametrize("document,message", [
    pytest.param(_t2i_doc([_item(kind="vibes")]), "unknown kind", id="unknown_kind"),
    pytest.param(_t2i_doc([_item(kind="ocr", text="  ")]), "has no 'text'",
                 id="blank_ocr_text"),
    pytest.param(_t2i_doc([_item(question="")]), "has no 'question'",
                 id="blank_vqa_question"),
    pytest.param(_t2i_doc([_item(weight=0.0)]), "weights must be > 0",
                 id="zero_weight"),
    pytest.param(
        {"checklist_set_id": "x", "version": 1, "prompt_set": "p",
         "prompt_set_version": 1,
         "t2i": [{"prompt_id": "t01", "items": [_item()]},
                 {"prompt_id": "t02", "items": [_item()]}]},
        "duplicate item id", id="duplicate_item_id_across_prompts",
    ),
    pytest.param(
        {**_t2i_doc([_item()]),
         "edit": [{"edit_id": "e01", "before_from": "t99",
                   "change": [_item(id="c")], "preserve": [_item(id="p")]}]},
        "names no t2i checklist", id="edit_row_points_at_no_prompt",
    ),
    pytest.param(
        {"checklist_set_id": "x", "version": 1, "prompt_set": "p",
         "prompt_set_version": 1,
         "t2v": [{"prompt_id": "v01", "motion": [],
                  "hold": [_item(id="h1", question="holds?")]}]},
        "empty checklist", id="empty_t2v_motion_group",
    ),
])
def test_a_malformed_checklist_fails_at_load(
    tmp_path: Path, document: dict, message: str,
) -> None:
    """A broken checklist must fail at LOAD, not silently score zero at eval
    time — a zero reads as 'the model rendered nothing'."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        adherence.load_checklists(path)


# ---------------------------------------------------------------------------
# OCR scoring
# ---------------------------------------------------------------------------

def test_text_match_is_insensitive_to_punctuation_and_case_but_not_spelling() -> None:
    exact, fuzzy = adherence.text_match("ETHIOPIA — GUJI", "single origin\nethiopia - guji")
    assert exact and fuzzy == 1.0
    # A real misspelling is not forgiven.
    _, fuzzy_bad = adherence.text_match("KIRKWOOD JUNCTION", "KLRKWOOB JUNCTOIN")
    assert fuzzy_bad < adherence.OCR_FUZZY_PASS
    # A missing word is a miss, not a pass.
    _, fuzzy_missing = adherence.text_match("HALF PRICE AFTER SIX", "HALF PRICE")
    assert fuzzy_missing < adherence.OCR_FUZZY_PASS


def test_absent_ocr_items_invert_the_verdict() -> None:
    """Edit removals ('the sign no longer says BASEMENT') are scored by the
    same machinery, inverted."""
    items = (
        adherence.ChecklistItem(id="gone", kind="ocr", text="BASEMENT", absent=True),
    )
    removed = adherence.score_ocr_items(items, "KOBAYASHI\nOPEN LATE")
    assert removed[0].verified
    still_there = adherence.score_ocr_items(items, "BASEMENT\nOPEN LATE")
    assert not still_there[0].verified


# ---------------------------------------------------------------------------
# judge orchestration
# ---------------------------------------------------------------------------

class StubJudge:
    """Stands in for the pod-side 7B judge. Records what it was asked so the
    cost contract (ONE call per image, not one per item) is testable."""

    model_ref = "stub-judge"

    def __init__(self, answers: dict[str, bool], reply_style: str = "json"):
        self.answers = answers
        self.reply_style = reply_style
        self.calls: list[tuple[int, str]] = []

    def ask(self, images: list, prompt: str) -> str:
        self.calls.append((len(images), prompt))
        questions = [
            line for line in prompt.splitlines()
            if line and line[0].isdigit() and ". " in line
        ]
        rows = []
        for n, line in enumerate(questions, start=1):
            text = line.split(". ", 1)[1]
            yes = self.answers.get(text, True)
            rows.append((n, "yes" if yes else "no"))
        if self.reply_style == "json":
            return json.dumps([{"n": n, "answer": a} for n, a in rows])
        return "\n".join(f"{n}. {a}" for n, a in rows)


def test_judge_asks_one_batched_question_set_per_image() -> None:
    """Cost discipline: a 10-item checklist is ONE forward pass, not ten."""
    loaded = promptset.checklists_for(SET)
    checklist = loaded.t2i["t03"]  # all-vqa, no quoted text
    judge = StubJudge({})
    score = adherence.score_t2i(checklist, _photoish(), judge=judge, page_text=None)
    assert len(judge.calls) == 1
    images, prompt = judge.calls[0]
    assert images == 1
    assert prompt.count("\n1. ") == 1
    assert len(checklist.items) >= 8
    assert score.measured and score.element_recall == 1.0


def test_judge_answers_parse_from_json_or_loose_text() -> None:
    items = tuple(
        adherence.ChecklistItem(id=f"i{n}", kind="vqa", question=f"q{n}?")
        for n in range(1, 4)
    )
    for style in ("json", "loose"):
        judge = StubJudge({"q2?": False}, reply_style=style)
        verdicts = adherence.score_vqa_items(judge, [_photoish()], items)
        assert [v.verified for v in verdicts] == [True, False, True], style


def test_unanswered_questions_are_unverified_not_silently_true() -> None:
    parsed = adherence.parse_judge_answers('[{"n": 1, "answer": "yes"}]', 3)
    assert parsed == {1: True}

    class SilentJudge:
        model_ref = "silent"

        def ask(self, images: list, prompt: str) -> str:
            return "I'd rather not say."

    items = tuple(
        adherence.ChecklistItem(id=f"i{n}", kind="vqa", question=f"q{n}?")
        for n in range(1, 4)
    )
    verdicts = adherence.score_vqa_items(SilentJudge(), [_photoish()], items)
    assert not any(v.verified for v in verdicts)
    assert all(v.detail == "unanswered" for v in verdicts)


def test_adherence_says_what_it_could_not_see() -> None:
    """The distinction that keeps a report honest: 'nobody looked' must never
    render as 'the model scored 0', and a score taken with one channel missing
    says so rather than passing itself off as complete."""
    checklist = promptset.checklists_for(SET).t2i["t01"]

    blind = adherence.score_t2i(checklist, _photoish(), judge=None, page_text=None)
    assert not blind.measured and blind.element_recall == 0.0
    assert "no judge" in blind.note and "no OCR" in blind.note

    ocr_only = adherence.score_t2i(
        checklist, _photoish(), judge=None,
        page_text="MORNING BAKE\nSOURDOUGH 4.50\nALMOND CROISSANT 3.75\n"
                  "CARDAMOM BUN 4.20\nSOLD OUT BY NOON",
    )
    assert ocr_only.measured and ocr_only.element_recall == 1.0
    assert ocr_only.text_exact == 1.0
    assert ocr_only.note.startswith("partial:") and "no judge" in ocr_only.note


# ---------------------------------------------------------------------------
# edit mode: the compliance / preservation duality
# ---------------------------------------------------------------------------

OLD_BOARD = ("MORNING BAKE\nSOURDOUGH 4.50\nALMOND CROISSANT 3.75\n"
             "CARDAMOM BUN 4.20\nSOLD OUT BY NOON")
NEW_BOARD = ("EVENING BAKE\nSOURDOUGH 4.50\nALMOND CROISSANT 3.75\n"
             "CARDAMOM BUN 4.20\nHALF PRICE AFTER SIX")
REPAINTED = "EVENING BAKE\nHALF PRICE AFTER SIX"

# The preserve questions of e01, answered "no" = the generator repainted
# everything it should have left alone.
PRESERVE_VIOLATED = {
    "Is the chalk texture and hand-lettering style the same in both images?": False,
    "Is the chalk wheat-sheaf drawing still present in the lower right corner, "
    "unchanged?": False,
    "Are the board's shape, position and the out-of-focus background unchanged "
    "between the two images?": False,
}


@pytest.mark.parametrize("answers,after_text,complied,preserved", [
    pytest.param({}, NEW_BOARD, True, True, id="a_perfect_edit_scores_both"),
    pytest.param({}, OLD_BOARD, False, True,
                 id="under_edit_fails_compliance_while_preservation_stays_clean"),
    pytest.param(PRESERVE_VIOLATED, REPAINTED, True, False,
                 id="over_edit_fails_preservation_while_compliance_is_clean"),
])
def test_the_edit_duality(
    answers: dict, after_text: str, complied: bool, preserved: bool,
) -> None:
    """Edit adherence is a duality, and the two halves must move independently:
    under-edit fails compliance, over-edit fails preservation."""
    loaded = promptset.checklists_for(SET)
    instruction = next(e for e in promptset.load(SET).edit if e.id == "e01").instruction
    score = adherence.score_edit(
        loaded.edit["e01"], _photoish(), _photoish(), instruction,
        judge=StubJudge(answers), after_text=after_text,
    )
    assert score.measured
    assert (score.compliance == 1.0) if complied else (score.compliance < 0.5)
    assert (score.preservation == 1.0) if preserved else (score.preservation < 0.5)


def test_edit_judge_sees_both_images_and_the_instruction() -> None:
    loaded = promptset.checklists_for(SET)
    judge = StubJudge({})
    adherence.score_edit(
        loaded.edit["e02"], _photoish(), _noise_image(3),
        "Make her smile warmly", judge=judge, after_text=None,
    )
    assert judge.calls, "the judge was never asked"
    for images, prompt in judge.calls:
        assert images == 2, "'did X change' is unanswerable from one image"
        assert "BEFORE" in prompt and "AFTER" in prompt
        assert "Make her smile warmly" in prompt
