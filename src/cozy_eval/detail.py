"""Fine-detail mode: the VLM rubric, the numeric pre-screen, and ONE detail verdict.

STABILITY: locked core (v0.x) for :func:`detail_verdict`'s call shape and the
:class:`DetailVerdict` it returns.

WHY THIS EXISTS. Our flat-region-LSB, coherence and adjacent-frame screens all
passed a composed H3 stack whose faces had melted, whose signage was pseudo-glyph
mush, and whose high-contrast edges rang with halos — because those defects are
TEMPORALLY COHERENT and live inside each frame. This module is the instrument
for that class.

THREE TIERS, and the ordering is deliberate:

* the VLM DETAIL RUBRIC is the PRIMARY catch. A vision-language model sees
  'melted face' and 'fake letters' the way a person does; four axes — text
  legibility, face/hand coherence, edge cleanliness, texture realism — scored
  per clip over the same ordered frame strip :mod:`cozy_eval.video` already
  builds. It is UNMEASURED without a :class:`~cozy_eval.judge.Judge`, exactly
  like the audio semantic tier is UNMEASURED without a transcriber.

* the NUMERIC PRE-SCREEN is the cheap filter. Reference-free ``text_legibility``
  (OCR confidence over detected text) and, when a same-seed control render
  exists, the paired ``dists`` / ``ringing_excess``. VALIDATED to separate the
  ie#634 labeled pairs — but the reference-free numbers are a SCREEN, never a
  gate on their own (a bare high-frequency number tracks scene, not quality; see
  :mod:`cozy_eval.metrics.detail`). The pre-screen may FLAG a clip for the VLM;
  it may never PASS one alone.

* the PAIRWISE VLM read, for the parity case: shown the candidate strip and the
  reference strip, which has the better fine detail. This is the capability-side
  read the quant/parity verdict wants when pixels cannot match.

Tri-state, like every verdict here: ``pass`` / ``reject`` / ``unmeasured``, and
UNMEASURED is never silently a pass.
"""

from __future__ import annotations

import time
from typing import Any

import msgspec

from .judge import Judge
from .metrics import detail as detail_metrics

PASS = "pass"
REJECT = "reject"
UNMEASURED = "unmeasured"

#: The four fine-detail axes. Each is a yes/no the VLM answers about the clip's
#: WORST offending region — 'yes' means the axis is CLEAN.
DETAIL_AXES: tuple[tuple[str, str], ...] = (
    ("detail_text_legible",
     ("Is every piece of TEXT / signage in these frames made of real, readable "
      "letters or characters (not text-like scribbles or fake glyphs)?")),
    ("detail_face_coherent",
     ("Is every FACE and HAND coherent and correctly formed (not melted, "
      "smeared, or with the wrong number of fingers)?")),
    ("detail_edge_clean",
     ("Are the high-contrast EDGES clean (free of bright/dark halos, ringing, or "
      "oversharpening rims beside them)?")),
    ("detail_texture_real",
     ("Do surfaces and fine TEXTURE look like real material (not mushy, "
      "wobbly, or hand-drawn-looking degradation)?")),
)

_DETAIL_PREAMBLE = (
    "You are grading the FINE-DETAIL FIDELITY of a generated video. You are "
    "shown {n} frames sampled in temporal order from one clip. Look CLOSELY at "
    "small regions — text, faces, hands, edges, and textures — not the overall "
    "composition. For each numbered question answer 'yes' only if it holds "
    "across the frames; if even one sampled frame clearly violates it, answer "
    "'no'. Reply with ONLY a JSON array like "
    '[{{"n": 1, "answer": "yes"}}, {{"n": 2, "answer": "no"}}] — one object per '
    "question, no other text."
)

_PAIRWISE_PREAMBLE = (
    "You are comparing the FINE-DETAIL FIDELITY of two generated videos of the "
    "same scene. The FIRST {n} frames are clip A; the next {n} frames are clip "
    "B. Which clip has better fine detail — more readable text, more coherent "
    "faces and hands, cleaner edges, more realistic texture? Reply with ONLY a "
    'JSON object like {{"winner": "A"}} or {{"winner": "B"}} or '
    '{{"winner": "tie"}} — no other text.'
)


def build_detail_judge_prompt(*, frame_count: int) -> str:
    lines = [_DETAIL_PREAMBLE.format(n=frame_count), "Questions:"]
    lines += [f"{i}. {q}" for i, (_, q) in enumerate(DETAIL_AXES, start=1)]
    return "\n".join(lines)


def score_detail_vlm(strip: list[Any], judge: Judge) -> dict[str, float]:
    """One judge call over a frame strip -> {axis: 1.0 clean | 0.0 defect}.

    Missing/unparseable answers are dropped (the axis is simply absent from the
    result), never scored 0 — an unanswered axis is UNMEASURED, not a failure.
    """
    from .metrics.adherence import parse_judge_answers

    prompt = build_detail_judge_prompt(frame_count=len(strip))
    answers = parse_judge_answers(judge.ask(strip, prompt), len(DETAIL_AXES))
    out: dict[str, float] = {}
    for i, (name, _) in enumerate(DETAIL_AXES, start=1):
        ans = answers.get(i)
        if ans is None:
            continue
        out[name] = 1.0 if ans else 0.0
    return out


def judge_detail_pairwise(cand_strip: list[Any], ref_strip: list[Any], judge: Judge) -> float:
    """Pairwise fine-detail preference: +1 candidate wins, -1 reference wins, 0 tie.

    Adapts the te#176 clip-judge shape: the two strips are concatenated (A =
    candidate, B = reference) into ONE call. UNMEASURED (returns NaN) if the
    reply names no winner.
    """
    import json
    import re

    n = min(len(cand_strip), len(ref_strip))
    prompt = _PAIRWISE_PREAMBLE.format(n=n)
    raw = judge.ask(list(cand_strip[:n]) + list(ref_strip[:n]), prompt)
    m = re.search(r"\{[^{}]*\}", raw or "")
    if not m:
        return float("nan")
    try:
        winner = str(json.loads(m.group(0)).get("winner", "")).strip().lower()
    except (ValueError, TypeError):
        return float("nan")
    return {"a": 1.0, "b": -1.0, "tie": 0.0}.get(winner, float("nan"))


class DetailVerdict(msgspec.Struct, kw_only=True):
    """One clip's fine-detail answer. ``measured`` and ``unmeasured`` are
    disjoint and together cover every number this function can produce."""

    verdict: str = UNMEASURED
    defects: list[str] = msgspec.field(default_factory=list)
    measured: dict[str, float] = msgspec.field(default_factory=dict)
    unmeasured: dict[str, str] = msgspec.field(default_factory=dict)
    flags: list[str] = msgspec.field(default_factory=list)
    notes: list[str] = msgspec.field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        head = f"detail {self.verdict.upper()}"
        if self.defects:
            head += " — " + "; ".join(self.defects)
        return f"{head} ({len(self.measured)} measured, {len(self.unmeasured)} unmeasured)"


def detail_verdict(
    strip: list[Any],
    *,
    reference_strip: list[Any] | None = None,
    judge: Judge | None = None,
    device: str = "cpu",
    ocr_min_conf: float = detail_metrics.LEGIBLE_CONF,
) -> DetailVerdict:
    """Score one clip's fine detail from its sampled frame strip.

    ``strip`` / ``reference_strip`` are lists of PIL images or ``(H,W,3)`` arrays
    — the same strip :func:`cozy_eval.video.run_video` samples. A verdict is
    ``reject`` when the VLM marks any axis as a defect OR the numeric pre-screen
    finds an illegible-text region; ``pass`` when something was measured and
    clean; ``unmeasured`` when nothing could be scored.
    """
    t0 = time.monotonic()
    result = DetailVerdict()

    # --- tier 1: numeric pre-screen (reference-free) ------------------------
    leg = detail_metrics.clip_text_legibility(strip, min_conf=ocr_min_conf)
    if leg.measured:
        result.measured["text_legibility"] = leg.mean_conf
        result.measured["text_legible_fraction"] = (
            leg.n_legible / leg.n_regions if leg.n_regions else 0.0
        )
        if leg.mean_conf < ocr_min_conf:
            result.flags.append(
                f"text_legibility {leg.mean_conf:.2f} < {ocr_min_conf:.2f} over "
                f"{leg.n_regions} detected region(s) — pseudo-glyph suspect, send to VLM"
            )
    else:
        result.unmeasured["text_legibility"] = leg.note

    overshoot = [detail_metrics.edge_overshoot(f) for f in strip]
    if overshoot:
        result.measured["edge_overshoot"] = float(sum(overshoot) / len(overshoot))

    # --- tier 2: reference-based (paired to a same-seed control) ------------
    if reference_strip is None:
        for name in ("dists", "ringing_excess"):
            result.unmeasured[name] = (
                "no reference strip: the texture/ringing distance is a PAIRED "
                "question. Render the same-seed control to measure it."
            )
    else:
        n = min(len(strip), len(reference_strip))
        try:
            dvals = [
                detail_metrics.dists_pair(reference_strip[i], strip[i], device)
                for i in range(n)
            ]
            result.measured["dists"] = float(sum(dvals) / len(dvals))
        except Exception as exc:  # torch/torchmetrics absent or shape mismatch
            result.unmeasured["dists"] = f"DISTS unavailable: {type(exc).__name__}: {exc}"
        rvals = [
            detail_metrics.ringing_excess(reference_strip[i], strip[i]) for i in range(n)
        ]
        result.measured["ringing_excess"] = float(sum(rvals) / len(rvals))

    # --- tier 3: the VLM rubric (primary catch) -----------------------------
    if judge is None:
        for name, _ in DETAIL_AXES:
            result.unmeasured[name] = (
                "no judge VLM: the fine-detail rubric is the primary catch for "
                "melted faces and pseudo-glyphs and needs a Judge"
            )
        result.unmeasured["detail_score"] = "no judge VLM"
    else:
        axes = score_detail_vlm(strip, judge)
        result.measured.update(axes)
        if axes:
            result.measured["detail_score"] = float(sum(axes.values()) / len(axes))
            for name, _ in DETAIL_AXES:
                if axes.get(name) == 0.0:
                    result.defects.append(f"VLM: {name} failed")
        if reference_strip is not None:
            pw = judge_detail_pairwise(strip, reference_strip, judge)
            if pw == pw:  # not NaN
                result.measured["detail_pref_delta"] = pw

    # --- tri-state ----------------------------------------------------------
    # A PASS needs a GATING-CAPABLE measurement, NOT merely something measured.
    # edge_overshoot is a scene-confounded diagnostic (report-only by
    # construction) — it can never PASS a clip alone, exactly the task's rule
    # that the cheap numeric tier may FLAG for the VLM but never pass on its own.
    gating = {
        "text_legibility", "text_legible_fraction", "dists", "ringing_excess",
        "detail_score", "detail_text_legible", "detail_face_coherent",
        "detail_edge_clean", "detail_texture_real",
    }
    has_gating = any(k in result.measured for k in gating)
    if result.defects:
        result.verdict = REJECT
    elif has_gating:
        result.verdict = PASS
    else:
        result.verdict = UNMEASURED
        result.notes.append(
            "detail UNMEASURED: only the scene-confounded diagnostic was "
            "computable (no judge, no reference, no detectable text) — the "
            "diagnostic cannot PASS a clip alone, so this is NOT a pass"
        )
    result.seconds = round(time.monotonic() - t0, 3)
    return result


__all__ = [
    "DETAIL_AXES",
    "PASS",
    "REJECT",
    "UNMEASURED",
    "DetailVerdict",
    "build_detail_judge_prompt",
    "detail_verdict",
    "judge_detail_pairwise",
    "score_detail_vlm",
]
