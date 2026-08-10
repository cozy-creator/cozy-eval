"""Video mode: the same four dimensions, scored over clips.

STABILITY: locked core (v0.x) for :func:`run_video`'s call shape and the modes
it stamps (``video-paired`` / ``video-reference-free``); the report it returns
is the same :class:`cozy_eval.suite.SuiteReport` schema.

What changes for video, and what deliberately does not:

* **similarity** stays per-frame — ``lpips``/``psnr``/``ssim``/``ms_ssim`` are
  per-frame MEANS under their usual names, plus the worst-frame tail
  (``lpips_frame_worst``) — AND gains the Δ-frame channel (``dframe_psnr``,
  ``dframe_ssim``): the frame-to-frame dynamics comparison where flicker is a
  first-order signal. See :mod:`cozy_eval.metrics.temporal`.
* **adherence** uses the same authored-checklist discipline with the
  motion/hold duality (:class:`cozy_eval.metrics.adherence.VideoChecklist`):
  a clip can fail by not moving or by not holding still. The judge sees an
  ORDERED strip of uniformly sampled frames in ONE call per clip — the same
  Judge protocol as images; a judge with native video input is just a faster
  implementation of the same contract. OCR items are read on every sampled
  frame and must PERSIST (majority of frames), because text that is legible in
  one lucky frame has not survived motion.
* **quality** composes ``cozy_eval.metrics.signal`` for single-arm temporal
  signal statistics (``luma_flicker``, ``jerk_ratio``).
* **preference** is UNMEASURED in video mode for now, and the report says so.
  A commercially-clean video preference model exists (UnifiedReward-2.0,
  MIT weights on an Apache-2.0 Qwen3-VL base) and its integration is tracked;
  a frame-mean image preference score is NOT used as a stand-in — an image
  model cannot see motion, and a number that ignores the axis under test is
  worse than an honest gap.

* **audio** is scored when ``audio=`` is passed, through the single verdict
  function :func:`cozy_eval.audio.audio_verdict` — signal statistics and the
  dual-mono/silence/clipping guards, paired fidelity to the reference arm,
  AV-sync, and the transcribe-then-judge semantic tier. The report's MODE gains
  ``+audio``, and a run WITHOUT it carries an explicit UNMEASURED note, because
  every video model we serve now emits audio and a green video report says
  nothing about the soundtrack.

Clips are ``(T, H, W, 3)`` RGB arrays (uint8 or float [0, 1]) or lists of PIL
frames — the pre-encode arrays a producer already holds. No ffmpeg dependency;
decoding video FILES is the caller's job.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import msgspec

from . import registry, suite
from .device import AUTO, resolve_device
from .errors import ConfigError
from .judge import Judge
from .metrics import adherence, ocr, similarity, temporal
from .metrics.adherence import (
    KIND_OCR,
    KIND_VQA,
    OCR_FUZZY_PASS,
    AdherenceScore,
    ChecklistItem,
    ItemVerdict,
    VideoChecklist,
    parse_judge_answers,
    text_match,
)

#: Frames shown to the judge (and read by OCR / CLIP) per clip. Uniform over
#: the clip, always including first and last frame.
DEFAULT_JUDGE_FRAMES = 8


class VideoSample(msgspec.Struct, kw_only=True):
    """One video eval row. ``checklist_id`` selects the authored t2v checklist;
    empty means the row is not scored for adherence."""

    prompt: str
    seed: int = 0
    checklist_id: str = ""


# ---------------------------------------------------------------------------
# judge orchestration — ONE structured call per clip
# ---------------------------------------------------------------------------

_VIDEO_PREAMBLE = (
    "You are grading whether a video generator followed its instructions. "
    "You are shown {n} frames sampled in temporal order from a single "
    "generated video: the first image is the first frame, the last image is "
    "the last frame. Judge motion and change ACROSS the frames, not any one "
    "frame alone. Answer each numbered question with yes or no; if the asked "
    "motion never happens, or something that should stay fixed changes, "
    "answer no. Reply with ONLY a JSON array like "
    '[{{"n": 1, "answer": "yes"}}, {{"n": 2, "answer": "no"}}] — one object '
    "per question, no other text."
)


def build_video_judge_prompt(items: tuple[ChecklistItem, ...], *, frame_count: int) -> str:
    lines = [_VIDEO_PREAMBLE.format(n=frame_count), "Questions:"]
    lines += [f"{i}. {item.question}" for i, item in enumerate(items, start=1)]
    return "\n".join(lines)


def score_video_ocr_items(
    items: tuple[ChecklistItem, ...], page_texts: list[str],
) -> list[ItemVerdict]:
    """OCR items over a sampled frame strip: PERSISTENCE, not best-frame.

    An item is present when it passes on a MAJORITY of the sampled frames —
    text that flickers into legibility once has not survived motion. ``score``
    is the mean fuzzy similarity across frames (inverted for ``absent`` items),
    so a near-miss still reads as a near-miss.
    """
    out = []
    for item in items:
        passes, fuzzies = 0, []
        for page in page_texts:
            exact, fuzzy = text_match(item.text, page)
            fuzzies.append(fuzzy)
            if exact or fuzzy >= OCR_FUZZY_PASS:
                passes += 1
        present = passes * 2 > len(page_texts)
        verified = (not present) if item.absent else present
        mean_fuzzy = sum(fuzzies) / len(fuzzies) if fuzzies else 0.0
        out.append(ItemVerdict(
            id=item.id, kind=KIND_OCR, verified=verified,
            score=(1.0 - mean_fuzzy) if item.absent else mean_fuzzy,
            detail=(
                f"{'absent-required ' if item.absent else ''}"
                f"passed {passes}/{len(page_texts)} frames, mean fuzzy {mean_fuzzy:.2f}"
            ),
        ))
    return out


def score_video(
    checklist: VideoChecklist, strip: list[Any], *,
    judge: Judge | None = None, page_texts: list[str] | None = None,
) -> AdherenceScore:
    """Motion compliance + static preservation for one clip's sampled strip.

    ONE judge call covers both groups' questions; OCR items are read from
    every frame of the strip (persistence rule). ``compliance`` carries the
    motion group, ``preservation`` the hold group — the same duality fields
    edit mode uses, because it is the same failure geometry over time.
    """
    groups = (("motion", checklist.motion), ("hold", checklist.hold))
    all_items = checklist.motion + checklist.hold
    vqa_items = tuple(i for i in all_items if i.kind == KIND_VQA)
    verdicts: list[ItemVerdict] = []
    missing = []

    for name, items in groups:
        ocr_items = tuple(i for i in items if i.kind == KIND_OCR)
        if ocr_items:
            if page_texts is None:
                missing.append(f"{len(ocr_items)} {name} ocr items (no OCR reader)")
            else:
                verdicts += score_video_ocr_items(ocr_items, page_texts)
    if vqa_items:
        if judge is None:
            missing.append(f"{len(vqa_items)} vqa items (no judge)")
        else:
            prompt = build_video_judge_prompt(vqa_items, frame_count=len(strip))
            answers = parse_judge_answers(judge.ask(strip, prompt), len(vqa_items))
            for i, item in enumerate(vqa_items, start=1):
                answered = answers.get(i)
                yes = bool(answered)
                verified = (not yes) if item.absent else yes
                verdicts.append(ItemVerdict(
                    id=item.id, kind=KIND_VQA, verified=verified,
                    score=1.0 if verified else 0.0,
                    detail="unanswered" if answered is None else ("yes" if yes else "no"),
                ))
    if not verdicts:
        return AdherenceScore(measured=False, note="; ".join(missing) or "empty checklist")

    def _group_recall(items: tuple[ChecklistItem, ...]) -> float:
        scored = tuple(i for i in items if any(v.id == i.id for v in verdicts))
        return adherence._recall(scored, verdicts) if scored else 0.0

    scored_all = tuple(i for i in all_items if any(v.id == i.id for v in verdicts))
    exact, fuzzy = adherence._text_rates(scored_all, verdicts)
    return AdherenceScore(
        measured=True,
        element_recall=adherence._recall(scored_all, verdicts),
        compliance=_group_recall(checklist.motion),
        preservation=_group_recall(checklist.hold),
        text_exact=exact, text_fuzzy=fuzzy, items=verdicts,
        note=("partial: " + "; ".join(missing)) if missing else "",
    )


def _group_measured(
    score: AdherenceScore, items: tuple[ChecklistItem, ...],
) -> bool:
    """Was ANY item of this group actually scored? A group nobody looked at
    must not report 0.0 — 'unmeasured' and 'failed everything' are different
    facts."""
    ids = {i.id for i in items}
    return any(v.id in ids for v in score.items)


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------

def _aligned(cand: Any, ref: Any, index: int) -> Any:
    """Candidate frames resized to the reference's HxW. Frame-count mismatch is
    an error, not a silent trim — a paired video verdict needs frame alignment."""
    if cand.shape[0] != ref.shape[0]:
        raise ConfigError(
            f"sample {index}: candidate has {cand.shape[0]} frames but the "
            f"reference has {ref.shape[0]} — paired video metrics are "
            "frame-aligned; render both arms at the same frame count"
        )
    if cand.shape[1:3] == ref.shape[1:3]:
        return cand
    from PIL import Image

    size = (ref.shape[2], ref.shape[1])  # PIL takes (W, H)
    return temporal.as_frames(
        [img.resize(size, Image.LANCZOS) for img in temporal.frame_images(cand)]
    )


def _adherence_pass(
    sample: VideoSample, checklists: Any, strip: list[Any],
    *, judge: Judge | None, use_ocr: bool, ocr_seconds: list[float],
) -> AdherenceScore | None:
    if checklists is None or not sample.checklist_id:
        return None
    entry = checklists.t2v.get(sample.checklist_id)
    if entry is None:
        return None
    page_texts: list[str] | None = None
    if use_ocr:
        _t = time.monotonic()
        pages = [ocr.read(img) for img in strip]
        ocr_seconds.append(time.monotonic() - _t)
        if any(p.measured for p in pages):
            page_texts = [p.text if p.measured else "" for p in pages]
    return score_video(entry, strip, judge=judge, page_texts=page_texts)


def run_video(
    samples: list[VideoSample],
    candidates: list[Any],
    *,
    references: list[Any] | None = None,
    checklists: Any = None,
    judge: Judge | None = None,
    use_ocr: bool = True,
    use_signal: bool = True,
    use_detail: bool = True,
    detail_judge: Judge | None = None,
    use_temporal_fidelity: bool = True,
    judge_frames: int = DEFAULT_JUDGE_FRAMES,
    device: str = AUTO,
    render_seconds: float = 0.0,
    audio: list[Any] | None = None,
    reference_audio: list[Any] | None = None,
    sample_rate: int = 0,
    fps: float = 0.0,
    audio_checklists: dict[str, Any] | None = None,
    transcriber: Any = None,
    audio_judge: Any = None,
    audio_fidelity_budget: tuple[Any, ...] = (),
) -> suite.SuiteReport:
    """Score a batch of clips. ``references`` present => video-paired mode.

    ``candidates`` / ``references`` are clips in any form
    :func:`cozy_eval.metrics.temporal.as_frames` accepts.

    ``audio`` attaches the soundtrack, one entry per sample (an
    :class:`~cozy_eval.metrics.audio.Audio`, an array with ``sample_rate``, or a
    media path). The report's MODE then says ``+audio``, and when it is omitted
    the report carries an explicit UNMEASURED note — because every video model we
    serve now emits audio, and a run that silently scored none of it must never
    read as a clean pass.
    """
    if len(samples) != len(candidates):
        raise ConfigError(
            f"got {len(samples)} samples but {len(candidates)} candidate clips; "
            "each sample is scored against its own render, so the lists must match"
        )
    paired = references is not None
    if paired and len(references) != len(samples):
        raise ConfigError(
            f"got {len(samples)} samples but {len(references)} reference clips; "
            "a paired run needs one reference per sample"
        )
    if audio is not None and len(audio) != len(samples):
        raise ConfigError(
            f"got {len(samples)} samples but {len(audio)} audio tracks; a clip's "
            "soundtrack is scored against its own sample, so the lists must match"
        )
    device = resolve_device(device)
    ocr_seconds: list[float] = []

    base_mode = "video-paired" if paired else "video-reference-free"
    report = suite.SuiteReport(
        mode=f"{base_mode}+audio" if audio is not None else base_mode,
        samples=len(samples),
        library_version=suite._library_version(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    rows = [
        suite.SampleRow(
            index=i, checklist_id=s.checklist_id, prompt=s.prompt[:200], seed=s.seed,
        )
        for i, s in enumerate(samples)
    ]

    cand_frames = [temporal.as_frames(c) for c in candidates]
    ref_frames = [temporal.as_frames(r) for r in references] if paired else None
    if paired:
        cand_frames = [
            _aligned(c, r, i)
            for i, (c, r) in enumerate(zip(cand_frames, ref_frames, strict=True))
        ]

    # --- similarity: per-frame aggregate + the Δ-frame channel (paired) -----
    if paired:
        t0 = time.monotonic()
        lp = similarity.lpips_model(device)
        report.seconds["lpips_load"] = round(time.monotonic() - t0, 2)
        lpips_s = pixel_s = dframe_s = 0.0
        for row, ref, cand in zip(rows, ref_frames, cand_frames, strict=True):
            ref_imgs = temporal.frame_images(ref)
            cand_imgs = temporal.frame_images(cand)
            t1 = time.monotonic()
            per_frame = [
                similarity.lpips_pair(lp, a, b, device)
                for a, b in zip(ref_imgs, cand_imgs, strict=True)
            ]
            lpips_s += time.monotonic() - t1
            row.lpips = sum(per_frame) / len(per_frame)
            row.values["lpips_frame_worst"] = max(per_frame)
            t1 = time.monotonic()
            pairs = list(zip(ref_imgs, cand_imgs, strict=True))
            row.psnr = _mean(similarity.psnr(a, b) for a, b in pairs)
            row.ssim = _mean(similarity.ssim(a, b) for a, b in pairs)
            row.ms_ssim = _mean(similarity.ms_ssim(a, b) for a, b in pairs)
            pixel_s += time.monotonic() - t1
            t1 = time.monotonic()
            dpsnr = temporal.dframe_psnr_series(ref, cand)
            row.values["dframe_psnr"] = sum(dpsnr) / len(dpsnr)
            row.values["dframe_psnr_worst"] = min(dpsnr)
            dssim = temporal.dframe_ssim_series(ref, cand)
            row.values["dframe_ssim"] = sum(dssim) / len(dssim)
            dframe_s += time.monotonic() - t1
        report.seconds["lpips"] = round(lpips_s, 2)
        report.seconds["pixel_metrics"] = round(pixel_s, 2)
        report.seconds["dframe"] = round(dframe_s, 2)
        report.models["lpips"] = "torchmetrics:lpips-alex"

    # --- adherence: clip continuity + the checklist pass --------------------
    def _strip(frames: Any) -> list[Any]:
        images = temporal.frame_images(frames)
        return [images[i] for i in temporal.sample_indices(frames.shape[0], judge_frames)]

    strips = [_strip(c) for c in cand_frames]
    ref_strips = [_strip(r) for r in ref_frames] if paired else None
    t0 = time.monotonic()
    clip, proc = similarity.clip_model(device)
    for row, sample, strip in zip(rows, samples, strips, strict=True):
        row.clip_cand = _mean(
            similarity.clip_score(clip, proc, img, sample.prompt, device) for img in strip
        )
        if paired:
            row.clip_ref = _mean(
                similarity.clip_score(clip, proc, img, sample.prompt, device)
                for img in ref_strips[row.index]
            )
            row.clip_delta = row.clip_cand - row.clip_ref
    report.seconds["clip"] = round(time.monotonic() - t0, 2)
    report.models["clip"] = similarity.CLIP_MODEL

    t0 = time.monotonic()
    scored_any = False
    for row, sample, strip in zip(rows, samples, strips, strict=True):
        cand_score = _adherence_pass(
            sample, checklists, strip,
            judge=judge, use_ocr=use_ocr, ocr_seconds=ocr_seconds,
        )
        if cand_score is None or not cand_score.measured:
            continue
        scored_any = True
        entry = checklists.t2v[sample.checklist_id]
        row.element_recall_cand = cand_score.element_recall
        row.text_exact = cand_score.text_exact
        row.text_fuzzy = cand_score.text_fuzzy
        if _group_measured(cand_score, entry.motion):
            row.values["motion_compliance"] = cand_score.compliance
        if _group_measured(cand_score, entry.hold):
            row.values["static_preservation"] = cand_score.preservation
        row.failed_items = [v.id for v in cand_score.items if not v.verified]
        if paired:
            ref_score = _adherence_pass(
                sample, checklists, ref_strips[row.index],
                judge=judge, use_ocr=use_ocr, ocr_seconds=ocr_seconds,
            )
            if ref_score is not None and ref_score.measured:
                row.element_recall_ref = ref_score.element_recall
                row.element_recall_delta = (
                    cand_score.element_recall - ref_score.element_recall
                )
    adherence_s = time.monotonic() - t0
    ocr_s = sum(ocr_seconds)
    report.seconds["ocr"] = round(ocr_s, 2)
    if judge is not None:
        report.seconds["judge_load"] = round(getattr(judge, "load_seconds", 0.0), 2)
        report.seconds["judge_infer"] = round(getattr(judge, "infer_seconds", 0.0), 2)
        report.judge_calls = int(getattr(judge, "call_count", 0) or 0)
        report.judge_vram_bytes = int(getattr(judge, "vram_bytes", 0))
        if report.judge_calls:
            report.seconds["judge_per_call"] = round(
                getattr(judge, "infer_seconds", 0.0) / report.judge_calls, 2
            )
    report.seconds["adherence_other"] = round(
        max(0.0, adherence_s - ocr_s - getattr(judge, "infer_seconds", 0.0)), 2
    )
    if scored_any:
        if judge is not None:
            report.models["judge"] = getattr(judge, "model_ref", adherence.DEFAULT_JUDGE)
        if use_ocr:
            report.models["ocr"] = adherence.DEFAULT_OCR
    else:
        report.notes.append(
            "element fidelity UNMEASURED: no t2v checklist matched these samples, "
            "or neither a judge VLM nor an OCR reader was available"
        )

    # --- quality: composed single-arm temporal signal -----------------------
    if use_signal:
        t0 = time.monotonic()
        for row, cand in zip(rows, cand_frames, strict=True):
            row.values.update(temporal.signal_stats(cand))
        report.seconds["signal"] = round(time.monotonic() - t0, 2)
        report.models["signal"] = temporal.SIGNAL_LIBRARY

    # --- temporal fidelity: do the movements match (optical flow) ------------
    # The motion axis screenshots miss. Reference-free warp_error on every clip;
    # flow_divergence + warp_error_delta additionally when a same-seed control is
    # present. opencv is the `video` extra — a run without it is UNMEASURED, not
    # a silent pass.
    if use_temporal_fidelity:
        try:
            import cv2  # noqa: F401
        except ImportError:
            report.notes.append(
                "TEMPORAL FIDELITY UNMEASURED: opencv not installed "
                "(pip install 'cozy-eval[video]') — flow_divergence / warp_error skipped"
            )
        else:
            t0 = time.monotonic()
            for row in rows:
                cand = cand_frames[row.index]
                if paired:
                    row.values.update(
                        temporal.temporal_fidelity(ref_frames[row.index], cand)
                    )
                else:
                    row.values["warp_error"] = temporal.warp_error(cand)
            report.seconds["temporal_fidelity"] = round(time.monotonic() - t0, 2)
            report.models["temporal_fidelity"] = temporal.FLOW_LIBRARY

    # --- audio: the axis a silent run must never be allowed to skip quietly ---
    if audio is None:
        report.notes.append(
            "AUDIO UNMEASURED: no soundtrack was passed to run_video. Every video "
            "model we serve emits audio (LTX-2.3 muxes AAC, MiniMax-H3 generates "
            "32 kHz stereo jointly), so a green report from this run says nothing "
            "about the soundtrack — pass audio=[...] to score it."
        )
    else:
        from . import audio as audio_mode
        from .metrics import audio as audio_metrics
        from .metrics import avsync

        t0 = time.monotonic()
        unmeasured_reasons: dict[str, str] = {}
        for row, sample, track in zip(rows, samples, audio, strict=True):
            av = audio_mode.audio_verdict(
                track,
                reference_audio[row.index] if reference_audio else None,
                sample_rate=sample_rate,
                frames=cand_frames[row.index] if fps > 0 else None,
                reference_frames=(
                    ref_frames[row.index] if (paired and fps > 0 and reference_audio)
                    else None
                ),
                fps=fps,
                checklist=(audio_checklists or {}).get(sample.checklist_id),
                transcriber=transcriber,
                audio_judge=audio_judge,
                fidelity_budget=audio_fidelity_budget,
            )
            row.values.update(av.measured)
            for defect in av.defects:
                report.notes.append(f"sample {row.index:02d} AUDIO DEFECT: {defect}")
            unmeasured_reasons.update(av.unmeasured)
        report.seconds["audio"] = round(time.monotonic() - t0, 2)
        report.models["audio"] = audio_metrics.AUDIO_LIBRARY
        if fps > 0:
            report.models["av_sync"] = avsync.AVSYNC_METHOD
            report.notes.append(avsync.LIPSYNC_GAP_NOTE)
        else:
            unmeasured_reasons.setdefault(
                "av_sync_offset_ms",
                "AV-sync needs the frame rate: pass fps= to run_video",
            )
        if transcriber is not None:
            report.models["transcriber"] = getattr(transcriber, "model_ref", "")
        if audio_judge is not None:
            report.models["audio_judge"] = getattr(audio_judge, "model_ref", "")
        measured_any = {k for row in rows for k in row.values}
        for name, reason in sorted(unmeasured_reasons.items()):
            if name not in measured_any:
                report.notes.append(f"{name} UNMEASURED: {reason}")

    # --- fine detail: melted faces, pseudo-glyphs, edge halos, texture mush ---
    if use_detail:
        from . import detail as detail_mode
        from .metrics import detail as detail_metrics

        t0 = time.monotonic()
        d_judge = detail_judge or judge
        detail_unmeasured: dict[str, str] = {}
        for row, strip in zip(rows, strips, strict=True):
            dv = detail_mode.detail_verdict(
                strip,
                reference_strip=ref_strips[row.index] if paired else None,
                judge=d_judge, device=device,
            )
            row.values.update(dv.measured)
            for defect in dv.defects:
                report.notes.append(f"sample {row.index:02d} DETAIL DEFECT: {defect}")
            for flag in dv.flags:
                report.notes.append(f"sample {row.index:02d} DETAIL FLAG: {flag}")
            detail_unmeasured.update(dv.unmeasured)
        report.seconds["detail"] = round(time.monotonic() - t0, 2)
        report.models["detail"] = detail_metrics.DETAIL_LIBRARY
        if d_judge is not None:
            report.models["detail_judge"] = getattr(d_judge, "model_ref", "")
        measured_detail = {k for row in rows for k in row.values}
        for name, reason in sorted(detail_unmeasured.items()):
            if name not in measured_detail:
                report.notes.append(f"{name} UNMEASURED: {reason}")

    # --- preference: an honest gap, stated -----------------------------------
    report.notes.append(
        "video preference UNMEASURED: image preference models cannot see motion "
        "and are not used as a proxy; the tracked integration is "
        "UnifiedReward-2.0 (MIT weights, video pair+point scoring)"
    )

    report.rows = rows
    reported: tuple[tuple[str, str], ...] = (
        ("lpips", "lpips"), ("psnr", "psnr"), ("ssim", "ssim"), ("ms_ssim", "ms_ssim"),
        ("clip_delta", "clip_delta"),
        ("element_recall", "element_recall_delta" if paired else "element_recall_cand"),
        ("text_exact", "text_exact"), ("text_fuzzy", "text_fuzzy"),
    )
    core = {name for name, _ in reported}
    for metric, field in reported:
        summary = suite.summarize(rows, metric, field=field)
        if summary.measured:
            report.dimensions.append(summary)
    for name in sorted({k for row in rows for k in row.values} - core):
        if name in registry.BY_NAME:
            summary = suite.summarize(rows, name)
            if summary.measured:
                report.dimensions.append(summary)

    total = round(sum(
        v for k, v in report.seconds.items() if k not in suite._DERIVED_SECONDS
    ), 2)
    report.seconds["total"] = total
    report.render_seconds = round(float(render_seconds), 2)
    if render_seconds > 0:
        report.metric_overhead = round(total / float(render_seconds), 4)
    return report


def _mean(values: Any) -> float:
    seq = list(values)
    return sum(seq) / len(seq)


__all__ = [
    "DEFAULT_JUDGE_FRAMES",
    "VideoSample",
    "build_video_judge_prompt",
    "run_video",
    "score_video",
    "score_video_ocr_items",
]
