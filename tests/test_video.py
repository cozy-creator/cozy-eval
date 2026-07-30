"""Video mode: frame handling, the Δ-frame channel, motion/hold checklists,
the video runner.

Integration style, no mocks except the pod-side judge VLM (stubbed exactly as
the image tests stub it). The load-bearing fixture is SYNTHETIC MOTION built
so the verdicts are forced: a "stable" candidate whose error is constant across
frames and a "flicker" candidate whose error alternates. Per-frame metrics
cannot tell them apart — the flicker arm even looks BETTER per-frame — and the
Δ-frame channel must separate them cleanly. That asymmetry is the whole reason
the channel exists.
"""

from __future__ import annotations

import json

import msgspec
import numpy as np
import pytest

from cozy_eval.bench import errors, promptset, suite, video
from cozy_eval.bench.metrics import adherence, similarity, temporal

SET = "hard-video-v1"


# ---------------------------------------------------------------------------
# synthetic clips
# ---------------------------------------------------------------------------

def _scene(size: int = 64, seed: int = 5) -> np.ndarray:
    """A textured static background — noise smoothed twice so it has structure
    at more than one scale, like the cmm test fixtures."""
    rng = np.random.default_rng(seed)
    base = rng.random((size, size, 3)).astype(np.float32)
    for _ in range(2):
        base = (
            base
            + np.roll(base, 1, 0) + np.roll(base, -1, 0)
            + np.roll(base, 1, 1) + np.roll(base, -1, 1)
        ) / 5.0
    return 0.2 + 0.6 * base


def _clip(offsets: list[int], size: int = 64, seed: int = 5) -> np.ndarray:
    """T frames of the scene with a bright square at x = offset[t]. Constant
    background, only the square moves — dynamics live entirely in the offsets."""
    frames = []
    for off in offsets:
        frame = _scene(size, seed).copy()
        frame[24:40, off:off + 16] = 0.95
        frames.append(frame)
    return np.stack(frames)


REF_OFFSETS = [4, 6, 8, 10, 12, 14]                 # constant velocity +2/frame

_ERR = 0.05


def _stable(ref: np.ndarray) -> np.ndarray:
    """A CONSTANT additive error: every frame off by the same amount, so the
    frame-to-frame dynamics are exactly the reference's."""
    return ref + _ERR


def _flicker(ref: np.ndarray) -> np.ndarray:
    """The SAME error magnitude per frame, alternating in sign: per-frame MSE
    is identical to the stable arm's, the dynamics are wrong."""
    signs = np.array([(-1.0) ** t for t in range(ref.shape[0])], dtype=np.float32)
    return ref + _ERR * signs[:, None, None, None]


# ---------------------------------------------------------------------------
# frame handling
# ---------------------------------------------------------------------------

def test_as_frames_normalizes_every_accepted_form_and_rejects_non_clips() -> None:
    """One convention downstream: float32 (T,H,W,3) in [0,1], whatever the
    producer handed over."""
    from PIL import Image

    arr = (np.random.default_rng(0).random((3, 32, 32, 3)) * 255).astype(np.uint8)
    frames = temporal.as_frames(arr)
    assert frames.shape == (3, 32, 32, 3)
    assert frames.dtype == np.float32
    assert 0.0 <= float(frames.min()) and float(frames.max()) <= 1.0

    pil = [Image.fromarray(f) for f in arr]
    assert np.allclose(frames, temporal.as_frames(pil))

    with pytest.raises(errors.ConfigError, match="expected"):
        temporal.as_frames(np.zeros((32, 32, 3), dtype=np.uint8))  # one frame, no T axis
    with pytest.raises(errors.ConfigError, match="at least 2 frames"):
        temporal.as_frames(np.zeros((1, 32, 32, 3), dtype=np.uint8))
    with pytest.raises(errors.ConfigError, match="empty"):
        temporal.as_frames([])


def test_sample_indices_are_uniform_and_keep_the_endpoints() -> None:
    """Motion is judged by its endpoints: the first and last frame must always
    be in the strip, whatever the clip length."""
    got = temporal.sample_indices(17, 8)
    assert got[0] == 0 and got[-1] == 16
    assert got == sorted(set(got))
    assert temporal.sample_indices(5, 8) == [0, 1, 2, 3, 4]
    assert temporal.sample_indices(3, 1) == [0]


# ---------------------------------------------------------------------------
# the Δ-frame channel — the test this lane exists for
# ---------------------------------------------------------------------------

def test_dframe_separates_flicker_from_stable_while_per_frame_cannot() -> None:
    """Both arms carry the SAME per-frame error magnitude — one constant, one
    alternating in sign. Per-frame PSNR cannot tell them apart (identical MSE
    by construction). The Δ-frame channel must separate them decisively: the
    stable arm's dynamics ARE the reference's, the flicker arm's alternate."""
    ref = _clip(REF_OFFSETS)
    stable = _stable(ref)
    flicker = _flicker(ref)

    def frame_psnr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean([
            similarity.psnr(x, y)
            for x, y in zip(temporal.frame_images(a), temporal.frame_images(b), strict=True)
        ]))

    # Per-frame: indistinguishable (same error magnitude every frame).
    assert abs(frame_psnr(ref, flicker) - frame_psnr(ref, stable)) < 0.5

    stable_d = temporal.dframe_psnr_series(ref, stable)
    flicker_d = temporal.dframe_psnr_series(ref, flicker)
    # Stable dynamics are IDENTICAL to the reference's: every Δ-frame matches.
    assert all(v == 99.0 for v in stable_d), stable_d
    # Flicker dynamics diverge on every step, by a wide margin.
    assert max(flicker_d) < 40.0, flicker_d
    assert min(stable_d) - max(flicker_d) > 20.0

    stable_s = temporal.dframe_ssim_series(ref, stable)
    flicker_s = temporal.dframe_ssim_series(ref, flicker)
    assert min(stable_s) > 0.99
    assert float(np.mean(flicker_s)) < float(np.mean(stable_s))

    # An identical clip is the degenerate case of the stable arm: the whole
    # series hits the finite sentinel rather than an infinity.
    assert temporal.dframe_psnr_series(ref, ref.copy()) == [99.0] * (len(REF_OFFSETS) - 1)
    assert min(temporal.dframe_ssim_series(ref, ref.copy())) > 0.99


def test_dframe_rejects_misaligned_clips() -> None:
    with pytest.raises(errors.ConfigError, match="differ in shape"):
        temporal.dframe_psnr_series(_clip(REF_OFFSETS), _clip(REF_OFFSETS[:4]))


# ---------------------------------------------------------------------------
# composed single-arm signal stats (real cozy_eval signal backend, no mock)
# ---------------------------------------------------------------------------

def test_signal_stats_compose_cmm_and_rank_flicker_above_smooth() -> None:
    # Constant frame-mean luma: the moving square covers the same area each
    # frame, so luma wobble is ~zero.
    steady = _clip(REF_OFFSETS)
    # PER-PIXEL smooth dynamics: a global luma ramp — every pixel changes by a
    # constant step per frame, so the second temporal difference is ~zero.
    # (Edge motion is no use as a low-jerk baseline: cmm's jerk is per-pixel,
    # and a pixel a moving edge crosses sees a step either way.)
    ramp = np.stack([_scene() * (1.0 + 0.03 * t) for t in range(8)])
    # Flicker: alternating global gain on a static scene.
    signs = np.array([(-1.0) ** t for t in range(8)], dtype=np.float32)
    flicker = np.clip(_scene()[None] * (1.0 + 0.04 * signs)[:, None, None, None], 0, 1)

    steady_s = temporal.signal_stats(steady)
    ramp_s = temporal.signal_stats(ramp)
    flicker_s = temporal.signal_stats(flicker)
    assert set(steady_s) == {"luma_flicker", "jerk_ratio"}
    assert flicker_s["luma_flicker"] > steady_s["luma_flicker"]
    assert flicker_s["jerk_ratio"] > ramp_s["jerk_ratio"]


# ---------------------------------------------------------------------------
# t2v checklists: the shipped set
# ---------------------------------------------------------------------------

def test_hard_video_v1_is_internally_consistent() -> None:
    """One comprehensive validator for the shipped video set: prompts and
    checklists cross-cover, the frozen rows and seeds have not moved, every row
    carries BOTH halves of the motion/hold duality, and the text-under-motion
    axis is checked by OCR persistence rather than a VLM's impression of the
    sign. Loading the set also exercises the additive t2v loader path.
    (Malformed t2v documents are covered by the loader table in
    test_bench_suite.py.)"""
    loaded = promptset.checklists_for(SET)
    ps = promptset.load(SET)

    assert set(loaded.t2v) == {p.id for p in ps.t2v}
    assert len(ps.t2v) == 16
    assert [p.seed for p in ps.t2v] == list(range(301, 317))
    for entry in loaded.t2v.values():
        assert len(entry.motion) >= 2, entry.prompt_id
        assert len(entry.hold) >= 3, entry.prompt_id

    for prompt in ps.t2v:
        if "text" not in prompt.tags:
            continue
        items = loaded.t2v[prompt.id].motion + loaded.t2v[prompt.id].hold
        assert any(i.kind == adherence.KIND_OCR for i in items), (
            f"{prompt.id} is tagged 'text' but has no ocr item"
        )


# ---------------------------------------------------------------------------
# judge orchestration over the frame strip
# ---------------------------------------------------------------------------

class StripJudge:
    """Stub for the pod-side judge; records calls so the ONE-call-per-clip
    cost contract and the strip size are testable."""

    model_ref = "stub-video-judge"

    def __init__(self, noes: set[str] = frozenset()):
        self.noes = set(noes)
        self.calls: list[tuple[int, str]] = []

    def ask(self, images: list, prompt: str) -> str:
        self.calls.append((len(images), prompt))
        rows = []
        for line in prompt.splitlines():
            if line and line[0].isdigit() and ". " in line:
                n, text = line.split(". ", 1)
                rows.append({"n": int(n), "answer": "no" if text in self.noes else "yes"})
        return json.dumps(rows)


def _strip() -> list:
    return temporal.frame_images(_clip(REF_OFFSETS))


def test_video_judge_gets_one_ordered_strip_call_per_clip() -> None:
    entry = promptset.checklists_for(SET).t2v["v01"]
    judge = StripJudge()
    score = video.score_video(entry, _strip(), judge=judge)
    assert len(judge.calls) == 1
    images, prompt = judge.calls[0]
    assert images == len(REF_OFFSETS)
    assert "temporal order" in prompt
    assert prompt.count("\n1. ") == 1
    assert score.measured and score.element_recall == 1.0
    assert score.compliance == 1.0 and score.preservation == 1.0


def test_under_motion_fails_compliance_and_instability_fails_preservation() -> None:
    entry = promptset.checklists_for(SET).t2v["v01"]
    motion_qs = {i.question for i in entry.motion}
    hold_qs = {i.question for i in entry.hold}

    frozen = video.score_video(entry, _strip(), judge=StripJudge(noes=motion_qs))
    assert frozen.compliance == 0.0 and frozen.preservation == 1.0

    unstable = video.score_video(entry, _strip(), judge=StripJudge(noes=hold_qs))
    assert unstable.compliance == 1.0 and unstable.preservation == 0.0


def test_video_without_judge_or_ocr_is_unmeasured_not_zero() -> None:
    entry = promptset.checklists_for(SET).t2v["v01"]
    score = video.score_video(entry, _strip(), judge=None, page_texts=None)
    assert not score.measured and "no judge" in score.note


def test_ocr_persistence_requires_a_majority_of_frames() -> None:
    """Text legible in one lucky frame has not survived motion."""
    items = (adherence.ChecklistItem(id="t", kind="ocr", text="AD ASTRA"),)
    mostly = ["AD ASTRA", "AD ASTRA", "AD ASTRA", "AB ASTRA"]
    rarely = ["AD ASTRA", "AS ASTRO", "A ASTRA", "ADASTRA WAIT NO"]
    assert video.score_video_ocr_items(items, mostly)[0].verified
    assert not video.score_video_ocr_items(items, rarely)[0].verified

    gone = (adherence.ChecklistItem(id="g", kind="ocr", text="BASEMENT", absent=True),)
    assert video.score_video_ocr_items(gone, ["OPEN LATE"] * 4)[0].verified
    assert not video.score_video_ocr_items(gone, ["BASEMENT"] * 4)[0].verified


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------

def _samples() -> list[video.VideoSample]:
    ps = promptset.load(SET)
    return [
        video.VideoSample(prompt=ps.t2v[0].prompt, seed=ps.t2v[0].seed, checklist_id="v01"),
        video.VideoSample(prompt=ps.t2v[1].prompt, seed=ps.t2v[1].seed, checklist_id="v02"),
    ]


def test_run_video_paired_reports_all_video_channels() -> None:
    refs = [_clip(REF_OFFSETS), _clip(REF_OFFSETS, seed=9)]
    cands = [_stable(refs[0]), _flicker(refs[1])]
    report = video.run_video(
        _samples(), cands, references=refs,
        checklists=promptset.checklists_for(SET),
        judge=StripJudge(), use_ocr=False, device="cpu",
    )
    assert report.mode == "video-paired"
    assert report.samples == 2 and len(report.rows) == 2
    for row in report.rows:
        assert row.lpips is not None and row.psnr is not None
        for key in ("lpips_frame_worst", "dframe_psnr", "dframe_psnr_worst", "dframe_ssim"):
            assert key in row.values, key
        assert row.clip_delta is not None
        assert row.element_recall_cand == 1.0
        assert row.values["motion_compliance"] == 1.0
    # The flicker arm's Δ-frame number must sit far below the stable arm's.
    assert report.rows[1].values["dframe_psnr"] < report.rows[0].values["dframe_psnr"] - 20
    summarized = {d.metric for d in report.dimensions}
    assert {"lpips", "dframe_psnr", "element_recall", "motion_compliance"} <= summarized
    if "luma_flicker" in {m for row in report.rows for m in row.values}:
        assert "luma_flicker" in summarized
    assert any("video preference UNMEASURED" in note for note in report.notes)
    assert set(report.seconds) <= set(suite.SECONDS_KEYS)
    assert set(report.models) <= set(suite.MODELS_KEYS)
    # The report stays a plain-JSON BenchReport.
    back = msgspec.json.decode(msgspec.json.encode(report), type=suite.BenchReport)
    assert back == report


def test_run_video_reference_free_skips_similarity_but_scores_the_rest() -> None:
    report = video.run_video(
        _samples()[:1], [_clip(REF_OFFSETS)],
        checklists=promptset.checklists_for(SET),
        judge=StripJudge(), use_ocr=False, device="cpu",
    )
    assert report.mode == "video-reference-free"
    row = report.rows[0]
    assert row.lpips is None and "dframe_psnr" not in row.values
    assert row.clip_cand is not None and row.clip_delta is None
    assert row.element_recall_cand == 1.0 and row.element_recall_delta is None


def test_run_video_rejects_mismatches() -> None:
    with pytest.raises(errors.ConfigError, match="lists must match"):
        video.run_video(_samples(), [_clip(REF_OFFSETS)])
    with pytest.raises(errors.ConfigError, match="same frame count"):
        video.run_video(
            _samples()[:1], [_clip(REF_OFFSETS[:4])],
            references=[_clip(REF_OFFSETS)], use_ocr=False, device="cpu",
        )


def test_worst_frame_tail_names_the_ruined_frame_not_the_average() -> None:
    """A clip faithful on average with ONE destroyed frame: the mean absorbs
    it, the tail must not."""
    ref = _clip(REF_OFFSETS)
    cand = ref.copy()
    rng = np.random.default_rng(11)
    cand[3] = rng.random(cand[3].shape).astype(np.float32)  # one frame of noise
    report = video.run_video(
        _samples()[:1], [cand], references=[ref], use_ocr=False, device="cpu",
    )
    row = report.rows[0]
    assert row.values["lpips_frame_worst"] > row.lpips * 2
    # The ruined transition also shows in the Δ-frame tail.
    assert row.values["dframe_psnr_worst"] < row.values["dframe_psnr"]
