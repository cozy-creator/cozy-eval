"""The NOISE/BLANK integrity floor (ce#10, from ie#634).

Every assertion here is a RED arm: a clip built to be the thing the floor must
catch, or the thing it must NOT catch. The `melt` arm is the SCOPE-HONESTY
test — it pins the documented blind spot as executable fact rather than prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from cozy_eval import integrity, registry
from cozy_eval.errors import ConfigError
from cozy_eval.metrics import temporal

SIZE = 64
FRAMES = 16


def _texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """A smooth structured scene — noise low-passed so neighbouring pixels
    correlate, the way real image content does."""
    raw = rng.random((h, w, 3), dtype=np.float32)
    k = np.ones(9, np.float32) / 9.0
    return np.stack(
        [np.convolve(raw[..., c].ravel(), k, "same").reshape(h, w) for c in range(3)],
        axis=-1,
    ).astype(np.float32)


def _pan(seed: int = 0, frames: int = FRAMES) -> np.ndarray:
    """A coherent 1 px/frame pan across a textured scene — a real render."""
    rng = np.random.default_rng(seed)
    scene = _texture(rng, SIZE, SIZE + frames)
    return np.stack([scene[:, i:i + SIZE] for i in range(frames)])


def _noise(frames: int = FRAMES) -> np.ndarray:
    """Independent noise per frame — the ie#615 production signature."""
    return np.random.default_rng(7).random((frames, SIZE, SIZE, 3), dtype=np.float32)


def _blank(frames: int = FRAMES) -> np.ndarray:
    return np.full((frames, SIZE, SIZE, 3), 0.5, np.float32)


def _cut(frames: int = FRAMES) -> np.ndarray:
    """Two coherent pans spliced — one adjacent pair is a hard cut."""
    return np.concatenate([_pan(0, frames // 2), _pan(3, frames // 2)])


def _melted(frames: int = FRAMES) -> np.ndarray:
    """The pan, blurred hard: the fp8-melt class this gate is BLIND to."""
    from numpy.lib.stride_tricks import sliding_window_view

    clip = _pan(0, frames)
    k = 9
    padded = np.pad(clip, ((0, 0), (k // 2, k // 2), (k // 2, k // 2), (0, 0)), mode="edge")
    return sliding_window_view(padded, (k, k), axis=(1, 2)).mean((-1, -2)).astype(np.float32)


# --- the numbers -----------------------------------------------------------

def test_corr_separates_noise_from_a_real_render_with_margin() -> None:
    """The whole gate in one assertion: noise and real video are not close."""
    assert temporal.adjacent_frame_corr(_noise()) < 0.4
    assert temporal.adjacent_frame_corr(_pan()) > 0.85
    # and the floor sits in the empty middle between them
    assert 0.4 < temporal.NOISE_CORR_FLOOR < 0.85


def test_median_over_pairs_survives_a_hard_cut() -> None:
    """A cut drives ONE pair to ~0; the median must not care, or every clip
    with an edit would be rejected as noise."""
    series = temporal.adjacent_frame_corr_series(_cut())
    assert min(series) < 0.4, "the spliced pair should read as unrelated"
    assert temporal.adjacent_frame_corr(_cut()) > 0.85
    assert integrity.output_integrity(_cut()).ok


def test_blank_output_is_caught_by_the_std_floor_not_the_correlation() -> None:
    """A constant-fill clip has no variance to correlate: the correlation is a
    meaningless 0.0, so the BLANK defect has to come from the contrast floor."""
    result = integrity.output_integrity(_blank())
    assert result.verdict == integrity.REJECT
    assert any(d.startswith("BLANK") for d in result.defects)
    assert temporal.frame_std_min(_blank()) == pytest.approx(0.0, abs=1e-6)


# --- the verdict -----------------------------------------------------------

def test_noise_clip_is_rejected_and_names_the_defect() -> None:
    result = integrity.output_integrity(_noise())
    assert result.verdict == integrity.REJECT
    assert not result.ok
    assert any(d.startswith("NOISE") for d in result.defects)
    assert "adjacent-frame correlation" in result.summary() or result.defects


def test_real_render_passes_and_carries_the_scope_note() -> None:
    """A PASS must never read as a quality pass — the scope note ships WITH it."""
    result = integrity.output_integrity(_pan())
    assert result.verdict == integrity.PASS
    assert result.ok
    assert integrity.SCOPE_NOTE in result.notes


def test_a_clip_too_short_to_have_a_pair_is_unmeasured_not_a_pass() -> None:
    result = integrity.output_integrity(np.zeros((1, SIZE, SIZE, 3), np.float32))
    assert result.verdict == integrity.UNMEASURED
    assert not result.ok


def test_malformed_frames_are_unmeasured_not_a_pass() -> None:
    result = integrity.output_integrity(np.zeros((4, SIZE, SIZE), np.float32))
    assert result.verdict == integrity.UNMEASURED
    assert not result.ok


def test_decimation_does_not_change_the_answer() -> None:
    """PIN — the gate decimates each frame to ~96 rows before converting to
    float, which is what buys the <10 ms serve-path budget (246 ms -> 8 ms on a
    1344x768 clip). It is only allowed to do that because the statistic is a
    coarse whole-frame correlation: decimated and full-resolution must agree,
    for the real arm AND the noise arm."""
    for clip in (_pan(), _noise()):
        full = temporal.integrity_stats(clip, target_h=10_000)["adjacent_frame_corr"]
        small = temporal.integrity_stats(clip, target_h=16)["adjacent_frame_corr"]
        assert full == pytest.approx(small, abs=0.05)


def test_only_the_sampled_frames_are_touched() -> None:
    """The cost must not grow with clip length: a clip 10x longer costs the
    same, because ~2*pairs frames are read and the rest are never converted."""
    short, long = _pan(0, 16), _pan(0, 160)
    assert temporal.adjacent_frame_corr(short) == pytest.approx(
        temporal.adjacent_frame_corr(long), abs=0.1)
    assert len(temporal.adjacent_frame_corr_series(long)) == temporal.INTEGRITY_PAIRS


def test_uint8_and_float_clips_score_the_same() -> None:
    """The serve path holds uint8; the eval path holds float in [0,1]."""
    clip = _pan()
    as_u8 = np.clip(clip * 255.0, 0, 255).astype(np.uint8)
    assert temporal.adjacent_frame_corr(as_u8) == pytest.approx(
        temporal.adjacent_frame_corr(clip), abs=0.01)


# --- SCOPE HONESTY: the documented blind spot, pinned -----------------------

def test_the_melt_class_scores_HIGHER_than_a_clean_render() -> None:
    """PIN — the reason this gate is one of THREE axes and never sells itself
    as a quality gate. Smearing removes high-frequency temporal variation, so a
    melted render looks temporally CLEANER: it scores higher and sails through.
    Measured on real arms too (melted 0.956 vs clean 0.916, ie#634)."""
    clean = temporal.adjacent_frame_corr(_pan())
    melted = temporal.adjacent_frame_corr(_melted())
    assert melted > clean, "if this ever inverts, re-read the scope note"
    assert integrity.output_integrity(_melted()).ok, (
        "the melt PASSES the integrity floor by design — detail detectors and "
        "the VLM rubric are the instruments for that class"
    )


# --- the production path ---------------------------------------------------

def test_run_video_always_records_integrity_and_flags_the_noise_row() -> None:
    """The real codepath, not the function in isolation. The screen has NO
    opt-out flag: every row carries both numbers, and a noise candidate lands a
    named note on the report instead of passing quietly."""
    from test_video import REF_OFFSETS, SET, StripJudge, _clip, _samples

    from cozy_eval import promptset, video

    cands = [_clip(REF_OFFSETS), _noise(frames=6)]
    report = video.run_video(
        _samples(), cands, checklists=promptset.checklists_for(SET),
        judge=StripJudge(), use_ocr=False, use_temporal_fidelity=False, device="cpu",
    )
    for row in report.rows:
        assert "adjacent_frame_corr" in row.values
        assert "frame_std_min" in row.values
    assert report.rows[0].values["adjacent_frame_corr"] > 0.85
    assert report.rows[1].values["adjacent_frame_corr"] < 0.4
    assert any("OUTPUT INTEGRITY" in n and "NOISE" in n for n in report.notes)
    assert not any("sample 00 OUTPUT INTEGRITY" in n for n in report.notes)
    assert report.models["integrity"] == temporal.INTEGRITY_LIBRARY


# --- registry wiring -------------------------------------------------------

def test_integrity_metrics_are_registered() -> None:
    assert int(registry.METRIC_SET_VERSION.rsplit("@", 1)[1]) >= 7
    spec = registry.BY_NAME["adjacent_frame_corr"]
    assert spec.dimension == registry.QUALITY
    assert not spec.paired
    assert spec.higher_is_better
    assert spec.gated, "the only floor here that is banked against real arms"
    assert not registry.BY_NAME["frame_std_min"].gated
    for dim in registry.DIMENSIONS:
        assert len([m for m in registry.by_dimension(dim) if m.headline]) == 1


def test_pair_starts_rejects_a_single_frame() -> None:
    with pytest.raises(ConfigError):
        temporal.adjacent_frame_corr_series(np.zeros((1, 4, 4, 3), np.float32))
