"""Track stability: the OBJECT axis.

Integration style, no mocks. The load-bearing fixture is a SYNTHETIC 3D-ish
scene: a textured field under a smooth camera pan, and the same pan with one
object whose feature points WARBLE — displaced by a small per-frame random
offset while every single frame stays a perfectly sharp, plausible image.

That asymmetry is the whole reason this family exists. The warble arm's frames
are individually clean, its whole-frame optical flow is dominated by the same
correct pan, and its frame-mean luma is unchanged; only something that follows a
POINT through time can see it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cozy_eval import errors, registry, tracks
from cozy_eval.metrics import temporal
from cozy_eval.metrics import tracks as T

cv2 = pytest.importorskip("cv2", reason="track stability needs the `video` extra")

SIZE = 256
FRAMES = 26


def _field(seed: int = 3, size: int = 512) -> np.ndarray:
    """A textured plane with structure at several scales — corners to track."""
    rng = np.random.default_rng(seed)
    base = rng.random((size, size)).astype(np.float32)
    for _ in range(2):
        base = (base + np.roll(base, 1, 0) + np.roll(base, -1, 0)
                + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5.0
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    # hard-edged blocks on top: unambiguous corners, so LK is not solving an
    # aperture problem on smooth noise
    for _ in range(60):
        y, x = rng.integers(0, size - 24, 2)
        base[y:y + rng.integers(8, 22), x:x + rng.integers(8, 22)] = rng.random()
    return base


_PLANE = _field()


def _pan(frames: int = FRAMES, size: int = SIZE, step: float = 2.0) -> np.ndarray:
    """A smooth camera pan across the plane: every point moves on a straight,
    constant-velocity image-plane trajectory. The clean control."""
    out = []
    for t in range(frames):
        x = round(40 + step * t)
        y = round(40 + step * t * 0.4)
        out.append(_PLANE[y:y + size, x:x + size])
    return np.stack([np.repeat(f[..., None], 3, axis=2) for f in out])


def _warble(frames: int = FRAMES, size: int = SIZE, step: float = 2.0,
            amp: float = 1.6, seed: int = 11) -> np.ndarray:
    """The same pan, but a REGION of the scene is re-displaced every frame.

    Each frame is a clean crop of the same sharp plane — no blur, no noise, no
    per-frame damage of any kind. What is wrong is only the TRAJECTORY of the
    points inside the region, which wander instead of moving smoothly. This is
    the failure the owner described: 'each static frame looks roughly correct,
    but objects lose their coherence across frames'.
    """
    rng = np.random.default_rng(seed)
    clip = _pan(frames, size, step)
    out = clip.copy()
    lo, hi = size // 4, 3 * size // 4
    for t in range(frames):
        dx, dy = (rng.normal(0, amp, 2)).round().astype(int)
        x = round(40 + step * t) + dx
        y = round(40 + step * t * 0.4) + dy
        patch = _PLANE[y + lo:y + hi, x + lo:x + hi]
        out[t, lo:hi, lo:hi] = np.repeat(patch[..., None], 3, axis=2)
    return out


def _static_noise(frames: int = FRAMES, size: int = 128) -> np.ndarray:
    """Independent structured noise per frame — nothing survives tracking."""
    return np.stack([
        np.repeat(_field(seed=200 + t, size=size)[..., None], 3, axis=2)
        for t in range(frames)
    ])


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------

def test_a_smooth_pan_keeps_its_tracks_and_scores_high() -> None:
    stats = T.track_stats(_pan())
    # Not 1.0 even on a perfect pan: points leave the frame at the trailing
    # edge. Survival is bounded by the camera move, which is why the GATE is a
    # ratio against a control that made the same move.
    assert stats.track_survival > 0.7
    assert stats.track_stability > 0.5
    assert stats.trackable
    assert stats.track_jitter < T.JITTER_KNEE


def test_warble_collapses_track_stability_while_every_frame_stays_clean() -> None:
    """THE RED TEST. Same sharp frames, same global pan, wrong trajectories."""
    clean, warbled = _pan(), _warble()
    good, bad = T.track_stats(clean), T.track_stats(warbled)
    assert bad.track_stability < good.track_stability * T.STABILITY_RATIO_FLOOR
    assert bad.track_jitter > good.track_jitter

    # ... and the families that already existed cannot see it. Per-frame damage
    # is what they measure, and there is none: the warble arm's frames are
    # crops of the SAME plane. warp_error is a whole-frame flow residual, so it
    # moves far less than the object-level number does.
    good_ratio = bad.track_stability / good.track_stability
    we_clean = temporal.warp_error(clean, pairs=8)
    we_warble = temporal.warp_error(warbled, pairs=8)
    assert good_ratio < 0.9
    assert we_warble / max(we_clean, 1e-9) < 1 / good_ratio


def test_a_fast_pan_is_not_penalized_for_being_fast() -> None:
    """Normalization by each track's own speed is what makes this true."""
    slow = T.track_stats(_pan(step=1.0))
    fast = T.track_stats(_pan(step=4.0))
    assert abs(slow.track_jitter - fast.track_jitter) < 0.05
    assert fast.track_stability > 0.4


def test_untrackable_content_is_unmeasured_never_a_verdict() -> None:
    noise = _static_noise()
    stats = T.track_stats(noise)
    assert not stats.trackable
    verdict = tracks.track_verdict(_pan(), noise)
    assert verdict.verdict == tracks.UNMEASURED
    assert not verdict.ok
    assert "UNTRACKABLE REFERENCE" in verdict.unmeasured["track_stability_ratio"]


def test_a_clip_too_short_for_a_second_derivative_raises() -> None:
    with pytest.raises(errors.ConfigError, match="at least 3 frames"):
        T.track_stats(_pan(frames=2))


def test_uint8_and_float_clips_read_the_same() -> None:
    clip = _pan(frames=12)
    as_u8 = np.clip(clip * 255.0, 0, 255).astype(np.uint8)
    a = T.track_stats(clip)
    b = T.track_stats(as_u8)
    assert abs(a.track_stability - b.track_stability) < 0.02
    assert abs(a.track_survival - b.track_survival) < 0.02


# ---------------------------------------------------------------------------
# determinism — a zero-change arm must score EXACTLY zero change
# ---------------------------------------------------------------------------

def test_scoring_a_clip_twice_is_bit_identical() -> None:
    clip = _pan(frames=14)
    a = T.track_stats(clip)
    b = T.track_stats(clip.copy())
    assert a.track_stability == b.track_stability
    assert a.track_survival == b.track_survival
    assert a.track_jitter == b.track_jitter
    assert a.track_rigidity_error == b.track_rigidity_error


def test_a_bit_identical_pair_scores_exactly_one_and_passes() -> None:
    clip = _pan(frames=14)
    paired = T.track_fidelity(clip, clip.copy())
    assert paired["track_stability_ratio"] == 1.0
    verdict = tracks.track_verdict(clip, clip.copy())
    assert verdict.verdict == tracks.PASS
    assert verdict.measured["track_stability_ratio"] == 1.0


def test_the_camera_fit_is_deterministic_and_not_ransac() -> None:
    """``cv2.estimateAffinePartial2D`` draws from a process-global RNG; two
    scores of the same clip would then differ, and no bit-identical pair could
    ever read exactly 1.0."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 10, (60, 2))
    b = a @ np.array([[0.99, -0.1], [0.1, 0.99]]).T + np.array([3.0, -2.0])
    b[:5] += 50.0  # outliers the robust fit must shrug off
    m1, t1 = T.similarity_fit(a, b)
    m2, t2 = T.similarity_fit(a, b)
    assert np.array_equal(m1, m2) and np.array_equal(t1, t2)
    assert np.allclose(m1, [[0.99, -0.1], [0.1, 0.99]], atol=0.02)
    assert np.allclose(t1, [3.0, -2.0], atol=0.5)


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------

def test_reference_free_reports_but_never_passes() -> None:
    """The absolute level is content-set, so it cannot gate — and the report
    has to SAY that rather than leave a low number looking like a defect."""
    verdict = tracks.track_verdict(_pan())
    assert verdict.verdict == tracks.UNMEASURED
    assert "track_stability" in verdict.measured
    assert "track_stability_ratio" in verdict.unmeasured
    assert any("CONTENT-DEPENDENT" in n for n in verdict.notes)


def test_a_warbling_candidate_is_rejected_and_the_defect_names_the_cause() -> None:
    verdict = tracks.track_verdict(_warble(), _pan())
    assert verdict.verdict == tracks.REJECT
    assert not verdict.ok
    defect = verdict.defects[0]
    assert "OBJECT WARBLE" in defect
    assert "track_stability_ratio" in defect
    assert "jitter" in defect or "survival" in defect or "neighbours" in defect


def test_a_clean_pass_carries_the_scope_note() -> None:
    verdict = tracks.track_verdict(_pan(frames=14), _pan(frames=14))
    assert verdict.verdict == tracks.PASS
    assert any("SCOPE" in n for n in verdict.notes)
    assert "blind to per-frame damage" in verdict.notes[0]


def test_the_paired_block_needs_no_frame_alignment() -> None:
    """Unlike every pixel-paired metric here: each arm is scored on its own
    trajectories, so arms of different length and size are comparable."""
    out = T.track_fidelity(_pan(frames=20), _pan(frames=14, size=192))
    assert 0.0 < out["track_stability_ratio"] < 2.0


# ---------------------------------------------------------------------------
# the registry contract
# ---------------------------------------------------------------------------

def test_the_family_is_declared_in_the_registry_and_only_the_ratio_gates() -> None:
    names = {"track_stability", "track_survival", "track_jitter",
             "track_rigidity_error", "track_stability_ratio"}
    assert names <= set(registry.BY_NAME)
    assert registry.BY_NAME["track_stability_ratio"].gated
    assert registry.BY_NAME["track_stability_ratio"].dimension == registry.SIMILARITY
    for name in names - {"track_stability_ratio"}:
        spec = registry.BY_NAME[name]
        assert spec.dimension == registry.QUALITY
        assert not spec.gated and not spec.paired
    assert registry.METRIC_SET_VERSION == "cozy-eval/metrics@8"


def test_the_floor_sits_in_the_measured_empty_middle() -> None:
    """The banked calibration is the provenance; this pins the constants the
    library ships against it (calibration/track-stability.json)."""
    assert 0.850 < T.STABILITY_RATIO_FLOOR < 0.929
    assert T.TRACKABILITY_FLOOR > 0.20  # loom's 0.18 control must stay excluded
