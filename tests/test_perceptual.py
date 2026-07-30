"""The clip adapter over the no-reference quality scorers. NIQE is closed form,
so the whole path is exercised for real with no weights and no network."""

import numpy as np
import pytest

from cozy_eval.backends import perceptual


def _clip(n=6, size=256, blur=False):
    """A textured scene with camera drift — the regime NSS metrics assume."""
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:size, 0:size] / size
    base = 96 + 64 * np.sin(2 * np.pi * x * 3) * np.cos(2 * np.pi * y * 2)
    spec = np.fft.rfft2(rng.normal(size=(size, size)))
    fy, fx = np.fft.fftfreq(size)[:, None], np.fft.rfftfreq(size)[None, :]
    pink = np.fft.irfft2(spec / np.maximum(np.hypot(fy, fx), 1.0 / size), s=(size, size))
    pink = pink / np.abs(pink).max() * (8 if blur else 48)
    gray = np.clip(base + pink, 0, 255) / 255.0
    return [np.roll(np.stack([gray] * 3, -1), i, axis=1).astype(np.float32) for i in range(n)]


def test_niqe_scores_a_clip_and_names_its_numbers():
    out = perceptual.score_frames(_clip(), model="niqe", stride=2)
    assert set(out) == {"niqe_mean", "niqe_worst", "niqe_frames"}
    assert out["niqe_frames"] == 3
    assert np.isfinite(out["niqe_mean"]) and out["niqe_mean"] > 0


def test_the_worst_frame_follows_the_metrics_direction():
    """NIQE is lower-is-better, so its tail is the HIGH percentile. Reporting
    p10 for it (the old behaviour) named the clip's best frames 'worst'."""
    out = perceptual.score_frames(_clip(), model="niqe", stride=1)
    assert out["niqe_worst"] >= out["niqe_mean"]


def test_frames_reach_the_scorer_in_image_range():
    """iter_frames yields 0..1; every scorer reads 0..255 and is silently wrong,
    not loud, on the wrong scale. A clip of one frame must score as that frame."""
    from PIL import Image

    from cozy_eval.bench.metrics import iqa

    frame = _clip(n=1)[0]
    direct = iqa.niqe(Image.fromarray(np.rint(frame * 255).astype(np.uint8)))
    assert abs(perceptual.score_frames([frame], model="niqe")["niqe_mean"] - direct) < 1e-9


def test_unknown_model_names_the_known_ones():
    with pytest.raises(ValueError, match="niqe"):
        perceptual.score_frames(_clip(), model="brisque")


def test_empty_clip_raises():
    with pytest.raises(ValueError, match="no frames"):
        perceptual.score_frames([], model="niqe")
