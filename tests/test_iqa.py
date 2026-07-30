"""NIQE is closed-form (no weights), so it is tested for real on synthetic
images. The torchmetrics-backed metrics (arniqa, clip_iqa) download weights and
are `heavy`-marked.

Registry wiring lives in `test_public_api.py`'s registry sweep.
"""

import numpy as np
import pytest

from cozy_eval.bench.metrics import iqa


def _textured(size=384, seed=7):
    """A natural-ish test image: smooth low-frequency structure + mild texture.
    Pure white noise or a flat field are both PATHOLOGICAL for NSS metrics, so
    tests use something with actual scene statistics."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    base = 96 + 64 * np.sin(2 * np.pi * x * 3) * np.cos(2 * np.pi * y * 2)
    spectrum = np.fft.rfft2(rng.normal(size=(size, size)))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    pink = np.fft.irfft2(spectrum / np.maximum(np.hypot(fy, fx), 1.0 / size), s=(size, size))
    pink = pink / np.abs(pink).max() * 48
    return np.clip(base + pink, 0, 255).astype(np.uint8)


def test_niqe_is_finite_and_deterministic_across_every_input_form():
    """Gray ndarray, RGB ndarray and PIL must all reach the same number: the
    metric reads luma, so the input container cannot change the score."""
    from PIL import Image

    gray = _textured()
    rgb = np.stack([gray] * 3, axis=-1)

    a, b = iqa.niqe(gray), iqa.niqe(gray)
    assert np.isfinite(a) and a > 0
    assert a == b
    # BT.601 luma weights sum to 0.9999, so the RGB path differs by a hair.
    assert abs(a - iqa.niqe(rgb)) < 1e-3
    assert abs(iqa.niqe(Image.fromarray(rgb)) - iqa.niqe(rgb)) < 1e-6


def test_white_noise_scores_worse_than_structure():
    """The defining property of a pristine-statistics model: white noise is far
    from natural scenes."""
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, size=(384, 384), dtype=np.uint8)
    assert iqa.niqe(noise) > iqa.niqe(_textured())


def test_small_image_raises():
    with pytest.raises(ValueError, match="needs at least 192x192"):
        iqa.niqe(np.zeros((100, 100), dtype=np.uint8))


def test_niqe_parity_with_banked_oracle():
    """Numbers banked from the NC reference (see parity/run_oracle.py); the NC
    package itself is never installed here. This is the test that caught a
    reimplementation whose values were 200% off with the ranking inverted."""
    import json
    from pathlib import Path

    from PIL import Image

    fixtures = Path(__file__).parent / "fixtures"
    spec = json.loads((fixtures / "oracle_niqe.json").read_text())
    for name, ref in spec["values"].items():
        ours = iqa.niqe(Image.open(fixtures / f"{name}.png"))
        assert abs(ours - ref) / ref < spec["tolerance_rel"], (
            f"{name}: ours={ours:.3f} oracle={ref:.3f}"
        )


def _banked(metric):
    import json
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    return fixtures, json.loads((fixtures / f"oracle_{metric}.json").read_text())


@pytest.mark.heavy
@pytest.mark.parametrize("metric", ["arniqa", "clip_iqa"])
def test_learned_metrics_score_in_range(metric):
    img = np.stack([_textured()] * 3, axis=-1)
    assert 0.0 <= getattr(iqa, metric)(img) <= 1.0


@pytest.mark.heavy
def test_clip_iqa_reproduces_the_oracle_under_the_oracle_prompts():
    """The only difference from the NC reference is the prompt set: ours is the
    paper's single canonical pair, pyiqa averages five. Given its five, the two
    implementations agree to 1e-4 — so the substitution is the same metric, and
    the default difference is a deliberate choice rather than a port error."""
    from PIL import Image

    fixtures, spec = _banked("clip_iqa")
    prompts = tuple(tuple(p) for p in spec["oracle_config"]["prompts"])
    for name, ref in spec["values"].items():
        ours = iqa.clip_iqa(Image.open(fixtures / f"{name}.png"), prompts=prompts)
        assert abs(ours - ref) / ref < spec["tolerance_rel"], (
            f"{name}: ours={ours:.6f} oracle={ref:.6f}"
        )


@pytest.mark.heavy
def test_arniqa_matches_its_banked_values():
    """arniqa deliberately diverges from the NC reference (antialiased half-scale
    vs its aliased decimation — see the json's `divergence`), so the banked
    guard is our own number: it catches a torchmetrics change silently moving
    the metric, which matters because nothing here carries an upper bound."""
    from PIL import Image

    fixtures, spec = _banked("arniqa")
    for name, ref in spec["ours"].items():
        ours = iqa.arniqa(Image.open(fixtures / f"{name}.png"))
        assert abs(ours - ref) / ref < spec["tolerance_rel"], (
            f"{name}: ours={ours:.6f} banked={ref:.6f}"
        )
