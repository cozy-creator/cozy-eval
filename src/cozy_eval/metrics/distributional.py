"""Optional deep-feature Fréchet distances — ``pip install cozy-eval[distributional]``.

**Read this before using it.** FVD-class distances are population statistics with
a well-known small-sample bias: published FVD is computed over thousands of
clips, and at the n=8-16 a per-artifact quality gate can afford, the estimator's
bias dominates the effect you are trying to measure. This library's *gate* does
not use a deep-feature Fréchet distance for that reason — its distributional
benchmark is a paired test over the no-reference score distributions, which is
valid at n=8 because the prompt set is identical between arms.

This module exists so a lane that CAN afford n in the hundreds gets the stronger
statistic without leaving the protocol machinery behind. It reports the sample
size next to every number and refuses below ``min_samples``.
"""

from __future__ import annotations

import numpy as np

from ..errors import SampleSizeError


def frechet(feat_a: np.ndarray, feat_b: np.ndarray, *, min_samples: int = 64) -> float:
    """Fréchet distance between two Gaussian-fitted feature populations.

    Full covariance; requires ``n >= min_samples`` on both sides. Uses SciPy's
    matrix square root when available and an eigendecomposition otherwise.
    """
    a, b = np.asarray(feat_a, np.float64), np.asarray(feat_b, np.float64)
    if min(len(a), len(b)) < min_samples:
        raise SampleSizeError(
            f"Frechet distance over {min(len(a), len(b))} samples is dominated by "
            f"estimator bias (need >= {min_samples}). Use the paired population "
            "benchmark instead — see GATE.md."
        )
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    try:
        from scipy.linalg import sqrtm
        covmean = sqrtm(ca @ cb)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except ImportError:                                        # pragma: no cover
        w, v = np.linalg.eigh(ca)
        root = v @ np.diag(np.sqrt(np.clip(w, 0, None))) @ v.T
        w2, v2 = np.linalg.eigh(root @ cb @ root)
        covmean = v2 @ np.diag(np.sqrt(np.clip(w2, 0, None))) @ v2.T
    return float(diff @ diff + np.trace(ca + cb - 2.0 * covmean))


def fvd(real_dir, generated_dir, *, model: str = "i3d", device: str = "cuda") -> float:
    """Content-debiased FVD via ``cd-fvd`` (MIT). Not reimplemented here.

    Vendoring yet another I3D checkpoint loader is how FVD numbers stopped being
    comparable between papers in the first place.

    Note (verified 2026-07-26): cd-fvd's PyPI release points ``model="videomae"``
    at a dead weights URL; the HuggingFace fix landed on ``main`` after the last
    upload. Use ``model="i3d"`` from PyPI, or install cd-fvd from git for
    VideoMAEv2 features.
    """
    try:
        from cdfvd import fvd as _fvd
    except ImportError as exc:                                 # pragma: no cover
        raise RuntimeError(
            "needs cd-fvd: pip install 'cozy-eval[distributional]'"
        ) from exc
    ev = _fvd.cdfvd(model, device=device)
    ev.compute_real_stats(ev.load_videos(str(real_dir), data_type="video_folder"))
    ev.compute_fake_stats(ev.load_videos(str(generated_dir), data_type="video_folder"))
    return float(ev.compute_fvd_from_stats())


def fvmd(real_dir, generated_dir, log_dir) -> float:           # pragma: no cover
    """Fréchet Video Motion Distance via ``fvmd`` (Apache-2.0). Not reimplemented.

    Motion-statistics distance over PIPs++ keypoint tracks. Reported to correlate
    with human judgement better than FVD or VBench on motion artifacts, which is
    the failure mode this library's temporal benchmark also targets — run both if
    you can afford the keypoint tracker.

    NOT declared as a dependency: fvmd 1.0.0 pins ``scipy==1.10.1``, which is
    unsatisfiable alongside anything modern. Install it into its own environment.
    """
    try:
        from fvmd import fvmd as _fvmd
    except ImportError as exc:
        raise RuntimeError(
            "needs fvmd: pip install 'cozy-eval[distributional]'"
        ) from exc
    return float(_fvmd(str(log_dir), str(generated_dir), str(real_dir)))
