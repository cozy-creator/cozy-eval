"""Dimension 4: technical quality — does the image LOOK good, no reference, no prompt.

STABILITY: experimental (v0.x). Metric NAMES (``arniqa``, ``clip_iqa``,
``niqe``) are locked by the registry; these function signatures are not.

ATTRIBUTION: the NIQE pristine-model MVG parameters in
``metrics/data/niqe_pristine.npz`` are adapted from scikit-video, which is
BSD-3-Clause licensed: Copyright (c) 2015-2018 Alex Reinhard and Todd Goodall.
Redistribution in binary form must reproduce that notice. The NIQE code itself
is our own implementation of Mittal et al. 2013. See PROVENANCE.md.


Orthogonal to adherence (a sharp, clean render of the wrong thing) and to
preference (which mixes aesthetics and prompt echo). Reference-free, so it
scores any render and never needs a paired baseline.

BUILD-vs-BUY per metric:

  arniqa    torchmetrics (Apache-2.0) — the strongest permissively-packaged
            learned NR metric available; dimension headline.
  clip_iqa  torchmetrics (Apache-2.0) — prompt-pair CLIP quality probe.
  niqe      OUR implementation of Mittal et al. 2013, "Making a 'Completely
            Blind' Image Quality Analyzer" — closed-form natural-scene
            statistics, no learned weights, runs anywhere numpy runs. The
            pristine-model MVG parameters are adapted from scikit-video
            (BSD-3, attributed in PROVENANCE.md).
  musiq     OUR PyTorch port of the Apache reference — see musiq.py.

Where we deliberately differ from `pyiqa`, the NC oracle these numbers are
banked against (`parity/run_oracle.py`, tolerances in tests/fixtures):

  clip_iqa  default prompt pair is the paper's ("Good photo."/"Bad photo.");
            pyiqa averages five hand-written pairs. Pass `prompts=` for its set —
            the implementations then agree to 1e-4, so the gap is the prompt
            choice, nothing else.
  arniqa    half-scale branch uses an ANTIALIASED resize. pyiqa decimates with
            `F.interpolate(..., mode="bilinear")`, which aliases exactly the
            high-frequency energy an NR quality metric reads (its whitenoise
            fixture scores 0.376 against our 0.576). Same weights otherwise:
            swapping the resize reproduces pyiqa to 1e-4.

NOT implemented, and why: BRISQUE (superseded for our purposes by NIQE, the
opinion-unaware successor from the same authors, and its SVR weights have no
clean provenance), MANIQA and TOPIQ (learned NR scorers with no permissive
checkpoint; TOPIQ's value IS its NC-trained weights). All three were reachable
through pyiqa and are dropped rather than ported — nothing in this library
called them. See PROVENANCE.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..device import AUTO, resolve_device
from ..errors import ConfigError

_MODEL_CACHE: dict[str, Any] = {}
_DATA = Path(__file__).parent / "data"

# NIQE constants from the paper: 96px patches at scale 1, two scales, patches
# kept when their sharpness clears 0.75 of the image maximum.
_NIQE_PATCH = 96
_NIQE_SHARPNESS_QUANTILE = 0.75


def free_models() -> None:
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# NIQE — Mittal, Soundararajan, Bovik 2013 (IEEE SPL). Closed form.
# ---------------------------------------------------------------------------

def _gaussian_window() -> Any:
    """7x7 gaussian, sigma 7/6, unit sum — the paper's local-statistics window."""
    import numpy as np

    half = 3
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    w = np.exp(-(x * x + y * y) / (2.0 * (7.0 / 6.0) ** 2))
    return w / w.sum()


def _mscn(gray: Any) -> tuple[Any, Any]:
    """Mean-subtracted contrast-normalized coefficients + the sigma field."""
    import numpy as np
    from scipy.ndimage import correlate

    w = _gaussian_window()
    mu = correlate(gray, w, mode="nearest")
    sigma = np.sqrt(np.abs(correlate(gray * gray, w, mode="nearest") - mu * mu))
    return (gray - mu) / (sigma + 1.0), sigma


def _aggd_fit(x: Any) -> tuple[float, float, float]:
    """Asymmetric GGD fit -> (shape alpha, left scale beta_l, right scale
    beta_r). Betas are SCALE parameters (std * sqrt(G(1/a)/G(3/a))) — the
    convention the pristine model was fit under; variances would not match."""
    import numpy as np
    from scipy.special import gamma as G

    shapes = np.arange(0.2, 10.001, 0.001)
    ratios = ((G(2.0 / shapes) / G(1.0 / shapes)) ** 2
              / (G(3.0 / shapes) / G(1.0 / shapes)))
    left, right = x[x < 0], x[x > 0]
    l_std = float(np.sqrt(np.mean(left * left))) if left.size else 0.0
    r_std = float(np.sqrt(np.mean(right * right))) if right.size else 0.0
    if l_std == 0.0 or r_std == 0.0:
        return 10.0, 0.0, 0.0
    gammahat = l_std / r_std
    rhat = float(np.mean(np.abs(x))) ** 2 / float(np.mean(x * x))
    rhatnorm = rhat * (gammahat**3 + 1.0) * (gammahat + 1.0) / (gammahat**2 + 1.0) ** 2
    alpha = float(shapes[np.argmin((ratios - rhatnorm) ** 2)])
    scale = float(np.sqrt(G(1.0 / alpha) / G(3.0 / alpha)))
    return alpha, l_std * scale, r_std * scale


def _patch_features(patch: Any) -> Any:
    """18 NSS features per patch, the pristine fit's exact convention:
    AGGD(mscn) -> [alpha, (beta_l+beta_r)/2], then per orientation
    (H, V, D1, D2) -> [alpha, eta, beta_l, beta_r] with
    eta = (beta_r - beta_l) * G(2/a)/G(1/a)."""
    import numpy as np
    from scipy.special import gamma as G

    alpha, bl, br = _aggd_fit(patch.ravel())
    feats = [alpha, (bl + br) / 2.0]
    pairs = (
        patch[:, :-1] * patch[:, 1:],            # horizontal
        patch[:-1, :] * patch[1:, :],            # vertical
        patch[:-1, :-1] * patch[1:, 1:],         # main diagonal
        patch[:-1, 1:] * patch[1:, :-1],         # secondary diagonal
    )
    for p in pairs:
        alpha, bl, br = _aggd_fit(p.ravel())
        eta = (br - bl) * (G(2.0 / alpha) / G(1.0 / alpha))
        feats += [alpha, eta, bl, br]
    return np.asarray(feats, dtype=np.float64)


def _to_gray(image: Any) -> Any:
    """RGB -> luma, float64, ITU-R BT.601 (Matlab rgb2gray) coefficients."""
    import numpy as np

    arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    arr = arr.astype(np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0] * 0.2989 + arr[..., 1] * 0.5870 + arr[..., 2] * 0.1140
    return arr


def _halve(gray: Any) -> Any:
    """Antialiased 2x downscale (the paper's second scale)."""
    import numpy as np
    from PIL import Image

    im = Image.fromarray(gray.astype(np.float32), mode="F")
    return np.asarray(
        im.resize((max(gray.shape[1] // 2, 1), max(gray.shape[0] // 2, 1)), Image.BICUBIC),
        dtype=np.float64,
    )


def niqe(image: Any) -> float:
    """Completely-blind quality: distance between this image's NSS statistics
    and a pristine-image model. LOWER is better; typical range ~2 (excellent)
    to ~10+ (badly distorted). Needs at least 192x192 pixels (two 96px patches
    per axis at scale 1)."""
    import numpy as np

    gray = _to_gray(image)
    h, w = gray.shape
    hs, ws = (h // _NIQE_PATCH) * _NIQE_PATCH, (w // _NIQE_PATCH) * _NIQE_PATCH
    if hs < 2 * _NIQE_PATCH or ws < 2 * _NIQE_PATCH:
        raise ConfigError(
            f"niqe: image is {h}x{w} but needs at least "
            f"{2 * _NIQE_PATCH}x{2 * _NIQE_PATCH} px (two {_NIQE_PATCH}px patches per axis)"
        )
    gray = gray[:hs, :ws]

    rows, cols = hs // _NIQE_PATCH, ws // _NIQE_PATCH
    scales: list[Any] = []
    mscn1, sigma1 = _mscn(gray)
    scales.append((mscn1, _NIQE_PATCH))
    mscn2, _ = _mscn(_halve(gray))
    scales.append((mscn2, _NIQE_PATCH // 2))

    # Patch selection happens at scale 1 on local sharpness, then the same
    # patch grid indexes both scales.
    sharpness = np.array([
        [sigma1[r * _NIQE_PATCH:(r + 1) * _NIQE_PATCH,
                c * _NIQE_PATCH:(c + 1) * _NIQE_PATCH].mean() for c in range(cols)]
        for r in range(rows)
    ])
    keep = sharpness > _NIQE_SHARPNESS_QUANTILE * sharpness.max()
    if not keep.any():
        keep = sharpness >= sharpness.max()

    features = []
    for r in range(rows):
        for c in range(cols):
            if not keep[r, c]:
                continue
            per_scale = []
            for mscn, size in scales:
                per_scale.append(_patch_features(
                    mscn[r * size:(r + 1) * size, c * size:(c + 1) * size]
                ))
            features.append(np.concatenate(per_scale))
    feats = np.asarray(features)

    pristine = np.load(_DATA / "niqe_pristine.npz")
    mu_p, cov_p = pristine["mu"], pristine["cov"]
    mu_d = feats.mean(axis=0)
    cov_d = np.cov(feats, rowvar=False) if feats.shape[0] > 1 else np.zeros_like(cov_p)
    diff = mu_p - mu_d
    inv = np.linalg.pinv((cov_p + cov_d) / 2.0)
    return float(np.sqrt(max(float(diff @ inv @ diff), 0.0)))


# ---------------------------------------------------------------------------
# torchmetrics integrations (Apache-2.0) — buy, don't build
# ---------------------------------------------------------------------------

def _image_tensor(image: Any) -> Any:
    """uint8 RGB -> float32 [1,3,H,W] in 0..1."""
    import numpy as np
    import torch

    arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    return torch.from_numpy(np.array(arr, copy=True)).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def arniqa(image: Any, *, regressor: str = "koniq10k", device: str = AUTO) -> float:
    """ARNIQA (Agnolucci et al. 2024) via torchmetrics; 0..1, higher is better."""
    import torch

    device = resolve_device(device)
    key = f"arniqa:{regressor}:{device}"
    if key not in _MODEL_CACHE:
        from torchmetrics.image.arniqa import ARNIQA

        _MODEL_CACHE[key] = ARNIQA(regressor_dataset=regressor).to(device).eval()
    model = _MODEL_CACHE[key]
    with torch.no_grad():
        return float(model(_image_tensor(image).to(device)))


#: The paper's canonical quality probe. `pyiqa` instead averages five pairs,
#: which is a different quantity, not a better-estimated one.
QUALITY_PROMPTS: tuple[Any, ...] = ("quality",)


def clip_iqa(image: Any, *, prompts: tuple[Any, ...] = QUALITY_PROMPTS,
             device: str = AUTO) -> float:
    """CLIP-IQA (Wang et al. 2023) via torchmetrics; 0..1, higher is better.

    ``prompts`` takes torchmetrics' antonym-pair form — a builtin name like
    ``"quality"`` or an explicit ``(positive, negative)`` tuple. More than one
    pair is averaged.

    Backbone weights: OpenAI CLIP — no licence tag on the HF card, MIT at the
    upstream repo; recorded as such in PROVENANCE.md.
    """
    import torch

    device = resolve_device(device)
    key = f"clip_iqa:{device}:{prompts}"
    if key not in _MODEL_CACHE:
        from torchmetrics.multimodal import CLIPImageQualityAssessment

        _MODEL_CACHE[key] = CLIPImageQualityAssessment(prompts=prompts).to(device).eval()
    model = _MODEL_CACHE[key]
    # data_range is 1.0, so the model wants 0..1 — feeding it 0..255 puts every
    # pixel far outside CLIP's normalization and silently returns nonsense.
    with torch.no_grad():
        out = model(_image_tensor(image).to(device))
    if isinstance(out, dict):
        return float(torch.stack([v for v in out.values()]).mean())
    return float(out)


__all__ = [
    "QUALITY_PROMPTS",
    "arniqa",
    "clip_iqa",
    "free_models",
    "niqe",
]
