"""Dimension 1: perceptual similarity to a reference render (PAIRED).

STABILITY: experimental (v0.x) for these function signatures; the metric NAMES
(``lpips``, ``ssim``, ``ms_ssim``, ``psnr``) are locked by the registry.

LPIPS is the gated headline. PSNR/SSIM/MS-SSIM are report-only — the
correlation study measured |r| 0.83-0.96 between all four on banked pairs, so
they are one signal, and gating four correlated numbers only multiplies the
chance of failing a good arm.

BUILD-vs-BUY (survey): every metric here comes from **torchmetrics**
(Apache-2.0, v1.9.0), not from our own code. Its SSIM/MS-SSIM/LPIPS are
CI-tested against the canonical implementations (skimage, pytorch_msssim, and
Zhang's original `lpips`), so adopting it deletes a hand-rolled numpy SSIM and
the standalone `lpips` package in one move. Defaults already match Wang 2004
(gaussian_kernel, sigma 1.5, K=(0.01, 0.03)).

Not adopted, and why: `pyiqa`/IQA-PyTorch is PolyForm **Noncommercial** —
repo-wide, no clean subset; `piq` is stale (2023) and redundant here.
"""

from __future__ import annotations

from typing import Any

from ..device import AUTO, resolve_device
from ..errors import ConfigError

# Keyed by (metric, model, device): a cache that ignores the device silently
# returns a model placed on the wrong one.
_MODEL_CACHE: dict[str, Any] = {}

_PSNR_CAP = 99.0  # identical images -> inf; report a finite sentinel
_SSIM_WIN = 11    # torchmetrics derives this from sigma=1.5; guard on it too
# Wang 2003 multi-scale weights.
_MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def free_models() -> None:
    _MODEL_CACHE.clear()


def _tensor(image: Any) -> Any:
    """uint8 RGB PIL/ndarray -> float32 [1,3,H,W] in 0..255."""
    import numpy as np
    import torch

    arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    t = torch.from_numpy(np.array(arr, copy=True)).float()
    return t.permute(2, 0, 1).unsqueeze(0)


def _pair(a: Any, b: Any, *, metric: str) -> tuple[Any, Any]:
    x, y = _tensor(a), _tensor(b)
    if x.shape != y.shape:
        raise ConfigError(
            f"{metric}: the two images differ in shape, "
            f"{tuple(x.shape[-2:])} vs {tuple(y.shape[-2:])} — a paired metric needs "
            "identically sized images; resize the candidate to the reference first"
        )
    if min(x.shape[-2:]) < _SSIM_WIN:
        raise ConfigError(
            f"{metric}: image is {tuple(x.shape[-2:])}, smaller than the "
            f"{_SSIM_WIN}px gaussian window this metric convolves with"
        )
    return x, y


def psnr(a: Any, b: Any) -> float:
    """PSNR in dB over uint8 RGB images. REPORT-ONLY: see the redundancy evidence in registry.py."""
    import numpy as np
    from torchmetrics.functional.image import peak_signal_noise_ratio

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ConfigError(
            f"psnr: the two images differ in shape, {x.shape} vs {y.shape}"
        )
    if float(np.mean((x - y) ** 2)) == 0.0:
        return _PSNR_CAP
    value = peak_signal_noise_ratio(_tensor(a), _tensor(b), data_range=255.0)
    return min(_PSNR_CAP, float(value))


def ssim(a: Any, b: Any) -> float:
    """Mean SSIM (Wang 2004) over uint8 images; 1.0 == identical."""
    from torchmetrics.functional.image import structural_similarity_index_measure

    x, y = _pair(a, b, metric="ssim")
    return float(structural_similarity_index_measure(
        x, y, data_range=255.0, gaussian_kernel=True, sigma=1.5,
    ))


def ms_ssim(a: Any, b: Any) -> float:
    """Multi-scale SSIM. The scale count drops (weights renormalized) on images
    too small for all five, so a small eval still returns a real number rather
    than raising — torchmetrics itself requires the full five."""
    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure,
    )

    x, y = _pair(a, b, metric="ms_ssim")
    levels, side = 0, min(x.shape[-2:])
    while levels < len(_MS_WEIGHTS) and side >= _SSIM_WIN:
        levels += 1
        side //= 2
    betas = _MS_WEIGHTS[:levels]
    total = sum(betas)
    return float(multiscale_structural_similarity_index_measure(
        x, y, data_range=255.0, gaussian_kernel=True, sigma=1.5,
        betas=tuple(b_ / total for b_ in betas),
    ))


def lpips_model(device: str = AUTO) -> Any:
    """torchmetrics' LPIPS — a direct BSD-2 port of Zhang's original, with the
    linear-layer weights shipped in-package (only the torchvision AlexNet
    backbone is fetched, ~245 MB, and is baked into the image)."""
    device = resolve_device(device)
    key = f"lpips:alex:{device}"
    if key not in _MODEL_CACHE:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        _MODEL_CACHE[key] = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True,
        ).to(device).eval()
    return _MODEL_CACHE[key]


def lpips_pair(model: Any, a: Any, b: Any, device: str = AUTO) -> float:
    import torch

    device = resolve_device(device)

    with torch.no_grad():
        return float(model(
            (_tensor(a) / 255.0).to(device), (_tensor(b) / 255.0).to(device),
        ))


CLIP_MODEL = "openai/clip-vit-base-patch32"


def clip_model(device: str = AUTO, *, model_ref: str = CLIP_MODEL) -> tuple[Any, Any]:
    device = resolve_device(device)
    key = f"clip:{model_ref}:{device}"
    if key not in _MODEL_CACHE:
        from transformers import CLIPModel, CLIPProcessor

        _MODEL_CACHE[key] = (
            CLIPModel.from_pretrained(model_ref).to(device).eval(),
            CLIPProcessor.from_pretrained(model_ref),
        )
    return _MODEL_CACHE[key]


def clip_score(
    model: Any, proc: Any, image: Any, prompt: str, device: str = AUTO,
) -> float:
    """torchmetrics-style CLIP score: 100 * max(0, cos(image_emb, text_emb)).

    Kept on plain transformers rather than ``torchmetrics.multimodal``: the
    model is already loaded and the extra would drag timm/einops/piq for a
    cosine we compute in four lines.
    """
    import torch

    device = resolve_device(device)
    inputs = proc(
        text=[prompt[:300]], images=[image], return_tensors="pt",
        padding=True, truncation=True,
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
        img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_e = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float(max(0.0, (img_e * txt_e).sum().item()) * 100.0)


__all__ = [
    "CLIP_MODEL",
    "clip_model",
    "clip_score",
    "free_models",
    "lpips_model",
    "lpips_pair",
    "ms_ssim",
    "psnr",
    "ssim",
]
