"""Video: frame handling, the Δ-frame temporal channel, and composed signal stats.

STABILITY: experimental (v0.x) for these function signatures; the metric NAMES
(``dframe_psnr``, ``dframe_ssim``, ``lpips_frame_worst``, ``luma_flicker``,
``jerk_ratio``) are locked by the registry.

THE Δ-FRAME CHANNEL is the axis per-frame metrics cannot see. A candidate can
match the reference frame-by-frame on LPIPS/PSNR and still FLICKER: its errors
alternate in sign from frame to frame while the reference's dynamics are smooth.
Differencing consecutive frames first — d_t = f_{t+1} - f_t — and then scoring
the candidate's difference-images against the reference's compares *what moved
between frames*, where flicker and smear are first-order signals instead of
noise buried under static content.

FRAME CONVENTION (matches cozy_eval and the conversion video gate):
``(T, H, W, 3)`` RGB, uint8 or float in [0, 1] — the PRE-ENCODE arrays a
producer already holds. A list of PIL images or of ``(H, W, 3)`` arrays is
accepted too. Nothing here decodes video files: the encoder is a separate
change that does not belong inside a generation verdict, and this package
takes no ffmpeg dependency for it.

SINGLE-ARM temporal statistics (flicker/jerk on one clip, no reference) are
NOT implemented here: ``cozy_eval.metrics.signal`` already owns them, so
:func:`signal_stats` composes it directly.
"""

from __future__ import annotations

from typing import Any

from ..errors import ConfigError

_PSNR_CAP = 99.0  # identical Δ-frames -> inf; the same finite sentinel similarity.py uses


def as_frames(source: Any) -> Any:
    """Normalize any accepted clip form to ``(T, H, W, 3)`` float32 in [0, 1].

    Accepts a ``(T, H, W, 3)`` ndarray (uint8 or float), a list/tuple of PIL
    images, or a list/tuple of ``(H, W, 3)`` arrays. A video needs at least two
    frames — a single frame is an image and belongs in the image suite.
    """
    import numpy as np

    if isinstance(source, (list, tuple)):
        if not source:
            raise ConfigError("as_frames: empty frame list")
        stacked = np.stack([
            np.asarray(f.convert("RGB") if hasattr(f, "convert") else f)
            for f in source
        ])
    else:
        stacked = np.asarray(source)
    if stacked.ndim != 4 or stacked.shape[-1] != 3:
        raise ConfigError(
            f"as_frames: expected (T, H, W, 3) RGB frames, got shape {stacked.shape}"
        )
    if stacked.shape[0] < 2:
        raise ConfigError(
            f"as_frames: a video needs at least 2 frames, got {stacked.shape[0]}; "
            "score single frames with the image suite"
        )
    frames = stacked.astype(np.float32)
    if stacked.dtype == np.uint8 or float(frames.max(initial=0.0)) > 1.5:
        frames = frames / 255.0
    return np.clip(frames, 0.0, 1.0)


def frame_images(frames: Any) -> list[Any]:
    """uint8 PIL views of ``(T, H, W, 3)`` float frames, for the image metrics."""
    import numpy as np
    from PIL import Image

    return [
        Image.fromarray(np.clip(f * 255.0, 0, 255).astype(np.uint8))
        for f in frames
    ]


def sample_indices(total: int, count: int) -> list[int]:
    """``count`` frame indices spread uniformly over ``[0, total)``, always
    including the first and last frame — motion is judged by its endpoints."""
    if total <= 0:
        raise ConfigError("sample_indices: empty clip")
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    step = (total - 1) / (count - 1)
    return sorted({round(i * step) for i in range(count)})


def dframes(frames: Any) -> Any:
    """Signed consecutive-frame differences, ``(T-1, H, W, 3)`` in [-1, 1]."""
    return frames[1:] - frames[:-1]


def dframe_psnr_series(reference: Any, candidate: Any) -> list[float]:
    """Per-step PSNR between the two clips' Δ-frames, in dB.

    ``data_range`` is 1.0 — the FRAME range, not the Δ range — so the numbers
    read on the same scale as frame PSNR: identical dynamics -> the 99.0
    sentinel, flicker against a smooth reference -> tens of dB below it.
    """
    import numpy as np

    _check_pair(reference, candidate, metric="dframe_psnr")
    out = []
    for dr, dc in zip(dframes(reference), dframes(candidate), strict=True):
        mse = float(np.mean((dr - dc) ** 2))
        if mse == 0.0:
            out.append(_PSNR_CAP)
        else:
            out.append(min(_PSNR_CAP, float(-10.0 * np.log10(mse))))
    return out


def dframe_ssim_series(reference: Any, candidate: Any) -> list[float]:
    """Per-step SSIM between the two clips' Δ-frames.

    SSIM is defined on images, not on signed differences, so each Δ-frame is
    affinely mapped to image range first (``(d + 1) / 2``) — structure and
    contrast of *what moved* are compared where the zero-motion level sits at
    mid-grey. Computed by the same torchmetrics path as frame SSIM.
    """
    import numpy as np

    from . import similarity

    _check_pair(reference, candidate, metric="dframe_ssim")
    out = []
    for dr, dc in zip(dframes(reference), dframes(candidate), strict=True):
        a = np.clip((dr + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        b = np.clip((dc + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        out.append(similarity.ssim(a, b))
    return out


def _check_pair(reference: Any, candidate: Any, *, metric: str) -> None:
    if reference.shape != candidate.shape:
        raise ConfigError(
            f"{metric}: the two clips differ in shape, {tuple(reference.shape)} vs "
            f"{tuple(candidate.shape)} — a paired temporal metric needs frame-aligned "
            "clips of the same length and size; resize/trim the candidate first"
        )


def signal_stats(frames: Any) -> dict[str, float]:
    """Single-arm temporal signal statistics, composed from cozy_eval's signal backend.

    Returns ``{"luma_flicker": ..., "jerk_ratio": ...}``.
    ``luma_flicker`` is the backend's ``flicker`` (frame-mean luma wobble, %);
    ``jerk_ratio`` keeps its name (second/first temporal difference of
    frame-mean luma; smooth motion sits low, flicker and judder push it up).
    """
    from cozy_eval.metrics.signal import score

    clip = score(frames)
    return {
        "luma_flicker": float(clip.flicker),
        "jerk_ratio": float(clip.jerk_ratio),
    }


SIGNAL_LIBRARY = "cozy-eval:signal"


__all__ = [
    "SIGNAL_LIBRARY",
    "as_frames",
    "dframe_psnr_series",
    "dframe_ssim_series",
    "dframes",
    "frame_images",
    "sample_indices",
    "signal_stats",
]
