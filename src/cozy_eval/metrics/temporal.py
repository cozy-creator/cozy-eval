"""Video: frame handling, the Δ-frame temporal channel, the optical-flow
temporal-fidelity family, and composed signal stats.

STABILITY: experimental (v0.x) for these function signatures; the metric NAMES
(``dframe_psnr``, ``dframe_ssim``, ``lpips_frame_worst``, ``luma_flicker``,
``jerk_ratio``, ``flow_divergence``, ``warp_error``, ``warp_error_delta``) are
locked by the registry.

THE TEMPORAL-FIDELITY FAMILY (``flow_divergence`` / ``warp_error`` /
``warp_error_delta``) answers a question screenshots cannot: *do the movements
match*. Every paired number beside it — LPIPS/PSNR/SSIM/DISTS/ringing — is
computed per FRAME; averaging stills is blind to motion, which is the axis a
sparse-attention or step-distill lane is most likely to damage silently. The
Δ-frame channel above is a pixel-difference proxy for motion; this family
compares the MOTION FIELDS themselves (Farneback optical flow), so a candidate
whose objects move along different trajectories is caught even when each frame
is individually faithful. ``warp_error`` is the reference-free half (shimmer /
boiling / flicker within one clip); ``flow_divergence`` and ``warp_error_delta``
are the paired halves against a same-seed control — valid for fp8-vs-bf16 /
quant-vs-dense pairs, exactly like the rest of the SIMILARITY dimension.

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

# ---------------------------------------------------------------------------
# the temporal-fidelity family — optical flow: do the movements match?
# ---------------------------------------------------------------------------

FLOW_LIBRARY = "cozy-eval:flow"

#: Working HEIGHT the flow is computed at. Farneback is O(pixels); a paired EPE
#: is normalized by the reference's own motion magnitude so it is robust to this
#: choice, and 384 keeps a 10 s 768p clip to a few seconds on CPU. Width follows
#: the aspect ratio, rounded even.
FLOW_TARGET_H = 384

#: How many consecutive (t, t+1) frame pairs to sample, spread uniformly over
#: the clip (endpoints included). Motion is judged from a spread of transitions,
#: not every one — 24 pairs is plenty and bounds cost on a long clip.
FLOW_PAIRS = 24

# Farneback parameters (pyr_scale, levels, winsize, iterations, poly_n,
# poly_sigma, flags). OpenCV's documented defaults for dense flow; the same
# settings the ie#636 A/B lane derived its ad-hoc numbers with, so the library
# and that report read on one scale.
_FARNEBACK = (0.5, 3, 15, 3, 5, 1.2, 0)


def _pair_starts(total: int, count: int) -> list[int]:
    """Uniformly-spread start indices ``k`` for consecutive ``(k, k+1)`` pairs."""
    if total < 2:
        raise ConfigError("optical flow needs at least 2 frames")
    last = total - 2  # k+1 must be in range
    if count >= last + 1:
        return list(range(last + 1))
    if count == 1:
        return [0]
    step = last / (count - 1)
    return sorted({round(i * step) for i in range(count)})


def _small_gray_frame(frame: Any, size: tuple[int, int]) -> Any:
    """One ``(H,W,3)`` float frame -> working-resolution uint8 grey."""
    import cv2
    import numpy as np

    u8 = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
    small = cv2.resize(u8, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def _work_size(h: int, w: int, target_h: int) -> tuple[int, int]:
    if h <= target_h:
        return w, h
    sw = max(2, round(w * target_h / h) & ~1)
    return sw, target_h


def flow_fields(frames: Any, *, pairs: int = FLOW_PAIRS,
                target_h: int = FLOW_TARGET_H) -> tuple[list[Any], dict[int, Any], list[int]]:
    """Farneback dense flow for a spread of consecutive frame pairs.

    Returns ``(flows, gray, starts)``: ``flows[i]`` is the ``(h, w, 2)`` flow
    from frame ``starts[i]`` to ``starts[i]+1`` at the working resolution, and
    ``gray`` is ``{index: uint8 grey frame}`` for JUST the sampled frames — a
    long clip is never fully downscaled, only the ~2*pairs frames flow needs.
    """
    import cv2

    frames = as_frames(frames)
    t, h, w = frames.shape[:3]
    starts = _pair_starts(t, pairs)
    size = _work_size(h, w, target_h)
    needed = sorted({k for k in starts} | {k + 1 for k in starts})
    gray = {i: _small_gray_frame(frames[i], size) for i in needed}
    flows = [
        cv2.calcOpticalFlowFarneback(gray[k], gray[k + 1], None, *_FARNEBACK)
        for k in starts
    ]
    return flows, gray, starts


def _magnitude(flow: Any) -> float:
    import numpy as np

    return float(np.hypot(flow[..., 0], flow[..., 1]).mean())


def warp_error_series(gray: Any, flows: Any, starts: list[int]) -> list[float]:
    """Per-pair reference-free temporal instability.

    Warp frame ``k`` toward ``k+1`` by the estimated flow and measure the mean
    absolute residual against the ACTUAL ``k+1``, normalized by that frame's
    contrast. Coherent motion warps cleanly (low residual); shimmer, boiling and
    per-frame noise cannot be predicted by any flow and leave a large residual —
    the reference-free signature of temporal instability, valid per clip
    regardless of same-seed trajectory re-roll.
    """
    import cv2
    import numpy as np

    h, w = gray[starts[0]].shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out: list[float] = []
    for flow, k in zip(flows, starts, strict=True):
        a = gray[k].astype(np.float32)
        b = gray[k + 1].astype(np.float32)
        warped = cv2.remap(a, gx + flow[..., 0], gy + flow[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out.append(float(np.abs(warped - b).mean() / max(float(b.std()), 1e-6)))
    return out


def warp_error(frames: Any, *, pairs: int = FLOW_PAIRS,
               target_h: int = FLOW_TARGET_H) -> float:
    """Reference-free mean temporal-instability (shimmer/boiling/flicker)."""
    import numpy as np

    flows, gray, starts = flow_fields(frames, pairs=pairs, target_h=target_h)
    series = warp_error_series(gray, flows, starts)
    return float(np.mean(series))


def flow_divergence(reference: Any, candidate: Any, *, pairs: int = FLOW_PAIRS,
                    target_h: int = FLOW_TARGET_H) -> dict[str, float]:
    """How far the candidate's MOTION FIELD diverges from the reference's.

    Computes flow for both arms at matched pair indices and returns the mean
    end-point error (EPE, working-resolution px) between the two flow fields,
    plus ``flow_divergence`` — that EPE normalized by the reference's own motion
    magnitude, so ``0`` is identical motion and ``1`` is a divergence the size of
    the reference's motion. The paired arms must be frame-aligned and the same
    length (:func:`as_frames` + the caller's alignment); a shape mismatch raises.

    Also reports each arm's mean motion magnitude, so a candidate that simply
    moves LESS (a smeared, under-moving lane) is distinguishable from one that
    moves DIFFERENTLY.
    """
    import numpy as np

    ref = as_frames(reference)
    cand = as_frames(candidate)
    _check_pair(ref, cand, metric="flow_divergence")
    fr, _gr, _starts = flow_fields(ref, pairs=pairs, target_h=target_h)
    fc, _, _ = flow_fields(cand, pairs=pairs, target_h=target_h)
    epe = float(np.mean([
        np.hypot(a[..., 0] - b[..., 0], a[..., 1] - b[..., 1]).mean()
        for a, b in zip(fr, fc, strict=True)
    ]))
    mag_ref = float(np.mean([_magnitude(f) for f in fr]))
    mag_cand = float(np.mean([_magnitude(f) for f in fc]))
    return {
        "flow_epe": epe,
        "flow_divergence": epe / max(mag_ref, 1e-6),
        "motion_mag_ref": mag_ref,
        "motion_mag_cand": mag_cand,
        "motion_mag_ratio": mag_cand / max(mag_ref, 1e-6),
    }


def temporal_fidelity(reference: Any, candidate: Any, *, pairs: int = FLOW_PAIRS,
                      target_h: int = FLOW_TARGET_H) -> dict[str, float]:
    """The paired temporal-fidelity block, flows computed once per arm.

    ``flow_divergence`` (motion-field EPE / reference motion), ``warp_error`` +
    ``warp_error_ref`` (each arm's reference-free instability) and
    ``warp_error_delta`` (candidate minus reference — the ADDED instability, a
    paired SIMILARITY number that cancels the shared scene the way #9's
    ``ringing_excess`` does).
    """
    import numpy as np

    ref = as_frames(reference)
    cand = as_frames(candidate)
    _check_pair(ref, cand, metric="temporal_fidelity")
    fr, gr, starts = flow_fields(ref, pairs=pairs, target_h=target_h)
    fc, gc, _ = flow_fields(cand, pairs=pairs, target_h=target_h)
    epe = float(np.mean([
        np.hypot(a[..., 0] - b[..., 0], a[..., 1] - b[..., 1]).mean()
        for a, b in zip(fr, fc, strict=True)
    ]))
    mag_ref = float(np.mean([_magnitude(f) for f in fr]))
    mag_cand = float(np.mean([_magnitude(f) for f in fc]))
    we_ref = float(np.mean(warp_error_series(gr, fr, starts)))
    we_cand = float(np.mean(warp_error_series(gc, fc, starts)))
    return {
        "flow_divergence": epe / max(mag_ref, 1e-6),
        "flow_epe": epe,
        "warp_error": we_cand,
        "warp_error_ref": we_ref,
        "warp_error_delta": we_cand - we_ref,
        "motion_mag_ref": mag_ref,
        "motion_mag_cand": mag_cand,
        "motion_mag_ratio": mag_cand / max(mag_ref, 1e-6),
    }


__all__ = [
    "FLOW_LIBRARY",
    "FLOW_PAIRS",
    "FLOW_TARGET_H",
    "SIGNAL_LIBRARY",
    "as_frames",
    "dframe_psnr_series",
    "dframe_ssim_series",
    "dframes",
    "flow_divergence",
    "flow_fields",
    "frame_images",
    "sample_indices",
    "signal_stats",
    "temporal_fidelity",
    "warp_error",
    "warp_error_series",
]
