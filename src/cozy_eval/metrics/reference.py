"""Reference metrics for the same-trajectory lane.

Composed, not reimplemented, wherever a maintained implementation exists:

* **VMAF** via ``ffmpeg_quality_metrics`` (MIT, actively maintained) when it is
  installed, falling back to a direct ``ffmpeg -lavfi libvmaf`` call. Netflix's
  model is the right instrument for post-decode changes and there is no reason to
  have an opinion of our own about it.
* **ColorVideoVDP** via ``cvvdp`` (MIT) when installed — a colour- and HDR-aware
  spatiotemporal full-reference video metric, strictly a better instrument than
  VMAF for this lane. Optional because it is heavier.
* **SSIM** via ``skimage.metrics.structural_similarity`` when scikit-image is
  installed; otherwise a built-in 8x8 block-windowed SSIM (global SSIM is far too
  insensitive on video — it stays at 0.998 through a visibly blocky re-encode).
* **LPIPS** via the ``lpips`` package when installed.
* **PSNR** is one line of numpy and has no library worth depending on.

Nothing in this module may be called without passing the
:func:`~cozy_eval.protocol.require_reference_lane` guard first; the gate
does that for you.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..frames import LUMA, iter_frames


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def _ssim_blocks(a: np.ndarray, b: np.ndarray, block: int = 8) -> float:
    la, lb = a @ LUMA, b @ LUMA
    h, w = la.shape
    h, w = h - h % block, w - w % block

    def tiles(x):
        x = x[:h, :w].reshape(h // block, block, w // block, block)
        return x.transpose(0, 2, 1, 3).reshape(-1, block * block)

    ta, tb = tiles(la), tiles(lb)
    m1, m2 = ta.mean(1), tb.mean(1)
    v1, v2 = ta.var(1), tb.var(1)
    cov = ((ta - m1[:, None]) * (tb - m2[:, None])).mean(1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * m1 * m2 + c1) * (2 * cov + c2)) / ((m1 ** 2 + m2 ** 2 + c1) * (v1 + v2 + c2))
    return float(s.mean())


@functools.cache
def _have_skimage() -> bool:
    try:
        import skimage.metrics  # noqa: F401
        return True
    except ImportError:
        return False


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if _have_skimage():
        from skimage.metrics import structural_similarity
        return float(structural_similarity(a @ LUMA, b @ LUMA, data_range=1.0))
    return _ssim_blocks(a, b)


@functools.cache
def have_vmaf() -> bool:
    """Is the local ffmpeg built with libvmaf?"""
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                         capture_output=True, text=True, check=False).stdout
    return " libvmaf " in out


def _harmonic_mean(vals: "list[float]") -> float:
    """Harmonic mean of per-frame VMAF, the pooling Netflix recommends: it
    weights the worst frames far more than the arithmetic mean, so a clip that
    is excellent for 90% and falls apart for 10% does not read as ~90."""
    finite = [v for v in vals if v is not None and v > 0]
    if not finite:
        return 0.0
    return float(len(finite) / np.sum([1.0 / v for v in finite]))


def _ffmpeg_bin() -> str:
    """The ffmpeg to shell out to. ``VMAF_FFMPEG`` overrides so a box whose
    system ffmpeg lacks libvmaf can point at a static build."""
    import os

    return os.environ.get("VMAF_FFMPEG", "ffmpeg")


def _probe_video(path: str | Path) -> tuple[int, int, str]:
    """``(width, height, fps_expr)`` of a clip's first video stream, via ffprobe."""
    import os

    ffmpeg = _ffmpeg_bin()
    d, base = os.path.split(ffmpeg)
    ffprobe = os.path.join(d, base.replace("ffmpeg", "ffprobe")) if d else "ffprobe"
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    w, h, fps = out.split(",")[:3]
    return int(w), int(h), fps


def vmaf(reference: str | Path, candidate: str | Path) -> dict[str, float]:
    """Netflix VMAF, candidate vs reference, harmonic-mean pooled.

    Resolution and framerate are ALIGNED before scoring: VMAF is a full-reference
    metric defined at one geometry and clock, so the candidate is scaled to the
    reference's resolution and resampled to its framerate in the filter graph.
    An unaligned pair otherwise scores a real fidelity loss where there was only
    a preset difference. Returns ``vmaf`` (harmonic mean — THE number),
    ``vmaf_mean`` (arithmetic, for continuity with historical rows) and
    ``vmaf_min`` (the worst frame, the tail).
    """
    ffmpeg = _ffmpeg_bin()
    rw, rh, rfps = _probe_video(reference)
    # [candidate] scaled+fps-matched to the reference, then [that][reference]libvmaf.
    align = f"scale={rw}:{rh}:flags=bicubic,fps={rfps},setsar=1"
    with tempfile.NamedTemporaryFile("r", suffix=".json") as out:
        graph = (
            f"[0:v]{align}[cand];"
            f"[1:v]setsar=1[ref];"
            f"[cand][ref]libvmaf=log_fmt=json:log_path={out.name}"
        )
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(candidate), "-i", str(reference),
             "-lavfi", graph, "-f", "null", "-"],
            check=False, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "libvmaf ffmpeg call failed; install an ffmpeg built with "
                "--enable-libvmaf or set VMAF_FFMPEG to one. stderr:\n"
                + (proc.stderr or "")[-800:]
            )
        data = json.loads(Path(out.name).read_text())
    pooled = data["pooled_metrics"]["vmaf"]
    per_frame = [f["metrics"]["vmaf"] for f in data.get("frames", [])]
    hmean = _harmonic_mean(per_frame) if per_frame else float(pooled.get("harmonic_mean", pooled["mean"]))
    return {
        "vmaf": hmean,
        "vmaf_mean": float(pooled["mean"]),
        "vmaf_min": float(pooled["min"]),
    }


@functools.cache
def _lpips_model():
    import lpips as _l
    return _l.LPIPS(net="alex")


def lpips_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Per-frame LPIPS via the ``lpips`` package (optional dependency)."""
    import torch
    model = _lpips_model()
    def t(x):
        return torch.from_numpy(x.transpose(2, 0, 1)[None] * 2.0 - 1.0).float()
    with torch.no_grad():
        return float(model(t(a), t(b)).item())


def cvvdp(reference: str | Path, candidate: str | Path,
          display: str = "standard_4k") -> dict[str, float]:
    """ColorVideoVDP (JOD units, higher is better). Needs ``[reference]`` extra."""
    try:
        import pycvvdp
    except ImportError as exc:                                 # pragma: no cover
        raise RuntimeError(
            "cvvdp not installed: pip install 'cozy-eval[reference]'"
        ) from exc
    metric = pycvvdp.cvvdp(display_name=display)
    jod, _ = metric.predict(str(candidate), str(reference), dim_order="FHWC")
    return {"cvvdp_jod": float(jod)}


def compare(reference, candidate, *, with_lpips: bool = False) -> dict[str, float]:
    """Per-frame reference metrics between two frame sources of the same take."""
    ps: list[float] = []
    ss: list[float] = []
    lp: list[float] = []
    n = 0
    for a, b in zip(iter_frames(reference), iter_frames(candidate), strict=True):
        if a.shape != b.shape:
            raise ValueError(
                f"frame shape mismatch {a.shape} vs {b.shape}: a resize is itself a "
                "change, declare ChangeKind.OUTPUT_RESIZE and compare at one size"
            )
        n += 1
        ps.append(psnr(a, b))
        ss.append(ssim(a, b))
        if with_lpips:
            lp.append(lpips_pair(a, b))
    if not n:
        raise ValueError("no overlapping frames")
    out = {
        "frames": float(n),
        "psnr_mean": float(np.mean(ps)),
        "psnr_worst": float(np.min(ps)),
        "ssim_mean": float(np.mean(ss)),
        "ssim_worst": float(np.min(ss)),
    }
    if lp:
        out["lpips_mean"] = float(np.mean(lp))
        out["lpips_worst"] = float(np.max(lp))
    return out
