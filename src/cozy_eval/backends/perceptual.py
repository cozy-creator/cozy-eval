"""Optional learned no-reference backends — ``pip install cozy-eval[perceptual]``.

We do not implement MUSIQ, CLIP-IQA, MANIQA, NIQE or BRISQUE; ``pyiqa`` already
maintains all of them behind one factory, so this module is a ~40-line adapter
that (a) makes them frame sources like every other backend, (b) aggregates a
per-frame score into a clip score, and (c) records which model and weights were
used inside the protocol stamp — an unstamped MUSIQ number is as unusable as an
unstamped LPIPS number.

**Licence, verified 2026-07-26.** ``pyiqa`` relicensed Apache-2.0 ->
PolyForm-Noncommercial-1.0.0 at **0.1.16** (released 2026-07-08). The extra pins
``<0.1.16`` to stay on the Apache-2.0 line; raising that ceiling is a licensing
decision, not a version bump. DOVER and FAST-VQA are deliberately absent: both
are S-Lab-1.0 (non-commercial) despite ``setup.py`` files that still declare
MIT/Apache.

These models download weights on first use and want a GPU to be quick, which is
why they are an extra rather than the core. The gate's calibrated thresholds are
on the signal backend; a perceptual score joins the report as an additional,
uncalibrated observation until somebody banks a clean/degraded population for it.
"""

from __future__ import annotations

import functools

import numpy as np

from ..frames import iter_frames

#: Models known to work as a per-frame no-reference score through pyiqa.
FRAME_MODELS = ("musiq", "clipiqa", "maniqa", "niqe", "brisque", "topiq_nr")


def available() -> bool:
    try:
        import pyiqa  # noqa: F401
        return True
    except ImportError:
        return False


@functools.cache
def _metric(name: str, device: str):
    import pyiqa
    return pyiqa.create_metric(name, device=device)


def score_frames(source, *, model: str = "musiq", device: str = "cpu",
                 stride: int = 8) -> dict[str, float]:
    """Aggregate a pyiqa per-frame no-reference score over a clip.

    ``stride`` subsamples frames: these models are two orders of magnitude more
    expensive than the signal backend and per-frame scores are highly correlated
    between neighbours.
    """
    if not available():                                        # pragma: no cover
        raise RuntimeError(
            "perceptual backend needs pyiqa: pip install 'cozy-eval[perceptual]'"
        )
    import torch
    m = _metric(model, device)
    vals: list[float] = []
    for i, f in enumerate(iter_frames(source)):
        if i % stride:
            continue
        t = torch.from_numpy(f.transpose(2, 0, 1)[None]).float().to(device)
        with torch.no_grad():
            vals.append(float(m(t).item()))
    if not vals:
        raise ValueError("no frames scored")
    return {
        f"{model}_mean": float(np.mean(vals)),
        f"{model}_p10": float(np.percentile(vals, 10)),
        f"{model}_frames": float(len(vals)),
    }
