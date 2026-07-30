"""Learned no-reference quality, aggregated over a clip.

The per-image scorers live in :mod:`cozy_eval.bench.metrics` — all of them
in-house (NIQE, MUSIQ) or Apache-2.0 (ARNIQA, CLIP-IQA via torchmetrics). This
module is the clip adapter: subsample frames, score each one, reduce to a mean
and a worst-case tail, and name the numbers so a protocol stamp can carry them.

There is no `pyiqa` here and there must never be again: it relicensed
Apache-2.0 -> PolyForm-Noncommercial at 0.1.16, which would have made every
consumer of this library non-commercial. See PROVENANCE.md.

The gate's calibrated thresholds are on the signal backend; a quality score
joins the report as an additional, uncalibrated observation until somebody banks
a clean/degraded population for it.
"""

from __future__ import annotations

import numpy as np

from ..bench.device import AUTO
from ..bench.registry import spec
from ..frames import iter_frames

#: Reference-free per-frame scorers, by registry name. `niqe` is closed form and
#: needs no weights; the rest download on first use and want a GPU to be quick.
FRAME_MODELS = ("niqe", "musiq", "arniqa", "clip_iqa")


def _scorer(model: str):
    if model not in FRAME_MODELS:
        raise ValueError(f"unknown frame model {model!r}; known: {list(FRAME_MODELS)}")
    from ..bench.metrics import iqa, musiq

    return getattr(musiq if model == "musiq" else iqa, model)


def available(model: str = "niqe") -> bool:
    """Whether `model`'s dependencies are importable. Weights are not checked —
    those download on first score."""
    try:
        _scorer(model)
    except ImportError:
        return False
    return True


def score_frames(source, *, model: str = "niqe", device: str = AUTO,
                 stride: int = 8) -> dict[str, float]:
    """Aggregate a per-frame no-reference score over a clip.

    ``stride`` subsamples frames: the learned models are two orders of magnitude
    more expensive than the signal backend and per-frame scores are highly
    correlated between neighbours.

    Returns ``{model}_mean`` and ``{model}_worst``. The tail is the 10th or 90th
    percentile depending on the metric's declared direction — a clip that is
    fine on average can still hold one ruined frame, and for NIQE (lower is
    better) that frame is at the TOP of the distribution.
    """
    score = _scorer(model)
    higher_is_better = spec(model).higher_is_better
    kwargs = {} if model == "niqe" else {"device": device}
    # iter_frames yields 0..1 float; every scorer here reads image-range data,
    # and all four are silently wrong (not loud) on a 0..1 input.
    vals = [float(score(np.rint(f * 255.0).astype(np.uint8), **kwargs))
            for i, f in enumerate(iter_frames(source)) if not i % stride]
    if not vals:
        raise ValueError("no frames scored")
    return {
        f"{model}_mean": float(np.mean(vals)),
        f"{model}_worst": float(np.percentile(vals, 10 if higher_is_better else 90)),
        f"{model}_frames": float(len(vals)),
    }
