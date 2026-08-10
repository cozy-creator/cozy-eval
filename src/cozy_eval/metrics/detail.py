"""Fine-detail fidelity: the numeric backend for the class the flat-region /
coherence / adjacent-frame screens MISS.

The failures Paul caught by eye on the composed H3 stack — melted faces,
PSEUDO-GLYPHS (text-like scribbles that are not real letters), edge HALOS /
ringing, fine-texture mush — are TEMPORALLY COHERENT. Adjacent-frame
correlation is normal on them; they are wrong in the detail of each frame, not
between frames. So none of the existing single-arm screens can see them.

WHAT SURVIVED VALIDATION, and what did NOT (ie#634 labeled crops + the
s20/s30/s50 same-seed step ladder, 2026-08-09):

* A bare reference-free high-frequency / sharpness number is BANNED, and the
  ban was re-confirmed here: on the ie#615 production-noise clip laplacian
  variance reads 592 vs 85 for a clean fal clip — the WORST clip scores the
  HIGHEST. ``edge_overshoot`` alone has the same disease: on the step ladder it
  RISES with steps (s50, the best arm, overshoots most) because genuine detail
  and ringing are the same signal reference-free.

* ``ringing_excess`` — candidate overshoot MINUS the reference's, at matched
  content — is the form that works, because subtracting a same-seed control
  cancels the scene. On the ie#634 crops (fal | ours, matched 2x zoom) the
  composed arm overshoots MORE on both (busker +0.045, signage +0.072). That is
  a paired SIMILARITY number, calibrated like every other paired number here.

* ``text_legibility`` (OCR confidence over detected text regions) is a real but
  NARROW detector: it flags a region OCR believes is text but cannot read
  confidently. Its blind spot is large and honest — on the ie#634 stylized
  neon-Japanese signage, where the pseudo-glyphs actually live, the OCR detector
  finds ZERO regions and the metric is UNMEASURED, not a pass. For non-Latin /
  heavily stylized signage the VLM rubric in :mod:`cozy_eval.detail` is the only
  instrument that catches the pseudo-glyph.

Everything here is numpy + the optional OCR reader + (for the paired texture
distance) torchmetrics DISTS. Nothing loads a VLM — that is the primary catch
and lives one layer up, exactly as the audio semantic tier sits above the audio
signal stats.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..frames import LUMA, iter_frames

DETAIL_LIBRARY = "cozy-eval:detail"
DISTS_MODEL = "torchmetrics:dists"

#: Fraction of the strongest-gradient pixels treated as an edge for the ringing
#: read. 0.05 = the top 5% of |grad|; a halo is the overshoot in the thin band
#: JUST BESIDE those edges, not on them.
EDGE_QUANTILE = 0.95
HALO_DILATION = 2

#: OCR confidence at/above which a recognized text region counts as legible.
#: rapidocr PP-OCR scores are per-line recognition confidence in [0, 1]; real
#: signage on the fal clips sits ~0.58-0.66, pseudo-glyph mush that still gets
#: recognized falls well under. Report-only: this is a pre-screen threshold, not
#: a shipped gate.
LEGIBLE_CONF = 0.60


def _luma(frame: Any) -> np.ndarray:
    a = np.asarray(frame, np.float32)
    if a.ndim == 3 and a.shape[-1] == 3:
        if a.max(initial=0.0) > 1.5:
            a = a / 255.0
        return a @ LUMA
    if a.max(initial=0.0) > 1.5:
        a = a / 255.0
    return a


def edge_overshoot(frame: Any) -> float:
    """Reference-free ringing DIAGNOSTIC: mean |unsharp residual| in the thin band
    beside strong edges, normalized by frame contrast.

    Ringing / halos put a bright-then-dark rim right beside high-contrast edges;
    this measures the energy in that rim. SCENE-CONFOUNDED on its own — genuine
    fine detail raises it too — so it is report-only and only trustworthy as the
    paired :func:`ringing_excess`. Kept because it is the quantity that
    difference makes meaningful.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    lum = _luma(frame)
    blur = gaussian_filter(lum, 1.2)
    resid = lum - blur
    gy, gx = np.gradient(blur)
    grad = np.hypot(gx, gy)
    edge = grad >= np.quantile(grad, EDGE_QUANTILE)
    band = binary_dilation(edge, iterations=HALO_DILATION) & ~edge
    if not band.any():
        return 0.0
    return float(np.abs(resid[band]).mean() / max(float(lum.std()), 1e-6))


def ringing_excess(reference: Any, candidate: Any) -> float:
    """Candidate edge-overshoot MINUS the reference's, at matched content.

    The validated ringing signal: same seed / same scene cancels, so a positive
    value is halo the candidate ADDED. Frames are compared one-to-one; a shape
    mismatch is the caller's to resolve (align first).
    """
    return edge_overshoot(candidate) - edge_overshoot(reference)


# ---------------------------------------------------------------------------
# text legibility — the OCR pre-screen for pseudo-glyphs
# ---------------------------------------------------------------------------

class TextLegibility:
    """One frame's text-legibility read. ``measured`` is False when the OCR
    reader found NO text-like region at all — which is UNMEASURED (there is
    nothing to judge), never a pass and never a fail."""

    __slots__ = ("mean_conf", "measured", "n_legible", "n_regions", "note")

    def __init__(self, *, measured: bool = False, n_regions: int = 0,
                 n_legible: int = 0, mean_conf: float = float("nan"), note: str = "") -> None:
        self.measured = measured
        self.n_regions = n_regions
        self.n_legible = n_legible
        self.mean_conf = mean_conf
        self.note = note


def text_legibility(image: Any, *, min_conf: float = LEGIBLE_CONF) -> TextLegibility:
    """Score whether the text regions of one frame hold READABLE glyphs.

    Uses the same OCR reader the adherence lane uses. A text detector that fires
    (a region looks like text) but whose recognizer confidence is low is the
    numeric signature of a pseudo-glyph. ``mean_conf`` is the mean per-region
    recognition confidence; ``n_legible`` counts regions at/above ``min_conf``.

    BLIND SPOT, stated: the PP-OCR detector does not fire on heavily stylized or
    non-Latin signage (the ie#634 neon-Japanese case), so exactly the hardest
    pseudo-glyphs come back UNMEASURED here and must be caught by the VLM rubric.
    """
    from . import ocr as ocr_mod

    if not ocr_mod.available():
        return TextLegibility(note="no OCR reader: pip install 'cozy-eval[ocr]'")
    page = ocr_mod.read(image)
    if not page.measured:
        return TextLegibility(note=f"OCR failed: {page.note}")
    confs = [float(c) for c in page.confidences]
    if not confs:
        return TextLegibility(
            note="no text region detected (UNMEASURED, not a pass — the OCR "
            "detector does not fire on stylized/non-Latin signage)"
        )
    return TextLegibility(
        measured=True,
        n_regions=len(confs),
        n_legible=sum(1 for c in confs if c >= min_conf),
        mean_conf=float(np.mean(confs)),
    )


def clip_text_legibility(strip: list[Any], *, min_conf: float = LEGIBLE_CONF) -> TextLegibility:
    """Aggregate legibility over a frame strip. Regions and legible counts sum;
    ``mean_conf`` is the confidence mean over all regions across frames. A clip
    with no detected text on ANY frame is UNMEASURED."""
    regions: list[float] = []
    legible = 0
    for img in strip:
        one = text_legibility(img, min_conf=min_conf)
        if one.measured:
            legible += one.n_legible
            # recover per-region confidences by re-reading would double-cost; we
            # already have counts+mean per frame, so weight the frame mean by its
            # region count to get a faithful pooled mean.
            regions.extend([one.mean_conf] * one.n_regions)
    if not regions:
        return TextLegibility(
            note="no text region detected on any sampled frame (UNMEASURED)"
        )
    return TextLegibility(
        measured=True, n_regions=len(regions), n_legible=legible,
        mean_conf=float(np.mean(regions)),
    )


# ---------------------------------------------------------------------------
# paired texture distance — DISTS, the registry's noted non-redundant gap
# ---------------------------------------------------------------------------

_DISTS_CACHE: dict[str, Any] = {}


def dists_model(device: str = "cpu") -> Any:
    """The torchmetrics DISTS model, cached. Texture-and-structure perceptual
    distance — the one similarity metric the registry flags as plausibly NOT
    redundant with LPIPS, and the one that reads 'real texture vs mush' where
    LPIPS reads global divergence."""
    key = f"dists:{device}"
    if key not in _DISTS_CACHE:
        import torch
        from torchmetrics.image import DeepImageStructureAndTextureSimilarity as _D

        m = _D().to(device)
        m.eval()
        _DISTS_CACHE[key] = (m, torch)
    return _DISTS_CACHE[key]


def dists_pair(reference: Any, candidate: Any, device: str = "cpu") -> float:
    """DISTS between two ``(H, W, 3)`` frames in [0, 1] (or uint8). Lower = closer."""
    model, torch = dists_model(device)
    r = _to_chw(reference, torch, device)
    c = _to_chw(candidate, torch, device)
    with torch.no_grad():
        return float(model(c, r))


def _to_chw(frame: Any, torch: Any, device: str) -> Any:
    a = np.asarray(frame, np.float32)
    if a.max(initial=0.0) > 1.5:
        a = a / 255.0
    return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().to(device)


def free_models() -> None:
    _DISTS_CACHE.clear()


# ---------------------------------------------------------------------------
# a whole-clip numeric snapshot, for the report
# ---------------------------------------------------------------------------

def detail_signal_stats(source: Any) -> dict[str, float]:
    """Reference-free numeric detail signals over a clip, for the report.

    Only ``edge_overshoot`` (the scene-confounded diagnostic) — text legibility
    needs its own UNMEASURED handling and is scored on the judge strip, not
    here. Streams frames so a long clip stays O(1) in memory.
    """
    vals: list[float] = []
    for rgb in iter_frames(source):
        vals.append(edge_overshoot(rgb))
    if not vals:
        return {}
    return {"edge_overshoot": float(np.mean(vals))}


__all__ = [
    "DETAIL_LIBRARY",
    "DISTS_MODEL",
    "LEGIBLE_CONF",
    "TextLegibility",
    "clip_text_legibility",
    "detail_signal_stats",
    "dists_model",
    "dists_pair",
    "edge_overshoot",
    "free_models",
    "ringing_excess",
    "text_legibility",
]
