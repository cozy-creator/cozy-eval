"""Every metric this library computes, one module per kind of measurement.

Which of these we wrote and which we compose from somebody else's package is a
licence fact recorded in PROVENANCE.md — not an axis of this package. A metric
belongs where it belongs by what it measures.

  similarity      LPIPS / SSIM / MS-SSIM / PSNR / CLIPScore between two renders
  reference       the same question for whole video FILES — libvmaf, ColorVideoVDP,
                  and torch-free PSNR/SSIM, so the reference lane runs on a bare
                  install
  adherence       authored-checklist scoring + VLM-judge orchestration
  geneval         compositional adherence, detector-scored
  ocr             the reader behind quoted-string checklist items
  vqascore        soft p(yes) alignment and Soft-TIFA aggregation
  preference      PickScore and alternates
  hpsv3           HPSv3 preference scoring
  quality         reference-free technical quality: NIQE, ARNIQA, CLIP-IQA, plus
                  the per-clip aggregator
  musiq           MUSIQ, our PyTorch port of the Apache reference
  signal          closed-form clip statistics (detail, tone, flicker, jerk);
                  numpy only, and what the gate's budgets are calibrated on
  temporal        the Δ-frame channel — dynamics no per-frame metric can see
  distributional  deep-feature Fréchet distances over a population

STABILITY: experimental (v0.x). The metric NAMES these modules produce are
locked — they are declared as data in :mod:`cozy_eval.registry` and consumers
key budgets and banked reports on them. The functions and classes here are not:
signatures, backends and internals may change in any 0.x release.

``signal`` and ``reference`` are numpy-only and always available; every other
module imports torch lazily, so a bare install stays torch-free.
"""

from __future__ import annotations

from . import reference, signal

__all__ = ["reference", "signal"]
