"""Scoring backends, one module per lane.

STABILITY: experimental (v0.x). The metric NAMES these modules produce are
locked — they are declared as data in :mod:`cozy_eval.bench.registry` and consumers
key budgets and banked reports on them. The functions and classes here are not:
signatures, backends and internals may change in any 0.x release.

Nothing in this package is imported by ``import cozy_eval.bench``, and no module here
imports torch at module scope, so a bare install stays torch-free.

  adherence   authored-checklist scoring + judge orchestration (checklist
              FORMAT and loader are locked core, the judge helpers are not)
  similarity  LPIPS / SSIM / MS-SSIM / PSNR via torchmetrics
  preference  PickScore and alternates
  ocr         rapidocr reader for quoted-string checklist items
  geneval     compositional adherence, detector-scored
  iqa         reference-free technical quality (NIQE, ARNIQA, CLIP-IQA)
  musiq       MUSIQ, our PyTorch port of the Apache reference
  vqascore    soft p(yes) alignment and Soft-TIFA aggregation
  hpsv3       HPSv3 preference scoring
"""

from __future__ import annotations

__all__: list[str] = []
