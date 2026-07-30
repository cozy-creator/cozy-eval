"""Measurement backends.

The library's opinion is about *protocol*, not about who computes the numbers.
Backends are therefore pluggable and, wherever a maintained package already owns
a metric, we call it rather than reimplement it:

* :mod:`.signal` — the always-available core. Elementary signal statistics
  (Laplacian variance, spectral high-frequency ratio, histogram entropy, first-
  and second-order temporal differences). These are primitives, not somebody's
  library; the core keeps them so the base install is numpy + ffmpeg.
* :mod:`.reference` — reference metrics for the same-trajectory lane. Uses
  ``libvmaf`` through ffmpeg when present and scikit-image's SSIM when present,
  falling back to the built-in windowed SSIM.
* :mod:`.perceptual` — ``pip install cozy-eval[perceptual]``. Wraps
  ``pyiqa`` (MUSIQ, CLIP-IQA, MANIQA, NIQE, BRISQUE) and, for video, DOVER-class
  scorers. Never reimplemented here.
* :mod:`.distributional` — ``pip install cozy-eval[distributional]``.
  Deep-feature Fréchet distances (I3D-FVD, V-JEPA/JEDi). Opt-in and explicitly
  NOT validated at the sample sizes this library's gate runs at; see GATE.md.
"""

from . import reference, signal

__all__ = ["reference", "signal"]
