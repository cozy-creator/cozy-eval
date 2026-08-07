"""AV-sync: does the sound land when the picture says it should.

STABILITY: experimental (v0.x). The metric NAMES (``av_sync_offset_ms``,
``av_sync_confidence``, ``av_sync_drift_ms``) are locked by the registry.

THE AXIS THAT ONLY EXISTS FOR JOINTLY-GENERATED AUDIO+VIDEO, and the one most
likely to break silently under acceleration: a cache sweep, a step-distill LoRA
or a sparse-attention lane can shift the audio stream against the picture
without moving a single per-frame or per-sample statistic. Every other number in
this library would stay green.

WHAT THIS IMPLEMENTS, AND THE GAP IT DOES NOT CLOSE
---------------------------------------------------

Lip-sync on dialogue and onset-alignment on non-speech events are DIFFERENT
sub-problems and are not collapsed here.

* **Non-speech event sync — implemented.** An audio onset envelope (half-wave
  rectified spectral flux; Bello et al., "A Tutorial on Onset Detection in Music
  Signals", IEEE TSAP 13(5), 2005) is cross-correlated against a VISUAL onset
  envelope built the same way from frame-difference energy. Correlating an
  acoustic envelope against a visual-change envelope to recover audio-visual
  correspondence is the classical approach — Hershey & Movellan, "Audio-Vision:
  Using Audio-Visual Synchrony to Locate Sounds", NIPS 1999; Slaney & Covell,
  "FaceSync: A Linear Operator for Measuring Synchronization of Video Facial
  Images and Audio Tracks", NIPS 2000. Closed form, NO trained weights, no
  licence exposure, CPU-cheap, and it runs today.

* **Lip-sync on dialogue — an honest gap, not a silent zero.** The published
  instruments are SyncNet-lineage (Chung & Zisserman, "Out of time: automated
  lip sync in the wild", ACCV 2016) and its successors; LSE-D / LSE-C as
  reported in the lip-sync literature are SyncNet outputs. Their weights are the
  method, and none of the widely-used checkpoints has a licence we can build a
  commercial gate on. :func:`sync_offset` therefore measures event sync and says
  so; it never reports a lip-sync number it cannot defend. The tracked candidate
  is Synchformer (Iashin et al., ICASSP 2024), pending a weights-licence check.

**Measurement honesty.** Onset correlation genuinely CANNOT measure a static
shot over ambient pad music: there are no shared events to align. When the
correlation curve has no prominent peak this returns ``measured=False`` with the
reason, rather than an offset read off noise. That is the whole point — a
confident 0 ms on unmeasurable content is worse than an admitted gap.

**Paired arms.** Absolute offset is confounded by content; the regression signal
is :func:`sync_drift` — candidate offset minus reference offset. Both travel in
the report.
"""

from __future__ import annotations

from typing import Any

import msgspec
import numpy as np

from ..errors import ConfigError
from . import audio as audio_metrics

#: Envelope grid, in seconds. 5 ms resolves sync far finer than one frame
#: (41.7 ms at 24 fps), which matters because a one-frame slip is already
#: audible on a hard transient.
HOP_SECONDS = 0.005

#: STFT window for the audio onset envelope.
N_FFT = 1024

#: Lags searched, in seconds. Beyond half a second the streams are not
#: mis-synced, they are unrelated.
MAX_LAG_SECONDS = 0.5

#: A correlation peak must stand this many standard deviations above the mean of
#: the lag curve to count as a peak at all. A flat curve means the content
#: carries no shared onsets, which is UNMEASURED, not 0 ms.
MIN_PROMINENCE = 3.0

#: ...and the peak itself must reach this Pearson r. Prominence alone can be
#: satisfied by a tiny bump on a very flat curve.
MIN_PEAK_R = 0.15


class SyncEstimate(msgspec.Struct, kw_only=True):
    """One clip's AV-sync read. ``measured=False`` carries its reason in ``note``."""

    measured: bool
    offset_ms: float = 0.0
    confidence: float = 0.0     # Pearson r at the winning lag
    prominence: float = 0.0     # (peak - mean) / sd of the lag curve
    note: str = ""


def onset_envelope(source: Any, sample_rate: int = 0, *,
                   hop_seconds: float = HOP_SECONDS, n_fft: int = N_FFT) -> Any:
    """Half-wave-rectified spectral flux of the programme signal.

    Bello et al. 2005. Rectification is what makes it an ONSET detector rather
    than a change detector: energy appearing counts, energy decaying does not.
    """
    clip = audio_metrics.as_audio(source, sample_rate)
    hop = max(1, round(hop_seconds * clip.sample_rate))
    spec = audio_metrics._stft_magnitude(clip.mono, n_fft=n_fft, hop=hop)
    flux = np.diff(spec, axis=1)
    return np.sum(np.maximum(flux, 0.0), axis=0)


def motion_envelope(frames: Any, fps: float, *,
                    hop_seconds: float = HOP_SECONDS,
                    length: int = 0, n_fft: int = N_FFT,
                    sample_rate: int = 0) -> Any:
    """Visual onset envelope on the audio envelope's grid.

    Frame-difference energy is the motion signal (the same Δ-frame channel
    :mod:`cozy_eval.metrics.temporal` already uses for flicker); differencing it
    again and half-wave rectifying gives a visual ONSET envelope — an impact,
    a cut or a mouth opening is a rise in motion energy, symmetric with the
    audio side.
    """
    if fps <= 0:
        raise ConfigError("motion_envelope: fps must be positive to place frames in time")
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ConfigError(
            f"motion_envelope: expected (T, H, W, 3) frames, got shape {frames.shape}"
        )
    energy = np.mean(np.abs(np.diff(frames.astype(np.float32), axis=0)), axis=(1, 2, 3))
    onsets = np.maximum(np.diff(energy), 0.0)
    if onsets.size == 0:
        return np.zeros(max(length, 1))
    # TIMESTAMPING, and it is worth a full frame of accuracy. energy[j] is
    # |f[j+1] - f[j]|: the change that made frame j+1 differ, so it belongs at
    # frame j+1's time, (j+1)/fps — NOT the interval midpoint. onsets[i] is the
    # RISE from energy[i] to energy[i+1], so it belongs at (i+2)/fps. Getting
    # this wrong costs exactly one frame (41.7 ms at 24 fps) of constant bias,
    # measured against the click-and-flash positive control in tests/.
    times = (np.arange(onsets.size) + 2.0) / float(fps)
    # The audio envelope's k-th value is the flux INTO STFT frame k+1.
    grid = (np.arange(length) + 1.0) * hop_seconds + (n_fft / 2.0) / float(
        sample_rate or round(1.0 / hop_seconds)
    )
    return np.interp(grid, times, onsets, left=0.0, right=0.0)


def _normalize(x: Any) -> Any:
    x = np.asarray(x, np.float64)
    sd = float(np.std(x))
    return np.zeros_like(x) if sd <= 0 else (x - float(np.mean(x))) / sd


def sync_offset(frames: Any, source: Any, sample_rate: int = 0, *, fps: float,
                hop_seconds: float = HOP_SECONDS,
                max_lag_seconds: float = MAX_LAG_SECONDS) -> SyncEstimate:
    """Audio-visual offset by onset cross-correlation.

    Positive ``offset_ms`` means the AUDIO IS LATE — the sound arrives after the
    picture event it belongs to.
    """
    clip = audio_metrics.as_audio(source, sample_rate)
    a_env = onset_envelope(clip, hop_seconds=hop_seconds)
    v_env = motion_envelope(
        frames, fps, hop_seconds=hop_seconds, length=a_env.size,
        sample_rate=clip.sample_rate,
    )
    n = min(a_env.size, v_env.size)
    if n < 16:
        return SyncEstimate(measured=False, note="clip too short for an onset correlation")
    a, v = _normalize(a_env[:n]), _normalize(v_env[:n])
    if not a.any():
        return SyncEstimate(
            measured=False,
            note="AV-sync UNMEASURED: the audio carries no onsets (silence or a "
                 "steady tone) — nothing to align against",
        )
    if not v.any():
        return SyncEstimate(
            measured=False,
            note="AV-sync UNMEASURED: the video carries no motion onsets (a "
                 "static shot) — nothing to align against",
        )
    limit = min(n - 8, round(max_lag_seconds / hop_seconds))
    if limit < 1:
        return SyncEstimate(measured=False, note="clip too short for the lag search")

    lags = np.arange(-limit, limit + 1)
    curve = np.empty(lags.size)
    for i, lag in enumerate(lags):
        # positive lag => audio late => audio[t + lag] aligns with video[t]
        x, y = (a[lag:], v[:n - lag]) if lag >= 0 else (a[:n + lag], v[-lag:])
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        curve[i] = float(np.dot(x, y)) / denominator if denominator > 0 else 0.0

    best = int(np.argmax(curve))
    peak = float(curve[best])
    sd = float(np.std(curve))
    prominence = 0.0 if sd <= 0 else (peak - float(np.mean(curve))) / sd
    offset_ms = float(lags[best]) * hop_seconds * 1000.0
    if prominence < MIN_PROMINENCE or peak < MIN_PEAK_R:
        return SyncEstimate(
            measured=False, offset_ms=offset_ms, confidence=peak, prominence=prominence,
            note=(
                f"AV-sync UNMEASURED: no prominent correlation peak (r={peak:.3f}, "
                f"prominence={prominence:.1f} sd, needs r>={MIN_PEAK_R} and "
                f">={MIN_PROMINENCE} sd) — this content has no shared audio/visual "
                "onsets to align, so an offset here would be read off noise"
            ),
        )
    return SyncEstimate(
        measured=True, offset_ms=offset_ms, confidence=peak, prominence=prominence,
    )


def sync_drift(reference: SyncEstimate, candidate: SyncEstimate) -> float | None:
    """Candidate offset minus reference offset, ms. ``None`` if either arm is
    UNMEASURED — a drift against an unknown baseline is not a number."""
    if not (reference.measured and candidate.measured):
        return None
    return candidate.offset_ms - reference.offset_ms


def shift_audio(source: Any, sample_rate: int = 0, *, milliseconds: float) -> Any:
    """Delay (or advance) audio by ``milliseconds``, zero-filled. The RED-ARM
    constructor: this is how a test builds a deliberately desynced soundtrack."""
    clip = audio_metrics.as_audio(source, sample_rate)
    n = round(abs(milliseconds) / 1000.0 * clip.sample_rate)
    if n == 0:
        return clip
    pad = np.zeros((n, clip.channels), np.float32)
    samples = (
        np.concatenate([pad, clip.samples[:-n]]) if milliseconds > 0
        else np.concatenate([clip.samples[n:], pad])
    )
    return audio_metrics.Audio(samples=samples, sample_rate=clip.sample_rate)


AVSYNC_METHOD = "cozy-eval:onset-correlation"

#: Recorded so the gap travels WITH the number rather than in a doc nobody reads.
LIPSYNC_GAP_NOTE = (
    "lip-sync (SyncNet-lineage LSE-D/LSE-C) NOT measured: the published "
    "checkpoints carry no licence we can gate a commercial product on. The "
    "number reported here is EVENT sync by onset correlation, which does not "
    "score dialogue. Tracked candidate: Synchformer (Iashin et al., ICASSP 2024)."
)


__all__ = [
    "AVSYNC_METHOD",
    "HOP_SECONDS",
    "LIPSYNC_GAP_NOTE",
    "MAX_LAG_SECONDS",
    "MIN_PEAK_R",
    "MIN_PROMINENCE",
    "N_FFT",
    "SyncEstimate",
    "motion_envelope",
    "onset_envelope",
    "shift_audio",
    "sync_drift",
    "sync_offset",
]
