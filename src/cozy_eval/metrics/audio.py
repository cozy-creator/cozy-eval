"""Audio: signal statistics, loudness, and paired fidelity to a reference arm.

STABILITY: experimental (v0.x) for these function signatures; the metric NAMES
they produce are locked by the registry.

WHY THIS EXISTS. Every video model we serve now emits audio — LTX-2.3 denoises
audio latents in the same loop and muxes AAC, MiniMax-H3 generates 32 kHz stereo
jointly with the picture. Until this module there was no ``audio`` anywhere in
the library, so a quant lane, a cache sweep or a step-distill LoRA could destroy
the soundtrack while every number in the harness stayed green. That is not
hypothetical: a cache sweep drove audio SNR 20.67 -> 13.72 dB on an arm whose
video SSIM still read 0.85.

AUDIO CONVENTION, the companion to ``(T, H, W, 3)`` frames: ``(N, C)`` float32
in [-1, 1] — N samples, C channels, channel-last for the same reason frames are
channel-last. A mono ``(N,)`` array is accepted and promoted. Nothing here
decodes files: :func:`cozy_eval.audio.read_audio` does that, behind the same
optional-ffmpeg stance :mod:`cozy_eval.frames` already takes.

THREE TIERS, and they answer different questions:

* **reference-free** (:func:`signal_stats`) — level, loudness, clipping,
  silence, DC, spectral flatness, and the STEREO pair. Runs on ANY arm with no
  reference at all, which is what makes it the floor of the gate: a dual-mono
  regression from a model that claims stereo is a real defect and a trivial one
  to catch. Every number here is content-INDEPENDENT enough to carry an absolute
  budget, unlike the paired numbers below.
* **paired** (:func:`paired_stats`) — SI-SDR, SNR, log-spectral distance and
  mel L1 against the bf16 anchor, i.e. the faithfulness question the standing
  quant-verdict policy makes the gold standard.
* AV-sync and the semantic (transcribe/judge) tier live in
  :mod:`cozy_eval.metrics.avsync` and :mod:`cozy_eval.audio`.

Loudness is ITU-R BS.1770-4 / EBU R128 (K-weighting + gated mean square), which
is a published standard with no trained weights, so it is implemented here
rather than depended on; ``parity/`` banks the numbers against pyloudnorm in a
throwaway venv. See PROVENANCE.md.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import msgspec
import numpy as np

from ..errors import BackendError, ConfigError

#: Below this the level is reported as this floor rather than -inf, so a silent
#: arm produces a number a budget can compare instead of an infinity that breaks
#: every mean downstream. Chosen well under 16-bit noise floor (-96 dBFS).
DB_FLOOR = -120.0

#: |sample| at or above this counts as clipped. Not 1.0 exactly: a decoder that
#: rounds through int16 lands on 32767/32768, and lossy codecs overshoot.
CLIP_LEVEL = 0.999

#: A block whose RMS is below this is silence. -60 dBFS is the conventional
#: broadcast "digital black" test level.
SILENCE_DBFS = -60.0

#: Analysis block for the silence/level scan, in seconds.
BLOCK_SECONDS = 0.05

#: ITU-R BS.1770-4 K-weighting stage prototypes (high shelf, then RLB high
#: pass). The recommendation tabulates coefficients at 48 kHz only; these are
#: the analog prototype parameters they bilinear-transform from, so the filter
#: is correct at 32 kHz too — which matters, because H3 emits 32 kHz.
_SHELF = (3.999843853973347, 0.7071752369554196, 1681.974450955533)   # G dB, Q, fc Hz
_HIGHPASS = (0.0, 0.5003270373238773, 38.13547087602444)

#: BS.1770-4 gating: 400 ms blocks, 75% overlap, absolute gate then a relative
#: gate 10 LU below the absolute-gated loudness.
_BLOCK_S = 0.400
_OVERLAP = 0.75
_ABSOLUTE_GATE_LUFS = -70.0
_RELATIVE_GATE_LU = -10.0

#: Per-channel weights, BS.1770-4 Table 3. Only the first two matter for the
#: stereo we actually generate; the surround weights are here so a 5.1 mux does
#: not silently get scored as if the surrounds were front channels.
_CHANNEL_WEIGHTS = (1.0, 1.0, 1.0, 1.41, 1.41)

#: Identical signals give an infinite SI-SDR/SNR; report this finite sentinel
#: instead, matching the 99.0 dB convention similarity.py uses for PSNR.
SNR_CAP = 99.0

#: Bounded search window for the encoder-delay alignment, in seconds. AAC
#: priming is ~2112 samples; container mux offsets are a few ms. Anything beyond
#: this is a different take, not a delay, and compensating it would manufacture
#: agreement that is not there.
MAX_ALIGN_SECONDS = 0.050


class Audio(msgspec.Struct, frozen=True, kw_only=True):
    """One arm's decoded audio. ``samples`` is ``(N, C)`` float32 in [-1, 1]."""

    samples: Any
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        return float(self.samples.shape[0]) / float(self.sample_rate)

    @property
    def mono(self) -> Any:
        """Channel mean — the programme signal the level statistics read."""
        return self.samples.mean(axis=1)


def as_audio(source: Any, sample_rate: int = 0) -> Audio:
    """Normalize an accepted audio form to :class:`Audio`.

    Accepts an :class:`Audio` (returned unchanged), an ``(N,)`` or ``(N, C)``
    array with ``sample_rate``, or a ``(array, sample_rate)`` tuple. Integer
    dtypes are scaled by their full scale; floats are taken as already in
    [-1, 1] and are NOT rescaled — a float array peaking at 0.3 is quiet audio,
    not audio that needs normalizing, and guessing would erase exactly the level
    regression this module exists to catch.
    """
    if isinstance(source, Audio):
        return source
    if isinstance(source, tuple) and len(source) == 2 and np.isscalar(source[1]):
        source, sample_rate = source
    if not sample_rate:
        raise ConfigError(
            "as_audio: sample_rate is required — a level, a loudness and a sync "
            "offset are all meaningless without it; pass sample_rate=... or an "
            "(array, sample_rate) tuple"
        )
    arr = np.asarray(source)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ConfigError(
            f"as_audio: expected (N,) or (N, C) samples, got shape {arr.shape}"
        )
    if arr.shape[0] < arr.shape[1]:
        raise ConfigError(
            f"as_audio: got shape {arr.shape} — audio is (N samples, C channels), "
            "channel-last like frames are; this looks transposed"
        )
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max + 1)
    else:
        arr = arr.astype(np.float32)
    if arr.shape[0] < 2:
        raise ConfigError("as_audio: need at least 2 samples")
    return Audio(samples=arr, sample_rate=int(sample_rate))


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

def db(value: float) -> float:
    """Amplitude ratio to dB, floored rather than -inf."""
    v = float(value)
    return DB_FLOOR if v <= 0.0 else max(DB_FLOOR, 20.0 * math.log10(v))


def rms_dbfs(x: Any) -> float:
    """RMS level in dBFS. Matches ffmpeg ``volumedetect``'s ``mean_volume``."""
    x = np.asarray(x, np.float64)
    return db(math.sqrt(float(np.mean(x * x)))) if x.size else DB_FLOOR


def peak_dbfs(x: Any) -> float:
    """Sample peak in dBFS. Matches ffmpeg ``volumedetect``'s ``max_volume``.

    SAMPLE peak, not true peak: BS.1770-4 Annex 2's 4x-oversampled true peak
    would read higher on an already-encoded file, and what we are catching is a
    generator that renders hot, not a delivery-spec violation.
    """
    x = np.asarray(x, np.float64)
    return db(float(np.max(np.abs(x)))) if x.size else DB_FLOOR


# ---------------------------------------------------------------------------
# ITU-R BS.1770-4 loudness
# ---------------------------------------------------------------------------

# The BILINEAR (De Man) forms, NOT the RBJ Audio EQ Cookbook forms. This is
# load-bearing and cost a real debugging pass: driven with the same prototype
# parameters, the cookbook high-shelf and this one differ by ~0.2 dB of gain
# around 1 kHz, which lands as a 0.22 LU loudness error on tonal material and
# ~0.03 LU on broadband. Only these forms reproduce BS.1770-4's own tabulated
# 48 kHz coefficients (Tables 1 and 2), which tests/test_audio.py asserts
# to 1e-11 — a stronger provenance claim than agreeing with any one library.


def _biquad_shelf(gain_db: float, q: float, fc: float, rate: int) -> tuple[Any, Any]:
    k = math.tan(math.pi * fc / rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.499666774155
    den = 1.0 + k / q + k * k
    b = np.array([
        (vh + vb * k / q + k * k) / den,
        2.0 * (k * k - vh) / den,
        (vh - vb * k / q + k * k) / den,
    ])
    a = np.array([1.0, 2.0 * (k * k - 1.0) / den, (1.0 - k / q + k * k) / den])
    return b, a


def _biquad_highpass(q: float, fc: float, rate: int) -> tuple[Any, Any]:
    k = math.tan(math.pi * fc / rate)
    den = 1.0 + k / q + k * k
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([1.0, 2.0 * (k * k - 1.0) / den, (1.0 - k / q + k * k) / den])
    return b, a


def _lfilter(b: Any, a: Any, x: Any) -> Any:
    try:
        from scipy.signal import lfilter
    except ImportError as exc:                                  # pragma: no cover
        raise BackendError(
            "loudness (ITU-R BS.1770-4) needs scipy for the K-weighting IIR: "
            "pip install 'cozy-eval[audio]'"
        ) from exc
    return lfilter(b, a, x, axis=0)


def k_weight(audio: Audio) -> Any:
    """BS.1770-4 K-weighting: head-shelf then RLB high pass, per channel."""
    rate = audio.sample_rate
    x = audio.samples.astype(np.float64)
    x = _lfilter(*_biquad_shelf(_SHELF[0], _SHELF[1], _SHELF[2], rate), x)
    return _lfilter(*_biquad_highpass(_HIGHPASS[1], _HIGHPASS[2], rate), x)


def loudness_lufs(audio: Audio) -> float:
    """Integrated loudness in LUFS, ITU-R BS.1770-4 with EBU R128 gating.

    Returns :data:`DB_FLOOR` when every block gates out — which is the honest
    answer for silence, not 0.
    """
    weighted = k_weight(audio)
    block = round(_BLOCK_S * audio.sample_rate)
    step = max(1, round(block * (1.0 - _OVERLAP)))
    if weighted.shape[0] < block:
        return DB_FLOOR
    starts = range(0, weighted.shape[0] - block + 1, step)
    weights = np.array([
        _CHANNEL_WEIGHTS[i] if i < len(_CHANNEL_WEIGHTS) else 1.0
        for i in range(audio.channels)
    ])
    # z[j] = weighted mean square of block j, summed over weighted channels
    z = np.array([
        float(np.sum(weights * np.mean(weighted[s:s + block] ** 2, axis=0)))
        for s in starts
    ])
    if not z.size:
        return DB_FLOOR

    def _loud(values: Any) -> float:
        total = float(np.mean(values))
        return DB_FLOOR if total <= 0 else -0.691 + 10.0 * math.log10(total)

    lj = np.array([DB_FLOOR if v <= 0 else -0.691 + 10.0 * math.log10(v) for v in z])
    keep = lj > _ABSOLUTE_GATE_LUFS
    if not keep.any():
        return DB_FLOOR
    relative = _loud(z[keep]) + _RELATIVE_GATE_LU
    keep &= lj > relative
    return _loud(z[keep]) if keep.any() else DB_FLOOR


# ---------------------------------------------------------------------------
# reference-free statistics
# ---------------------------------------------------------------------------

def spectral_flatness(audio: Audio, *, n_fft: int = 1024) -> float:
    """Wiener entropy: geometric over arithmetic mean of the power spectrum.

    ~0 for a tone, ~1 for white noise. A generator whose soundtrack collapses
    into hiss climbs here; one that collapses into a drone falls.
    """
    spec = _stft_magnitude(audio.mono, n_fft=n_fft, hop=n_fft // 4)
    power = np.maximum(spec ** 2, 1e-20)
    geo = np.exp(np.mean(np.log(power), axis=0))
    arith = np.mean(power, axis=0)
    ok = arith > 1e-18
    return float(np.mean(geo[ok] / arith[ok])) if ok.any() else 0.0


def _blocks(x: Any, rate: int, seconds: float) -> Any:
    n = max(1, round(seconds * rate))
    usable = (x.shape[0] // n) * n
    if usable < n:
        return x[None, :]
    return x[:usable].reshape(-1, n)


def signal_stats(source: Any, sample_rate: int = 0) -> dict[str, float]:
    """Reference-free audio statistics. No reference arm, no prompt, no weights.

    Stereo keys (``audio_side_dbfs``, ``audio_channel_correlation``,
    ``audio_stereo_separation_db``) are ABSENT for mono input rather than zero —
    a mono file has no stereo image to be wrong about, and the caller must be
    able to tell "mono source" from "stereo source that collapsed".

    ``audio_lufs`` is likewise ABSENT when scipy is not installed. It is the one
    statistic here with a dependency, and one missing optional package must not
    take the whole reference-free tier down with it — the dual-mono and silence
    guards are the floor of the gate and they need nothing but numpy.
    """
    audio = as_audio(source, sample_rate)
    mono = audio.mono
    out: dict[str, float] = {
        "audio_rms_dbfs": rms_dbfs(audio.samples),
        "audio_peak_dbfs": peak_dbfs(audio.samples),
        "audio_clip_fraction": float(
            np.mean(np.abs(audio.samples) >= CLIP_LEVEL)
        ),
        "audio_dc_offset": float(np.max(np.abs(np.mean(audio.samples, axis=0)))),
        "audio_spectral_flatness": spectral_flatness(audio),
    }
    with contextlib.suppress(BackendError):     # no scipy: key absent, not zero
        out["audio_lufs"] = loudness_lufs(audio)
    blocks = _blocks(mono, audio.sample_rate, BLOCK_SECONDS)
    block_rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1))
    silent = np.array([db(v) for v in block_rms]) < SILENCE_DBFS
    out["audio_silence_fraction"] = float(np.mean(silent))

    if audio.channels >= 2:
        left, right = audio.samples[:, 0], audio.samples[:, 1]
        side = (left - right) / 2.0
        out["audio_side_dbfs"] = rms_dbfs(side)
        out["audio_stereo_separation_db"] = out["audio_rms_dbfs"] - out["audio_side_dbfs"]
        sl, sr = float(np.std(left)), float(np.std(right))
        out["audio_channel_correlation"] = (
            1.0 if sl <= 0 or sr <= 0 else float(np.corrcoef(left, right)[0, 1])
        )
    return out


# ---------------------------------------------------------------------------
# paired fidelity
# ---------------------------------------------------------------------------

def align_lag(reference: Any, candidate: Any, rate: int, *,
              max_seconds: float = MAX_ALIGN_SECONDS) -> int:
    """Integer-sample lag of ``candidate`` behind ``reference``, bounded.

    Codec priming and container mux offsets shift an arm by a couple of
    milliseconds; uncorrected, that alone can cost tens of dB of SNR and would
    read as damage. The search is deliberately BOUNDED — beyond
    :data:`MAX_ALIGN_SECONDS` a shift is a different take, not a delay, and
    sliding to find agreement would be manufacturing it.
    """
    limit = round(max_seconds * rate)
    if limit < 1:
        return 0
    a = np.asarray(reference, np.float64)
    b = np.asarray(candidate, np.float64)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    limit = min(limit, n - 1)
    floor = rate // 100                       # need >=10 ms of overlap to score a lag
    if limit < 1 or n <= floor:
        return 0

    # FFT cross-correlation, NOT a lag loop. At 32 kHz the +-50 ms window is 3201
    # lags and a real clip is half a million samples; the naive loop is ~10^9
    # multiply-adds per pair and dominated the whole metric pass.
    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    lags = np.arange(-limit, limit + 1)
    dots = corr[lags]                         # negative indices wrap to the tail

    # Per-lag norms of the two OVERLAPPING segments, from prefix sums — the same
    # normalization the loop did, so the answer is unchanged, only the cost.
    sa = np.concatenate([[0.0], np.cumsum(a * a)])
    sb = np.concatenate([[0.0], np.cumsum(b * b)])
    pos = lags >= 0
    len_x = n - np.abs(lags)
    # np.where evaluates BOTH branches, so every index is clipped into [0, n]
    # first; the unused branch's value is discarded but must still be legal.
    norm_a = np.where(
        pos,
        sa[n] - sa[np.clip(lags, 0, n)],
        sa[np.clip(n + lags, 0, n)],
    )
    norm_b = np.where(
        pos,
        sb[np.clip(n - lags, 0, n)],
        sb[n] - sb[np.clip(-lags, 0, n)],
    )
    denominator = np.sqrt(np.maximum(norm_a, 0.0) * np.maximum(norm_b, 0.0))
    scores = np.where(denominator > 0, dots / np.where(denominator > 0, denominator, 1.0), 0.0)
    scores[len_x < floor] = -np.inf
    if not np.isfinite(scores).any():
        return 0
    return int(lags[int(np.argmax(scores))])


def _stft_magnitude(x: Any, *, n_fft: int, hop: int) -> Any:
    """``(freq, time)`` magnitude spectrogram with a Hann window."""
    x = np.asarray(x, np.float64)
    if x.shape[0] < n_fft:
        x = np.pad(x, (0, n_fft - x.shape[0]))
    window = np.hanning(n_fft)
    count = 1 + (x.shape[0] - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(count, n_fft), strides=(x.strides[0] * hop, x.strides[0]),
    )
    return np.abs(np.fft.rfft(frames * window, axis=1)).T


def _mel_filterbank(rate: int, n_fft: int, bands: int = 64) -> Any:
    def to_mel(f: Any) -> Any:
        return 2595.0 * np.log10(1.0 + np.asarray(f, np.float64) / 700.0)

    def from_mel(m: Any) -> Any:
        return 700.0 * (10.0 ** (np.asarray(m, np.float64) / 2595.0) - 1.0)

    edges = from_mel(np.linspace(to_mel(0.0), to_mel(rate / 2.0), bands + 2))
    bins = np.floor((n_fft + 1) * edges / rate).astype(int)
    bank = np.zeros((bands, n_fft // 2 + 1))
    for i in range(bands):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            bank[i, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        if hi > mid:
            bank[i, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return bank


def si_sdr_db(reference: Any, candidate: Any) -> float:
    """Scale-invariant signal-to-distortion ratio in dB.

    Le Roux, Wisdom, Erdogan & Hershey, "SDR - half-baked or well done?",
    ICASSP 2019. Closed form, no weights. Scale-invariant is the point: a lane
    that changes output GAIN is not a lane that damaged the audio, and plain SNR
    would punish it as if it were.
    """
    ref = np.asarray(reference, np.float64)
    cand = np.asarray(candidate, np.float64)
    ref = ref - ref.mean()
    cand = cand - cand.mean()
    energy = float(np.dot(ref, ref))
    if energy <= 0.0:
        return DB_FLOOR
    projection = (float(np.dot(cand, ref)) / energy) * ref
    noise = cand - projection
    num, den = float(np.dot(projection, projection)), float(np.dot(noise, noise))
    if den <= 0.0:
        return SNR_CAP
    return min(SNR_CAP, 10.0 * math.log10(num / den)) if num > 0 else DB_FLOOR


def snr_db(reference: Any, candidate: Any) -> float:
    """Plain SNR in dB: reference energy over error energy, no scale freedom.

    Kept beside SI-SDR because it is the number the H3/LTX lanes already quote
    (the cache sweep that read 20.67 -> 13.72 dB was this), so historical rows
    stay comparable.
    """
    ref = np.asarray(reference, np.float64)
    cand = np.asarray(candidate, np.float64)
    num = float(np.dot(ref, ref))
    err = ref - cand
    den = float(np.dot(err, err))
    if num <= 0.0:
        return DB_FLOOR
    if den <= 0.0:
        return SNR_CAP
    return min(SNR_CAP, 10.0 * math.log10(num / den))


def log_spectral_distance(reference: Any, candidate: Any, *, n_fft: int = 1024) -> float:
    """Root-mean-square log-spectral distance in dB.

    Gray & Markel, "Distance measures for speech processing", IEEE TASSP 1976.
    Closed form. Unlike SNR it survives a phase difference, so it separates
    "the spectrum changed" from "the waveform moved".
    """
    a = _stft_magnitude(reference, n_fft=n_fft, hop=n_fft // 4)
    b = _stft_magnitude(candidate, n_fft=n_fft, hop=n_fft // 4)
    n = min(a.shape[1], b.shape[1])
    if n == 0:
        return 0.0
    la = 20.0 * np.log10(np.maximum(a[:, :n], 1e-10))
    lb = 20.0 * np.log10(np.maximum(b[:, :n], 1e-10))
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def mel_l1(reference: Any, candidate: Any, rate: int, *, n_fft: int = 1024) -> float:
    """Mean absolute difference of log-mel spectrograms.

    The perceptually-weighted sibling of the log-spectral distance, and the loss
    every neural vocoder is trained under, so its scale is familiar.
    """
    bank = _mel_filterbank(rate, n_fft)
    a = _stft_magnitude(reference, n_fft=n_fft, hop=n_fft // 4)
    b = _stft_magnitude(candidate, n_fft=n_fft, hop=n_fft // 4)
    n = min(a.shape[1], b.shape[1])
    if n == 0:
        return 0.0
    la = np.log(np.maximum(bank @ a[:, :n], 1e-10))
    lb = np.log(np.maximum(bank @ b[:, :n], 1e-10))
    return float(np.mean(np.abs(la - lb)))


def paired_stats(reference: Any, candidate: Any, sample_rate: int = 0, *,
                 align: bool = True) -> dict[str, float]:
    """Fidelity of ``candidate`` to ``reference``. The faithfulness question.

    Both arms are compared on the programme (channel-mean) signal, with the
    stereo image handled separately by the reference-free pair. Sample rates
    must match — resampling one arm to score it against the other would put the
    resampler's error inside the measurement.
    """
    ref = as_audio(reference, sample_rate)
    cand = as_audio(candidate, sample_rate)
    if ref.sample_rate != cand.sample_rate:
        raise ConfigError(
            f"paired audio: reference is {ref.sample_rate} Hz and candidate is "
            f"{cand.sample_rate} Hz — render both arms at the same rate; "
            "resampling here would score the resampler, not the model"
        )
    a, b = ref.mono, cand.mono
    out: dict[str, float] = {}
    lag = align_lag(a, b, ref.sample_rate) if align else 0
    out["audio_align_lag_ms"] = 1000.0 * lag / ref.sample_rate
    if lag > 0:
        a, b = a[lag:], b[:b.shape[0] - lag]
    elif lag < 0:
        a, b = a[:a.shape[0] + lag], b[-lag:]
    n = min(a.shape[0], b.shape[0])
    if n < ref.sample_rate // 100:
        raise ConfigError(
            "paired audio: fewer than 10 ms overlap after alignment — the two "
            "arms are not the same take"
        )
    a, b = a[:n], b[:n]
    out["audio_si_sdr"] = si_sdr_db(a, b)
    out["audio_snr_db"] = snr_db(a, b)
    out["audio_lsd_db"] = log_spectral_distance(a, b)
    out["audio_mel_l1"] = mel_l1(a, b, ref.sample_rate)
    out["audio_lufs_delta"] = loudness_lufs(cand) - loudness_lufs(ref)
    return out


AUDIO_LIBRARY = "cozy-eval:audio"


__all__ = [
    "AUDIO_LIBRARY",
    "BLOCK_SECONDS",
    "CLIP_LEVEL",
    "DB_FLOOR",
    "MAX_ALIGN_SECONDS",
    "SILENCE_DBFS",
    "SNR_CAP",
    "Audio",
    "align_lag",
    "as_audio",
    "db",
    "k_weight",
    "log_spectral_distance",
    "loudness_lufs",
    "mel_l1",
    "paired_stats",
    "peak_dbfs",
    "rms_dbfs",
    "si_sdr_db",
    "signal_stats",
    "snr_db",
    "spectral_flatness",
]
