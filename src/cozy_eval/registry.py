"""What an evaluation is allowed to measure, and which numbers gate.

ONE table for the whole library — every metric, whatever module computes it and
whoever wrote that module.

STABILITY: locked core (v0.x). See the stability section of the README.

FOUR DIMENSIONS. The design rule is that a suite needs a small number of
*mostly independent* numbers: if two metrics are almost perfectly correlated
they are one metric wearing two hats, and reporting both only adds chances to
fail a good model on the same underlying signal.

  similarity  — perceptual distance to a reference render.  PAIRED only.
  adherence   — did the render contain what the prompt specified.
  preference  — would a human prefer this render.
  quality     — does the render look good on its own terms.  Reference-free.

AUDIO IS NOT A FIFTH DIMENSION, deliberately. The four dimensions are
QUESTIONS, not media: 'is it faithful to the reference', 'did it contain what
was asked', 'would a human prefer it', 'is it well formed on its own'. Those
questions are the same for a soundtrack, so audio SNR is similarity, a
sound-of-X checklist is adherence, and clipping/silence/dual-mono are quality.
The payoff is mechanical rather than aesthetic: the tri-state parity verdict in
``cozy_eval.verdict`` reads FAITHFULNESS_DIMENSIONS and PARITY_DIMENSIONS, so a
quant arm that wrecks the audio reaches ``reject`` through the machinery that
already existed, with no audio-specific verdict path to keep in sync.

Each dimension has exactly ONE gated headline number plus its worst-case tail.
Everything else is report-only: computed because it is nearly free and useful
to a human reader, never able to fail a candidate on its own.

NOT declared here, deliberately: the population gate's derived indices
(``imaging_index``, ``jerk_excess``, ``flicker_ratio``, ``significant_features``,
``population_frechet``). Those are declared as data too — in
:data:`cozy_eval.benchmarks.VIDEO_BUDGETS` — because they are two-sided budgets
on statistics computed ACROSS a prompt population, and ``MetricSpec`` has one
direction and one row. The split is metric-vs-threshold, not layer-vs-layer.

Redundancy here was MEASURED, not assumed — see ``REDUNDANCY_EVIDENCE``.

**Every metric this package implements is declared in ``BUILTIN`` below, as
data.** A ``MetricSpec`` is pure metadata — no torch, no weights — so the table
is complete on ``import cozy_eval`` whether or not the module that computes a
given number was imported. That is deliberate: if metrics registered themselves
as an import side effect, ``REGISTRY``, ``GATED`` and ``headline()`` would
answer differently depending on what the caller happened to import, and a
report's ``metric_set`` would not identify the set that produced it.

:func:`register` is the extension point for metrics that live OUTSIDE this
package. Do not edit ``BUILTIN`` to add one, and do not call ``register()`` from
inside this package.

NAMING RULE, enforced by ``test_metric_names_follow_one_scheme``: a metric that
implements a *published, named* method keeps that name verbatim (``lpips``,
``niqe``, ``musiq``, ``arniqa``, ``clip_iqa``, ``ssim``, ``ms_ssim``, ``psnr``,
``vqascore``). Anything we derive ourselves is ``<subject>_<quantity>``
(``element_recall``, ``edit_compliance``, ``pref_delta``, ``composition_recall``,
``hpsv3_score``). Names are lowercase ``[a-z0-9_]`` and are part of the locked
API — budgets and banked reports key on them.
"""

from __future__ import annotations

import re
from typing import Any

import msgspec

from .errors import RegistryError

#: Identifies the METRIC SET a report was produced with. Same ``cozy-eval/<thing>@<n>``
#: convention as ``suite.REPORT_SCHEMA``. Bump when a metric's meaning changes.
METRIC_SET_VERSION = "cozy-eval/metrics@4"

# A metric name is an API key: budgets, banked reports and report rows all
# address metrics by it.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

SIMILARITY = "similarity"
ADHERENCE = "adherence"
PREFERENCE = "preference"
QUALITY = "quality"
DIMENSIONS: tuple[str, ...] = (SIMILARITY, ADHERENCE, PREFERENCE, QUALITY)

# REDUNDANCY, MEASURED. 44 banked A/B pairs: 20 fp8-quantized renders of one
# model at 1328^2 on a hard prompt set, 8 more from a second model, 8 from a
# third, and 8 same-checkpoint/different-seed control rows. Pearson / Spearman
# WITHIN the single-model population (the cleanest read — one model, one prompt
# set, one recipe):
#
#     lpips vs ms_ssim   -0.960 / -0.961     <- the same number, inverted
#     lpips vs ssim      -0.873 / -0.829
#     lpips vs psnr      -0.826 / -0.814
#     ssim  vs psnr      +0.874 / +0.922
#     clip_delta vs ALL of the above:  |r| <= 0.14   <- genuinely orthogonal
#
# So the four pixel/perceptual metrics are ONE dimension wearing four hats.
# LPIPS keeps the gate (learned perceptual distance, best human agreement of
# the four); SSIM/MS-SSIM/PSNR are computed and REPORTED because they cost
# ~nothing beside LPIPS and a human reads 0.81 more easily than a perceptual
# distance. clip_delta survives on the evidence: uncorrelated with similarity,
# so it measures something else.
#
# The honest gaps, stated rather than hidden: DISTS (texture-oriented) is the
# one similarity metric that plausibly is NOT redundant with LPIPS and has not
# been measured here; and element_recall vs clip_delta has no measurement yet
# because the correlation study predates any judged row.
REDUNDANCY_EVIDENCE = "correlation study, 44 banked pairs, 2026-07-27"


class MetricSpec(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    dimension: str
    version: str
    model_ref: str = ""       # HF id / package of the scoring model; "" = closed form
    gated: bool = False       # False = computed and recorded, never fails a candidate
    headline: bool = False    # THE number for its dimension; exactly one per dimension
    higher_is_better: bool = True
    paired: bool = True       # needs a reference render; False = scores one image
    note: str = ""


BUILTIN: tuple[MetricSpec, ...] = (
    # --- similarity (paired only) -----------------------------------------
    MetricSpec(
        name="lpips", dimension=SIMILARITY, version="1.0",
        model_ref="torchmetrics:lpips-alex",
        gated=True, headline=True, higher_is_better=False,
        note="dimension headline: learned perceptual distance to the reference",
    ),
    MetricSpec(
        name="ssim", dimension=SIMILARITY, version="1.0", note=(
            "report-only: r=-0.87 with lpips on banked pairs. Kept because a "
            "human reads 0.82 more easily than an LPIPS distance."
        ),
    ),
    MetricSpec(
        name="ms_ssim", dimension=SIMILARITY, version="1.0", note=(
            "report-only: r=-0.96 with lpips — effectively the same number."
        ),
    ),
    MetricSpec(
        name="psnr", dimension=SIMILARITY, version="1.0", higher_is_better=True,
        note=(
            "report-only. Pixel identity punishes same-seed trajectory "
            "divergence identically whether the candidate got worse or merely "
            "different. Kept so historical rows stay comparable."
        ),
    ),
    MetricSpec(
        name="vmaf", dimension=SIMILARITY, version="1.0",
        model_ref="libvmaf:vmaf_v0.6.1", note=(
            "reference lane only, whole video files through ffmpeg's libvmaf. "
            "Report-only here because its budgets are per-content: a VMAF that "
            "is excellent for animation is mediocre for film grain."
        ),
    ),
    MetricSpec(
        name="cvvdp", dimension=SIMILARITY, version="1.0",
        model_ref="cvvdp:standard_4k", note=(
            "reference lane only. ColorVideoVDP JOD units — a perceptual "
            "difference predictor that models colour and motion, unlike the "
            "luma-only classics beside it. Report-only for the same reason."
        ),
    ),
    # --- adherence --------------------------------------------------------
    MetricSpec(
        name="element_recall", dimension=ADHERENCE, version="1.0", gated=True,
        headline=True, paired=False, note=(
            "dimension headline: fraction of the prompt's authored checklist "
            "items verified present. Reference-free — scores any render. In a "
            "PAIRED comparison the candidate-minus-reference delta is what gates."
        ),
    ),
    MetricSpec(
        name="text_exact", dimension=ADHERENCE, version="1.0", paired=False,
        model_ref="rapidocr", note="OCR exact-match rate over the prompt's quoted strings",
    ),
    MetricSpec(
        name="text_fuzzy", dimension=ADHERENCE, version="1.0", paired=False,
        model_ref="rapidocr", note="OCR normalized-similarity mean over the same strings",
    ),
    MetricSpec(
        name="edit_compliance", dimension=ADHERENCE, version="1.0", paired=False,
        note="EDIT mode: did the instructed change happen (judged on before+after)",
    ),
    MetricSpec(
        name="edit_preservation", dimension=ADHERENCE, version="1.0", paired=False,
        note="EDIT mode: did everything the instruction did not mention stay put",
    ),
    MetricSpec(
        name="clip_delta", dimension=ADHERENCE, version="1.0",
        model_ref="openai/clip-vit-base-patch32", gated=True, note=(
            "FALLBACK GATE: gates only when element_recall is UNMEASURED (no "
            "judge/OCR on this image), so an adherence-blind run is never an "
            "ungated run. Orthogonal to similarity (|r|<=0.14) so it is not "
            "redundant with that dimension, but it is a bag-of-concepts proxy "
            "for the question element_recall answers directly."
        ),
    ),
    # --- preference -------------------------------------------------------
    MetricSpec(
        name="pref_delta", dimension=PREFERENCE, version="1.0",
        model_ref="yuvalkirstain/PickScore_v1", gated=True,
        headline=True, note=(
            "dimension headline: candidate-minus-reference human-preference "
            "score. The absolute scale is arbitrary and NOT comparable across "
            "prompts — only same-prompt deltas are meaningful. A continuous "
            "CLIP-H scalar resolves the small same-prompt deltas a generative "
            "1-10 judge would round to zero. NOTE: these weights carry no "
            "declared licence; see PROVENANCE.md before relying on them."
        ),
    ),
    MetricSpec(
        name="pref_score", dimension=PREFERENCE, version="1.0",
        model_ref="yuvalkirstain/PickScore_v1", paired=False,
        note="the raw per-image preference score; what an absolute leaderboard reads",
    ),
    MetricSpec(
        name="hpsv3_score", dimension=PREFERENCE, version="1.0",
        model_ref="MizzenAI/HPSv3", paired=False, note=(
            "HPSv3 mu (uncertainty-aware preference reward, Qwen2-VL-7B "
            "backbone); the HPDv3 accuracy leader, so it is the recommended "
            "reference-free/leaderboard scorer. Report-only in the paired "
            "gate: measured on 52 banked pairs, it buys no same-prompt delta "
            "resolution over PickScore at ~80x the load cost."
        ),
    ),
    # --- adherence, compositional (detector-scored) ------------------------
    MetricSpec(
        name="composition_correct", dimension=ADHERENCE, version="1.0",
        model_ref="IDEA-Research/grounding-dino-tiny", paired=False, note=(
            "GenEval-style boolean per image (ALL include clauses AND no exclude); "
            "the mean over a prompt set is the published-GenEval-comparable number"
        ),
    ),
    MetricSpec(
        name="composition_recall", dimension=ADHERENCE, version="1.0",
        model_ref="IDEA-Research/grounding-dino-tiny", paired=False, note=(
            "graded fraction of compositional terms satisfied; detector-scored "
            "sibling of element_recall for specs with structured object clauses"
        ),
    ),
    # --- adherence, soft (VLM p(yes) rather than a parsed yes/no) ----------
    MetricSpec(
        name="vqascore", dimension=ADHERENCE, version="1.0",
        model_ref="Qwen/Qwen3-VL-8B-Instruct", paired=False, note=(
            "soft single-number alignment, p(yes) for 'does this figure show "
            "{prompt}'; no decomposition, so it survives prompts with no checklist"
        ),
    ),
    MetricSpec(
        name="atom_score", dimension=ADHERENCE, version="1.0",
        model_ref="Qwen/Qwen3-VL-8B-Instruct", paired=False, note=(
            "Soft-TIFA arithmetic mean of per-atom p(yes) over the versioned "
            "template decomposition; soft sibling of element_recall with delta "
            "resolution a hard parse rounds away"
        ),
    ),
    # --- quality (reference-free, no prompt) -------------------------------
    MetricSpec(
        name="arniqa", dimension=QUALITY, version="1.0",
        model_ref="torchmetrics:arniqa-koniq10k", paired=False, headline=True,
        note=(
            "dimension headline: learned no-reference quality, 0..1. Report-only "
            "for now — quality gating is a policy decision that waits for banked "
            "distributions."
        ),
    ),
    MetricSpec(
        name="clip_iqa", dimension=QUALITY, version="1.0",
        model_ref="torchmetrics:clip-iqa", paired=False, note=(
            "CLIP prompt-pair quality probe, 0..1; OpenAI CLIP backbone (licence "
            "caveat in PROVENANCE.md)"
        ),
    ),
    MetricSpec(
        name="niqe", dimension=QUALITY, version="1.0",
        paired=False, higher_is_better=False, note=(
            "closed-form natural-scene statistics distance, no weights; the only "
            "quality number that runs without torch"
        ),
    ),
    MetricSpec(
        name="musiq", dimension=QUALITY, version="1.0",
        model_ref="google-research/musiq:koniq (Apache-2.0)", paired=False, note=(
            "multi-scale quality transformer at native resolution; our PyTorch "
            "port of the Apache reference + checkpoints (no permissive PyTorch "
            "path existed). koniq MOS scale, ~0-100"
        ),
    ),
    # --- video (report-only until video distributions are banked; a threshold
    # --- is only set where measured values exist to calibrate it) ------------
    MetricSpec(
        name="lpips_frame_worst", dimension=SIMILARITY, version="1.0",
        model_ref="torchmetrics:lpips-alex", higher_is_better=False, note=(
            "video: the worst single-frame LPIPS within one clip pair. The "
            "per-frame MEAN travels under lpips itself; this is its tail — a "
            "clip that is faithful on average can still hold one ruined frame"
        ),
    ),
    MetricSpec(
        name="dframe_psnr", dimension=SIMILARITY, version="1.0", note=(
            "video: mean PSNR between the two clips' consecutive-frame "
            "DIFFERENCE images — frame-to-frame dynamics preservation. A "
            "candidate can match per-frame LPIPS yet flicker; alternating "
            "errors are first-order here and invisible per-frame. dB on the "
            "frame scale (data_range 1.0)"
        ),
    ),
    MetricSpec(
        name="dframe_psnr_worst", dimension=SIMILARITY, version="1.0", note=(
            "video: the worst single Δ-frame step of dframe_psnr — one botched "
            "transition (a popped frame) that the mean would absorb"
        ),
    ),
    MetricSpec(
        name="dframe_ssim", dimension=SIMILARITY, version="1.0", note=(
            "video: mean SSIM between the two clips' Δ-frames, mapped to image "
            "range about mid-grey — structural agreement of what moved"
        ),
    ),
    MetricSpec(
        name="motion_compliance", dimension=ADHERENCE, version="1.0", paired=False,
        note=(
            "video checklists: did the asked-for dynamics HAPPEN (a static clip "
            "fails here). The motion half of the motion/hold duality; "
            "element_recall stays the weighted union"
        ),
    ),
    MetricSpec(
        name="static_preservation", dimension=ADHERENCE, version="1.0", paired=False,
        note=(
            "video checklists: did everything meant to stay constant STAY "
            "(identity drift and background reshuffling fail here). The hold "
            "half of the motion/hold duality"
        ),
    ),
    MetricSpec(
        name="luma_flicker", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:signal", paired=False,
        higher_is_better=False, note=(
            "video, composed from cozy_eval's signal backend (its `flicker`): "
            "frame-mean luma wobble in percent, reference-free"
        ),
    ),
    MetricSpec(
        name="jerk_ratio", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:signal", paired=False,
        higher_is_better=False, note=(
            "video, composed from cozy-eval's signal backend: second/first temporal "
            "difference of frame-mean luma; smooth motion sits low, flicker "
            "and judder push it up"
        ),
    ),
    # --- audio, paired fidelity (SIMILARITY: the faithfulness question) ------
    MetricSpec(
        name="audio_si_sdr", dimension=SIMILARITY, version="1.0",
        model_ref="cozy-eval:audio", gated=True, note=(
            "AUDIO GATE. Scale-invariant signal-to-distortion ratio in dB "
            "against the reference arm (Le Roux et al., ICASSP 2019). Closed "
            "form, no weights. Scale-invariant deliberately: a lane that only "
            "changed output GAIN did not damage the audio, and plain SNR would "
            "punish it as if it had. This is the number a quant/cache/distill "
            "arm has to hold — a cache sweep once drove audio SNR 20.67 -> "
            "13.72 dB while the same arm's video SSIM still read 0.85."
        ),
    ),
    MetricSpec(
        name="audio_snr_db", dimension=SIMILARITY, version="1.0",
        model_ref="cozy-eval:audio", note=(
            "report-only: plain SNR against the reference arm, no scale freedom. "
            "Kept beside audio_si_sdr because it is the number the H3/LTX lanes "
            "already quote, so historical rows stay comparable."
        ),
    ),
    MetricSpec(
        name="audio_lsd_db", dimension=SIMILARITY, version="1.0",
        model_ref="cozy-eval:audio", higher_is_better=False, note=(
            "report-only: RMS log-spectral distance in dB (Gray & Markel, IEEE "
            "TASSP 1976). Survives a phase difference, so it separates 'the "
            "spectrum changed' from 'the waveform moved' — the audio analogue of "
            "the take-divergence problem protocol.py exists for."
        ),
    ),
    MetricSpec(
        name="audio_mel_l1", dimension=SIMILARITY, version="1.0",
        model_ref="cozy-eval:audio", higher_is_better=False, note=(
            "report-only: mean absolute log-mel spectrogram difference. The "
            "perceptually-weighted sibling of audio_lsd_db, on the scale every "
            "neural vocoder is trained under."
        ),
    ),
    MetricSpec(
        name="audio_lufs_delta", dimension=SIMILARITY, version="1.0",
        model_ref="cozy-eval:audio", higher_is_better=False, note=(
            "report-only, SIGNED: candidate-minus-reference integrated loudness in "
            "LU. SIMILARITY rather than QUALITY because it is a comparison to the "
            "reference arm, and QUALITY means reference-free here. It exists "
            "because audio_si_sdr is blind to gain BY CONSTRUCTION, so a pure "
            "level shift would otherwise leave no trace anywhere in the report. "
            "Signed, so its tail reads as the loudest drift, not the largest."
        ),
    ),
    MetricSpec(
        name="audio_align_lag_ms", dimension=SIMILARITY, version="1.0",
        model_ref="cozy-eval:audio", higher_is_better=False, note=(
            "report-only DIAGNOSTIC, signed: the bounded encoder-delay lag "
            "compensated before the paired numbers were taken. A non-zero value "
            "means the two arms were not sample-aligned as delivered; a value at "
            "the search bound means the alignment did not converge and the "
            "paired numbers should be distrusted."
        ),
    ),
    MetricSpec(
        name="av_sync_drift_ms", dimension=SIMILARITY, version="1.0",
        model_ref="cozy-eval:onset-correlation", gated=True,
        higher_is_better=False, note=(
            "AUDIO GATE. |candidate AV-sync offset - reference AV-sync offset|, "
            "in ms. THE paired sync number: absolute offset is confounded by "
            "content, drift is not, and any constant instrument bias cancels. "
            "Measured only where BOTH arms had alignable onsets; UNMEASURED "
            "otherwise, never 0."
        ),
    ),
    # --- audio, reference-free signal statistics (QUALITY) ------------------
    MetricSpec(
        name="audio_rms_dbfs", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, note=(
            "programme level in dBFS, matching ffmpeg volumedetect's "
            "mean_volume. TWO-SIDED (too quiet and too hot are both faults), so "
            "the direction flag is nominal and the real bounds live in "
            "cozy_eval.audio.AUDIO_DEFECTS as data — same metric-vs-threshold "
            "split the video population budgets use."
        ),
    ),
    MetricSpec(
        name="audio_peak_dbfs", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, higher_is_better=False, note=(
            "sample peak in dBFS (volumedetect's max_volume). SAMPLE peak, not "
            "BS.1770 true peak: what this catches is a generator rendering into "
            "the rail, not a delivery-spec violation."
        ),
    ),
    MetricSpec(
        name="audio_lufs", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, note=(
            "integrated loudness, ITU-R BS.1770-4 with EBU R128 gating. A "
            "published standard with no trained weights, so it is implemented "
            "here rather than depended on; parity banked against pyloudnorm "
            "(MIT). Two-sided like audio_rms_dbfs."
        ),
    ),
    MetricSpec(
        name="audio_clip_fraction", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, higher_is_better=False, note=(
            "fraction of samples at or above 0.999 full scale"
        ),
    ),
    MetricSpec(
        name="audio_silence_fraction", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, higher_is_better=False, note=(
            "fraction of 50 ms blocks below -60 dBFS. The 'the generator "
            "stopped' detector — a silent soundtrack is the failure mode no "
            "video metric can see."
        ),
    ),
    MetricSpec(
        name="audio_dc_offset", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, higher_is_better=False, note=(
            "largest per-channel mean sample value; a broken decode or vocoder, "
            "never a creative choice"
        ),
    ),
    MetricSpec(
        name="audio_spectral_flatness", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, higher_is_better=False, note=(
            "Wiener entropy of the power spectrum, ~0 for a tone and ~1 for "
            "white noise. TWO-SIDED — a soundtrack collapsing into hiss climbs "
            "and one collapsing into a drone falls — so the direction flag is "
            "nominal and this is report-only."
        ),
    ),
    MetricSpec(
        name="audio_side_dbfs", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, note=(
            "level of the side signal (L-R)/2 in dBFS. ABSENT for mono input "
            "rather than zero: a mono source has no stereo image to be wrong "
            "about, and that is a different fact from a stereo source that "
            "collapsed."
        ),
    ),
    MetricSpec(
        name="audio_stereo_separation_db", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, higher_is_better=False, note=(
            "DUAL-MONO GUARD: programme RMS minus side RMS. A model that claims "
            "stereo and emits identical channels is a real regression and a "
            "trivial one to catch — the side signal falls to the level floor and "
            "this climbs to ~99 dB. fal's MiniMax-H3 output sits at 2.5-20 dB "
            "over 18 clips; the anchor clip reads 8.4 dB."
        ),
    ),
    MetricSpec(
        name="audio_channel_correlation", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:audio", paired=False, higher_is_better=False, note=(
            "Pearson r between L and R. Two-sided (r=1 is dual mono, r=-1 is "
            "polarity-inverted); report-only beside audio_stereo_separation_db, "
            "which is the gated form of the same fault."
        ),
    ),
    # --- audio, AV-sync single-arm (QUALITY) --------------------------------
    MetricSpec(
        name="av_sync_offset_ms", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:onset-correlation", paired=False,
        higher_is_better=False, note=(
            "SIGNED audio-visual offset in ms by onset cross-correlation "
            "(Bello et al. 2005 for the audio envelope; Hershey & Movellan, "
            "NIPS 1999 for the audio-against-visual-change correlation). "
            "Positive = audio late. Closed form, NO trained weights. Report-only: "
            "the absolute offset is content-confounded, av_sync_drift_ms is the "
            "gated paired read. Does NOT score lip-sync — see PROVENANCE.md."
        ),
    ),
    MetricSpec(
        name="av_sync_confidence", dimension=QUALITY, version="1.0",
        model_ref="cozy-eval:onset-correlation", paired=False, note=(
            "Pearson r at the winning lag. Reported so an offset always travels "
            "with the strength of the evidence for it; below the peak/prominence "
            "floor the offset is not reported at all."
        ),
    ),
    # --- audio, semantic adherence -----------------------------------------
    MetricSpec(
        name="audio_event_recall", dimension=ADHERENCE, version="1.0",
        paired=False, gated=True, note=(
            "AUDIO GATE, capability side: fraction of the prompt's authored "
            "SOUND-EVENT checklist items an audio-capable judge verified "
            "('is there a sound of rain'). Transcription cannot answer these, so "
            "it is deliberately a separate instrument from audio_speech_*."
        ),
    ),
    MetricSpec(
        name="audio_speech_exact", dimension=ADHERENCE, version="1.0",
        paired=False, model_ref="openai/whisper (MIT code AND weights)", note=(
            "transcribe-then-judge: exact-match rate over the prompt's required "
            "spoken strings. Reuses the OCR lane's fuzzy matcher, because "
            "reading a required string out of recognised text is the same "
            "operation whether it came from pixels or from samples."
        ),
    ),
    MetricSpec(
        name="audio_speech_fuzzy", dimension=ADHERENCE, version="1.0",
        paired=False, model_ref="openai/whisper (MIT code AND weights)", note=(
            "normalized-similarity mean over the same spoken strings"
        ),
    ),
)

# Live registry. Rebuilt in place by register() so that modules holding a
# reference to THIS MODULE (``from cozy_eval import registry``) always see the
# current set. Do not do ``from .registry import REGISTRY`` — that snapshots.
REGISTRY: tuple[MetricSpec, ...] = BUILTIN
BY_NAME: dict[str, MetricSpec] = {}
GATED: tuple[str, ...] = ()
REPORT_ONLY: tuple[str, ...] = ()


def _rebuild() -> None:
    # Rebinding at module level is the point: holders of `from . import
    # registry` must see the live set.
    global BY_NAME, GATED, REPORT_ONLY  # noqa: PLW0603
    BY_NAME = {m.name: m for m in REGISTRY}
    GATED = tuple(m.name for m in REGISTRY if m.gated)
    REPORT_ONLY = tuple(m.name for m in REGISTRY if not m.gated)


_rebuild()


def register(spec: MetricSpec, *, replace: bool = False) -> MetricSpec:
    """Add a metric to the live registry.

    The extension point for metrics that live OUTSIDE this package. Raises on a
    duplicate name unless ``replace=True``, and refuses a second headline for a
    dimension that already has one — the one-headline-per-dimension invariant is
    what makes a report readable, so it is enforced at registration rather than
    discovered later at read time.
    """
    global REGISTRY  # noqa: PLW0603 — see _rebuild
    if not NAME_PATTERN.match(spec.name):
        raise RegistryError(
            f"metric name {spec.name!r} is not a valid key: names are lowercase "
            f"[a-z][a-z0-9_]* because budgets and banked reports address metrics by name"
        )
    if spec.dimension not in DIMENSIONS:
        raise RegistryError(
            f"metric {spec.name!r} declares unknown dimension {spec.dimension!r}; "
            f"known: {list(DIMENSIONS)}"
        )
    if not spec.version:
        raise RegistryError(f"metric {spec.name!r} must declare a version")
    existing = BY_NAME.get(spec.name)
    if existing is not None and not replace:
        raise RegistryError(
            f"metric {spec.name!r} is already registered; pass replace=True to override"
        )
    if spec.headline:
        clash = [
            m.name for m in REGISTRY
            if m.dimension == spec.dimension and m.headline and m.name != spec.name
        ]
        if clash:
            raise RegistryError(
                f"dimension {spec.dimension!r} already has headline metric {clash[0]!r}; "
                f"{spec.name!r} cannot also be the headline"
            )
    REGISTRY = (*(m for m in REGISTRY if m.name != spec.name), spec)
    _rebuild()
    return spec


def unregister(name: str) -> None:
    """Remove a metric. Mainly for tests that register a throwaway spec."""
    global REGISTRY  # noqa: PLW0603 — see _rebuild
    REGISTRY = tuple(m for m in REGISTRY if m.name != name)
    _rebuild()


def by_dimension(dimension: str) -> tuple[MetricSpec, ...]:
    return tuple(m for m in REGISTRY if m.dimension == dimension)


def headline(dimension: str) -> MetricSpec:
    """The one number that names a dimension."""
    picks = [m for m in by_dimension(dimension) if m.headline]
    if len(picks) != 1:
        raise RegistryError(
            f"dimension {dimension!r} must have exactly one headline metric, "
            f"has {[m.name for m in picks]}"
        )
    return picks[0]


def spec(name: str) -> MetricSpec:
    """Look up one metric, with an error that lists what does exist."""
    found = BY_NAME.get(name)
    if found is None:
        raise RegistryError(
            f"unknown metric {name!r}; registered: {sorted(BY_NAME)}"
        )
    return found


def reference_free() -> tuple[MetricSpec, ...]:
    """Metrics that score a single render — the absolute-evaluation mode."""
    return tuple(m for m in REGISTRY if not m.paired)


def registry_json() -> list[dict[str, Any]]:
    return [msgspec.to_builtins(m) for m in REGISTRY]


__all__ = [
    "ADHERENCE",
    "BUILTIN",
    "BY_NAME",
    "DIMENSIONS",
    "GATED",
    "METRIC_SET_VERSION",
    "NAME_PATTERN",
    "PREFERENCE",
    "QUALITY",
    "REDUNDANCY_EVIDENCE",
    "REGISTRY",
    "REPORT_ONLY",
    "SIMILARITY",
    "MetricSpec",
    "by_dimension",
    "headline",
    "reference_free",
    "register",
    "registry_json",
    "spec",
    "unregister",
]
