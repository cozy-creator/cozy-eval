"""Audio mode: ingest, the defect budget, and THE ONE audio verdict function.

STABILITY: locked core (v0.x) for :func:`audio_verdict`'s call shape and the
:class:`AudioVerdict` it returns — the H3 quality gates (te#171-B parity,
pgw#1011 technique gates, the fal-parity arm) call exactly this, so it is the
part that has to stay still.

WHY ONE FUNCTION. There are three audio tiers and they fail differently, but a
gate wants ONE answer. :func:`audio_verdict` runs whatever the inputs support,
records what it could NOT run and why, and returns a tri-state consistent with
the standing quant-verdict policy (faithfulness to the bf16 anchor is the gold
standard, capability parity the fallback).

THE THREE OUTCOMES, and the fourth that is not an outcome:

* absolute DEFECTS — silence, clipping, dual-mono, DC — fail on their own,
  against a budget this module DOES ship. That is the deliberate exception to
  the library's "thresholds are not shipped" rule, and the reason is that these
  are content-INDEPENDENT engineering faults: a soundtrack that is 90% digital
  silence is broken for every prompt, every family and every resolution, unlike
  an LPIPS budget which is only meaningful against the population it was
  calibrated on.
* paired FIDELITY against the reference arm, judged by a caller-supplied budget
  like every other paired number in this library.
* SEMANTIC adherence — speech read back by a transcriber, sound events judged
  by an audio-capable VLM.
* and **UNMEASURED**, which is never silently a pass. Everything the run could
  not measure lands in :attr:`AudioVerdict.unmeasured` with the reason, and a
  verdict with nothing measured is ``UNMEASURED``, not ``pass``.

INGEST. :func:`read_audio` decodes a file through the ffmpeg CLI, the same
optional-soft-dependency stance :mod:`cozy_eval.frames` already takes for video.
Producers that hold the pre-mux waveform should pass the array instead and never
touch a codec.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgspec

from . import registry, suite
from .errors import ConfigError, DecodeError
from .judge import AudioJudge, Transcriber
from .metrics import adherence, avsync
from .metrics import audio as audio_metrics
from .metrics.adherence import (
    KIND_OCR,
    KIND_VQA,
    OCR_FUZZY_PASS,
    ChecklistItem,
    ItemVerdict,
    parse_judge_answers,
    text_match,
)
from .metrics.audio import Audio, as_audio

PASS = "pass"
REJECT = "reject"
UNMEASURED = "unmeasured"


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def probe_audio(path: str | Path) -> tuple[int, int, float]:
    """``(sample_rate, channels, duration)`` of the first audio stream.

    Raises :class:`~cozy_eval.errors.DecodeError` when the file has NO audio
    stream — for a model that claims to generate audio, a missing track is the
    loudest possible defect and must not read as "0 measured, all fine".
    """
    try:
        raw = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DecodeError(f"ffprobe failed on {path}: {exc}") from exc
    streams = [s for s in json.loads(raw)["streams"] if s.get("codec_type") == "audio"]
    if not streams:
        raise DecodeError(
            f"no audio stream in {path} — for an audio-bearing model that is a "
            "defect, not an absence of data"
        )
    s = streams[0]
    return int(s["sample_rate"]), int(s["channels"]), float(s.get("duration") or 0.0)


def read_audio(path: str | Path, *, sample_rate: int = 0, channels: int = 0) -> Audio:
    """Decode a media file's audio to :class:`~cozy_eval.metrics.audio.Audio`.

    Defaults preserve the file's own rate and channel count, because a silent
    downmix would erase the stereo collapse this module exists to catch.
    """
    native_rate, native_channels, _ = probe_audio(path)
    rate = sample_rate or native_rate
    chans = channels or native_channels
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", str(path), "-vn", "-f", "f32le",
             "-acodec", "pcm_f32le", "-ar", str(rate), "-ac", str(chans), "-"],
            capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DecodeError(f"ffmpeg failed to decode audio from {path}: {exc}") from exc
    import numpy as np

    samples = np.frombuffer(raw, np.float32)
    if samples.size < chans * 2:
        raise DecodeError(f"decoded no usable audio from {path}")
    return as_audio(samples.reshape(-1, chans), rate)


# ---------------------------------------------------------------------------
# the defect budget — shipped, and here is why each number is what it is
# ---------------------------------------------------------------------------

class Defect(msgspec.Struct, frozen=True, kw_only=True):
    """One absolute, content-independent audio fault, with its evidence."""

    metric: str
    low: float | None = None
    high: float | None = None
    provenance: str = ""

    def breached(self, value: float) -> bool:
        if self.low is not None and value < self.low:
            return True
        return self.high is not None and value > self.high

    def describe(self) -> str:
        if self.low is not None and self.high is not None:
            return f"{self.low:g} <= {self.metric} <= {self.high:g}"
        if self.low is not None:
            return f"{self.metric} >= {self.low:g}"
        return f"{self.metric} <= {self.high:g}"


#: Calibrated on the 18-clip fal MiniMax-H3 corpus (2026-08-07, 1344x768 @24fps,
#: 32 kHz stereo AAC — ie#612's billed benchmark run), which is a KNOWN-GOOD
#: population: real product output from the model we are trying to beat. Each
#: limit is set outside the whole observed range with the margin recorded.
AUDIO_DEFECTS: tuple[Defect, ...] = (
    Defect(metric="audio_rms_dbfs", low=-50.0, provenance=(
        "programme level. fal H3 n=18 spans -33.3 to -18.7 dBFS; -50 sits 17 dB "
        "under the quietest real clip and catches an arm whose soundtrack "
        "collapsed to near-nothing without being fully digital-silent."
    )),
    Defect(metric="audio_peak_dbfs", high=-0.1, provenance=(
        "fal H3 n=18 peaks span -14.7 to -0.45 dBFS — none reaches full scale. "
        "A peak at or above -0.1 dBFS means the render is pinned to the rail."
    )),
    Defect(metric="audio_clip_fraction", high=0.001, provenance=(
        "fal H3 n=18: ZERO clipped samples in every clip. 0.1% is a tolerance "
        "for lossy-codec overshoot, not a licence to clip."
    )),
    Defect(metric="audio_silence_fraction", high=0.50, provenance=(
        "fraction of 50 ms blocks under -60 dBFS. fal H3 n=18 max 0.05. Half the "
        "clip silent is a generator that stopped, not a quiet passage; the limit "
        "is deliberately 10x the worst observed so a legitimately sparse "
        "soundtrack is not failed."
    )),
    Defect(metric="audio_dc_offset", high=0.01, provenance=(
        "fal H3 n=18 max 0.0002. 0.01 (=-40 dBFS of DC) is a broken decode or a "
        "broken vocoder, never a creative choice."
    )),
    Defect(metric="audio_stereo_separation_db", high=60.0, provenance=(
        "DUAL-MONO GUARD, stereo arms only. programme RMS minus side ((L-R)/2) "
        "RMS. fal H3 n=18 spans 2.5 to 20.0 dB — genuinely stereo. Collapsing "
        "L and R to identical samples drives the side signal to the -120 dBFS "
        "floor and the separation to ~99 dB, so 60 dB separates 'narrow stereo' "
        "from 'not stereo at all' with ~40 dB of margin on both sides."
    )),
)


# ---------------------------------------------------------------------------
# semantic tier: the audio checklist
# ---------------------------------------------------------------------------

class AudioChecklist(msgspec.Struct, frozen=True, kw_only=True):
    """Authored audio expectations for one prompt, in two groups.

    ``speech`` items reuse ``kind="ocr"`` deliberately: reading a required
    string out of recognised text is the SAME operation whether the text came
    from pixels or from samples, so the fuzzy matcher, the pass threshold and
    the ``text_exact``/``text_fuzzy`` rates are shared rather than duplicated.
    ``events`` items are ``kind="vqa"`` and go to an :class:`AudioJudge`.
    """

    prompt_id: str
    speech: tuple[ChecklistItem, ...] = ()
    events: tuple[ChecklistItem, ...] = ()


class AudioAdherence(msgspec.Struct, kw_only=True):
    measured: bool = False
    event_recall: float = 0.0
    speech_exact: float = 0.0
    speech_fuzzy: float = 0.0
    items: list[ItemVerdict] = msgspec.field(default_factory=list)
    note: str = ""


_AUDIO_PREAMBLE = (
    "You are grading whether an audio generator followed its instructions. You "
    "are given the generated audio track of one video. Judge only what you can "
    "HEAR — do not assume a sound is present because the prompt asked for it. "
    "Answer each numbered question with yes or no. Reply with ONLY a JSON array "
    'like [{{"n": 1, "answer": "yes"}}, {{"n": 2, "answer": "no"}}] — one object '
    "per question, no other text."
)


def build_audio_judge_prompt(items: tuple[ChecklistItem, ...]) -> str:
    lines = [_AUDIO_PREAMBLE, "Questions:"]
    lines += [f"{i}. {item.question}" for i, item in enumerate(items, start=1)]
    return "\n".join(lines)


def score_audio_adherence(
    checklist: AudioChecklist, clip: Audio, *,
    transcriber: Transcriber | None = None, audio_judge: AudioJudge | None = None,
) -> AudioAdherence:
    """Speech read-back + sound-event judging for one clip."""
    verdicts: list[ItemVerdict] = []
    missing: list[str] = []

    if checklist.speech:
        if transcriber is None:
            missing.append(f"{len(checklist.speech)} speech items (no transcriber)")
        else:
            transcript = transcriber.transcribe(clip.samples, clip.sample_rate)
            for item in checklist.speech:
                exact, fuzzy = text_match(item.text, transcript)
                present = exact or fuzzy >= OCR_FUZZY_PASS
                verified = (not present) if item.absent else present
                verdicts.append(ItemVerdict(
                    id=item.id, kind=KIND_OCR, verified=verified,
                    score=(1.0 - fuzzy) if item.absent else fuzzy,
                    detail=f"{'absent-required ' if item.absent else ''}fuzzy {fuzzy:.2f}",
                ))

    if checklist.events:
        if audio_judge is None:
            missing.append(f"{len(checklist.events)} sound-event items (no audio judge)")
        else:
            prompt = build_audio_judge_prompt(checklist.events)
            answers = parse_judge_answers(
                audio_judge.ask(clip.samples, clip.sample_rate, prompt),
                len(checklist.events),
            )
            for i, item in enumerate(checklist.events, start=1):
                answered = answers.get(i)
                yes = bool(answered)
                verified = (not yes) if item.absent else yes
                verdicts.append(ItemVerdict(
                    id=item.id, kind=KIND_VQA, verified=verified,
                    score=1.0 if verified else 0.0,
                    detail="unanswered" if answered is None else ("yes" if yes else "no"),
                ))

    if not verdicts:
        return AudioAdherence(measured=False, note="; ".join(missing) or "empty checklist")
    scored_events = tuple(i for i in checklist.events if any(v.id == i.id for v in verdicts))
    scored_speech = tuple(i for i in checklist.speech if any(v.id == i.id for v in verdicts))
    exact, fuzzy = adherence._text_rates(scored_speech, verdicts)
    return AudioAdherence(
        measured=True,
        event_recall=adherence._recall(scored_events, verdicts) if scored_events else 0.0,
        speech_exact=exact, speech_fuzzy=fuzzy, items=verdicts,
        note=("partial: " + "; ".join(missing)) if missing else "",
    )


# ---------------------------------------------------------------------------
# THE verdict
# ---------------------------------------------------------------------------

class AudioVerdict(msgspec.Struct, kw_only=True):
    """One clip's audio answer. What every H3 quality gate reads.

    ``measured`` and ``unmeasured`` are disjoint and together account for every
    metric this function knows how to produce — so "did anyone look at the
    stereo image" is answerable from the result alone.
    """

    verdict: str = UNMEASURED           # pass | reject | unmeasured
    defects: list[str] = msgspec.field(default_factory=list)
    measured: dict[str, float] = msgspec.field(default_factory=dict)
    unmeasured: dict[str, str] = msgspec.field(default_factory=dict)
    fidelity_reasons: list[str] = msgspec.field(default_factory=list)
    sync: avsync.SyncEstimate | None = None
    reference_sync: avsync.SyncEstimate | None = None
    adherence: AudioAdherence | None = None
    notes: list[str] = msgspec.field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        head = f"audio {self.verdict.upper()}"
        if self.defects:
            head += " — " + "; ".join(self.defects)
        elif self.fidelity_reasons:
            head += " — " + "; ".join(self.fidelity_reasons)
        return f"{head} ({len(self.measured)} measured, {len(self.unmeasured)} unmeasured)"


def audio_verdict(
    candidate: Any,
    reference: Any = None,
    *,
    sample_rate: int = 0,
    frames: Any = None,
    reference_frames: Any = None,
    fps: float = 0.0,
    checklist: AudioChecklist | None = None,
    transcriber: Transcriber | None = None,
    audio_judge: AudioJudge | None = None,
    defects: tuple[Defect, ...] = AUDIO_DEFECTS,
    fidelity_budget: tuple[Any, ...] = (),
) -> AudioVerdict:
    """Score one clip's audio and return a single verdict.

    ``candidate`` / ``reference`` are :class:`Audio`, arrays with
    ``sample_rate``, or paths (decoded through ffmpeg). ``frames`` +
    ``fps`` enable AV-sync; supplying ``reference_frames`` too makes the sync
    read a paired DRIFT, which is the number a regression gate should trust —
    absolute offset is confounded by content, drift is not.

    ``fidelity_budget`` is a
    :data:`cozy_eval.verdict.Budget` of
    :class:`~cozy_eval.verdict.Threshold`, calibrated by the caller against the
    family being gated, exactly like every other paired budget here.
    """
    t0 = time.monotonic()
    result = AudioVerdict()
    cand = _coerce(candidate, sample_rate, "candidate")
    ref = _coerce(reference, sample_rate, "reference") if reference is not None else None

    # --- tier 1: reference-free signal statistics + the absolute defects ----
    stats = audio_metrics.signal_stats(cand)
    result.measured.update(stats)
    if cand.channels < 2:
        result.unmeasured["audio_stereo_separation_db"] = (
            "mono source: there is no stereo image to score. This is only a "
            "defect if the model claims stereo — check the arm, not the metric."
        )
        result.unmeasured["audio_channel_correlation"] = "mono source"
    for defect in defects:
        value = stats.get(defect.metric)
        if value is None:
            continue
        if defect.breached(value):
            result.defects.append(
                f"{defect.metric} {value:.4g} breaches {defect.describe()}"
            )

    # --- tier 2: paired fidelity to the reference arm -----------------------
    if ref is None:
        for name in ("audio_si_sdr", "audio_snr_db", "audio_lsd_db", "audio_mel_l1"):
            result.unmeasured[name] = (
                "no reference arm: faithfulness is a PAIRED question. Render the "
                "bf16 anchor at the same prompt and seed to measure it."
            )
    else:
        paired = audio_metrics.paired_stats(ref, cand, align=True)
        result.measured.update(paired)
        result.fidelity_reasons = _fidelity_breaches(paired, fidelity_budget)
        if not fidelity_budget:
            result.notes.append(
                "paired audio fidelity was MEASURED but not GATED: no "
                "fidelity_budget was supplied, so these numbers cannot fail the "
                "arm. Calibrate one against this family before relying on it."
            )

    # --- tier 3: AV-sync ----------------------------------------------------
    if frames is None or fps <= 0:
        result.unmeasured["av_sync_offset_ms"] = (
            "AV-sync needs the picture too: pass frames=(T,H,W,3) and fps="
        )
    else:
        result.sync = avsync.sync_offset(frames, cand, fps=fps)
        result.notes.append(avsync.LIPSYNC_GAP_NOTE)
        if result.sync.measured:
            result.measured["av_sync_offset_ms"] = result.sync.offset_ms
            result.measured["av_sync_confidence"] = result.sync.confidence
        else:
            result.unmeasured["av_sync_offset_ms"] = result.sync.note
        if reference_frames is not None and ref is not None:
            result.reference_sync = avsync.sync_offset(reference_frames, ref, fps=fps)
            drift = avsync.sync_drift(result.reference_sync, result.sync)
            if drift is None:
                result.unmeasured["av_sync_drift_ms"] = (
                    "sync drift needs BOTH arms measured; at least one arm's "
                    "content carries no alignable onsets"
                )
            else:
                # MAGNITUDE: the registry direction has to mean something, and
                # "the sync moved 80 ms" is the fault whichever way it went. The
                # signed value stays available on .sync / .reference_sync.
                result.measured["av_sync_drift_ms"] = abs(drift)
        else:
            result.unmeasured["av_sync_drift_ms"] = (
                "no reference arm frames: drift is the paired read of sync and "
                "is the one a regression gate should trust"
            )

    # --- tier 4: semantic ---------------------------------------------------
    if checklist is None:
        result.unmeasured["audio_event_recall"] = (
            "no audio checklist for this prompt: nothing authored to check the "
            "soundtrack against"
        )
    else:
        scored = score_audio_adherence(
            checklist, cand, transcriber=transcriber, audio_judge=audio_judge,
        )
        result.adherence = scored
        if scored.measured:
            if any(v.kind == KIND_VQA for v in scored.items):
                result.measured["audio_event_recall"] = scored.event_recall
            if any(v.kind == KIND_OCR for v in scored.items):
                result.measured["audio_speech_exact"] = scored.speech_exact
                result.measured["audio_speech_fuzzy"] = scored.speech_fuzzy
        else:
            result.unmeasured["audio_event_recall"] = (
                f"audio adherence UNMEASURED: {scored.note}"
            )

    # --- the tri-state ------------------------------------------------------
    if result.defects or result.fidelity_reasons:
        result.verdict = REJECT
    elif result.measured:
        result.verdict = PASS
    else:
        result.verdict = UNMEASURED
        result.notes.append(
            "audio UNMEASURED: nothing could be scored on this input, so this is "
            "NOT a pass"
        )
    result.seconds = round(time.monotonic() - t0, 3)
    return result


def _coerce(source: Any, sample_rate: int, role: str) -> Audio:
    if isinstance(source, (str, Path)):
        return read_audio(source)
    try:
        return as_audio(source, sample_rate)
    except ConfigError as exc:
        raise ConfigError(f"{role} audio: {exc}") from exc


def _fidelity_breaches(paired: dict[str, float], budget: tuple[Any, ...]) -> list[str]:
    out: list[str] = []
    for threshold in budget:
        spec = registry.BY_NAME.get(threshold.metric)
        value = paired.get(threshold.metric)
        if spec is None or value is None:
            continue
        hib = spec.higher_is_better
        for limit, label in (
            (getattr(threshold, "mean_limit", None), "budget"),
            (getattr(threshold, "tail_limit", None), "tail budget"),
        ):
            if limit is None:
                continue
            if (value < limit) if hib else (value > limit):
                out.append(f"{threshold.metric} {value:.4g} vs {label} {limit:.4g}")
                break
    return out


# ---------------------------------------------------------------------------
# batch runner
# ---------------------------------------------------------------------------

class AudioSample(msgspec.Struct, kw_only=True):
    """One audio eval row. ``checklist_id`` selects the authored audio
    checklist; empty means the row is not scored for audio adherence."""

    prompt: str = ""
    seed: int = 0
    checklist_id: str = ""


def run_audio(
    samples: list[AudioSample],
    candidates: list[Any],
    *,
    references: list[Any] | None = None,
    sample_rate: int = 0,
    frames: list[Any] | None = None,
    reference_frames: list[Any] | None = None,
    fps: float = 0.0,
    checklists: dict[str, AudioChecklist] | None = None,
    transcriber: Transcriber | None = None,
    audio_judge: AudioJudge | None = None,
    fidelity_budget: tuple[Any, ...] = (),
) -> suite.SuiteReport:
    """Score a batch of soundtracks into the standard report schema.

    Mode is ``audio-paired`` / ``audio-reference-free``. Defects are recorded in
    the report's ``notes`` naming the row, because a defect is a fact about one
    render and a dimension summary would average it away.
    """
    if len(samples) != len(candidates):
        raise ConfigError(
            f"got {len(samples)} samples but {len(candidates)} candidate clips; "
            "each sample is scored against its own render, so the lists must match"
        )
    paired = references is not None
    if paired and len(references) != len(samples):
        raise ConfigError(
            f"got {len(samples)} samples but {len(references)} reference clips; "
            "a paired run needs one reference per sample"
        )
    report = suite.SuiteReport(
        mode="audio-paired" if paired else "audio-reference-free",
        samples=len(samples),
        library_version=suite._library_version(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    rows = [
        suite.SampleRow(
            index=i, checklist_id=s.checklist_id, prompt=s.prompt[:200], seed=s.seed,
        )
        for i, s in enumerate(samples)
    ]
    t0 = time.monotonic()
    unmeasured_reasons: dict[str, str] = {}
    for row, sample, cand in zip(rows, samples, candidates, strict=True):
        verdict = audio_verdict(
            cand,
            references[row.index] if paired else None,
            sample_rate=sample_rate,
            frames=frames[row.index] if frames else None,
            reference_frames=reference_frames[row.index] if reference_frames else None,
            fps=fps,
            checklist=(checklists or {}).get(sample.checklist_id),
            transcriber=transcriber,
            audio_judge=audio_judge,
            fidelity_budget=fidelity_budget,
        )
        row.values.update(verdict.measured)
        row.failed_items += [
            v.id for v in (verdict.adherence.items if verdict.adherence else [])
            if not v.verified
        ]
        for defect in verdict.defects:
            report.notes.append(f"sample {row.index:02d} DEFECT: {defect}")
        unmeasured_reasons.update(verdict.unmeasured)
    report.seconds["audio"] = round(time.monotonic() - t0, 2)
    report.seconds["total"] = report.seconds["audio"]
    report.models["audio"] = audio_metrics.AUDIO_LIBRARY
    if frames is not None and fps > 0:
        report.models["av_sync"] = avsync.AVSYNC_METHOD
        report.notes.append(avsync.LIPSYNC_GAP_NOTE)
    if transcriber is not None:
        report.models["transcriber"] = getattr(transcriber, "model_ref", "")
    if audio_judge is not None:
        report.models["audio_judge"] = getattr(audio_judge, "model_ref", "")

    report.rows = rows
    for name in sorted({k for row in rows for k in row.values}):
        if name in registry.BY_NAME:
            summary = suite.summarize(rows, name)
            if summary.measured:
                report.dimensions.append(summary)
    measured_names = {d.metric for d in report.dimensions}
    for name, reason in sorted(unmeasured_reasons.items()):
        if name not in measured_names:
            report.notes.append(f"{name} UNMEASURED: {reason}")
    return report


__all__ = [
    "AUDIO_DEFECTS",
    "PASS",
    "REJECT",
    "UNMEASURED",
    "AudioAdherence",
    "AudioChecklist",
    "AudioSample",
    "AudioVerdict",
    "Defect",
    "audio_verdict",
    "build_audio_judge_prompt",
    "probe_audio",
    "read_audio",
    "run_audio",
    "score_audio_adherence",
]
