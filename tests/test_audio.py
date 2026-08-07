"""Audio axis: the RED arms are the proof, not the green one.

A green report on good audio proves nothing — it is what the library already
produced when it could not hear at all. Every test here that matters builds a
DELIBERATELY BROKEN soundtrack and asserts the gate fails it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cozy_eval import audio as audio_mode
from cozy_eval import registry
from cozy_eval import verdict as verdict_mod
from cozy_eval.audio import AUDIO_DEFECTS, AudioChecklist, AudioSample, audio_verdict, run_audio
from cozy_eval.errors import ConfigError
from cozy_eval.metrics import audio as A
from cozy_eval.metrics import avsync as S
from cozy_eval.metrics.adherence import ChecklistItem

SR = 32000
DURATION = 4.0
FPS = 24.0


def _stereo(seed: int = 0, seconds: float = DURATION) -> A.Audio:
    """A plausible stereo programme: two decorrelated noise-shaped tones."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    left = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.normal(size=n)
    right = 0.2 * np.sin(2 * np.pi * 223 * t) + 0.02 * rng.normal(size=n)
    return A.as_audio(np.stack([left, right], axis=1).astype(np.float32), SR)


# ---------------------------------------------------------------------------
# ingest and shape discipline
# ---------------------------------------------------------------------------

def test_as_audio_promotes_mono_and_refuses_a_transposed_array() -> None:
    mono = A.as_audio(np.zeros(SR, np.float32) + 0.1, SR)
    assert mono.samples.shape == (SR, 1)
    assert mono.channels == 1
    with pytest.raises(ConfigError, match="transposed"):
        A.as_audio(np.zeros((2, SR), np.float32), SR)


def test_as_audio_requires_a_sample_rate() -> None:
    """A level, a loudness and a sync offset are all meaningless without it."""
    with pytest.raises(ConfigError, match="sample_rate is required"):
        A.as_audio(np.zeros((SR, 2), np.float32))


def test_float_input_is_not_silently_normalized() -> None:
    """A float array peaking at 0.3 is QUIET AUDIO, not audio needing a rescale.
    Guessing here would erase the level regression this module exists to catch."""
    quiet = A.as_audio((_stereo().samples * 0.01), SR)
    level = A.signal_stats(quiet)["audio_rms_dbfs"]
    assert level < -50.0                    # 40 dB under the un-scaled fixture
    assert level == pytest.approx(A.signal_stats(_stereo())["audio_rms_dbfs"] - 40.0, abs=0.1)


# ---------------------------------------------------------------------------
# RED ARM 1: the soundtrack went silent
# ---------------------------------------------------------------------------

def test_silent_soundtrack_is_rejected() -> None:
    silent = A.as_audio(np.zeros((int(DURATION * SR), 2), np.float32), SR)
    result = audio_verdict(silent)
    assert result.verdict == audio_mode.REJECT
    assert any("audio_silence_fraction" in d for d in result.defects)
    assert any("audio_rms_dbfs" in d for d in result.defects)


def test_half_silent_soundtrack_is_rejected_but_a_quiet_passage_is_not() -> None:
    good = _stereo().samples.copy()
    half = good.copy()
    half[: int(half.shape[0] * 0.6)] = 0.0
    assert audio_verdict(A.as_audio(half, SR)).verdict == audio_mode.REJECT
    sparse = good.copy()
    sparse[: sparse.shape[0] // 5] = 0.0          # 20% silence: legitimate
    assert audio_verdict(A.as_audio(sparse, SR)).verdict == audio_mode.PASS


# ---------------------------------------------------------------------------
# RED ARM 2: a stereo model emitted dual mono
# ---------------------------------------------------------------------------

def test_dual_mono_is_rejected_while_real_stereo_passes() -> None:
    real = _stereo()
    assert audio_verdict(real).verdict == audio_mode.PASS

    collapsed = A.as_audio(np.repeat(real.mono[:, None], 2, axis=1), SR)
    result = audio_verdict(collapsed)
    assert result.verdict == audio_mode.REJECT
    assert any("audio_stereo_separation_db" in d for d in result.defects)
    assert result.measured["audio_channel_correlation"] == pytest.approx(1.0)
    assert result.measured["audio_side_dbfs"] == A.DB_FLOOR


def test_mono_input_reports_the_stereo_axis_as_unmeasured_not_as_a_defect() -> None:
    """A mono source has no stereo image to be wrong about. That is a different
    fact from a stereo source that collapsed, and the report must say which."""
    mono = A.as_audio(_stereo().mono, SR)
    result = audio_verdict(mono)
    assert "audio_stereo_separation_db" not in result.measured
    assert "mono source" in result.unmeasured["audio_stereo_separation_db"]
    assert not any("stereo" in d for d in result.defects)


# ---------------------------------------------------------------------------
# RED ARM 3: the render is pinned to the rail
# ---------------------------------------------------------------------------

def test_clipping_is_rejected() -> None:
    hot = np.clip(_stereo().samples * 8.0, -1.0, 1.0)
    result = audio_verdict(A.as_audio(hot, SR))
    assert result.verdict == audio_mode.REJECT
    assert any("audio_clip_fraction" in d or "audio_peak_dbfs" in d for d in result.defects)


def test_dc_offset_is_rejected() -> None:
    broken = _stereo().samples + 0.05
    result = audio_verdict(A.as_audio(broken, SR))
    assert any("audio_dc_offset" in d for d in result.defects)


# ---------------------------------------------------------------------------
# paired fidelity — the faithfulness question
# ---------------------------------------------------------------------------

def test_identical_arms_cap_and_degradation_falls_monotonically() -> None:
    ref = _stereo()
    assert A.paired_stats(ref, ref)["audio_si_sdr"] == A.SNR_CAP

    rng = np.random.default_rng(1)
    previous = A.SNR_CAP
    for sigma in (0.001, 0.005, 0.02, 0.1):
        noisy = A.as_audio(
            ref.samples + rng.normal(0, sigma, ref.samples.shape).astype(np.float32), SR
        )
        value = A.paired_stats(ref, noisy)["audio_si_sdr"]
        assert value < previous, sigma
        previous = value


def test_si_sdr_ignores_a_pure_gain_change_and_snr_does_not() -> None:
    """A lane that only changed output GAIN did not damage the audio. That is
    the whole reason SI-SDR is the gated number and plain SNR is report-only."""
    ref = _stereo()
    louder = A.as_audio(ref.samples * 0.5, SR)
    paired = A.paired_stats(ref, louder)
    assert paired["audio_si_sdr"] == pytest.approx(A.SNR_CAP)
    assert paired["audio_snr_db"] < 10.0
    assert paired["audio_lufs_delta"] == pytest.approx(-6.02, abs=0.1)


def test_encoder_delay_is_compensated_within_the_bound_and_not_beyond_it() -> None:
    ref = _stereo()
    shifted = S.shift_audio(ref, milliseconds=2.0)          # AAC-priming scale
    paired = A.paired_stats(ref, shifted)
    assert paired["audio_align_lag_ms"] == pytest.approx(-2.0, abs=0.2)
    assert paired["audio_si_sdr"] > 20.0

    far = S.shift_audio(ref, milliseconds=300.0)            # a different take
    assert A.paired_stats(ref, far)["audio_si_sdr"] < 10.0


def test_paired_stats_refuse_a_sample_rate_mismatch() -> None:
    """Resampling one arm to score it against the other would put the
    resampler's own error inside the measurement."""
    ref = _stereo()
    other = A.Audio(samples=ref.samples, sample_rate=16000)
    with pytest.raises(ConfigError, match="same rate"):
        A.paired_stats(ref, other)


def test_a_fidelity_budget_can_fail_a_paired_arm() -> None:
    ref = _stereo()
    rng = np.random.default_rng(3)
    wrecked = A.as_audio(
        ref.samples + rng.normal(0, 0.08, ref.samples.shape).astype(np.float32), SR
    )
    budget = (verdict_mod.Threshold(metric="audio_si_sdr", mean_limit=15.0),)
    result = audio_verdict(wrecked, ref, fidelity_budget=budget)
    assert result.verdict == audio_mode.REJECT
    assert any("audio_si_sdr" in r for r in result.fidelity_reasons)


def test_measured_fidelity_without_a_budget_is_not_a_gate() -> None:
    ref = _stereo()
    result = audio_verdict(_stereo(seed=9), ref)
    assert result.verdict == audio_mode.PASS
    assert "audio_si_sdr" in result.measured
    assert any("not GATED" in n for n in result.notes)


# ---------------------------------------------------------------------------
# loudness
# ---------------------------------------------------------------------------

#: ITU-R BS.1770-4 Tables 1 and 2 — the recommendation's OWN 48 kHz coefficients.
#: Reproducing these is a stronger provenance claim than agreeing with any one
#: library, and it is what caught the RBJ-vs-bilinear filter-form bug: driven
#: with identical prototype parameters the cookbook high-shelf is ~0.2 dB off
#: around 1 kHz, worth 0.22 LU on tonal material.
BS1770_STAGE1_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
BS1770_STAGE1_A = (1.0, -1.69065929318241, 0.73248077421585)
BS1770_STAGE2_B = (1.0, -2.0, 1.0)
BS1770_STAGE2_A = (1.0, -1.99004745483398, 0.99007225036621)


def test_k_weighting_reproduces_the_standards_tabulated_48khz_coefficients() -> None:
    shelf_b, shelf_a = A._biquad_shelf(A._SHELF[0], A._SHELF[1], A._SHELF[2], 48000)
    assert shelf_b == pytest.approx(BS1770_STAGE1_B, abs=1e-11)
    assert shelf_a == pytest.approx(BS1770_STAGE1_A, abs=1e-11)
    hp_b, hp_a = A._biquad_highpass(A._HIGHPASS[1], A._HIGHPASS[2], 48000)
    assert hp_b == pytest.approx(BS1770_STAGE2_B, abs=1e-11)
    assert hp_a == pytest.approx(BS1770_STAGE2_A, abs=1e-11)


def test_loudness_matches_the_banked_pyloudnorm_oracle() -> None:
    """Banked by parity/run_loudness_oracle.py in a throwaway venv; pyloudnorm
    never enters the dependency tree. Regenerate with that script."""
    fixture = Path(__file__).parent / "fixtures" / "oracle_loudness.json"
    if not fixture.exists():
        pytest.skip("loudness oracle not banked")
    banked = json.loads(fixture.read_text())
    sys.path.insert(0, str(Path(__file__).parent.parent / "parity"))
    import run_loudness_oracle as oracle

    for key, expected in banked["values"].items():
        name, _, rate = key.rpartition("@")
        rate = int(rate)
        got = A.loudness_lufs(A.as_audio(oracle.signals(rate)[name], rate))
        assert got == pytest.approx(expected, abs=banked["tolerance_lu"]), key


def test_loudness_tracks_gain_and_floors_on_silence() -> None:
    ref = _stereo()
    base = A.loudness_lufs(ref)
    louder = A.loudness_lufs(A.as_audio(ref.samples * 2.0, SR))
    assert louder - base == pytest.approx(6.02, abs=0.1)
    assert A.loudness_lufs(A.as_audio(np.zeros((SR, 2), np.float32), SR)) == A.DB_FLOOR


# ---------------------------------------------------------------------------
# AV-sync: the positive control is what licenses the UNMEASURED verdicts
# ---------------------------------------------------------------------------

def _click_and_flash(offset_ms: float = 0.0, seed: int = 7, events: int = 10):
    """Hard audio-visual events at known times: a one-frame flash and a click."""
    rng = np.random.default_rng(seed)
    n_frames, n_samples = int(DURATION * FPS), int(DURATION * SR)
    times = np.sort(rng.uniform(0.5, DURATION - 0.6, events))
    frames = np.full((n_frames, 24, 24, 3), 0.15, np.float32)
    frames += rng.normal(0, 0.01, frames.shape).astype(np.float32)
    samples = np.zeros((n_samples, 2), np.float32)
    envelope = np.exp(-np.arange(800) / 120.0)
    click = (envelope * np.sin(2 * np.pi * 1200 * np.arange(800) / SR) * 0.6).astype(np.float32)
    for t in times:
        frame = round(t * FPS)
        if 0 <= frame < n_frames:
            frames[frame] += 0.7
        start = round((t + offset_ms / 1000.0) * SR)
        if 0 <= start < n_samples - click.size:
            samples[start:start + click.size, 0] += click
            samples[start:start + click.size, 1] += click
    return np.clip(frames, 0, 1), A.as_audio(np.clip(samples, -1, 1), SR)


@pytest.mark.parametrize("injected", [0.0, 40.0, -40.0, 120.0, -120.0])
def test_sync_offset_recovers_an_injected_offset(injected: float) -> None:
    """THE instrument validation. Without this, an UNMEASURED verdict elsewhere
    is indistinguishable from a broken estimator."""
    frames, clip = _click_and_flash(injected)
    estimate = S.sync_offset(frames, clip, fps=FPS)
    assert estimate.measured, estimate.note
    assert estimate.offset_ms == pytest.approx(injected, abs=25.0)


def test_uncorrelated_audio_is_unmeasured_rather_than_confidently_zero() -> None:
    frames, _ = _click_and_flash(seed=7)
    _, unrelated = _click_and_flash(seed=99)
    estimate = S.sync_offset(frames, unrelated, fps=FPS)
    assert not estimate.measured
    assert "no prominent correlation peak" in estimate.note


def test_silent_audio_and_static_video_each_name_their_own_reason() -> None:
    frames, clip = _click_and_flash()
    silent = A.as_audio(np.zeros_like(clip.samples), SR)
    assert "no onsets" in S.sync_offset(frames, silent, fps=FPS).note
    static = np.full_like(frames, 0.3)
    assert "no motion onsets" in S.sync_offset(static, clip, fps=FPS).note


def test_sync_drift_is_the_paired_read_and_needs_both_arms() -> None:
    ref_frames, ref_audio = _click_and_flash(0.0)
    cand_frames, cand_audio = _click_and_flash(150.0)
    result = audio_verdict(
        cand_audio, ref_audio, frames=cand_frames,
        reference_frames=ref_frames, fps=FPS,
    )
    assert result.measured["av_sync_drift_ms"] == pytest.approx(150.0, abs=30.0)

    _, unrelated = _click_and_flash(seed=99)
    blind = audio_verdict(
        unrelated, ref_audio, frames=cand_frames,
        reference_frames=ref_frames, fps=FPS,
    )
    assert "av_sync_drift_ms" not in blind.measured
    assert "at least one arm" in blind.unmeasured["av_sync_drift_ms"]


def test_the_lipsync_gap_travels_with_every_sync_number() -> None:
    frames, clip = _click_and_flash()
    result = audio_verdict(clip, frames=frames, fps=FPS)
    assert any("lip-sync" in n and "Synchformer" in n for n in result.notes)


# ---------------------------------------------------------------------------
# semantic tier
# ---------------------------------------------------------------------------

class _Transcriber:
    model_ref = "stub:whisper"

    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, samples, sample_rate) -> str:
        return self.text


class _AudioJudge:
    model_ref = "stub:audio-judge"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def ask(self, samples, sample_rate, prompt) -> str:
        self.prompt = prompt
        return self.reply


CHECKLIST = AudioChecklist(
    prompt_id="storm",
    speech=(ChecklistItem(id="line", kind="ocr", text="get inside now"),),
    events=(
        ChecklistItem(id="thunder", kind="vqa", question="Is there a thunderclap?"),
        ChecklistItem(id="rain", kind="vqa", question="Is there the sound of rain?"),
    ),
)


def test_speech_and_events_are_separate_instruments() -> None:
    result = audio_verdict(
        _stereo(), checklist=CHECKLIST,
        transcriber=_Transcriber("quick, get inside now, it's coming"),
        audio_judge=_AudioJudge('[{"n": 1, "answer": "yes"}, {"n": 2, "answer": "no"}]'),
    )
    assert result.measured["audio_speech_exact"] == pytest.approx(1.0)
    assert result.measured["audio_event_recall"] == pytest.approx(0.5)


def test_a_missing_transcriber_does_not_zero_the_speech_axis() -> None:
    """Half the instruments present must not read as half the items failed."""
    result = audio_verdict(
        _stereo(), checklist=CHECKLIST,
        audio_judge=_AudioJudge('[{"n": 1, "answer": "yes"}, {"n": 2, "answer": "yes"}]'),
    )
    assert "audio_speech_exact" not in result.measured
    assert result.measured["audio_event_recall"] == pytest.approx(1.0)
    assert "no transcriber" in (result.adherence.note if result.adherence else "")


def test_no_instruments_at_all_is_unmeasured_not_zero() -> None:
    result = audio_verdict(_stereo(), checklist=CHECKLIST)
    assert "audio_event_recall" not in result.measured
    assert "UNMEASURED" in result.unmeasured["audio_event_recall"]


# ---------------------------------------------------------------------------
# the verdict's own honesty rules
# ---------------------------------------------------------------------------

def test_every_unrunnable_axis_names_a_reason() -> None:
    result = audio_verdict(_stereo())
    assert result.unmeasured
    for name, reason in result.unmeasured.items():
        assert reason.strip(), name
        assert name not in result.measured, name


def test_a_run_that_measured_nothing_is_not_a_pass() -> None:
    """The failure this whole issue exists to prevent: a silent 'no audio
    measured' reading as a clean gate."""
    empty = audio_verdict(_stereo(), defects=())
    assert empty.verdict == audio_mode.PASS   # signal stats always run
    stripped = audio_mode.AudioVerdict()
    assert stripped.verdict == audio_mode.UNMEASURED


def test_every_defect_names_a_registered_metric_and_carries_provenance() -> None:
    for defect in AUDIO_DEFECTS:
        assert defect.metric in registry.BY_NAME, defect.metric
        assert len(defect.provenance) > 40, defect.metric


def test_run_audio_summarizes_into_the_standard_report() -> None:
    samples = [AudioSample(prompt="a", seed=1), AudioSample(prompt="b", seed=2)]
    report = run_audio(samples, [_stereo(0), _stereo(1)], references=[_stereo(2), _stereo(3)])
    assert report.mode == "audio-paired"
    assert report.summary("audio_si_sdr") is not None
    assert report.summary("audio_stereo_separation_db") is not None
    assert report.models["audio"] == A.AUDIO_LIBRARY


def test_run_audio_records_a_defect_against_its_own_row() -> None:
    silent = A.as_audio(np.zeros((int(DURATION * SR), 2), np.float32), SR)
    report = run_audio(
        [AudioSample(prompt="ok"), AudioSample(prompt="broken")], [_stereo(), silent],
    )
    assert any("sample 01 DEFECT" in n for n in report.notes)
    assert not any("sample 00 DEFECT" in n for n in report.notes)


def test_mismatched_list_lengths_are_an_error() -> None:
    with pytest.raises(ConfigError, match="must match"):
        run_audio([AudioSample(prompt="a")], [_stereo(), _stereo()])
