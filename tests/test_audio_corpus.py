"""The audio axis against REAL generated media, not synthetic waveforms.

Corpus: the 18 billed fal MiniMax-H3 generations from ie#612's benchmark run
(2026-08-07) — 1344x768 @24 fps, 32 kHz stereo AAC, t2va/i2va/ref2va at 5/10/15 s.
It is a KNOWN-GOOD population: real product output from the model we are trying
to beat, which is what makes it usable to calibrate an absolute defect budget.

THE SANITY ANCHOR. ie#612 measured one of these clips with ffmpeg's own
``volumedetect`` and banked three numbers in the tracker: programme -20.8 dB,
peak -5.6 dB, and a side signal at -29.2 dB, "8.4 dB below programme". This
module's numpy path has to reproduce all three, or it is not measuring what it
claims to. That check is :func:`test_reproduces_the_banked_stereo_anchor`.

Skipped when the corpus is not on disk — it is 140 MB of billed generations, not
a repo fixture.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from cozy_eval import audio as audio_mode
from cozy_eval.audio import AUDIO_DEFECTS, audio_verdict, probe_audio, read_audio
from cozy_eval.metrics import audio as A
from cozy_eval.metrics import avsync as S

CORPUS = Path.home() / "cozy/samples/fal-h3-bench-20260807/out"

#: The clip ie#612 quoted, and the three numbers it quoted for it.
ANCHOR = "t2va-5s-r0.mp4"
ANCHOR_RMS_DBFS = -20.8
ANCHOR_PEAK_DBFS = -5.6
ANCHOR_SIDE_DBFS = -29.2
ANCHOR_SEPARATION_DB = 8.4

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(
        not CORPUS.is_dir() or not list(CORPUS.glob("*.mp4")),
        reason=f"fal MiniMax-H3 sample corpus not on disk at {CORPUS}",
    ),
]


def _clips() -> list[Path]:
    return sorted(CORPUS.glob("*.mp4"))


@pytest.fixture(scope="module")
def anchor() -> A.Audio:
    return read_audio(CORPUS / ANCHOR)


def _frames(path: Path, scale: int = 8) -> np.ndarray:
    """Decode to a small RGB array — AV-sync reads motion energy, and full
    resolution buys nothing while costing a lot of RAM per clip."""
    width, height = 1344 // scale, 768 // scale
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-vf", f"scale={width}:{height}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# THE ANCHOR
# ---------------------------------------------------------------------------

def test_reproduces_the_banked_stereo_anchor(anchor: A.Audio) -> None:
    """ie#612 banked -20.8 / -5.6 / -29.2 dB for this clip from ffmpeg
    volumedetect. If our numbers drift from those, the instrument is wrong."""
    stats = A.signal_stats(anchor)
    assert stats["audio_rms_dbfs"] == pytest.approx(ANCHOR_RMS_DBFS, abs=0.1)
    assert stats["audio_peak_dbfs"] == pytest.approx(ANCHOR_PEAK_DBFS, abs=0.1)
    assert stats["audio_side_dbfs"] == pytest.approx(ANCHOR_SIDE_DBFS, abs=0.1)
    assert stats["audio_stereo_separation_db"] == pytest.approx(ANCHOR_SEPARATION_DB, abs=0.1)


def test_the_anchor_clip_is_32khz_stereo_as_the_model_card_claims(anchor: A.Audio) -> None:
    rate, channels, _ = probe_audio(CORPUS / ANCHOR)
    assert (rate, channels) == (32000, 2)
    assert anchor.sample_rate == 32000
    assert anchor.channels == 2


# ---------------------------------------------------------------------------
# the corpus as a known-good population
# ---------------------------------------------------------------------------

def test_every_clip_has_an_audio_stream() -> None:
    """For a model that generates audio jointly, a missing track is the loudest
    possible defect — and ie#612 recorded that all 18 carry one."""
    for clip in _clips():
        rate, channels, duration = probe_audio(clip)
        assert (rate, channels) == (32000, 2), clip.name
        assert duration > 4.0, clip.name


def test_the_whole_corpus_passes_the_shipped_defect_budget() -> None:
    """The budget is calibrated ON this population, so this is a consistency
    check rather than a discovery: a budget that fails its own known-good
    population is mis-set."""
    for clip in _clips():
        result = audio_verdict(read_audio(clip))
        assert result.verdict == audio_mode.PASS, f"{clip.name}: {result.defects}"


def test_the_corpus_is_genuinely_stereo_with_margin_to_the_dual_mono_limit() -> None:
    limit = next(d for d in AUDIO_DEFECTS if d.metric == "audio_stereo_separation_db")
    separations = {}
    for clip in _clips():
        stats = A.signal_stats(read_audio(clip))
        separations[clip.name] = stats["audio_stereo_separation_db"]
    assert max(separations.values()) < limit.high - 30.0, separations
    assert min(separations.values()) > 0.0, separations


# ---------------------------------------------------------------------------
# RED ARMS built from real media
# ---------------------------------------------------------------------------

def test_collapsing_a_real_clip_to_dual_mono_is_rejected(anchor: A.Audio) -> None:
    collapsed = A.as_audio(np.repeat(anchor.mono[:, None], 2, axis=1), anchor.sample_rate)
    result = audio_verdict(collapsed)
    assert result.verdict == audio_mode.REJECT
    assert any("audio_stereo_separation_db" in d for d in result.defects)


def test_muting_a_real_clip_is_rejected(anchor: A.Audio) -> None:
    muted = A.as_audio(np.zeros_like(anchor.samples), anchor.sample_rate)
    assert audio_verdict(muted).verdict == audio_mode.REJECT


def test_a_degraded_arm_falls_against_the_real_reference(anchor: A.Audio) -> None:
    """The faithfulness question on real media: a same-length degraded arm has
    to lose SI-SDR monotonically as the damage grows."""
    rng = np.random.default_rng(11)
    previous = A.SNR_CAP
    for sigma in (0.002, 0.01, 0.05):
        arm = A.as_audio(
            anchor.samples + rng.normal(0, sigma, anchor.samples.shape).astype(np.float32),
            anchor.sample_rate,
        )
        value = A.paired_stats(anchor, arm)["audio_si_sdr"]
        assert value < previous, sigma
        previous = value
    assert previous < 20.0


def test_a_desynced_real_clip_shows_up_as_drift_when_sync_is_measurable() -> None:
    """Guarded by the same honesty rule as everything else: if this content has
    no alignable onsets, the drift is UNMEASURED and the test asserts THAT,
    because a fabricated drift would be worse than an admitted gap."""
    clip = CORPUS / ANCHOR
    frames, track = _frames(clip), read_audio(clip)
    shifted = S.shift_audio(track, milliseconds=200.0)
    result = audio_verdict(
        shifted, track, frames=frames, reference_frames=frames, fps=24.0,
    )
    if "av_sync_drift_ms" in result.measured:
        assert result.measured["av_sync_drift_ms"] == pytest.approx(200.0, abs=40.0)
    else:
        assert "carries no alignable onsets" in result.unmeasured["av_sync_drift_ms"]


# ---------------------------------------------------------------------------
# the finding this corpus actually produced
# ---------------------------------------------------------------------------

def test_event_sync_is_honestly_unmeasurable_on_this_corpus() -> None:
    """A MEASURED FINDING, asserted so it cannot rot silently.

    None of the 18 fal H3 clips carries hard audio transients — the generated
    soundtracks are continuous ambience and score-like beds (onset-envelope skew
    0.2-1.7, against 4-14 for their own picture). Onset correlation therefore
    has nothing to align, and every clip returns UNMEASURED with a reason rather
    than an offset read off noise.

    The consequence for the H3 program, and the reason this is a test and not a
    comment: **AV-sync cannot be gated on general prompts.** It needs a prompt
    subset authored to contain hard synchronised events — a door slam, a clap,
    footsteps, dialogue — and until that subset exists, ``av_sync_drift_ms``
    will report UNMEASURED on the H3 lanes. If a future model starts emitting
    transients, this test flips and should be re-read, not deleted.
    """
    measured, unmeasured = [], []
    for clip in _clips()[:6]:
        estimate = S.sync_offset(_frames(clip), read_audio(clip), fps=24.0)
        (measured if estimate.measured else unmeasured).append(clip.name)
        if not estimate.measured:
            assert estimate.note.startswith("AV-sync UNMEASURED"), clip.name
    assert not measured, (
        "fal H3 clips became event-sync measurable — re-read the finding in this "
        f"test's docstring rather than deleting it: {measured}"
    )
