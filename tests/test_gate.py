"""End-to-end tests: real frames, real scoring, real verdicts. No mocks.

The synthetic clips here are procedural, not random noise: a textured scene with
smooth camera motion, which is the regime the calibrated thresholds live in. Each
test degrades that scene along ONE axis and asserts that the benchmark responsible
for that axis fires and the others do not — the property the whole design rests on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cozy_eval import (
    ChangeKind,
    ClipScore,
    Protocol,
    TrajectoryPerturbingError,
    Verdict,
    classify_change,
    measure_null_control,
    require_reference_lane,
    run_population_gate,
    run_reference_gate,
    score_clip,
)
from cozy_eval.protocol import Lane, ProtocolError

H, W, T = 96, 128, 24
PROMPTS = tuple(f"scene{i}" for i in range(8))


def clip(seed: int) -> np.ndarray:
    """A deterministic textured scene with smooth pan and a bit of local motion."""
    rng = np.random.default_rng(seed)
    tex = rng.random((H * 2, W * 2, 3)).astype(np.float32)
    # give the texture real spatial structure at several scales
    for _ in range(2):
        tex = (tex + np.roll(tex, 1, 0) + np.roll(tex, 1, 1)) / 3.0
    base = 0.25 + 0.5 * tex
    frames = np.empty((T, H, W, 3), np.float32)
    for t in range(T):
        dy, dx = t, (t * 2) % W
        frames[t] = base[dy:dy + H, dx:dx + W]
    return np.clip(frames, 0, 1)


def blur(frames: np.ndarray, alpha: float) -> np.ndarray:
    p = np.pad(frames, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    v = p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]
    b = (v[:, :, :-2] + 2 * v[:, :, 1:-1] + v[:, :, 2:]) / 16.0
    return (1 - alpha) * frames + alpha * b


def gain_jitter(frames: np.ndarray, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = 1.0 + amp * rng.standard_normal(len(frames)).astype(np.float32)
    return np.clip(frames * g[:, None, None, None], 0, 1)


def desaturate(frames: np.ndarray, keep: float) -> np.ndarray:
    lum = (frames @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    return np.clip(lum + keep * (frames - lum), 0, 1)


def proto(*changes: ChangeKind, n: int = 8, **over) -> Protocol:
    kw = dict(
        family="test", reference_arm="ref", candidate_arm="cand", changes=changes,
        width=W, height=H, frames=T, steps=8, seeds=(1,), prompts=PROMPTS[:n],
        execution_lane="compiled", hardware="cpu-test",
    )
    return Protocol(**{**kw, **over})


def score_pairs_of(transform) -> list:
    return [(score_clip(clip(i)), score_clip(transform(clip(i), i))) for i in range(8)]


def gate(transform, **kw) -> tuple:
    pairs = score_pairs_of(transform)
    r = run_population_gate(pairs, proto(ChangeKind.WEIGHT_QUANTIZATION), **kw)
    return r, {k: v.passed for k, v in r.benchmarks.items()}


def null_control(offset: int = 100, **over):
    """A zero-change arm: the same scene family drawn at another seed."""
    pairs = score_pairs_of(lambda f, i: clip(i + offset))
    return measure_null_control(pairs, proto(ChangeKind.SEED, **over))


# --------------------------------------------------------------------------- #
# the two-lane rule
# --------------------------------------------------------------------------- #

def test_reference_metrics_refused_for_trajectory_perturbing_change():
    with pytest.raises(TrajectoryPerturbingError, match="different take"):
        run_reference_gate([clip(0)], [clip(0)],
                           proto(ChangeKind.WEIGHT_QUANTIZATION, n=1))


def test_reference_metrics_allowed_for_post_latent_change():
    base = clip(0)
    report = run_reference_gate([base], [base], proto(ChangeKind.VAE_DECODE_DTYPE, n=1))
    assert report.lane is Lane.SAME_TRAJECTORY
    assert report.benchmarks["reference"].values["psnr_mean"] > 90


def test_one_perturbing_change_contaminates_a_mixed_set():
    assert classify_change(ChangeKind.VIDEO_ENCODER) is Lane.SAME_TRAJECTORY
    assert classify_change(ChangeKind.VIDEO_ENCODER,
                           ChangeKind.COMPILE) is Lane.POPULATION


def test_unclassified_change_is_an_error_not_a_default():
    with pytest.raises(ProtocolError, match="unclassified"):
        classify_change("something-new")


def test_population_gate_rejects_a_post_latent_protocol():
    pairs = [(score_clip(clip(i)), score_clip(clip(i))) for i in range(8)]
    with pytest.raises(ProtocolError, match="post-latent"):
        run_population_gate(pairs, proto(ChangeKind.VIDEO_ENCODER))


# --------------------------------------------------------------------------- #
# the three benchmarks are three different instruments
# --------------------------------------------------------------------------- #

def test_null_change_passes_every_benchmark():
    report, passed = gate(lambda f, i: f)
    assert report.verdict is Verdict.PASS, report.summary()
    assert passed == {"imaging": True, "temporal": True, "distributional": True}


def test_softening_is_caught_by_imaging():
    report, passed = gate(lambda f, i: blur(f, 0.5))
    assert report.verdict is Verdict.FAIL
    assert passed["imaging"] is False
    assert passed["temporal"] is True          # blur does not destabilise frames


def test_temporal_only_failure_is_invisible_to_the_imaging_benchmark():
    """Per-frame gain jitter leaves every per-frame statistic intact."""
    report, passed = gate(lambda f, i: gain_jitter(f, 0.03, i))
    assert passed["temporal"] is False, report.summary()
    assert passed["imaging"] is True
    assert report.verdict is Verdict.FAIL


def test_distributional_only_failure_is_invisible_to_the_other_two():
    """A small uniform desaturation moves no index past its budget, but the
    paired population test sees a consistent shift across all eight prompts."""
    report, passed = gate(lambda f, i: desaturate(f, 0.90))
    assert passed["imaging"] is True, report.summary()
    assert passed["temporal"] is True
    assert passed["distributional"] is False
    assert report.verdict is Verdict.FAIL


# --------------------------------------------------------------------------- #
# sample-size semantics
# --------------------------------------------------------------------------- #

def test_single_prompt_cannot_produce_a_pass():
    pairs = [(score_clip(clip(0)), score_clip(clip(0)))]
    report = run_population_gate(pairs, proto(ChangeKind.WEIGHT_QUANTIZATION, n=1))
    assert report.verdict is Verdict.INDETERMINATE
    assert report.benchmarks["distributional"].passed is None
    assert any("measures the take" in r for r in report.reasons)


def test_cross_pod_arms_cannot_produce_a_pass():
    pairs = [(score_clip(clip(i)), score_clip(clip(i))) for i in range(8)]
    p = proto(ChangeKind.WEIGHT_QUANTIZATION)
    p = Protocol(**{**{k: getattr(p, k) for k in p.__slots__}, "same_pod": False})
    assert run_population_gate(pairs, p).verdict is Verdict.INDETERMINATE


def test_every_result_carries_its_protocol():
    report, _ = gate(lambda f, i: f)
    stamp = report.to_dict()["protocol"]["stamp"]
    for token in ("128x96", "x24f", "8 steps", "n=8 prompts", "compiled both arms"):
        assert token in stamp


# --------------------------------------------------------------------------- #
# persisting a score (a consumer caches these between runs)
# --------------------------------------------------------------------------- #

def test_clip_scores_survive_a_json_round_trip_and_regate_identically(tmp_path):
    """The 2-D ``features`` array is the field a naive round-trip loses."""
    pairs = score_pairs_of(lambda f, i: blur(f, 0.5))
    restored = [tuple(ClipScore.from_dict(json.loads(json.dumps(s.to_dict())))
                      for s in pair) for pair in pairs]
    assert restored == pairs
    assert restored[0][0].features.shape == (T, 6)
    assert restored[0][0].features.dtype == np.float64

    path = tmp_path / "score.json"
    pairs[0][0].save(path)
    assert ClipScore.load(path) == pairs[0][0]

    before = run_population_gate(pairs, proto(ChangeKind.WEIGHT_QUANTIZATION))
    after = run_population_gate(restored, proto(ChangeKind.WEIGHT_QUANTIZATION))
    assert after.to_dict()["benchmarks"] == before.to_dict()["benchmarks"]


# --------------------------------------------------------------------------- #
# a degenerate arm carries no signal, and that is not a FAIL
# --------------------------------------------------------------------------- #

def test_a_degenerate_candidate_is_no_signal_not_a_confident_fail():
    """Uniformly black frames used to score imaging_index 0.0 and FAIL."""
    black = np.zeros((T, H, W, 3), np.float32)
    report = run_population_gate(
        [(score_clip(clip(i)), score_clip(black)) for i in range(8)],
        proto(ChangeKind.WEIGHT_QUANTIZATION))
    assert report.verdict is Verdict.DEGENERATE, report.summary()
    assert report.verdict is not Verdict.FAIL
    assert "imaging" not in report.benchmarks           # no number to rank on
    assert any("NO SIGNAL" in r for r in report.reasons)
    assert json.dumps(report.to_dict())                 # and it is still persistable


def test_a_degenerate_reference_makes_every_ratio_undefined():
    white = np.ones((T, H, W, 3), np.float32)
    report = run_population_gate(
        [(score_clip(white), score_clip(clip(i))) for i in range(8)],
        proto(ChangeKind.WEIGHT_QUANTIZATION))
    assert report.verdict is Verdict.DEGENERATE
    assert any("reference carries NO SIGNAL" in r for r in report.reasons)


# --------------------------------------------------------------------------- #
# API surface: one calling convention, one verdict vocabulary
# --------------------------------------------------------------------------- #

def test_a_packed_sequence_of_changes_says_so_instead_of_unclassified():
    changes = (ChangeKind.COMPILE, ChangeKind.SCHEDULER)
    with pytest.raises(TypeError, match="separate argument"):
        classify_change(changes)
    with pytest.raises(TypeError, match="separate argument"):
        require_reference_lane(changes)
    assert classify_change(*changes) is Lane.POPULATION


def test_benchmarks_and_reports_speak_one_verdict_vocabulary():
    report, _ = gate(lambda f, i: blur(f, 0.5))
    assert report.verdict is Verdict.FAIL
    assert report.benchmarks["imaging"].verdict is Verdict.FAIL
    assert report.benchmarks["temporal"].verdict is Verdict.PASS
    one_prompt = run_population_gate([(score_clip(clip(0)), score_clip(clip(0)))],
                                     proto(ChangeKind.WEIGHT_QUANTIZATION, n=1))
    assert one_prompt.benchmarks["distributional"].passed is None
    assert one_prompt.benchmarks["distributional"].verdict is Verdict.INDETERMINATE


# --------------------------------------------------------------------------- #
# the null control: which budgets is this family allowed to be judged on?
# --------------------------------------------------------------------------- #

def test_a_budget_the_zero_change_control_trips_cannot_fail_the_candidate():
    """The measured gatebackfill finding: a control trips pFQD at zero model change,
    so a candidate 'failing' it has been told nothing."""
    control = null_control()
    assert control.transfers["population_frechet"] is False
    assert control.transfers["imaging_index"] is True

    take_noise = lambda f, i: clip(i + 300)              # noqa: E731 — another draw
    uncontrolled, _ = gate(take_noise)
    assert uncontrolled.verdict is Verdict.FAIL          # what the library said before

    report, passed = gate(take_noise, null_control=control)
    assert report.verdict is Verdict.INDETERMINATE, report.summary()
    assert passed["distributional"] is True              # on its trusted budget only
    row = next(b for b in report.benchmarks["distributional"].budgets
               if "population_frechet" in b["budget"])
    assert row["passed"] is False and row["disregarded"] is True
    assert any("not transfer to this family" in r for r in report.reasons)
    assert report.to_dict()["null_control"]["untrusted"]


def test_a_budget_the_control_proves_transfers_still_fails_the_candidate():
    report, passed = gate(lambda f, i: desaturate(f, 0.90), null_control=null_control())
    assert report.verdict is Verdict.FAIL, report.summary()
    assert passed["distributional"] is False             # significant_features: control 0


def test_a_control_that_is_not_a_control_is_refused():
    pairs = score_pairs_of(lambda f, i: clip(i + 100))
    with pytest.raises(ProtocolError, match="same weights at a different seed"):
        measure_null_control(pairs, proto(ChangeKind.WEIGHT_QUANTIZATION))
    black = score_clip(np.zeros((T, H, W, 3), np.float32))
    with pytest.raises(ProtocolError, match="carry no signal"):
        measure_null_control([(black, black)] * 8, proto(ChangeKind.SEED))

    # A control rendered under other conditions does not describe this run.
    report, _ = gate(lambda f, i: f, null_control=null_control(steps=99))
    assert report.verdict is Verdict.INDETERMINATE
    assert any("not rendered under this run's conditions" in r for r in report.reasons)
