"""Fine-detail fidelity: the numeric detectors, the VLM rubric, the verdict.

Integration style, no mocks except the pod-side judge VLM (stubbed exactly as
the video tests stub it). The load-bearing fixtures are SYNTHETIC: a clean step
edge vs the SAME edge with a ringing overshoot added — the reference-free number
cannot separate them (ringing looks like sharpness) but the paired
``ringing_excess`` must, which is the whole reason the paired form exists.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cozy_eval import detail, registry
from cozy_eval.metrics import detail as detail_metrics

# ---------------------------------------------------------------------------
# synthetic frames: a clean edge and the same edge with ringing added
# ---------------------------------------------------------------------------

def _edge(size: int = 96, seed: int = 3) -> np.ndarray:
    """A textured scene with several mid-contrast bars — many edges, plausible
    plateaus, like a real frame's structure. RGB float in [0, 1]."""
    rng = np.random.default_rng(seed)
    lum = np.full((size, size), 0.5, np.float32)
    for i in range(1, 6):
        c = i * size // 6
        lum[:, c:c + 4] += 0.3 if i % 2 else -0.3     # alternating bars
    lum += rng.normal(0, 0.02, lum.shape).astype(np.float32)   # light texture
    return np.repeat(np.clip(lum, 0, 1)[..., None], 3, axis=2)


def _ringing(size: int = 96, seed: int = 3) -> np.ndarray:
    """The same scene run through an UNSHARP MASK — the canonical ringing
    generator: it adds a bright/dark overshoot rim at every edge."""
    from scipy.ndimage import gaussian_filter

    frame = _edge(size, seed)
    lum = frame[..., 0]
    sharp = np.clip(lum + 1.5 * (lum - gaussian_filter(lum, 1.5)), 0, 1)
    return np.repeat(sharp[..., None], 3, axis=2)


def test_ringing_excess_separates_halo_from_clean_edge() -> None:
    clean, rung = _edge(), _ringing()
    # reference-free overshoot: the rung frame reads higher, but the number is
    # scene-confounded so it is only report-only.
    assert detail_metrics.edge_overshoot(rung) > detail_metrics.edge_overshoot(clean)
    # the PAIRED form is the validated one: candidate=rung against reference=clean
    # is a clear positive; the reverse is a clear negative; identical is ~0.
    assert detail_metrics.ringing_excess(clean, rung) > 0.02
    assert detail_metrics.ringing_excess(rung, clean) < -0.02
    assert abs(detail_metrics.ringing_excess(clean, clean)) < 1e-6


def test_edge_overshoot_flat_frame_is_zero() -> None:
    flat = np.full((64, 64, 3), 0.5, np.float32)
    assert detail_metrics.edge_overshoot(flat) == 0.0


def test_text_legibility_unmeasured_without_reader(monkeypatch) -> None:
    # Force the "no OCR reader" branch — UNMEASURED, never a silent pass/zero.
    monkeypatch.setattr(detail_metrics, "text_legibility", detail_metrics.text_legibility)
    from cozy_eval.metrics import ocr as ocr_mod

    monkeypatch.setattr(ocr_mod, "available", lambda: False)
    res = detail_metrics.text_legibility(np.zeros((32, 32, 3), np.uint8))
    assert res.measured is False
    assert "OCR" in res.note


def test_dists_pair_identical_is_near_zero() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    rng = np.random.default_rng(0)
    img = rng.random((64, 64, 3)).astype(np.float32)
    d = detail_metrics.dists_pair(img, img)
    assert d < 0.05


# ---------------------------------------------------------------------------
# the VLM rubric
# ---------------------------------------------------------------------------

class RubricJudge:
    """Stub judge that answers 'no' for named axis questions (matched by a
    substring of the question), 'yes' otherwise. Records the call count so the
    ONE-call-per-clip contract is testable."""

    model_ref = "stub-detail-judge"

    def __init__(self, fail_substrings: set[str] = frozenset(), pairwise="A"):
        self.fail = set(fail_substrings)
        self.pairwise = pairwise
        self.calls: list[tuple[int, str]] = []

    def ask(self, images: list, prompt: str) -> str:
        self.calls.append((len(images), prompt))
        if "winner" in prompt:
            return json.dumps({"winner": self.pairwise})
        rows = []
        for line in prompt.splitlines():
            if line and line[0].isdigit() and ". " in line:
                n, text = line.split(". ", 1)
                ans = "no" if any(s in text for s in self.fail) else "yes"
                rows.append({"n": int(n), "answer": ans})
        return json.dumps(rows)


def _strip(n: int = 4) -> list[np.ndarray]:
    return [_edge() for _ in range(n)]


def test_detail_rubric_is_one_call_and_scores_four_axes() -> None:
    judge = RubricJudge()
    axes = detail.score_detail_vlm(_strip(), judge)
    assert len(judge.calls) == 1
    assert set(axes) == {name for name, _ in detail.DETAIL_AXES}
    assert all(v == 1.0 for v in axes.values())


def test_detail_rubric_marks_the_failing_axis() -> None:
    judge = RubricJudge(fail_substrings={"melted"})   # the face/hand question
    axes = detail.score_detail_vlm(_strip(), judge)
    assert axes["detail_face_coherent"] == 0.0
    assert axes["detail_text_legible"] == 1.0


def test_pairwise_parses_winner() -> None:
    assert detail.judge_detail_pairwise(_strip(), _strip(), RubricJudge(pairwise="A")) == 1.0
    assert detail.judge_detail_pairwise(_strip(), _strip(), RubricJudge(pairwise="B")) == -1.0
    assert detail.judge_detail_pairwise(_strip(), _strip(), RubricJudge(pairwise="tie")) == 0.0


# ---------------------------------------------------------------------------
# the verdict tri-state
# ---------------------------------------------------------------------------

def test_verdict_diagnostic_only_is_unmeasured_not_pass() -> None:
    # No judge, no reference, synthetic edge with no detectable text: only the
    # scene-confounded edge_overshoot is computable, which must NOT pass.
    v = detail.detail_verdict(_strip(), judge=None)
    assert v.verdict == detail.UNMEASURED
    assert "edge_overshoot" in v.measured
    assert v.verdict != detail.PASS


def test_verdict_rejects_on_vlm_defect() -> None:
    judge = RubricJudge(fail_substrings={"fake glyphs"})   # text axis fails
    v = detail.detail_verdict(_strip(), judge=judge)
    assert v.verdict == detail.REJECT
    assert any("detail_text_legible" in d for d in v.defects)
    assert v.measured["detail_score"] < 1.0


def test_verdict_passes_on_clean_vlm() -> None:
    v = detail.detail_verdict(_strip(), judge=RubricJudge())
    assert v.verdict == detail.PASS
    assert v.measured["detail_score"] == 1.0


def test_verdict_reference_tier_measures_dists_and_ringing() -> None:
    pytest.importorskip("torch")
    ref = [_edge() for _ in range(3)]
    cand = [_ringing() for _ in range(3)]
    v = detail.detail_verdict(cand, reference_strip=ref, judge=None)
    assert "dists" in v.measured
    assert "ringing_excess" in v.measured
    assert v.measured["ringing_excess"] > 0.0        # candidate added halos
    assert v.verdict == detail.PASS                   # a paired number is gating-capable


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------

def test_new_metrics_registered_at_v5() -> None:
    assert registry.METRIC_SET_VERSION == "cozy-eval/metrics@5"
    for name in ("dists", "ringing_excess", "edge_overshoot", "text_legibility",
                 "detail_score", "detail_text_legible", "detail_face_coherent",
                 "detail_edge_clean", "detail_texture_real", "detail_pref_delta"):
        assert name in registry.BY_NAME, name
    # headline invariant preserved — no new headline stolen
    for dim in registry.DIMENSIONS:
        assert len([m for m in registry.by_dimension(dim) if m.headline]) == 1


def test_detail_metrics_land_in_the_right_dimensions() -> None:
    assert registry.BY_NAME["dists"].dimension == registry.SIMILARITY
    assert registry.BY_NAME["ringing_excess"].dimension == registry.SIMILARITY
    assert registry.BY_NAME["edge_overshoot"].dimension == registry.QUALITY
    assert registry.BY_NAME["text_legibility"].dimension == registry.QUALITY
    assert registry.BY_NAME["detail_score"].dimension == registry.QUALITY
    assert registry.BY_NAME["detail_pref_delta"].dimension == registry.PREFERENCE
    # none of the detail numbers is gated on its own yet (report-only until banked)
    assert not registry.BY_NAME["dists"].gated
    assert not registry.BY_NAME["edge_overshoot"].gated
