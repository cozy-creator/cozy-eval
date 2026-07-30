"""Aggregation is pure math and tested exactly. The SoftJudge itself needs a
real VLM, so its tests are `heavy`-marked and run on the smallest Apache-2.0
image-text-to-text checkpoint that behaves (SmolVLM-256M) — structure and
discrimination, not accuracy.

Registry wiring lives in `test_public_api.py`'s registry sweep.
"""

import math

import pytest

from cozy_eval.metrics.vqascore import AtomScore, aggregate


def atom(p, id="a", question="q"):
    return AtomScore(id=id, question=question, p=p)


@pytest.mark.parametrize("probs,measured,atom_mean,prompt_score", [
    pytest.param([], False, 0.0, 0.0, id="empty_is_unmeasured"),
    pytest.param([0.8], True, 0.8, 0.8, id="a_single_atom_is_both_means"),
    # The whole point of the geometric mean: it is NOT the arithmetic one.
    pytest.param([1.0, 0.01], True, 0.505, math.sqrt(0.01),
                 id="arithmetic_vs_geometric"),
    # One violated constraint sinks the prompt score while the mean shrugs.
    pytest.param([0.99, 0.99, 0.99, 0.001], True, 0.74275,
                 (0.99 ** 3 * 0.001) ** 0.25,
                 id="one_violated_atom_sinks_prompt_score_not_mean"),
])
def test_aggregate(probs, measured, atom_mean, prompt_score):
    """Soft-TIFA: a prompt is only as satisfied as its least-satisfied atom,
    so one wrong object cannot be averaged away."""
    out = aggregate([atom(p) for p in probs])
    assert out.measured is measured
    assert out.atom_mean == pytest.approx(atom_mean)
    assert out.prompt_score == pytest.approx(prompt_score, rel=1e-4)


def test_zero_probability_is_clamped():
    """log(0) must never raise, and a certain miss must never make the whole
    prompt score exactly zero — it is clamped, not special-cased."""
    out = aggregate([atom(0.0)])
    assert out.measured and out.prompt_score > 0.0


@pytest.mark.heavy
class TestSoftJudge:
    @pytest.fixture(scope="class")
    def judge(self):
        import torch

        from cozy_eval.metrics.vqascore import VlmSoftJudge

        # CPU prefills of SmolVLM run ~10 min/question on a shared box; take
        # the GPU when there is one.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return VlmSoftJudge("HuggingFaceTB/SmolVLM-256M-Instruct", device=device)

    def _solid(self, color):
        from PIL import Image

        return Image.new("RGB", (256, 256), color)

    def test_p_yes_is_a_probability_that_discriminates(self, judge):
        red = self._solid((220, 20, 20))
        p_red = judge.p_yes([red], "Is this image mostly red? Answer yes or no.")
        p_blue = judge.p_yes([red], "Is this image mostly blue? Answer yes or no.")
        assert 0.0 <= p_blue <= 1.0 and 0.0 <= p_red <= 1.0
        assert p_red > p_blue

    def test_vqascore_and_soft_tifa_run(self, judge):
        from cozy_eval.metrics.adherence import ChecklistItem
        from cozy_eval.metrics.vqascore import soft_tifa, vqascore

        img = self._solid((30, 30, 220))
        s = vqascore(judge, img, "a plain blue image")
        assert 0.0 <= s <= 1.0
        items = (
            ChecklistItem(id="a1", kind="vqa", question="Is the image blue?"),
            ChecklistItem(id="a2", kind="vqa", question="Is there a dog in the image?"),
        )
        out = soft_tifa(judge, img, items)
        assert out.measured and len(out.atoms) == 2
        assert judge.call_count >= 3
