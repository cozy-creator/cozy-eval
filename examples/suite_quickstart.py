"""Runnable on CPU in a few seconds, with no model downloads.

Shows the four things you need to use cozy_eval:

  1. the authored prompt set + its cross-validated checklists
  2. scoring adherence with a judge (stubbed here — a real one is a VLM)
  3. the edit change/preserve duality
  4. the tri-state verdict over a paired comparison

Run:  python examples/suite_quickstart.py
"""

from __future__ import annotations

from PIL import Image

from cozy_eval import promptset, registry, verdict
from cozy_eval.metrics import adherence, similarity

SET = "hard-eval-v1"


class StubJudge:
    """Stands in for a vision-language judge.

    ``cozy_eval.Judge`` is a Protocol, not a base class: anything with an
    ``ask(images, prompt) -> str`` method IS a judge. The real local one is
    ``adherence.VlmJudge("Qwen/Qwen3-VL-8B-Instruct")`` — Apache-2.0 weights,
    ~18 GB — but a hosted model behind HTTP, or this five-line stub, plug in
    exactly the same way.
    """

    model_ref = "stub"

    def __init__(self, answers: dict[int, str]) -> None:
        self.answers, self.call_count = answers, 0

    def ask(self, images: list, prompt: str) -> str:
        self.call_count += 1
        n = prompt.count("\n") and len(self.answers)
        return "[" + ", ".join(
            f'{{"n": {i}, "answer": "{self.answers.get(i, "yes")}"}}' for i in range(1, n + 1)
        ) + "]"


def main() -> None:
    # ---- 1. the benchmark is prompts AND the assertions about them ---------
    ps = promptset.load(SET)
    checklists = promptset.checklists_for(SET)  # raises if the two ever drift
    print(f"{ps.set_id}@{ps.version}: {len(ps.t2i)} prompts, {len(ps.edit)} edit triads")
    print(f"metric set: {registry.METRIC_SET_VERSION}, {len(registry.REGISTRY)} metrics")
    for dim in registry.DIMENSIONS:
        print(f"  {dim:<11} headline={registry.headline(dim).name}")

    checklist = checklists.t2i["t01"]
    print(f"\nt01 has {len(checklist.items)} authored items, e.g.:")
    for item in checklist.items[:3]:
        detail = item.text if item.kind == "ocr" else item.question
        print(f"  [{item.kind}] {detail[:70]}")

    # ---- 2. adherence, scored by a judge ----------------------------------
    # One call per image covering the WHOLE checklist, not one call per item.
    vqa_count = sum(1 for i in checklist.items if i.kind == "vqa")
    judge = StubJudge({n: "yes" for n in range(1, vqa_count + 1)})
    img = Image.new("RGB", (256, 256), "white")
    score = adherence.score_t2i(checklist, img, judge=judge, page_text=None)
    print(f"\nelement_recall = {score.element_recall:.2f} "
          f"(measured={score.measured}, judge calls={judge.call_count})")
    print(f"  note: {score.note}")

    # ---- 3. the edit duality ----------------------------------------------
    edit = checklists.edit["e01"]
    print(f"\nedit e01: {len(edit.change)} change items, {len(edit.preserve)} preserve items")
    print("  under-editing fails compliance; over-editing fails preservation")

    # ---- 4. similarity + the tri-state verdict ----------------------------
    a = Image.new("RGB", (128, 128), (120, 120, 120))
    b = Image.new("RGB", (128, 128), (122, 121, 119))
    print(f"\nssim(a,b) = {similarity.ssim(a, b):.4f}   psnr = {similarity.psnr(a, b):.1f} dB")

    budget = (
        verdict.Threshold(metric="lpips", mean_limit=0.35, tail_limit=0.60),
        verdict.Threshold(metric="pref_delta", mean_limit=-0.05, tail_limit=-0.30),
    )
    cases = {
        "close to the reference": [
            verdict.Measurement(metric="lpips", mean=0.20, worst=0.45),
            verdict.Measurement(metric="pref_delta", mean=0.01, worst=-0.02),
        ],
        "different, but does the job": [
            verdict.Measurement(metric="lpips", mean=0.68, worst=0.85, worst_row="t07"),
            verdict.Measurement(metric="pref_delta", mean=0.00, worst=-0.10),
        ],
        "different AND degraded": [
            verdict.Measurement(metric="lpips", mean=0.68, worst=0.85, worst_row="t07"),
            verdict.Measurement(metric="pref_delta", mean=-0.40, worst=-0.90),
        ],
        "different, parity never checked": [
            verdict.Measurement(metric="lpips", mean=0.68, worst=0.85, worst_row="t07"),
        ],
    }
    print("\ntri-state verdicts:")
    for label, measurements in cases.items():
        v = verdict.evaluate(measurements, budget)
        review = " (needs a human)" if v.needs_human_review else ""
        print(f"  {label:<34} -> {v.verdict}{review}")
    print("\n  note the last one: an unmeasured dimension never rescues a "
          "divergent\n  candidate — 'parity was never checked' is not 'parity holds'.")


if __name__ == "__main__":
    main()
