"""Regenerate the calibration evidence in ``banked-pairs.json`` from real renders.

Every threshold in ``cozy_eval.benchmarks`` is fixed by this script's
output. It scores banked clips from real production and probe lanes, plus a set
of synthetic single-axis degradations of the clean reference arm, and reports
each of the three benchmarks on every pair. A threshold that does not separate
the known-clean pairs from the known-degraded ones is not a threshold.

The clips are not redistributed with the library (they are multi-GB internal
renders). Point ``--samples-root`` at them to reproduce; the committed JSON is
the evidence.

    python calibration/run_banked.py --samples-root ~/cozy/samples
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cozy_eval import benchmarks as B
from cozy_eval import resources
from cozy_eval.frames import LUMA, iter_frames
from cozy_eval.metrics.signal import ClipScore, score

LTX8 = ("forge", "chef", "market", "portrait", "fabric", "street", "water", "forest")


def _blur3(rgb):
    p = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    v = p[:-2] + 2.0 * p[1:-1] + p[2:]
    return (v[:, :-2] + 2.0 * v[:, 1:-1] + v[:, 2:]) / 16.0


SYNTHETIC = {
    "soften50": lambda f, rng: 0.5 * f + 0.5 * _blur3(f),
    "soften25": lambda f, rng: 0.75 * f + 0.25 * _blur3(f),
    "desat92": lambda f, rng: np.clip((f @ LUMA)[..., None]
                                      + 0.92 * (f - (f @ LUMA)[..., None]), 0, 1),
    "flicker2": lambda f, rng: np.clip(f * (1.0 + 0.02 * rng.standard_normal()), 0, 1),
    "flicker1": lambda f, rng: np.clip(f * (1.0 + 0.01 * rng.standard_normal()), 0, 1),
    "noise02": lambda f, rng: np.clip(
        f + 0.02 * rng.standard_normal(f.shape).astype(np.float32), 0, 1),
}


def clips(root: Path) -> dict[str, Path]:
    c: dict[str, Path] = {}
    for p in LTX8:
        for arm in ("A-comp", "C2-comp"):
            c[f"ltx8:{arm}:{p}"] = root / f"w8a8-audit/h100-seed8b-{arm}-{p}.mp4"
    for arm in ("A-eager", "A-comp", "C2-eager", "C2-comp", "B-eager"):
        c[f"ltx1:{arm}"] = root / f"ltx-w8a8-probe/h100-1-{arm}-blacksmith.mp4"
    for n in ("bw4_p1_bf16c", "bw4_p0_fp8c", "bw4_p1_fp8c", "ag12_p1_bf16c", "ag12_p1_fp8c"):
        c[f"fp8ab:{n}"] = root / f"wan22-fp8ab/out/{n}.mp4"
    for n in ("basewan4_p0", "basewan4_p1", "turbo4_buggy_p0", "turbo4_corrected_s3_p0",
              "turbo8_buggy_p0", "turbo8_corrected_p0", "turbo12_buggy_p0",
              "turbo12_corrected_p0"):
        c[f"show:{n}"] = root / f"animegen-showcase/out/{n}.mp4"
    for n in ("seko2_p0", "naive_seko2_p0"):
        c[f"probe:{n}"] = root / f"wan22-probe/out/{n}.mp4"
    return c


def pairs_for(root: Path):
    """label -> (list[(ref_key, cand_key)], expectation, note)."""
    P = {}
    P["LTX 2.3 w8a8-pcs compiled vs bf16 compiled"] = (
        [(f"ltx8:A-comp:{p}", f"ltx8:C2-comp:{p}") for p in LTX8],
        "CLEAN", "the shipped LTX recipe; n=8, H100, 1280x704x121f, both compiled")
    P["  ^ the same recipe judged on ONE prompt"] = (
        [("ltx1:A-comp", "ltx1:C2-comp")], "CLEAN (n=1 trap)",
        "1920x1088; a single prompt reads as a 22% sharpness loss that eight prompts show is take noise")
    P["LTX compile-only control (no quant at all)"] = (
        [("ltx1:A-eager", "ltx1:A-comp")], "CLEAN control",
        "the pair that scores LPIPS 0.196-0.249, consuming the whole fp8 reference budget")
    P["LTX fp8-storage cast (demoted rung)"] = (
        [("ltx1:A-eager", "ltx1:B-eager")], "n=1 observation", "")
    P["Wan 2.2 4-step per-tensor fp8"] = (
        [("show:basewan4_p0", "fp8ab:bw4_p0_fp8c"),
         ("show:basewan4_p1", "fp8ab:bw4_p1_fp8c")], "DEGRADED",
        "torchao Float8DynamicActivationFloat8WeightConfig() defaults: per-TENSOR both sides; cross-run reference, same recipe/seed/steps")
    P["Wan 2.2 4-step per-tensor fp8 (same-pod ref)"] = (
        [("fp8ab:bw4_p1_bf16c", "fp8ab:bw4_p1_fp8c")], "DEGRADED",
        "the tightest available degraded observation: one prompt, one pod")
    P["Wan 2.2 12-step per-tensor fp8"] = (
        [("fp8ab:ag12_p1_bf16c", "fp8ab:ag12_p1_fp8c")], "CLEAN",
        "same quantization, more steps - the damage is step-count dependent")
    P["animegen 4-step double-shift mush"] = (
        [("show:turbo4_corrected_s3_p0", "show:turbo4_buggy_p0")], "SEVERELY BAD",
        "diffusers set_timesteps double-shift; the worst banked failure")
    P["animegen 8-step double-shift"] = (
        [("show:turbo8_corrected_p0", "show:turbo8_buggy_p0")], "near-null",
        "same bug, enough steps to absorb it")
    P["animegen 12-step double-shift"] = (
        [("show:turbo12_corrected_p0", "show:turbo12_buggy_p0")], "near-null", "")
    P["Wan naive timestep grid"] = (
        [("probe:seko2_p0", "probe:naive_seko2_p0")], "bad, SEMANTIC",
        "background-figure cloning + flatter grade: a compositional failure this metric class cannot see")
    for name in SYNTHETIC:
        P[f"SYNTHETIC {name}"] = (
            [(f"ltx8:A-comp:{p}", f"syn:{name}:{p}") for p in LTX8],
            "single-axis control", "known-amplitude degradation of the clean arm")
    return P


def _score_banked(item):
    key, path = item
    return key, score(path)


def _score_synth(item):
    name, prompt, path = item
    fn = SYNTHETIC[name]
    rng = np.random.default_rng(abs(hash((name, prompt))) % (2 ** 32))
    return f"syn:{name}:{prompt}", score(fn(f.astype(np.float32), rng).astype(np.float32)
                                         for f in iter_frames(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel scorers; default: half the cozy-eval thread budget")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "banked-pairs.json")
    a = ap.parse_args()

    banked = {k: v for k, v in clips(a.samples_root).items() if v.exists()}
    missing = sorted(set(clips(a.samples_root)) - set(banked))
    if missing:
        print(f"WARNING: {len(missing)} clips missing: {missing[:5]}...")

    scores: dict[str, ClipScore] = {}
    budget = resources.active()
    n_workers = resources.worker_count(a.workers)
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=resources.pool_worker_init,
                             initargs=(max(1, budget.threads // n_workers),)) as ex:
        for k, s in ex.map(_score_banked, banked.items()):
            scores[k] = s
        synth_items = [(n, p, banked[f"ltx8:A-comp:{p}"]) for n in SYNTHETIC for p in LTX8
                       if f"ltx8:A-comp:{p}" in banked]
        for k, s in ex.map(_score_synth, synth_items):
            scores[k] = s

    out = {
        "library": "cozy-eval",
        "what": "calibration evidence: three benchmarks on known-clean and known-degraded pairs",
        "clip_scores": {k: v.metrics() for k, v in scores.items()},
        "pairs": {},
    }
    hdr = (f"{'pair':46s} {'expected':17s} {'IMG':>6s} {'wIMG':>6s} {'jerkX':>6s} "
           f"{'flick':>6s} {'nsig':>4s} {'pFQD':>7s}  verdicts")
    print(hdr)
    print("-" * len(hdr))
    for label, (keys, expect, note) in pairs_for(a.samples_root).items():
        if any(k not in scores for k, _ in keys) or any(k not in scores for _, k in keys):
            continue
        pairs = [(scores[r], scores[c]) for r, c in keys]
        res = {"imaging": B.imaging(pairs), "temporal": B.temporal(pairs),
               "distributional": B.distributional(pairs)}
        i, t, d = (res[k].values for k in ("imaging", "temporal", "distributional"))
        out["pairs"][label] = {
            "expectation": expect, "note": note, "n": len(pairs),
            "clips": [{"reference": r, "candidate": c} for r, c in keys],
            "benchmarks": {k: v.to_dict() for k, v in res.items()},
        }
        v = "".join({True: "P", False: "F", None: "-"}[res[k].passed]
                    for k in ("imaging", "temporal", "distributional"))
        ns = f"{d['significant_features']:4.0f}" if "significant_features" in d else "   -"
        fq = f"{d['population_frechet']:7.3f}" if "population_frechet" in d else "      -"
        print(f"{label:46s} {expect:17s} {i['imaging_index']:6.3f} "
              f"{i['imaging_worst_prompt']:6.3f} {t['jerk_excess']:+6.3f} "
              f"{t['flicker_ratio']:6.3f} {ns} {fq}  {v}")
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}  (verdict letters = imaging/temporal/distributional; "
          f"'-' = not evaluated below the n>= {B.HARD_MIN_PROMPTS} floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
