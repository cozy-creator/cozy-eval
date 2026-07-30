"""The three benchmarks, and the thresholds calibrated on banked known-good/bad pairs.

Each benchmark answers a question the other two cannot:

1. :func:`imaging` — *did individual frames get worse?* Softening, detail loss,
   flattening, over-sharpening. Per-frame, no-reference, aggregated over prompts.
2. :func:`temporal` — *did the frames stop agreeing with each other?* Exposure
   flicker, per-frame jitter, fine-detail shimmer, motion collapse. Quantization
   noise shows up here in ways no per-frame statistic can see.
3. :func:`distributional` — *did the population of outputs shift?* A paired test
   across the whole prompt set. This is the only one of the three that is honest
   about sample size, and the only one that survives take divergence, because a
   single prompt's score ratio is dominated by which take you happened to draw.

Every threshold below carries its calibration provenance. See GATE.md for the
full validation table and the separation margins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _stats
from .backends.signal import (
    DETAIL_METRICS,
    IMAGING_METRICS,
    TONAL_METRICS,
    ClipScore,
)
from .protocol import Verdict

#: Minimum paired prompts for a population verdict. Below this the gate returns
#: INDETERMINATE rather than a number: at n=1 the LTX w8a8 pair scores an imaging
#: index of 0.934 on one prompt and 1.005 over eight — the single prompt is take
#: noise, and a gate that believed it would have failed a shipped-clean recipe.
MIN_PROMPTS = 8

#: Absolute floor below which no population statistic is reported at all.
HARD_MIN_PROMPTS = 6

#: Paired log-spread floor. A deterministic post-process has ~zero paired
#: variance and would otherwise manufacture an unbounded effect size.
PAIRED_SD_FLOOR = 0.01

#: A metric must move at least this much to count as a real finding, however
#: significant it is. Statistical significance without a practical effect is a
#: sample-size artifact, not damage.
PRACTICAL_EFFECT = 0.02


@dataclass(frozen=True, slots=True)
class Budget:
    """One threshold, with the evidence that fixed it."""

    name: str
    low: float | None
    high: float | None
    provenance: str

    def check(self, value: float) -> bool:
        if self.low is not None and value < self.low:
            return False
        return not (self.high is not None and value > self.high)

    def describe(self) -> str:
        if self.low is not None and self.high is not None:
            return f"{self.low} <= {self.name} <= {self.high}"
        if self.low is not None:
            return f"{self.name} >= {self.low}"
        return f"{self.name} <= {self.high}"


#: Calibrated on the banked pairs in ``calibration/banked-pairs.json``.
#: Clean population: LTX-2.3-distilled w8a8-pcs compiled vs bf16 compiled, n=8,
#: 1280x704 x121f, H100-80GB-HBM3, both arms compiled, same pod.
VIDEO_BUDGETS: tuple[Budget, ...] = (
    Budget("imaging_index", 0.92, 1.25,
           "clean n=8 LTX w8a8 = 1.005; nearest degraded = 0.869 (Wan 4-step "
           "per-tensor fp8, same-pod ref) and 0.869 (synthetic 25% blur blend); "
           "0.626 for the animegen 4-step double-shift mush. Upper bound catches "
           "noise injection: synthetic sigma=0.02 per-pixel noise scores 1.645."),
    Budget("imaging_worst_prompt", 0.85, None,
           "worst single prompt in the clean n=8 population = 0.941; the "
           "degraded Wan population's worst prompt = 0.647."),
    Budget("jerk_excess", None, 0.04,
           "clean n=8 = +0.013 (per-prompt sigma_log 0.0139, the tightest "
           "statistic in the suite); synthetic 2% per-frame gain jitter = +0.056; "
           "animegen 4-step mush = +0.053; pixel noise = +0.104. Synthetic 1% "
           "jitter passes at +0.022, fixing the sensitivity floor at ~1.5% "
           "frame-to-frame gain noise. THIS is the operative temporal statistic."),
    Budget("flicker_ratio", None, 1.25,
           "clean n=8 = 1.043, but per-prompt sigma_log is 0.18 - an order of "
           "magnitude looser than jerk - so this is a wide backstop for gross "
           "exposure instability, not a marginal-case budget. Nothing banked or "
           "synthetic trips it: the 2% gain-jitter control reaches 1.248 and is "
           "caught on jerk instead. Deliberately NOT tightened to make that "
           "control fail; 1.20 would sit ~2 sigma from the clean median."),
    Budget("significant_features", None, 0.0,
           "count of imaging features with Holm-adjusted paired p<0.05 AND a >=2% "
           "median effect. Clean n=8 = 0; synthetic 8% desaturation = 1 (and that "
           "case passes BOTH other benchmarks: imaging index 0.991, jerk 1.000); "
           "synthetic 25% blur = 3."),
    Budget("population_frechet", None, 0.20,
           "paired per-frame Frechet distance on the six signal features, "
           "whitened by the reference population. Clean n=8 = 0.068; synthetic "
           "25% blur = 0.286; per-pixel noise = 4.792. PROVISIONAL: one clean "
           "population only — re-estimate as more clean arms bank."),
)

#: Same-trajectory lane. Calibrated on a real post-latent change: x264 re-encode
#: of two banked LTX clips at CRF 18/23/28/34.
REFERENCE_BUDGETS: tuple[Budget, ...] = (
    Budget("psnr_mean", 36.0, None,
           "x264 re-encode anchors on banked LTX clips: CRF 18 = 40.6-44.4 dB, "
           "CRF 23 = 37.6-42.4, CRF 28 = 33.9-38.7, CRF 34 = 30.4-35.2. A "
           "VAE-decode-dtype change should land far above this; 36 dB fails a "
           "CRF-28-class regression."),
    Budget("psnr_worst", 32.0, None,
           "worst frame at CRF 23 = 35.4 dB, at CRF 28 = 32.2 dB."),
    Budget("ssim_mean", 0.97, None,
           "block-windowed SSIM; PROVISIONAL — the CRF anchors barely move "
           "windowed SSIM (0.996 at CRF 34 on low-motion content), so PSNR is "
           "the operative budget until a post-latent pair is banked."),
)


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    #: Tri-state. ``None`` = not evaluated: below the sample floor, or every
    #: budget it would have used is untrusted for this family (see
    #: :mod:`cozy_eval.control`).
    passed: bool | None
    values: dict[str, float]
    budgets: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        """The same vocabulary :attr:`GateReport.verdict <cozy_eval.GateReport>`
        speaks. ``passed=None`` is INDETERMINATE, which is not a pass."""
        return {True: Verdict.PASS, False: Verdict.FAIL,
                None: Verdict.INDETERMINATE}[self.passed]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "verdict": str(self.verdict),
                "values": self.values, "budgets": self.budgets, "notes": self.notes}


Pair = tuple[ClipScore, ClipScore]      # (reference, candidate)


def _log_ratios(pairs: list[Pair], metric: str) -> list[float]:
    return [math.log(max(getattr(c, metric), 1e-12) / max(getattr(r, metric), 1e-12))
            for r, c in pairs]


def _median_ratio(pairs: list[Pair], metric: str) -> float:
    return float(np.exp(np.median(_log_ratios(pairs, metric))))


def _geo(values) -> float:
    return float(np.exp(np.mean(np.log(list(values)))))


#: Mapping ``budget name -> why it is not trustworthy here`` (a null control's
#: measurement). A disregarded budget is still measured and still reported — it
#: just cannot decide a verdict for this family.
Untrusted = dict[str, str]


def _budgets_for(names, budgets, values,
                 untrusted: Untrusted | None = None) -> tuple[bool | None, list[dict[str, Any]]]:
    """Evaluate the budgets in ``names``. Returns ``None`` if all were disregarded."""
    untrusted = untrusted or {}
    rows: list[dict[str, Any]] = []
    ok, evaluated = True, 0
    for b in budgets:
        if b.name not in names:
            continue
        v = values[b.name]
        p = b.check(v)
        row = {"budget": b.describe(), "value": v, "passed": p, "provenance": b.provenance}
        if b.name in untrusted:
            row["disregarded"] = True
            row["disregarded_because"] = untrusted[b.name]
        else:
            evaluated += 1
            ok = ok and p
        rows.append(row)
    return (ok if evaluated else None), rows


def _disregarded_note(rows: list[dict[str, Any]]) -> list[str]:
    out = [f"{r['budget']} measured {r['value']:.4f} but is DISREGARDED: "
           f"{r['disregarded_because']}"
           for r in rows if r.get("disregarded")]
    if out and all(r.get("disregarded") for r in rows):
        out.append("every budget on this benchmark is untrusted for this family — "
                   "the numbers are reported, the benchmark is not evaluated.")
    return out


# --------------------------------------------------------------------------- #
# Benchmark 1 — frame/imaging quality
# --------------------------------------------------------------------------- #

def imaging(pairs: list[Pair], budgets=VIDEO_BUDGETS,
            untrusted: Untrusted | None = None) -> BenchmarkResult:
    """Per-frame quality retention: detail (weight 2/3) and tone (weight 1/3).

    Reported as ratios to the reference arm, aggregated as the median over
    prompts so one divergent take cannot drive the verdict.
    """
    per_metric = {m: _median_ratio(pairs, m) for m in IMAGING_METRICS}
    detail = _geo(per_metric[m] for m in DETAIL_METRICS)
    tonal = _geo(per_metric[m] for m in TONAL_METRICS)
    per_prompt = [
        _geo(getattr(c, m) / getattr(r, m) for m in DETAIL_METRICS) ** (2 / 3)
        * _geo(getattr(c, m) / getattr(r, m) for m in TONAL_METRICS) ** (1 / 3)
        for r, c in pairs
    ]
    values = {
        "imaging_index": detail ** (2 / 3) * tonal ** (1 / 3),
        "imaging_worst_prompt": float(np.min(per_prompt)),
        "detail_retention": detail,
        "tonal_retention": tonal,
        **{f"ratio_{m}": v for m, v in per_metric.items()},
    }
    ok, rows = _budgets_for({"imaging_index", "imaging_worst_prompt"}, budgets, values,
                            untrusted)
    return BenchmarkResult("imaging", ok, values, rows, _disregarded_note(rows))


# --------------------------------------------------------------------------- #
# Benchmark 2 — temporal quality
# --------------------------------------------------------------------------- #

def temporal(pairs: list[Pair], budgets=VIDEO_BUDGETS,
             untrusted: Untrusted | None = None) -> BenchmarkResult:
    """Frame-to-frame stability relative to the reference arm."""
    jerk = _median_ratio(pairs, "jerk_ratio")
    shim = _median_ratio(pairs, "shimmer")
    flick = _median_ratio(pairs, "flicker")
    motion = _median_ratio(pairs, "motion_energy")
    values = {
        "jerk_excess": jerk - 1.0,
        "jerk_ratio": jerk,
        "flicker_ratio": flick,
        "shimmer_ratio": shim,
        "motion_ratio": motion,
    }
    ok, rows = _budgets_for({"jerk_excess", "flicker_ratio"}, budgets, values, untrusted)
    notes = _disregarded_note(rows)
    if motion < 0.80:
        notes.append(
            f"motion_energy fell to {motion:.3f}x the reference — advisory, not a "
            "budget: motion energy is content-driven and its per-prompt spread on "
            "the clean population is sigma_log 0.095. The animegen 4-step mush "
            "scores 0.523 here and the degraded Wan fp8 population 0.807."
        )
    if abs(math.log(shim)) > 0.12:
        notes.append(
            f"fine-detail shimmer moved {shim:.3f}x — advisory: shimmer also falls "
            "under simple softening, so it does not separate temporal from imaging "
            "damage on its own."
        )
    return BenchmarkResult("temporal", ok, values, rows, notes)


# --------------------------------------------------------------------------- #
# Benchmark 3 — distributional fidelity
# --------------------------------------------------------------------------- #

def distributional(pairs: list[Pair], budgets=VIDEO_BUDGETS,
                   untrusted: Untrusted | None = None) -> BenchmarkResult:
    """Population-level paired test over the whole prompt set.

    Why a paired test and not FVD: at the n a per-artifact gate can afford, a
    deep-feature Fréchet distance is dominated by estimator bias. The prompt set
    here is identical between arms, so content is differenced out by construction
    and a paired test over the per-clip no-reference scores has real power at
    n=8. The Fréchet term is retained on the six signal features, where n is the
    frame count (hundreds) rather than the prompt count.
    """
    n = len(pairs)
    if n < HARD_MIN_PROMPTS:
        return BenchmarkResult(
            "distributional", None, {"n": float(n)}, [],
            [(f"not evaluated: a population test needs n>={HARD_MIN_PROMPTS} paired "
              f"prompts (protocol calls for {MIN_PROMPTS}), got {n}.")],
        )

    dz, pvals, effect = {}, {}, {}
    for m in IMAGING_METRICS:
        d = np.asarray(_log_ratios(pairs, m))
        sd = max(float(d.std(ddof=1)), PAIRED_SD_FLOOR)
        dz[m] = float(d.mean() / sd)
        effect[m] = float(np.exp(np.median(d)))
        pvals[m] = _stats.paired_t(d.tolist())[1]
    adj = _stats.holm(pvals)
    sig = [m for m in adj
           if adj[m] < 0.05 and abs(effect[m] - 1.0) >= PRACTICAL_EFFECT]

    # Paired per-frame Frechet on the signal features, whitened by the pooled
    # reference population so the statistic is scale-free and comparable across
    # families.
    pooled = np.concatenate([r.features for r, _ in pairs], 0)
    scale = pooled.std(0) + 1e-9
    fq = []
    for r, c in pairs:
        a, b = r.features / scale, c.features / scale
        d = a.mean(0) - b.mean(0)
        va, vb = a.var(0), b.var(0)
        fq.append(float(d @ d + np.sum(va + vb - 2.0 * np.sqrt(va * vb))))

    values = {
        "n": float(n),
        "significant_features": float(len(sig)),
        "rms_effect_size": float(np.sqrt(np.mean([v ** 2 for v in dz.values()]))),
        "max_effect_size": float(max(abs(v) for v in dz.values())),
        "population_frechet": float(np.median(fq)),
        "population_frechet_worst": float(np.max(fq)),
        **{f"dz_{m}": v for m, v in dz.items()},
        **{f"holm_p_{m}": v for m, v in adj.items()},
    }
    ok, rows = _budgets_for({"significant_features", "population_frechet"}, budgets, values,
                            untrusted)
    notes = _disregarded_note(rows)
    if sig:
        notes.append("shifted features (Holm p<0.05, >=2% effect): "
                     + ", ".join(f"{m} x{effect[m]:.3f}" for m in sig))
    if n < MIN_PROMPTS:
        notes.append(f"n={n} is below the protocol's n={MIN_PROMPTS}; the test ran but "
                     "is under-powered and a PASS here is weak evidence.")
    return BenchmarkResult("distributional", ok, values, rows, notes)
