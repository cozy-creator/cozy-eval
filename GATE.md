# The gate protocol

This is the document a producer lane follows verbatim. It defines what to render,
which metrics are admissible, what the thresholds are, and where those thresholds
came from. Deviating from it silently is how a wrong verdict ships.

---

## 1. Pick the lane. This is not optional.

A quality comparison between two arms is meaningful only once you have decided
what the change did to the **sampling trajectory**.

| The change… | Lane | Admissible metrics | Valid at |
|---|---|---|---|
| leaves the latent trajectory identical and alters only what happens after it — VAE decode dtype, decoder tiling, colour conversion, output resize, video encoder, container mux | `same-trajectory` | PSNR, SSIM, LPIPS, VMAF (reference metrics) | n = 1 |
| perturbs the trajectory — weight/activation quantization, weight storage cast, `torch.compile`, attention backend, scheduler, step count, guidance, LoRA attach/fuse, denoiser dtype, offload placement, hardware SKU, new weights | `population` | no-reference statistics only, over a paired prompt set | n ≥ 8 |

`classify_change()` implements the table; `require_reference_lane()` raises
`TrajectoryPerturbingError` rather than returning a number. One perturbing change
contaminates a mixed set.

### Why the refusal exists

Measured CPU-only on banked LTX-2.3 renders at 1920×1088 (`~/cozy/samples/w8a8-audit/`):

1. **The no-op consumes the whole budget.** A compile-only control — zero
   quantization change — scores LPIPS **0.196–0.249** against a fleet fp8 budget
   of 0.25.
2. **It is divergence, not drift.** The distance is already **0.29–0.41 at frame 0**
   and flat-to-falling across the clip. Accumulating numerical error grows; a
   different take starts far apart and stays there.
3. **The ranking inverts.** An fp8-storage-cast arm and an unscaled-w8a8 arm carry
   *identical weight bytes*, but the cast arm computes its GEMMs in bf16 and is
   therefore strictly the more accurate path. It scores **0.3875 vs 0.3052** —
   worse. A metric that ranks a strictly-better arm below a strictly-worse one
   cannot choose between arms.

The same logic applies to images; it is only less visible there, because one
1024² frame diverges less than 121 frames do.

---

## 2. Render conditions (population lane)

Both arms MUST share all of these, and the protocol stamp MUST record them:

* **n ≥ 8 paired prompts.** Hard floor 6; below 6 the distributional benchmark is
  not evaluated at all. At n = 1 the *shipped-clean* LTX w8a8 recipe scores an
  imaging index of **0.934** on one prompt and **1.005** over eight. One prompt
  measures the take.
* **Same seeds, same prompt set, same order.** The prompts do not need to be the
  te#79 fixed set, but they must be fixed for the family and reused across runs.
* **Same pod.** Host CPU and GPU variant move these numbers on their own
  (an H100 PCIe and an H100 SXM are not interchangeable). Cross-pod arms return
  `INDETERMINATE`.
* **Same execution lane.** Both arms compiled, or both eager — never one of each.
  Compile alone perturbs the trajectory, so a compiled-candidate-vs-eager-reference
  comparison measures compile plus quantization and cannot attribute either.
  **The arms must be the ones you will serve**: if production serves compiled,
  gate compiled.
* **Native production resolution and step count.** A 512² proxy understates native
  damage by roughly 2.5× on image families; there is no reason to expect video to
  be kinder.
* Frames are scored **pre-encode** where possible. Pass decoded frames straight to
  `score_clip`; the encoder is itself a `same-trajectory` change and does not
  belong inside a quantization verdict.

---

## 3. The three benchmarks

Each answers a question the other two cannot. Section 5 shows a degradation that
only one of them catches, for each of the three.

### B1 — Imaging (per-frame, no-reference)

Six per-frame statistics, split into **detail** (Laplacian variance, spectral
high-frequency ratio above quarter-Nyquist, local contrast) and **tone** (luma
standard deviation, RGB saturation, 64-bin luma histogram entropy). Each is taken
as a ratio to the reference arm and aggregated as the **median across prompts**,
so one divergent take cannot drive the verdict.

```
imaging_index = detail_retention^(2/3) · tonal_retention^(1/3)
```

Catches: softening, detail loss, flattening/greying, and — via the upper bound —
noise injection and over-sharpening.

### B2 — Temporal

* `jerk_ratio` = mean |L[t] − 2L[t−1] + L[t−2]| / mean |L[t] − L[t−1]|. Smooth
  motion keeps this low whatever the motion magnitude; per-frame instability
  raises it. **The tightest statistic in the suite**: per-prompt σ_log = 0.0139 on
  the clean population.
* `flicker` = frame-mean luma standard deviation as a percentage of the mean.
* `shimmer`, `motion_energy` — advisory diagnostics, not budgets (see §6).

Catches: exposure flicker, frame-to-frame jitter, motion collapse.

### B3 — Distributional fidelity

A **paired test across the prompt set**: per-metric paired *t* on log-ratios with
Holm-Bonferroni correction, gated additionally on a ≥2 % practical effect, plus a
paired per-frame Fréchet distance on the six signal features whitened by the
reference population.

Catches: consistent population-level shifts too small to move any single index,
and — by construction — it is the only benchmark that cannot be fooled by take
noise, because it asks whether the shift is *consistent across prompts*.

**Why not FVD / JEDi.** A deep-feature Fréchet distance is a population statistic
whose estimator bias dominates at n in the single digits; published FVD uses
thousands of clips. A per-artifact producer gate cannot afford that. Because the
prompt set here is *identical between arms*, content is differenced out by
construction and a paired test has real power at n = 8. `[distributional]` ships
`frechet()` for lanes that can afford n ≥ 64, and it refuses below that.

---

## 4. Thresholds and their provenance

Calibrated on the pairs in §5. Clean reference population throughout:
**LTX-2.3-distilled, w8a8-pcs compiled vs bf16 compiled, n = 8 prompts,
1280×704 × 121 f, 8 steps, H100-80GB-HBM3, both arms compiled, same pod.**

| Budget | Value | Clean measures | Nearest failing case |
|---|---|---|---|
| `imaging_index` | 0.92 ≤ x ≤ 1.25 | 1.005 | 0.869 (Wan 4-step per-tensor fp8, same-pod ref); 1.645 (σ=0.02 pixel noise) |
| `imaging_worst_prompt` | ≥ 0.85 | 0.941 (worst of 8) | 0.647 (Wan fp8 population's worst prompt) |
| `jerk_excess` | ≤ +0.04 | +0.013 | +0.056 (2 % per-frame gain jitter); +0.053 (animegen 4-step mush); +0.104 (pixel noise) |
| `flicker_ratio` | ≤ 1.25 | 1.043 | **nothing banked trips it** — see below |
| `significant_features` | = 0 | 0 | 1 (8 % desaturation); 3 (25 % blur blend) |
| `population_frechet` | ≤ 0.20 | 0.068 | 0.286 (25 % blur blend); 4.792 (pixel noise) |

**Same-trajectory lane** — anchored on a real post-latent change: x264 re-encode
of banked LTX clips at CRF 18/23/28/34, measuring 40.6–44.4 / 37.6–42.4 /
33.9–38.7 / 30.4–35.2 dB mean PSNR.

| Budget | Value | Rationale |
|---|---|---|
| `psnr_mean` | ≥ 36 dB | passes a CRF-23-class change, fails CRF-28-class |
| `psnr_worst` | ≥ 32 dB | worst frame at CRF 23 = 35.4 dB, at CRF 28 = 32.2 dB |
| `ssim_mean` | ≥ 0.97 | **provisional** — windowed SSIM barely moves on these anchors; PSNR is operative |

**`jerk_excess` is the operative temporal statistic; `flicker_ratio` is a wide
backstop.** The 2 % gain-jitter control raises flicker to 1.248 — inside the 1.25
budget — and is caught on jerk (+0.056) instead. The budget is not tightened to make
the control fail: `flicker`'s per-prompt spread on the clean population is
σ_log = 0.18, an order of magnitude looser than jerk's 0.0139, so a threshold that
would trip on 1.248 would be about 2σ from the clean median and start failing clean
arms. Flicker earns its place by catching gross exposure instability, not marginal
cases.

Sensitivity floors, stated honestly: the temporal axis resolves ~1.5 % frame-to-frame
gain noise (1 % jitter passes at jerk +0.022, 2 % fails at +0.056). The imaging axis
resolves roughly a 25 % blur blend. The distributional axis resolves an 8 %
saturation shift held consistently across 8 prompts.

`population_frechet` is fixed against **one** clean population and is the weakest
threshold here. Re-estimate it as more clean arms bank.

---

## 4b. The null control — which budgets is this family allowed to be judged on?

Every threshold in §4 was fixed on ONE family at one resolution. A threshold is
only a threshold where the *clean* population sits comfortably inside it, and
that is a per-family fact. **Measure it; do not assume it.**

A **null control** is an arm with identical weights and different seeds. Zero
model change, so anything it trips is take spread. Declare it with
`ChangeKind.SEED` — the only change a control may carry — and run it through
`measure_null_control()`:

```python
control = measure_null_control(control_pairs, control_protocol)   # ChangeKind.SEED
report  = run_population_gate(pairs, protocol, null_control=control)
```

The gate then reports a budget the control trips as **disregarded** — still
measured, still printed, never decisive — fails only on budgets the control
proved trustworthy, and ceilings the verdict at `INDETERMINATE` whenever
anything was disregarded, because a PASS on partial evidence is not a PASS.

Measured, 2026-07-27, two image families, n = 8 each, one control arm apiece:

| Budget | null controls measured | Transfers? |
|---|---|---|
| `significant_features` = 0 | 0 and 0 | **yes** — and it is what caught a real over-sharpening arm |
| `imaging_index` 0.92–1.25 | 1.0087 and 1.0025 | **yes**, and tight |
| `imaging_worst_prompt` ≥ 0.85 | 0.9182 and **0.6102** | family-dependent |
| `population_frechet` ≤ 0.20 | **0.4688** and **1.3275** | **no** — 2.3× and 6.6× the budget at zero model change |

Without the control, the second family reads a confident `FAIL` on two budgets
its own zero-change arm fails harder. That is a threshold-transfer artifact, and
demoting an artifact on it is exactly the class of wrong verdict this library
exists to refuse. **Carry a control arm on any family outside the calibrated
envelope of §4 — it is one extra arm on a pod you already bought** (measured:
$0.13 of a $0.38 pod).

---

## 4c. Degenerate arms: NO SIGNAL is not a FAIL

A candidate whose frames are a constant fill — uniformly black, white or flat —
scores an imaging index of 0.0 and an unbounded Fréchet distance against any real
reference. Those numbers are arithmetic, not evidence: they invite ranking one
broken arm against another. Both population gates detect it up front and return
`DEGENERATE` with no benchmark numbers at all. A degenerate *reference* is
refused for the same reason — every statistic here is a ratio to it.

Real case: a Wan 2.2 per-row fp8 arm that emitted 8/8 uniformly black clips
previously returned `FAIL`, `imaging_index 0.0000`, `population_frechet 13001`.
It now returns `DEGENERATE`. Look at the render; do not rank the arm.

The same-trajectory lane is deliberately NOT covered: there, the reference render
is the same take, so PSNR against it measures the blackout correctly and a FAIL
is the right, non-misleading verdict.

---

## 5. Validation table

`imaging_index` / worst-prompt / `jerk_excess` / `flicker_ratio` /
significant-feature count / paired Fréchet. Verdict letters are
imaging·temporal·distributional; `-` means not evaluated below the n ≥ 6 floor.
Regenerate with `python calibration/run_banked.py --samples-root …`; the evidence
is `calibration/banked-pairs.json`.

| Pair | Expected | IMG | wIMG | jerkX | flick | nsig | pFQD | Verdict |
|---|---|---|---|---|---|---|---|---|
| LTX 2.3 w8a8-pcs compiled vs bf16 compiled (n=8) | CLEAN | **1.005** | 0.941 | +0.013 | 1.043 | 0 | 0.068 | `PPP` |
| ↳ the same recipe judged on ONE prompt | CLEAN (trap) | 0.934 | 0.934 | −0.015 | 1.014 | – | – | `PP-` indeterminate |
| LTX compile-only control (no quant at all) | CLEAN control | 0.980 | 0.980 | +0.000 | 0.993 | – | – | `PP-` indeterminate |
| LTX fp8-storage cast (demoted rung) | observation | 0.957 | 0.957 | −0.011 | 1.157 | – | – | `PP-` indeterminate |
| Wan 2.2 4-step per-tensor fp8 (n=2) | **DEGRADED** | **0.727** | 0.647 | +0.018 | 0.899 | – | – | **`FP-`** |
| Wan 2.2 4-step per-tensor fp8, same-pod ref | **DEGRADED** | **0.869** | 0.869 | −0.001 | 0.786 | – | – | **`FP-`** |
| Wan 2.2 12-step per-tensor fp8 | CLEAN | 1.009 | 1.009 | +0.005 | 1.004 | – | – | `PP-` |
| animegen 4-step double-shift mush | **SEVERELY BAD** | **0.626** | 0.626 | **+0.053** | 0.706 | – | – | **`FF-`** |
| animegen 8-step double-shift | near-null | 1.038 | 1.038 | −0.009 | 0.881 | – | – | `PP-` |
| animegen 12-step double-shift | near-null | 1.038 | 1.038 | +0.008 | 0.951 | – | – | `PP-` |
| Wan naive timestep grid | bad, **semantic** | 0.976 | 0.976 | +0.009 | 0.837 | – | – | `PP-` **misses it — §6** |
| SYNTHETIC soften, 50 % blur blend (n=8) | imaging axis | **0.748** | 0.690 | −0.022 | 1.000 | 3 | 1.218 | `FPF` |
| SYNTHETIC soften, 25 % blur blend (n=8) | imaging axis | **0.869** | 0.841 | −0.011 | 1.000 | 3 | 0.286 | `FPF` |
| SYNTHETIC desaturate to 92 % (n=8) | tonal axis | 0.991 | 0.991 | +0.000 | 1.000 | **1** | 0.008 | **`PPF`** |
| SYNTHETIC 2 % per-frame gain jitter (n=8) | temporal axis | 0.998 | 0.996 | **+0.056** | 1.248 | 0 | 0.006 | **`PFP`** |
| SYNTHETIC 1 % per-frame gain jitter (n=8) | below floor | 0.999 | 0.998 | +0.022 | 1.068 | 0 | 0.001 | `PPP` |
| SYNTHETIC σ=0.02 per-pixel noise (n=8) | mixed | **1.645** | 1.161 | +0.104 | 0.996 | 3 | 4.790 | `FFF` |

**Separation margins.** Clean population 1.005 vs the tightest real degraded
observation 0.869 → the 0.92 threshold sits +9.0 % above the clean measurement and
−5.6 % below the nearest failure. On the temporal axis, +0.013 clean vs +0.053
failing, threshold +0.04: +0.027 / −0.013.

**Each benchmark catches something the others miss** — the whole reason there are
three:

* **imaging only** (`FP-`) — the Wan 4-step per-tensor fp8 arm. Index 0.727, jerk
  +0.018 which is *inside* the clean band, and the population test cannot run at
  n=2. This is the real, banked, production-relevant failure.
* **temporal only** (`PFP`) — 2 % per-frame gain jitter. Imaging index 0.998, zero
  significant paired features, Fréchet 0.006 — all three pass — and `jerk_excess`
  +0.056 fails.
* **distributional only** (`PPF`) — an 8 % desaturation held consistently across all
  eight prompts. Imaging index 0.991 and jerk +0.000 both pass comfortably; the
  paired test flags `saturation` at Holm p < 0.05 with a 2.7 % effect.

---

## 6. What this gate does NOT measure

State these to anyone reading a PASS.

* **Semantic and compositional failure.** The banked Wan naive-timestep-grid pair
  is a real bug — it produces *background-figure cloning* and a flatter grade —
  and this gate passes it (index 0.976, every axis silent). Signal statistics
  cannot see a duplicated subject, a dropped prompt element, or text that renders
  as gibberish. That needs a CLIP/VLM-class scorer or a human, and it is a fourth
  axis, deliberately out of scope here.
* **Absolute quality.** Every number is a ratio to a reference arm. A gate PASS
  says "no worse than the reference", never "good".
* **Aesthetics and prompt adherence.**
* **Motion energy and shimmer** are reported but not budgeted: both are strongly
  content-driven (motion σ_log = 0.095 per prompt on the clean population) and
  shimmer also falls under plain softening, so neither separates its own axis.
  They appear as advisory notes.
* **Audio**, for families that generate it.
* Thresholds are calibrated on **LTX-2.3 and Wan-2.2 class video at 720p–1080p,
  4–12 steps**. A family far outside that (very low resolution, very long
  clips, heavily stylised flat-shaded output) should re-derive them from its own
  clean population before trusting a PASS.

---

## 7. Producer lane checklist

1. Declare every `ChangeKind` the candidate arm introduces. Do not guess; if the
   change is not in the enum, add it and decide its lane deliberately.
2. If the lane is `same-trajectory`: `run_reference_gate`, n ≥ 1, done.
3. Otherwise render **both arms** on one pod, same seeds, ≥ 8 prompts, both in the
   serving execution lane, at production resolution and step count.
4. Outside §4's calibrated envelope — any image family, any family far from
   LTX/Wan-class 720p–1080p video — render a **third arm on the same pod**: the
   same checkpoint at seeds + 1, and `measure_null_control(...)` it (§4b).
5. `score_pairs(...)` → `run_population_gate(pairs, protocol, null_control=control)`.
6. Persist `report.to_dict()` next to the artifact. It contains the verdict, every
   measured value, every budget with its provenance, which budgets were
   disregarded and why, the control's own numbers, and the protocol stamp.
   `ClipScore.to_dict()` / `from_dict()` persist the per-clip scores losslessly if
   you want to re-gate without re-rendering.
7. `INDETERMINATE` is not a pass, and neither is `DEGENERATE`. Never promote an
   artifact on either.
8. A FAIL is a **verdict**, not a fault: publish the report, keep the artifact
   staged, do not promote it.
