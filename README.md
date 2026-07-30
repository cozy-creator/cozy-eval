# cozy-eval

**Quality benchmarking for generative image and video models, with the validity
rules enforced by the API.**

Two questions, one library, one set of validity rules over both:

* *How far did the pixels move, and is that damage or just a different take?*
  Lanes, protocol stamping, null controls, population gates.
* *[Did the model make what was asked for?](#did-it-make-what-was-asked-for)*
  Checklist adherence, preference, standalone quality — MIT reimplementations of
  methods the ecosystem otherwise only ships under non-commercial licences.

Everything is organized by **what it measures**, not by who wrote it. Every
metric lives in `cozy_eval.metrics`, every metric is declared in one
`cozy_eval.registry`, and the same protocol/null-control/population rules govern
all of them.

There are good libraries for computing quality metrics. `ffmpeg-quality-metrics`
and `cvvdp` do full-reference video properly; `torchmetrics` owns FID/KID/IS and
the permissively-packaged learned scorers; `cd-fvd` and `fvmd` do population
distances. This library composes those and reimplements only what is otherwise
locked behind a non-commercial licence.

What no library does is stop you from computing the **wrong metric**. That is what
this one is for.

---

## The problem

You quantized a video model and you want to know if it got worse. The obvious move
— render the same prompt at the same seed on both arms and measure LPIPS or PSNR
between them — is the move essentially every production quantization harness makes.
`torchao`'s Flux benchmark scores same-seed LPIPS against a high-precision baseline.
DeepCompressor/SVDQuant's eval config lists `["psnr", "lpips", "ssim"]` against the
BF16 render directory, and Nunchaku's CI **hard-gates merges** on
`assert lpips < expected_lpips * 1.15`. NVIDIA's TensorRT-ModelOpt diffusers
examples ship no quality eval at all, and where they acknowledge divergence the
advice is *"we suggest to run a few more times and choose the best one."*

That measurement is invalid, and here is the evidence, taken CPU-only on banked
LTX-2.3 renders at 1920×1088:

1. **The no-op consumes the entire budget.** A compile-only control — zero
   quantization change — scores LPIPS **0.196–0.249** against a fleet fp8 budget
   of 0.25.
2. **It is divergence, not drift.** Distance is already **0.29–0.41 at frame 0**
   and flat-to-falling across the clip. Numerical error accumulates; a different
   take starts far apart and stays there.
3. **The ranking inverts.** An fp8-storage-cast arm and an unscaled-w8a8 arm carry
   *identical weight bytes*. The cast arm computes its GEMMs in bf16 and is
   therefore strictly the more accurate path — and it scores **worse**
   (0.3875 vs 0.3052).

A metric that ranks a strictly-better arm below a strictly-worse one cannot be
used to choose between arms. Anything that perturbs the sampling trajectory —
quantization, `torch.compile`, an attention-backend swap, a scheduler change, a
LoRA attach — produces a *different take of the same prompt*, and distance to the
reference render measures how different the take is, not how damaged it is.

The academic side has largely avoided the trap (Q-Diffusion, ViDiT-Q and friends
report FID/FVD/VBench, i.e. population metrics) but states its reason as sample
size rather than validity, so the production side never got the message.

## What this library adds

| | |
|---|---|
| **Two lanes, enforced** | `classify_change()` types every change as post-latent or trajectory-perturbing. `require_reference_lane()` **raises** rather than returning a number. One perturbing change contaminates a mixed set. |
| **Protocol stamping** | Every result carries resolution, frame count, steps, seeds, prompt set, execution lane, hardware and same-pod status. Cross-pod or mismatched-lane arms return `INDETERMINATE`, never `PASS`. |
| **Population semantics** | Trajectory-perturbing comparisons need n ≥ 8 paired prompts. At n = 1 a *shipped-clean* w8a8 recipe scores an imaging index of 0.934 on one prompt and 1.005 over eight. One prompt measures the take. |
| **Three orthogonal benchmarks** | Imaging, temporal, distributional — with a banked degradation each one catches that the other two miss. |
| **Thresholds with provenance** | Every budget names the known-good and known-bad populations that fixed it and its separation margin. Provisional ones say so. |
| **Null-control arms** | A budget is only a budget where a *zero-change* arm sits inside it. `measure_null_control()` measures that per family; budgets its control trips are disregarded, and a run with disregarded budgets cannot rise above `INDETERMINATE`. Measured: null controls trip `population_frechet` at 2.3× and 6.6× the budget with **zero** model change. |
| **NO-SIGNAL, not a confident FAIL** | An arm of constant frames returns `DEGENERATE`, not `FAIL` with an imaging index of 0.0. There is nothing to compare; ranking it would be a category error. |
| **Statistical honesty** | Paired *t* with Holm correction, effect sizes, and a practical-effect floor so significance without magnitude cannot fail an artifact. No package surveyed reports a confidence interval or effect size on a quality delta. |

Composed where a maintained permissive implementation exists: `ffmpeg-quality-metrics`
and `cvvdp` for full-reference video, `scikit-image` for SSIM, `lpips` for LPIPS,
`torchmetrics` for ARNIQA/CLIP-IQA, `cd-fvd` and `fvmd` for population distances.
The base install is **numpy + msgspec only** — everything else is an extra.

## Install

```bash
pip install cozy-eval                      # numpy + the ffmpeg CLI
pip install "cozy-eval[reference]"         # VMAF / ColorVideoVDP / SSIM / LPIPS
pip install "cozy-eval[quality]"           # NIQE / MUSIQ / ARNIQA / CLIP-IQA
pip install "cozy-eval[distributional]"    # cd-fvd, fvmd
```

> **Licence note.** No dependency here is non-commercial, and none ever will be.
> `pyiqa` was dropped outright when it relicensed Apache-2.0 →
> PolyForm-Noncommercial-1.0.0 at 0.1.16 (2026-07-08): NIQE and MUSIQ are ours,
> ARNIQA and CLIP-IQA come from torchmetrics, and BRISQUE/MANIQA/TOPIQ are gone
> rather than ported. DOVER and FAST-VQA are deliberately not wrapped — both are
> S-Lab-1.0 (non-commercial) while their `setup.py` files still declare
> MIT/Apache. `parity/` keeps the replacements honest against the banked oracle.

## Use

```python
from cozy_eval import (
    ChangeKind, Protocol, score_pairs, run_population_gate,
)

protocol = Protocol(
    family="ltx-2.3-distilled",
    reference_arm="bf16 compiled",
    candidate_arm="w8a8-pcs compiled",
    changes=(ChangeKind.WEIGHT_QUANTIZATION, ChangeKind.ACTIVATION_QUANTIZATION),
    width=1280, height=704, frames=121, steps=8,
    seeds=(8,),
    prompts=("forge", "chef", "market", "portrait",
             "fabric", "street", "water", "forest"),
    execution_lane="compiled",              # both arms, or it is INDETERMINATE
    hardware="NVIDIA H100 80GB HBM3",
)

pairs = score_pairs(reference_clips, candidate_clips)   # paths, dirs, or arrays
report = run_population_gate(pairs, protocol)
print(report.summary())
```

```
PASS  [population]
ltx-2.3-distilled | w8a8-pcs compiled vs bf16 compiled | 1280x704 x121f | 8 steps |
n=8 prompts | compiled both arms | NVIDIA H100 80GB HBM3 | same pod | lane=population
  PASS  imaging
        ok  0.92 <= imaging_index <= 1.25   measured 1.0050
        ok  imaging_worst_prompt >= 0.85    measured 0.9410
  PASS  temporal
        ok  jerk_excess <= 0.04             measured 0.0132
        ok  flicker_ratio <= 1.25           measured 1.0435
  PASS  distributional
        ok  significant_features <= 0.0     measured 0.0000
        ok  population_frechet <= 0.2       measured 0.0680
```

And the refusal:

```python
>>> run_reference_gate(refs, cands, protocol)   # protocol says WEIGHT_QUANTIZATION
TrajectoryPerturbingError: reference metrics (PSNR/SSIM/LPIPS/VMAF) are invalid for
['weight-quantization']: these change the sampling trajectory, so the reference
render is a different take of the prompt and distance to it measures divergence,
not damage. …
```

Post-latent changes take the reference lane and are valid at n = 1:

```python
run_reference_gate(before, after, protocol_with(ChangeKind.VAE_DECODE_DTYPE))
```

Images are the single-frame case — same lanes, same rules,
`run_image_population_gate` drops the temporal benchmark.

### Null controls: is this family allowed to be judged on these budgets?

The thresholds were calibrated on one family. Render a third arm on the same pod
— the **same checkpoint at different seeds** — and let it decide which budgets
you may believe:

```python
from cozy_eval import ChangeKind, measure_null_control

control = measure_null_control(control_pairs, control_protocol)   # ChangeKind.SEED only
report = run_population_gate(pairs, protocol, null_control=control)
```

```
INDETERMINATE  [population]
  null control n=8: DOES NOT TRANSFER — population_frechet measured 0.4688
  PASS  imaging
        ok  0.92 <= imaging_index <= 1.25   measured 1.0739
  PASS  distributional
        ok  significant_features <= 0.0     measured 0.0000
        --- population_frechet <= 0.2       measured 0.1825
        note: … DISREGARDED: the null control — identical weights, seeds only —
              measured 0.4688 …, so this budget does not transfer to this family.
  ! budgets disregarded on the null control's evidence … the best available
    verdict is INDETERMINATE.
```

Measured on two image families: null controls read `population_frechet` 0.4688
and 1.3275 against a 0.20 budget, and one read `imaging_worst_prompt` 0.6102
against 0.85 — at zero model change. Without the control, one of those families
reads a confident FAIL on budgets its own null arm fails harder. `imaging_index`
and `significant_features` transferred on both. Details: [GATE.md §4b](GATE.md).

### Persisting scores

`ClipScore.to_dict()` / `ClipScore.from_dict()` round-trip exactly through plain
JSON, including the `(frames, 6)` per-frame feature matrix, so a lane can cache
scores and re-gate without re-rendering. `ClipScore.metrics()` is the flat scalar
view for report rows. `cozy-eval score --json` writes the lossless form.

## CLI

```bash
cozy-eval score clip.mp4 other.mp4
cozy-eval gate --reference a.mp4 --candidate b.mp4 … \
    --change weight-quantization --execution-lane compiled …
cozy-eval compare --reference a.mp4 --candidate b.mp4 --vmaf \
    --change vae-decode-dtype …
```

## The three benchmarks

1. **Imaging** — per-frame, no-reference. Detail (Laplacian variance, spectral
   HF ratio, local contrast) and tone (contrast, saturation, histogram entropy),
   as ratios to the reference arm, median-aggregated across prompts.
2. **Temporal** — frame-to-frame. Jerk ratio (second temporal difference over the
   first), exposure flicker, shimmer, motion energy. Quantization noise often
   shows up here and nowhere else.
3. **Distributional** — a paired test across the whole prompt set, with
   Holm-corrected significance and a practical-effect floor. Not FVD: at n = 8 a
   deep-feature Fréchet distance is dominated by estimator bias. Because the
   prompt set is identical between arms, content is differenced out by
   construction and a paired test has real power.

Each catches a degradation the other two miss — see the validation table in
[GATE.md](GATE.md).

## Did it make what was asked for?

The gate above tells you whether pixels moved for a *valid* reason. The suite
answers the other question: is the render actually what the prompt requested?
It exists because the good evaluation code in this space is locked up —
GenEval2 is CC-BY-NC, pyiqa relicensed to PolyForm-Noncommercial, DOVER and
FAST-VQA are S-Lab non-commercial. This library reimplements the published
methods from the papers under MIT, with the audit trail in
[`PROVENANCE.md`](PROVENANCE.md).

Four mostly-independent dimensions, each with exactly one gated headline
number; everything else is report-only:

| dimension | what it asks | headline | needs a reference? | module |
|---|---|---|---|---|
| **similarity** | how far did the pixels move | `lpips` | yes | `metrics/similarity.py`, `metrics/reference.py` |
| **adherence** | did it contain what was asked for | `element_recall` | no | `metrics/adherence.py`, `geneval.py`, `ocr.py`, `vqascore.py` |
| **preference** | would a human prefer it | `pref_delta` | no | `metrics/preference.py`, `hpsv3.py` |
| **quality** | does it look good on its own terms | `arniqa` | no | `metrics/quality.py`, `musiq.py`, `signal.py` |

The Δ-frame temporal channel (`metrics/temporal.py`) and the population
distances (`metrics/distributional.py`) report into the dimensions above; the
gate's own two-sided budgets over a whole prompt population live in
`cozy_eval.benchmarks`, which is a threshold table, not a metric table.

```python
from cozy_eval import promptset, suite

report = suite.run(samples, candidates,
                   checklists=promptset.checklists_for("hard-eval-v1"))
print(report.summary("element_recall"))
```

The load-bearing design decisions:

* **Authored, versioned checklists.** `element_recall` is the weighted fraction
  of a prompt's authored checklist verified present (`ocr` items read literally;
  `vqa` items answered by a VLM judge, one structured call per image). Checklists
  are versioned with their prompt set — never generated per run, so a score is
  reproducible. Shipped sets: `hard-eval-v1` (t2i + edit), `hard-video-v1`
  (16 frozen t2v prompts with motion/hold dualities).
* **Editing is a duality**: `edit_compliance` (the instructed change happened)
  vs `edit_preservation` (everything else stayed put) — under- and over-editing
  fail on opposite halves.
* **Video** (`cozy_eval.video.run_video`): per-frame aggregation with
  worst-frame tails, a Δ-frame temporal channel per-frame metrics cannot see,
  and motion/hold checklists judged on one ordered frame strip per clip.
* **Tri-state parity verdict** (`free_win` / `conditional_parity` / `reject`): a
  candidate that fulfilled the request *differently but equally well* is not a
  failure — the case a pixel-distance metric cannot express.
* **Registry as data.** `import cozy_eval` sees the complete metric table
  without importing a single scoring backend; external metrics join through
  `register()`.

Scoring backends are extras, so the base install stays torch-free:

```bash
pip install "cozy-eval[similarity]"      # LPIPS / SSIM / MS-SSIM / PSNR
pip install "cozy-eval[judge]"           # VLM judge, CLIP fallback, Grounding DINO + SigLIP2
pip install "cozy-eval[ocr]"             # OCR items (rapidocr, Apache-2.0)
pip install "cozy-eval[preference]"      # PickScore and alternates
pip install "cozy-eval[quality]"         # ARNIQA / CLIP-IQA / MUSIQ port / NIQE
pip install "cozy-eval[hpsv3]"           # HPSv3 preference scorer (16 GB weights)
pip install "cozy-eval[video]"           # frame handling + Δ-frame channel
pip install "cozy-eval[all]"
```

**Model licences — read before you ship.** The library is MIT; the models it
can load are not all MIT. It never defaults to weights that cannot be used
commercially, and every model it touches has a verified row in
[`PROVENANCE.md`](PROVENANCE.md) — including "none stated", where that is the
truth. `parity/` holds the harness that keeps the replacements honest against the
non-commercial oracles — NIQE within 2.4%, MUSIQ within 4.5% with identical
rankings, CLIP-IQA bit-identical under the oracle's own prompt set, ARNIQA
deliberately diverged (antialiased half-scale) with the divergence isolated and
recorded. The oracle NUMBERS, not code, are banked in `tests/fixtures/`.

**Stability**: everything re-exported from the `cozy_eval` package root (metric
names, the registry, the report schema, checklist/prompt-set formats, the
verdicts, the Judge protocols, the protocol/lane rules) is locked for 0.x;
everything under `cozy_eval.metrics.*` and `cozy_eval.decompose` is experimental.

## Documentation

* **[GATE.md](GATE.md)** — the protocol a producer lane follows verbatim: lanes,
  render conditions, thresholds with calibration provenance, the full validation
  table with separation margins, and an explicit list of what this gate does
  *not* measure.
* **[PROVENANCE.md](PROVENANCE.md)** — per module: which paper it implements,
  whether the implementation is original, and the real licence of every
  dependency and model weight involved.
* **[calibration/](calibration/)** — the evidence. `run_banked.py` regenerates
  every threshold from real renders plus synthetic single-axis controls.

## Status

Alpha. The thresholds are calibrated on LTX-2.3 and Wan-2.2 class video at
720p–1080p, 4–12 steps; a family far outside that should carry a null-control arm
(and, better, re-derive the budgets from its own clean population).
`population_frechet` is fixed against a single clean population, is the weakest
number in the table, and is the budget null controls most often disprove.

## Licence

MIT. See [`LICENSE`](LICENSE).
