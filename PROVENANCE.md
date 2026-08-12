# Provenance

Why this file exists: this library is MIT, and it reimplements methods that are
also implemented in non-commercially-licensed projects. Algorithms and metrics
are not copyrightable and published research is meant to be reimplemented — but
that claim should be *checkable*, not asserted. So every module records which
paper or public spec it implements, whether it is an original implementation,
and the licence of everything it depends on, code and weights alike.

Two hard rules govern the whole repository:

1. **No code from a non-commercial project is copied or lightly paraphrased into
   this package.** Not from GenEval2 (CC-BY-NC), not from pyiqa (PolyForm
   Noncommercial + NTU S-Lab), not from DOVER/FAST-VQA (S-Lab). Reading such code
   to understand behaviour is fine; what ships here is originally authored.
   Relicensing someone else's expression as MIT would destroy exactly the
   advantage this library exists to create.
2. **The library never defaults to, downloads, or recommends weights that cannot
   be used commercially.** An MIT codebase that pulls non-commercial weights on
   first run is non-commercial software with extra steps. Any NC-weight
   integration, if one is ever added, sits behind an explicit opt-in flag and
   emits a loud licence warning at load.

Licence verification date: **2026-07-27** (video-mode rows: **2026-07-28**;
dependency table re-audited at latest resolvable versions **2026-07-30**).
Every row below was read from the actual HF model-card YAML frontmatter or the
repository `LICENSE` file, not inferred from a family name or a blog post.
Re-verify before a release: upstream projects relicense (pyiqa went Apache-2.0
→ PolyForm-Noncommercial at 0.1.16 on 2026-07-08, mid-flight for anyone
depending on it). **And the HF `license:` field is a claim, not a grant — read
the base model's `config.json` when a small fine-tune claims a more permissive
licence than its backbone.** Three survey claims were reversed by doing exactly
that (2026-08-11): "VideoMAE-v2 is Apache-2.0" (it is `cc-by-nc-4.0` on HF and
the upstream GitHub repo carries **no licence at all**), "COVER is an MIT DOVER
replacement" (its `swin_backbone.py` and `head.py` are **byte-identical** to
DOVER's non-commercial source with the notice removed), and
"`videophy_2_auto` is MIT" (its `config.json` is `model_type: llama`, vocab
32000, 4096 hidden, 32 layers — mPLUG-Owl-video on **LLaMA-7B**, the same trap
shape as the LiFT-Critic row below).

---

## Module provenance

| module | implements | origin | notes |
|---|---|---|---|
| `registry.py` | — | **original** | The four-dimension design and the one-gated-headline-per-dimension rule are ours. The redundancy correlations recorded in it are our own measurement on 44 banked A/B pairs. |
| `verdict.py` | — | **original** | The faithfulness-vs-capability-parity tri-state is our own policy. |
| `metrics/adherence.py` | checklist scoring, VLM-judge orchestration, edit change/preserve duality | **original** | Follows the prompt/parse *structure* of Qwen-Image-Bench / Q-Judger (Apache-2.0 code and weights), which is genuinely checklist-shaped. The change/preserve split follows the SC/PQ structure described by VIEScore (MIT). No code copied from either. |
| `metrics/similarity.py` | LPIPS, SSIM, MS-SSIM, PSNR | **thin wrapper over torchmetrics (Apache-2.0)** | We deliberately do not reimplement these — torchmetrics is permissive, maintained, and CI-tested against skimage, pytorch_msssim, and Zhang's reference LPIPS. Reimplementing them would be worse code for no licence benefit. |
| `metrics/ocr.py` | OCR item reading | **thin wrapper over rapidocr (Apache-2.0)** | Fallback to python-doctr. |
| `metrics/preference.py` | preference scoring | **original wrapper**; the scoring models are third-party | See the weights table. |
| `catalog.py`, `promptset.py` | prompt/checklist pairing and cross-validation | **original** | |
| `checklists/`, `promptsets/` | the authored prompt set and its checklists | **original content**, authored by us | |
| `metrics/geneval.py` | GenEval scoring semantics (thresholds, dead-zone position rule, exact counts via exclude clauses) | **original**; semantics mirror the MIT original (`djghosh13/geneval`) for score comparability | Backends deliberately differ: Grounding DINO (open-vocab, Apache) instead of mmdet Mask2Former; SigLIP2 instead of OpenAI CLIP for colour binding (same three prompt templates). Bbox crops instead of instance-mask composites — boxes are all an open-vocab detector gives. |
| `metrics/quality.py` (NIQE) | Mittal, Soundararajan & Bovik, *Making a "Completely Blind" Image Quality Analyzer* (IEEE SPL 2013) | **original implementation** | Pristine-model MVG parameters (`metrics/data/niqe_pristine.npz`) adapted from scikit-video (**BSD-3-Clause**) with attribution. Parity vs the NC oracle: max 2.4% rel on banked fixtures (`tests/fixtures/oracle_niqe.json`). |
| `metrics/quality.py` (arniqa, clip_iqa) | ARNIQA (Agnolucci et al. 2024), CLIP-IQA (Wang et al. 2023) | **thin wrappers over torchmetrics (Apache-2.0)** | Deliberately not reimplemented — permissive implementations already exist. Both are banked against the NC oracle: `clip_iqa` is bit-identical to it under the oracle's five-pair prompt set (our default is the paper's single canonical pair); `arniqa` deliberately diverges on the half-scale resize — we antialias, the oracle decimates — and swapping the resize reproduces it to 1e-4. `tests/fixtures/oracle_{clip_iqa,arniqa}.json`. |
| `metrics/musiq.py` | Ke et al., *MUSIQ: Multi-scale Image Quality Transformer* (ICCV 2021) | **PyTorch port ADAPTED with attribution from the Apache-2.0 reference** (`google-research/musiq`, flax) | Loads Google's released Apache checkpoints via our own npz converter — no permissive PyTorch path existed before. TF GAUSSIAN resize, SAME patching and the v1 nearest hash rule reproduced; parity vs the NC oracle max 4.5% rel, identical rankings (`tests/fixtures/oracle_musiq.json`). |
| `metrics/vqascore.py` | VQAScore (Lin et al. 2024, Apache-2.0) single-number p("yes"); Soft-TIFA aggregation (GenEval2 paper, arXiv:2512.16853 — METHOD only) | **original** | No GenEval2 code or data touched; atoms come from our own `decompose/` templates. Soft p(yes) normalized vs p(no) over one answer token. |
| `metrics/hpsv3.py` | HPSv3 (Ma et al. 2025) | **ADAPTED with attribution from the MIT original** (`MizzenAI/HPSv3`) | Ported off its `transformers==4.45.2` pin to modern transformers. The INSTRUCTION text and ranknet head shape are the trained model's contract, reproduced byte-exact. Checkpoint key remap 4.45-era -> v5 layout. |
| `decompose/` | template decomposition into atomic claims (TIFA/DSG shape, Apache-2.0; GenEval2's deterministic-template idea — method only) | **original**, including the 40-object/18-attribute/12-relation vocabulary and all prompt/question templates | GenEval2's 800-prompt dataset is CC-BY-NC and is NOT used; our case generator replaces it with deterministic, versioned, MIT data. |
| `metrics/temporal.py` | Δ-frame channel: PSNR/SSIM (closed-form / Wang 2004) computed over consecutive-frame difference images | **original** | The frame-difference framing follows common practice in video-codec and VAE evaluation (frame-to-frame dynamics preservation); PSNR and SSIM themselves come from the same paths `metrics/similarity.py` uses. Single-arm temporal stats are NOT implemented here — composed from `metrics/signal.py` (MIT, this repository). |
| `video.py` | video runner; motion/hold checklist scoring over an ordered frame strip; OCR persistence rule | **original** | Same Judge protocol as images; one call per clip. The VBench temporal-flickering dimension (Apache-2.0) was verified to use NO pretrained model — supporting evidence that closed-form temporal channels are the norm, not a shortcut. |
| `metrics/signal.py` | Laplacian variance, spectral HF ratio, histogram entropy, first/second temporal differences | **original** | Elementary signal statistics, not anybody's library. numpy-only; the always-available core the gate's budgets are calibrated on. |
| `metrics/audio.py` | levels/clipping/silence/DC/spectral-flatness; ITU-R BS.1770-4 integrated loudness; SI-SDR, SNR, log-spectral distance, log-mel L1 | **original implementation** | All closed form, no trained weights. Loudness follows **ITU-R BS.1770-4** (K-weighting + gated mean square) with the analog prototype parameters `pyloudnorm` (**MIT**, Steinmetz & Reiss) bilinear-transforms from — necessary because the recommendation tabulates coefficients at 48 kHz only and our audio is 32 kHz. Parity banked against pyloudnorm in a throwaway venv (`parity/`). SI-SDR follows Le Roux et al. 2019; the biquad forms are the RBJ Audio EQ Cookbook's, which is universally republished. Level statistics are cross-checked against ffmpeg `volumedetect` on real media in `tests/test_audio_corpus.py`. |
| `metrics/avsync.py` | audio-visual offset by onset cross-correlation | **original implementation** | Audio onset envelope is half-wave-rectified spectral flux (Bello et al. 2005); the visual envelope is the rectified derivative of frame-difference energy, symmetric with it. Correlating an acoustic envelope against a visual-change envelope to recover AV correspondence is the classical framing — Hershey & Movellan (NIPS 1999), Slaney & Covell (NIPS 2000). NO trained weights and no third-party code. It scores EVENT sync, not lip-sync; see the audio licence survey below for why the lip-sync instruments are not shipped. |
| `audio.py` | audio ingest, the absolute defect budget, the one audio verdict function, the audio checklist | **original** | `AUDIO_DEFECTS` is the deliberate exception to "thresholds are not shipped": silence, clipping, dual-mono and DC are content-INDEPENDENT engineering faults, unlike an LPIPS budget. Calibrated on the 18-clip fal MiniMax-H3 corpus (ie#612, 2026-08-07) as a known-good population, with each limit's margin recorded in its `provenance` field. Speech items reuse the OCR lane's fuzzy matcher because reading a required string out of recognised text is the same operation whether it came from pixels or samples. |
| `metrics/reference.py` | PSNR / SSIM / VMAF / ColorVideoVDP over whole video files | **original wrapper**; the metrics are third-party | `libvmaf` through the ffmpeg CLI, `cvvdp` and `scikit-image` when installed, with a torch-free windowed-SSIM fallback so the reference lane runs on a bare install. |
| `metrics/distributional.py` | Fréchet distance over a feature population; I3D-FVD / JEDi wrappers | **original** (closed-form Fréchet); wrappers over `cd-fvd`, `fvmd` | Opt-in and explicitly NOT validated at the sample sizes this library's gate runs at — see GATE.md. |
| `benchmarks.py`, `gate.py`, `control.py`, `protocol.py` | lanes, protocol stamping, population gating, null controls, the calibrated budget table | **original** | The two-lane rule and the null-control policy are this library's thesis; every budget carries the populations that fixed it. |
| `promptsets/hard_video_v1.json`, `checklists/hard_video_v1.json` | the frozen 16-prompt t2v set (seeds 301-316) and its authored motion/hold checklists | **original content**, authored by us | Prompt rows are byte-identical to our internal frozen source; the checklists are new. |

### Papers and specs these methods come from

- **GenEval** — Ghosh, Hajishirzi & Schmidt, *GenEval: An Object-Focused
  Framework for Evaluating Text-to-Image Alignment* (NeurIPS 2023). Object
  presence, counting, colour, position, two-object composition scored with an
  object detector. Note the original `djghosh13/geneval` is **MIT**; only
  **GenEval2** (Meta) is CC-BY-NC.
- **TIFA** — Hu et al., *TIFA: Accurate and Interpretable Text-to-Image
  Faithfulness Evaluation with Question Answering* (ICCV 2023). Apache-2.0.
- **DSG** — Cho et al., *Davidsonian Scene Graph* (ICLR 2024). `j-min/DSG` is
  Apache-2.0.
- **VQAScore** — Lin et al., *Evaluating Text-to-Visual Generation with
  Image-to-Text Generation* (ECCV 2024). Wrapper Apache-2.0; see the weights
  caveat below.
- **LPIPS** — Zhang et al., *The Unreasonable Effectiveness of Deep Features as a
  Perceptual Metric* (CVPR 2018). BSD-2-Clause.
- **SSIM / MS-SSIM** — Wang et al. (2004 / 2003).
- **CLIPScore** — Hessel et al. (EMNLP 2021).
- **HPS** — Wu et al., *Human Preference Score* v2 / v3.
- **PickScore** — Kirstain et al., *Pick-a-Pic* (NeurIPS 2023).
- **VIEScore** — Ku et al. (ACL 2024). MIT.
- **MUSIQ** — Ke et al., *MUSIQ: Multi-scale Image Quality Transformer*
  (ICCV 2021). Reference code + checkpoints Apache-2.0 (google-research).
- **NIQE** — Mittal, Soundararajan & Bovik, *Making a "Completely Blind" Image
  Quality Analyzer* (IEEE SPL 2013). Closed-form; parameters are reproducible
  research artifacts, independently re-shipped by BSD/MIT projects.
- **Soft-TIFA** — Kamath et al. (GenEval2 paper, arXiv:2512.16853). The
  aggregation method (soft per-atom p(yes), arithmetic atom mean, geometric
  prompt score) is implemented from the paper; the CC-BY-NC repo is untouched.
- **ARNIQA** — Agnolucci et al. (WACV 2024). Apache-2.0, shipped by torchmetrics.
- **CLIP-IQA** — Wang et al. (AAAI 2023). Original repo is S-Lab (NC); the
  torchmetrics implementation we depend on is Apache-2.0.
- **VBench** — Huang et al. (CVPR 2024). Apache-2.0 repo; its
  `temporal_flickering` dimension is model-free (mean absolute difference of
  consecutive frames) and its `motion_smoothness` depends on the AMT
  interpolator, which is CC-BY-NC (see the traps table). Methods only; no code
  used. **The NC taint is INSTALL-WIDE, verified 2026-08-11**: `pyiqa` is a hard
  `requirements.txt` entry (pulled in by `vbench/imaging_quality.py`, which does
  `from pyiqa.archs.musiq_arch import MUSIQ`), so a stock `pip install` of VBench
  puts PolyForm-Noncommercial code in the tree *regardless of which dimension is
  called*. If a VBench dimension is ever adopted it must be a **re-hosted
  clean-dependency build**, never a stock install — the escape hatches are
  verified: the original MUSIQ + SPAQ checkpoints ship Apache-2.0 from
  google-research (which is what `metrics/musiq.py` already ports), and FILM
  (Apache-2.0) or RIFE (MIT) are clean AMT swaps. Commercially-clean per-video
  dimensions after that surgery: `dynamic_degree`, `subject_consistency`,
  `background_consistency`, `overall_consistency`, `temporal_flickering`.
  **Dynamic Degree's top-5% pooling was evaluated for our over-smoothing gate
  and MEASURED not to transfer** (`calibration/motion-magnitude.json`); note
  also that its published per-video output is BINARY and its dimension score is
  a set-level fraction, so it is not a per-clip gauge in the first place.
- **FVD** — Unterthiner et al. (2018). Licence is actually fine (Apache-2.0
  I3D, checkpoint included) — **rejected on statistical grounds**: like FID it
  needs a large sample population, and this suite scores 16-clip runs.

---


### Audio and AV-sync: the methods, and the licence survey behind them

Surveyed 2026-08-07, primary sources only (LICENSE files, HF model-card
`license:` fields, PyPI metadata, ITU pages). **Adopt over invent** was applied
first; what is implemented here is what had no clean implementation to adopt.

- **ITU-R BS.1770-4** — *Algorithms to measure audio programme loudness and
  true-peak audio level*. A published standard, no weights. `pyloudnorm`
  (**MIT**) is an independent implementation of the same algorithm — NOT ITU
  reference source — which is why it is safe as a parity oracle.
- **SI-SDR** — Le Roux, Wisdom, Erdogan & Hershey, *SDR — half-baked or well
  done?* (ICASSP 2019). Closed form. `torchmetrics` (**Apache-2.0**) ships a
  pure-torch SI-SDR/SI-SNR/SNR/SDR with no third-party dependency, and is the
  right thing to adopt for anyone already on torch; ours is numpy so the audio
  tier runs on a base install without torch.
- **Log-spectral distance** — Gray & Markel, *Distance measures for speech
  processing* (IEEE TASSP 1976). Closed form.
- **Onset detection** — Bello, Daudet, Abdallah, Duxbury, Davies & Sandler,
  *A Tutorial on Onset Detection in Music Signals* (IEEE TSAP 13(5):1035–1047,
  2005). The spectral-flux onset function.
- **Audio-visual synchrony by correlation** — Hershey & Movellan, *Audio Vision:
  Using Audio-Visual Synchrony to Locate Sounds* (NIPS 12, 1999, pp. 813–819);
  Slaney & Covell, *FaceSync: A Linear Operator for Measuring Synchronization of
  Video Facial Images and Audio Tracks* (NIPS 13, 2000, pp. 814–820). Note
  FaceSync *fits* a linear operator, so it is cited for the framing, not
  reimplemented.
- **Whisper** — Radford et al. 2022. Code **AND weights MIT**, stated verbatim
  in the repo README. The recommended backing model for the `Transcriber`
  protocol; `faster-whisper` and `whisper.cpp` are also MIT.

**REJECTED — PESQ (ITU-T P.862), and it is a trap worth stating loudly.** The
`pesq` PyPI package is *labelled* MIT, but it embeds a modified copy of the ITU
P.862 ANSI-C reference (`dsp.c`, `pesqdsp.c`, `pesqmod.c`, …) whose header
carries the PESQ Intellectual Property Rights Notice: rights assigned to
Psytechnics Limited and OPTICOM GmbH, users *may not* "alter, duplicate, modify,
adapt, or translate in whole or in part any aspect of the PESQ Algorithm and or
PESQ Software", and permitted use is conformance testing **provided results are
not used commercially**. The packager's MIT label cannot override the embedded
rights-holders' notice, and the package modifies the source the notice forbids
modifying. ITU also deleted P.862 on 2024-01-05 as out of date. **Consequence
for this repository: never `pip install torchmetrics[audio]`** — that extra pins
`pesq>=0.0.4,<0.0.5` and drags the encumbered code into the tree. Plain
`torchmetrics` is fine. P.863 (POLQA), its successor, is commercially licensed
and is likewise not an option.

**NOT SHIPPED, licence-clean but impractical today.** ViSQOL v3 (google/visqol)
is **Apache-2.0** including its in-repo model files, and is the right
full-reference perceptual audio metric to reach for — but it builds through
Bazel with no PyPI wheel, so it is a source install, not a CI dependency.
Recorded as the intended upgrade path for `audio_lsd_db`.

**LIP-SYNC: the honest gap.** SyncNet (Chung & Zisserman, ACCV 2016) is the
standard instrument and LSE-D / LSE-C are its outputs. `joonson/syncnet_python`
is **MIT**, but the *weights* — which are the method — are published by VGG as
"CC-BY … for research purposes", and that stated intent sits awkwardly against a
commercial gate. Synchformer (Iashin et al., ICASSP 2024) and SparseSync (BMVC
2022) are **MIT code** with released checkpoints that carry **no
weights-specific licence statement at all** — repo-MIT is a reasonable read but
it is inference, not a statement. So this library ships the closed-form event
sync above, states that it does not score dialogue, and tracks Synchformer as
the candidate pending a weights-licence confirmation. `avsync.LIPSYNC_GAP_NOTE`
puts that sentence in every report that carries a sync number, rather than
leaving it in a document nobody reads.

## Code dependencies

| package | licence | verified at | role |
|---|---|---|---|
| `msgspec` | BSD-3-Clause | PyPI/GitHub | all data structures; the only base dependency |
| `torchmetrics` | **Apache-2.0** | github.com/Lightning-AI/torchmetrics `LICENSE` | LPIPS/SSIM/MS-SSIM/PSNR |
| `torch`, `torchvision` | BSD-3-Clause | github.com/pytorch | tensor math |
| `transformers` | Apache-2.0 | github.com/huggingface/transformers | CLIP, judge VLM, preference models |
| `rapidocr` | **Apache-2.0 code AND weights** | project docs | OCR items |
| `numpy` | BSD-3-Clause | | array ops |
| `pillow` | MIT-CMU | | image IO in examples/tests |
| `scipy` | BSD-3-Clause | github.com/scipy | NIQE (`ndimage.correlate`, `special.gamma`). Its wheels bundle OpenBLAS/`libgfortran` (GPL-3.0 **with the GCC Runtime Library Exception**) and `libquadmath` (LGPL-2.1) — the standard manylinux scientific-Python situation, explicitly permitted for non-GPL programs. Noted because a naive licence scanner flags the string. |
| `piq` | Apache-2.0 (PyPI classifier; `license` field blank) | PyPI | required at runtime by torchmetrics' `clip_iqa` default checkpoint path — re-verified against torchmetrics 1.9.0, which still does `import piq; piq.clip_iqa.clip.load()` |
| `onnxruntime` | MIT | PyPI classifier | rapidocr runtime |
| `safetensors`, `huggingface-hub` | Apache-2.0 | github.com/huggingface | HPSv3 checkpoint fetch + load |
| `python-doctr` | Apache-2.0 | github.com/mindee/doctr | OPTIONAL OCR fallback, deliberately **not declared** in any extra — bring your own |
| `cozy_eval.metrics.signal` | MIT (this repository) | root `LICENSE` | single-arm temporal signal stats (`luma_flicker`, `jerk_ratio`) are composed from its `score_clip`; numpy-only, always available in the base install |

Transitive closure audited 2026-07-27 against the actual installed metadata of
every extra: **no CC-BY-NC, PolyForm-Noncommercial, S-Lab, AGPL or GPL package
appears anywhere**, directly or transitively. `torchvision` and `torch_fidelity`
(both permissive) arrive via `torchmetrics[image]`; `certifi` is MPL-2.0
(file-level, does not propagate) and `tqdm` is dual MPL-2.0/MIT.

Not used, and why — these are the licence traps, recorded so nobody re-walks them:

| project | licence | verdict |
|---|---|---|
| **pyiqa / IQA-PyTorch** | **PolyForm-Noncommercial-1.0.0 + NTU S-Lab** (`chaofengc/IQA-PyTorch` `LICENSE`) | UNUSABLE, and as of 2026-07-30 **not reachable from any extra**: the `perceptual` extra that pinned `pyiqa<0.1.16` is deleted. Relicensed from Apache-2.0 at 0.1.16 (2026-07-08). Survives only as the dev-only parity oracle (0.1.15, the last Apache release), installed in a throwaway venv by `parity/run_oracle.py`; the NUMBERS are committed, the package never is. |
| **GenEval2** | **CC BY-NC 4.0**, Meta Platforms (`facebookresearch/GenEval2` `LICENSE`) | UNUSABLE. Reimplement the method from the GenEval paper instead. |
| **DOVER / FAST-VQA / FasterVQA / Q-Align / Q-Bench** | **S-Lab License 1.0** (non-commercial) — while their `setup.py` still declares MIT/Apache | UNUSABLE. The declared metadata is wrong; read the `LICENSE`. Everything out of NTU S-Lab carries this, which removes most of the no-reference video-quality canon in one stroke. |
| **COVER** (`vztu/COVER`) | claims MIT | ⛔ **TRAP, verified by md5 2026-08-11**: `swin_backbone.py` and `head.py` are **byte-identical to DOVER's** non-commercial source with the notice removed (`conv_backbone.py` / `evaluator.py` lightly modified), and S-Lab clause 1 requires retaining the notice. The obvious "MIT DOVER replacement" is not one. **Do not ship COVER.** |
| **VideoPhy-2 auto-evaluator** (`videophysics/videocon_physics` lineage) | card says `mit` | ⛔ **TRAP**: `config.json` is `model_type: llama`, vocab 32000, hidden 4096, 32 layers on an `mplug_owl_vision_model` tower — mPLUG-Owl-video on LLaMA-7B, and Meta's terms reach through. Physics plausibility stays UNCOVERED rather than shipping this. |
| **VideoMAE v1 / v2 backbones** (via `cd-fvd`'s `model="videomae"`) | `OpenGVLab/VideoMAEv2-Large` is **`cc-by-nc-4.0`**; upstream GitHub repo licence **NONE**; `cd-fvd` vendors the source into `cdfvd/third_party/VideoMAEv2/` **with no LICENSE in that directory** | UNUSABLE. `metrics/distributional.py` defaults to `model="i3d"`; **that default is now also a LICENCE decision, not only the broken-URL one it started as.** |
| **CoTracker / CoTracker3** | **CC-BY-NC-4.0** (re-confirmed 2026-08-11) | UNUSABLE — which is why the track-stability family uses BSD pyramidal Lucas-Kanade. Permissive alternatives if a dense tracker is ever wanted: BootsTAPIR / TAPNet (Apache-2.0), Track-On (MIT). |
| **LiFT / MJ-Video / T2VQA / DEVIL / T2V-CompBench** | **no LICENSE file at all** | UNUSABLE — all rights reserved is strictly worse than non-commercial. The dominant blocker in this field is "unlicensed", not "NC". |
| **Ultralytics YOLO** | **AGPL-3.0** (paid Enterprise Licence exists) | UNUSABLE as a dependency. Copyleft would reach our users' products. |
| **piq** | Apache-2.0 | USED (quality extra): torchmetrics' `clip_iqa` default checkpoint path requires `piq>=0.8` at runtime — found by the pod smoke, not the docs. Otherwise still redundant with torchmetrics. |
| **t2v_metrics (VQAScore)** | Apache-2.0 wrapper | wrapper usable; **its checkpoints are not** — see weights table. |
| **EvalCrafter** | **NO LICENSE file at all** (GitHub API: null) | UNUSABLE as code — all rights reserved by default. Its sub-metrics are reimplementable from the paper. Also depends on DOVER (S-Lab NC). |
| **AMT frame interpolation** (`MCG-NKU/AMT`) | **CC-BY-NC-4.0**, code AND checkpoints (commercial contact by email) | UNUSABLE. This is what makes a straight port of VBench `motion_smoothness` impossible; a permissive interpolator swap (RIFE claims MIT — unverified) would be a genuine contribution. |
| **T2VQA** | no license (GitHub API: null) | UNUSABLE. |

### Commercial dual-licensing fact-find

Asked because it is worth knowing, though it changes nothing about our approach
(we are not buying a licence to copy code — we are reimplementing methods):

- **GenEval2** — no public commercial-licence offering found in the repo, docs,
  or search. Treat as not offered publicly. **UNVERIFIED.**
- **pyiqa / IQA-PyTorch** — no public purchase page; only an author contact email
  surfaced. **UNVERIFIED whether a commercial licence is actually sold.**

---

## Model weights

**This table is the one to read before shipping.** Code licence and weights
licence are different facts and are recorded separately. `cozy-eval` is MIT;
these weights are not ours to relicense, and we do not claim they are.

| model | code licence | **WEIGHTS licence** | status | used for |
|---|---|---|---|---|
| `CodeGoat24/UnifiedReward-2.0-qwen3vl-2b` | MIT | **MIT** (HF frontmatter) | ✅ clean | preference (alternate) |
| `CodeGoat24/UnifiedReward-2.0-qwen3vl-8b` | MIT | **MIT** (HF frontmatter); base `Qwen/Qwen3-VL-8B-Instruct` is Apache-2.0 | ✅ clean | **the tracked VIDEO preference integration** — its card documents video pairwise+pointwise scoring; the only commercially clean video preference model found (survey 2026-07-28). Not yet wired; video preference reports UNMEASURED until it is. |
| `TIGER-Lab/VideoScore` | — | **Apache-2.0** (HF frontmatter); base Mantis-8B-Idefics2 is Apache-2.0 | ✅ clean | video preference fallback candidate; not integrated |
| `KwaiVGI/VideoReward` | — | **Apache-2.0** (HF frontmatter); base Qwen2-VL-2B-Instruct is Apache-2.0 | ✅ clean | video preference fallback candidate; not integrated |
| **`Fudan-FUXI/LiFT-Critic-13b-lora`** | — | card frontmatter says "mit" but it is a **LoRA on VILA1.5-13b = CC-BY-NC-4.0, "non-commercial use only", LLaMA terms** | ⛔ **TRAP** | **never used.** The card's own licence tag is misleading; the base model's terms reach through a LoRA. |
| `THUDM/VisionReward-Video` | — | custom `cogvlm2` licence | ⛔ | not used |
| `CodeGoat24/UnifiedReward-2.0-qwen-7b` | MIT | **MIT** (HF frontmatter) | ✅ clean | preference (alternate) |
| `Qwen/Qwen3-VL-8B-Instruct` | — | **Apache-2.0** (HF frontmatter) | ✅ clean | **default judge VLM** |
| `Qwen/Qwen2.5-VL-7B-Instruct` | — | **Apache-2.0** (HF frontmatter) | ✅ clean | judge alternate |
| **`Qwen/Qwen2.5-VL-3B-Instruct`** | — | **Qwen RESEARCH Licence — NON-COMMERCIAL** ("FOR NON-COMMERCIAL PURPOSES ONLY", HF `LICENSE`) | ⛔ **TRAP** | **never used.** Same family as the Apache-2.0 7B. Family reputation is not a licence. |
| `MizzenAI/HPSv3` | MIT | **Apache-2.0** (HF frontmatter) | ✅ clean | preference — the 4.45.2 pin is GONE: `metrics/hpsv3.py` ports inference to modern transformers |
| `IDEA-Research/grounding-dino-tiny` / `-base` | Apache-2.0 (repo `LICENSE` + HF frontmatter) | **Apache-2.0** (HF frontmatter) | ✅ clean | **default detector** for compositional scoring; transformers-native |
| `google/siglip2-base-patch16-224` | Apache-2.0 | **Apache-2.0** (HF frontmatter) | ✅ clean | colour/attribute zero-shot binding. Preferred over `openai/clip-vit-*`, whose HF cards carry NO licence tag (MIT provable only via the upstream GitHub repo) |
| MUSIQ checkpoints (`gs://gresearch/musiq/{koniq,spaq,paq2piq}_ckpt.npz`) | Apache-2.0 (google-research repo root) | **Apache-2.0 by repo-root licence; no per-checkpoint override found** — medium confidence, flagged | ✅ used | `metrics/musiq.py`; the ava checkpoint (10-class head) is not wired |
| ARNIQA regressor weights (via torchmetrics) | Apache-2.0 (`miccunifi/ARNIQA`) | **Apache-2.0** | ✅ clean | quality headline |
| CLIP-IQA backbone (OpenAI CLIP via torchmetrics/piq) | MIT (upstream GitHub) | **no HF licence tag; MIT at `openai/CLIP`** | ⚠️ soft | `clip_iqa`; documented rather than assumed |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | Apache-2.0 | **Apache-2.0** (HF frontmatter) | ✅ clean | tests only — smallest judge that exercises the SoftJudge path on CPU |
| HPSv2 (`tgxs002/HPSv2`) | Apache-2.0 | Apache-2.0 **per repo README prose only** — no machine-readable HF licence tag on the raw checkpoint | ⚠️ soft | not integrated |
| `THUDM/ImageReward` | Apache-2.0 | **Apache-2.0** (HF frontmatter) | ✅ clean | preference (alternate) |
| **`yuvalkirstain/PickScore_v1`** | MIT (GitHub) | **NONE STATED** — the model card carries no `license` field | ⚠️ **owner-accepted risk** | **current default `pref_score`/`pref_delta`** |
| `openai/clip-vit-base-patch32` / `-large-patch14` | MIT (openai/CLIP) | **NONE STATED** on HF; model card says deployed use is "out of scope" (an ethical note, not a licence clause) | ⚠️ owner-accepted risk | `clip_delta` fallback |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | — | **MIT** (HF frontmatter); same soft "out of scope" card note | ✅ clean | PickScore backbone |
| LPIPS backbone (torchvision AlexNet/VGG) | BSD-2 (LPIPS), BSD-3 (torchvision) | BSD-3 umbrella; torchvision publishes no separate weights licence | ⚠️ soft, universally relied on | `lpips` |
| rapidocr PP-OCR ONNX models | Apache-2.0 | **Apache-2.0** | ✅ clean | OCR items |

### The no-licence-field decision, recorded honestly

`yuvalkirstain/PickScore_v1` and `openai/clip-vit-*` publish **no licence field**.
That is an absence of a grant, not a grant.

They are used here anyway, as an explicit, recorded decision by the project
owner on 2026-07-27: *no stated licence is treated as usable for our purposes,
the risk is accepted, and the integration would be removed or replaced on
complaint.* PickScore was chosen on merit — a continuous CLIP-H scalar resolves
the small same-prompt deltas that a generative 1-10 judge rounds to zero, at the
cost of a single forward pass.

**This is written down rather than laundered.** These weights are not described
as MIT anywhere in this project, because they are not. Downstream users get the
true status in this table and make their own call — which is the entire reason
the WEIGHTS column exists separately from the code column.

Note the asymmetry, which is deliberate: *"no terms stated"* and *"terms that
forbid it"* are different facts. The hard rule above is unchanged for
affirmatively non-commercial weights (CC-BY-NC, Qwen Research, S-Lab,
revenue-capped OpenRAIL) — those stay out regardless of accuracy.

### Detector selection for the GenEval-style reimplementation

Detectors are chosen **licence first, accuracy second**. A slightly weaker
Apache-2.0 detector beats a stronger non-commercial one, because a benchmark
nobody can use commercially is the exact problem this library exists to solve.

| detector | weights licence | usable |
|---|---|---|
| `IDEA-Research/grounding-dino-tiny` / `-base` | Apache-2.0 | ✅ |
| `google/owlv2-base-patch16-ensemble` | Apache-2.0 | ✅ open-vocabulary |
| `facebook/detr-resnet-50` | Apache-2.0 | ✅ |
| `PekingU/rtdetr_r50vd` | Apache-2.0 | ✅ |
| `shi-labs/oneformer_coco_swin_large` | MIT | ✅ |
| `facebook/mask2former-*` (HF-hosted) | **`license: other`, no terms resolvable** | ⚠️ avoid the HF mirror. The Meta source repo is MIT and mmdetection is Apache-2.0; GenEval itself consumes Mask2Former via mmdetection, not these HF repos. |
| Ultralytics YOLO | AGPL-3.0 | ⛔ |

---

## Resolved — gaps that were closed rather than shipped around

Both of the original deferrals have since been built the clean way. Recorded
here because *how* they were unblocked is the reusable part.

- **VQAScore, end-to-end — SHIPPED** as `metrics/vqascore.py`. The block was
  never the method: the `t2v_metrics` wrapper is Apache-2.0, but every
  checkpoint it ships is research-only (CLIP-FlanT5 "research use only";
  LLaVA-1.5 / InstructBLIP inherit LLaMA/Vicuna non-commercial lineage), and a
  permissive wrapper around non-permissive weights is not a permissive metric.
  Reimplementing the soft-p("yes") method against `Qwen/Qwen3-VL-8B-Instruct`
  (Apache-2.0, verified) removes the weights problem entirely.
- **MUSIQ — SHIPPED** as `metrics/musiq.py`, from the Apache-2.0
  google-research reference plus our own TF→PyTorch checkpoint converter. pyiqa
  was never the only path to it; it was only the most convenient one.

## Deferred — capabilities still not shipped

Recorded rather than shipped, per rule 1. A gap here is a correct outcome; a
quietly relicensed module is not.

- **TOPIQ.** The architecture is public but the value is in NC-trained weights,
  and there is no permissive checkpoint. Retraining on KonIQ-10k is a GPU
  project, not a port.
- **Q-Align.** S-Lab non-commercial. Its discrete-level LMM prompting could be
  reimplemented over Qwen3-VL later.
- **BRISQUE.** Was reachable through the deleted `pyiqa` extra and is now gone,
  not ported. NIQE is the opinion-*unaware* successor from the same authors and
  is shipped; BRISQUE additionally needs an SVR trained on LIVE, whose weights
  have no clean provenance. Nothing in this library called it.
- **pyiqa's remaining learned no-reference scorers** (MANIQA, TOPIQ, LIQE,
  HyperIQA, DBCNN, NIMA). MANIQA and TOPIQ were likewise reachable through the
  deleted extra and are dropped by name rather than ported. Not blocked on
  licence for most of them — permissive ORIGINALS exist — just not worth
  porting until something asks for them. pyiqa is not the bottleneck.
- **The MUSIQ `ava` checkpoint** (10-class head) is not wired; `spaq` and
  `paq2piq` are wired but unverified against an oracle.
- **Video preference scoring.** Not a licence gap — `UnifiedReward-2.0-qwen3vl-8b`
  is MIT on an Apache base and video-capable — an integration gap: wiring it
  faithfully needs its own inference recipe reproduced and validated. Until
  then video preference reports UNMEASURED, and no image-model frame-mean proxy
  is substituted.
- **Motion smoothness (VBench-style).** Blocked by the AMT interpolator's
  CC-BY-NC licence; would need a permissively licensed interpolator swap
  (RIFE claims MIT — unverified) to build cleanly.
- **Full-reference perceptual audio quality (ViSQOL).** Apache-2.0 and correct
  for the job; blocked on packaging, not licence — Bazel build, no wheel.
- **Lip-sync scoring.** See the audio survey above: SyncNet's weights are
  research-worded CC-BY and Synchformer's carry no weights-specific statement.
  Event sync ships; dialogue sync does not, and reports say so.
- **Fréchet Audio Distance.** The distributional audio sibling of the FVD path
  already here. Licence-clean end to end (`fadtk` MIT, `frechet_audio_distance`
  MIT, LAION CLAP CC0/Apache-2.0) except that the classic VGGish `.ckpt` carries
  no weights-specific statement; CLAP embeddings avoid that. Not shipped because
  the population sizes our gate runs at do not support it — the same honesty
  caveat `metrics/distributional.py` already carries.
- **No-reference audio quality (DNSMOS / NISQA / UTMOS / Audiobox-Aesthetics).**
  Audiobox-Aesthetics is **CC-BY 4.0** (Attribution, *not* NonCommercial) and is
  therefore the clean candidate for a learned no-reference audio score; the
  speech-MOS family is scoped to speech and would mis-score a music bed. Not
  shipped: the reference-free tier here is currently all closed-form, and adding
  a learned scorer needs banked distributions before it can gate anything.
- **AV-sync on general prompts.** Measured on the 18-clip fal MiniMax-H3 corpus:
  every clip is UNMEASURABLE for event sync because the generated soundtracks
  carry no hard transients (onset-envelope skew 0.2–1.7, against 4–14 for their
  own picture). Gating sync needs a prompt subset authored to contain
  synchronised events — door slam, clap, footsteps, dialogue. Recorded in
  `tests/test_audio_corpus.py` as an asserted finding.
