"""Regenerate ``motion-magnitude.json`` — why ``motion_mag_ratio`` does NOT gate.

A MEASURED NEGATIVE, banked so it is not re-proposed. ``motion_mag_ratio`` is
the obvious companion for a two-sided over-smoothing gate: an arm that collapses
into a slower take should move less than its control. It does not work.

The published fix for the obvious objection — that a whole-frame mean is
dominated by camera pan — is VBench's Dynamic Degree pooling, the top 5% most
active pixels. This script measures BOTH poolings, and each of them over 1 and 3
temporal windows (min-pooled), across the same owner-labeled sets that fixed the
track-stability bounds:

* **oversmoothed** — the arm the companion has to catch.
* **identical** / **re-render floor** — the band it must stay silent inside.
* **bit-exact** — must read exactly 1.0.
* **rejected** — the warble class, for context; it is the track family's job.

The verdict is the overlap: at every pooling tried, the arm that must fire sits
INSIDE the band that must not. See ``registry.py``'s ``motion_mag_ratio`` note.

    python calibration/run_motion.py --samples-root ~/cozy/samples
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cozy_eval import registry, resources
from cozy_eval.metrics import temporal as TP

FREESEL = "h3-freesel-20260811/out/idx"
FLEET = "h3-indexer-20260810/out/idx-fleet"
IDX = "h3-indexer-20260810/out/idx"
R2 = "pgw1081-r2confirm-20260811/evidence"
COV15 = "h3-15s-coverage-20260811/out/idx"

#: (label, class, reference, candidate) — the same labeled corpus as
#: ``run_tracks.py``, read through the flow family instead of the tracker.
PAIRS: list[tuple[str, str, str, str]] = [
    ("cov15:courier/mn-c10", "oversmoothed",
     f"{COV15}/courier-15s__dense-r1.mp4", f"{COV15}/courier-15s__mn-c10-r1.mp4"),
    ("bitexact:busker dense", "bit-exact",
     f"{FLEET}/busker-5s__dense-r1.mp4", f"{FLEET}/busker-5s__dense-r1.mp4"),
    ("bitexact:samurai exactB", "bit-exact",
     f"{R2}/samurai-30__exactB.mp4", f"{R2}/samurai-30__exactB.mp4"),
    *[(f"r2:{cell}{suf}", "identical",
       f"{R2}/{cell}{suf}-30__exactB.mp4", f"{R2}/{cell}{suf}-30__r2.mp4")
      for cell in ("busker", "loom", "samurai") for suf in ("", "B", "C")],
    ("r2w:samurai", "identical",
     f"{R2}/samurai-30__exactB.mp4", f"{R2}/samurai-30__r2w.mp4"),
    ("r2w:samuraiB", "identical",
     f"{R2}/samuraiB-30__exactB.mp4", f"{R2}/samuraiB-30__r2w.mp4"),
    ("floor:busker exact(pod+line)", "re-render floor",
     f"{R2}/busker-30__exactB.mp4", f"{R2}/busker-30__exact.mp4"),
    ("floor:samurai exact(pod+line)", "re-render floor",
     f"{R2}/samurai-30__exactB.mp4", f"{R2}/samurai-30__exact.mp4"),
    ("floor:loom exact(pod+line)", "re-render floor",
     f"{R2}/loom-30__exactB.mp4", f"{R2}/loom-30__exact.mp4"),
    ("floor:loom exactB2(pod)", "re-render floor",
     f"{R2}/loom-30__exactB.mp4", f"{R2}/loom-30__exactB2.mp4"),
    *[(f"freesel:{cell}/{seed}/{arm}", "rejected",
       f"{FREESEL}/{cell}-5s__dense-{seed}.mp4",
       f"{FREESEL}/{cell}-5s__{arm}-{seed}.mp4")
      for cell in ("busker", "carpet", "glassblower", "whale")
      for seed in ("s20260808", "s20260811")
      for arm in ("mn-k16", "mp-k16", "ph-k16")],
    *[(f"fleet:{cell}/{arm}", "rejected",
       f"{FLEET}/{cell}-5s__dense-r1.mp4", f"{FLEET}/{cell}-5s__{arm}-r1.mp4")
      for cell in ("busker", "glassblower", "whale")
      for arm in ("ph-k16", "ph-k32", "mp-k16", "mp-k32")],
    ("negative:busker jl-k32", "known-bad",
     f"{FLEET}/busker-5s__dense-r1.mp4", f"{IDX}/busker-5s__jl-k32-r1.mp4"),
    ("negative:busker g8-k32", "known-bad",
     f"{FLEET}/busker-5s__dense-r1.mp4", f"{IDX}/busker-5s__g8-k32-r1.mp4"),
    ("oracle:busker oracle-k16", "oracle-topk",
     f"{FLEET}/busker-5s__dense-r1.mp4", f"{FLEET}/busker-5s__oracle-k16-r1.mp4"),
]

#: Spatial poolings. 1.0 is the shipped whole-frame mean; 0.05 is VBench Dynamic
#: Degree's top-5%-of-pixels trim, the published answer to the pan objection.
FRACS = (1.0, 0.10, 0.05, 0.01)
#: Temporal poolings: whole clip, and the WORST of N contiguous windows.
WINDOWS = (1, 3, 4)
CLASSES = ("oversmoothed", "bit-exact", "identical", "re-render floor",
           "rejected", "known-bad", "oracle-topk")


def decode(path: Path) -> np.ndarray:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split("x"))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", *resources.ffmpeg_thread_args(), "-i", str(path),
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def magnitude_series(flows: list, frac: float) -> list[float]:
    """Per-pair pooled flow magnitude. ``frac=1.0`` is :func:`temporal._magnitude`."""
    out = []
    for fl in flows:
        m = np.hypot(fl[..., 0], fl[..., 1]).ravel()
        if frac >= 1.0:
            out.append(float(m.mean()))
        else:
            k = max(1, int(m.size * frac))
            out.append(float(m[np.argpartition(m, m.size - k)[m.size - k:]].mean()))
    return out


def window_min(ref: list[float], cand: list[float], windows: int) -> float:
    from itertools import pairwise

    r, c = np.asarray(ref), np.asarray(cand)
    edges = np.linspace(0, len(r), windows + 1).astype(int)
    return min(
        float(c[a:b].mean() / max(float(r[a:b].mean()), 1e-6))
        for a, b in pairwise(edges) if b > a
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-root", type=Path, default=Path.home() / "cozy/samples")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "motion-magnitude.json")
    args = ap.parse_args()
    resources.configure()

    cache: dict[str, dict[float, list[float]]] = {}
    rows: list[dict] = []
    for label, cls, ref_rel, cand_rel in PAIRS:
        if not (args.samples_root / ref_rel).exists() or \
           not (args.samples_root / cand_rel).exists():
            print(f"  MISSING {label}", file=sys.stderr)
            continue
        t0 = time.monotonic()
        for rel in (ref_rel, cand_rel):
            if rel not in cache:
                flows, _, _ = TP.flow_fields(decode(args.samples_root / rel))
                cache[rel] = {f: magnitude_series(flows, f) for f in FRACS}
        row = {"pair": label, "class": cls, "reference": ref_rel, "candidate": cand_rel}
        for f in FRACS:
            r, c = cache[ref_rel][f], cache[cand_rel][f]
            row[f"ratio@{f}"] = float(np.mean(c) / max(float(np.mean(r)), 1e-6))
            for w in WINDOWS[1:]:
                row[f"ratio@{f}/min{w}"] = window_min(r, c, w)
        row["seconds"] = round(time.monotonic() - t0, 2)
        rows.append(row)
        print(f"  {label:32s} mean={row['ratio@1.0']:.4f} "
              f"top5%={row['ratio@0.05']:.4f} top5%/min3={row['ratio@0.05/min3']:.4f}",
              flush=True)

    keys = [f"ratio@{f}" for f in FRACS] + [
        f"ratio@{f}/min{w}" for f in FRACS for w in WINDOWS[1:]]
    bands = {
        key: {
            cls: {
                "n": len([r for r in rows if r["class"] == cls]),
                "min": min((r[key] for r in rows if r["class"] == cls), default=None),
                "max": max((r[key] for r in rows if r["class"] == cls), default=None),
            }
            for cls in CLASSES if any(r["class"] == cls for r in rows)
        }
        for key in keys
    }
    # THE VERDICT: does the arm that must fire sit outside the band that must not?
    separation = {}
    for key, band in bands.items():
        null_lo = min(band[c]["min"] for c in ("identical", "re-render floor", "bit-exact")
                      if c in band)
        null_hi = max(band[c]["max"] for c in ("identical", "re-render floor", "bit-exact")
                      if c in band)
        over = band.get("oversmoothed", {})
        separation[key] = {
            "null_band": [null_lo, null_hi],
            "oversmoothed": [over.get("min"), over.get("max")],
            "separates": bool(over and (over["max"] < null_lo or over["min"] > null_hi)),
        }

    doc = {
        "library": "cozy-eval",
        "metric_set": registry.METRIC_SET_VERSION,
        "what": ("motion_mag_ratio calibration: the MEASURED NEGATIVE that keeps it "
                 "report-only. Every pooling puts the over-smoothed arm inside the "
                 "band the metric must not fire on."),
        "settings": {"flow_pairs": TP.FLOW_PAIRS, "flow_target_h": TP.FLOW_TARGET_H,
                     "spatial_fracs": list(FRACS), "temporal_windows": list(WINDOWS),
                     "threads": resources.active().threads},
        "pairs": rows,
        "bands": bands,
        "separation": separation,
        "any_pooling_separates": any(v["separates"] for v in separation.values()),
    }
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print()
    for key, sep in separation.items():
        print(f"{key:22s} null {sep['null_band'][0]:.3f}-{sep['null_band'][1]:.3f}  "
              f"oversmoothed {sep['oversmoothed'][0]:.3f}  "
              f"separates={sep['separates']}")
    print(f"ANY POOLING SEPARATES: {doc['any_pooling_separates']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
