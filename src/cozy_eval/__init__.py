"""cozy-eval — quality benchmarking for generative image and video models,
with the validity rules built into the API.

The measurements here are ordinary. The opinion is not:

* **Two lanes, enforced.** Reference metrics (PSNR/SSIM/LPIPS/VMAF) are valid only
  when the compared change leaves the sampling trajectory intact. Ask for them on
  a quantization or compile comparison and the library raises.
* **Protocol stamping.** Every result carries resolution, frame count, steps,
  seeds, prompt set, execution lane and hardware. A number without those is not
  a result.
* **Population semantics.** Trajectory-perturbing comparisons are gated over
  n>=8 paired prompts, because a single prompt measures the take, not the recipe.
* **Thresholds with provenance.** Every budget names the known-good and
  known-bad populations that fixed it, and its separation margin.
* **Null controls.** A budget only holds where a zero-change arm (same weights,
  different seeds) sits inside it. ``measure_null_control`` measures that per
  family; budgets its control trips are disregarded, and a run with disregarded
  budgets cannot rise above INDETERMINATE.
* **No signal is not a failure.** An arm of constant frames returns DEGENERATE,
  never a confident FAIL with an imaging index of 0.0.

Quick start::

    from cozy_eval import (
        ChangeKind, Protocol, score_pairs, run_population_gate,
    )

    proto = Protocol(
        family="ltx-2.3-distilled",
        reference_arm="bf16 compiled", candidate_arm="w8a8-pcs compiled",
        changes=(ChangeKind.WEIGHT_QUANTIZATION, ChangeKind.ACTIVATION_QUANTIZATION),
        width=1280, height=704, frames=121, steps=8,
        seeds=(8,), prompts=("forge", "chef", "market", "portrait",
                             "fabric", "street", "water", "forest"),
        execution_lane="compiled", hardware="NVIDIA H100 80GB HBM3",
    )
    pairs = score_pairs(reference_clips, candidate_clips)
    print(run_population_gate(pairs, proto).summary())
"""

from .backends.signal import ClipScore
from .backends.signal import score as score_clip
from .benchmarks import (
    MIN_PROMPTS,
    REFERENCE_BUDGETS,
    VIDEO_BUDGETS,
    BenchmarkResult,
    Budget,
    distributional,
    imaging,
    temporal,
)
from .control import NullControl, measure_null_control
from .frames import iter_frames, probe
from .gate import (
    GateReport,
    run_population_gate,
    run_reference_gate,
    score_pairs,
)
from .image import IMAGE_BUDGETS, run_image_population_gate, score_image
from .protocol import (
    ChangeKind,
    Lane,
    Protocol,
    ProtocolError,
    TrajectoryPerturbingError,
    Verdict,
    classify_change,
    require_reference_lane,
)

__version__ = "0.1.0"

__all__ = [
    "IMAGE_BUDGETS",
    "MIN_PROMPTS",
    "REFERENCE_BUDGETS",
    "VIDEO_BUDGETS",
    "BenchmarkResult",
    "Budget",
    "ChangeKind",
    "ClipScore",
    "GateReport",
    "Lane",
    "NullControl",
    "Protocol",
    "ProtocolError",
    "TrajectoryPerturbingError",
    "Verdict",
    "classify_change",
    "distributional",
    "imaging",
    "iter_frames",
    "measure_null_control",
    "probe",
    "require_reference_lane",
    "run_image_population_gate",
    "run_population_gate",
    "run_reference_gate",
    "score_clip",
    "score_image",
    "score_pairs",
    "temporal",
]
