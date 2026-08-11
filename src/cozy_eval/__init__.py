"""cozy-eval — quality evaluation for generative image and video models, with
the validity rules built into the API.

Everything here is an evaluation; they differ in what they measure, not in kind.
Six dimensions, in :mod:`cozy_eval.metrics`:

  similarity      how far did the pixels move from a reference render
  adherence       did the render contain what the prompt asked for
  preference      would a human prefer it
  quality         does it look good on its own terms, no reference, no prompt
  temporal        did the frames stop agreeing with each other
  distributional  did the population of outputs shift

The measurements are ordinary. The opinion is not, and it governs all of them:

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
* **Registry as data.** ``import cozy_eval`` sees the complete metric table
  without importing a single scoring backend; external metrics join through
  :func:`register`.

Quick start — the gate::

    from cozy_eval import ChangeKind, Protocol, score_pairs, run_population_gate

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

Quick start — the prompt-adherence suite::

    from cozy_eval import promptset, suite

    report = suite.run(samples, candidates,
                       checklists=promptset.checklists_for("hard-eval-v1"))
    print(report.summary("element_recall"))

STABILITY (v0.x)
----------------

Locked core: everything re-exported here — the metric names, the registry, the
report schema, the checklist and prompt-set formats, the verdicts, the Judge
protocols, the protocol/lane rules. Experimental: the concrete scoring code
under :mod:`cozy_eval.metrics` and the generator in :mod:`cozy_eval.decompose`.
The metric NAMES those modules produce are locked by the registry; their
functions and internals may change in any 0.x release.
"""

__version__ = "0.1.0"

from . import resources

# FIRST, before numpy/torch/OpenCV land: their thread pools are sized from the
# environment at first use and never resized. See cozy_eval.resources.
resources.configure()

from . import (  # noqa: E402
    audio,
    catalog,
    decompose,
    detail,
    device,
    errors,
    integrity,
    judge,
    metrics,
    promptset,
    registry,
    suite,
    tracks,
    verdict,
    video,
)
from .audio import (  # noqa: E402
    AUDIO_DEFECTS,
    AudioChecklist,
    AudioSample,
    AudioVerdict,
    Defect,
    audio_verdict,
    probe_audio,
    read_audio,
    run_audio,
)
from .benchmarks import (  # noqa: E402
    MIN_PROMPTS,
    REFERENCE_BUDGETS,
    VIDEO_BUDGETS,
    BenchmarkResult,
    Budget,
    distributional,
    imaging,
    temporal,
)
from .catalog import add_checklist_set, checklist_set  # noqa: E402
from .control import NullControl, measure_null_control  # noqa: E402
from .detail import (  # noqa: E402
    DetailVerdict,
    detail_verdict,
    score_detail_vlm,
)
from .device import AUTO, resolve_device  # noqa: E402
from .errors import (  # noqa: E402
    BackendError,
    ConfigError,
    CozyEvalError,
    DataError,
    DecodeError,
    ProtocolError,
    RegistryError,
    SampleSizeError,
    TrajectoryPerturbingError,
)
from .frames import iter_frames, probe  # noqa: E402
from .gate import GateReport, run_population_gate, run_reference_gate, score_pairs  # noqa: E402
from .image import IMAGE_BUDGETS, run_image_population_gate, score_image  # noqa: E402
from .integrity import OutputIntegrity, output_integrity  # noqa: E402
from .judge import AudioJudge, Judge, SoftJudge, Transcriber  # noqa: E402
from .metrics.adherence import (  # noqa: E402
    Checklist,
    ChecklistItem,
    ChecklistSet,
    EditChecklist,
    VideoChecklist,
    load_checklists,
)
from .metrics.audio import Audio, as_audio  # noqa: E402
from .metrics.avsync import SyncEstimate, sync_offset  # noqa: E402
from .metrics.signal import ClipScore  # noqa: E402
from .metrics.signal import score as score_clip  # noqa: E402
from .promptset import PromptSet, add_prompt_set, checklists_for  # noqa: E402
from .protocol import (  # noqa: E402
    ChangeKind,
    Lane,
    Protocol,
    Verdict,
    classify_change,
    require_reference_lane,
)
from .registry import METRIC_SET_VERSION, MetricSpec, register  # noqa: E402
from .resources import ComputeBudget  # noqa: E402
from .suite import DimensionSummary, Sample, SampleRow, SuiteReport  # noqa: E402
from .tracks import TrackVerdict, track_verdict  # noqa: E402
from .verdict import Measurement, ParityVerdict, Threshold, evaluate  # noqa: E402
from .video import VideoSample, run_video  # noqa: E402

__all__ = [
    "AUDIO_DEFECTS",
    "AUTO",
    "IMAGE_BUDGETS",
    "METRIC_SET_VERSION",
    "MIN_PROMPTS",
    "REFERENCE_BUDGETS",
    "VIDEO_BUDGETS",
    "Audio",
    "AudioChecklist",
    "AudioJudge",
    "AudioSample",
    "AudioVerdict",
    "BackendError",
    "BenchmarkResult",
    "Budget",
    "ChangeKind",
    "Checklist",
    "ChecklistItem",
    "ChecklistSet",
    "ClipScore",
    "ComputeBudget",
    "ConfigError",
    "CozyEvalError",
    "DataError",
    "DecodeError",
    "Defect",
    "DetailVerdict",
    "DimensionSummary",
    "EditChecklist",
    "GateReport",
    "Judge",
    "Lane",
    "Measurement",
    "MetricSpec",
    "NullControl",
    "OutputIntegrity",
    "ParityVerdict",
    "PromptSet",
    "Protocol",
    "ProtocolError",
    "RegistryError",
    "Sample",
    "SampleRow",
    "SampleSizeError",
    "SoftJudge",
    "SuiteReport",
    "SyncEstimate",
    "Threshold",
    "TrackVerdict",
    "TrajectoryPerturbingError",
    "Transcriber",
    "Verdict",
    "VideoChecklist",
    "VideoSample",
    "__version__",
    "add_checklist_set",
    "add_prompt_set",
    "as_audio",
    "audio",
    "audio_verdict",
    "catalog",
    "checklist_set",
    "checklists_for",
    "classify_change",
    "decompose",
    "detail",
    "detail_verdict",
    "device",
    "distributional",
    "errors",
    "evaluate",
    "imaging",
    "integrity",
    "iter_frames",
    "judge",
    "load_checklists",
    "measure_null_control",
    "metrics",
    "output_integrity",
    "probe",
    "probe_audio",
    "promptset",
    "read_audio",
    "register",
    "registry",
    "require_reference_lane",
    "resolve_device",
    "resources",
    "run_audio",
    "run_image_population_gate",
    "run_population_gate",
    "run_reference_gate",
    "run_video",
    "score_clip",
    "score_detail_vlm",
    "score_image",
    "score_pairs",
    "suite",
    "sync_offset",
    "temporal",
    "track_verdict",
    "tracks",
    "verdict",
    "video",
]
