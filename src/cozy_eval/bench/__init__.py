"""cozy_eval.bench — an open benchmark for image generation and image editing.

The question this suite answers is *did the model make what was asked for?*,
in four mostly-independent dimensions (similarity, adherence, preference,
quality), with exactly one gated headline number per dimension.

Two modes, for images and for video:

  reference-free   score one render on its own. What a leaderboard reads.
  paired           score a candidate against a reference render. What a
                   regression gate reads; the verdict is a DELTA.

Video runs through :func:`cozy_eval.bench.video.run_video` (modes ``video-paired``
/ ``video-reference-free``): the same four dimensions over clips, with
per-frame aggregation, worst-frame tails, a Δ-frame temporal-consistency
channel, and motion/hold checklists judged on an ordered frame strip.

Scoring backends are optional extras — importing this package does not import
torch. ``cozy_eval.bench.metrics.*`` and ``cozy_eval.bench.suite`` pull their backends in
lazily, so checklist authoring and validation run on a bare install.

STABILITY (v0.x)
----------------

Everything re-exported from this module is the **locked core**: it does not
change incompatibly within 0.x, because consumers build gates on it.

    registry      the metric table, the register() extension point, the
                  one-headline-per-dimension rule, and the metric NAMES
    suite         run() and the report schema (BenchReport / SampleRow /
                  DimensionSummary)
    verdict       the tri-state policy and its Threshold / Measurement inputs
    checklists    the checklist JSON format and load_checklists()
    promptset     prompt sets and the prompt<->checklist cross-validation
    judge         the Judge / SoftJudge protocols
    errors        the exception hierarchy
    device        the "auto" device convention

Everything else — in particular the concrete scoring backends under
``cozy_eval.bench.metrics.*`` and the generator in ``cozy_eval.bench.decompose`` — is
**experimental**: the metric NAMES those modules produce are locked by the
registry, but their functions, classes and internals may change in any 0.x
release. Each such module says so in its own docstring.
"""

from __future__ import annotations

from . import catalog, device, errors, judge, promptset, registry, suite, verdict, video
from .catalog import add_checklist_set, checklist_set
from .device import AUTO, resolve_device
from .errors import (
    BackendError,
    ConfigError,
    CozyBenchError,
    DataError,
    RegistryError,
)
from .judge import Judge, SoftJudge
from .metrics.adherence import (
    Checklist,
    ChecklistItem,
    ChecklistSet,
    EditChecklist,
    VideoChecklist,
    load_checklists,
)
from .promptset import PromptSet, add_prompt_set, checklists_for
from .registry import METRIC_SET_VERSION, MetricSpec, register
from .suite import BenchReport, BenchSample, DimensionSummary, SampleRow
from .verdict import Budget, Measurement, Threshold, Verdict, evaluate
from .video import VideoSample, run_video

from cozy_eval import __version__

__all__ = [
    "AUTO",
    "METRIC_SET_VERSION",
    "BackendError",
    "BenchReport",
    "BenchSample",
    "Budget",
    "Checklist",
    "ChecklistItem",
    "ChecklistSet",
    "ConfigError",
    "CozyBenchError",
    "DataError",
    "DimensionSummary",
    "EditChecklist",
    "Judge",
    "Measurement",
    "MetricSpec",
    "PromptSet",
    "RegistryError",
    "SampleRow",
    "SoftJudge",
    "Threshold",
    "Verdict",
    "VideoChecklist",
    "VideoSample",
    "__version__",
    "add_checklist_set",
    "add_prompt_set",
    "catalog",
    "checklist_set",
    "checklists_for",
    "device",
    "errors",
    "evaluate",
    "judge",
    "load_checklists",
    "promptset",
    "register",
    "registry",
    "resolve_device",
    "run_video",
    "suite",
    "verdict",
    "video",
]
