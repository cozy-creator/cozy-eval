"""The benchmark suite: four dimensions, two modes, one report.

STABILITY: locked core (v0.x) for the REPORT SCHEMA and :func:`run`'s
call shape. The report is the part of this library that outlives the code — a
banked report has to stay readable by a version that did not write it — so its
keys, its versioning and its JSON-safety are treated as the API they are.

**paired** — candidate vs reference, same prompt and seed. All measurable
dimensions run; the verdict is a DELTA. This is what a regression gate reads.

**reference-free** — score one model's renders on their own. Similarity is
skipped (it has no meaning without a reference); everything else is an absolute
number, which is what a model leaderboard reads.

Cost discipline: every stage is timed and the totals travel IN the report, so
"the metric pass got expensive" is visible rather than folklore. Target is a
small fraction of the render pass being scored.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import msgspec

from . import registry
from .device import AUTO, resolve_device
from .errors import ConfigError, RegistryError
from .judge import Judge
from .metrics import adherence, ocr, preference, similarity

#: Version of the report STRUCTURE. Bump when a key changes meaning or leaves.
REPORT_SCHEMA = "cozy-eval/report@2"

#: Keys :attr:`SuiteReport.seconds` may carry. Stable — a dashboard may key on
#: them. Absent means the stage did not run, never means zero.
SECONDS_KEYS = (
    "lpips_load", "lpips", "pixel_metrics", "dframe", "clip", "ocr",
    "judge_load", "judge_infer", "judge_per_call", "adherence_other",
    "preference_load", "preference_infer", "signal", "audio", "detail", "total",
)

#: Keys :attr:`SuiteReport.models` may carry, mapped to the model actually used.
MODELS_KEYS = (
    "lpips", "clip", "judge", "ocr", "preference", "signal",
    "audio", "av_sync", "transcriber", "audio_judge", "detail", "detail_judge",
)


class Sample(msgspec.Struct, kw_only=True):
    """One eval row. ``checklist_id`` selects the authored element checklist;
    empty means this row has no checklist and is not scored for adherence."""

    prompt: str
    seed: int = 0
    checklist_id: str = ""
    instruction: str = ""   # edit mode: the edit instruction
    is_edit: bool = False


class SampleRow(msgspec.Struct, kw_only=True):
    """Per-sample metrics — a tail verdict must be able to name its
    own worst row, which requires the row to carry its own numbers.

    The named fields are the core metric set. ``values`` is the extension slot
    that mirrors :func:`cozy_eval.registry.register`: any metric without a
    dedicated field here — a registered third-party metric, or one of the
    compositional/quality/soft lanes — reports through it under its registry
    name, and :func:`summarize` reads it transparently.
    """

    index: int
    checklist_id: str = ""
    prompt: str = ""
    seed: int = 0
    # similarity (paired only)
    lpips: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    ms_ssim: float | None = None
    # adherence
    clip_ref: float | None = None
    clip_cand: float | None = None
    clip_delta: float | None = None
    element_recall_ref: float | None = None
    element_recall_cand: float | None = None
    element_recall_delta: float | None = None
    text_exact: float | None = None
    text_fuzzy: float | None = None
    edit_compliance: float | None = None
    edit_preservation: float | None = None
    # preference
    pref_ref: float | None = None
    pref_cand: float | None = None
    pref_delta: float | None = None
    # any other registered metric, keyed by its registry name
    values: dict[str, float] = msgspec.field(default_factory=dict)
    # which checklist items failed, so a bad row explains itself
    failed_items: list[str] = msgspec.field(default_factory=list)


class DimensionSummary(msgspec.Struct, kw_only=True):
    dimension: str
    metric: str
    measured: bool
    gated: bool
    higher_is_better: bool
    mean: float = 0.0
    worst: float = 0.0
    worst_index: int = -1
    worst_row: str = ""
    #: The SampleRow field this summary read, when it differs from the metric
    #: name (a paired delta vs a raw score). Structured, not prose.
    source_field: str = ""
    note: str = ""


class SuiteReport(msgspec.Struct, kw_only=True):
    """A run's results. Self-describing: it names its own schema version, the
    metric set that produced it, and the library version that wrote it, so a
    report banked today is still interpretable by a later version."""

    schema_version: str = REPORT_SCHEMA
    library_version: str = ""
    created_at: str = ""          # ISO-8601 UTC
    mode: str = "paired"          # "paired" | "reference-free"
    metric_set: str = registry.METRIC_SET_VERSION
    samples: int = 0
    rows: list[SampleRow] = msgspec.field(default_factory=list)
    dimensions: list[DimensionSummary] = msgspec.field(default_factory=list)
    models: dict[str, str] = msgspec.field(default_factory=dict)
    seconds: dict[str, float] = msgspec.field(default_factory=dict)
    render_seconds: float = 0.0
    metric_overhead: float = 0.0  # metric seconds / render seconds
    judge_calls: int = 0
    judge_vram_bytes: int = 0
    preference_vram_bytes: int = 0
    notes: list[str] = msgspec.field(default_factory=list)

    def summary(self, metric: str) -> DimensionSummary | None:
        return next((d for d in self.dimensions if d.metric == metric), None)


# Keys that are per-unit views of another key, not additional wall time.
_DERIVED_SECONDS = frozenset({"judge_per_call", "total"})


def _value(row: SampleRow, field: str) -> float | None:
    """A row's number for one field: a named attribute, else the ``values``
    extension slot."""
    found = getattr(row, field, None)
    if found is None:
        found = row.values.get(field)
    return None if found is None else float(found)


def _values(rows: list[SampleRow], field: str) -> list[tuple[int, float]]:
    out = []
    for row in rows:
        value = _value(row, field)
        if value is not None:
            out.append((row.index, value))
    return out


def summarize(
    rows: list[SampleRow], metric: str, *, field: str = "",
) -> DimensionSummary:
    """Mean + worst row for one metric. ``field`` names the SampleRow attribute
    (or ``values`` key) when it differs from the metric name — a paired delta
    versus a raw score."""
    spec = registry.BY_NAME.get(metric)
    if spec is None:
        raise RegistryError(
            f"cannot summarize unknown metric {metric!r}; register it first with "
            f"cozy_eval.registry.register(). registered: {sorted(registry.BY_NAME)}"
        )
    field = field or metric
    pairs = _values(rows, field)
    base = DimensionSummary(
        dimension=spec.dimension, metric=metric, measured=bool(pairs),
        gated=spec.gated, higher_is_better=spec.higher_is_better,
        source_field="" if field == metric else field,
    )
    if not pairs:
        return base
    values = [v for _, v in pairs]
    worst_idx, worst_val = (
        max(pairs, key=lambda p: p[1]) if not spec.higher_is_better
        else min(pairs, key=lambda p: p[1])
    )
    row = next((r for r in rows if r.index == worst_idx), None)
    base.mean = sum(values) / len(values)
    base.worst = worst_val
    base.worst_index = worst_idx
    base.worst_row = (row.checklist_id or f"sample {worst_idx:02d}") if row else ""
    return base


def _adherence_for(
    sample: Sample, checklists: Any, image: Any, before: Any,
    *, judge: Judge | None, use_ocr: bool, ocr_seconds: list[float],
) -> adherence.AdherenceScore | None:
    if checklists is None or not sample.checklist_id:
        return None
    if use_ocr:
        _t = time.monotonic()
        page = ocr.read(image)
        ocr_seconds.append(time.monotonic() - _t)
    else:
        page = None
    text = page.text if (page is not None and page.measured) else None
    if sample.is_edit:
        entry = checklists.edit.get(sample.checklist_id)
        if entry is None or before is None:
            return None
        return adherence.score_edit(
            entry, before, image, sample.instruction, judge=judge, after_text=text,
        )
    entry = checklists.t2i.get(sample.checklist_id)
    if entry is None:
        return None
    return adherence.score_t2i(entry, image, judge=judge, page_text=text)


def run(
    samples: list[Sample],
    candidates: list[Any],
    *,
    references: list[Any] | None = None,
    befores: list[Any] | None = None,
    checklists: Any = None,
    judge: Judge | None = None,
    use_ocr: bool = True,
    use_preference: bool = True,
    device: str = AUTO,
    render_seconds: float = 0.0,
    preference_model: str = "",
) -> SuiteReport:
    """Score a batch. ``references`` present => paired mode."""
    if len(samples) != len(candidates):
        raise ConfigError(
            f"got {len(samples)} samples but {len(candidates)} candidate images; "
            "each sample is scored against its own render, so the lists must match"
        )
    paired = references is not None
    if paired and len(references) != len(samples):
        raise ConfigError(
            f"got {len(samples)} samples but {len(references)} reference images; "
            "a paired run needs one reference per sample"
        )
    device = resolve_device(device)
    ocr_seconds: list[float] = []

    report = SuiteReport(
        mode="paired" if paired else "reference-free", samples=len(samples),
        library_version=_library_version(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    rows = [
        SampleRow(
            index=i, checklist_id=s.checklist_id, prompt=s.prompt[:200], seed=s.seed,
        )
        for i, s in enumerate(samples)
    ]

    # --- similarity (paired only) -----------------------------------------
    if paired:
        t0 = time.monotonic()
        lp = similarity.lpips_model(device)
        report.seconds["lpips_load"] = round(time.monotonic() - t0, 2)
        lpips_s = pixel_s = 0.0
        for row, _sample, ref, cand in zip(rows, samples, references, candidates, strict=True):
            if cand.size != ref.size:
                cand = cand.resize(ref.size)  # noqa: PLW2901 — scoring wants the pair aligned
            t1 = time.monotonic()
            row.lpips = similarity.lpips_pair(lp, ref, cand, device)
            lpips_s += time.monotonic() - t1
            t1 = time.monotonic()
            row.psnr = similarity.psnr(ref, cand)
            row.ssim = similarity.ssim(ref, cand)
            row.ms_ssim = similarity.ms_ssim(ref, cand)
            pixel_s += time.monotonic() - t1
        report.seconds["lpips"] = round(lpips_s, 2)
        report.seconds["pixel_metrics"] = round(pixel_s, 2)
        report.models["lpips"] = "torchmetrics:lpips-alex"

    # --- adherence: clip (continuity) -------------------------------------
    t0 = time.monotonic()
    clip, proc = similarity.clip_model(device)
    for row, sample, cand in zip(rows, samples, candidates, strict=True):
        row.clip_cand = similarity.clip_score(clip, proc, cand, sample.prompt, device)
        if paired:
            row.clip_ref = similarity.clip_score(
                clip, proc, references[row.index], sample.prompt, device
            )
            row.clip_delta = row.clip_cand - row.clip_ref
    report.seconds["clip"] = round(time.monotonic() - t0, 2)
    report.models["clip"] = similarity.CLIP_MODEL

    # --- adherence: element fidelity --------------------------------------
    t0 = time.monotonic()
    scored_any = False
    for row, sample, cand in zip(rows, samples, candidates, strict=True):
        before = befores[row.index] if befores else None
        cand_score = _adherence_for(
            sample, checklists, cand, before,
            judge=judge, use_ocr=use_ocr, ocr_seconds=ocr_seconds,
        )
        if cand_score is None or not cand_score.measured:
            continue
        scored_any = True
        row.element_recall_cand = cand_score.element_recall
        row.text_exact = cand_score.text_exact
        row.text_fuzzy = cand_score.text_fuzzy
        if sample.is_edit:
            row.edit_compliance = cand_score.compliance
            row.edit_preservation = cand_score.preservation
        row.failed_items = [v.id for v in cand_score.items if not v.verified]
        if paired:
            ref_score = _adherence_for(
                sample, checklists, references[row.index], before,
                judge=judge, use_ocr=use_ocr, ocr_seconds=ocr_seconds,
            )
            if ref_score is not None and ref_score.measured:
                row.element_recall_ref = ref_score.element_recall
                row.element_recall_delta = (
                    cand_score.element_recall - ref_score.element_recall
                )
    adherence_s = time.monotonic() - t0
    ocr_s = sum(ocr_seconds)
    report.seconds["ocr"] = round(ocr_s, 2)
    if judge is not None:
        report.seconds["judge_load"] = round(getattr(judge, "load_seconds", 0.0), 2)
        report.seconds["judge_infer"] = round(getattr(judge, "infer_seconds", 0.0), 2)
        report.judge_calls = int(getattr(judge, "call_count", 0) or 0)
        report.judge_vram_bytes = int(getattr(judge, "vram_bytes", 0))
        if report.judge_calls:
            report.seconds["judge_per_call"] = round(
                getattr(judge, "infer_seconds", 0.0) / report.judge_calls, 2
            )
    report.seconds["adherence_other"] = round(
        max(0.0, adherence_s - ocr_s - getattr(judge, "infer_seconds", 0.0)), 2
    )
    if scored_any:
        if judge is not None:
            report.models["judge"] = getattr(judge, "model_ref", adherence.DEFAULT_JUDGE)
        if use_ocr:
            report.models["ocr"] = adherence.DEFAULT_OCR
    else:
        report.notes.append(
            "element fidelity UNMEASURED: no checklist matched these samples, or "
            "neither a judge VLM nor an OCR reader was available on this image"
        )

    # --- preference -------------------------------------------------------
    if use_preference:
        t0 = time.monotonic()
        prompts = [s.prompt for s in samples]
        pref_kw = {"device": device}
        if preference_model:
            pref_kw["model_ref"] = preference_model
        cand_pref = preference.primary_scores(candidates, prompts, **pref_kw)
        report.seconds["preference_load"] = round(cand_pref.load_seconds, 2)
        if cand_pref.measured:
            report.models["preference"] = cand_pref.model_ref
            for row, score in zip(rows, cand_pref.scores, strict=True):
                row.pref_cand = score
            if paired:
                ref_pref = preference.primary_scores(references, prompts, **pref_kw)
                for row, delta in zip(rows, preference.deltas(cand_pref, ref_pref), strict=True):
                    row.pref_ref = ref_pref.scores[row.index]
                    row.pref_delta = delta
        else:
            report.notes.append(f"preference UNMEASURED: {cand_pref.note}")
        total_pref = time.monotonic() - t0
        report.seconds["preference_infer"] = round(
            max(0.0, total_pref - report.seconds.get("preference_load", 0.0)), 2
        )
        report.preference_vram_bytes = int(cand_pref.vram_bytes)

    report.rows = rows
    # (metric name in the registry, the row field that carries it). They differ
    # where a metric reads a delta in paired mode and a raw score without one.
    reported: tuple[tuple[str, str], ...] = (
        ("lpips", "lpips"), ("psnr", "psnr"), ("ssim", "ssim"), ("ms_ssim", "ms_ssim"),
        ("clip_delta", "clip_delta"),
        ("element_recall", "element_recall_delta" if paired else "element_recall_cand"),
        ("text_exact", "text_exact"), ("text_fuzzy", "text_fuzzy"),
        ("edit_compliance", "edit_compliance"),
        ("edit_preservation", "edit_preservation"),
        ("pref_delta", "pref_delta"), ("pref_score", "pref_cand"),
    )
    core = {name for name, _ in reported}
    for metric, field in reported:
        summary = summarize(rows, metric, field=field)
        if summary.measured:
            report.dimensions.append(summary)
    # Anything a caller wrote into SampleRow.values summarizes the same way, so
    # a registered metric this loop knows nothing about still reaches the report.
    for name in sorted({k for row in rows for k in row.values} - core):
        if name in registry.BY_NAME:
            summary = summarize(rows, name)
            if summary.measured:
                report.dimensions.append(summary)

    total = round(sum(
        v for k, v in report.seconds.items() if k not in _DERIVED_SECONDS
    ), 2)
    report.seconds["total"] = total
    report.render_seconds = round(float(render_seconds), 2)
    if render_seconds > 0:
        report.metric_overhead = round(total / float(render_seconds), 4)
    return report


def _library_version() -> str:
    from . import __version__

    return __version__


def free_models() -> None:
    """Drop every cached scoring model, in every metric module that has been
    imported. Modules never imported are left alone rather than imported just
    to clear an empty cache."""
    import importlib
    import sys

    for name in (
        "similarity", "preference", "ocr", "iqa", "geneval", "musiq",
    ):
        module = sys.modules.get(f"{__package__}.metrics.{name}")
        if module is None:
            continue
        importlib.import_module(module.__name__).free_models()


__all__ = [
    "MODELS_KEYS",
    "REPORT_SCHEMA",
    "SECONDS_KEYS",
    "DimensionSummary",
    "Sample",
    "SampleRow",
    "SuiteReport",
    "free_models",
    "run",
    "summarize",
]
