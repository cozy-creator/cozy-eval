"""The v0.x stability contract, enforced rather than documented.

Every test here guards a promise the README makes to an outside user. They are
deliberately about the SHAPE of the public surface — names, versions,
serialization, error types — not about whether any particular metric is right.

Tests named ``test_regression_*`` pin a bug this library actually shipped or
nearly shipped; each one fails if that specific defect is reintroduced.
"""

from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys

import msgspec
import pytest

import cozy_eval
from cozy_eval import errors, registry, suite

# Modules that must never register a metric, import torch, or otherwise have a
# side effect at import time.
METRIC_MODULES = (
    "adherence", "similarity", "reference", "preference", "ocr", "geneval",
    "quality", "musiq", "vqascore", "hpsv3", "temporal", "signal",
    "distributional",
)

#: The locked core: modules the README promises by name off the package root.
CORE_MODULES = (
    "catalog", "decompose", "device", "errors", "judge", "metrics", "promptset",
    "registry", "suite", "verdict", "video",
)


# ---------------------------------------------------------------------------
# the registry is complete, self-describing, and independent of import order
# ---------------------------------------------------------------------------

def test_regression_every_dimension_has_a_headline_on_a_bare_import() -> None:
    """PIN — the registry import-order crash. QUALITY was declared in
    DIMENSIONS while its metrics registered only as a side effect of importing
    metrics.iqa, so `headline("quality")` raised for anyone who had not
    imported that module — including the shipped example. Runs in a subprocess
    because the defect is invisible once any test has imported a metric."""
    out = subprocess.run(
        [sys.executable, "-c", (
            "import cozy_eval, cozy_eval.registry as r, json, sys\n"
            "print(json.dumps({\n"
            "  'headlines': {d: r.headline(d).name for d in r.DIMENSIONS},\n"
            "  'count': len(r.REGISTRY),\n"
            "  'heavy': [m for m in ('torch','transformers','torchmetrics','scipy')\n"
            "            if m in sys.modules],\n"
            "}))"
        )],
        capture_output=True, text=True, check=True,
    )
    got = json.loads(out.stdout)
    assert got["headlines"] == {
        "similarity": "lpips",
        "adherence": "element_recall",
        "preference": "pref_delta",
        "quality": "arniqa",
    }
    assert got["count"] == len(registry.BUILTIN)
    assert got["heavy"] == [], f"importing cozy_eval pulled in {got['heavy']}"


def test_regression_importing_a_metric_module_does_not_change_the_registry() -> None:
    """PIN — the other half of the import-order crash. Metrics are declared as
    data in registry.BUILTIN. If a module registered itself on import, the
    registry — and a report's metric_set — would depend on what the caller
    happened to import."""
    before = registry.REGISTRY
    for name in METRIC_MODULES:
        importlib.import_module(f"cozy_eval.metrics.{name}")
    assert registry.REGISTRY is before


#: Per-metric facts the individual metric test modules used to assert one
#: function at a time. Held here so the registry has ONE integrity test.
#: name -> (dimension, gated, headline, higher_is_better, paired)
EXPECTED_SPECS = {
    "lpips":              ("similarity", True,  True,  False, True),
    "psnr":               ("similarity", False, False, True,  True),
    "ssim":               ("similarity", False, False, True,  True),
    "ms_ssim":            ("similarity", False, False, True,  True),
    "lpips_frame_worst":  ("similarity", False, False, False, True),
    "dframe_psnr":        ("similarity", False, False, True,  True),
    "vmaf":               ("similarity", False, False, True,  True),
    "vmaf_min":           ("similarity", False, False, True,  True),
    "cvvdp":              ("similarity", False, False, True,  True),
    "flow_divergence":    ("similarity", False, False, False, True),
    "warp_error_delta":   ("similarity", False, False, False, True),
    "warp_error":         ("quality",    False, False, False, False),
    "element_recall":     ("adherence",  True,  True,  True,  False),
    "clip_delta":         ("adherence",  True,  False, True,  True),
    "vqascore":           ("adherence",  False, False, True,  False),
    "atom_score":         ("adherence",  False, False, True,  False),
    "composition_recall": ("adherence",  False, False, True,  False),
    "motion_compliance":  ("adherence",  False, False, True,  False),
    "pref_delta":         ("preference", True,  True,  True,  True),
    "pref_score":         ("preference", False, False, True,  False),
    "hpsv3_score":        ("preference", False, False, True,  False),
    "arniqa":             ("quality",    False, True,  True,  False),
    "clip_iqa":           ("quality",    False, False, True,  False),
    "niqe":               ("quality",    False, False, False, False),
    "musiq":              ("quality",    False, False, True,  False),
    "luma_flicker":       ("quality",    False, False, False, False),
}

#: Published, named methods keep their published name; everything we derive
#: ourselves is <subject>_<quantity>.
PUBLISHED_NAMES = {
    "lpips", "ssim", "ms_ssim", "psnr", "niqe", "musiq", "arniqa",
    "clip_iqa", "vqascore", "vmaf", "cvvdp", "dists",
}


def test_registry_integrity_sweep() -> None:
    """One sweep over the whole registry, replacing the per-metric "registers
    correctly" test each metric module used to carry. Covers the naming scheme
    (names are API keys — budgets and banked reports address metrics by them),
    exactly one headline per dimension, gated/report-only partitioning,
    self-description, the reference-free set, and the declared flags of every
    metric a module-level test used to assert on its own."""
    for spec in registry.REGISTRY:
        assert registry.NAME_PATTERN.match(spec.name), spec.name
        assert spec.dimension in registry.DIMENSIONS, spec.name
        assert spec.version, spec.name
        if spec.name not in PUBLISHED_NAMES:
            assert "_" in spec.name, (
                f"{spec.name!r}: a metric we derive ourselves is <subject>_<quantity>; "
                "only a published, named method keeps a bare name"
            )
        assert (spec.name in registry.GATED) == spec.gated, spec.name
        assert (spec.name in registry.REPORT_ONLY) != spec.gated, spec.name

    for dimension in registry.DIMENSIONS:
        head = registry.headline(dimension)
        assert head.dimension == dimension
        assert len([m for m in registry.by_dimension(dimension) if m.headline]) == 1

    # Self-describing: registry_json() is the wire form of the same set.
    rows = registry.registry_json()
    assert {r["name"] for r in rows} == set(registry.BY_NAME)
    assert all(r["version"] and r["dimension"] in registry.DIMENSIONS for r in rows)
    assert registry.METRIC_SET_VERSION

    # Reference-free is the absolute-evaluation set: it scores one render.
    free = {m.name for m in registry.reference_free()}
    assert {"element_recall", "pref_score", "niqe", "arniqa"} <= free
    assert "lpips" not in free  # meaningless without a reference
    assert all(not m.paired for m in registry.by_dimension(registry.QUALITY))

    for name, expected in EXPECTED_SPECS.items():
        spec = registry.BY_NAME[name]
        assert (spec.dimension, spec.gated, spec.headline,
                spec.higher_is_better, spec.paired) == expected, name


# ---------------------------------------------------------------------------
# the public surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CORE_MODULES)
def test_locked_core_modules_are_reachable_from_the_package_root(module: str) -> None:
    """The README promises these by name as the locked core."""
    assert getattr(cozy_eval, module) is not None
    assert module in cozy_eval.__all__


def test_every_exported_name_exists_package_wide() -> None:
    """__all__ is the export list `from x import *` and every doc generator
    reads; a name in it that does not resolve is a broken public surface.
    Ordering is enforced by ruff RUF022, not re-litigated here."""
    missing = [n for n in cozy_eval.__all__ if not hasattr(cozy_eval, n)]
    assert not missing, f"cozy_eval.__all__ names that do not exist: {missing}"

    for name in (
        [f"cozy_eval.metrics.{m}" for m in METRIC_MODULES]
        + [f"cozy_eval.{m}" for m in CORE_MODULES]
    ):
        module = importlib.import_module(name)
        missing = [n for n in getattr(module, "__all__", []) if not hasattr(module, n)]
        assert not missing, f"{name}.__all__ names that do not exist: {missing}"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

def test_the_error_hierarchy_is_additive() -> None:
    """One root so a caller can catch everything, and each class still
    subclasses the builtin it replaced — so `except ValueError` written against
    an earlier version keeps working."""
    for name in errors.__all__:
        if name != "CozyEvalError":
            assert issubclass(getattr(errors, name), errors.CozyEvalError), name
    for cls in (errors.ConfigError, errors.DataError, errors.RegistryError,
                errors.ProtocolError):
        assert issubclass(cls, ValueError)
    for cls in (errors.BackendError, errors.DecodeError, errors.SampleSizeError,
                errors.TrajectoryPerturbingError):
        assert issubclass(cls, RuntimeError)


def test_unknown_names_are_reported_with_what_does_exist() -> None:
    """An actionable message names the alternatives."""
    with pytest.raises(errors.ConfigError, match="known:"):
        cozy_eval.checklist_set("nope")
    with pytest.raises(errors.ConfigError, match="known:"):
        cozy_eval.promptset.load("nope")
    with pytest.raises(errors.RegistryError, match="registered:"):
        registry.spec("nope")
    with pytest.raises(errors.RegistryError, match="registered:"):
        suite.summarize([], "nope")


# ---------------------------------------------------------------------------
# the report schema — the API that outlives the code
# ---------------------------------------------------------------------------

def test_a_report_describes_itself_and_round_trips_through_json() -> None:
    """A banked report is read back by a version that did not write it, so it
    names its own schema/metric-set/library versions and must be plain JSON —
    no NaN, no tuples, no non-string keys."""
    report = suite.SuiteReport(
        library_version=cozy_eval.__version__,
        created_at="2026-07-27T00:00:00+00:00",
        mode="paired", samples=1,
        rows=[suite.SampleRow(index=0, lpips=0.2, values={"niqe": 3.5})],
        models={"lpips": "torchmetrics:lpips-alex"},
        seconds={"total": 1.5},
    )
    assert report.schema_version == suite.REPORT_SCHEMA
    assert report.metric_set == registry.METRIC_SET_VERSION
    assert report.library_version == cozy_eval.__version__
    # Both version strings use ONE convention, so a reader can parse either.
    for value in (report.schema_version, report.metric_set):
        assert value.startswith("cozy-eval/") and "@" in value

    raw = msgspec.json.encode(report)
    json.loads(raw)  # plain-JSON parseable, not just msgspec-decodable
    assert msgspec.json.decode(raw, type=suite.SuiteReport) == report


def test_the_values_slot_carries_metrics_the_core_schema_has_no_field_for() -> None:
    """SampleRow.values is the report-side twin of registry.register(): a
    registered metric with no dedicated column still summarizes."""
    rows = [
        suite.SampleRow(index=0, values={"niqe": 3.0}),
        suite.SampleRow(index=1, values={"niqe": 5.0}),
    ]
    summary = suite.summarize(rows, "niqe")
    assert summary.measured
    assert summary.dimension == registry.QUALITY
    assert summary.mean == 4.0
    # niqe is lower-is-better, so the WORST row is the highest number.
    assert summary.worst == 5.0
    assert summary.worst_index == 1


def test_regression_summarize_reports_its_source_field_structurally() -> None:
    """PIN — the DimensionSummary note leak. The paired/unpaired field
    indirection was smuggled into `note` as the prose `field=...`, putting an
    implementation detail into the serialized schema. It is a structured field
    now, and `note` stays free for a human sentence."""
    rows = [suite.SampleRow(index=0, element_recall_delta=-0.1)]
    summary = suite.summarize(rows, "element_recall", field="element_recall_delta")
    assert summary.source_field == "element_recall_delta"
    assert summary.note == ""
    # A metric read from its own name declares no indirection at all.
    assert suite.summarize(rows, "lpips").source_field == ""


def test_regression_unmeasured_is_not_zero_and_zero_is_not_unmeasured() -> None:
    """PIN — the falsy-sentinel landmine. A row's value is looked up with an
    `is None` test, never a truthiness test: `None` means NOBODY LOOKED (the
    whole verdict policy turns on it) while a real `0.0` is a measurement.
    Reading either with `if not value` collapses them into each other."""
    absent = suite.summarize([suite.SampleRow(index=0)], "lpips")
    assert not absent.measured
    assert absent.mean == 0.0  # the struct default, guarded by `measured`

    # A genuine zero — the worst possible element_recall — is MEASURED.
    zeros = suite.summarize(
        [suite.SampleRow(index=0, element_recall_cand=0.0),
         suite.SampleRow(index=1, element_recall_cand=0.5)],
        "element_recall", field="element_recall_cand",
    )
    assert zeros.measured
    assert zeros.mean == 0.25
    assert zeros.worst == 0.0 and zeros.worst_index == 0

    # Same rule in the values extension slot.
    from_values = suite.summarize(
        [suite.SampleRow(index=0, values={"text_exact": 0.0})], "text_exact",
    )
    assert from_values.measured and from_values.worst == 0.0


# ---------------------------------------------------------------------------
# device convention and model caches
# ---------------------------------------------------------------------------

def test_auto_is_the_one_device_convention_everywhere() -> None:
    """Two lanes built this and defaulted opposite ways (cuda vs cpu). One
    convention now, checked by signature so it cannot drift back."""
    from cozy_eval import video
    from cozy_eval.metrics import (
        adherence,
        geneval,
        musiq,
        preference,
        quality,
        vqascore,
    )

    assert cozy_eval.resolve_device("cpu") == "cpu"
    assert cozy_eval.resolve_device("cuda:1") == "cuda:1"
    assert cozy_eval.resolve_device(cozy_eval.AUTO) in ("cpu", "cuda")

    entry_points = (
        suite.run, video.run_video, quality.arniqa, quality.clip_iqa,
        quality.score_frames, musiq.musiq,
        preference.primary_scores, preference.pickscore_scores,
        adherence.VlmJudge.__init__, vqascore.VlmSoftJudge.__init__,
        geneval.GroundingDino.__init__, geneval.SiglipColors.__init__,
    )
    for func in entry_points:
        param = inspect.signature(func).parameters.get("device")
        assert param is not None and param.default == cozy_eval.AUTO, func.__qualname__
    assert len(entry_points) == 12


def test_regression_model_caches_key_on_the_model_and_the_device() -> None:
    """PIN — the wrong-model cache key. `preference._scorer` cached under the
    literal key "pref", so a second `model_ref` in the same process silently
    got the FIRST model's scores; similarity's caches ignored `device` the same
    way. Loaders are primed with sentinels instead of real weights, so the key
    is exercised for real without downloading gigabytes."""
    from cozy_eval.metrics import preference, similarity

    preference.free_models()
    similarity.free_models()
    try:
        preference._CACHE["pref:org/model-a:cpu"] = ("model-a", "proc-a")
        preference._CACHE["pref:org/model-b:cpu"] = ("model-b", "proc-b")
        assert preference._scorer("org/model-a", "cpu") == ("model-a", "proc-a")
        assert preference._scorer("org/model-b", "cpu") == ("model-b", "proc-b")

        similarity._MODEL_CACHE["lpips:alex:cpu"] = "lpips-on-cpu"
        similarity._MODEL_CACHE["lpips:alex:cuda:3"] = "lpips-on-cuda3"
        assert similarity.lpips_model("cpu") == "lpips-on-cpu"
        assert similarity.lpips_model("cuda:3") == "lpips-on-cuda3"

        similarity._MODEL_CACHE[f"clip:{similarity.CLIP_MODEL}:cpu"] = "clip-default"
        similarity._MODEL_CACHE["clip:other/clip:cpu"] = "clip-other"
        assert similarity.clip_model("cpu") == "clip-default"
        assert similarity.clip_model("cpu", model_ref="other/clip") == "clip-other"
    finally:
        preference.free_models()
        similarity.free_models()


def test_free_models_covers_every_module_that_caches_one() -> None:
    """A cache that free_models() forgets is a leaked multi-GB model."""
    for name in METRIC_MODULES:
        module = importlib.import_module(f"cozy_eval.metrics.{name}")
        caches = [
            v for k, v in vars(module).items()
            if k.endswith("_CACHE") or k == "_CACHE"
        ]
        if caches:
            assert hasattr(module, "free_models"), f"{name} caches but cannot be freed"
    suite.free_models()  # must not raise with backends imported
