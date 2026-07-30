"""The registry extension point third-party metrics register through.

Registration rules are enforced AT REGISTRATION rather than discovered at read
time, so a bad spec can never reach a report.
"""

from __future__ import annotations

import msgspec
import pytest

from cozy_eval.bench import errors, registry
from cozy_eval.bench.registry import MetricSpec

SPEC = MetricSpec(
    name="ext_metric", dimension=registry.ADHERENCE, version="0.1", paired=False,
)


@pytest.fixture(autouse=True)
def _clean() -> None:
    yield
    for name in ("ext_metric", "ext_headline"):
        registry.unregister(name)


def _replace(spec: MetricSpec, **kw: object) -> MetricSpec:
    return msgspec.structs.replace(spec, **kw)


def test_a_registered_metric_joins_the_live_registry_without_touching_builtin() -> None:
    """It becomes visible to every read path — and BUILTIN, the shipped set a
    report's metric_set names, is left exactly as it was."""
    before = registry.BUILTIN
    registry.register(SPEC)
    assert registry.BY_NAME["ext_metric"] is SPEC
    assert SPEC in registry.by_dimension(registry.ADHERENCE)
    assert SPEC in registry.reference_free()
    assert "ext_metric" in registry.REPORT_ONLY
    assert any(r["name"] == "ext_metric" for r in registry.registry_json())

    assert registry.BUILTIN is before
    assert SPEC not in registry.BUILTIN

    registry.register(_replace(SPEC, gated=True), replace=True)
    assert "ext_metric" in registry.GATED
    assert "ext_metric" not in registry.REPORT_ONLY


def test_duplicate_registration_is_refused_unless_replacing() -> None:
    registry.register(SPEC)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(SPEC)
    replaced = registry.register(_replace(SPEC, version="0.2"), replace=True)
    assert registry.BY_NAME["ext_metric"].version == "0.2"
    assert replaced.version == "0.2"


def test_a_second_headline_for_a_dimension_is_refused() -> None:
    """One headline per dimension is what makes a report readable, so it is
    enforced at registration rather than discovered at read time."""
    clash = MetricSpec(
        name="ext_headline", dimension=registry.ADHERENCE, version="0.1",
        headline=True, paired=False,
    )
    with pytest.raises(ValueError, match="already has headline metric"):
        registry.register(clash)
    # and the incumbent is untouched
    assert registry.headline(registry.ADHERENCE).name == "element_recall"


@pytest.mark.parametrize("name,dimension", [
    ("Not Lower", registry.ADHERENCE),
    ("1leading", registry.ADHERENCE),
    ("has-dash", registry.ADHERENCE),
    ("_private", registry.ADHERENCE),
    ("", registry.ADHERENCE),
    ("ext_bogus", "vibes"),
])
def test_register_refuses_a_spec_that_cannot_address_a_report(
    name: str, dimension: str,
) -> None:
    """A metric name is a report key and a budget key; a dimension is a report
    section. Neither may be something a reader cannot resolve."""
    with pytest.raises(errors.RegistryError):
        registry.register(MetricSpec(name=name, dimension=dimension, version="0.1"))
