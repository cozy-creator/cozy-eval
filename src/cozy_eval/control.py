"""The null-control arm: which budgets are you allowed to believe on this family?

Every threshold in :mod:`~cozy_evalmarks` was fixed on one family
at one resolution. A threshold is only a threshold where the *clean* population
sits comfortably inside it — and that is a per-family fact, not a library fact.

**Measured, not argued** (gatebackfill lane, 2026-07-27, two image families, two
controls, n=8 each). A null control is an arm with **identical weights** and
different seeds: zero model change, so anything it trips is take spread, not
damage. On images:

===========================  ===================================  ==============
budget                       null controls measured               transfers?
===========================  ===================================  ==============
``significant_features``     0 and 0                              **yes**
``imaging_index``            1.0087 and 1.0025 (budget .92-1.25)  **yes**
``imaging_worst_prompt``     0.9182 and **0.6102** (budget .85)   family-dependent
``population_frechet``       **0.4688** and **1.3275** (b. 0.20)  **no** (2.3x, 6.6x)
===========================  ===================================  ==============

A candidate that "fails" ``population_frechet`` on such a family has told you
nothing: its zero-change control fails it harder. Without the control that reads
as a confident FAIL, and one of those two families would have been demoted for a
threshold-transfer artifact.

So the control is first-class here. :func:`measure_null_control` runs the same
benchmarks over the control arm and returns a :class:`NullControl` whose
``untrusted`` set is handed to a gate:

.. code-block:: python

    control = measure_null_control(control_pairs, control_protocol)
    report = run_population_gate(pairs, protocol, null_control=control)

The gate then: reports every budget the control trips as **disregarded** (still
measured, still printed, never decisive), fails only on budgets the control
proved trustworthy, and — because a PASS on partial evidence is not a PASS —
ceilings the verdict at INDETERMINATE whenever anything was disregarded.

Cost: one extra arm on a pod you already bought ($0.13 of a $0.38 pod, measured).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import benchmarks as B
from .metrics.signal import ClipScore
from .protocol import ChangeKind, Protocol, ProtocolError

Pair = tuple[ClipScore, ClipScore]

#: Protocol fields a control must share with the run it validates. A control
#: rendered at another resolution, step count or execution lane says nothing
#: about the budgets of the run under test.
CONTROL_MUST_MATCH = ("family", "width", "height", "frames", "steps",
                      "execution_lane", "hardware")


@dataclass(frozen=True, slots=True)
class NullControl:
    """What a zero-change arm measured, and therefore what you may conclude."""

    protocol: Protocol
    #: Every value the benchmarks produced for the control arm.
    values: dict[str, float] = field(default_factory=dict)
    #: budget name -> did the control stay inside it (i.e. does it transfer here).
    transfers: dict[str, bool] = field(default_factory=dict)
    #: budget name -> why it cannot decide a verdict for this family.
    untrusted: dict[str, str] = field(default_factory=dict)
    #: Budgets the control could not measure (e.g. the population test below its
    #: sample floor). No evidence either way; treated as trusted but unproven.
    unverified: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return self.protocol.n

    def mismatches(self, protocol: Protocol) -> list[str]:
        """Render conditions on which this control does not describe ``protocol``."""
        out = []
        for f in CONTROL_MUST_MATCH:
            a, b = getattr(self.protocol, f), getattr(protocol, f)
            if a != b:
                out.append(f"{f}: control {a!r} vs candidate {b!r}")
        return out

    def line(self) -> str:
        if not self.untrusted:
            good = ", ".join(sorted(self.transfers))
            return f"null control n={self.n}: every budget transfers ({good})"
        return ("null control n={}: DOES NOT TRANSFER — {}".format(
            self.n, "; ".join(f"{k} measured {self.values[k]:.4f}"
                              for k in sorted(self.untrusted))))

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol.to_dict(), "values": self.values,
                "transfers": self.transfers, "untrusted": self.untrusted,
                "unverified": list(self.unverified), "summary": self.line()}


def measure_null_control(
    pairs: list[Pair],
    protocol: Protocol,
    *,
    budgets: tuple[B.Budget, ...] = B.VIDEO_BUDGETS,
    temporal: bool | None = None,
) -> NullControl:
    """Score a zero-change arm and rule on which of ``budgets`` transfer.

    ``pairs`` is ``[(arm_a_score, arm_b_score), ...]`` for two draws of the SAME
    checkpoint at different seeds, in the protocol's prompt order. ``protocol``
    must declare :attr:`ChangeKind.SEED` and nothing else — a control with a
    model change in it is not a control, and this raises rather than pretending
    otherwise.

    ``temporal`` defaults to "whenever the clips have more than one frame".
    """
    if tuple(protocol.changes) != (ChangeKind.SEED,):
        raise ProtocolError(
            f"a null control declares exactly (ChangeKind.SEED,), got {protocol.changes}: "
            "the arm must be the same weights at a different seed. Any other change "
            "makes it a candidate, and a candidate cannot validate its own budgets."
        )
    if not pairs:
        raise ProtocolError("a null control needs paired renders")
    degenerate = [i for i, (a, b) in enumerate(pairs) if a.is_degenerate or b.is_degenerate]
    if degenerate:
        raise ProtocolError(
            f"null control prompts {degenerate} carry no signal (constant frames): "
            "a degenerate control measures nothing and cannot validate any budget."
        )

    if temporal is None:
        temporal = any(a.n_frames > 1 for a, _ in pairs)
    results = [B.imaging(pairs, budgets), B.distributional(pairs, budgets)]
    if temporal:
        results.append(B.temporal(pairs, budgets))

    values: dict[str, float] = {}
    for r in results:
        values.update(r.values)

    transfers: dict[str, bool] = {}
    untrusted: dict[str, str] = {}
    unverified: list[str] = []
    for b in budgets:
        if b.name not in values:
            unverified.append(b.name)
            continue
        ok = b.check(values[b.name])
        transfers[b.name] = ok
        if not ok:
            untrusted[b.name] = (
                f"the null control — identical weights, seeds only — measured "
                f"{values[b.name]:.4f} against {b.describe()}, so this budget does not "
                f"transfer to {protocol.family} at {protocol.resolution}. It is take "
                "spread, not damage, and it cannot decide a verdict here."
            )
    return NullControl(protocol, values, transfers, untrusted, tuple(unverified))


__all__ = ["CONTROL_MUST_MATCH", "NullControl", "measure_null_control"]
