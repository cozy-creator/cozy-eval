"""Named checklist sets, so a caller names a set instead of carrying one.

STABILITY: locked core (v0.x).

A checklist is evaluation DATA that must be versioned and reviewable, not
something a caller invents per request — otherwise two runs of "the same"
benchmark are not comparable and nobody can tell.

Register your own with :func:`add_checklist_set`; the bundled sets are a
starting point, not the whole world.
"""

from __future__ import annotations

from pathlib import Path

from .errors import ConfigError
from .metrics.adherence import ChecklistSet, load_checklists

_HERE = Path(__file__).parent
_DATA = _HERE / "checklists"

CHECKLIST_SETS: dict[str, Path] = {
    # Curated hard set: 20 t2i checklists + 8 before/instruction/after triads.
    "hard-eval-v1": _DATA / "cozy_hard_eval_v1.json",
    # Magicbrush-style edit instructions over real photographs, 20 triads.
    "magicbrush-edit-v1": _DATA / "magicbrush_edit_v1.json",
    # 16 t2v motion/hold checklists for the hard video prompt set.
    "hard-video-v1": _DATA / "hard_video_v1.json",
}

_CACHE: dict[str, ChecklistSet] = {}


def add_checklist_set(name: str, path: Path | str) -> None:
    """Make a checklist file addressable by name."""
    CHECKLIST_SETS[name] = Path(path)
    _CACHE.pop(name, None)


def checklist_set(name: str) -> ChecklistSet:
    if name not in CHECKLIST_SETS:
        raise ConfigError(
            f"unknown checklist set {name!r}; known: {sorted(CHECKLIST_SETS)}. "
            "Register your own with cozy_eval.bench.catalog.add_checklist_set(name, path)."
        )
    if name not in _CACHE:
        _CACHE[name] = load_checklists(CHECKLIST_SETS[name])
    return _CACHE[name]


__all__ = ["CHECKLIST_SETS", "add_checklist_set", "checklist_set"]
