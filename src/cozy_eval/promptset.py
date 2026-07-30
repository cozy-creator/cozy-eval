"""Prompt sets, and the rule that a checklist set must match the one it describes.

STABILITY: locked core (v0.x).

A benchmark is prompts + the assertions that say what each prompt asked for. Ship
them apart and they drift: a prompt gets reworded, its checklist keeps grading
the old wording, and the score silently stops meaning anything. So the two are
versioned together and :func:`checklists_for` refuses a mismatch at load rather
than letting it surface as an unexplained zero at eval time.

Bundled: ``hard-eval-v1`` — 20 text-to-image prompts and 8 edit triads,
deliberately hard, covering readable in-image text, hands, and faces across
photoreal and stylized registers. Each row carries a fixed seed so a run
reproduces. ``hard-video-v1`` — 16 text-to-video prompts (seeds 301-316)
covering motion, camera moves, temporal identity, physics, and text under
motion; FROZEN — a reworded row is a new set, never an edit in place.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import msgspec

from .catalog import checklist_set
from .errors import ConfigError, DataError
from .metrics.adherence import ChecklistSet

_DATA = Path(__file__).parent / "promptsets"

PROMPT_SETS: dict[str, Path] = {
    "hard-eval-v1": _DATA / "hard_eval_v1.json",
    "hard-video-v1": _DATA / "hard_video_v1.json",
}

# Prompt set -> the checklist set authored against it.
CHECKLIST_FOR: dict[str, str] = {
    "hard-eval-v1": "hard-eval-v1",
    "hard-video-v1": "hard-video-v1",
}


class Prompt(msgspec.Struct, frozen=True, kw_only=True):
    id: str
    seed: int
    prompt: str
    tags: tuple[str, ...] = ()


class EditCase(msgspec.Struct, frozen=True, kw_only=True):
    id: str
    seed: int
    instruction: str
    before_from: str          # the t2i prompt whose render is this edit's BEFORE
    tags: tuple[str, ...] = ()


class PromptSet(msgspec.Struct, frozen=True, kw_only=True):
    set_id: str
    version: int
    t2i: tuple[Prompt, ...]
    edit: tuple[EditCase, ...]
    t2v: tuple[Prompt, ...] = ()
    recipe: dict = {}
    purpose: str = ""


def add_prompt_set(name: str, path: Path | str, *, checklists: str = "") -> None:
    """Make a prompt-set file addressable by name."""
    PROMPT_SETS[name] = Path(path)
    if checklists:
        CHECKLIST_FOR[name] = checklists
    load.cache_clear()


@functools.lru_cache(maxsize=8)
def load(name: str) -> PromptSet:
    if name not in PROMPT_SETS:
        raise ConfigError(
            f"unknown prompt set {name!r}; known: {sorted(PROMPT_SETS)}. "
            "Register your own with cozy_eval.promptset.add_prompt_set(name, path)."
        )
    path = PROMPT_SETS[name]
    try:
        spec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: not valid JSON: {exc}") from exc
    except OSError as exc:
        raise DataError(f"{path}: cannot be read: {exc}") from exc
    for key in ("set_id", "version"):
        if key not in spec:
            raise DataError(f"{path}: prompt set is missing required key {key!r}")
    return PromptSet(
        set_id=spec["set_id"],
        version=int(spec["version"]),
        purpose=spec.get("purpose", ""),
        recipe=spec.get("recipe", {}),
        t2i=tuple(
            Prompt(
                id=r["id"], seed=int(r["seed"]), prompt=r["prompt"],
                tags=tuple(r.get("tags", ())),
            )
            for r in spec.get("t2i", ())
        ),
        edit=tuple(
            EditCase(
                id=r["id"], seed=int(r["seed"]), instruction=r["instruction"],
                before_from=r["before_from"], tags=tuple(r.get("tags", ())),
            )
            for r in spec.get("edit", ())
        ),
        t2v=tuple(
            Prompt(
                id=r["id"], seed=int(r["seed"]), prompt=r["prompt"],
                tags=tuple(r.get("tags", ())),
            )
            for r in spec.get("t2v", ())
        ),
    )


def by_tag(name: str, tag: str) -> tuple[Prompt, ...]:
    return tuple(r for r in load(name).t2i if tag in r.tags)


def checklists_for(name: str) -> ChecklistSet:
    """The checklist set authored against this prompt set, cross-validated.

    A checklist for a prompt that does not exist, or a prompt with no checklist,
    is an authoring error and must surface HERE — not as a silent zero later.
    """
    ps = load(name)
    if name not in CHECKLIST_FOR:
        raise ConfigError(
            f"prompt set {name!r} has no registered checklist set; pass "
            "checklists=<name> to add_prompt_set, or load the checklists directly"
        )
    loaded = checklist_set(CHECKLIST_FOR[name])
    if loaded.prompt_set != ps.set_id or loaded.prompt_set_version != ps.version:
        raise DataError(
            f"checklist set {loaded.checklist_set_id!r} targets "
            f"{loaded.prompt_set}@{loaded.prompt_set_version}, "
            f"not {ps.set_id}@{ps.version} — prompts and their checklists are "
            "versioned together so a reworded prompt cannot keep an old checklist"
        )
    prompts = {r.id for r in ps.t2i}
    edits = {r.id for r in ps.edit}
    if set(loaded.t2i) != prompts:
        raise DataError(
            f"t2i checklist coverage mismatch for {name!r}: no checklist for "
            f"{sorted(prompts - set(loaded.t2i))}, checklist for nonexistent prompt "
            f"{sorted(set(loaded.t2i) - prompts)}"
        )
    if set(loaded.edit) != edits:
        raise DataError(
            f"edit checklist coverage mismatch for {name!r}: no checklist for "
            f"{sorted(edits - set(loaded.edit))}, checklist for nonexistent edit "
            f"{sorted(set(loaded.edit) - edits)}"
        )
    videos = {r.id for r in ps.t2v}
    if set(loaded.t2v) != videos:
        raise DataError(
            f"t2v checklist coverage mismatch for {name!r}: no checklist for "
            f"{sorted(videos - set(loaded.t2v))}, checklist for nonexistent prompt "
            f"{sorted(set(loaded.t2v) - videos)}"
        )
    return loaded


__all__ = [
    "CHECKLIST_FOR",
    "PROMPT_SETS",
    "EditCase",
    "Prompt",
    "PromptSet",
    "add_prompt_set",
    "by_tag",
    "checklists_for",
    "load",
]
