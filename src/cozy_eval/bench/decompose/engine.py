"""Case -> (prompt text, detector spec, atom checklist), all by fixed templates.

A Case is the structured ground truth of what a prompt asks for. Everything
else derives from it deterministically, so a frozen case file IS the benchmark:
same cases, same prompts, same questions, forever.

Lane routing: colour attributes and the four geometric relations are
detector-scorable and appear in the CompositionalSpec; materials, patterns and
soft relations (3D, interaction verbs) are judge-lane only and appear as atoms.
Every constraint appears in the atoms either way — the judge lane is complete,
the detector lane is the cheap subset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import msgspec

from ..metrics.adherence import KIND_VQA, ChecklistItem
from ..metrics.geneval import COLORS, CompositionalSpec, ExcludeTerm, ObjectTerm
from . import vocab
from .vocab import NUMBER_WORDS, article, plural

TASKS = ("single_object", "counting", "colors", "materials", "two_object",
         "color_attr", "position", "relation", "composite")


class Entity(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    count: int = 1
    attribute: str = ""


class Rel(msgspec.Struct, frozen=True, kw_only=True):
    kind: str
    subject: int = 0
    object: int = 1


class Case(msgspec.Struct, frozen=True, kw_only=True):
    case_id: str
    task: str
    entities: tuple[Entity, ...]
    relations: tuple[Rel, ...] = ()
    seed: int = 0


class CaseSet(msgspec.Struct, frozen=True, kw_only=True):
    set_id: str
    version: int
    cases: tuple[Case, ...]


# ---------------------------------------------------------------------------
# prompt text
# ---------------------------------------------------------------------------

def _noun_phrase(e: Entity) -> str:
    noun = f"{e.attribute} {e.name}".strip()
    if e.count == 1:
        return f"{article(noun)} {noun}"
    nouns = plural(e.name) if not e.attribute else f"{e.attribute} {plural(e.name)}"
    return f"{NUMBER_WORDS[e.count]} {nouns}"


def prompt_text(case: Case) -> str:
    phrases = [_noun_phrase(e) for e in case.entities]
    if case.relations and len(case.entities) == 2 and len(case.relations) == 1:
        rel = case.relations[0]
        a, b = phrases[rel.subject], phrases[rel.object]
        return f"a photo of {a} {rel.kind} {b}"
    if len(phrases) == 1:
        body = phrases[0]
    elif len(phrases) == 2:
        body = f"{phrases[0]} and {phrases[1]}"
    else:
        body = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    out = f"a photo of {body}"
    for rel in case.relations if len(case.entities) > 2 or len(case.relations) > 1 else ():
        a, b = case.entities[rel.subject], case.entities[rel.object]
        out += f", with the {a.name} {rel.kind} the {b.name}"
    return out


# ---------------------------------------------------------------------------
# atoms (judge lane) — complete coverage of every constraint
# ---------------------------------------------------------------------------

def atoms(case: Case) -> tuple[ChecklistItem, ...]:
    items: list[ChecklistItem] = []
    for i, e in enumerate(case.entities):
        base = f"{case.case_id}.e{i}"
        items.append(ChecklistItem(
            id=f"{base}.presence", kind=KIND_VQA,
            question=f"Is there at least one {e.name} in the image?",
        ))
        if e.count > 1:
            items.append(ChecklistItem(
                id=f"{base}.count", kind=KIND_VQA,
                question=(f"Are there exactly {NUMBER_WORDS[e.count]} "
                          f"{plural(e.name)} in the image?"),
            ))
        if e.attribute:
            noun = e.name if e.count == 1 else plural(e.name)
            verb = "is" if e.count == 1 else "are"
            det = "the" if e.count == 1 else f"all {NUMBER_WORDS[e.count]}"
            items.append(ChecklistItem(
                id=f"{base}.attr", kind=KIND_VQA,
                question=f"{verb.capitalize()} {det} {noun} {e.attribute}?",
            ))
    for j, rel in enumerate(case.relations):
        a, b = case.entities[rel.subject], case.entities[rel.object]
        items.append(ChecklistItem(
            id=f"{case.case_id}.r{j}", kind=KIND_VQA,
            question=f"Is the {a.name} {rel.kind} the {b.name} in the image?",
        ))
    return tuple(items)


# ---------------------------------------------------------------------------
# detector lane spec — the closed-form subset
# ---------------------------------------------------------------------------

def to_spec(case: Case) -> CompositionalSpec:
    """Colour attributes and geometric relations carry over; materials,
    patterns and soft relations stay judge-lane (they are still in the atoms).
    Counts are exact, GenEval-style: include(N) + exclude(N+1)."""
    geometric = {
        (r.subject): r for r in case.relations if r.kind in vocab.GEOMETRIC_RELATIONS
    }
    include = []
    exclude = []
    for i, e in enumerate(case.entities):
        rel = geometric.get(i)
        include.append(ObjectTerm(
            name=e.name, count=e.count,
            color=e.attribute if e.attribute in COLORS else "",
            relation=rel.kind if rel else "",
            relative_to=rel.object if rel else -1,
        ))
        if e.count > 1:
            exclude.append(ExcludeTerm(name=e.name, count=e.count + 1))
    return CompositionalSpec(
        spec_id=case.case_id, prompt=prompt_text(case),
        include=tuple(include), exclude=tuple(exclude),
        tags=(case.task,),
    )


# ---------------------------------------------------------------------------
# generation — deterministic grid sampling
# ---------------------------------------------------------------------------

def _pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    out = []
    for _ in range(n):
        a, b = rng.sample(vocab.OBJECTS, 2)
        out.append((a, b))
    return out


def generate(*, seed: int = 0, per_task: int = 12) -> tuple[Case, ...]:
    """The compositional benchmark grid. Same seed, same cases, always."""
    rng = random.Random(seed)
    cases: list[Case] = []

    def add(task: str, entities: list[Entity], relations: list[Rel] | None = None):
        index = sum(1 for c in cases if c.task == task)
        cases.append(Case(
            case_id=f"{task}_{index:03d}",
            task=task, entities=tuple(entities), relations=tuple(relations or ()),
            seed=rng.randrange(2**31),
        ))

    for name in rng.sample(vocab.OBJECTS, min(per_task, len(vocab.OBJECTS))):
        add("single_object", [Entity(name=name)])
    for _ in range(per_task):
        add("counting", [Entity(name=rng.choice(vocab.OBJECTS), count=rng.choice(vocab.COUNTS))])
    for _ in range(per_task):
        add("colors", [Entity(
            name=rng.choice(vocab.INANIMATE), attribute=rng.choice(vocab.COLORS),
        )])
    for _ in range(per_task):
        add("materials", [Entity(
            name=rng.choice(vocab.INANIMATE),
            attribute=rng.choice(vocab.MATERIALS + vocab.PATTERNS),
        )])
    for a, b in _pairs(rng, per_task):
        add("two_object", [Entity(name=a), Entity(name=b)])
    for a, b in _pairs(rng, per_task):
        ca, cb = rng.sample(vocab.COLORS, 2)
        add("color_attr", [Entity(name=a, attribute=ca), Entity(name=b, attribute=cb)])
    for a, b in _pairs(rng, per_task):
        add("position", [Entity(name=a), Entity(name=b)],
            [Rel(kind=rng.choice(vocab.GEOMETRIC_RELATIONS))])
    for a, b in _pairs(rng, per_task):
        add("relation", [Entity(name=a), Entity(name=b)],
            [Rel(kind=rng.choice(vocab.SOFT_RELATIONS))])
    for _ in range(per_task):
        names = rng.sample(vocab.OBJECTS, 3)
        ents = [
            Entity(name=names[0], attribute=rng.choice(vocab.COLORS)),
            Entity(name=names[1], count=rng.choice((2, 3))),
            Entity(name=names[2], attribute=rng.choice(vocab.MATERIALS)),
        ]
        add("composite", ents, [Rel(
            kind=rng.choice(vocab.GEOMETRIC_RELATIONS + vocab.SOFT_RELATIONS),
            subject=0, object=2,
        )])
    return tuple(cases)


# ---------------------------------------------------------------------------
# serialization + suite-format emitters
# ---------------------------------------------------------------------------

def save_cases(cases: tuple[Case, ...], path: str | Path, *, set_id: str, version: int) -> None:
    Path(path).write_bytes(msgspec.json.format(
        msgspec.json.encode(CaseSet(set_id=set_id, version=version, cases=cases)), indent=2,
    ))


def load_cases(path: str | Path) -> CaseSet:
    return msgspec.json.decode(Path(path).read_bytes(), type=CaseSet)


def to_prompt_set(cs: CaseSet) -> dict:
    """The promptset.py JSON shape; write with json.dump and register via
    ``cozy_eval.bench.promptset.add_prompt_set``."""
    return {
        "set_id": cs.set_id, "version": cs.version,
        "purpose": "generated compositional benchmark (decompose.engine, deterministic)",
        "t2i": [
            {"id": c.case_id, "seed": c.seed, "prompt": prompt_text(c),
             "tags": [c.task]}
            for c in cs.cases
        ],
        "edit": [],
    }


def to_checklist_set(cs: CaseSet) -> dict:
    """The adherence checklist JSON shape (kind=vqa atoms), loadable by
    ``cozy_eval.bench.load_checklists``."""
    return {
        "checklist_set_id": f"{cs.set_id}-atoms",
        "version": cs.version,
        "prompt_set": cs.set_id,
        "prompt_set_version": cs.version,
        "t2i": [
            {"prompt_id": c.case_id,
             "items": [msgspec.to_builtins(item) for item in atoms(c)]}
            for c in cs.cases
        ],
        "edit": [],
    }


SETS_DIR = Path(__file__).parent / "sets"
BUILTIN_SET = "compositional-v1"


def register_builtin() -> CaseSet:
    """Register the frozen compositional set with the prompt/checklist
    catalogs and return its cases. Idempotent."""
    from .. import catalog, promptset

    catalog.add_checklist_set(
        f"{BUILTIN_SET}-atoms", SETS_DIR / f"{BUILTIN_SET}_checklists.json",
    )
    promptset.add_prompt_set(
        BUILTIN_SET, SETS_DIR / f"{BUILTIN_SET}_prompts.json",
        checklists=f"{BUILTIN_SET}-atoms",
    )
    return load_cases(SETS_DIR / f"{BUILTIN_SET}_cases.json")


def write_artifacts(cs: CaseSet, directory: str | Path) -> tuple[Path, Path, Path]:
    """cases + prompt set + checklist set, ready to register."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases_path = directory / f"{cs.set_id}_cases.json"
    save_cases(cs.cases, cases_path, set_id=cs.set_id, version=cs.version)
    prompts_path = directory / f"{cs.set_id}_prompts.json"
    prompts_path.write_text(json.dumps(to_prompt_set(cs), indent=2))
    checklists_path = directory / f"{cs.set_id}_checklists.json"
    checklists_path.write_text(json.dumps(to_checklist_set(cs), indent=2))
    return cases_path, prompts_path, checklists_path


__all__ = [
    "BUILTIN_SET",
    "SETS_DIR",
    "TASKS",
    "Case",
    "CaseSet",
    "Entity",
    "Rel",
    "atoms",
    "generate",
    "load_cases",
    "prompt_text",
    "register_builtin",
    "save_cases",
    "to_checklist_set",
    "to_prompt_set",
    "to_spec",
    "write_artifacts",
]
