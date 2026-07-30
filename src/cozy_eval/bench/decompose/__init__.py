"""Deterministic prompt decomposition and generation.

STABILITY: experimental (v0.x). The EMITTED formats are stable — they are the
locked prompt-set and checklist JSON shapes — but this package's own types and
functions may change.

One structured :class:`~cozy_eval.bench.decompose.engine.Case` is the source of
truth; the prompt TEXT, the detector-lane :class:`CompositionalSpec`, and the
judge-lane atom checklist are all derived from it by fixed templates. That is
what makes the benchmark reproducible: no eval-time LLM decomposition, no
unversioned question generator — the decomposition IS the data, versioned.

Method: template decomposition into atomic yes/no claims follows GenEval2's
Soft-TIFA design (arXiv:2512.16853; CC-BY-NC code/data NOT used — templates,
vocabulary, and prompts here are our own) and TIFA/DSG (Apache-2.0).
"""

from .engine import (
    BUILTIN_SET,
    SETS_DIR,
    TASKS,
    Case,
    CaseSet,
    Entity,
    Rel,
    atoms,
    generate,
    load_cases,
    prompt_text,
    register_builtin,
    save_cases,
    to_checklist_set,
    to_prompt_set,
    to_spec,
    write_artifacts,
)
from .vocab import (
    ANIMATE,
    ATTRIBUTES,
    COUNTS,
    GEOMETRIC_RELATIONS,
    INANIMATE,
    OBJECTS,
    SOFT_RELATIONS,
    article,
    plural,
)

__all__ = [
    "ANIMATE",
    "ATTRIBUTES",
    "BUILTIN_SET",
    "COUNTS",
    "GEOMETRIC_RELATIONS",
    "INANIMATE",
    "OBJECTS",
    "SETS_DIR",
    "SOFT_RELATIONS",
    "TASKS",
    "Case",
    "CaseSet",
    "Entity",
    "Rel",
    "article",
    "atoms",
    "generate",
    "load_cases",
    "plural",
    "prompt_text",
    "register_builtin",
    "save_cases",
    "to_checklist_set",
    "to_prompt_set",
    "to_spec",
    "write_artifacts",
]
