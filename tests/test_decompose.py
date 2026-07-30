"""The decompose engine is the benchmark's data source, so the tests pin the
contract hard: same seed -> byte-identical artifacts, every constraint appears
in the atoms, and the emitted files pass the suite's own validating loaders.
"""

import json

import pytest

from cozy_eval.bench.decompose import engine, vocab
from cozy_eval.bench.decompose.engine import Case, CaseSet, Entity, Rel
from cozy_eval.bench.metrics.adherence import load_checklists
from cozy_eval.bench.metrics.geneval import validate_spec


def case(task="two_object", entities=(), relations=(), case_id="c0"):
    return Case(case_id=case_id, task=task, entities=tuple(entities), relations=tuple(relations))


# ---------------------------------------------------------------------------
# vocab
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,word,expected", [
    ("article", "owl", "an"),
    ("article", "dog", "a"),
    ("plural", "sheep", "sheep"),
    ("plural", "butterfly", "butterflies"),
    ("plural", "bus", "buses"),
    ("plural", "dog", "dogs"),
])
def test_article_and_plural_rules(fn, word, expected):
    assert getattr(vocab, fn)(word) == expected


def test_vocab_lanes_are_disjoint():
    """A relation is geometric (checkable by boxes) or soft (judge only), never
    both — the two lanes score through different machinery."""
    assert not set(vocab.GEOMETRIC_RELATIONS) & set(vocab.SOFT_RELATIONS)
    assert len(set(vocab.OBJECTS)) == 40


# ---------------------------------------------------------------------------
# prompt text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c,expected", [
    pytest.param(case(entities=[Entity(name="owl")]),
                 "a photo of an owl", id="single"),
    pytest.param(case(entities=[Entity(name="cat", count=3)]),
                 "a photo of three cats", id="counting"),
    pytest.param(case(entities=[Entity(name="car", attribute="red")]),
                 "a photo of a red car", id="attribute"),
    pytest.param(case(entities=[Entity(name="dog"), Entity(name="teapot")]),
                 "a photo of a dog and a teapot", id="two_object"),
    pytest.param(case(entities=[Entity(name="cat"), Entity(name="dog")],
                      relations=[Rel(kind="left of")]),
                 "a photo of a cat left of a dog", id="relation_phrasing"),
    pytest.param(case(entities=[Entity(name="cat", attribute="red"),
                                Entity(name="dog", count=2),
                                Entity(name="chair", attribute="wooden")],
                      relations=[Rel(kind="on top of", subject=0, object=2)]),
                 "a photo of a red cat, two dogs, and a wooden chair, "
                 "with the cat on top of the chair", id="composite_appends_a_clause"),
])
def test_prompt_text(c, expected):
    assert engine.prompt_text(c) == expected


# ---------------------------------------------------------------------------
# atoms and spec projection
# ---------------------------------------------------------------------------

def test_every_constraint_becomes_an_atom():
    """Presence, attribute, count and relation each earn their own question —
    an unasked constraint is an unscored one."""
    c = case(
        entities=[Entity(name="cat", attribute="red"), Entity(name="dog", count=3)],
        relations=[Rel(kind="chasing", subject=0, object=1)],
    )
    questions = [i.question for i in engine.atoms(c)]
    # presence x2 + attribute + count + relation = 5
    assert len(questions) == 5
    assert any("at least one cat" in q for q in questions)
    assert any("at least one dog" in q for q in questions)
    assert "Is the cat red?" in questions
    assert any("exactly three dogs" in q for q in questions)
    assert "Is the cat chasing the dog in the image?" in questions

    # A counted entity's attribute is asked of ALL of them.
    plural = case(entities=[Entity(name="car", count=2, attribute="blue")])
    assert "Are all two cars blue?" in [i.question for i in engine.atoms(plural)]


@pytest.mark.parametrize("c,check", [
    pytest.param(
        case(task="counting", entities=[Entity(name="dog", count=3)]),
        lambda s: s.include[0].count == 3 and s.exclude[0].count == 4,
        id="counting_encodes_an_exact_count_as_include_n_exclude_n_plus_1",
    ),
    pytest.param(
        case(entities=[Entity(name="cat"), Entity(name="dog")],
             relations=[Rel(kind="above", subject=0, object=1)]),
        lambda s: s.include[0].relation == "above" and s.include[0].relative_to == 1,
        id="a_geometric_relation_carries_over",
    ),
    pytest.param(
        case(entities=[Entity(name="cat"), Entity(name="dog")],
             relations=[Rel(kind="chasing", subject=0, object=1)]),
        lambda s: all(t.relation == "" for t in s.include),
        id="a_soft_relation_stays_out_of_the_box_spec",
    ),
    pytest.param(
        case(entities=[Entity(name="chair", attribute="wooden"),
                       Entity(name="car", attribute="red")]),
        lambda s: s.include[0].color == "" and s.include[1].color == "red",
        id="material_stays_out_but_colour_carries",
    ),
])
def test_to_spec_projects_only_what_boxes_can_check(c, check):
    """The spec is what the DETECTOR is asked; anything a box cannot settle
    (soft relations, materials) stays behind for the judge."""
    projected = engine.to_spec(c)
    validate_spec(projected)
    assert check(projected)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def test_generate_is_deterministic_and_every_case_it_emits_is_well_formed():
    """Same seed -> same cases; different seed -> different cases. Every case
    covers a declared task, validates as a spec, and carries unique ids at both
    the case and the atom level."""
    assert engine.generate(seed=7) == engine.generate(seed=7)
    assert engine.generate(seed=7) != engine.generate(seed=8)

    cases = engine.generate(seed=0, per_task=8)
    assert {c.task for c in cases} == set(engine.TASKS)
    assert len({c.case_id for c in cases}) == len(cases)
    for c in cases:
        validate_spec(engine.to_spec(c))
        atom_ids = [i.id for i in engine.atoms(c)]
        assert len(atom_ids) == len(set(atom_ids)), c.case_id


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def test_written_artifacts_load_back_through_the_suite_loaders(tmp_path):
    """The engine's output is only useful if the benchmark's own validating
    loaders accept it — cases round-trip, prompts line up, checklists load."""
    cs = CaseSet(set_id="compo-test", version=1, cases=engine.generate(seed=3, per_task=4))
    cases_path, prompts_path, checklists_path = engine.write_artifacts(cs, tmp_path)

    assert engine.load_cases(cases_path).cases == cs.cases
    assert len(json.loads(prompts_path.read_text())["t2i"]) == len(cs.cases)
    loaded = load_checklists(checklists_path)
    assert loaded.prompt_set == "compo-test"
    assert set(loaded.t2i) == {c.case_id for c in cs.cases}


def test_the_frozen_builtin_set_has_not_drifted(tmp_path):
    """The shipped compositional-v1 files must equal a fresh generation from
    the recorded seed — the benchmark data is code output, and this pins it
    byte for byte, which also proves the write path is deterministic."""
    cs = CaseSet(set_id="compositional-v1", version=1,
                 cases=engine.generate(seed=20260727, per_task=12))
    for fresh in engine.write_artifacts(cs, tmp_path):
        shipped = engine.SETS_DIR / fresh.name
        assert shipped.read_bytes() == fresh.read_bytes(), fresh.name


def test_the_builtin_set_cross_validates_against_its_checklists():
    from cozy_eval.bench import promptset

    cases = engine.register_builtin()
    assert len(cases.cases) == 108
    loaded = promptset.checklists_for(engine.BUILTIN_SET)
    assert set(loaded.t2i) == {c.case_id for c in cases.cases}
