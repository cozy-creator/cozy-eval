"""Compositional scoring is closed-form over typed detections, so every rule is
tested here with synthetic detections and zero model weights. The rules under
test are GenEval's published semantics (arXiv:2310.11513, MIT) — parity with
the paper's decision table is the point, so the table below mirrors its tasks
(two_object, counting, colors, position, color_attr) directly, one named row
per rule.
"""

import msgspec
import pytest

from cozy_eval.metrics.geneval import (
    CompositionalSpec,
    Detection,
    ExcludeTerm,
    ObjectTerm,
    needs_counting,
    relative_position,
    score_spec,
    validate_spec,
    vocabulary,
)


def det(label, score=0.95, box=(0, 0, 10, 10), color=""):
    return Detection(label=label, score=score, box=tuple(float(v) for v in box), color=color)


def spec(spec_id="s", include=(), exclude=(), prompt="p"):
    return CompositionalSpec(
        spec_id=spec_id, prompt=prompt, include=tuple(include), exclude=tuple(exclude),
    )


def dogs(n, score=0.95):
    """n distinct dogs in a row, so NMS keeps all of them."""
    return [det("dog", score=score, box=(i * 20, 0, i * 20 + 10, 10)) for i in range(n)]


# ---------------------------------------------------------------------------
# relative_position — the dead-zone rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    pytest.param((0, 0, 10, 10), (100, 0, 110, 10), {"left of"}, id="clear_left"),
    pytest.param((100, 100, 110, 110), (0, 0, 10, 10), {"right of", "below"},
                 id="clear_right_and_below_at_45_degrees"),
    pytest.param((0, 0, 10, 10), (0, 100, 10, 110), {"above"},
                 id="above_uses_image_coords_smaller_y_is_higher"),
    pytest.param((0, 0, 50, 50), (10, 10, 60, 60), frozenset(),
                 id="overlapping_boxes_have_no_relation"),
    # Offset 15px against a 0.1*(100+100)=20px dead zone.
    pytest.param((0, 0, 100, 100), (15, 0, 115, 100), frozenset(),
                 id="dead_zone_swallows_small_offsets"),
    # Offset 30px survives the dead zone, but the normalized x-component
    # (10/30) is BELOW 0.5 — GenEval's conservative read on near-overlap.
    pytest.param((0, 0, 100, 100), (30, 0, 130, 100), frozenset(),
                 id="just_past_the_dead_zone_still_names_nothing"),
    pytest.param((0, 0, 100, 100), (300, 0, 400, 100), {"left of"},
                 id="far_offset_names_the_axis"),
])
def test_relative_position(a, b, expected):
    assert relative_position(a, b) == expected


# ---------------------------------------------------------------------------
# spec validation + helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_spec,message", [
    pytest.param(
        spec(include=[ObjectTerm(name="dog"),
                      ObjectTerm(name="cat", relation="left of", relative_to=0)]),
        None, id="a_valid_spec_passes",
    ),
    pytest.param(spec(), "empty spec", id="empty_spec"),
    pytest.param(
        spec(include=[ObjectTerm(name="a"),
                      ObjectTerm(name="b", relation="behind", relative_to=0)]),
        "unknown relation", id="unknown_relation",
    ),
    pytest.param(
        spec(include=[ObjectTerm(name="a", relation="left of", relative_to=0)]),
        "relative to itself", id="self_relation",
    ),
    pytest.param(
        spec(include=[ObjectTerm(name="a", relation="left of", relative_to=3)]),
        "out of range", id="relative_to_out_of_range",
    ),
    pytest.param(
        spec(include=[ObjectTerm(name="a", count=0)]),
        "count", id="zero_include_count",
    ),
    pytest.param(
        spec(include=[ObjectTerm(name="a")], exclude=[ExcludeTerm(name="b", count=0)]),
        "count", id="zero_exclude_count",
    ),
])
def test_validate_spec(bad_spec, message):
    if message is None:
        validate_spec(bad_spec)
        return
    with pytest.raises(ValueError, match=message):
        validate_spec(bad_spec)


def test_spec_helpers_report_the_vocabulary_and_whether_counting_is_needed():
    """`vocabulary` is what the detector is prompted with, so order and
    de-duplication matter; `needs_counting` picks the strict threshold."""
    s = spec(
        include=[ObjectTerm(name="dog"), ObjectTerm(name="cat"),
                 ObjectTerm(name="dog", color="red")],
        exclude=[ExcludeTerm(name="person")],
    )
    assert vocabulary(s) == ("dog", "cat", "person")
    assert not needs_counting(spec(include=[ObjectTerm(name="a"), ObjectTerm(name="b")]))
    assert needs_counting(spec(include=[ObjectTerm(name="a", count=3)]))


# ---------------------------------------------------------------------------
# score_spec — the GenEval decision table
# ---------------------------------------------------------------------------

CAR_AND_DOG = spec(include=[
    ObjectTerm(name="car", color="red"), ObjectTerm(name="dog", color="blue"),
])
CAT_LEFT_OF_DOG = spec(include=[
    ObjectTerm(name="dog"),
    ObjectTerm(name="cat", relation="left of", relative_to=0),
])
THREE_DOGS = spec(include=[ObjectTerm(name="dog", count=3)],
                  exclude=[ExcludeTerm(name="dog", count=4)])

# (spec, detections, correct, term_recall or None, detail fragment, failed term prefix)
DECISION_TABLE = [
    # --- single and two object -------------------------------------------
    pytest.param(spec(include=[ObjectTerm(name="dog")]), [det("dog")],
                 True, 1.0, "", "", id="single_object_present"),
    pytest.param(spec(include=[ObjectTerm(name="dog")]), [det("cat")],
                 False, 0.0, "found 0 of 1", "", id="single_object_absent"),
    pytest.param(spec(include=[ObjectTerm(name="dog")]), [det("dog", score=0.2)],
                 False, 0.0, "", "", id="a_low_confidence_detection_is_not_present"),
    pytest.param(spec(include=[ObjectTerm(name="dog"), ObjectTerm(name="cat")]),
                 [det("dog")], False, 0.5, "", "", id="two_object_partial_recall"),
    # --- counting ---------------------------------------------------------
    pytest.param(THREE_DOGS, dogs(3), True, 1.0, "", "",
                 id="exact_count_via_include_3_plus_exclude_4"),
    pytest.param(THREE_DOGS, dogs(4), False, None, "", "exclude",
                 id="a_fourth_object_breaks_the_exclude_clause"),
    # Two dogs, one at 0.5: counts at conf 0.3 but not at counting conf 0.9.
    pytest.param(spec(include=[ObjectTerm(name="dog", count=2)]),
                 [det("dog", score=0.95), det("dog", score=0.5, box=(20, 0, 30, 10))],
                 False, None, "", "", id="counting_uses_the_strict_threshold"),
    pytest.param(spec(include=[ObjectTerm(name="dog", count=1)]),
                 [det("dog", score=0.95), det("dog", score=0.5, box=(20, 0, 30, 10))],
                 True, None, "", "", id="the_loose_threshold_still_finds_one"),
    # A phantom 0.5-score fourth dog must NOT fail the exclude clause when the
    # spec is a counting spec (the whole image is read at 0.9).
    pytest.param(THREE_DOGS, [*dogs(3), det("dog", score=0.5, box=(80, 0, 90, 10))],
                 True, None, "", "", id="exclude_shares_the_strict_threshold"),
    # Identical boxes are exact duplicates: NMS at IoU 1.0 keeps one.
    pytest.param(spec(include=[ObjectTerm(name="dog", count=2)]),
                 [det("dog"), det("dog", score=0.93)],
                 False, None, "", "", id="duplicate_boxes_are_deduped"),
    # --- colors -----------------------------------------------------------
    pytest.param(spec(include=[ObjectTerm(name="car", color="red")]),
                 [det("car", color="red")], True, 1.0, "", "", id="color_match"),
    pytest.param(spec(include=[ObjectTerm(name="car", color="red")]),
                 [det("car", color="blue")], False, 0.0, "", "", id="color_mismatch"),
    pytest.param(spec(include=[ObjectTerm(name="car", color="red")]), [det("car")],
                 False, 0.0, "", "", id="an_unclassified_detection_is_not_a_color_match"),
    # "a red car and a blue dog": binding, not bag-of-colors.
    pytest.param(CAR_AND_DOG,
                 [det("car", color="red"), det("dog", color="blue", box=(20, 0, 30, 10))],
                 True, 1.0, "", "", id="color_attr_binds_each_colour_to_its_object"),
    pytest.param(CAR_AND_DOG,
                 [det("car", color="blue"), det("dog", color="red", box=(20, 0, 30, 10))],
                 False, 0.0, "", "", id="swapped_colours_fail_both_terms"),
    # --- position ---------------------------------------------------------
    pytest.param(CAT_LEFT_OF_DOG,
                 [det("dog", box=(200, 0, 220, 20)), det("cat", box=(0, 0, 20, 20))],
                 True, 1.0, "", "", id="relation_satisfied"),
    pytest.param(CAT_LEFT_OF_DOG,
                 [det("dog", box=(0, 0, 20, 20)), det("cat", box=(200, 0, 220, 20))],
                 False, None, "relation not observed", "", id="relation_violated"),
    pytest.param(CAT_LEFT_OF_DOG, [det("cat", box=(0, 0, 20, 20))],
                 False, None, "", "", id="a_relation_needs_both_objects"),
    # Two cats, only one is left of the dog — exists-semantics passes.
    pytest.param(CAT_LEFT_OF_DOG,
                 [det("dog", box=(200, 0, 220, 20)), det("cat", box=(0, 0, 20, 20)),
                  det("cat", box=(400, 0, 420, 20))],
                 True, None, "", "", id="any_satisfying_pair_counts"),
    # "a dog left of a dog" must not match one dog against itself.
    pytest.param(spec(include=[ObjectTerm(name="dog"),
                               ObjectTerm(name="dog", relation="left of", relative_to=0)]),
                 [det("dog", box=(0, 0, 20, 20))],
                 False, None, "", "", id="a_same_class_relation_needs_distinct_instances"),
]


@pytest.mark.parametrize(
    "compositional_spec,detections,correct,recall,detail,failed_prefix", DECISION_TABLE,
)
def test_score_spec_decision_table(
    compositional_spec, detections, correct, recall, detail, failed_prefix,
):
    out = score_spec(compositional_spec, detections)
    assert out.correct is correct
    if recall is not None:
        assert out.term_recall == recall
    if detail:
        assert any(detail in v.detail for v in out.terms), [v.detail for v in out.terms]
    if failed_prefix:
        assert any(v.term.startswith(failed_prefix) and not v.satisfied for v in out.terms)


def test_a_spec_and_its_score_round_trip_through_json():
    """Both structs are report payloads, so they must survive a plain-JSON
    encode/decode with no custom hooks."""
    s = spec(
        include=[ObjectTerm(name="car", color="red", relation="above", relative_to=1),
                 ObjectTerm(name="dog", count=2)],
        exclude=[ExcludeTerm(name="dog", count=3)],
    )
    assert msgspec.json.decode(msgspec.json.encode(s), type=CompositionalSpec) == s

    out = score_spec(spec(include=[ObjectTerm(name="dog")]), [det("dog")])
    rebuilt = msgspec.convert(msgspec.to_builtins(out), type(out))
    assert rebuilt.correct == out.correct


@pytest.mark.heavy
class TestBackends:
    """Real Grounding DINO + SigLIP2 on photographic-ish synthetic scenes.
    Structure and gross discrimination, not benchmark accuracy."""

    @pytest.fixture(scope="class")
    def detector(self):
        from cozy_eval.metrics.geneval import GroundingDino

        return GroundingDino(device="cpu")

    def _scene(self):
        # A red disc left of a blue square on gray — unambiguous geometry.
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (512, 320), (128, 128, 128))
        d = ImageDraw.Draw(im)
        d.ellipse((40, 100, 160, 220), fill=(210, 30, 30))
        d.rectangle((330, 100, 470, 240), fill=(30, 60, 210))
        return im

    def test_detector_returns_typed_detections_and_siglip_names_their_colors(
        self, detector,
    ):
        from cozy_eval.metrics.geneval import Detection, SiglipColors

        scene = self._scene()
        dets = detector.detect(scene, ("circle", "square"))
        assert all(0.0 <= d.score <= 1.0 for d in dets)
        assert all(len(d.box) == 4 and d.box[2] > d.box[0] for d in dets)
        assert {d.label for d in dets} <= {"circle", "square"}

        colors = SiglipColors(device="cpu")
        red_disc = Detection(label="circle", score=1.0, box=(40, 100, 160, 220))
        blue_square = Detection(label="square", score=1.0, box=(330, 100, 470, 240))
        assert colors.classify(scene, red_disc) == "red"
        assert colors.classify(scene, blue_square) == "blue"

    def test_score_image_end_to_end(self, detector):
        from cozy_eval.metrics.geneval import SiglipColors, score_image

        compositional_spec = CompositionalSpec(
            spec_id="smoke", prompt="a red circle left of a blue square",
            include=(
                ObjectTerm(name="circle", color="red"),
                ObjectTerm(name="square", color="blue", relation="right of", relative_to=0),
            ),
        )
        out = score_image(
            compositional_spec, self._scene(), detector, colors=SiglipColors(device="cpu"),
        )
        assert out.spec_id == "smoke"
        assert out.terms and 0.0 <= out.term_recall <= 1.0
