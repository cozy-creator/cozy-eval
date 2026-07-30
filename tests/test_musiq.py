"""Preprocessing is pure numpy and tested exactly. Model + checkpoint tests are
`heavy` (163 MB Apache checkpoint); the banked-oracle parity test runs there
too — rankings and values were verified against the NC reference (see
tests/fixtures/oracle_musiq.json).

Registry wiring lives in `test_public_api.py`'s registry sweep.
"""

import numpy as np
import pytest

from cozy_eval.bench.metrics import musiq


def test_hash_ids_match_the_tf_v1_nearest_rule():
    # count < grid: floor(i * 10 / count)
    assert musiq._hash_ids(5).tolist() == [0, 2, 4, 6, 8]
    # count == grid: identity
    assert musiq._hash_ids(10).tolist() == list(range(10))
    # count > grid: values stay in range and are non-decreasing
    ids = musiq._hash_ids(23)
    assert ids.min() == 0 and ids.max() == 9
    assert all(a <= b for a, b in zip(ids, ids[1:], strict=False))


def test_gaussian_resize_normalizes_its_kernel_and_preserves_a_constant():
    """A resampling kernel whose rows do not sum to 1 shifts the image's mean —
    the fastest way to be subtly, invisibly wrong."""
    _, w = musiq._resize_kernel(384, 224)
    assert np.allclose(w.sum(axis=1), 1.0)

    out = musiq._gaussian_resize(np.full((100, 160, 3), 0.25), 60, 96)
    assert out.shape == (60, 96, 3)
    assert np.allclose(out, 0.25)


def test_preprocess_emits_the_multiscale_sequence_layout_deterministically():
    img = np.random.default_rng(0).integers(0, 256, (384, 384, 3), dtype=np.uint8)
    patches, spatial, scale, mask = musiq.preprocess(img)
    # 224 scale padded to ceil(224/32)^2=49, 384 scale to 144, native 12x12.
    assert patches.shape == (49 + 144 + 144, 32 * 32 * 3)
    assert set(np.unique(scale)) == {0, 1, 2}
    assert mask.sum() == 49 + 144 + 144
    assert spatial.max() < musiq.GRID * musiq.GRID

    again = musiq.preprocess(img)
    for x, y in zip((patches, spatial, scale, mask), again, strict=True):
        assert np.array_equal(x, y)


def test_preprocess_masks_the_padding_of_a_non_square_image():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    patches, _, _, mask = musiq.preprocess(img)
    # 224-scale render is 112x224 -> 4x7=28 real patches of budget 49.
    assert mask[:49].sum() == 28
    assert patches[mask == 0].sum() == 0  # padding rows are all zero


@pytest.mark.heavy
class TestModel:
    @pytest.fixture(scope="class")
    def model(self):
        return musiq.load_model("koniq")

    def test_checkpoint_loads_strict(self, model):
        assert sum(p.numel() for p in model.parameters()) == 27_125_825

    def test_parity_with_banked_oracle(self):
        import json
        from pathlib import Path

        from PIL import Image

        fixtures = Path(__file__).parent / "fixtures"
        spec = json.loads((fixtures / "oracle_musiq.json").read_text())
        for name, ref in spec["values"].items():
            ours = musiq.musiq(Image.open(fixtures / f"{name}.png"))
            assert abs(ours - ref) / ref < spec["tolerance_rel"], (
                f"{name}: ours={ours:.3f} oracle={ref:.3f}"
            )
