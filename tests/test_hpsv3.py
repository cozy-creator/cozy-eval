"""The 16 GB weights never load in CI; what CAN be proven without them is
proven hard: the checkpoint key remap, the trained-contract prompt assembly
byte-for-byte, and the full reward forward path (backbone -> ranknet head ->
special-token pooling) on a tiny randomly-initialized Qwen2-VL config.

Registry wiring lives in `test_public_api.py`'s registry sweep.
"""

import pytest

from cozy_eval.bench.metrics import hpsv3

transformers = pytest.importorskip("transformers")


def tiny_model(special_ids=(999,)):
    from transformers import Qwen2VLConfig
    from transformers.models.qwen2_vl.configuration_qwen2_vl import (
        Qwen2VLTextConfig,
        Qwen2VLVisionConfig,
    )

    text = Qwen2VLTextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=1000,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        bos_token_id=0, eos_token_id=1,
    )
    vision = Qwen2VLVisionConfig(
        depth=2, embed_dim=32, hidden_size=32, num_heads=2,
        patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
    )
    cfg = Qwen2VLConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
    cls = hpsv3._make_model_class()
    return cls(cfg, special_token_ids=special_ids).eval()


def test_prompt_assembly_reproduces_the_trained_contract():
    """Spot-check the load-bearing bytes: the trailing double spaces, the exact
    suffix the model was trained with, and the pixel budget the message carries
    — a reward model reads its own training format or it reads noise."""
    assert "**Visual Quality:**  \n" in hpsv3.INSTRUCTION
    assert hpsv3.INSTRUCTION.endswith("Textual prompt - {text_prompt}\n\n\n")
    assert hpsv3.SUFFIX == (
        "\nPlease provide the overall ratings of this image: <|Reward|>\n\nEND\n"
    )

    msg = hpsv3.build_message("a red car", image="IMG")
    assert msg[0]["role"] == "user"
    image_part, text_part = msg[0]["content"]
    assert image_part["min_pixels"] == image_part["max_pixels"] == hpsv3.PIXELS
    assert "a red car" in text_part["text"]
    assert text_part["text"].endswith(hpsv3.SUFFIX)


@pytest.mark.parametrize("state,expected", [
    pytest.param(
        {"visual.blocks.0.norm1.weight": 1,
         "model.layers.0.self_attn.q_proj.weight": 2,
         "model.embed_tokens.weight": 3,
         "lm_head.weight": 4,
         "rm_head.0.weight": 5},
        {"model.visual.blocks.0.norm1.weight": 1,
         "model.language_model.layers.0.self_attn.q_proj.weight": 2,
         "model.language_model.embed_tokens.weight": 3,
         "lm_head.weight": 4,
         "rm_head.0.weight": 5},
        id="legacy_keys_translate",
    ),
    pytest.param(
        {"model.language_model.embed_tokens.weight": 1,
         "model.visual.blocks.0.norm1.weight": 2},
        None, id="modern_keys_pass_through_unchanged",
    ),
])
def test_remap(state, expected):
    assert hpsv3._remap(state, tiny_model()) == (state if expected is None else expected)


def test_remap_covers_every_backbone_key():
    """Round-trip guarantee: legacy-ifying the modern names and remapping them
    back reproduces the model's own state dict keys exactly — so no released
    checkpoint key can quietly fail to land."""
    import re

    model = tiny_model()
    modern = set(model.state_dict())
    legacy = {}
    for k in modern:
        k2 = re.sub(r"^model\.visual\.", "visual.", k)
        k2 = re.sub(r"^model\.language_model\.", "model.", k2)
        legacy[k2] = None
    assert set(hpsv3._remap(legacy, model)) == modern


def test_reward_forward_pools_on_the_special_token_deterministically():
    import torch

    model = tiny_model(special_ids=(999,))
    input_ids = torch.tensor([
        [5, 6, 7, 999, 1],
        [8, 9, 999, 1, 1],
    ])
    mask = (input_ids != 1).long()
    with torch.no_grad():
        a = model.reward(input_ids=input_ids, attention_mask=mask)
        b = model.reward(input_ids=input_ids, attention_mask=mask)
    assert a.shape == (2, 2)  # [batch, (mu, sigma)]
    assert torch.equal(a, b)
    assert a.dtype == torch.float32


def test_head_shape_matches_released_checkpoint():
    model = tiny_model()
    keys = {k: tuple(v.shape) for k, v in model.rm_head.state_dict().items()}
    assert keys == {
        "0.weight": (1024, 32), "0.bias": (1024,),
        "3.weight": (16, 1024), "3.bias": (16,),
        "5.weight": (2, 16), "5.bias": (2,),
    }
