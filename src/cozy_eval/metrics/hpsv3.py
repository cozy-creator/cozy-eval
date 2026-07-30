"""HPSv3 human-preference scoring, ported to modern transformers.

STABILITY: experimental (v0.x). The metric NAME ``hpsv3_score`` is locked by the
registry; ``Hpsv3Scorer`` and the remap helpers are not.

ATTRIBUTION: adapted from MizzenAI/HPSv3, which is MIT licensed. The MIT notice
travels with this derivative work; the INSTRUCTION text and the ranknet head
shape are reproduced verbatim because they are the trained model's contract.
See PROVENANCE.md.

HPSv3 (Ma et al. 2025, MizzenAI/HPSv3 — MIT code, weight file on an MIT-tagged
HF repo) is the human-preference accuracy leader (76.9 on HPDv3 vs PickScore
65.6). Upstream pins ``transformers==4.45.2`` because it hand-rolls Qwen2-VL's
vision-embed merge against long-gone internals; modern ``Qwen2VLModel`` does
that merge itself, so this port collapses to: backbone forward -> ranknet head
-> read the two outputs (mu, sigma) at the ``<|Reward|>`` token, score = mu.

ADAPTED with attribution from the MIT original (hpsv3/model/qwen2vl_trainer.py,
hpsv3/dataset/data_collator_qwen.py); the INSTRUCTION text and head shape are
the trained model's contract and are reproduced verbatim. See PROVENANCE.md.

The checkpoint predates the transformers 4.52+ submodule reshuffle
(``visual.*`` -> ``model.visual.*``, ``model.*`` -> ``model.language_model.*``);
:func:`_remap` translates using the class's own conversion table, so the port
tracks upstream transformers rather than a frozen key list.

Weights are ~16 GB: load on the pod that renders, never eagerly. Nothing here
imports torch at module import time.
"""

from __future__ import annotations

from typing import Any

from ..device import AUTO, resolve_device

DEFAULT_CHECKPOINT = "MizzenAI/HPSv3"
DEFAULT_BASE = "Qwen/Qwen2-VL-7B-Instruct"
REWARD_TOKEN = "<|Reward|>"
PIXELS = 256 * 28 * 28  # upstream fixes min == max

# Upstream's evaluation instruction, verbatim (MIT; the model was trained on
# this exact text — paraphrasing it would move the scores).
INSTRUCTION = '\nYou are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. \n\n**Visual Quality:**  \nEvaluate the overall visual quality of the image. The following sub-dimensions should be considered:\n- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.\n- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.\n- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).\n- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.\n- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. \n\n**Text Alignment:**  \nAssess how well the image matches the textual prompt across the following sub-dimensions:\n- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.\n- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.\n- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.\n- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.\n- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.\nTextual prompt - {text_prompt}\n\n\n'

SUFFIX = f"""
Please provide the overall ratings of this image: {REWARD_TOKEN}

END
"""


def build_message(prompt: str, image: Any) -> list[dict[str, Any]]:
    """One scoring conversation, upstream's exact shape (special-token path)."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image, "min_pixels": PIXELS, "max_pixels": PIXELS},
            {"type": "text", "text": INSTRUCTION.format(text_prompt=prompt) + SUFFIX},
        ],
    }]


# The 4.45-era layout the checkpoint was saved under -> today's submodule
# layout. Used when the transformers class no longer carries its own
# conversion table (v5 dropped `_checkpoint_conversion_mapping`).
_LEGACY_KEY_MAP = {
    r"^visual\.": "model.visual.",
    r"^model\.(?!language_model\.|visual\.)": "model.language_model.",
}


def _remap(state_dict: dict[str, Any], model: Any) -> dict[str, Any]:
    """Translate pre-4.52 checkpoint keys into the current layout, preferring
    the model class's own conversion table when it exists."""
    import re

    mapping = getattr(type(model), "_checkpoint_conversion_mapping", None) or _LEGACY_KEY_MAP
    out = {}
    for key, value in state_dict.items():
        out_key = key
        for pattern, repl in mapping.items():
            new_key, n = re.subn(pattern, repl, key)
            if n:
                out_key = new_key
                break
        out[out_key] = value
    return out


def _build_head(hidden_size: int, output_dim: int = 2) -> Any:
    """Upstream's ranknet head, default kwargs (the released 7B shape):
    keys rm_head.0 / rm_head.3 / rm_head.5."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(hidden_size, 1024),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(1024, 16),
        nn.ReLU(),
        nn.Linear(16, output_dim),
    )


def _make_model_class() -> type:
    """Class built lazily so importing this module never imports torch."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration

    class HPSv3RewardModel(Qwen2VLForConditionalGeneration):
        def __init__(self, config, *, output_dim: int = 2, special_token_ids: tuple[int, ...] = ()):
            super().__init__(config)
            # v5 composite configs stop proxying text attrs; 4.x kept them flat.
            hidden = getattr(config, "hidden_size", None) or config.get_text_config().hidden_size
            self.rm_head = _build_head(hidden, output_dim).to(torch.float32)
            self.special_token_ids = tuple(special_token_ids)

        def reward(
            self, *, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None,
        ):
            """[B, 2] — (mu, sigma) read at the reward-token position."""
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            )
            logits = self.rm_head(outputs.last_hidden_state.float())
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in self.special_token_ids:
                mask |= input_ids == token_id
            return logits[mask].view(input_ids.shape[0], -1)

    return HPSv3RewardModel


class Hpsv3Scorer:
    """Resident HPSv3 model; score = mu, same-prompt deltas are the meaningful
    read (like every preference scalar). ~16 GB in bf16 — pod-side."""

    def __init__(
        self, *, checkpoint: str = DEFAULT_CHECKPOINT, base: str = DEFAULT_BASE,
        device: str = AUTO,
    ):
        device = resolve_device(device)
        import safetensors.torch
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoProcessor

        self.model_ref = checkpoint
        self.processor = AutoProcessor.from_pretrained(
            base, min_pixels=PIXELS, max_pixels=PIXELS,
        )
        self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [REWARD_TOKEN]}
        )
        special_ids = tuple(
            self.processor.tokenizer.convert_tokens_to_ids([REWARD_TOKEN])
        )
        config = AutoConfig.from_pretrained(base)
        cls = _make_model_class()
        # Build in bf16 (an fp32 8B skeleton is ~33 GB RAM for nothing —
        # load_state_dict converts into param dtype anyway). The rm_head is
        # re-pinned to float32 before load so its checkpoint values are
        # copied at full precision, matching upstream's autocast(float32).
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = cls(config, special_token_ids=special_ids)
        finally:
            torch.set_default_dtype(default)
        model.rm_head.to(torch.float32)
        model.resize_token_embeddings(len(self.processor.tokenizer))
        path = hf_hub_download(checkpoint, "HPSv3.safetensors")
        state = safetensors.torch.load_file(path, device="cpu")
        if "model" in state:
            state = state["model"]
        model.load_state_dict(_remap(state, model), strict=True)
        self.model = model.to(device).eval()
        self.device = device

    def score(self, images: list[Any], prompts: list[str]) -> list[float]:
        import torch

        texts, pils = [], []
        for image, prompt in zip(images, prompts, strict=True):
            msg = build_message(prompt, image)
            texts.append(self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True,
            ))
            pils.append(image.convert("RGB") if hasattr(image, "convert") else image)
        inputs = self.processor(
            text=texts, images=pils, padding=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            pooled = self.model.reward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
        return [float(v) for v in pooled[:, 0]]


__all__ = [
    "DEFAULT_BASE",
    "DEFAULT_CHECKPOINT",
    "INSTRUCTION",
    "PIXELS",
    "REWARD_TOKEN",
    "SUFFIX",
    "Hpsv3Scorer",
    "build_message",
]
