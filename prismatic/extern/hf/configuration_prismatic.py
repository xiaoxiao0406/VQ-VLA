"""
configuration_prismatic.py

HuggingFace-style configuration definition for Prismatic VLMs, inheriting from `transformers.PretrainedConfig`.
Default configuration specifies `siglip-224px+7b`.
"""

import os
from typing import Any, Dict, List, Optional, Union

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

from prismatic.action_vqvae import ActionVQVAELossWrapper

# from prismatic.vqvae.vae_temporal import VqVae

# === Utilities for Mapping Prismatic names to HF names ===
# fmt: off
VISION_BACKBONE_TO_RESOLUTION: Dict[str, List[int]] = {
    "clip-vit-l": [224], "siglip-vit-so400m": [224], "dinov2-vit-l": [224], "in1k-vit-l": [224],

    "clip-vit-l-336px": [336],
    "siglip-vit-so400m-384px": [384],

    "dinoclip-vit-l-336px": [336, 336],
    "dinosiglip-vit-so-224px": [224, 224],
    "dinosiglip-vit-so-384px": [384, 384],
}
VISION_BACKBONE_TO_TIMM_ID: Dict[str, List[str]] = {
    "clip-vit-l": ["vit_large_patch14_clip_224.openai"],
    "clip-vit-l-336px": ["vit_large_patch14_clip_336.openai"],

    "dinov2-vit-l": ["vit_large_patch14_reg4_dinov2.lvd142m"],
    "in1k-vit-l": ["vit_large_patch16_224.augreg_in21k_ft_in1k"],

    "siglip-vit-so400m": ["vit_so400m_patch14_siglip_224"],
    "siglip-vit-so400m-384px": ["vit_so400m_patch14_siglip_384"],

    "dinoclip-vit-l-336px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_large_patch14_clip_336.openai"],
    "dinosiglip-vit-so-224px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
    "dinosiglip-vit-so-384px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_384"],
}
TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    "clip-vit-l": ["quick_gelu"], "clip-vit-l-336px": ["quick_gelu"],
    "dinov2-vit-l": [None], "in1k-vit-l": [None],
    "siglip-vit-so400m": [None], "siglip-vit-so400m-384px": [None],
    "dinoclip-vit-l-336px": [None, "quick_gelu"],
    "dinosiglip-vit-so-224px": [None, None], "dinosiglip-vit-so-384px": [None, None]
}

LLM_BACKBONE_TO_HF_PATH = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf", "llama2-13b-pure": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",

    "vicuna-v15-7b": "lmsys/vicuna-7b-v1.5", "vicuna-v15-13b": "lmsys/vicuna-13b-v1.5",

    "mistral-v0.1-7b-pure": "mistralai/Mistral-7B-v0.1",
    "mistral-v0.1-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",

    "phi-2-3b": "microsoft/phi-2",
}
LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama", "llama2-13b-pure": "llama", "llama2-7b-chat": "llama", "llama2-13b-chat": "llama",
    "vicuna-v15-7b": "llama", "vicuna-v15-13b": "llama",

    "mistral-v0.1-7b-pure": "mistral", "mistral-v0.1-7b-instruct": "mistral",

    "phi-2-3b": "phi",
}

EVAL_TASK_TO_CONDITION = {
    "libero_spatial_no_noops": ("franka",20),
    "libero_object_no_noops": ("franka",20),
    "libero_goal_no_noops": ("franka",20),
    "libero_10_no_noops": ("franka",20),
    "libero_90_no_noops": ("franka",20),
    "bridge_orig": ("widowx",5)
}

VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION.keys())
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH)
# fmt: on


class PrismaticConfig(PretrainedConfig):
    model_type: str = "prismatic"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "vicuna-v15-7b",
        arch_specifier: str = "no-align+gelu-mlp",
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = "letterbox",
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        if vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(f"Vision backbone `{vision_backbone_id}` not in {VALID_VISION_BACKBONES = }")

        if llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(f"LLM backbone `{llm_backbone_id}` not in {VALID_LLM_BACKBONES = }")

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>> supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(self.vision_backbone_id.startswith(v) for v in ["dinoclip", "dinosiglip"])
        )

        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[self.vision_backbone_id]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTION[self.vision_backbone_id]
        self.image_resize_strategy = image_resize_strategy

        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[self.llm_backbone_id]
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](**text_config)
            if text_config is not None
            else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
        )

        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))


class OpenVLAConfig(PrismaticConfig):
    model_type: str = "openvla"

    def __init__(
        self,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
        n_action_bins: int = 256,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        super().__init__(**kwargs)


class ConvVLAConfig(PrismaticConfig):
    model_type: str = "openvla"

    def __init__(
        self,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
        n_action_bins: int = 256,
        codebook_size: int = 256,
        vq_layer_group: int = 4,
        window_size: int = 5,
        action_chunk_size: int = 1,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.codebook_size, self.n_action_bins = norm_stats, codebook_size, n_action_bins
        self.output_token_num = vq_layer_group
        self.action_chunk_size = action_chunk_size
        # self.vqvae_model_config = {
        #     "config_path": "train_vae/scripts/action_vqvae_config",
        #     "eval": True,
        #     "use_action_type_pe": True,
        #     "use_time_pe": True,
        # }
        self.vqvae_model = ActionVQVAELossWrapper(
            "train_vae/scripts/action_vqvae_config",
            is_eval=True,
            use_action_type_pe=True,
            use_time_pe=True,
        )

        super().__init__(**kwargs)
