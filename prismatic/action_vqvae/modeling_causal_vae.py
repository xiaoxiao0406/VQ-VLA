from typing import Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vq_model import VQEncoderOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange
from timm.models.layers import trunc_normal_

from prismatic.action_vqvae.modeling_enc_dec import (
    ActionVQVaeDecoder,
    ActionVQVaeEncoder,
    DecoderOutput,
)
from prismatic.action_vqvae.residual_vq import ResidualVQ
from prismatic.action_vqvae.vqvae_utils import get_tensor


class ActionVQVAE(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # encoder related parameters
        encoder_in_channels: int = 1,
        encoder_out_channels: int = 1,
        encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
        encoder_down_block_types: Tuple[str, ...] = (
            "DownEncoderBlockCausal2D",
            "DownEncoderBlockCausal2D",
            "DownEncoderBlockCausal2D",
            "DownEncoderBlockCausal2D",
        ),
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        encoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        encoder_act_fn: str = "silu",
        encoder_norm_num_groups: int = 32,
        encoder_double_z: bool = True,
        encoder_type: str = "causal_vae_conv",
        # decoder related
        decoder_in_channels: int = 1,
        decoder_out_channels: int = 1,
        decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3),
        decoder_up_block_types: Tuple[str, ...] = (
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
        ),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        decoder_act_fn: str = "silu",
        decoder_norm_num_groups: int = 32,
        decoder_type: str = "causal_vae_conv",
        vq_embed_dim: int = 128,
        num_vq_embeddings: int = 256,
        device: str = "cuda",
        action_window_size: int = 5,
        vq_groups: int = 4,
    ):
        super().__init__()

        print(f"The latent dimmension channes is {encoder_out_channels}")
        # pass init params to Encoder

        self.encoder = ActionVQVaeEncoder(
            in_channels=encoder_in_channels,
            out_channels=encoder_out_channels,
            down_block_types=encoder_down_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            act_fn=encoder_act_fn,
            norm_num_groups=encoder_norm_num_groups,
            double_z=True,
            block_dropout=encoder_block_dropout,
        )

        # pass init params to Decoder
        self.decoder = ActionVQVaeDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            up_block_types=decoder_up_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            norm_num_groups=decoder_norm_num_groups,
            act_fn=decoder_act_fn,
            block_dropout=decoder_block_dropout,
            device=device,
            action_window_size=action_window_size,
        )
        latent_channels = 128
        self.vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels

        self.vqvae_groups = vq_groups
        self.vqvae_n_embed = 256
        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}

        # self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)
        # self.quant_conv= nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.vq_layer = ResidualVQ(
            dim=self.vq_embed_dim,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
            kmeans_init=True,
            # sync_codebook = False, # important! if not set, loss will be different when the number of gpus are different
        )

        self.apply(self._init_weights)

        # self.start_event = torch.cuda.Event(enable_timing=True)
        # self.end_event = torch.cuda.Event(enable_timing=True)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if state.ndimension() == 2:
            state = state.unsqueeze(0)
        # state = einops.rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)  # .to(self.device)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> VQEncoderOutput:
        x = self.preprocess(x)
        h = self.encoder(x)
        # h = self.quant_conv(h)
        h = h.reshape(h.shape[0], -1)
        if not return_dict:
            return (h,)

        return VQEncoderOutput(latents=h)

    @apply_forward_hook
    def decode(
        self,
        h: torch.Tensor,
        robot_type=None,
        frequency=None,
        force_not_quantize: bool = False,
        return_dict: bool = True,
        shape=None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        # h = self.post_quant_conv(h)
        dec = self.decoder(h, robot_type, frequency)

        return dec  # , commit_loss

    def forward(
        self, sample: torch.Tensor, robot_type=None, frequency=None, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor, ...]]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.autoencoders.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoders.vq_model.VQEncoderOutput`] is returned, otherwise a
                plain `tuple` is returned.
        """
        sample = sample.to(self.device, dtype=torch.bfloat16)
        latents = self.encode(sample).latents

        B = latents.shape[0]  # B
        state_rep_flat = latents.view(latents.size(0), -1, latents.size(1))
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(B, -1)
        dec = self.decode(state_vq, robot_type, frequency)

        recon_loss = torch.nn.MSELoss()(sample, dec)
        vq_loss_state = torch.sum(vq_loss_state)
        loss = recon_loss + vq_loss_state * 5
        return vq_loss_state, recon_loss, loss


class ActionVQVAEPE(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # encoder related parameters
        encoder_in_channels: int = 1,
        encoder_out_channels: int = 1,
        encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
        encoder_down_block_types: Tuple[str, ...] = (
            "DownEncoderBlockCausal2D",
            "DownEncoderBlockCausal2D",
            "DownEncoderBlockCausal2D",
            "DownEncoderBlockCausal2D",
        ),
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        encoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        encoder_act_fn: str = "silu",
        encoder_norm_num_groups: int = 32,
        encoder_double_z: bool = True,
        encoder_type: str = "causal_vae_conv",
        # decoder related
        decoder_in_channels: int = 1,
        decoder_out_channels: int = 1,
        decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3),
        decoder_up_block_types: Tuple[str, ...] = (
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
        ),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        decoder_act_fn: str = "silu",
        decoder_norm_num_groups: int = 32,
        decoder_type: str = "causal_vae_conv",
        vq_embed_dim: int = 128,
        num_vq_embeddings: int = 256,
        num_frequencies: int = 10,
        min_freq: float = 0.0,
        max_freq: float = 8.0,
        temporal_compression_ratio: int = 5,
        device: str = "cuda",
        use_action_type_pe: bool = False,
        use_time_pe: bool = False,
    ):
        super().__init__()

        print(f"The latent dimmension channes is {encoder_out_channels}")
        # pass init params to Encoder

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.temporal_compression_ratio = temporal_compression_ratio
        encoder_in_channels = num_frequencies * 2 + 1

        # assume temporal_compression_ratio == T
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies)
        time_emb_ = (
            torch.arange(self.temporal_compression_ratio).float() / self.temporal_compression_ratio * 2 * torch.pi
        )
        time_emb = time_emb_[..., None] * freqs
        time_emb = torch.sin(torch.cat([time_emb, time_emb + torch.pi / 2.0], dim=-1))
        time_emb = torch.cat([time_emb, time_emb_[..., None]], dim=-1)
        self.register_buffer("time_emb", time_emb[None, ..., None])

        self.xyz_emb = nn.Parameter(torch.randn(1, encoder_in_channels, 1, 3), requires_grad=True)
        self.euler_emb = nn.Parameter(torch.randn(1, encoder_in_channels, 1, 3), requires_grad=True)
        self.gripper_emb = nn.Parameter(torch.randn(1, encoder_in_channels, 1, 1), requires_grad=True)

        self.use_action_type_pe = use_action_type_pe
        self.use_time_pe = use_time_pe

        self.encoder = ActionVQVaeEncoder(
            in_channels=encoder_in_channels,
            out_channels=encoder_out_channels,
            down_block_types=encoder_down_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            act_fn=encoder_act_fn,
            norm_num_groups=encoder_norm_num_groups,
            double_z=True,
            block_dropout=encoder_block_dropout,
        )

        # pass init params to Decoder
        self.decoder = ActionVQVaeDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            up_block_types=decoder_up_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            norm_num_groups=decoder_norm_num_groups,
            act_fn=decoder_act_fn,
            block_dropout=decoder_block_dropout,
            device=device,
        )
        latent_channels = 128
        self.vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels

        self.vqvae_groups = 4  # 4
        self.vqvae_n_embed = 256  # 256
        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}

        # self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)
        # self.quant_conv= nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.vq_layer = ResidualVQ(
            dim=self.vq_embed_dim,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
            kmeans_init=True,
            # sync_codebook = False, # important! if not set, loss will be different when the number of gpus are different
        )

        self.apply(self._init_weights)

        # self.start_event = torch.cuda.Event(enable_timing=True)
        # self.end_event = torch.cuda.Event(enable_timing=True)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_action_type_emb(self, action):
        if action.ndim == 3:
            action = rearrange(action, "b t n -> b 1 t n")
        b, c, t, n = action.shape
        assert n == 7, n
        action = action + torch.cat([self.xyz_emb, self.euler_emb, self.gripper_emb], dim=-1)
        return action

    def apply_time_emb(self, action):
        if action.ndim == 3:
            action = rearrange(action, "b t n -> b 1 t n")
        b, c, t, n = action.shape  # c = 21, t = 8, n = 7
        assert n == 7, n
        action = rearrange(action, "b c t n -> b t c n")
        action = self.time_emb.to(dtype=action.dtype, device=action.device) + action
        return rearrange(action, "b t c n -> b c t n", b=b, t=t, n=n)

    def preprocess(self, action):
        if not torch.is_tensor(action):
            action = get_tensor(action, self.device)
        if action.ndimension() == 2:
            action = action.unsqueeze(0)
        action = action.to(self.device)
        if self.use_action_type_pe:
            action = self.apply_action_type_emb(action)
        if self.use_time_pe:
            action = self.apply_time_emb(action)
        # action = einops.rearrange(action, "N T A -> N (T A)")
        return action  # .to(self.device)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> VQEncoderOutput:
        x = self.preprocess(x)
        h = self.encoder(x)
        # h = self.quant_conv(h)
        h = h.reshape(h.shape[0], -1)
        if not return_dict:
            return (h,)

        return VQEncoderOutput(latents=h)

    @apply_forward_hook
    def decode(
        self,
        h: torch.Tensor,
        robot_type=None,
        frequency=None,
        force_not_quantize: bool = False,
        return_dict: bool = True,
        shape=None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        # h = self.post_quant_conv(h)
        dec = self.decoder(h, robot_type, frequency)

        return dec  # , commit_loss

    def forward(
        self, sample: torch.Tensor, robot_type=None, frequency=None, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor, ...]]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.autoencoders.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoders.vq_model.VQEncoderOutput`] is returned, otherwise a
                plain `tuple` is returned.
        """
        latents = self.encode(sample).latents

        B = latents.shape[0]  # B
        state_rep_flat = latents.view(latents.size(0), -1, latents.size(1))
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(B, -1)
        dec = self.decode(state_vq, robot_type, frequency)
        recon_loss = torch.nn.MSELoss()(sample, dec)
        vq_loss_state = torch.sum(vq_loss_state)
        loss = recon_loss + vq_loss_state * 5
        return vq_loss_state, recon_loss, loss
