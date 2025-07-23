# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from diffusers.utils import BaseOutput
from einops import rearrange

from .modeling_block import (
    get_down_block,
    get_up_block,
)
from .modeling_causal_conv import CausalConv2d, CausalGroupNorm


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor


class ActionVQVaeEncoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 128,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlockCausal2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: Tuple[int, ...] = (2,),
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        block_dropout: Tuple[int, ...] = (0.0,),
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.num_res_blocks = num_res_blocks
        self.conv_in = CausalConv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])
        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
                dropout=block_dropout[i],
            )
            self.down_blocks.append(down_block)
            if i < len(down_block_types) - 1:
                self.conv_blocks.append(CausalConv2d(output_channel, output_channel, kernel_size=3, stride=2))
            else:
                self.conv_blocks.append(nn.Identity(output_channel))

        self.res_blocks = get_down_block(
            "DownEncoderBlockCausal2D",
            num_layers=self.num_res_blocks,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
        )
        # out

        self.conv_norm_out = CausalGroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        self.conv_out = CausalConv2d(block_out_channels[-1], out_channels, kernel_size=3, stride=1)

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        # sample : [B,5,7]

        sample = self.conv_in(sample)

        # for down_block in self.down_blocks:
        for i, down_block in enumerate(self.down_blocks):
            sample = down_block(sample)
            if i < len(self.down_blocks) - 1:
                sample = self.conv_blocks[i](sample)

        sample = self.res_blocks(sample)  # B,512,1,1

        # post-process
        sample = self.conv_norm_out(sample)  # B,512,5,7 ->B,512,5,7
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class ActionVQVaeDecoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlockCausal2D",),
        spatial_up_sample: Tuple[bool, ...] = (True,),
        temporal_up_sample: Tuple[bool, ...] = (False,),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: Tuple[int, ...] = (2,),
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        interpolate: bool = True,
        block_dropout: Tuple[int, ...] = (0.0,),
        temporal_downsample: Tuple[bool, ...] = (True, True, True),
        num_res_blocks: int = 4,
        device="cuda",
        action_window_size=5,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.num_res_blocks = num_res_blocks
        self.temporal_downsample = temporal_downsample
        self.conv_in = CausalConv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
        )
        self.action_window_size = action_window_size

        self.res_blocks = get_up_block(
            "UpDecoderBlockCausal2D",
            num_layers=self.num_res_blocks,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
        )

        self.up_blocks = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])
        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            j = len(up_block_types) - 1 - i
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block[i],
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temb_channels=None,
                resnet_time_scale_shift="default",
                interpolate=interpolate,
                dropout=block_dropout[i],
            )
            self.up_blocks.insert(0, up_block)
            if j > 0:
                if self.temporal_downsample[j - 1]:
                    t_stride, s_stride = 2, 2
                    self.conv_blocks.insert(
                        0, CausalConv2d(output_channel, output_channel * t_stride * s_stride, kernel_size=3)
                    )
                else:
                    self.conv_blocks.insert(
                        0,
                        nn.Identity(output_channel),
                    )

        # out
        self.projection = nn.Linear(8, 7)
        self.conv_norm_out = CausalGroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv2d(block_out_channels[0], out_channels, kernel_size=3, stride=1)

        self.gradient_checkpointing = False
        self.device = device

    def forward(self, sample: torch.FloatTensor, robot_type=None, frequency=None) -> torch.FloatTensor:
        # breakpoint()
        r"""The forward method of the `Decoder` class."""
        sample = sample.unsqueeze(-1).unsqueeze(-1)
        sample = self.conv_in(sample)
        sample = self.res_blocks(sample)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # middle
        # sample = self.mid_block(sample, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        sample = sample.to(upscale_dtype)
        # up
        # for i, up_block in enumerate(self.up_blocks):
        for i, up_block in enumerate(reversed(self.up_blocks)):
            i = len(self.up_blocks) - 1 - i
            sample = up_block(sample)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                s_stride = 2 if self.temporal_downsample[i - 1] else 1
                sample = self.conv_blocks[i - 1](sample)
                sample = rearrange(
                    sample,
                    "B  (C ts asize) T A-> B C (T ts) (A asize)",
                    ts=t_stride,
                    asize=s_stride,
                )

        sample = self.projection(sample)
        # sample:[B,128,1,1]
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        # breakpoint()
        sample = sample[:, 0, -self.action_window_size :, :]  # B,5,7

        return sample
