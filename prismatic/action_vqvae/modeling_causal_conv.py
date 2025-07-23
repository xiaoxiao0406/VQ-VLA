from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from torch import Tensor


def divisible_by(num, den):
    return (num % den) == 0


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def is_odd(n):
    return not divisible_by(n, 2)


class CausalGroupNorm(nn.GroupNorm):

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            t = x.shape[2]
            x = rearrange(x, "b c t d -> (b t) c d")
            x = super().forward(x)
            x = rearrange(x, "(b t) c d -> b c t d", t=t)
        elif x.dim() == 5:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = super().forward(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        else:
            raise ValueError(f"The dim of input:{x.dim()} is wrong.")

        return x


class CausalConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = cast_tuple(kernel_size, 2)

        time_kernel_size, action_kernel_size = kernel_size
        self.time_kernel_size = time_kernel_size
        dilation = kwargs.pop("dilation", 1)
        self.pad_mode = pad_mode

        if isinstance(stride, int):
            stride = (stride, stride)

        time_pad = dilation * (time_kernel_size - 1)
        action_pad = action_kernel_size // 2

        self.temporal_stride = stride[0]
        self.time_pad = time_pad
        self.time_causal_padding = (time_pad, 0, action_pad, action_pad)
        self.time_uncausal_padding = (action_pad, action_pad, 0, 0)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # temporal_chunk: whether to use the temporal chunk
        # breakpoint()
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # B,1,7,9
        x = self.conv(x)  # B,128,5,7
        return x
