import torch
from torch import nn
import numpy as np

class UNETAttentionBlock(nn.Module):

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(channels)
        self.self_attention = nn.MultiheadAttention(embed_dim=channels, num_heads=n_head, batch_first=True)
        # no cross attention
        # and no layernorm2 as a result

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Channels, H, W)
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (B, C, H, W) -> (B, C, H * W)
        x = x.view((n, c, h * w))
        # (B, C, H * W) -> (B, H * W, C)
        x = x.transpose(-1, -2)

        # normalization + self attention with skip connection
        residue_short = x
        x = self.layernorm1(x)
        x += self.self_attention(x, x, x, need_weights=False)[0]
        x += residue_short
        
        # (B, H * W, C) -> (B, C, H * W)
        x = x.transpose(-1, -2)

        # (B, C, H * W) -> (B, C, H, W)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long