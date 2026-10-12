import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        t_emb = self.act(self.time_proj(t_emb))
        h = h + t_emb.unsqueeze(-1).unsqueeze(-1)

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return out + x


class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        has_attention: bool = False,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.has_attention = has_attention

        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(in_ch, out_channels, time_emb_dim, dropout))
            if has_attention:
                self.attentions.append(SelfAttention(out_channels))

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, t_emb)
            if self.has_attention:
                x = self.attentions[i](x)

        x = self.downsample(x)
        return x


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: int,
        has_attention: bool = False,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.has_attention = has_attention
        self.out_channels = out_channels

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)

        first_block_in = in_channels + skip_channels
        self.res_blocks.append(ResidualBlock(first_block_in, out_channels, time_emb_dim, dropout))
        if has_attention:
            self.attentions.append(SelfAttention(out_channels))

        for i in range(1, num_res_blocks):
            self.res_blocks.append(ResidualBlock(out_channels, out_channels, time_emb_dim, dropout))
            if has_attention:
                self.attentions.append(SelfAttention(out_channels))

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)

        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, t_emb)
            if self.has_attention:
                x = self.attentions[i](x)

        return x


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (2, 4),
        num_heads: int = 4,
        dropout: float = 0.0,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.condition_dim = condition_dim

        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        if condition_dim > 0:
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.num_levels = len(channel_mult)
        in_ch = model_channels
        skip_channels = [model_channels]
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            has_attn = i in attention_resolutions
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_emb_dim, has_attn, num_res_blocks, dropout)
            )
            in_ch = out_ch
            skip_channels.append(out_ch)

        self.mid_res = ResidualBlock(in_ch, in_ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(in_ch, num_heads)
        self.mid_res2 = ResidualBlock(in_ch, in_ch, time_emb_dim, dropout)

        self.up_blocks = nn.ModuleList()
        for idx, (i, mult) in enumerate(reversed(list(enumerate(channel_mult)))):
            out_ch = model_channels * mult
            has_attn = i in attention_resolutions
            skip_ch = skip_channels[self.num_levels - 1 - idx]
            self.up_blocks.append(
                UpBlock(in_ch, out_ch, skip_ch, time_emb_dim, has_attn, num_res_blocks, dropout)
            )
            in_ch = out_ch

        self.output_norm = nn.GroupNorm(32, model_channels)
        self.output_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)

        if condition is not None and self.condition_dim > 0:
            c_emb = self.condition_proj(condition)
            t_emb = t_emb + c_emb

        h = self.input_conv(x)

        skips = [h]
        for block in self.down_blocks:
            h = block(h, t_emb)
            skips.append(h)

        h = self.mid_res(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        for i, block in enumerate(self.up_blocks):
            skip = skips[self.num_levels - 1 - i]
            h = block(h, skip, t_emb)

        h = F.silu(self.output_norm(h))
        h = self.output_conv(h)

        return h
