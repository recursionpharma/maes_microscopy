# © Recursion Pharmaceuticals 2024
from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
from timm.models.helpers import checkpoint_seq
from timm.models.vision_transformer import Block, Mlp, VisionTransformer

from masking import transformer_random_masking
from vit import channel_agnostic_vit

# If interested in training new MAEs, combine an encoder and decoder into a new module, and you should
# leverage the flattening and unflattening utilities as needed from mae_utils.py.
# Be sure to use an encoder-decoder Linear projection layer to match encoder dims with decoder dimensions.
# As described in the paper, images are self-standardized at the start.


class SelfStandardize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_standardize = nn.LazyInstanceNorm2d(
            affine=False, track_running_stats=False
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = pixels.float() / 255.0
        return self.self_standardize(x)


class MAEEncoder(nn.Module):
    def __init__(
        self,
        vit_backbone: VisionTransformer,
        max_in_chans: int = 6,
        channel_agnostic: bool = False,
    ) -> None:
        super().__init__()
        if channel_agnostic:
            self.vit_backbone = channel_agnostic_vit(
                vit_backbone, max_in_chans=max_in_chans
            )
        else:
            self.vit_backbone = vit_backbone
        self.max_in_chans = max_in_chans
        self.channel_agnostic = channel_agnostic

    @property
    def embed_dim(self) -> int:
        return int(self.vit_backbone.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit_backbone.forward_features(x)
        x = self.vit_backbone.forward_head(x)
        return x  # type: ignore[no-any-return]

    def forward_masked(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        constant_noise: Union[torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.vit_backbone.patch_embed(x)
        x = self.vit_backbone._pos_embed(x)  # adds class token
        x_ = x[:, 1:, :]  # no class token
        x_, mask, ind_restore = transformer_random_masking(
            x_, mask_ratio, constant_noise
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # add class token
        x = self.vit_backbone.norm_pre(x)

        if self.vit_backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.vit_backbone.blocks, x)
        else:
            x = self.vit_backbone.blocks(x)
        x = self.vit_backbone.norm(x)
        return x, mask, ind_restore


class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),  # type: ignore[assignment]
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embeddings = None  # to be overwritten by MAE class
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embeddings
        x = self.blocks(x)
        x = self.norm(x)
        return x  # type: ignore[no-any-return]

    def forward_masked(
        self, x: torch.Tensor, ind_restore: torch.Tensor
    ) -> torch.Tensor:
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ind_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # remove class token
        x_ = torch.gather(
            x_, dim=1, index=ind_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # add class token

        x = x + self.pos_embeddings
        x = self.blocks(x)
        x = self.norm(x)
        return x  # type: ignore[no-any-return]


class CrossAttention(nn.Module):
    def __init__(
        self, embed_dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CAMAEDecoder(nn.Module):
    def __init__(
        self,
        num_modalities: int = 6,
        tokens_per_modality: int = 256,
        embed_dim: int = 256,
        depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),  # type: ignore[assignment]
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.tokens_per_modality = tokens_per_modality
        self.embed_dim = embed_dim
        self.pos_embeddings = None  # to be overwritten by MAE class
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.placeholder = nn.Parameter(
            torch.zeros(1, 1, embed_dim), requires_grad=False
        )
        self.modality_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                for modality in range(self.num_modalities)
            ]
        )

        self.cross_attention = CrossAttention(embed_dim=self.embed_dim)
        self.mlp = Mlp(self.embed_dim, hidden_features=int(self.embed_dim * mlp_ratio))

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        Block(
                            embed_dim,
                            num_heads,
                            mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                        )
                        for i in range(depth)
                    ]
                )
                for modality in range(self.num_modalities)
            ]
        )
        # self.norm = norm_layer(embed_dim)  # we decided to drop the last layer norm
        self.context_norm = norm_layer(embed_dim)
        self.query_norm = norm_layer(embed_dim)
        self.out_norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_m_s = []

        modality_tokens_concat = torch.cat(
            [
                self.placeholder,
            ]  # placeholder for class token
            + [
                m_t.repeat(1, self.tokens_per_modality, 1)
                for m_t in self.modality_tokens
            ],
            dim=1,
        )

        x = (
            x + self.pos_embeddings + modality_tokens_concat
        )  # add pos and tiled modality tokens
        x_ = x[:, 1:, :]  # no class token
        for m, decoder in enumerate(
            self.decoders
        ):  # iterate through modalities and decoders
            x_m = x_[
                :, m * self.tokens_per_modality : (m + 1) * self.tokens_per_modality, :
            ]
            x_m = self.cross_attention(self.query_norm(x_m), self.context_norm(x_))
            x_m = x_m + self.mlp(self.out_norm(x_m))
            x_m = decoder(x_m)
            x_m_s.append(x_m)
        x_m_s = torch.cat(x_m_s, dim=1)  # concat all tokens
        # x_m_s = self.norm(x_m_s)  # we decided to drop the last layer norm
        x_m_s = torch.cat([x[:, :1, :], x_m_s], dim=1)  # add back class token

        return x_m_s

    def forward_masked(
        self, x: torch.Tensor, ind_restore: torch.Tensor
    ) -> torch.Tensor:
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ind_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # remove class token
        x_ = torch.gather(
            x_, dim=1, index=ind_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # add class token
        x = self.forward(x)
        return x
