# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pe_init(seq_len: int, embed_dim: int) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32)
        * -(math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(seq_len, embed_dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class GeGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(inputs).chunk(2, dim=-1)
        return x * F.gelu(gate)


class RLTCrossAttentionLayer(nn.Module):
    """Cross-attention transformer block used by the RLT token encoder/decoder."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.Dropout(dropout_rate),
            GeGLU(mlp_dim),
            nn.Linear(mlp_dim, embed_dim),
        )

    @staticmethod
    def _key_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        return ~mask.to(dtype=torch.bool)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cross_key_padding_mask = self._key_padding_mask(cross_mask)
        if cross_key_padding_mask is not None:
            cross_key_padding_mask = cross_key_padding_mask.to(device=y.device)

        residual = x
        x_norm = self.self_norm(x)
        x = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = residual + x

        residual = x
        x_norm = self.cross_norm(x)
        x = self.cross_attn(
            x_norm,
            y,
            y,
            key_padding_mask=cross_key_padding_mask,
            need_weights=False,
        )[0]
        x = residual + x

        return x + self.mlp(self.mlp_norm(x))


class RLTTokenEncoder(nn.Module):
    """Compress VLA prefix embeddings into a small set of RL tokens."""

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        embed_dim: int = 2048,
        num_rl_tokens: int = 1,
        prefix_seq_len: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.num_rl_tokens = int(num_rl_tokens)
        self.prefix_seq_len = int(prefix_seq_len)

        self.input_proj = (
            nn.Linear(self.input_dim, self.embed_dim)
            if self.input_dim != self.embed_dim
            else nn.Identity()
        )
        self.q_embed = nn.Parameter(
            sinusoidal_pe_init(self.num_rl_tokens, self.embed_dim)
        )
        self.y_pos_enc = nn.Parameter(
            sinusoidal_pe_init(self.prefix_seq_len, self.embed_dim)
        )
        self.layers = nn.ModuleList(
            [
                RLTCrossAttentionLayer(
                    self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        prefix_embs = self.input_proj(prefix_embs)
        seq_len = prefix_embs.shape[-2]
        if seq_len > self.prefix_seq_len:
            raise ValueError(
                f"prefix sequence length {seq_len} exceeds configured "
                f"prefix_seq_len {self.prefix_seq_len}."
            )

        batch_size = prefix_embs.shape[0]
        x = self.q_embed.unsqueeze(0).expand(batch_size, -1, -1)
        y = prefix_embs + self.y_pos_enc[:seq_len].to(dtype=prefix_embs.dtype)

        for layer in self.layers:
            x = layer(x, y, cross_mask=mask)
        return x


class RLTTokenDecoder(nn.Module):
    """Reconstruct VLA prefix embeddings from RL tokens."""

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        embed_dim: int = 2048,
        num_rl_tokens: int = 1,
        prefix_seq_len: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.num_rl_tokens = int(num_rl_tokens)
        self.prefix_seq_len = int(prefix_seq_len)

        self.q_embed = nn.Parameter(sinusoidal_pe_init(prefix_seq_len, embed_dim))
        self.y_pos_enc = nn.Parameter(sinusoidal_pe_init(num_rl_tokens, embed_dim))
        self.layers = nn.ModuleList(
            [
                RLTCrossAttentionLayer(
                    self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = (
            nn.Linear(self.embed_dim, self.input_dim)
            if self.input_dim != self.embed_dim
            else nn.Identity()
        )

    def forward(self, rl_tokens: torch.Tensor, target_seq_len: int | None = None):
        target_seq_len = int(target_seq_len or self.prefix_seq_len)
        if target_seq_len > self.prefix_seq_len:
            raise ValueError(
                f"target sequence length {target_seq_len} exceeds configured "
                f"prefix_seq_len {self.prefix_seq_len}."
            )

        batch_size = rl_tokens.shape[0]
        x = self.q_embed[:target_seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        y = rl_tokens + self.y_pos_enc[: rl_tokens.shape[-2]].to(dtype=rl_tokens.dtype)

        for layer in self.layers:
            x = layer(x, y)
        return self.output_proj(x)


class RLTTokenTransformer(nn.Module):
    """PyTorch implementation of openpi-RLT's RL-token encoder-decoder.

    Defaults produce a flattened z_rl feature of 2048 dimensions:
    num_rl_tokens=1 and embed_dim=2048.
    """

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        embed_dim: int = 2048,
        num_rl_tokens: int = 1,
        prefix_seq_len: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.num_rl_tokens = int(num_rl_tokens)
        self.prefix_seq_len = int(prefix_seq_len)

        common_kwargs = {
            "input_dim": self.input_dim,
            "embed_dim": self.embed_dim,
            "num_rl_tokens": self.num_rl_tokens,
            "prefix_seq_len": self.prefix_seq_len,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "dropout_rate": dropout_rate,
        }
        self.encoder = RLTTokenEncoder(**common_kwargs)
        self.decoder = RLTTokenDecoder(**common_kwargs)

    @property
    def z_dim(self) -> int:
        return self.num_rl_tokens * self.embed_dim

    def encode(
        self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.encoder(prefix_embs, mask)

    def encode_flat(
        self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.encode(prefix_embs, mask).reshape(prefix_embs.shape[0], -1)

    def decode(
        self, rl_tokens: torch.Tensor, target_seq_len: int | None = None
    ) -> torch.Tensor:
        return self.decoder(rl_tokens, target_seq_len)

    def reconstruct(
        self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rl_tokens = self.encode(prefix_embs, mask)
        reconstructed = self.decode(rl_tokens, prefix_embs.shape[-2])
        return reconstructed, rl_tokens

    def loss(
        self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reconstructed, rl_tokens = self.reconstruct(prefix_embs, mask)
        target = prefix_embs.detach().to(dtype=torch.float32)
        reconstructed = reconstructed.to(dtype=torch.float32)
        sq_error = torch.square(reconstructed - target)

        if mask is not None:
            mask_expanded = mask.to(device=sq_error.device, dtype=sq_error.dtype)[
                ..., None
            ]
            sq_error = sq_error * mask_expanded
            denom = torch.clamp(mask_expanded.sum() * prefix_embs.shape[-1], min=1.0)
            mse = sq_error.sum() / denom
        else:
            mse = sq_error.mean()

        return mse, {
            "mse": mse,
            "z_rl": rl_tokens.reshape(prefix_embs.shape[0], -1),
        }

    def forward(
        self, prefix_embs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss(prefix_embs, mask)
