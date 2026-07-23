#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch
from torch import nn


class DriftingTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        memory_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.self_norm(hidden_states)
        attention_output, _ = self.self_attention(normed, normed, normed, need_weights=False)
        hidden_states = hidden_states + self.dropout(attention_output)

        normed = self.cross_norm(hidden_states)
        attention_output, _ = self.cross_attention(
            normed,
            memory,
            memory,
            key_padding_mask=~memory_attention_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states + self.dropout(self.feed_forward(self.ff_norm(hidden_states)))


class DriftingTransformer(nn.Module):
    """One-step conditional transformer used by the drifting action generator."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        attend_text_every_n_blocks: int,
    ) -> None:
        super().__init__()
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        self.blocks = nn.ModuleList(
            [
                DriftingTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)

    @staticmethod
    def _ensure_nonempty_mask(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool().clone()
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask[empty_rows, 0] = True
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        backbone_attention_mask = self._ensure_nonempty_mask(backbone_attention_mask)
        image_attention_mask = image_mask.bool() & backbone_attention_mask
        text_attention_mask = ~image_mask.bool() & backbone_attention_mask

        for index, block in enumerate(self.blocks):
            candidate_mask = (
                text_attention_mask if index % self.attend_text_every_n_blocks == 0 else image_attention_mask
            )
            has_selected_tokens = candidate_mask.any(dim=1, keepdim=True)
            candidate_mask = torch.where(has_selected_tokens, candidate_mask, backbone_attention_mask)
            hidden_states = block(
                hidden_states=hidden_states,
                memory=memory,
                memory_attention_mask=self._ensure_nonempty_mask(candidate_mask),
            )
        return self.final_norm(hidden_states)
