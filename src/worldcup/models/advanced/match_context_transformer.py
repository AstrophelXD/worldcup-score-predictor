from __future__ import annotations

import torch
from torch import nn


class TeamSequenceEncoder(nn.Module):
    """Temporal encoder for recent team match sequences."""

    def __init__(
        self,
        seq_dim: int,
        d_model: int,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.input_proj = nn.Linear(seq_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _encode(self, sequence: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(sequence, src_key_padding_mask=padding_mask)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(sequence)
        outputs: list[torch.Tensor] = []
        for idx in range(sequence.shape[0]):
            if not mask[idx].any():
                outputs.append(torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype))
                continue
            seq_slice = x[idx : idx + 1]
            pad_mask = ~mask[idx : idx + 1]
            if self.gradient_checkpointing and self.training:
                encoded = torch.utils.checkpoint.checkpoint(
                    self._encode,
                    seq_slice,
                    pad_mask,
                    use_reentrant=False,
                )
            else:
                encoded = self._encode(seq_slice, pad_mask)
            weights = mask[idx].float().unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum().clamp(min=1.0)
            outputs.append(pooled.squeeze(0))
        return torch.stack(outputs, dim=0)


class MatchContextTransformer(nn.Module):
    """Fuse tabular, sequence, graph, and odds tokens into a match embedding."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _encode(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens, src_key_padding_mask=padding_mask)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
        full_mask = torch.cat([cls_mask, token_mask], dim=1)
        if self.gradient_checkpointing and self.training:
            encoded = torch.utils.checkpoint.checkpoint(
                self._encode,
                x,
                ~full_mask,
                use_reentrant=False,
            )
        else:
            encoded = self._encode(x, ~full_mask)
        return encoded[:, 0, :]
