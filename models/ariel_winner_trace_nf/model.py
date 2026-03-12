"""Winner-family independent conditional neural spline flows for Ariel tracedata."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import zuko

from .constants import TARGET_COLUMNS


@dataclass
class ModelConfig:
    context_dim: int = 118
    hidden_features: int = 152
    transforms: int = 18
    bins: int = 19
    hidden_layers: int = 5


class IndependentNSF(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = [int(config.hidden_features)] * int(config.hidden_layers)
        self.flows = nn.ModuleList(
            [
                zuko.flows.NSF(
                    features=1,
                    context=int(config.context_dim),
                    hidden_features=hidden,
                    transforms=int(config.transforms),
                    bins=int(config.bins),
                )
                for _ in TARGET_COLUMNS
            ]
        )

    @torch.no_grad()
    def sample(self, context: torch.Tensor, num_samples: int) -> torch.Tensor:
        sampled = []
        for flow in self.flows:
            values = flow(context).sample((num_samples,)).squeeze(-1).permute(1, 0)
            sampled.append(values.unsqueeze(-1))
        return torch.cat(sampled, dim=-1)

