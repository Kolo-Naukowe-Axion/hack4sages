"""Winner-style independent conditional neural spline flows for ADC2023 gases."""

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

    def negative_log_likelihood(self, context: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        losses = []
        per_target = {}
        for idx, (name, flow) in enumerate(zip(TARGET_COLUMNS, self.flows)):
            nll = -flow(context).log_prob(targets[:, idx : idx + 1]).mean()
            losses.append(nll)
            per_target[name] = float(nll.detach().cpu().item())
        total = torch.stack(losses).sum()
        return total, per_target

    @torch.no_grad()
    def sample(self, context: torch.Tensor, num_samples: int) -> torch.Tensor:
        sampled = []
        for flow in self.flows:
            values = flow(context).sample((num_samples,)).squeeze(-1).permute(1, 0)
            sampled.append(values.unsqueeze(-1))
        return torch.cat(sampled, dim=-1)

