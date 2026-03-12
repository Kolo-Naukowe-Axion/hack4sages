"""Purely classical TauREx regressor with a proven conv backbone and residual refinement."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn

from .constants import AUX_COLUMNS, TARGET_COLUMNS


def _make_group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ResidualSpectralBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = _make_group_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _make_group_norm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act(x)


class SpectralAttentionPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, channels // 2)
        self.scorer = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


class SpectralEncoder(nn.Module):
    def __init__(self, input_channels: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            _make_group_norm(32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualSpectralBlock(32, 32, dropout),
            ResidualSpectralBlock(32, 64, dropout),
            ResidualSpectralBlock(64, 96, dropout),
        )
        self.attention_pool = SpectralAttentionPool(96)
        self.proj = nn.Sequential(
            nn.Linear(96 * 2, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        x = self.stem(spectra)
        x = self.blocks(x)
        mean_pooled = x.mean(dim=-1)
        attn_pooled = self.attention_pool(x)
        return self.proj(torch.cat([mean_pooled, attn_pooled], dim=-1))


class AuxEncoder(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(AUX_COLUMNS), 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.GELU(),
        )

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        return self.net(aux)


class FusionEncoder(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96 + 32, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

    def forward(self, spectral_feat: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([spectral_feat, aux_feat], dim=-1))


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(TARGET_COLUMNS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class RefinementProjector(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, hidden_width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class RefinementBlock(nn.Module):
    def __init__(self, latent_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        hidden_width = max(64, latent_dim * 2)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(latent_dim, hidden_width, dropout) for _ in range(layers)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(latent))


@dataclass
class ModelConfig:
    spectral_input_channels: int = 4
    dropout: float = 0.05
    refinement_width: int = 32
    refinement_layers: int = 2
    classical_only: bool = False
    use_amp: bool = True


class TauRExNoQuantRegressor(nn.Module):
    def __init__(
        self,
        spectral_encoder: SpectralEncoder,
        aux_encoder: AuxEncoder,
        fusion_encoder: FusionEncoder,
        classical_head: RegressionHead,
        refinement_projector: Optional[RefinementProjector],
        refinement_block: Optional[RefinementBlock],
        refinement_head: Optional[RegressionHead],
        device: torch.device,
        amp_dtype: Optional[torch.dtype],
        classical_only: bool,
    ) -> None:
        super().__init__()
        self.spectral_encoder = spectral_encoder.to(device)
        self.aux_encoder = aux_encoder.to(device)
        self.fusion_encoder = fusion_encoder.to(device)
        self.classical_head = classical_head.to(device)
        self.refinement_projector = refinement_projector.to(device) if refinement_projector is not None else None
        self.refinement_block = refinement_block.to(device) if refinement_block is not None else None
        self.refinement_head = refinement_head.to(device) if refinement_head is not None else None
        self.device_ref = device
        self.amp_dtype = amp_dtype
        self.classical_only = bool(classical_only)
        self.head_context_dim = 128 + 96 + 32

    def backbone_modules(self) -> tuple[nn.Module, ...]:
        return (self.spectral_encoder, self.aux_encoder, self.fusion_encoder, self.classical_head)

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.backbone_modules():
            yield from module.parameters()

    def refinement_parameters(self) -> Iterable[nn.Parameter]:
        if self.refinement_projector is not None:
            yield from self.refinement_projector.parameters()
        if self.refinement_block is not None:
            yield from self.refinement_block.parameters()
        if self.refinement_head is not None:
            yield from self.refinement_head.parameters()

    def set_backbone_trainable(self, enabled: bool) -> None:
        for module in self.backbone_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)

    def forward(
        self,
        aux: torch.Tensor,
        spectra: torch.Tensor,
        enable_refinement: bool = True,
        refinement_scale: float = 1.0,
    ) -> torch.Tensor:
        autocast_enabled = self.device_ref.type == "cuda" and self.amp_dtype is not None
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if autocast_enabled
            else nullcontext()
        )

        with autocast_ctx:
            spectral_feat = self.spectral_encoder(spectra)
            aux_feat = self.aux_encoder(aux)
            fused = self.fusion_encoder(spectral_feat, aux_feat)
            head_context = torch.cat([fused, spectral_feat, aux_feat], dim=-1)
            classical_pred = self.classical_head(head_context)

        if self.classical_only or not enable_refinement or refinement_scale <= 0.0:
            return classical_pred.float()

        if self.refinement_projector is None or self.refinement_block is None or self.refinement_head is None:
            raise RuntimeError("Refinement path is missing required modules.")

        head_context = head_context.float()
        latent = self.refinement_projector(fused.float())
        refinement_features = self.refinement_block(latent)
        refinement = self.refinement_head(torch.cat([head_context, refinement_features], dim=-1))
        return classical_pred.float() + float(refinement_scale) * refinement.float()


def resolve_amp_dtype(device: torch.device, use_amp: bool) -> Optional[torch.dtype]:
    if not use_amp or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_model(config: ModelConfig, device: torch.device) -> TauRExNoQuantRegressor:
    refinement_projector: Optional[RefinementProjector] = None
    refinement_block: Optional[RefinementBlock] = None
    refinement_head: Optional[RegressionHead] = None
    if not config.classical_only:
        refinement_projector = RefinementProjector(128, config.refinement_width, config.dropout)
        refinement_block = RefinementBlock(config.refinement_width, config.refinement_layers, config.dropout)
        refinement_head = RegressionHead(128 + 96 + 32 + config.refinement_width, 192, config.dropout)

    return TauRExNoQuantRegressor(
        spectral_encoder=SpectralEncoder(config.spectral_input_channels, config.dropout),
        aux_encoder=AuxEncoder(config.dropout),
        fusion_encoder=FusionEncoder(config.dropout),
        classical_head=RegressionHead(128 + 96 + 32, 192, config.dropout),
        refinement_projector=refinement_projector,
        refinement_block=refinement_block,
        refinement_head=refinement_head,
        device=device,
        amp_dtype=resolve_amp_dtype(device, config.use_amp),
        classical_only=config.classical_only,
    )
