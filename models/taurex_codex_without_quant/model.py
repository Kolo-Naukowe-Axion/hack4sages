"""Classical TauREx regressors with baseline and FiLM multiscale variants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import AUX_COLUMNS, TARGET_COLUMNS

SUPPORTED_ARCHITECTURES = ("legacy_conv_refiner", "pyramid_film")


def _make_group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _compute_derivative(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] < 2:
        return torch.zeros_like(values)
    left = values[..., :1]
    middle = 0.5 * (values[..., 2:] - values[..., :-2])
    right = values[..., -1:] - values[..., -2:-1]
    return torch.cat([values[..., 1:2] - left, middle, right], dim=-1)


class NoQuantRegressorBase(nn.Module, ABC):
    device_ref: torch.device
    amp_dtype: Optional[torch.dtype]
    classical_only: bool

    @abstractmethod
    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def refinement_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def set_backbone_trainable(self, enabled: bool) -> None:
        raise NotImplementedError


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
        self.output_dim = 96
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
            nn.Linear(96 * 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        )

    def forward(self, spectra: torch.Tensor, aux_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        del aux_context
        x = self.stem(spectra)
        x = self.blocks(x)
        mean_pooled = x.mean(dim=-1)
        attn_pooled = self.attention_pool(x)
        return self.proj(torch.cat([mean_pooled, attn_pooled], dim=-1))


class AuxEncoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = 32
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
        )

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        return self.net(aux)


class FusionEncoder(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.output_dim = 128
        self.net = nn.Sequential(
            nn.Linear(96 + 32, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
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


class GatedResidualMLPBlock(nn.Module):
    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden_width = max(width, width * expansion)
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden_width * 2)
        self.fc2 = nn.Linear(hidden_width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(self.norm(x))
        hidden = F.glu(hidden, dim=-1)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        return x + hidden


class FiLMGenerator(nn.Module):
    def __init__(self, aux_dim: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_dim, max(aux_dim, channels)),
            nn.GELU(),
            nn.Linear(max(aux_dim, channels), channels * 2),
        )

    def forward(self, aux_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.net(aux_feat).chunk(2, dim=-1)
        return 1.0 + 0.25 * torch.tanh(gamma), 0.25 * torch.tanh(beta)


class SqueezeExcitation1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(16, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpectralFeatureExpander(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.expanded_channels = input_channels + 3

    def forward(self, spectra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = spectra[:, 0:1, :]
        noise = spectra[:, 1:2, :]
        first_derivative = _compute_derivative(spectrum)
        second_derivative = _compute_derivative(first_derivative)
        local_contrast = spectrum - F.avg_pool1d(spectrum, kernel_size=7, stride=1, padding=3)
        expanded = torch.cat([spectra, first_derivative, second_derivative, local_contrast], dim=1)
        summary = torch.cat(
            [
                spectrum.mean(dim=-1),
                spectrum.std(dim=-1, unbiased=False),
                noise.mean(dim=-1),
                noise.std(dim=-1, unbiased=False),
                first_derivative.abs().mean(dim=-1),
                second_derivative.abs().mean(dim=-1),
                local_contrast.std(dim=-1, unbiased=False),
            ],
            dim=-1,
        )
        return expanded, summary


class FiLMMultiScaleSpectralBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aux_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        hidden_channels = max(in_channels, out_channels)
        self.norm = _make_group_norm(in_channels)
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.branch_small = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.branch_large = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.se = SqueezeExcitation1d(out_channels)
        self.film = FiLMGenerator(aux_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        hidden = self.in_proj(self.norm(x))
        hidden = self.branch_small(hidden) + self.branch_large(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.out_proj(hidden)
        hidden = self.se(hidden)
        gamma, beta = self.film(aux_feat)
        hidden = hidden * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
        hidden = self.dropout(hidden)
        return self.act(hidden + residual)


class FiLMSpectralEncoder(nn.Module):
    def __init__(self, input_channels: int, aux_dim: int, base_width: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = base_width * 2
        self.expander = SpectralFeatureExpander(input_channels)
        widths = (base_width, base_width + base_width // 2, base_width * 2, base_width * 2)
        self.stem = nn.Sequential(
            nn.Conv1d(self.expander.expanded_channels, widths[0], kernel_size=7, padding=3),
            _make_group_norm(widths[0]),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                FiLMMultiScaleSpectralBlock(widths[0], widths[0], aux_dim, dilation=1, dropout=dropout),
                FiLMMultiScaleSpectralBlock(widths[0], widths[1], aux_dim, dilation=2, dropout=dropout),
                FiLMMultiScaleSpectralBlock(widths[1], widths[2], aux_dim, dilation=4, dropout=dropout),
                FiLMMultiScaleSpectralBlock(widths[2], widths[3], aux_dim, dilation=1, dropout=dropout),
            ]
        )
        self.attention_pool = SpectralAttentionPool(widths[-1])
        self.summary_proj = nn.Sequential(
            nn.Linear(7, base_width),
            nn.LayerNorm(base_width),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(widths[-1] * 3 + base_width, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        )

    def forward(self, spectra: torch.Tensor, aux_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if aux_context is None:
            raise RuntimeError("FiLMSpectralEncoder requires aux_context.")
        expanded, summary = self.expander(spectra)
        x = self.stem(expanded)
        for block in self.blocks:
            x = block(x, aux_context)
        pooled = torch.cat(
            [
                x.mean(dim=-1),
                x.amax(dim=-1),
                self.attention_pool(x),
                self.summary_proj(summary),
            ],
            dim=-1,
        )
        return self.proj(pooled)


class FiLMAuxEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = width
        self.input_proj = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(
            GatedResidualMLPBlock(width, expansion=2, dropout=dropout),
            GatedResidualMLPBlock(width, expansion=2, dropout=dropout),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(width),
            nn.GELU(),
        )

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.input_proj(aux)))


class FiLMFusionEncoder(nn.Module):
    def __init__(self, spectral_dim: int, aux_dim: int, base_width: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = base_width * 3
        self.input_proj = nn.Linear(spectral_dim + aux_dim, self.output_dim)
        self.blocks = nn.Sequential(
            GatedResidualMLPBlock(self.output_dim, expansion=2, dropout=dropout),
            GatedResidualMLPBlock(self.output_dim, expansion=2, dropout=dropout),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        )

    def forward(self, spectral_feat: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
        fused = self.input_proj(torch.cat([spectral_feat, aux_feat], dim=-1))
        return self.out(self.blocks(fused))


class TargetwiseRegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        bottleneck = max(32, hidden_dim // 2)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, bottleneck),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(bottleneck, 1),
                )
                for _ in TARGET_COLUMNS
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(features)
        return torch.cat([head(hidden) for head in self.heads], dim=-1)


@dataclass
class ModelConfig:
    spectral_input_channels: int = 4
    aux_input_dim: int = len(AUX_COLUMNS)
    dropout: float = 0.05
    refinement_width: int = 32
    refinement_layers: int = 2
    classical_only: bool = False
    use_amp: bool = True
    architecture: str = "legacy_conv_refiner"
    spectral_width: int = 64


def _resolve_architecture_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "baseline": "baseline",
        "conv_refiner": "baseline",
        "legacy": "baseline",
        "legacy_conv": "baseline",
        "legacy_conv_refiner": "baseline",
        "film_multiscale": "film_multiscale",
        "multiscale_film": "film_multiscale",
        "film": "film_multiscale",
        "pyramid_film": "film_multiscale",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported architecture={name!r}. Expected one of {SUPPORTED_ARCHITECTURES}.")
    return aliases[normalized]


class TauRExNoQuantRegressor(NoQuantRegressorBase):
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


class FiLMMultiScaleTauRExRegressor(NoQuantRegressorBase):
    def __init__(
        self,
        spectral_encoder: FiLMSpectralEncoder,
        aux_encoder: FiLMAuxEncoder,
        fusion_encoder: FiLMFusionEncoder,
        classical_head: TargetwiseRegressionHead,
        refinement_projector: Optional[RefinementProjector],
        refinement_block: Optional[RefinementBlock],
        refinement_head: Optional[TargetwiseRegressionHead],
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
        self.head_context_dim = (
            self.fusion_encoder.output_dim + self.spectral_encoder.output_dim + self.aux_encoder.output_dim
        )

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
            aux_feat = self.aux_encoder(aux)
            spectral_feat = self.spectral_encoder(spectra, aux_context=aux_feat)
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


def build_model(config: ModelConfig, device: torch.device) -> NoQuantRegressorBase:
    architecture = _resolve_architecture_name(config.architecture)
    amp_dtype = resolve_amp_dtype(device, config.use_amp)

    if architecture == "baseline":
        refinement_projector: Optional[RefinementProjector] = None
        refinement_block: Optional[RefinementBlock] = None
        refinement_head: Optional[RegressionHead] = None
        if not config.classical_only:
            refinement_projector = RefinementProjector(128, config.refinement_width, config.dropout)
            refinement_block = RefinementBlock(config.refinement_width, config.refinement_layers, config.dropout)
            refinement_head = RegressionHead(128 + 96 + 32 + config.refinement_width, 192, config.dropout)

        return TauRExNoQuantRegressor(
            spectral_encoder=SpectralEncoder(config.spectral_input_channels, config.dropout),
            aux_encoder=AuxEncoder(config.aux_input_dim, config.dropout),
            fusion_encoder=FusionEncoder(config.dropout),
            classical_head=RegressionHead(128 + 96 + 32, 192, config.dropout),
            refinement_projector=refinement_projector,
            refinement_block=refinement_block,
            refinement_head=refinement_head,
            device=device,
            amp_dtype=amp_dtype,
            classical_only=config.classical_only,
        )

    base_width = max(32, int(config.spectral_width))
    aux_encoder = FiLMAuxEncoder(input_dim=config.aux_input_dim, width=base_width, dropout=config.dropout)
    spectral_encoder = FiLMSpectralEncoder(
        input_channels=config.spectral_input_channels,
        aux_dim=aux_encoder.output_dim,
        base_width=base_width,
        dropout=config.dropout,
    )
    fusion_encoder = FiLMFusionEncoder(
        spectral_dim=spectral_encoder.output_dim,
        aux_dim=aux_encoder.output_dim,
        base_width=base_width,
        dropout=config.dropout,
    )
    head_context_dim = fusion_encoder.output_dim + spectral_encoder.output_dim + aux_encoder.output_dim
    head_hidden_dim = max(192, base_width * 4)
    refinement_projector = None
    refinement_block = None
    refinement_head = None
    if not config.classical_only:
        refinement_projector = RefinementProjector(
            fusion_encoder.output_dim,
            config.refinement_width,
            config.dropout,
        )
        refinement_block = RefinementBlock(config.refinement_width, config.refinement_layers, config.dropout)
        refinement_head = TargetwiseRegressionHead(
            head_context_dim + config.refinement_width,
            head_hidden_dim,
            config.dropout,
        )

    return FiLMMultiScaleTauRExRegressor(
        spectral_encoder=spectral_encoder,
        aux_encoder=aux_encoder,
        fusion_encoder=fusion_encoder,
        classical_head=TargetwiseRegressionHead(head_context_dim, head_hidden_dim, config.dropout),
        refinement_projector=refinement_projector,
        refinement_block=refinement_block,
        refinement_head=refinement_head,
        device=device,
        amp_dtype=amp_dtype,
        classical_only=config.classical_only,
    )
