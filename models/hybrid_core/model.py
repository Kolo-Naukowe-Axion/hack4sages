"""Shared hybrid regressor OOP core with ABC, Strategy, and Factory patterns."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

DEFAULT_AUX_INPUT_DIM = 8
DEFAULT_NUM_TARGETS = 5
HEAD_CONTEXT_DIM = 128 + 96 + 32


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
    def __init__(self, dropout: float, aux_input_dim: int = DEFAULT_AUX_INPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_input_dim, 32),
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
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float,
        num_targets: int = DEFAULT_NUM_TARGETS,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def import_pennylane() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for hybrid mode. Install models/requirements-ariel-quantum.txt."
        ) from exc
    return qml


class QuantumProjector(nn.Module):
    def __init__(self, dropout: float, n_qubits: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_qubits),
            nn.LayerNorm(n_qubits),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(fused)) * math.pi


class QuantumBlock(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        depth: int,
        quantum_device_name: str,
        init_scale: float,
        use_async: bool,
    ) -> None:
        super().__init__()
        if depth % 2 != 0:
            raise ValueError("qnn_depth must be even for the current variational block.")

        self.n_qubits = int(n_qubits)
        self.depth = int(depth)
        self.quantum_device_name = quantum_device_name
        self.qml = import_pennylane()
        device_kwargs: dict[str, Any] = {"wires": self.n_qubits}
        if quantum_device_name.startswith("lightning."):
            device_kwargs["c_dtype"] = np.complex64
        if quantum_device_name == "lightning.gpu" and use_async:
            device_kwargs["use_async"] = True
        try:
            self.device = self.qml.device(quantum_device_name, **device_kwargs)
        except TypeError as exc:
            if quantum_device_name == "lightning.gpu" and "use_async" in device_kwargs and "use_async" in str(exc):
                device_kwargs.pop("use_async", None)
                self.device = self.qml.device(quantum_device_name, **device_kwargs)
            else:
                raise
        self.num_blocks = self.depth // 2
        self.weights = nn.Parameter(
            float(init_scale) * torch.randn(3 * self.n_qubits * self.num_blocks, dtype=torch.float32)
        )
        self.qnode = self._build_qnode()

    def _build_qnode(self) -> Any:
        qml = self.qml
        n_qubits = self.n_qubits
        num_blocks = self.num_blocks
        device = self.device

        @qml.qnode(device, interface="torch", diff_method="adjoint")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            for qubit in range(n_qubits):
                qml.RY(inputs[..., qubit], wires=qubit)

            param_index = 0
            for _ in range(num_blocks):
                for qubit in range(n_qubits):
                    qml.RY(weights[param_index], wires=qubit)
                    param_index += 1
                for qubit in range(n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
                for qubit in range(n_qubits):
                    qml.RZ(weights[param_index], wires=qubit)
                    param_index += 1
                for qubit in range(n_qubits):
                    qml.CRX(weights[param_index], wires=[qubit, (qubit + 1) % n_qubits])
                    param_index += 1

            return [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]

        return circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.qnode(inputs.to(dtype=torch.float32), self.weights)
        if isinstance(outputs, (list, tuple)):
            outputs = torch.stack(tuple(outputs), dim=-1)
        return outputs.to(dtype=torch.float32)


@dataclass
class ModelConfig:
    spectral_input_channels: int = 4
    dropout: float = 0.1
    qnn_qubits: int = 8
    qnn_depth: int = 2
    qnn_init_scale: float = 0.1
    quantum_device: str = "lightning.qubit"
    quantum_use_async: bool = False
    classical_only: bool = False
    use_amp: bool = True
    aux_input_dim: int = DEFAULT_AUX_INPUT_DIM
    num_targets: int = DEFAULT_NUM_TARGETS


@dataclass
class PredictionContext:
    fused: torch.Tensor
    head_context: torch.Tensor
    classical_pred: torch.Tensor
    quantum_scale: float
    regressor: HybridArielRegressor


class PredictionBranchStrategy(ABC):
    @abstractmethod
    def apply(self, context: PredictionContext) -> torch.Tensor:
        raise NotImplementedError


class ClassicalBranchStrategy(PredictionBranchStrategy):
    def apply(self, context: PredictionContext) -> torch.Tensor:
        return context.classical_pred.float()


class QuantumBranchStrategy(PredictionBranchStrategy):
    def apply(self, context: PredictionContext) -> torch.Tensor:
        model = context.regressor
        if model.projector is None or model.quantum_block is None or model.quantum_head is None:
            raise RuntimeError("Hybrid model is missing quantum components.")

        fused = context.fused.float()
        head_context = context.head_context.float()
        classical_pred = context.classical_pred.float()
        quantum_angles = model.projector(fused)
        quantum_features = model.quantum_block(quantum_angles.float().to(model.quantum_torch_device))
        if quantum_features.device != model.classical_device:
            quantum_features = quantum_features.to(model.classical_device)
        quantum_correction = model.quantum_head(torch.cat([head_context, quantum_features], dim=-1))
        gate = torch.tanh(model.quantum_gate).view(1, -1)
        return classical_pred + float(context.quantum_scale) * gate * quantum_correction.float()


_CLASSICAL_BRANCH_STRATEGY = ClassicalBranchStrategy()
_QUANTUM_BRANCH_STRATEGY = QuantumBranchStrategy()


class HybridRegressorBase(nn.Module, ABC):
    classical_device: torch.device
    amp_dtype: Optional[torch.dtype]
    classical_only: bool
    head_context_dim: int

    spectral_encoder: SpectralEncoder
    aux_encoder: AuxEncoder
    fusion_encoder: FusionEncoder
    classical_head: RegressionHead

    @abstractmethod
    def forward(
        self,
        aux: torch.Tensor,
        spectra: torch.Tensor,
        enable_quantum: bool = True,
        quantum_scale: float = 1.0,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def quantum_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def quantum_adapter_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def set_backbone_trainable(self, enabled: bool) -> None:
        raise NotImplementedError

    def backbone_modules(self) -> tuple[nn.Module, ...]:
        return (self.spectral_encoder, self.aux_encoder, self.fusion_encoder, self.classical_head)

    def _autocast_context(self):
        autocast_enabled = self.classical_device.type == "cuda" and self.amp_dtype is not None
        if autocast_enabled:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _compute_classical_prediction(
        self,
        aux: torch.Tensor,
        spectra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._autocast_context():
            spectral_feat = self.spectral_encoder(spectra)
            aux_feat = self.aux_encoder(aux)
            fused = self.fusion_encoder(spectral_feat, aux_feat)
            head_context = torch.cat([fused, spectral_feat, aux_feat], dim=-1)
            classical_pred = self.classical_head(head_context)
        return fused, head_context, classical_pred


class ClassicalOnlyHybridRegressor(HybridRegressorBase):
    def __init__(
        self,
        spectral_encoder: SpectralEncoder,
        aux_encoder: AuxEncoder,
        fusion_encoder: FusionEncoder,
        classical_head: RegressionHead,
        classical_device: torch.device,
        amp_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()
        self.spectral_encoder = spectral_encoder.to(classical_device)
        self.aux_encoder = aux_encoder.to(classical_device)
        self.fusion_encoder = fusion_encoder.to(classical_device)
        self.classical_head = classical_head.to(classical_device)
        self.classical_device = classical_device
        self.amp_dtype = amp_dtype
        self.classical_only = True
        self.head_context_dim = HEAD_CONTEXT_DIM

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.backbone_modules():
            yield from module.parameters()

    def quantum_parameters(self) -> Iterable[nn.Parameter]:
        return iter(())

    def quantum_adapter_parameters(self) -> Iterable[nn.Parameter]:
        return iter(())

    def set_backbone_trainable(self, enabled: bool) -> None:
        for module in self.backbone_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)

    def forward(
        self,
        aux: torch.Tensor,
        spectra: torch.Tensor,
        enable_quantum: bool = True,
        quantum_scale: float = 1.0,
    ) -> torch.Tensor:
        del enable_quantum, quantum_scale
        _, _, classical_pred = self._compute_classical_prediction(aux, spectra)
        return classical_pred.float()


class HybridArielRegressor(HybridRegressorBase):
    def __init__(
        self,
        spectral_encoder: SpectralEncoder,
        aux_encoder: AuxEncoder,
        fusion_encoder: FusionEncoder,
        classical_head: RegressionHead,
        projector: QuantumProjector,
        quantum_block: QuantumBlock,
        quantum_head: RegressionHead,
        classical_device: torch.device,
        quantum_torch_device: torch.device,
        amp_dtype: Optional[torch.dtype],
        num_targets: int = DEFAULT_NUM_TARGETS,
    ) -> None:
        super().__init__()
        self.spectral_encoder = spectral_encoder.to(classical_device)
        self.aux_encoder = aux_encoder.to(classical_device)
        self.fusion_encoder = fusion_encoder.to(classical_device)
        self.classical_head = classical_head.to(classical_device)
        self.projector = projector.to(classical_device)
        self.quantum_block = quantum_block.to(quantum_torch_device)
        self.quantum_head = quantum_head.to(classical_device)
        self.quantum_gate = nn.Parameter(
            torch.zeros(num_targets, dtype=torch.float32, device=classical_device)
        )
        self.classical_device = classical_device
        self.quantum_torch_device = quantum_torch_device
        self.amp_dtype = amp_dtype
        self.classical_only = False
        self.head_context_dim = HEAD_CONTEXT_DIM

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.backbone_modules():
            yield from module.parameters()

    def quantum_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.quantum_block.parameters()

    def quantum_adapter_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.projector.parameters()
        yield from self.quantum_head.parameters()
        yield self.quantum_gate

    def set_backbone_trainable(self, enabled: bool) -> None:
        for module in self.backbone_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)

    def _select_branch_strategy(
        self,
        enable_quantum: bool,
        quantum_scale: float,
    ) -> PredictionBranchStrategy:
        if not enable_quantum or quantum_scale <= 0.0:
            return _CLASSICAL_BRANCH_STRATEGY
        return _QUANTUM_BRANCH_STRATEGY

    def forward(
        self,
        aux: torch.Tensor,
        spectra: torch.Tensor,
        enable_quantum: bool = True,
        quantum_scale: float = 1.0,
    ) -> torch.Tensor:
        fused, head_context, classical_pred = self._compute_classical_prediction(aux, spectra)
        strategy = self._select_branch_strategy(enable_quantum, quantum_scale)
        context = PredictionContext(
            fused=fused,
            head_context=head_context,
            classical_pred=classical_pred,
            quantum_scale=quantum_scale,
            regressor=self,
        )
        return strategy.apply(context)


def resolve_amp_dtype(device: torch.device, use_amp: bool) -> Optional[torch.dtype]:
    if not use_amp or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_model(config: ModelConfig, device: torch.device) -> HybridRegressorBase:
    amp_dtype = resolve_amp_dtype(device, config.use_amp)
    spectral_encoder = SpectralEncoder(config.spectral_input_channels, config.dropout)
    aux_encoder = AuxEncoder(config.dropout, config.aux_input_dim)
    fusion_encoder = FusionEncoder(config.dropout)
    classical_head = RegressionHead(
        HEAD_CONTEXT_DIM,
        192,
        config.dropout,
        num_targets=config.num_targets,
    )

    if config.classical_only:
        return ClassicalOnlyHybridRegressor(
            spectral_encoder=spectral_encoder,
            aux_encoder=aux_encoder,
            fusion_encoder=fusion_encoder,
            classical_head=classical_head,
            classical_device=device,
            amp_dtype=amp_dtype,
        )

    quantum_torch_device = device
    if config.quantum_device != "lightning.gpu":
        quantum_torch_device = torch.device("cpu")

    return HybridArielRegressor(
        spectral_encoder=spectral_encoder,
        aux_encoder=aux_encoder,
        fusion_encoder=fusion_encoder,
        classical_head=classical_head,
        projector=QuantumProjector(config.dropout, config.qnn_qubits),
        quantum_block=QuantumBlock(
            config.qnn_qubits,
            config.qnn_depth,
            config.quantum_device,
            config.qnn_init_scale,
            config.quantum_use_async,
        ),
        quantum_head=RegressionHead(
            HEAD_CONTEXT_DIM + config.qnn_qubits,
            192,
            config.dropout,
            num_targets=config.num_targets,
        ),
        classical_device=device,
        quantum_torch_device=quantum_torch_device,
        amp_dtype=amp_dtype,
        num_targets=config.num_targets,
    )
