"""Model definition for the five-qubit hybrid regressor."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.hybrid_core.model import (
    ClassicalBranchStrategy,
    ClassicalOnlyHybridRegressor,
    FusionEncoder,
    HybridArielRegressor,
    HybridRegressorBase,
    ModelConfig as _ModelConfig,
    PredictionBranchStrategy,
    PredictionContext,
    QuantumBlock,
    QuantumBranchStrategy,
    QuantumProjector,
    RegressionHead,
    SpectralEncoder,
    AuxEncoder,
    build_model as _build_model,
    resolve_amp_dtype,
)


@dataclass
class ModelConfig(_ModelConfig):
    qnn_qubits: int = 5


def build_model(config: ModelConfig, device: torch.device) -> HybridRegressorBase:
    return _build_model(config, device)


__all__ = [
    "AuxEncoder",
    "ClassicalBranchStrategy",
    "ClassicalOnlyHybridRegressor",
    "FusionEncoder",
    "HybridArielRegressor",
    "HybridRegressorBase",
    "ModelConfig",
    "PredictionBranchStrategy",
    "PredictionContext",
    "QuantumBlock",
    "QuantumBranchStrategy",
    "QuantumProjector",
    "RegressionHead",
    "SpectralEncoder",
    "build_model",
    "resolve_amp_dtype",
]
