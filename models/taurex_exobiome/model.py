"""Model definition for the TauREx five-gas hybrid regressor."""

from models.hybrid_core.model import (
    ClassicalBranchStrategy,
    ClassicalOnlyHybridRegressor,
    FusionEncoder,
    HybridArielRegressor,
    HybridRegressorBase,
    ModelConfig,
    PredictionBranchStrategy,
    PredictionContext,
    QuantumBlock,
    QuantumBranchStrategy,
    QuantumProjector,
    RegressionHead,
    SpectralEncoder,
    AuxEncoder,
    build_model,
    resolve_amp_dtype,
)

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
