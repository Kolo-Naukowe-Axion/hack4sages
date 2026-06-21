"""Shared OOP core for hybrid classical/quantum exobiome regressors."""

from .model import (
    ClassicalBranchStrategy,
    ClassicalOnlyHybridRegressor,
    HybridArielRegressor,
    HybridRegressorBase,
    ModelConfig,
    PredictionBranchStrategy,
    PredictionContext,
    QuantumBranchStrategy,
    build_model,
    resolve_amp_dtype,
)

__all__ = [
    "ClassicalBranchStrategy",
    "ClassicalOnlyHybridRegressor",
    "HybridArielRegressor",
    "HybridRegressorBase",
    "ModelConfig",
    "PredictionBranchStrategy",
    "PredictionContext",
    "QuantumBranchStrategy",
    "build_model",
    "resolve_amp_dtype",
]
