"""Five-qubit Codex Ariel hybrid quantum regressor."""

from __future__ import annotations

from typing import Any

__all__ = ["TrainingConfig", "build_model", "run_training_experiment"]


def __getattr__(name: str) -> Any:
    if name == "build_model":
        from .model import build_model

        return build_model
    if name in {"TrainingConfig", "run_training_experiment"}:
        from .training import TrainingConfig, run_training_experiment

        return {"TrainingConfig": TrainingConfig, "run_training_experiment": run_training_experiment}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
