"""Compatibility helpers for building FMPE models with Dingo."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch


def _import_dingo() -> Any:
    try:
        from dingo.core.posterior_models.build_model import build_model_from_kwargs
    except ImportError as exc:
        raise ImportError(
            "dingo-gw core is required for FMPE training. Install it with "
            "`python -m pip install --no-deps dingo-gw==0.8.3` and then install "
            "the platform-specific requirements file for this workflow "
            "(`models/sbi_ariel_crossgen/requirements-mac.txt` on macOS, "
            "`models/sbi_ariel_crossgen/requirements-vast.txt` on Vast/Linux). "
            f"Original import error: {exc}"
        ) from exc
    return build_model_from_kwargs


def expand_hidden_dims(hidden_dims_layers: list[int], multiplicative_factor_layers: int) -> list[int]:
    expanded: list[int] = []
    for width in hidden_dims_layers:
        expanded.extend([int(width)] * int(multiplicative_factor_layers))
    return expanded


def prepare_model_settings(settings: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(settings)
    prepared.setdefault("model", {})
    model_type = prepared["model"].get("posterior_model_type", prepared["model"].get("type", "flow_matching"))
    prepared["model"]["posterior_model_type"] = model_type
    prepared["model"].pop("type", None)
    prepared["model"].pop("prior", None)
    posterior_kwargs = prepared["model"].setdefault("posterior_kwargs", {})
    posterior_kwargs["input_dim"] = int(prepared["task"]["dim_theta"])
    posterior_kwargs["context_dim"] = int(prepared["task"]["dim_x"])
    posterior_kwargs["hidden_dims"] = expand_hidden_dims(
        posterior_kwargs["hidden_dims_layers"],
        posterior_kwargs["multiplicative_factor_layers"],
    )
    return prepared


def move_model_to_device(model: Any, device: torch.device) -> Any:
    model.device = device
    if hasattr(model, "network"):
        model.network = model.network.to(device)
    return model


def build_posterior_model(settings: dict[str, Any], device: torch.device) -> Any:
    build_model_from_kwargs = _import_dingo()
    prepared = prepare_model_settings(settings)
    model = build_model_from_kwargs(settings={"train_settings": prepared}, device="cpu")
    move_model_to_device(model, device)
    model.optimizer_kwargs = prepared["training"]["optimizer"]
    model.scheduler_kwargs = prepared["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()
    return model, prepared


def load_posterior_model(checkpoint_path: str | Path, device: torch.device) -> Any:
    build_model_from_kwargs = _import_dingo()
    model = build_model_from_kwargs(filename=str(checkpoint_path), device="cpu")
    move_model_to_device(model, device)
    return model


def build_resume_payload(model: Any, extra: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model_kwargs": model.model_kwargs,
        "model_state_dict": model.network.state_dict(),
        "epoch": model.epoch,
        "version": getattr(model, "version", None),
        "metadata": getattr(model, "metadata", None),
        "optimizer_kwargs": getattr(model, "optimizer_kwargs", None),
        "scheduler_kwargs": getattr(model, "scheduler_kwargs", None),
        "optimizer_state_dict": model.optimizer.state_dict() if getattr(model, "optimizer", None) is not None else None,
        "scheduler_state_dict": model.scheduler.state_dict() if getattr(model, "scheduler", None) is not None else None,
    }
    if getattr(model, "context", None) is not None:
        payload["context"] = model.context
    if getattr(model, "event_metadata", None) is not None:
        payload["event_metadata"] = model.event_metadata
    payload.update(extra)
    return payload
