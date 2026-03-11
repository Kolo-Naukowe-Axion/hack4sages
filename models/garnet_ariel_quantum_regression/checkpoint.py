"""Checkpoint bridge for the Garnet Ariel quantum regression port."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from models.ariel_quantum_regression.dataset import ArrayStandardizer, SpectralStandardizer

from .constants import DEFAULT_CHECKPOINT_DIR


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Torch is required to load the Ariel hybrid checkpoint.") from exc
    return torch


@dataclass(frozen=True)
class CheckpointBundle:
    checkpoint_dir: Path
    checkpoint_payload: dict[str, Any]
    config: dict[str, Any]
    model: Any
    model_config: Any
    aux_scaler: ArrayStandardizer
    target_scaler: ArrayStandardizer
    spectral_scaler: SpectralStandardizer

    @property
    def state_dict(self) -> dict[str, Any]:
        return self.checkpoint_payload["model_state_dict"]

    @property
    def qnn_qubits(self) -> int:
        return int(self.config["qnn_qubits"])

    @property
    def qnn_depth(self) -> int:
        return int(self.config["qnn_depth"])

    @property
    def quantum_gate(self) -> np.ndarray:
        return self.state_dict["quantum_gate"].detach().cpu().numpy().astype(np.float32)

    @property
    def quantum_weights(self) -> np.ndarray:
        return self.state_dict["quantum_block.weights"].detach().cpu().numpy().astype(np.float32)


class FrozenArielHybridBridge:
    """Expose explicit frozen stages from the original hybrid model."""

    def __init__(self, bundle: CheckpointBundle) -> None:
        self.bundle = bundle
        self.model = bundle.model
        self.torch = import_torch()

    def _as_tensor(self, values: Any) -> Any:
        if isinstance(values, self.torch.Tensor):
            return values.to(device=self.model.classical_device, dtype=self.torch.float32)
        return self.torch.as_tensor(values, dtype=self.torch.float32, device=self.model.classical_device)

    def encode_features(self, aux: Any, spectra: Any) -> dict[str, Any]:
        aux_tensor = self._as_tensor(aux)
        spectra_tensor = self._as_tensor(spectra)
        with self.torch.inference_mode():
            spectral_feat = self.model.spectral_encoder(spectra_tensor)
            aux_feat = self.model.aux_encoder(aux_tensor)
            fused = self.model.fusion_encoder(spectral_feat, aux_feat)
            head_context = self.torch.cat([fused, spectral_feat, aux_feat], dim=-1)
        return {
            "aux": aux_tensor,
            "spectra": spectra_tensor,
            "spectral_feat": spectral_feat,
            "aux_feat": aux_feat,
            "fused": fused,
            "head_context": head_context,
        }

    def classical_predict(self, head_context: Any) -> Any:
        with self.torch.inference_mode():
            return self.model.classical_head(self._as_tensor(head_context)).float()

    def project_quantum_angles(self, fused: Any) -> Any:
        if self.model.projector is None:
            raise RuntimeError("Checkpoint does not contain a quantum projector.")
        with self.torch.inference_mode():
            return self.model.projector(self._as_tensor(fused)).float()

    def combine_predictions(self, head_context: Any, quantum_features: Any, quantum_scale: float = 1.0) -> Any:
        if self.model.quantum_head is None:
            raise RuntimeError("Checkpoint does not contain a quantum correction head.")
        head_context_tensor = self._as_tensor(head_context)
        quantum_tensor = self._as_tensor(quantum_features)
        with self.torch.inference_mode():
            classical_pred = self.model.classical_head(head_context_tensor).float()
            quantum_correction = self.model.quantum_head(
                self.torch.cat([head_context_tensor, quantum_tensor], dim=-1)
            ).float()
            gate = self.torch.tanh(self.model.quantum_gate).view(1, -1)
            return classical_pred + float(quantum_scale) * gate * quantum_correction


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_model(checkpoint_payload: dict[str, Any], config: dict[str, Any]) -> tuple[Any, Any]:
    torch = import_torch()
    from models.ariel_quantum_regression.model import (
        AuxEncoder,
        FusionEncoder,
        ModelConfig,
        QuantumProjector,
        RegressionHead,
        SpectralEncoder,
        TARGET_COLUMNS,
    )

    class FrozenHybridModules(torch.nn.Module):
        """Minimal checkpoint-compatible module set without PennyLane dependencies."""

        def __init__(self) -> None:
            super().__init__()
            self.spectral_encoder = SpectralEncoder(
                int(config.get("spectral_input_channels", 4)),
                float(config["dropout"]),
            )
            self.aux_encoder = AuxEncoder(float(config["dropout"]))
            self.fusion_encoder = FusionEncoder(float(config["dropout"]))
            self.classical_head = RegressionHead(128 + 96 + 32, 192, float(config["dropout"]))
            self.projector = QuantumProjector(float(config["dropout"]), int(config["qnn_qubits"]))
            self.quantum_head = RegressionHead(
                128 + 96 + 32 + int(config["qnn_qubits"]),
                192,
                float(config["dropout"]),
            )
            self.quantum_gate = torch.nn.Parameter(torch.zeros(len(TARGET_COLUMNS), dtype=torch.float32))
            self.classical_device = torch.device("cpu")

    model_config = ModelConfig(
        spectral_input_channels=int(config.get("spectral_input_channels", 4)),
        dropout=float(config["dropout"]),
        qnn_qubits=int(config["qnn_qubits"]),
        qnn_depth=int(config["qnn_depth"]),
        qnn_init_scale=float(config["qnn_init_scale"]),
        quantum_device="default.qubit",
        quantum_use_async=False,
        classical_only=bool(config["classical_only"]),
        use_amp=False,
    )
    model = FrozenHybridModules()
    model.load_state_dict(
        {
            key: value
            for key, value in checkpoint_payload["model_state_dict"].items()
            if key.startswith(
                (
                    "spectral_encoder.",
                    "aux_encoder.",
                    "fusion_encoder.",
                    "classical_head.",
                    "projector.",
                    "quantum_head.",
                    "quantum_gate",
                )
            )
        },
        strict=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_config


def load_checkpoint_bundle(checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> CheckpointBundle:
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    torch = import_torch()
    payload = torch.load(checkpoint_root / "best_model.pt", map_location="cpu")
    config = payload.get("config") or _load_json(checkpoint_root / "config.json")
    scalers = _load_json(checkpoint_root / "scalers.json")
    model, model_config = _load_model(payload, config)
    return CheckpointBundle(
        checkpoint_dir=checkpoint_root,
        checkpoint_payload=payload,
        config=config,
        model=model,
        model_config=model_config,
        aux_scaler=ArrayStandardizer.from_state_dict(scalers["aux_scaler"]),
        target_scaler=ArrayStandardizer.from_state_dict(scalers["target_scaler"]),
        spectral_scaler=SpectralStandardizer.from_state_dict(scalers["spectral_scaler"]),
    )


def load_default_checkpoint_bundle() -> CheckpointBundle:
    return load_checkpoint_bundle(DEFAULT_CHECKPOINT_DIR)


def build_frozen_bridge(checkpoint_dir: Optional[str | Path] = None, bundle: Optional[CheckpointBundle] = None) -> FrozenArielHybridBridge:
    bundle = bundle or load_checkpoint_bundle(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
    return FrozenArielHybridBridge(bundle)
