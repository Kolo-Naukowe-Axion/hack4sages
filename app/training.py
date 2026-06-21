"""Osoba 03 training orchestration for the ExoBiome app.

This layer is intentionally thin: the heavy model implementation stays in
``models.ariel_quantum_regression.training`` while the app gets a stable OOP
facade for two-stage training, callbacks, metrics, and artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from models.ariel_quantum_regression.training import TrainingConfig


MetricMap = Mapping[str, Any]


class Callback(Protocol):
    """Observer contract used by the app/UI/logging layer."""

    def __call__(self, event: "TrainingEvent") -> None: ...


class StageRunner(Protocol):
    """Dependency-injection contract for real or fake training backends."""

    def __call__(self, config: TrainingConfig) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class StageSpec:
    """Immutable description of a single training stage."""

    name: str
    config: TrainingConfig
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or any(part in self.name for part in ("/", "\\")):
            raise ValueError("StageSpec.name must be a non-empty path-safe name.")
        object.__setattr__(self, "config", replace(self.config))


@dataclass(frozen=True)
class StageResult:
    """Collected metrics and artifacts for one completed stage."""

    stage_name: str
    output_dir: Path
    config: TrainingConfig
    best_checkpoint: Path
    artifacts: Mapping[str, Path]
    validation_metrics: MetricMap
    holdout_metrics: MetricMap
    run_summary: MetricMap

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "best_checkpoint", Path(self.best_checkpoint))
        object.__setattr__(self, "config", replace(self.config))
        object.__setattr__(self, "artifacts", _freeze_mapping(self.artifacts))
        object.__setattr__(self, "validation_metrics", _freeze_mapping(self.validation_metrics))
        object.__setattr__(self, "holdout_metrics", _freeze_mapping(self.holdout_metrics))
        object.__setattr__(self, "run_summary", _freeze_mapping(self.run_summary))


@dataclass(frozen=True)
class TwoStageResult:
    """Classical pretrain followed by hybrid fine-tune."""

    stage1: StageResult
    stage2: StageResult

    @property
    def best_checkpoint(self) -> Path:
        return self.stage2.best_checkpoint


@dataclass(frozen=True)
class TwoStagePlan:
    """Dry-run preview of the default two-stage workflow."""

    stage1: StageSpec
    stage2: StageSpec

    @property
    def stages(self) -> tuple[StageSpec, StageSpec]:
        return (self.stage1, self.stage2)

    @property
    def expected_best_checkpoint(self) -> Path:
        return Path(self.stage2.config.output_dir) / "best_model.pt"


@dataclass(frozen=True)
class TrainingEvent:
    """Deterministic callback payload; no wall-clock timestamp on purpose."""

    stage_name: str
    status: str
    output_dir: Path
    message: str = ""
    metrics: MetricMap = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "metrics", _freeze_mapping(self.metrics))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "output_dir": str(self.output_dir),
            "message": self.message,
            "metrics": dict(self.metrics),
        }


def default_stage_runner(config: TrainingConfig) -> Mapping[str, Any]:
    """Run the repo's existing training experiment."""

    from models.ariel_quantum_regression.training import run_training_experiment

    return run_training_experiment(config)


def clone_config(config: TrainingConfig, **updates: Any) -> TrainingConfig:
    """Pure dataclass update helper used by stage strategies."""

    return replace(config, **updates)


def make_classical_pretrain_spec(
    base_config: TrainingConfig,
    output_dir: str | Path,
    *,
    prepared_cache_dir: str | Path | None,
) -> StageSpec:
    """Default stage 1: train only the classical backbone/head."""

    config = clone_config(
        base_config,
        output_dir=str(output_dir),
        prepared_cache_dir=None if prepared_cache_dir is None else str(prepared_cache_dir),
        init_checkpoint_path=None,
        classical_only=True,
        quantum_warmup_epochs=0,
    )
    return StageSpec(
        name="stage1_classical",
        config=config,
        description="Classical-only pretraining stage.",
    )


def make_hybrid_finetune_spec(
    base_config: TrainingConfig,
    output_dir: str | Path,
    init_checkpoint_path: str | Path,
    *,
    prepared_cache_dir: str | Path | None,
) -> StageSpec:
    """Default stage 2: load stage 1 and enable the hybrid branch."""

    config = clone_config(
        base_config,
        output_dir=str(output_dir),
        prepared_cache_dir=None if prepared_cache_dir is None else str(prepared_cache_dir),
        init_checkpoint_path=str(init_checkpoint_path),
        classical_only=False,
    )
    return StageSpec(
        name="stage2_hybrid",
        config=config,
        description="Hybrid fine-tuning stage initialized from stage 1.",
    )


def read_json_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON file; return an empty mapping when the file is absent."""

    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text())


def read_stage_metrics(output_dir: str | Path) -> tuple[MetricMap, MetricMap]:
    """Return ``(validation_metrics, holdout_metrics)`` for a stage."""

    root = Path(output_dir)
    return (
        read_json_file(root / "validation_metrics.json"),
        read_json_file(root / "holdout_metrics.json"),
    )


def select_best_checkpoint(output_dir: str | Path) -> Path:
    """Prefer ``best_model.pt`` and fall back to ``last_model.pt``."""

    root = Path(output_dir)
    for name in ("best_model.pt", "last_model.pt"):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model checkpoint found in {root}.")


def collect_artifacts(output_dir: str | Path) -> Mapping[str, Path]:
    """Collect the stable artifacts Osoba 04/UI code can depend on."""

    root = Path(output_dir)
    names = (
        "best_model.pt",
        "last_model.pt",
        "history.csv",
        "training_state.json",
        "config.json",
        "scalers.json",
        "validation_metrics.json",
        "holdout_metrics.json",
        "validation_predictions.csv",
        "holdout_predictions.csv",
        "testdata_predictions.csv",
        "run_summary.json",
        "artifacts_manifest.json",
        "failed_stage.json",
    )
    return {name: root / name for name in names if (root / name).exists()}


def flatten_result_metrics(result: StageResult) -> dict[str, float]:
    """Small numeric summary suitable for callbacks and UI status."""

    out: dict[str, float] = {}
    for prefix, metrics in (
        ("validation", result.validation_metrics),
        ("holdout", result.holdout_metrics),
    ):
        for key in ("rmse_mean", "mae_mean"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                out[f"{prefix}_{key}"] = float(value)
    return out


def format_metric_summary(result: StageResult) -> str:
    """Human-readable one-line metric summary for logs/report snippets."""

    metrics = flatten_result_metrics(result)
    if not metrics:
        return f"{result.stage_name}: metrics unavailable"
    values = " | ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
    return f"{result.stage_name}: {values}"


def has_nonfinite_metric(metrics: Mapping[str, Any]) -> bool:
    """True when any numeric metric is NaN or infinite."""

    return any(
        isinstance(value, (int, float)) and not math.isfinite(float(value))
        for value in metrics.values()
    )


def make_fail_on_nan_callback() -> Callback:
    """Callback that fails the run when collected metrics contain NaN/inf."""

    def fail_on_nan(event: TrainingEvent) -> None:
        if event.status == "completed" and has_nonfinite_metric(event.metrics):
            raise RuntimeError(f"Non-finite metric detected after {event.stage_name}.")

    return fail_on_nan


def make_jsonl_logger(path: str | Path) -> Callback:
    """Build a simple JSONL event logger callback."""

    target = Path(path)

    def log_event(event: TrainingEvent) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_json_dict(), sort_keys=True) + "\n")

    return log_event


def stage_spec_to_json_dict(spec: StageSpec) -> dict[str, Any]:
    """Serialize a stage spec for dry-run previews and manifests."""

    return {
        "name": spec.name,
        "description": spec.description,
        "config": _config_payload(spec.config),
    }


def plan_to_json_dict(plan: TwoStagePlan) -> dict[str, Any]:
    """Serialize a two-stage plan without running any training."""

    return {
        "stages": [stage_spec_to_json_dict(stage) for stage in plan.stages],
        "expected_best_checkpoint": str(plan.expected_best_checkpoint),
    }


def artifact_manifest_payload(result: StageResult) -> dict[str, Any]:
    """Build the reproducibility manifest for one completed stage."""

    artifacts = {name: str(path) for name, path in sorted(result.artifacts.items())}
    artifacts.setdefault("artifacts_manifest.json", str(result.output_dir / "artifacts_manifest.json"))
    return {
        "stage_name": result.stage_name,
        "output_dir": str(result.output_dir),
        "best_checkpoint": str(result.best_checkpoint),
        "artifacts": artifacts,
        "metrics": {
            "validation": dict(result.validation_metrics),
            "holdout": dict(result.holdout_metrics),
            "summary": flatten_result_metrics(result),
        },
        "run_summary": dict(result.run_summary),
        "config": _config_payload(result.config),
    }


def write_artifact_manifest(result: StageResult, path: str | Path | None = None) -> Path:
    """Write ``artifacts_manifest.json`` for downstream app/demo consumers."""

    target = Path(path) if path is not None else result.output_dir / "artifacts_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_ready(artifact_manifest_payload(result)), indent=2, sort_keys=True) + "\n")
    return target


def failure_report_payload(stage_name: str, output_dir: str | Path, config: TrainingConfig, exc: Exception) -> dict[str, Any]:
    """Build a deterministic failure report for one failed stage."""

    return {
        "stage_name": stage_name,
        "output_dir": str(Path(output_dir)),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "config": _config_payload(config),
    }


def write_failure_report(stage_name: str, output_dir: str | Path, config: TrainingConfig, exc: Exception) -> Path:
    """Persist ``failed_stage.json`` before re-raising a training error."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "failed_stage.json"
    target.write_text(json.dumps(_json_ready(failure_report_payload(stage_name, root, config, exc)), indent=2, sort_keys=True) + "\n")
    return target


class Trainer:
    """OOP coordinator for the app-facing two-stage training workflow."""

    def __init__(
        self,
        base_config: TrainingConfig | None = None,
        *,
        run_root: str | Path | None = None,
        stage_runner: StageRunner | None = None,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        self.base_config = replace(base_config or _new_training_config())
        self.run_root = _resolve_run_root(self.base_config, run_root)
        self.stage_runner = stage_runner or default_stage_runner
        self.callbacks = tuple(callbacks)

    def plan_two_stage(self) -> TwoStagePlan:
        prepared_cache_dir = self._prepared_cache_dir()
        stage1 = make_classical_pretrain_spec(
            self.base_config,
            self.run_root / "stage1_classical",
            prepared_cache_dir=prepared_cache_dir,
        )
        stage2 = make_hybrid_finetune_spec(
            self.base_config,
            self.run_root / "stage2_hybrid",
            Path(stage1.config.output_dir) / "best_model.pt",
            prepared_cache_dir=prepared_cache_dir,
        )
        return TwoStagePlan(stage1=stage1, stage2=stage2)

    def run_stage(self, spec: StageSpec) -> StageResult:
        output_dir = Path(spec.config.output_dir)
        self._emit(TrainingEvent(spec.name, "started", output_dir, spec.description))
        try:
            self.stage_runner(spec.config)
            result = build_stage_result(spec.name, output_dir, spec.config)
            write_artifact_manifest(result)
            result = build_stage_result(spec.name, output_dir, spec.config)
        except Exception as exc:
            write_failure_report(spec.name, output_dir, spec.config, exc)
            self._emit(TrainingEvent(spec.name, "failed", output_dir, str(exc)))
            raise
        self._emit(
            TrainingEvent(
                spec.name,
                "completed",
                output_dir,
                format_metric_summary(result),
                flatten_result_metrics(result),
            )
        )
        return result

    def run_two_stage(self) -> TwoStageResult:
        plan = self.plan_two_stage()
        stage1 = self.run_stage(plan.stage1)
        stage2_spec = make_hybrid_finetune_spec(
            self.base_config,
            self.run_root / "stage2_hybrid",
            stage1.best_checkpoint,
            prepared_cache_dir=self._prepared_cache_dir(),
        )
        stage2 = self.run_stage(stage2_spec)
        return TwoStageResult(stage1=stage1, stage2=stage2)

    def _prepared_cache_dir(self) -> str | None:
        if self.base_config.prepared_cache_dir is not None:
            return self.base_config.prepared_cache_dir
        return str((self.run_root / "prepared_cache").resolve())

    def _emit(self, event: TrainingEvent) -> None:
        for callback in self.callbacks:
            callback(event)


def build_stage_result(stage_name: str, output_dir: str | Path, config: TrainingConfig) -> StageResult:
    root = Path(output_dir)
    validation_metrics, holdout_metrics = read_stage_metrics(root)
    return StageResult(
        stage_name=stage_name,
        output_dir=root,
        config=config,
        best_checkpoint=select_best_checkpoint(root),
        artifacts=collect_artifacts(root),
        validation_metrics=validation_metrics,
        holdout_metrics=holdout_metrics,
        run_summary=read_json_file(root / "run_summary.json"),
    )


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def _config_payload(config: TrainingConfig) -> Mapping[str, Any]:
    to_json_dict = getattr(config, "to_json_dict", None)
    if callable(to_json_dict):
        return to_json_dict()
    if is_dataclass(config):
        return asdict(config)
    return {key: value for key, value in vars(config).items() if not key.startswith("_")}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _resolve_run_root(config: TrainingConfig, run_root: str | Path | None) -> Path:
    if run_root is None:
        return config.resolved_output_dir().resolve()
    root = Path(run_root).expanduser()
    if root.is_absolute():
        return root.resolve()
    return (config.resolved_project_root() / root).resolve()


def _new_training_config() -> TrainingConfig:
    from models.ariel_quantum_regression.training import TrainingConfig

    return TrainingConfig()


__all__ = [
    "Callback",
    "StageRunner",
    "StageSpec",
    "StageResult",
    "TwoStagePlan",
    "TwoStageResult",
    "TrainingEvent",
    "Trainer",
    "artifact_manifest_payload",
    "build_stage_result",
    "clone_config",
    "collect_artifacts",
    "default_stage_runner",
    "failure_report_payload",
    "flatten_result_metrics",
    "format_metric_summary",
    "has_nonfinite_metric",
    "make_classical_pretrain_spec",
    "make_fail_on_nan_callback",
    "make_hybrid_finetune_spec",
    "make_jsonl_logger",
    "plan_to_json_dict",
    "read_json_file",
    "read_stage_metrics",
    "select_best_checkpoint",
    "stage_spec_to_json_dict",
    "write_artifact_manifest",
    "write_failure_report",
]
