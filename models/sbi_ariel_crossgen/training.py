"""Custom FMPE training loop with explicit split control and resumable batches."""

from __future__ import annotations

import json
import math
import os
import random
import time
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from .dataset import load_datasets
from .dingo_compat import build_posterior_model, build_resume_payload, load_posterior_model, move_model_to_device


@dataclass
class TrainerState:
    epoch: int = 1
    global_step: int = 0
    batch_in_epoch: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    patience_bad_epochs: int = 0
    epoch_loss_sum: float = 0.0
    epoch_loss_count: int = 0
    wandb_run_id: Optional[str] = None


class IndexBatchSampler(Sampler[list[int]]):
    def __init__(self, indices: np.ndarray, batch_size: int, start_batch: int = 0) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.start_batch = int(start_batch)

    def __iter__(self) -> Iterable[list[int]]:
        start_index = self.start_batch * self.batch_size
        for offset in range(start_index, len(self.indices), self.batch_size):
            yield self.indices[offset : offset + self.batch_size].tolist()

    def __len__(self) -> int:
        remaining = max(0, len(self.indices) - self.start_batch * self.batch_size)
        return math.ceil(remaining / self.batch_size)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device: torch.device) -> None:
    if os.name != "nt":
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except (AttributeError, RuntimeError):
            pass
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
    return torch.device(device_name)


def build_loader_kwargs(
    training_cfg: dict[str, Any],
    *,
    persistent_workers: Optional[bool] = None,
) -> dict[str, Any]:
    num_workers = int(training_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        if persistent_workers is None:
            persistent_workers = bool(training_cfg.get("persistent_workers", False))
        kwargs["persistent_workers"] = bool(persistent_workers)
        if training_cfg.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(training_cfg["prefetch_factor"])
    return kwargs


def build_eval_loader(dataset, training_cfg: dict[str, Any]) -> DataLoader:
    batch_size = int(training_cfg.get("eval_batch_size", training_cfg["batch_size"]))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **build_loader_kwargs(training_cfg, persistent_workers=False),
    )


def build_train_loader(dataset, training_cfg: dict[str, Any], seed: int, epoch: int, start_batch: int) -> DataLoader:
    permutation = np.random.default_rng(seed + epoch).permutation(len(dataset)).astype(np.int64)
    batch_sampler = IndexBatchSampler(permutation, int(training_cfg["batch_size"]), start_batch=start_batch)
    return DataLoader(dataset, batch_sampler=batch_sampler, **build_loader_kwargs(training_cfg))


def close_dataloader(dataloader: Optional[DataLoader]) -> None:
    if dataloader is None:
        return
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
        dataloader._iterator = None
    gc.collect()


def maybe_run_periodic_rmse(
    model: Any,
    settings: dict[str, Any],
    dataset,
    training_device: torch.device,
    epoch: int,
    global_step: int,
    run_path: Path,
    wandb_module: Optional[Any],
) -> Optional[dict[str, Any]]:
    from .evaluate import evaluate_split, resolve_evaluation_device

    evaluation_cfg = settings.get("evaluation", {})
    explicit_epochs = evaluation_cfg.get("periodic_rmse_epochs")
    if explicit_epochs is not None:
        allowed_epochs = {int(value) for value in explicit_epochs}
        if epoch not in allowed_epochs:
            return None
    else:
        every_epochs = int(evaluation_cfg.get("periodic_rmse_every_epochs", 0) or 0)
        if every_epochs <= 0 or epoch % every_epochs != 0:
            return None

    eval_device = resolve_evaluation_device(settings)
    posterior_samples = int(evaluation_cfg.get("periodic_posterior_samples", evaluation_cfg["posterior_samples"]))
    context_batch_size = int(evaluation_cfg.get("periodic_context_batch_size", evaluation_cfg.get("context_batch_size", 8)))
    max_rows = evaluation_cfg.get("periodic_rmse_max_rows")
    row_selection_seed = int(evaluation_cfg.get("periodic_rmse_seed", settings.get("seed", 42)))

    original_device = training_device
    moved = eval_device != original_device
    if moved:
        move_model_to_device(model, eval_device)

    try:
        subset_rows = min(len(dataset), int(max_rows)) if max_rows not in (None, 0) else len(dataset)
        print(
            f"Epoch {epoch} RMSE monitor | device={eval_device} rows={subset_rows}/{len(dataset)} posterior_samples={posterior_samples}",
            flush=True,
        )
        metrics, _ = evaluate_split(
            model=model,
            dataset=dataset,
            device=eval_device,
            posterior_samples=posterior_samples,
            context_batch_size=context_batch_size,
            include_predictions=False,
            max_rows=max_rows,
            row_selection_seed=row_selection_seed,
        )
    finally:
        if moved:
            move_model_to_device(model, original_device)
        model.network.train()

    record = {
        "epoch": epoch,
        "global_step": global_step,
        "device": str(eval_device),
        "posterior_samples": posterior_samples,
        "num_rows": int(metrics["num_rows"]),
        "full_num_rows": int(metrics["full_num_rows"]),
        "mean_rmse_mean": float(metrics["mean"]["rmse_mean"]),
        "median_rmse_mean": float(metrics["median"]["rmse_mean"]),
        "mean_rmse": metrics["mean"]["rmse"],
        "median_rmse": metrics["median"]["rmse"],
    }
    append_jsonl(run_path / "validation_rmse.jsonl", record)
    print(
        "Epoch {epoch} RMSE | device={device} posterior_samples={samples} "
        "mean_rmse={mean_rmse:.6f} median_rmse={median_rmse:.6f}".format(
            epoch=record["epoch"],
            device=record["device"],
            samples=record["posterior_samples"],
            mean_rmse=record["mean_rmse_mean"],
            median_rmse=record["median_rmse_mean"],
        ),
        flush=True,
    )
    if wandb_module is not None:
        wandb_module.log(
            {
                "epoch": epoch,
                "global_step": global_step,
                "val_rmse_mean": record["mean_rmse_mean"],
                "val_rmse_median": record["median_rmse_mean"],
            },
            step=global_step,
        )
    return record


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_history_row(path: Path, epoch: int, train_loss: float, val_loss: float, lr: float) -> None:
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(f"{epoch}\t{train_loss}\t{val_loss}\t{lr}\n")


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norms.append(parameter.grad.detach().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def step_scheduler(scheduler: Optional[Any], validation_loss: float) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_loss)
        return
    scheduler.step()


def evaluate_loss(model: Any, dataloader: DataLoader, device: torch.device) -> float:
    total_loss = 0.0
    total_examples = 0
    model.network.eval()
    with torch.no_grad():
        for theta, context in dataloader:
            theta = theta.to(device, non_blocking=device.type == "cuda")
            context = context.to(device, non_blocking=device.type == "cuda")
            loss = model.loss(theta, context)
            batch_examples = len(theta)
            total_loss += float(loss.item()) * batch_examples
            total_examples += batch_examples
    model.network.train()
    return total_loss / max(total_examples, 1)


def save_atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def save_resume_checkpoint(
    model: Any,
    state: TrainerState,
    run_dir: Path,
    settings: dict[str, Any],
) -> Path:
    payload = build_resume_payload(
        model,
        {
            "global_step": state.global_step,
            "batch_in_epoch": state.batch_in_epoch,
            "best_val_loss": state.best_val_loss,
            "best_epoch": state.best_epoch,
            "patience_bad_epochs": state.patience_bad_epochs,
            "epoch_loss_sum": state.epoch_loss_sum,
            "epoch_loss_count": state.epoch_loss_count,
            "wandb_run_id": state.wandb_run_id,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "settings_snapshot": settings,
        },
    )
    if torch.cuda.is_available():
        payload["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    checkpoint_path = run_dir / "resume_latest.pt"
    save_atomic_torch(checkpoint_path, payload)
    return checkpoint_path


def load_resume_state(checkpoint_path: Path, device: torch.device) -> tuple[Any, TrainerState, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    settings = payload["settings_snapshot"]
    model, settings = build_posterior_model(settings, device)
    model.network.load_state_dict(payload["model_state_dict"])
    if payload.get("optimizer_state_dict") is not None:
        model.optimizer.load_state_dict(payload["optimizer_state_dict"])
    if payload.get("scheduler_state_dict") is not None and getattr(model, "scheduler", None) is not None:
        model.scheduler.load_state_dict(payload["scheduler_state_dict"])
    model.epoch = int(payload["epoch"])
    state = TrainerState(
        epoch=int(payload["epoch"]),
        global_step=int(payload.get("global_step", 0)),
        batch_in_epoch=int(payload.get("batch_in_epoch", 0)),
        best_val_loss=float(payload.get("best_val_loss", float("inf"))),
        best_epoch=int(payload.get("best_epoch", 0)),
        patience_bad_epochs=int(payload.get("patience_bad_epochs", 0)),
        epoch_loss_sum=float(payload.get("epoch_loss_sum", 0.0)),
        epoch_loss_count=int(payload.get("epoch_loss_count", 0)),
        wandb_run_id=payload.get("wandb_run_id"),
    )
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.random.set_rng_state(payload["torch_random_state"])
    if torch.cuda.is_available() and payload.get("cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_random_state_all"])
    return model, state, settings


def maybe_init_wandb(run_dir: Path, settings: dict[str, Any], state: TrainerState, resume: bool) -> Optional[Any]:
    logging_cfg = settings.get("logging", {})
    if not logging_cfg.get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing with local batch_metrics.jsonl only.", flush=True)
        return None

    run_id = state.wandb_run_id
    if run_id is None:
        run_id = wandb.util.generate_id()
        state.wandb_run_id = run_id

    init_kwargs = {
        "project": logging_cfg.get("project", "hack4sages-crossgen-sbi-ariel"),
        "dir": str(run_dir),
        "config": settings,
        "id": run_id,
        "resume": "must" if resume else "allow",
        "name": logging_cfg.get("name", run_dir.name),
    }
    if logging_cfg.get("entity"):
        init_kwargs["entity"] = logging_cfg["entity"]
    if logging_cfg.get("tags"):
        init_kwargs["tags"] = list(logging_cfg["tags"])

    wandb.init(**init_kwargs)
    return wandb


def apply_requested_runtime_overrides(settings: dict[str, Any], requested_settings: dict[str, Any]) -> dict[str, Any]:
    requested_training = requested_settings.get("training", {})
    settings.setdefault("training", {})
    for key in (
        "batch_size",
        "device",
        "max_steps",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "eval_batch_size",
        "checkpoint_every_batches",
        "epochs",
        "patience",
    ):
        if key in requested_training:
            settings["training"][key] = requested_training[key]
    settings["logging"] = requested_settings.get("logging", settings.get("logging", {}))
    settings["evaluation"] = requested_settings.get("evaluation", settings.get("evaluation", {}))
    settings.setdefault("dataset", {})
    settings["dataset"]["path"] = requested_settings["dataset"]["path"]
    return settings


def train_model(
    settings: dict[str, Any],
    run_dir: str | Path,
    prepared_data_override: Optional[str | Path] = None,
    resume_mode: str = "auto",
    include_poseidon_eval: bool = False,
) -> dict[str, Any]:
    from .evaluate import run_regression_evaluation

    requested_settings = json.loads(json.dumps(settings))
    run_path = Path(run_dir).expanduser().resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(settings, prepared_data_override=prepared_data_override)

    device = resolve_device(settings["training"]["device"])
    configure_runtime(device)

    settings_path = run_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    (run_path / "prepared_manifest.json").write_text(json.dumps(datasets["train"].manifest, indent=2, sort_keys=True) + "\n")

    resume_path = run_path / "resume_latest.pt"
    if resume_mode == "auto" and resume_path.exists():
        model, state, settings = load_resume_state(resume_path, device)
        settings = apply_requested_runtime_overrides(settings, requested_settings)
        datasets = load_datasets(settings, prepared_data_override=prepared_data_override)
        resumed = True
    elif resume_mode not in {"auto", "never"}:
        model, state, settings = load_resume_state(Path(resume_mode), device)
        settings = apply_requested_runtime_overrides(settings, requested_settings)
        datasets = load_datasets(settings, prepared_data_override=prepared_data_override)
        resumed = True
    else:
        set_seed(int(settings.get("seed", 42)))
        model, settings = build_posterior_model(settings, device)
        state = TrainerState()
        resumed = False

    wandb_module = maybe_init_wandb(run_path, settings, state, resume=resumed)

    train_loader_settings = settings["training"]
    val_loader = build_eval_loader(datasets["validation"], train_loader_settings)
    batch_metrics_path = run_path / "batch_metrics.jsonl"
    history_path = run_path / "history.txt"
    best_model_path = run_path / "best_model.pt"

    max_epochs = int(train_loader_settings["epochs"])
    checkpoint_every_batches = int(train_loader_settings.get("checkpoint_every_batches", 10))
    max_steps = train_loader_settings.get("max_steps")
    patience = int(train_loader_settings["patience"])
    status = "completed"
    evaluation_cfg = settings.get("evaluation", {})
    run_final_evaluation = evaluation_cfg.get("run_after_training")
    if run_final_evaluation is None:
        run_final_evaluation = device.type != "mps"
    run_final_evaluation = bool(run_final_evaluation)

    if resumed and state.batch_in_epoch == 0:
        if state.epoch > max_epochs:
            status = "completed"
        elif state.patience_bad_epochs >= patience:
            status = "stopped_early"
        else:
            status = "continue"
    else:
        status = "continue"

    while status == "continue" and state.epoch <= max_epochs:
        model.epoch = state.epoch
        train_loader = build_train_loader(
            datasets["train"],
            train_loader_settings,
            seed=int(settings.get("seed", 42)),
            epoch=state.epoch,
            start_batch=state.batch_in_epoch,
        )
        if state.batch_in_epoch == 0:
            state.epoch_loss_sum = 0.0
            state.epoch_loss_count = 0

        print(f"Start training epoch {state.epoch} on {device}.", flush=True)
        model.network.train()

        try:
            for batch_idx, batch in enumerate(train_loader, start=state.batch_in_epoch + 1):
                batch_start = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                theta, context = batch
                theta = theta.to(device, non_blocking=device.type == "cuda")
                context = context.to(device, non_blocking=device.type == "cuda")

                model.optimizer.zero_grad(set_to_none=True)
                loss = model.loss(theta, context)
                loss.backward()
                grad_norm = compute_grad_norm(model.network.parameters())
                model.optimizer.step()

                batch_seconds = time.perf_counter() - batch_start
                batch_examples = len(theta)
                state.global_step += 1
                state.batch_in_epoch = batch_idx
                state.epoch_loss_sum += float(loss.item()) * batch_examples
                state.epoch_loss_count += batch_examples

                record = {
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "batch_in_epoch": batch_idx,
                    "loss": float(loss.item()),
                    "lr": current_lr(model.optimizer),
                    "batch_seconds": batch_seconds,
                    "examples_per_second": batch_examples / max(batch_seconds, 1.0e-12),
                    "grad_norm": grad_norm,
                    "peak_cuda_memory_mb": (
                        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                append_jsonl(batch_metrics_path, record)
                print(
                    "Epoch {epoch} Batch {batch} Step {step} | loss={loss:.6f} lr={lr:.6e} "
                    "sec={seconds:.3f} ex/s={eps:.1f} grad={grad:.3f} peak_cuda_mb={peak:.1f}".format(
                        epoch=record["epoch"],
                        batch=record["batch_in_epoch"],
                        step=record["global_step"],
                        loss=record["loss"],
                        lr=record["lr"],
                        seconds=record["batch_seconds"],
                        eps=record["examples_per_second"],
                        grad=record["grad_norm"],
                        peak=record["peak_cuda_memory_mb"],
                    ),
                    flush=True,
                )
                if wandb_module is not None:
                    wandb_module.log(record, step=state.global_step)

                if checkpoint_every_batches > 0 and state.global_step % checkpoint_every_batches == 0:
                    save_resume_checkpoint(model, state, run_path, settings)

                if max_steps is not None and state.global_step >= int(max_steps):
                    save_resume_checkpoint(model, state, run_path, settings)
                    status = "stopped_max_steps"
                    break
        finally:
            close_dataloader(train_loader)

        if status == "stopped_max_steps":
            break

        train_loss = state.epoch_loss_sum / max(state.epoch_loss_count, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        step_scheduler(getattr(model, "scheduler", None), val_loss)
        lr = current_lr(model.optimizer)
        write_history_row(history_path, state.epoch, train_loss, val_loss, lr)

        epoch_record = {
            "epoch": state.epoch,
            "epoch_train_loss": float(train_loss),
            "epoch_val_loss": float(val_loss),
            "epoch_lr": lr,
            "global_step": state.global_step,
        }
        print(
            f"Epoch {state.epoch} complete | train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.6e}",
            flush=True,
        )
        if wandb_module is not None:
            wandb_module.log(epoch_record, step=state.global_step)

        try:
            maybe_run_periodic_rmse(
                model=model,
                settings=settings,
                dataset=datasets["validation"],
                training_device=device,
                epoch=state.epoch,
                global_step=state.global_step,
                run_path=run_path,
                wandb_module=wandb_module,
            )
        except Exception as exc:
            append_jsonl(
                run_path / "validation_rmse.jsonl",
                {
                    "epoch": state.epoch,
                    "global_step": state.global_step,
                    "status": "error",
                    "error": str(exc),
                },
            )
            print(f"Epoch {state.epoch} RMSE monitor skipped due to error: {exc}", flush=True)

        if val_loss < state.best_val_loss:
            state.best_val_loss = float(val_loss)
            state.best_epoch = int(state.epoch)
            state.patience_bad_epochs = 0
            model.save_model(str(best_model_path), save_training_info=False)
        else:
            state.patience_bad_epochs += 1

        state.epoch += 1
        state.batch_in_epoch = 0
        state.epoch_loss_sum = 0.0
        state.epoch_loss_count = 0
        model.epoch = state.epoch
        save_resume_checkpoint(model, state, run_path, settings)

        if state.patience_bad_epochs >= patience:
            status = "stopped_early"
            break

    if status == "continue":
        status = "completed"

    if (status == "completed" or status == "stopped_early") and run_final_evaluation:
        save_resume_checkpoint(model, state, run_path, settings)
        close_dataloader(val_loader)
        print(
            "Training finished with status={status}. Starting final evaluation on {device}. "
            "This can take a while because posterior sampling runs after training completes.".format(
                status=status,
                device=settings.get("evaluation", {}).get("device", device),
            ),
            flush=True,
        )
        evaluation_outputs = run_regression_evaluation(
            settings=settings,
            run_dir=run_path,
            prepared_data_override=prepared_data_override,
            include_poseidon_eval=include_poseidon_eval,
        )
    else:
        close_dataloader(val_loader)
        evaluation_outputs = {}
        if status == "completed" or status == "stopped_early":
            print(
                "Training finished with status={status}. Automatic final evaluation is disabled; "
                "run run_regression_evaluation(...) manually when you want metrics.".format(status=status),
                flush=True,
            )

    summary = {
        "status": status,
        "run_dir": str(run_path),
        "resume_checkpoint": str(run_path / "resume_latest.pt"),
        "best_model_path": str(best_model_path),
        "best_epoch": state.best_epoch,
        "best_val_loss": state.best_val_loss,
        "global_step": state.global_step,
        "next_epoch": state.epoch,
        "run_final_evaluation": run_final_evaluation,
        "evaluation_outputs": evaluation_outputs,
    }
    (run_path / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if wandb_module is not None:
        wandb_module.finish()
    return summary
