"""Custom FMPE training loop for ADC2023 five-gas retrieval."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from .runtime import (
    append_jsonl,
    build_eval_loader,
    build_train_loader,
    close_dataloader,
    compute_grad_norm,
    configure_runtime,
    current_lr,
    evaluate_loss,
    resolve_device,
    set_seed,
    step_scheduler,
    write_history_row,
)

from .dataset import load_datasets
from .dingo_compat import build_posterior_model, build_resume_payload
from .evaluate import evaluate_split, resolve_evaluation_device, run_regression_evaluation


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
    best_monitor_rmse_mean: float = float("inf")
    best_monitor_epoch: int = 0
    best_monitor_estimator: Optional[str] = None


def save_atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def save_resume_checkpoint(model: Any, state: TrainerState, run_dir: Path, settings: dict[str, Any]) -> Path:
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
            "best_monitor_rmse_mean": state.best_monitor_rmse_mean,
            "best_monitor_epoch": state.best_monitor_epoch,
            "best_monitor_estimator": state.best_monitor_estimator,
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
        best_monitor_rmse_mean=float(payload.get("best_monitor_rmse_mean", float("inf"))),
        best_monitor_epoch=int(payload.get("best_monitor_epoch", 0)),
        best_monitor_estimator=payload.get("best_monitor_estimator"),
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
        "project": logging_cfg.get("project", "hack4sages-adc2023-fmpe"),
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
    context_batch_size = int(evaluation_cfg.get("periodic_context_batch_size", evaluation_cfg.get("context_batch_size", 16)))
    max_rows = evaluation_cfg.get("periodic_rmse_max_rows")
    row_selection_seed = int(evaluation_cfg.get("periodic_rmse_seed", settings.get("seed", 42)))
    progress_every_batches = int(evaluation_cfg.get("periodic_progress_every_batches", 0) or 0)

    original_device = training_device
    moved = eval_device != original_device
    if moved:
        from .dingo_compat import move_model_to_device

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
            progress_label=f"epoch {epoch} RMSE monitor",
            progress_every_batches=progress_every_batches,
        )
    finally:
        if moved:
            from .dingo_compat import move_model_to_device

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
        "Epoch {epoch} RMSE | mean={mean_rmse:.6f} median={median_rmse:.6f} "
        "rows={rows} samples={samples} device={device}".format(
            epoch=record["epoch"],
            mean_rmse=record["mean_rmse_mean"],
            median_rmse=record["median_rmse_mean"],
            rows=record["num_rows"],
            samples=record["posterior_samples"],
            device=record["device"],
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


def train_model(
    settings: dict[str, Any],
    run_dir: str | Path,
    prepared_data_override: Optional[str | Path] = None,
    resume_mode: str = "auto",
) -> dict[str, Any]:
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
    best_model_by_loss_path = run_path / "best_model_by_loss.pt"
    best_model_by_mrmse_path = run_path / "best_model_by_mrmse.pt"
    best_monitor_path = run_path / "best_monitor_snapshot.json"

    max_epochs = int(train_loader_settings["epochs"])
    checkpoint_every_batches = int(train_loader_settings.get("checkpoint_every_batches", 10))
    max_steps = train_loader_settings.get("max_steps")
    patience = int(train_loader_settings["patience"])
    status = "continue"

    if resumed and state.batch_in_epoch == 0:
        if state.epoch > max_epochs:
            status = "completed"
        elif state.patience_bad_epochs >= patience:
            status = "stopped_early"

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
                theta, context = batch
                theta = theta.to(device, non_blocking=device.type == "cuda")
                context = context.to(device, non_blocking=device.type == "cuda")

                model.optimizer.zero_grad(set_to_none=True)
                loss = model.loss(theta, context)
                loss.backward()
                grad_norm = compute_grad_norm(model.network.parameters())
                model.optimizer.step()

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
                    "grad_norm": grad_norm,
                }
                append_jsonl(batch_metrics_path, record)
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
        print(
            "Epoch {epoch} complete | train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.6e} "
            "best_val_loss={best_val:.6f} best_rmse={best_rmse}".format(
                epoch=state.epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
                best_val=state.best_val_loss,
                best_rmse=(
                    f"{state.best_monitor_rmse_mean:.6f}"
                    if np.isfinite(state.best_monitor_rmse_mean)
                    else "n/a"
                ),
            ),
            flush=True,
        )

        if wandb_module is not None:
            wandb_module.log(
                {
                    "epoch": state.epoch,
                    "epoch_train_loss": float(train_loss),
                    "epoch_val_loss": float(val_loss),
                    "epoch_lr": lr,
                    "global_step": state.global_step,
                },
                step=state.global_step,
            )

        rmse_record = None
        try:
            rmse_record = maybe_run_periodic_rmse(
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

        if rmse_record is not None:
            mean_rmse = float(rmse_record["mean_rmse_mean"])
            median_rmse = float(rmse_record["median_rmse_mean"])
            monitor_rmse = mean_rmse if mean_rmse <= median_rmse else median_rmse
            estimator = "mean" if mean_rmse <= median_rmse else "median"
            if monitor_rmse < state.best_monitor_rmse_mean:
                state.best_monitor_rmse_mean = monitor_rmse
                state.best_monitor_epoch = int(state.epoch)
                state.best_monitor_estimator = estimator
                model.save_model(str(best_model_by_mrmse_path), save_training_info=False)
                best_monitor_path.write_text(
                    json.dumps(
                        {
                            "epoch": state.best_monitor_epoch,
                            "rmse_mean": state.best_monitor_rmse_mean,
                            "selected_point_estimate": state.best_monitor_estimator,
                            "record": rmse_record,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )

        if val_loss < state.best_val_loss:
            state.best_val_loss = float(val_loss)
            state.best_epoch = int(state.epoch)
            state.patience_bad_epochs = 0
            model.save_model(str(best_model_path), save_training_info=False)
            model.save_model(str(best_model_by_loss_path), save_training_info=False)
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

    close_dataloader(val_loader)

    if not best_model_by_mrmse_path.exists() and best_model_by_loss_path.exists():
        shutil.copy2(best_model_by_loss_path, best_model_by_mrmse_path)
        best_monitor_path.write_text(
            json.dumps(
                {
                    "epoch": state.best_epoch,
                    "rmse_mean": None,
                    "selected_point_estimate": None,
                    "fallback_to_best_loss": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    evaluation_cfg = settings.get("evaluation", {})
    run_final_evaluation = evaluation_cfg.get("run_after_training")
    if run_final_evaluation is None:
        run_final_evaluation = device.type != "mps"
    evaluation_outputs = (
        run_regression_evaluation(settings=settings, run_dir=run_path, prepared_data_override=prepared_data_override)
        if (status in {"completed", "stopped_early"} and bool(run_final_evaluation))
        else {}
    )

    summary = {
        "status": status,
        "run_dir": str(run_path),
        "resume_checkpoint": str(run_path / "resume_latest.pt"),
        "best_model_path": str(best_model_path),
        "best_model_by_loss_path": str(best_model_by_loss_path),
        "best_model_by_mrmse_path": str(best_model_by_mrmse_path),
        "best_epoch": state.best_epoch,
        "best_val_loss": state.best_val_loss,
        "best_monitor_rmse_mean": state.best_monitor_rmse_mean,
        "best_monitor_epoch": state.best_monitor_epoch,
        "best_monitor_estimator": state.best_monitor_estimator,
        "global_step": state.global_step,
        "next_epoch": state.epoch,
        "run_final_evaluation": bool(run_final_evaluation),
        "evaluation_outputs": evaluation_outputs,
    }
    (run_path / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if wandb_module is not None:
        wandb_module.finish()
    return summary
