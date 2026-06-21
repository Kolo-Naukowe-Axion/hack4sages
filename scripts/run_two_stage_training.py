#!/usr/bin/env python3
"""Run or preview the Osoba 03 two-stage training workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.training import (  # noqa: E402
    Trainer,
    make_fail_on_nan_callback,
    make_jsonl_logger,
    plan_to_json_dict,
)


@dataclass
class PreviewTrainingConfig:
    """Lightweight config used only when ``--dry-run`` runs without ML deps."""

    project_root: str = str(PROJECT_ROOT)
    data_root: str = "data/ariel-ml-dataset"
    output_dir: str = "outputs/osoba03_two_stage"
    prepared_cache_dir: str | None = None
    init_checkpoint_path: str | None = None
    classical_only: bool = False
    quantum_warmup_epochs: int = 5
    max_epochs: int = 30
    batch_size: int = 64
    eval_batch_size: int = 128
    train_limit: int | None = None
    val_limit: int | None = None
    holdout_limit: int | None = None
    test_limit: int | None = None
    use_amp: bool = True

    def resolved_project_root(self) -> Path:
        return Path(self.project_root).expanduser().resolve()

    def resolved_output_dir(self) -> Path:
        root = Path(self.output_dir).expanduser()
        if root.is_absolute():
            return root
        return self.resolved_project_root() / root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Repository root used to resolve relative paths.")
    parser.add_argument("--data-root", default="data/ariel-ml-dataset", help="Dataset root relative to project root or absolute.")
    parser.add_argument("--run-root", default="outputs/osoba03_two_stage", help="Root directory for stage outputs.")
    parser.add_argument("--events-jsonl", default=None, help="Optional JSONL event log path. Defaults to <run-root>/events.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Print the two-stage plan without running training.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override TrainingConfig.max_epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override TrainingConfig.batch_size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override TrainingConfig.eval_batch_size.")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit training records for smoke runs.")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit validation records for smoke runs.")
    parser.add_argument("--holdout-limit", type=int, default=None, help="Limit holdout records for smoke runs.")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test records for smoke runs.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Any:
    try:
        from models.ariel_quantum_regression.training import TrainingConfig
    except ModuleNotFoundError as exc:
        if not args.dry_run:
            raise RuntimeError("Full training needs the model environment, including torch and optional dataset dependencies.") from exc
        TrainingConfig = PreviewTrainingConfig

    cfg = TrainingConfig(project_root=args.project_root, data_root=args.data_root)
    updates: dict[str, Any] = {}
    for attr in ("max_epochs", "batch_size", "eval_batch_size", "train_limit", "val_limit", "holdout_limit", "test_limit"):
        value = getattr(args, attr)
        if value is not None:
            updates[attr] = value
    if args.no_amp:
        updates["use_amp"] = False
    return replace(cfg, **updates) if updates else cfg


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    run_root = Path(args.run_root).expanduser()
    events_path = Path(args.events_jsonl).expanduser() if args.events_jsonl else run_root / "events.jsonl"
    trainer = Trainer(
        cfg,
        run_root=run_root,
        callbacks=(make_jsonl_logger(events_path), make_fail_on_nan_callback()),
    )

    if args.dry_run:
        print(json.dumps(plan_to_json_dict(trainer.plan_two_stage()), indent=2, sort_keys=True))
        return 0

    result = trainer.run_two_stage()
    print(
        json.dumps(
            {
                "best_checkpoint": str(result.best_checkpoint),
                "stage1_manifest": str(result.stage1.output_dir / "artifacts_manifest.json"),
                "stage2_manifest": str(result.stage2.output_dir / "artifacts_manifest.json"),
                "events_jsonl": str(events_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
