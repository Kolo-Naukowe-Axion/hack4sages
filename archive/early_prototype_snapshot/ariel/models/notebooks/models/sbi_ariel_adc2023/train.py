"""CLI entrypoint for ADC2023 five-gas FMPE training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ADC2023 five-gas FMPE model.")
    parser.add_argument("--settings", required=True, help="Path to the YAML settings file.")
    parser.add_argument("--prepared-data", default=None, help="Optional prepared split directory override.")
    parser.add_argument("--run-dir", required=True, help="Directory where checkpoints, logs, and outputs will be written.")
    parser.add_argument("--resume", default="auto", help="Resume mode: auto, never, or an explicit checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional device override, for example cpu or cuda.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional debug limit for total optimizer steps.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging for this run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings_path = Path(args.settings).expanduser().resolve()
    settings = yaml.safe_load(settings_path.read_text())

    if args.prepared_data is not None:
        settings["dataset"]["path"] = str(Path(args.prepared_data).expanduser().resolve())
    if args.device is not None:
        settings["training"]["device"] = args.device
    if args.max_steps is not None:
        settings["training"]["max_steps"] = int(args.max_steps)
    if args.no_wandb:
        settings.setdefault("logging", {})
        settings["logging"]["use_wandb"] = False

    summary = train_model(
        settings=settings,
        run_dir=args.run_dir,
        prepared_data_override=args.prepared_data,
        resume_mode=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

