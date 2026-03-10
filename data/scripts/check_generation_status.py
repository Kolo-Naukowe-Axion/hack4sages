#!/usr/bin/env python3
"""Display progress for a local pRT ADC2023 generation run."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--log-lines", type=int, default=12)
    return parser.parse_args()


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    total = int(max(seconds, 0.0))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_timestamp(timestamp: Optional[str]) -> str:
    if not timestamp:
        return "unknown"
    try:
        value = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _pid_status(pid: Optional[int]) -> str:
    if pid is None:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "not running"
    except PermissionError:
        return "running"
    return "running"


def _tail_lines(path: Path, line_count: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-line_count:]


def _fallback_status(output_root: Path) -> Dict[str, Any]:
    shard_root = output_root / "work" / "shards"
    shard_count = len(sorted(shard_root.glob("shard_*.npz")))
    return {
        "stage": "unknown",
        "generated_samples": None,
        "persisted_samples": None,
        "sample_count": None,
        "shards_completed": shard_count,
        "shards_total": None,
        "message": "no progress.json found yet",
        "last_update_utc": None,
        "elapsed_seconds": None,
        "estimated_remaining_seconds": None,
        "generation_rate_samples_per_minute": None,
        "log_path": None,
        "pid": None,
    }


def _load_status(output_root: Path) -> Dict[str, Any]:
    progress_path = output_root / "work" / "progress.json"
    if progress_path.exists():
        return json.loads(progress_path.read_text())
    return _fallback_status(output_root)


def _render(status: Dict[str, Any], log_lines: int) -> str:
    sample_count = status.get("sample_count")
    generated = status.get("generated_samples")
    percent = None
    if sample_count and generated is not None:
        percent = 100.0 * generated / sample_count

    lines = [
        f"Output root: {status.get('output_root', 'unknown')}",
        f"Stage: {status.get('stage', 'unknown')}",
        f"Process: {_pid_status(status.get('pid'))} (pid={status.get('pid', 'unknown')})",
        f"Started: {_format_timestamp(status.get('run_started_utc'))}",
        f"Last update: {_format_timestamp(status.get('last_update_utc'))}",
        f"Elapsed: {_format_duration(status.get('elapsed_seconds'))}",
    ]
    if generated is not None and sample_count:
        lines.append(f"Samples: {generated:,}/{sample_count:,} ({percent:.2f}%)")
    else:
        lines.append("Samples: unknown")

    shards_completed = status.get("shards_completed")
    shards_total = status.get("shards_total")
    if shards_completed is not None and shards_total is not None:
        lines.append(f"Shards: {shards_completed}/{shards_total}")
    elif shards_completed is not None:
        lines.append(f"Shards completed: {shards_completed}")

    if status.get("current_shard_ordinal") is not None:
        current_generated = status.get("current_shard_generated") or 0
        current_total = None
        if status.get("current_shard_start") is not None and status.get("current_shard_stop") is not None:
            current_total = status["current_shard_stop"] - status["current_shard_start"]
        if current_total is not None:
            lines.append(
                f"Current shard: {status['current_shard_ordinal']}/{shards_total} "
                f"({current_generated}/{current_total} samples)"
            )
        else:
            lines.append(f"Current shard ordinal: {status['current_shard_ordinal']}")

    rate = status.get("generation_rate_samples_per_minute")
    if rate is not None:
        lines.append(f"Rate: {rate:.1f} samples/min")
    eta = status.get("estimated_remaining_seconds")
    if eta is not None:
        lines.append(f"ETA: {_format_duration(eta)}")

    if status.get("message"):
        lines.append(f"Message: {status['message']}")
    if status.get("error"):
        lines.append(f"Error: {status['error']}")

    log_path = status.get("log_path")
    if log_path:
        log_path_obj = Path(log_path)
        lines.append(f"Log: {log_path_obj}")
        tail = _tail_lines(log_path_obj, log_lines)
        if tail:
            lines.append("")
            lines.append(f"Last {len(tail)} log lines:")
            lines.extend(tail)

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    while True:
        status = _load_status(args.output_root)
        if args.watch:
            sys.stdout.write("\033[2J\033[H")
        print(_render(status, args.log_lines), flush=True)
        if not args.watch:
            return
        if status.get("stage") in {"completed", "failed"} and _pid_status(status.get("pid")) != "running":
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
