"""Reproducible re-evaluation helper (non-invasive).

Every re-eval writes a self-describing run under ``reports/reeval/<date>/<model>/``:

  - ``metrics.json``      the re-derived metrics (per-gas RMSE + mean + rows)
  - ``predictions.csv``   (only when the re-eval *produces* new predictions)
  - ``manifest.json``     full provenance so a reviewer can reproduce the number:
      git commit, python + package versions, checkpoint path + sha256,
      input files + sha256, data root, split manifest, seed, quantum_scale,
      method, and the reported-vs-rederived delta + status. The status is
      ``verified``/``discrepancy`` only against a fixed external tolerance; runs
      judged against a self-derived band report ``within-seed-band`` instead
      (see ``tolerance_kind`` in ``write_reeval_run``).

Design: import-light (stdlib only at load; pandas is used by callers, not here).
It never imports or modifies any model package -> safe for ported/SOTA models.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REEVAL_ROOT = REPO_ROOT / "reports" / "reeval"

GASES = ["log_H2O", "log_CO2", "log_CO", "log_CH4", "log_NH3"]

# Strict tolerance for deterministic recomputation.
# ADOPTED, NOT DERIVED: the five
# recompute-from-predictions cases land 1e-08 .. 6.9e-07 from their committed scalars, i.e. with a
# margin of >=70x, so no verdict in this directory turns on the exact value. Any threshold between
# ~1e-6 and ~1e-3 would classify every current run identically.
DEFAULT_TOLERANCE = 5e-5

DEFAULT_PACKAGES = (
    "numpy", "pandas", "torch", "pennylane", "pennylane-lightning",
    "zuko", "tensorflow", "scikit-learn", "h5py",
)


def sha256(path: str | Path | None) -> str | None:
    """Streaming sha256 of a file; None if the path is missing."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root: Path = REPO_ROOT) -> str:
    """Current commit sha; suffixed with '-dirty' if the tree has changes."""
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True,
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def env_versions(packages=DEFAULT_PACKAGES) -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            out[pkg] = "not-installed"
    return out


def rmse_from_predictions(df, gases=GASES) -> dict:
    """Per-gas + mean RMSE from a dataframe with true_<gas>/pred_<gas> columns.

    Raises on any non-finite cell. A NaN would otherwise propagate into ``rmse_mean``; the status
    algebra in ``write_reeval_run`` would then read ``abs(nan) <= tol`` as False and emit
    ``discrepancy`` (fail-safe, so the verdict would not lie), but ``json.dumps`` writes a bare
    ``NaN`` token, which is not valid JSON and breaks any strict reader of the manifest. Failing
    loudly at the source is cheaper than shipping an unparseable record.
    """
    import numpy as np

    cols = [f"{p}_{g}" for g in gases for p in ("pred", "true")]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"prediction frame is missing columns: {missing}")
    n_nonfinite = int(sum((~np.isfinite(df[c].to_numpy(dtype=float))).sum() for c in cols))
    if n_nonfinite:
        raise ValueError(
            f"{n_nonfinite} non-finite cell(s) in the prediction frame; refusing to emit a metric "
            f"that would serialise as a bare NaN token"
        )

    per = {
        g: float(np.sqrt(np.mean((df[f"pred_{g}"] - df[f"true_{g}"]) ** 2)))
        for g in gases
    }
    return {"rmse": per, "rmse_mean": float(np.mean(list(per.values()))), "rows": int(len(df))}


def write_reeval_run(
    model: str,
    metrics: dict,
    *,
    method: str,
    date: str | None = None,
    predictions=None,
    inputs: dict[str, str | Path] | None = None,
    reported: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    tolerance_kind: str = "fixed",
    checkpoint: str | Path | None = None,
    data_root: str | Path | None = None,
    split_manifest: str | Path | None = None,
    seed: int | None = None,
    quantum_scale=None,
    packages=DEFAULT_PACKAGES,
    notes: str | None = None,
    truth_check: dict | None = None,
    reeval_root: Path = REEVAL_ROOT,
) -> tuple[Path, str | None, float | None]:
    """Write metrics + manifest (+ optional predictions) for one re-eval run.

    Returns (run_dir, status, delta_vs_reported).
    ``predictions``: a pandas DataFrame of NEW predictions to persist. For a pure
    recompute from an existing CSV, leave it None and pass that CSV via ``inputs``
    so its sha256 is recorded without duplicating the file.

    ``tolerance_kind`` decides what the status *means*, and the two must not be
    conflated when the numbers are cited:

      ``"fixed"``      tolerance is an external constant (default 5e-5). A pass is
                       a genuine reproduction -> status ``verified``.
      ``"seed-band"``  tolerance was derived from the run's own spread (e.g. 2 sigma
                       over noise seeds), so a pass only says the reported value sits
                       inside a band this run defined -> status ``within-seed-band``.
                       Such a status is NOT a reproduction and must be labelled as
                       such wherever it is quoted.

    ``truth_check``: result of verifying the run's ``true_*`` column against an INDEPENDENT load of
    the labels. Recomputing RMSE from a prediction CSV only proves the scalar was derived correctly
    from that CSV -- it says nothing about whether the CSV's own truth column is the real ground
    truth. Callers that can load the labels separately should pass the comparison here so the
    manifest records which of the two questions was actually answered.
    """
    if tolerance_kind not in ("fixed", "seed-band"):
        raise ValueError(f"tolerance_kind must be 'fixed' or 'seed-band', got {tolerance_kind!r}")
    date = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    run_dir = Path(reeval_root) / date / model
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    pred_name = None
    if predictions is not None:
        pred_name = "predictions.csv"
        predictions.to_csv(run_dir / pred_name, index=False)

    input_provenance = None
    if inputs:
        input_provenance = {
            name: {"path": str(p), "sha256": sha256(p)} for name, p in inputs.items()
        }

    status: str | None = None
    delta: float | None = None
    if reported is not None and metrics.get("rmse_mean") is not None:
        delta = round(metrics["rmse_mean"] - float(reported), 8)
        within = abs(delta) <= tolerance
        if tolerance_kind == "fixed":
            status = "verified" if within else "discrepancy"
        else:
            status = "within-seed-band" if within else "outside-seed-band"

    manifest = {
        "model": model,
        "method": method,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": env_versions(packages),
        # `present` disambiguates the two readings of `sha256: null`, which are otherwise identical
        # in the record: "this method needs no checkpoint" (path is None -- e.g. a pure recompute
        # from a CSV) versus "a checkpoint was declared but the file is not there" (path is a
        # string, file absent -- the real case of noquant_snapshot_poseidon, whose weights a14
        # lists under missing_paths). Without it a typo'd path looks like a deliberate omission.
        "checkpoint": {"path": str(checkpoint) if checkpoint else None,
                       "sha256": sha256(checkpoint),
                       "declared": checkpoint is not None,
                       "present": bool(checkpoint is not None and Path(checkpoint).is_file())},
        "inputs": input_provenance,
        "truth_check": truth_check,
        "data_root": str(data_root) if data_root else None,
        "split_manifest": str(split_manifest) if split_manifest else None,
        "seed": seed,
        "quantum_scale": quantum_scale,
        "rmse_mean": metrics.get("rmse_mean"),
        "rows": metrics.get("rows"),
        "reported": float(reported) if reported is not None else None,
        "delta_vs_reported": delta,
        "status": status,
        "tolerance": tolerance,
        "tolerance_kind": tolerance_kind,
        "outputs": {"metrics": "metrics.json", "predictions": pred_name},
        "notes": notes,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir, status, delta
