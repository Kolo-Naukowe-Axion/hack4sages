$ErrorActionPreference = "Stop"

$WorkflowDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $WorkflowDir "..\.."))
$EnvDir = if ($env:ENV_DIR) { $env:ENV_DIR } else { Join-Path $HOME ".venvs\five-qubit-codex" }
$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "py" }
$PythonArgs = if ($env:PYTHON_ARGS) { $env:PYTHON_ARGS.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries) } else { @("-3.11") }

& $PythonExe @PythonArgs -m venv $EnvDir
$VenvPython = Join-Path $EnvDir "Scripts\python.exe"

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install `
  "numpy>=1.26" `
  "pandas>=2.2" `
  "h5py>=3.10" `
  "pyarrow>=17" `
  "scikit-learn>=1.5" `
  "torch>=2.3" `
  "pennylane>=0.44,<0.45" `
  "pennylane-lightning>=0.44,<0.45"

& $VenvPython -c @"
import importlib

for name in ("torch", "h5py", "pandas", "pennylane"):
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'unknown')}")

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
except Exception as exc:
    print("Torch backend probe failed:", exc)
"@

Write-Host ""
Write-Host "Environment ready."
Write-Host "Hybrid training:"
Write-Host "$VenvPython -u $RepoRoot\models\five_qubit_codex\run_five_qubit_codex.py --data-root $RepoRoot\data\ariel-ml-dataset --output-dir $RepoRoot\outputs\five_qubit_codex_live --batch-size 64 --eval-batch-size 128 --max-epochs 30 --log-every-batches 20"
Write-Host ""
Write-Host "Cross-validation:"
Write-Host "$VenvPython -u $RepoRoot\models\five_qubit_codex\run_five_qubit_codex_cross_validation.py --data-root $RepoRoot\data\ariel-ml-dataset --output-dir $RepoRoot\outputs\five_qubit_codex_cv --num-folds 5 --val-fraction 0.1 --batch-size 64 --eval-batch-size 128 --max-epochs 30 --log-every-batches 20"
