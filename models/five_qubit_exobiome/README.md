# Five-qubit ExoBiome

`five_qubit_exobiome` is a standalone 5-qubit copy of the Ariel hybrid quantum regressor. It keeps the same two-stage training flow as the 8-qubit package, but defaults to `qnn_qubits=5` and writes its own outputs under `outputs/five_qubit_exobiome*`.

The training metric is validation `mRMSE` (`rmse_mean` across the five gases), computed once per epoch and used for checkpoint selection.

## Local Commands

Set up the environment:

```bash
bash models/five_qubit_exobiome/setup_five_qubit_exobiome_env.sh
```

Run a single training job:

```bash
$HOME/.venvs/five-qubit-exobiome/bin/python -u models/five_qubit_exobiome/run_five_qubit_exobiome.py \
  --data-root data/ariel-ml-dataset \
  --output-dir outputs/five_qubit_exobiome_live
```

Run the standard two-stage workflow:

```bash
bash models/five_qubit_exobiome/run_five_qubit_exobiome_two_stage.sh
```

## Remote Commands From Your Mac

Sync the repo to the Ubuntu machine:

```bash
bash models/five_qubit_exobiome/sync_five_qubit_exobiome_remote.sh
```

One-time remote environment bootstrap:

```bash
ssh iwo@100.103.127.124 \
  'ENV_DIR=$HOME/.venvs/five-qubit-exobiome PYTHON_BIN=python3.11 bash /home/iwo/hack4sages-crossgen/models/five_qubit_exobiome/setup_five_qubit_exobiome_env.sh'
```

Launch the two-stage remote run in `tmux`:

```bash
bash models/five_qubit_exobiome/launch_five_qubit_exobiome_remote.sh
```

Preview the live remote log in your terminal:

```bash
bash models/five_qubit_exobiome/monitor_five_qubit_exobiome_remote.sh
```

Attach to the remote `tmux` session:

```bash
bash models/five_qubit_exobiome/monitor_five_qubit_exobiome_remote.sh attach
```

Remote outputs land under:

```text
/home/iwo/hack4sages-crossgen/outputs/five_qubit_exobiome_two_stage_YYYYMMDD_HHMMSS
```
