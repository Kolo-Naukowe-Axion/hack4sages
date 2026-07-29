"""A07 — Did the model "learn to silence" the quantum branch, or did the gate never leave its init?

Proves/disproves finding K7. QUANTUM_ADVANTAGE_VERDICT.md reads gate ~= 0.046 as evidence that
"the model itself figured this out ... it downweighted the quantum contribution to near-zero".
But models/*/model.py initialises `quantum_gate = nn.Parameter(torch.zeros(5))`, so the gate GREW
from exactly 0 during the 6 usable epochs and was still growing when the run was stopped.

Those two readings imply opposite conclusions and cannot be distinguished from a single checkpoint.
The distinguishing evidence is the gate trajectory, which requires either per-epoch checkpoints or
a re-run with gate logging — this check reports what CAN be established from the artifacts and
states exactly what is missing.

PASS criterion: the checkpoint's gate is a converged quantity, i.e. |gate| moved away from its
initialisation and the run terminated by its own stopping rule (a05), and per-gas gates agree in sign.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a07_gate_dynamics",
    finding="K7 — 'the model learned to silence the quantum branch' is unsupported: the gate is zero-initialised and was still growing",
    question="Is gate~=0.046 a converged equilibrium or an artifact of zero-init plus a truncated schedule?",
    criterion="gate is converged (run terminated by its own rule) AND per-gas gates agree in sign",
)


def main() -> None:
    import pandas as pd
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(A.REPO))
    import inspect
    from models.ariel_exobiome import model as mm
    src = inspect.getsource(mm.HybridArielRegressor.__init__)
    zero_init = "torch.zeros(len(TARGET_COLUMNS)" in src

    model, ck = A.load_exobiome()
    raw = model.quantum_gate.detach().numpy()
    g = np.tanh(raw)
    cfg = json.loads((A.EXOBIOME_ARTIFACT / "config.json").read_text())
    hist = pd.read_csv(A.EXOBIOME_ARTIFACT / "history.csv")
    state = json.loads((A.EXOBIOME_ARTIFACT / "training_state.json").read_text())
    best_epoch = int(state["best_epoch"])
    epochs_run = int(hist["epoch"].max())
    steps_per_epoch = 33138 // int(cfg["batch_size"])

    signs = np.sign(g)
    sign_agreement = bool(len(set(signs.tolist())) == 1)
    converged = (epochs_run >= int(cfg["max_epochs"])) or \
                (epochs_run - best_epoch >= int(cfg["early_stop_patience"]))

    issues = []
    if zero_init:
        issues.append("quantum_gate is initialised to EXACTLY zeros -> |gate|=0.046 is a distance travelled "
                      "from zero, not a learned suppression")
    if not converged:
        issues.append(f"run ended at epoch {epochs_run}/{cfg['max_epochs']} without exhausting patience "
                      f"{cfg['early_stop_patience']} -> the gate was not at equilibrium")
    if not sign_agreement:
        neg = [A.TARGETS[i] for i in range(len(g)) if g[i] < 0]
        pos = [A.TARGETS[i] for i in range(len(g)) if g[i] >= 0]
        issues.append(f"per-gas gates disagree in sign (neg: {neg}; pos: {pos}) -> mean|tanh(gate)| is not "
                      "an adequate summary of the branch")
    issues.append(f"the gate trained for {best_epoch} epochs x ~{steps_per_epoch} steps at "
                  f"quantum_lr={cfg['quantum_lr']} with weight_decay={cfg['weight_decay']} while "
                  f"quantum_scale was simultaneously ramping 0->{hist['quantum_scale'].max():.3f}; the "
                  "gate gradient is proportional to that scale, so the two effects are confounded")

    payload = {
        "gate_raw": raw.tolist(), "gate_tanh": g.tolist(),
        "gate_tanh_per_gas": dict(zip(A.TARGETS, g.tolist())),
        "mean_abs_tanh_gate": float(np.abs(g).mean()),
        "zero_initialised": zero_init, "sign_agreement_across_gases": sign_agreement,
        "epochs_run": epochs_run, "best_epoch": best_epoch, "max_epochs": int(cfg["max_epochs"]),
        "converged_by_own_rule": bool(converged),
        "quantum_lr": cfg["quantum_lr"], "approx_optimizer_steps_before_selection": best_epoch * steps_per_epoch,
        "circuit_weight_abs_mean": float(model.quantum_block.weights.detach().abs().mean()),
        "circuit_init_scale": cfg["qnn_init_scale"],
        "issues": issues,
        "missing_evidence_to_settle_it": [
            "per-epoch gate values (log tanh(quantum_gate) every epoch, or save per-epoch checkpoints)",
            "a run to convergence at fixed quantum_scale=1.0 (no ramp), so gate growth is not confounded",
            ">=5 seeds, to see whether the gate lands in the same place",
        ],
    }
    print(f"  gate tanh per gas: {dict(zip(A.TARGETS, np.round(g, 4).tolist()))}")
    print(f"  mean|tanh(gate)| = {payload['mean_abs_tanh_gate']:.4f}   zero-initialised: {zero_init}   "
          f"sign agreement: {sign_agreement}   converged: {converged}")
    for i in issues:
        print(f"   - {i}")
    # Galaz PASS MUSI istniec, inaczej check jest niefalsyfikowalny w strone spelnienia kryterium:
    # `criterion=` definiuje warunek PASS (bramka zbiezna AND znaki zgodne), a poprzednia wersja
    # w najlepszym razie zwracala WARN, wiec kryterium nie moglo zostac spelnione zadnymi danymi.
    if converged and sign_agreement:
        _status = "PASS"
    elif (zero_init and not converged) or not sign_agreement:
        _status = "FAIL"
    else:
        _status = "WARN"
    CHECK.emit(_status, payload,
               inputs=[A.EXOBIOME_ARTIFACT / "best_model.pt"], out=args.out)


if __name__ == "__main__":
    main()
