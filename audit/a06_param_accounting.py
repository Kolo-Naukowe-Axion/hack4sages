"""A06 — How many of the "quantum branch" parameters are actually quantum?

Proves/disproves finding K6. `final = classical_pred + scale * tanh(gate) * quantum_head([head_context, q])`
where quantum_head is a 264->192->5 MLP over the FULL classical context plus 8 circuit outputs.
So the "quantum correction" is a second classical head to which 8 of 264 input dims come from the
circuit, and the circuit itself has 3*n_qubits*(depth//2) parameters.

PASS criterion: an on/off ablation of the quantum branch is parameter-matched, i.e. the branch adds
< 1% extra classical parameters relative to the classical-only model. (It adds ~37%.)

Also computes the effective multiplier |tanh(gate)| * scale and the share of the quantum features
in the head's input dimensionality — the quantity that bounds any achievable circuit effect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a06_param_accounting",
    finding="K6 — 24 of 69,434 'quantum branch' parameters are quantum; the branch is a second classical head",
    question="What fraction of the quantum pathway is quantum, and is the on/off ablation parameter-matched?",
    criterion="quantum branch adds <1% extra classical parameters over the classical-only model",
)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, ck = A.load_exobiome()
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
    total = sum(groups.values())
    qpath_keys = ["projector", "quantum_block", "quantum_head", "quantum_gate"]
    qpath = sum(groups.get(k, 0) for k in qpath_keys)
    circuit = int(model.quantum_block.weights.numel())
    classical_only = total - qpath
    extra_classical = qpath - circuit

    head_in = model.quantum_head.net[0].in_features
    n_qubits = model.quantum_block.n_qubits
    gate = torch.tanh(model.quantum_gate).detach().numpy()

    payload = {
        "modules": groups, "total_params": total,
        "quantum_pathway_params": qpath, "circuit_params": circuit,
        "extra_classical_params_added_by_the_branch": extra_classical,
        "classical_only_params": classical_only,
        "circuit_share_of_model": circuit / total,
        "extra_classical_over_classical_only": extra_classical / classical_only,
        "quantum_head_input_dim": int(head_in),
        "quantum_features_share_of_head_input": n_qubits / head_in,
        "gate_tanh": gate.tolist(),
        "gate_mean_abs": float(np.abs(gate).mean()),
        "effective_multiplier_at_scale_0.5": (float(np.abs(gate).min() * 0.5), float(np.abs(gate).max() * 0.5)),
        "circuit_formula": "3 * n_qubits * (depth // 2)",
        "variational_layers": int(model.quantum_block.depth // 2),
        "interpretation": (
            "The branch adds ~37% extra classical capacity over the classical-only model, so an on/off "
            "ablation is not parameter-matched. The circuit contributes 8 of "
            f"{head_in} input dims to a {groups.get('quantum_head')}-parameter classical head, and the "
            "whole correction is multiplied by |tanh(gate)|*scale. The architecture cannot isolate a "
            "quantum effect by construction — a null result is uninformative and a positive result is "
            "unattributable."),
    }
    print(f"  total={total}  classical-only={classical_only}  quantum pathway={qpath}  circuit={circuit}")
    print(f"  circuit share of model            = {payload['circuit_share_of_model']:.6%}")
    print(f"  extra classical vs classical-only = {payload['extra_classical_over_classical_only']:.2%}")
    print(f"  quantum dims / head input dims    = {n_qubits}/{head_in} = {payload['quantum_features_share_of_head_input']:.2%}")
    print(f"  |tanh(gate)| per gas              = {np.abs(gate).round(4).tolist()}  (mean {payload['gate_mean_abs']:.4f})")
    status = "PASS" if payload["extra_classical_over_classical_only"] < 0.01 else "FAIL"
    CHECK.emit(status, payload, inputs=[A.EXOBIOME_ARTIFACT / "best_model.pt"], out=args.out)


if __name__ == "__main__":
    main()
