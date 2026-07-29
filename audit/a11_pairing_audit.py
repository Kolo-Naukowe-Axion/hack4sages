"""A11 — Do two runs presented as an A/B comparison differ only in the factor under study?

Proves/disproves finding P3. reports/taurex_model_comparison.md compares a quantum snapshot against
a "noquant" snapshot as if the quantum branch were the only difference. They differ in best_epoch
(5 vs 59), classical_lr (20x), warm start vs scratch, batch size, patience, freeze/ramp schedule,
the reported quantum_scale, the taurex_ignore_poseidon flag, the residual-branch parameter count,
and — behind the same loss_name="mse" string — two different loss functions.

PASS criterion: the two configs differ in at most the declared factor (plus an allowlist).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a11_pairing_audit",
    finding="P3 — the quantum-vs-noquant comparison is unmatched on ~10 axes, including two different losses under one name",
    question="Which configuration fields differ between two runs presented as an ablation pair?",
    criterion="differences confined to the declared factor under study",
)

DEFAULT_A = "reports/ariel_quantum_taurex_snapshot_20260312_1003"
DEFAULT_B = "reports/taurex_noquant_taurex_snapshot_20260312_133054"
IGNORE_PREFIXES = ("project_root", "data_root", "output_dir", "prepared_cache_dir", "run_name")
DECLARED_FACTOR = {"quantum_device", "classical_only", "qnn_qubits", "qnn_depth", "qnn_init_scale"}


def loss_semantics() -> dict:
    """Same `loss_name='mse'` string, two different objectives across the compared packages."""
    out = {}
    for pkg, marker, meaning in (
        ("models/taurex_exobiome/training.py", "nn.MSELoss()",
         "MSE on z-scored targets -> implicit per-gas weight 1/sigma_g^2"),
        ("models/taurex_exobiome_without_quant/training.py", "OriginalScaleMSELoss",
         "MSE on ORIGINAL-scale targets -> equal per-gas weight; package also offers 'mrmse', "
         "which directly optimises the reported metric"),
    ):
        p = A.REPO / pkg
        out[pkg] = {"marker_present": (marker in p.read_text()) if p.exists() else None, "meaning": meaning}
    return out


QUANTUM_BRANCH_MODULES = ("projector", "quantum_block", "quantum_head", "quantum_gate")
CONTROL_BRANCH_MODULES = ("refinement_projector", "refinement_block", "refinement_head")

# Stale sprzed 2026-07-28, zostawione WYLACZNIE jako punkt odniesienia dla asercji regresji.
# Nie wolno ich raportowac jako pomiaru — patrz komentarz w residual_branch_params().
PREVIOUSLY_HARDCODED = {"quantum_arm_total": 69434, "classical_control_arm_total": 85733}


def _module_params(model) -> dict:
    """Suma parametrow po modulach najwyzszego poziomu, tak jak w a06:41-43."""
    g: dict[str, int] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        g[top] = g.get(top, 0) + int(p.numel())
    return g


def residual_branch_params(ca: dict, cb: dict) -> dict:
    """Liczba parametrow galezi rezydualnej w obu ramionach — WYLICZONA z named_parameters().

    Bylo: dwie stale WPISANE w kod (69 434 i 85 733), nigdy nieporownane z zadnym modelem.
    Pierwsza jest potwierdzona niezaleznie przez a06 (`quantum_pathway_params`), druga NICZYM —
    a to z niej brala sie teza "23 % wiecej pojemnosci" w polu `note`, czyli caly wniosek o
    niesparowaniu pojemnosci stal na liczbie bez zrodla. Liczba udajaca pomiar jest gorsza od
    braku liczby, bo nie da sie jej sfalsyfikowac przebiegiem.

    Teraz kazde ramie jest budowane z konfiguracji SWOJEGO snapshotu i sumowane po
    `named_parameters()`. Gdy ramienia nie da sie zbudowac, wchodzi wartosc historyczna
    z jawnym `source: "hardcoded, unverified"` i `error` — nigdy nie udajac pomiaru.

    `quantum_device` z configu (lightning.gpu) jest podmieniane na lightning.qubit: liczba
    parametrow nie zalezy od backendu symulatora, a lightning.gpu nie jest w tym srodowisku.
    """
    import torch
    sys.path.insert(0, str(A.REPO))
    out: dict = {}

    def fallback(key: str, err: Exception) -> dict:
        return {"total": PREVIOUSLY_HARDCODED[key], "source": "hardcoded, unverified",
                "error": f"{type(err).__name__}: {err}",
                "warning": "liczba NIE jest pomiarem tego przebiegu — nie cytowac jako zmierzona"}

    try:
        from models.taurex_exobiome.model import ModelConfig as QCfg, build_model as q_build
        qm = q_build(QCfg(spectral_input_channels=4, dropout=float(ca.get("dropout", 0.05)),
                          qnn_qubits=int(ca.get("qnn_qubits", 8)), qnn_depth=int(ca.get("qnn_depth", 2)),
                          qnn_init_scale=float(ca.get("qnn_init_scale", 0.1)),
                          quantum_device="lightning.qubit", quantum_use_async=False,
                          classical_only=False, use_amp=False), torch.device("cpu"))
        g = _module_params(qm)
        out["quantum_arm"] = {
            "modules": {k: g.get(k, 0) for k in QUANTUM_BRANCH_MODULES},
            "total": sum(g.get(k, 0) for k in QUANTUM_BRANCH_MODULES),
            "circuit_params": int(qm.quantum_block.weights.numel()),
            "gated": True, "gate_init": "zeros",
            "source": "measured from models.taurex_exobiome.build_model(named_parameters)"}
    except Exception as exc:  # pragma: no cover
        out["quantum_arm"] = fallback("quantum_arm_total", exc)

    try:
        from models.taurex_exobiome_without_quant.model import (ModelConfig as NCfg,
                                                                build_model as n_build)
        nkw = dict(spectral_input_channels=4, dropout=float(cb.get("dropout", 0.05)),
                   refinement_width=int(cb.get("refinement_width", 32)),
                   refinement_layers=int(cb.get("refinement_layers", 2)),
                   classical_only=False, use_amp=False)
        if cb.get("architecture"):
            nkw["architecture"] = str(cb["architecture"])
        nm = n_build(NCfg(**nkw), torch.device("cpu"))
        g = _module_params(nm)
        out["classical_control_arm"] = {
            "modules": {k: g.get(k, 0) for k in CONTROL_BRANCH_MODULES},
            "total": sum(g.get(k, 0) for k in CONTROL_BRANCH_MODULES),
            "gated": False, "gate_init": None,
            "source": "measured from models.taurex_exobiome_without_quant.build_model(named_parameters)"}
    except Exception as exc:  # pragma: no cover
        out["classical_control_arm"] = fallback("classical_control_arm_total", exc)

    qt = out["quantum_arm"]["total"]
    ct = out["classical_control_arm"]["total"]
    out["control_over_quantum_capacity"] = (ct / qt) if qt else None
    out["both_arms_measured"] = all(
        v.get("source", "").startswith("measured") for k, v in out.items() if isinstance(v, dict))
    out["matches_previously_hardcoded"] = {
        "quantum_arm": qt == PREVIOUSLY_HARDCODED["quantum_arm_total"],
        "classical_control_arm": ct == PREVIOUSLY_HARDCODED["classical_control_arm_total"]}
    out["note"] = (f"the 'control' carries {(ct / qt - 1) * 100:.1f}% MORE residual capacity than the "
                   "arm it controls for, and no gate" if qt else
                   "capacity ratio unavailable — an arm could not be built")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=DEFAULT_A)
    ap.add_argument("--b", default=DEFAULT_B)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pa, pb = A.REPO / args.a, A.REPO / args.b

    # Bylo: brak config.json -> {} -> zero roznic -> zero confoundow -> PASS, czyli "ramiona sa
    # sparowane" orzekane na podstawie NIEPRZECZYTANIA zadnego pliku. Udowodnione:
    #   --a reports/typo_does_not_exist --b reports/also_not_there  ->  "0 differing fields", [PASS].
    # Literowka w sciezce zamieniala dowod P3 w PASS. To ten sam wzorzec "brak danych = sukces",
    # ktory naprawiono w run_all.py:73-84 i a04:73-94. Brak wejscia musi byc bledem wejscia.
    load_errors = []
    for label, p in (("a", pa), ("b", pb)):
        cf = p / "config.json"
        if not cf.exists():
            load_errors.append(f"--{label}: brak {cf.relative_to(A.REPO) if str(cf).startswith(str(A.REPO)) else cf}"
                               f" (katalog {'istnieje' if p.exists() else 'NIE istnieje'})")
    if load_errors:
        payload = {"run_a": args.a, "run_b": args.b, "config_load_errors": load_errors,
                   "n_differing_fields": None, "differences": None, "confounding_differences": None,
                   "interpretation": ("Nie porownano niczego: brakuje co najmniej jednego config.json. "
                                      "Ten wynik NIE jest dowodem sparowania ramion.")}
        for e in load_errors:
            print(f"  BLAD WEJSCIA: {e}")
        CHECK.emit("FAIL", payload, out=args.out)
        return

    ca = json.loads((pa / "config.json").read_text())
    cb = json.loads((pb / "config.json").read_text())
    sa = json.loads((pa / "training_state.json").read_text()) if (pa / "training_state.json").exists() else {}
    sb = json.loads((pb / "training_state.json").read_text()) if (pb / "training_state.json").exists() else {}

    diffs = {}
    for k in sorted(set(ca) | set(cb)):
        if k.startswith(IGNORE_PREFIXES):
            continue
        if ca.get(k) != cb.get(k):
            diffs[k] = {"a": ca.get(k), "b": cb.get(k), "declared_factor": k in DECLARED_FACTOR}
    for k in ("best_epoch", "quantum_scale"):
        if sa.get(k) != sb.get(k):
            diffs[f"state.{k}"] = {"a": sa.get(k), "b": sb.get(k), "declared_factor": False}

    confounds = {k: v for k, v in diffs.items() if not v["declared_factor"]}

    # Bylo: loss_semantics() liczone i wyrzucane do payloadu, ale NIE wchodzace do statusu — a to
    # najmocniejszy punkt P3 (jeden string loss_name="mse", dwie rozne funkcje straty). Obecnosc
    # oba markerow jest roznica konfiguracji ukryta poza config.json, wiec nalezy do confoundow.
    losses = loss_semantics()
    markers = {k: v["marker_present"] for k, v in losses.items()}
    if all(markers.values()):
        confounds["loss_function_behind_loss_name_mse"] = {
            "a": "nn.MSELoss() na targetach z-scored (waga per-gaz 1/sigma_g^2)",
            "b": "OriginalScaleMSELoss na targetach w skali oryginalnej (waga per-gaz rowna)",
            "declared_factor": False,
            "evidence": "oba markery obecne w kodzie obu pakietow (patrz loss_semantics)"}
    elif any(v is None for v in markers.values()):
        confounds["loss_function_undecidable"] = {
            "a": None, "b": None, "declared_factor": False,
            "evidence": f"nie da sie odczytac markerow straty: {markers}"}

    residual = residual_branch_params(ca, cb)
    payload = {"run_a": args.a, "run_b": args.b, "n_differing_fields": len(diffs),
               "differences": diffs, "confounding_differences": confounds,
               "n_confounding_differences": len(confounds),
               "loss_semantics": losses, "residual_branch_params": residual,
               "interpretation": ("Every confounding difference is an alternative explanation for the "
                                 "measured delta. With this many, the delta is not attributable to the "
                                 "quantum branch at all.")}
    print(f"  {len(diffs)} differing config/state fields, {len(confounds)} confounds in total:")
    for k, v in confounds.items():
        print(f"     {k:34} A={v['a']!r:28} B={v['b']!r}")
    print(f"  residual branch: quantum {residual['quantum_arm']['total']} vs control "
          f"{residual['classical_control_arm']['total']} params "
          f"({'zmierzone' if residual['both_arms_measured'] else 'NIE oba zmierzone'})")
    CHECK.emit("FAIL" if confounds else "PASS", payload,
               inputs=[pa / "config.json", pb / "config.json"], out=args.out)


if __name__ == "__main__":
    main()
