"""Training & dataset view (Osoba 4).

Surfaces Osoba 3's two-stage training - the Trainer's dry-run plan (Strategy +
Factory, immutable StageSpec), the per-epoch history, and the TrainingEvent
stream replayed through a Callback (Observer) - plus the model's real accuracy
over the whole validation set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.data.types import GASES
from app.training import Trainer, TrainingEvent, plan_to_json_dict
from app.ui import components

_REPO = Path(__file__).resolve().parents[2]
_EVENTS_PATH = _REPO / "examples" / "osoba03_training_events.jsonl"
_RF_META = Path(__file__).resolve().parent / "artifacts" / "rf_baseline_meta.json"


@st.cache_data(show_spinner=False)
def _plan() -> dict:
    from models.ariel_quantum_regression.training import TrainingConfig

    return plan_to_json_dict(Trainer(TrainingConfig()).plan_two_stage())


@st.cache_data(show_spinner=False)
def _history(checkpoint_dir: str) -> pd.DataFrame:
    path = Path(checkpoint_dir) / "history.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _events() -> list[dict]:
    if not _EVENTS_PATH.exists():
        return []
    return [json.loads(line) for line in _EVENTS_PATH.read_text().splitlines() if line.strip()]


@st.cache_data(show_spinner=False)
def _metrics(checkpoint_dir: str) -> dict:
    val = Path(checkpoint_dir) / "validation_metrics.json"
    quantum = json.loads(val.read_text()) if val.exists() else {}
    rf = json.loads(_RF_META.read_text()) if _RF_META.exists() else {}
    return {"quantum": quantum, "rf": rf}


@st.cache_data(show_spinner=False)
def _parity(checkpoint_dir: str, n: int = 400) -> pd.DataFrame:
    path = Path(checkpoint_dir) / "validation_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    table = table.sample(min(n, len(table)), random_state=0)
    rows = [
        {"gaz": components.gas_label(g), "prawda": float(t), "predykcja": float(p)}
        for g in GASES
        for t, p in zip(table[f"true_{g}"], table[f"pred_{g}"])
    ]
    return pd.DataFrame(rows)


def render(engine) -> None:
    checkpoint_dir = str(engine.checkpoint_dir)
    with st.container(border=True):
        st.markdown("##### :material/model_training: Jak model powstał i jak dobry jest na całym zbiorze")
        plan_tab, training_tab, dataset_tab = st.tabs(["Plan treningu", "Krzywe i zdarzenia", "Wyniki na zbiorze"])
        with plan_tab:
            _render_plan()
        with training_tab:
            _render_training(checkpoint_dir)
        with dataset_tab:
            _render_dataset(checkpoint_dir)


def _render_plan() -> None:
    st.caption(
        "Trener (Osoba 3) buduje plan dwuetapowy na żywo - wzorce Strategy + Factory i "
        "niemutowalne StageSpec, bez uruchamiania treningu."
    )
    st.graphviz_chart(
        'digraph{rankdir=LR;bgcolor="transparent";'
        'node[shape=box,style="rounded,filled",fillcolor="#EAF1F8",color="#0F52BA",fontname="Inter",fontsize=11];'
        'a[label="Etap 1\\nklasyczny pretrain\\n(kwant zamrożony)"];'
        'b[label="best_model.pt",shape=note,fillcolor="#FBFCFE"];'
        'c[label="Etap 2\\nhybryda finetune\\n(kwant: warmup → ramp)"];'
        "a->b->c;}"
    )
    stage1, stage2 = _plan()["stages"]
    cols = st.columns(2)
    cols[0].markdown(
        f"**{stage1['name']}** · `classical_only={stage1['config'].get('classical_only')}` · "
        f"`quantum_warmup_epochs={stage1['config'].get('quantum_warmup_epochs')}`"
    )
    cols[1].markdown(
        f"**{stage2['name']}** · `classical_only={stage2['config'].get('classical_only')}` · "
        "init z `best_model.pt` etapu 1"
    )


def _render_training(checkpoint_dir: str) -> None:
    history = _history(checkpoint_dir)
    if not history.empty:
        st.caption("Krzywe mRMSE po epokach; zacieniony obszar = etap hybrydowy (kwant aktywny).")
        st.altair_chart(components.training_curve_chart(history), width="stretch")
    st.caption(
        "Strumień zdarzeń, który Trener emituje do callbacków (wzorzec **Obserwator**); "
        "UI rejestruje własny callback i odtwarza zapisany log."
    )
    _replay_events()


def _replay_events() -> None:
    log: list[dict] = []

    def observer(event: TrainingEvent) -> None:  # implements Osoba 3's Callback contract (Observer)
        row = {"etap": event.stage_name, "status": event.status}
        row.update({key: round(float(value), 3) for key, value in event.metrics.items()})
        log.append(row)

    for raw in _events():
        observer(
            TrainingEvent(
                stage_name=raw["stage_name"],
                status=raw["status"],
                output_dir=raw["output_dir"],
                message=raw.get("message", ""),
                metrics=raw.get("metrics", {}),
            )
        )
    if log:
        st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)


def _render_dataset(checkpoint_dir: str) -> None:
    metrics = _metrics(checkpoint_dir)
    quantum_rmse = metrics["quantum"].get("rmse_mean")
    rf_rmse = metrics["rf"].get("mRMSE_holdout")
    st.caption("Prawdziwa skuteczność na całym zbiorze walidacyjnym (4142 planety), nie z jednej planety.")
    cols = st.columns(2)
    if quantum_rmse is not None:
        cols[0].metric("Model kwantowy · mRMSE", f"{quantum_rmse:.3f}", border=True, help="4142 planety")
    if rf_rmse is not None:
        cols[1].metric("Baseline RF · mRMSE", f"{rf_rmse:.3f}", border=True, help="held-out")
    if metrics["quantum"].get("rmse"):
        st.altair_chart(components.per_gas_rmse_chart(metrics["quantum"]["rmse"]), width="stretch")
    parity = _parity(checkpoint_dir)
    if not parity.empty:
        st.caption("Parity-plot: predykcja vs prawda (próbka planet); im bliżej przekątnej, tym lepiej.")
        st.altair_chart(components.parity_chart(parity), width="stretch")
