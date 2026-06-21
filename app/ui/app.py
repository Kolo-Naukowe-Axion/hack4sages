"""ExoBiome - Streamlit UI (Osoba 4).

Biosignature-retrieval demo: pick or upload an exoplanet transmission spectrum,
run live inference, and compare the predicted atmospheric composition against
ground truth and the full-quantum model. Run from the repo root:

    .venv-app/bin/streamlit run app/ui/app.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import streamlit as st

# Make the repo importable when launched via `streamlit run app/ui/app.py`.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Auto-wire the sibling data worktree (../hack4sages-data) when present, so the
# curated dropdown works with zero env setup. Bundled examples + upload work
# without any dataset.
_SIBLING_DATA = _ROOT.parent / "hack4sages-data" / "data" / "ariel-ml-dataset"
if "EXOBIOME_DATA" not in os.environ and (_SIBLING_DATA / "TrainingData").is_dir():
    os.environ["EXOBIOME_DATA"] = str(_SIBLING_DATA)

from app.data import loading  # noqa: E402
from app.data.types import DataError, PlanetRecord  # noqa: E402
from app.ui import components  # noqa: E402
from app.ui import inference_adapter as ia  # noqa: E402

st.set_page_config(page_title="ExoBiome", page_icon="🪐", layout="wide")

_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


# --- cached resources & data -------------------------------------------------


@st.cache_resource(show_spinner="Ładowanie checkpointu modelu...")
def get_engine() -> ia.InferenceEngine:
    return ia.load_engine()


@st.cache_data(show_spinner=False)
def get_curated() -> list:
    try:
        return loading.list_curated_planets()
    except DataError:
        return []


def example_files() -> list[Path]:
    return sorted(_EXAMPLES_DIR.glob("*.csv")) if _EXAMPLES_DIR.is_dir() else []


# --- input selection (sidebar) -----------------------------------------------


def _safe_parse(source) -> PlanetRecord | None:
    try:
        return loading.parse_upload(source)
    except DataError as exc:
        st.sidebar.error(f"Nieprawidłowy plik: {exc}")
        return None


def pick_record() -> PlanetRecord | None:
    curated = get_curated()
    examples = example_files()

    modes: list[str] = []
    if curated:
        modes.append("Kuratowana planeta")
    if examples:
        modes.append("Przykład")
    modes.append("Wczytaj plik CSV")

    with st.sidebar:
        st.header(":material/biotech: ExoBiome")
        st.caption("Retrieval składu atmosfery egzoplanety z widma transmisyjnego.")
        mode = st.radio("Źródło widma", modes, label_visibility="collapsed")

        if mode == "Kuratowana planeta":
            labels = {cp.label: cp.planet_id for cp in curated}
            choice = st.selectbox(
                "Planeta", list(labels), help="Realne planety ADC2023 o różnej chemii."
            )
            try:
                return loading.load_by_id(labels[choice])
            except DataError as exc:
                st.error(f"Nie udało się wczytać planety: {exc}")
                return None

        if mode == "Przykład":
            files = {path.stem: path for path in examples}
            choice = st.selectbox("Przykład", list(files))
            return _safe_parse(io.StringIO(files[choice].read_text()))

        st.caption("Jednowierszowy CSV: 8 cech aux + flux_0..51 + noise_0..51.")
        if examples:
            st.download_button(
                "Pobierz szablon CSV",
                examples[0].read_text(),
                file_name="exobiome_template.csv",
                icon=":material/download:",
                width="stretch",
            )
        upload = st.file_uploader("Plik CSV", type=["csv"])
        if upload is None:
            return None
        return _safe_parse(io.StringIO(upload.getvalue().decode("utf-8")))


# --- rendering ---------------------------------------------------------------


def render(record: PlanetRecord, engine: ia.InferenceEngine) -> None:
    truth_known = record.truth is not None
    st.subheader(f"Planeta `{record.planet_id}`")
    st.caption(
        "Ground truth z danych" if truth_known else "Brak ground truth w rekordzie (zostanie pobrany z referencji, jeśli dostępny)."
    )

    st.markdown("#### Widmo transmisyjne")
    st.altair_chart(components.spectrum_chart(record), width="stretch")

    classical = engine.predict(record)
    preds = [classical]
    quantum = ia.full_quantum_prediction(engine.checkpoint_dir, record.planet_id)
    if quantum is not None:
        preds.append(quantum)
    truth = record.truth or ia.reference_truth(engine.checkpoint_dir, record.planet_id)
    rows, agg = ia.compare(truth, preds)

    st.markdown("#### Przewidziany skład atmosfery vs prawda")
    st.altair_chart(components.comparison_chart(rows), width="stretch")

    if agg:
        st.markdown("##### Średni RMSE (5 gazów)")
        cols = st.columns(len(agg))
        for col, (model, metric) in zip(cols, agg.items()):
            col.metric(model, f"{metric['rmse_mean']:.3f}", help="mRMSE względem ground truth")

    with st.expander("Tabela porownania (log-VMR)", icon=":material/table:"):
        st.dataframe(components.comparison_dataframe(rows), width="stretch")

    if quantum is None:
        st.caption(
            ":material/info: Brak referencyjnej predykcji pełnego modelu kwantowego dla tej "
            "planety - pokazana jest sama głowica klasyczna vs prawda."
        )


def main() -> None:
    st.title("ExoBiome")
    st.caption(
        "Hybrydowy kwantowo-klasyczny model przewiduje log-VMR pięciu gazów "
        "(H2O, CO2, CO, CH4, NH3) z widma transmisyjnego Ariel."
    )

    try:
        engine = get_engine()
    except Exception as exc:  # noqa: BLE001 - surface any checkpoint/load failure to the UI
        st.error(f"Nie udało się załadować modelu z checkpointu: {exc}")
        st.stop()

    record = pick_record()
    if record is None:
        st.info(
            "Wybierz planetę, przykład albo wczytaj własny plik CSV w panelu po lewej.",
            icon=":material/arrow_back:",
        )
        return

    try:
        render(record, engine)
    except DataError as exc:
        st.error(f"Błąd danych: {exc}")


main()
