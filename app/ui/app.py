"""ExoBiome - Streamlit UI (Osoba 4).

Biosignature-retrieval demo: pick or upload an exoplanet transmission spectrum,
run live inference, and compare the predicted atmospheric composition against a
Random Forest baseline, the full quantum model, and ground truth. Run from the
repo root:

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
# curated dropdown works with zero env setup. Examples + upload work without it.
_SIBLING_DATA = _ROOT.parent / "hack4sages-data" / "data" / "ariel-ml-dataset"
if "EXOBIOME_DATA" not in os.environ and (_SIBLING_DATA / "TrainingData").is_dir():
    os.environ["EXOBIOME_DATA"] = str(_SIBLING_DATA)

from app.data import loading  # noqa: E402
from app.data.types import GASES, DataError, GroundTruth, PlanetRecord  # noqa: E402
from app.ui import components  # noqa: E402
from app.ui import inference_adapter as ia  # noqa: E402

st.set_page_config(page_title="ExoBiome", page_icon=":material/science:", layout="wide")

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


def dominant_gas(truth: GroundTruth | None) -> str | None:
    if truth is None:
        return None
    return components.gas_label(max(GASES, key=lambda gas: truth.log_vmr[gas]))


# --- sidebar: source only ----------------------------------------------------


def _safe_parse(source) -> PlanetRecord | None:
    try:
        return loading.parse_upload(source)
    except DataError as exc:
        st.sidebar.error(f"Nieprawidłowy plik: {exc}", icon=":material/error:")
        return None


def pick_record() -> PlanetRecord | None:
    curated = get_curated()
    examples = example_files()

    modes: list[str] = []
    if curated:
        modes.append("Kuratowana")
    if examples:
        modes.append("Przykład")
    modes.append("Upload CSV")

    record: PlanetRecord | None = None
    with st.sidebar:
        st.markdown("### Dane wejściowe")
        mode = (
            st.segmented_control(
                "Źródło widma", modes, default=modes[0], selection_mode="single", width="stretch"
            )
            or modes[0]
        )

        if mode == "Kuratowana":
            labels = {cp.label: cp.planet_id for cp in curated}
            choice = st.selectbox("Planeta", list(labels), help="Realne planety ADC2023 o różnej chemii.")
            try:
                record = loading.load_by_id(labels[choice])
            except DataError as exc:
                st.error(f"Nie udało się wczytać: {exc}", icon=":material/error:")
        elif mode == "Przykład":
            files = {path.stem: path for path in examples}
            choice = st.selectbox("Przykład", list(files))
            record = _safe_parse(io.StringIO(files[choice].read_text()))
        else:
            st.caption("Jednowierszowy CSV: 8 cech aux + flux_0..51 + noise_0..51.")
            if examples:
                st.download_button(
                    "Pobierz szablon",
                    examples[0].read_text(),
                    file_name="exobiome_template.csv",
                    icon=":material/download:",
                    width="stretch",
                )
            upload = st.file_uploader("Plik CSV", type=["csv"])
            if upload is not None:
                record = _safe_parse(io.StringIO(upload.getvalue().decode("utf-8")))

        st.space("medium")
        st.caption("ExoBiome · Hack4SAGES 2026")
    return record


# --- main content ------------------------------------------------------------


def hero() -> None:
    st.title("ExoBiome")
    st.markdown(
        ":violet-badge[hybryda kwantowo-klasyczna] :blue-badge[Ariel ADC2023] :gray-badge[5 gazów]"
    )


def intro() -> None:
    st.markdown(
        "Aplikacja przewiduje **skład atmosfery egzoplanety** z jej **widma transmisyjnego**. "
        "Gdy planeta przechodzi na tle gwiazdy, gazy pochłaniają światło w swoich pasmach - to "
        "chemiczny odcisk palca, który czyta model. Prawdę znamy, bo widma są **symulowane** "
        "(Ariel Data Challenge 2023): sami ustawiamy skład, a model rozwiązuje zadanie odwrotne "
        "(widmo → skład). Porównujemy trzy modele - **baseline Random Forest**, naszą **głowicę "
        "klasyczną** i **pełny model kwantowy** - z prawdą."
    )


def kpi_row(record: PlanetRecord, truth: GroundTruth | None, agg: dict) -> None:
    best = min(agg.items(), key=lambda kv: kv[1]["rmse_mean"]) if agg else None
    cols = st.columns(4)
    cols[0].metric("Planeta", record.planet_id, border=True)
    cols[1].metric(
        "Dominujący gaz", dominant_gas(truth) or "-", border=True, help="Gaz o najwyższym udziale wg prawdy"
    )
    cols[2].metric("Ground truth", "tak" if truth is not None else "nie", border=True)
    cols[3].metric(
        "Najlepszy mRMSE", f"{best[1]['rmse_mean']:.3f}" if best else "-", border=True, help=best[0] if best else None
    )


def spectrum_section(record: PlanetRecord) -> None:
    with st.container(border=True):
        st.markdown("##### :material/show_chart: Widmo transmisyjne")
        st.caption(
            "Linia = głębokość tranzytu vs długość fali, pasmo = szum pomiaru. Przerywane "
            "znaczniki pokazują, gdzie pochłaniają gazy (np. zafalowanie pod „CO2” to ślad CO2). "
            "Te wzory to sygnał, który czyta model."
        )
        st.altair_chart(components.spectrum_chart(record), width="stretch")


def results_section(rows: list, agg: dict, quantum_present: bool) -> None:
    with st.container(border=True):
        st.markdown("##### :material/query_stats: Skład atmosfery - modele vs prawda")
        st.caption(
            "Wszystko jako słupki: prawda + 3 modele, pogrupowane po gazie. Wyższy słupek = "
            "więcej gazu (log-VMR). Im bliżej słupka „prawda”, tym lepiej."
        )
        st.altair_chart(components.comparison_chart(rows), width="stretch")

        if agg:
            st.markdown("**Średni błąd na planetę (mRMSE, mniej = lepiej)**")
            base_rmse = agg.get(ia.BASELINE_MODEL_NAME, {}).get("rmse_mean")
            cols = st.columns(len(agg))
            for col, (model, metric) in zip(cols, agg.items()):
                rmse = metric["rmse_mean"]
                show_delta = base_rmse is not None and model != ia.BASELINE_MODEL_NAME
                col.metric(
                    model,
                    f"{rmse:.3f}",
                    delta=f"{rmse - base_rmse:+.2f}" if show_delta else None,
                    delta_color="inverse",
                    border=True,
                    help="różnica vs baseline" if show_delta else "klasyczny Random Forest",
                )
            st.caption(
                "baseline (RF) = klasyczny Random Forest · klasyczny = głowica klasyczna naszej "
                "hybrydy (na żywo) · kwantowy = pełny model kwantowy (z checkpointu)"
            )

        with st.expander("Tabela porównania (log-VMR)", icon=":material/table_chart:"):
            st.dataframe(components.comparison_dataframe(rows), width="stretch")

        if not quantum_present:
            st.caption(
                ":material/info: Brak predykcji pełnego modelu kwantowego dla tej planety - "
                "pokazane: baseline (RF) + klasyczny vs prawda."
            )


def main() -> None:
    try:
        engine = get_engine()
    except Exception as exc:  # noqa: BLE001 - surface any checkpoint/load failure
        st.title("ExoBiome")
        st.error(f"Nie udało się załadować modelu z checkpointu: {exc}", icon=":material/error:")
        st.stop()

    record = pick_record()

    _, center, _ = st.columns([1, 22, 1])
    with center:
        hero()
        intro()
        if record is None:
            st.info(
                "Wybierz planetę, przykład albo wczytaj plik CSV w panelu po lewej.",
                icon=":material/arrow_back:",
            )
            return

        try:
            classical = engine.predict(record)
            quantum = ia.full_quantum_prediction(engine.checkpoint_dir, record.planet_id)
            baseline = ia.rf_prediction(record)
            preds = ([baseline] if baseline is not None else []) + [classical]
            preds += [quantum] if quantum is not None else []
            truth = record.truth or ia.reference_truth(engine.checkpoint_dir, record.planet_id)
            rows, agg = ia.compare(truth, preds)
        except DataError as exc:
            st.error(f"Błąd danych: {exc}", icon=":material/error:")
            return

        kpi_row(record, truth, agg)
        spectrum_section(record)
        results_section(rows, agg, quantum is not None)


main()
