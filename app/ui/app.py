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
# planet list comes from the real ADC2023 dataset. Upload works without it.
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


# --- input: planet picker / upload (in the main area) ------------------------


def _safe_parse(source) -> PlanetRecord | None:
    try:
        return loading.parse_upload(source)
    except DataError as exc:
        st.error(f"Nieprawidłowy plik: {exc}", icon=":material/error:")
        return None


def _planet_options() -> tuple[str, dict]:
    """One unified planet list: real ADC2023 planets when the dataset is present,
    otherwise the bundled example planets. The user never sees the distinction."""
    curated = get_curated()
    if curated:
        return "curated", {cp.label: cp.planet_id for cp in curated}
    examples = example_files()
    if examples:
        return "example", {path.stem: path for path in examples}
    return "none", {}


def input_controls() -> PlanetRecord | None:
    kind, options = _planet_options()
    sources = (["Planeta z listy"] if options else []) + ["Wczytaj plik CSV"]
    source = (
        st.segmented_control(
            "Źródło widma", sources, default=sources[0], selection_mode="single", label_visibility="collapsed"
        )
        or sources[0]
    )

    if source == "Planeta z listy" and options:
        choice = st.selectbox(
            "Wybierz planetę",
            list(options),
            label_visibility="collapsed",
            help="Realne planety ADC2023 o różnej chemii - kliknij, aby wybrać.",
        )
        if kind == "curated":
            try:
                return loading.load_by_id(options[choice])
            except DataError as exc:
                st.error(f"Nie udało się wczytać: {exc}", icon=":material/error:")
                return None
        return _safe_parse(io.StringIO(options[choice].read_text()))

    cols = st.columns([3, 1], vertical_alignment="bottom")
    upload = cols[0].file_uploader("Plik CSV (1 wiersz: 8 cech aux + flux_0..51 + noise_0..51)", type=["csv"])
    examples = example_files()
    if examples:
        cols[1].download_button(
            "Szablon",
            examples[0].read_text(),
            file_name="exobiome_template.csv",
            icon=":material/download:",
            width="stretch",
        )
    if upload is not None:
        return _safe_parse(io.StringIO(upload.getvalue().decode("utf-8")))
    return None


# --- main content ------------------------------------------------------------


def intro() -> None:
    st.markdown(
        "Aplikacja przewiduje **skład atmosfery egzoplanety** z jej **widma transmisyjnego**. "
        "Gdy planeta przechodzi na tle gwiazdy, gazy pochłaniają światło w swoich pasmach - to "
        "chemiczny odcisk palca, który czyta model. Prawdę znamy, bo widma są **symulowane** "
        "(Ariel Data Challenge 2023): sami ustawiamy skład, a model rozwiązuje zadanie odwrotne "
        "(widmo → skład). Porównujemy trzy modele - **baseline Random Forest**, naszą **głowicę "
        "klasyczną** i **pełny model kwantowy** - z prawdą."
    )


def kpi_row(record: PlanetRecord, truth: GroundTruth | None) -> None:
    cols = st.columns(2)
    cols[0].metric("Planeta", record.planet_id, border=True)
    cols[1].metric(
        "Dominujący gaz", dominant_gas(truth) or "-", border=True, help="Gaz o najwyższym udziale wg prawdy"
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
            st.markdown("**Δ do prawdy**  ·  mRMSE = średnia odległość predykcji do prawdy (mniej = bliżej)")
            order = [m for m in (ia.BASELINE_MODEL_NAME, ia.CLASSICAL_MODEL_NAME, ia.QUANTUM_MODEL_NAME) if m in agg]
            best = min(order, key=lambda m: agg[m]["rmse_mean"]) if order else None
            cols = st.columns(len(order) + 1)
            cols[0].metric("prawda (cel)", "0.000", border=True, help="punkt odniesienia - idealne dopasowanie")
            for col, model in zip(cols[1:], order):
                rmse = agg[model]["rmse_mean"]
                col.metric(
                    model,
                    f"{rmse:.3f}",
                    border=True,
                    help="najbliżej prawdy ze wszystkich modeli" if model == best else "odległość do prawdy (mRMSE)",
                )
            if best:
                st.caption(
                    f"Najbliżej prawdy: **{best}** ({agg[best]['rmse_mean']:.3f}). baseline (RF) = klasyczny "
                    "Random Forest · klasyczny = głowica klasyczna (na żywo) · kwantowy = pełny model kwantowy."
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

    _, center, _ = st.columns([1, 22, 1])
    with center:
        st.title("ExoBiome")
        intro()
        record = input_controls()
        if record is None:
            st.info(
                "Wybierz planetę z listy albo wczytaj własne widmo (CSV) powyżej.",
                icon=":material/arrow_upward:",
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

        kpi_row(record, truth)
        spectrum_section(record)
        results_section(rows, agg, quantum is not None)


main()
