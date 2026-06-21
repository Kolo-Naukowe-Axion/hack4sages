"""ExoBiome - Streamlit UI (Osoba 4).

Hybrid OOP + FP front-end: pure chart builders and spectrum perturbations (FP)
feed the object-oriented predictors and inference engine (OOP). Run from the
repo root:

    .venv-app/bin/streamlit run app/ui/app.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Use the sibling data worktree when present; examples + upload work without it.
_SIBLING_DATA = _ROOT.parent / "hack4sages-data" / "data" / "ariel-ml-dataset"
if "EXOBIOME_DATA" not in os.environ and (_SIBLING_DATA / "TrainingData").is_dir():
    os.environ["EXOBIOME_DATA"] = str(_SIBLING_DATA)

from app.data import loading  # noqa: E402
from app.data.types import GASES, DataError, GroundTruth, PlanetRecord, Prediction  # noqa: E402
from app.ui import components, whatif  # noqa: E402
from app.ui import inference_adapter as ia  # noqa: E402

st.set_page_config(page_title="ExoBiome", page_icon=":material/science:", layout="wide")

_ROOT_DIR = Path(__file__).resolve().parent
_PLANET = (_ROOT_DIR / "assets" / "planet.svg").read_text()
_EXAMPLES_DIR = _ROOT_DIR / "examples"


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


def _parse(source) -> PlanetRecord | None:
    try:
        return loading.parse_upload(source)
    except DataError as exc:
        st.error(f"Nieprawidłowy plik: {exc}", icon=":material/error:")
        return None


def _planet_options() -> tuple[str, dict]:
    curated = get_curated()
    if curated:
        return "curated", {cp.label: cp.planet_id for cp in curated}
    examples = example_files()
    if examples:
        return "example", {path.stem: path for path in examples}
    return "none", {}


def pick_record() -> PlanetRecord | None:
    kind, options = _planet_options()
    sources = (["Planeta z listy"] if options else []) + ["Wczytaj plik CSV"]
    source = (
        st.segmented_control("Źródło widma", sources, default=sources[0], label_visibility="collapsed")
        or sources[0]
    )
    if source == "Planeta z listy" and options:
        choice = st.selectbox(
            "Wybierz planetę", list(options), label_visibility="collapsed",
            help="Realne planety ADC2023 o różnej chemii.",
        )
        if kind == "curated":
            try:
                return loading.load_by_id(options[choice])
            except DataError as exc:
                st.error(f"Nie udało się wczytać: {exc}", icon=":material/error:")
                return None
        return _parse(io.StringIO(options[choice].read_text()))

    cols = st.columns([3, 1], vertical_alignment="bottom")
    upload = cols[0].file_uploader(
        "Wczytaj widmo (CSV)", type=["csv"], help="1 wiersz: 8 cech aux + flux_0..51 + noise_0..51"
    )
    examples = example_files()
    if examples:
        cols[1].download_button(
            "Szablon", examples[0].read_text(), file_name="exobiome_template.csv",
            icon=":material/download:", width="stretch",
        )
    return _parse(io.StringIO(upload.getvalue().decode("utf-8"))) if upload is not None else None


def header() -> None:
    cols = st.columns([1, 11], vertical_alignment="center", gap="small")
    cols[0].image(_PLANET, width=52)
    cols[1].title("ExoBiome")


def intro() -> None:
    st.markdown(
        "Aplikacja przewiduje **skład atmosfery egzoplanety** z jej **widma transmisyjnego**. "
        "Gdy planeta przechodzi na tle gwiazdy, gazy pochłaniają światło w swoich pasmach - to "
        "chemiczny odcisk palca, który czyta model. Prawdę znamy, bo widma są **symulowane** "
        "(Ariel Data Challenge 2023): sami ustawiamy skład, a model rozwiązuje zadanie odwrotne. "
        "Porównujemy trzy modele - **baseline Random Forest**, naszą **głowicę klasyczną** i "
        "**pełny model kwantowy** - z prawdą."
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
            "Linia = głębokość tranzytu vs długość fali, pasmo = szum. Przerywane znaczniki "
            "pokazują, gdzie pochłaniają gazy (np. zafalowanie pod „CO2” to ślad CO2)."
        )
        st.altair_chart(components.spectrum_chart(record), width="stretch")


def results_section(rows: list, agg: dict, quantum_present: bool) -> None:
    with st.container(border=True):
        st.markdown("##### :material/query_stats: Skład atmosfery - modele vs prawda")
        st.caption(
            "Wszystko jako słupki: prawda + 3 modele, pogrupowane po gazie. Wyższy słupek = "
            "więcej gazu. Im bliżej słupka „prawda”, tym lepiej."
        )
        st.altair_chart(components.comparison_chart(rows), width="stretch")
        if agg:
            st.markdown("**Δ do prawdy**  ·  mRMSE = średnia odległość predykcji do prawdy (mniej = bliżej)")
            order = [m for m in (ia.BASELINE_MODEL_NAME, ia.CLASSICAL_MODEL_NAME, ia.QUANTUM_MODEL_NAME) if m in agg]
            best = min(order, key=lambda m: agg[m]["rmse_mean"]) if order else None
            for col, model in zip(st.columns(len(order)), order):
                col.metric(
                    model, f"{agg[model]['rmse_mean']:.3f}", border=True,
                    help="najbliżej prawdy" if model == best else "odległość do prawdy (mRMSE)",
                )
            if best:
                st.caption(
                    f"Najbliżej prawdy: **{best}** ({agg[best]['rmse_mean']:.3f}). baseline (RF) = Random "
                    "Forest · klasyczny = głowica klasyczna · kwantowy = pełny model kwantowy."
                )
        with st.expander("Tabela porównania (log-VMR)", icon=":material/table_chart:"):
            st.dataframe(components.comparison_dataframe(rows), width="stretch")
        if not quantum_present:
            st.caption(
                ":material/info: Brak predykcji modelu kwantowego dla tej planety - baseline (RF) + klasyczny vs prawda."
            )


def whatif_section(
    record: PlanetRecord, engine: ia.InferenceEngine, base: Prediction, truth: GroundTruth | None
) -> None:
    with st.container(border=True):
        st.markdown("##### :material/tune: Eksperyment - jak kształt widma wpływa na skład?")
        st.caption(
            "Każdy ruch suwaka buduje nowe, niemutowalne widmo czystymi funkcjami (FP) i od nowa "
            "odpala model (OOP). Zobacz, jak kształt widma przekłada się na przewidziany skład."
        )
        c = st.columns(3)
        depth = c[0].slider(
            "Głębokość cech ×", 0.0, 2.5, 1.0, 0.1,
            help="Siła sygnału gazów: <1 = słabsze cechy absorpcji (mniej gazu), >1 = mocniejsze.",
        )
        noise = c[1].slider(
            "Szum ×", 0.0, 5.0, 0.0, 0.25,
            help="Jakość pomiaru: symuluje szum instrumentu/obserwacji - większy = bardziej zaszumione widmo.",
        )
        slope = c[2].slider(
            "Nachylenie", -0.5, 0.5, 0.0, 0.05,
            help="Przechylenie kontinuum, np. systematyka instrumentu albo aktywność gwiazdy.",
        )
        st.caption(
            "Suwaki modelują realne zjawiska: **głębokość cech** = jak silnie gaz pochłania (sygnał), "
            "**szum** = gorsza jakość pomiaru, **nachylenie** = przechył kontinuum (instrument / gwiazda)."
        )

        record2 = whatif.perturbed_record(record, depth=depth, noise=noise, slope=slope)
        changed = engine.predict(record2)

        left, right = st.columns(2)
        with left:
            st.caption("Zmienione widmo")
            st.altair_chart(components.spectrum_chart(record2, height=320), width="stretch")
        with right:
            st.caption("Skład: oryginał vs po zmianie" + (" vs prawda" if truth is not None else ""))
            rows, _ = ia.compare(
                truth,
                [Prediction("oryginał", dict(base.log_vmr)), Prediction("po zmianie", dict(changed.log_vmr))],
            )
            st.altair_chart(components.comparison_chart(rows, height=320), width="stretch")

        shift = (sum((base.log_vmr[g] - changed.log_vmr[g]) ** 2 for g in GASES) / len(GASES)) ** 0.5
        st.caption(f"Zmiana predykcji względem oryginału: **{shift:.3f}** (mRMSE).")


def main() -> None:
    st.html(
        "<style>@keyframes exobiomeReveal{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}"
        '[data-testid="stVerticalBlockBorderWrapper"]{animation:exobiomeReveal .4s ease both}</style>'
    )
    try:
        engine = get_engine()
    except Exception as exc:  # noqa: BLE001
        st.title("ExoBiome")
        st.error(f"Nie udało się załadować modelu: {exc}", icon=":material/error:")
        st.stop()

    _, center, _ = st.columns([1, 22, 1])
    with center:
        header()
        intro()
        record = pick_record()
        if record is None:
            st.info(
                "Wybierz planetę z listy albo wczytaj własne widmo (CSV) powyżej.",
                icon=":material/arrow_upward:",
            )
            return

        try:
            predictions = [pred for p in ia.build_predictors(engine) if (pred := p.predict(record)) is not None]
            truth = record.truth or ia.reference_truth(engine.checkpoint_dir, record.planet_id)
            rows, agg = ia.compare(truth, predictions)
        except DataError as exc:
            st.error(f"Błąd danych: {exc}", icon=":material/error:")
            return

        quantum_present = any(pred.model_name == ia.QUANTUM_MODEL_NAME for pred in predictions)
        classical = next((pred for pred in predictions if pred.model_name == ia.CLASSICAL_MODEL_NAME), None)

        kpi_row(record, truth)
        spectrum_section(record)
        results_section(rows, agg, quantum_present)
        if classical is not None:
            whatif_section(record, engine, classical, truth)


main()
