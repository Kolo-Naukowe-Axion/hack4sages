"""Pure Altair/pandas chart builders for the ExoBiome UI (Osoba 4).

No Streamlit calls here - every function returns an Altair chart or a DataFrame,
so the visual logic stays testable without a running app. ``app.py`` does the
``st.*`` rendering and lets the Streamlit theme colour the charts (no hard-coded
colours, so the chart palette follows ``.streamlit/config.toml``).
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from app.data.types import GASES, ComparisonRow, PlanetRecord

GROUND_TRUTH = "prawda"

# Approximate mid-infrared absorption bands (micrometres) of the target gases,
# used to annotate the spectrum so it is clear which feature comes from which gas.
ABSORPTION_BANDS: list[tuple[float, str]] = [(1.4, "H2O"), (2.3, "CH4"), (4.3, "CO2"), (4.6, "CO")]


def gas_label(gas: str) -> str:
    """``"log_H2O"`` -> ``"H2O"`` for axis/legend labels."""
    return gas.replace("log_", "")


GAS_ORDER = [gas_label(g) for g in GASES]


def _x_axis(record: PlanetRecord) -> tuple[np.ndarray, str, bool]:
    """Wavelength grid when it is real and varying, else the bin index.

    Returns ``(values, axis_title, is_wavelength)``.
    """
    wl = np.asarray(record.spectrum.wavelength, dtype=float)
    if np.isfinite(wl).all() and float(np.ptp(wl)) > 0.0:
        return wl, "długość fali [µm]", True
    return np.arange(record.spectrum.flux.shape[0], dtype=float), "bin widma", False


def spectrum_dataframe(record: PlanetRecord) -> pd.DataFrame:
    x, _, _ = _x_axis(record)
    flux = np.asarray(record.spectrum.flux, dtype=float)
    noise = np.abs(np.asarray(record.spectrum.noise, dtype=float))
    # Plain "x" column: Vega-Lite treats "[" / "]" in a field name as a nested
    # accessor; the human axis title is applied in spectrum_chart instead.
    return pd.DataFrame({"x": x, "flux": flux, "lo": flux - noise, "hi": flux + noise})


def spectrum_chart(record: PlanetRecord, height: int = 300) -> alt.LayerChart:
    """Transmission spectrum: flux line with a +/- noise band and gas-band markers.

    The y-axis zooms to the data (``zero=False``) because transit depths vary by
    < 1e-3, so a zero-based axis would flatten the spectral features to a line.
    """
    df = spectrum_dataframe(record)
    _, xtitle, is_wl = _x_axis(record)
    x = alt.X("x:Q", title=xtitle, scale=alt.Scale(zero=False, nice=False))
    base = alt.Chart(df)
    band = base.mark_area(opacity=0.16, color="#A6C5D7").encode(
        x=x, y=alt.Y("lo:Q", scale=alt.Scale(zero=False)), y2="hi:Q"
    )
    line = base.mark_line(strokeWidth=2, color="#0F52BA").encode(
        x=x,
        y=alt.Y("flux:Q", title="głębokość tranzytu", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("x:Q", title="λ [µm]", format=".2f"),
            alt.Tooltip("flux:Q", title="głębokość", format=".5f"),
        ],
    )
    layers: list[alt.Chart] = [band, line]
    if is_wl:
        bands = pd.DataFrame(ABSORPTION_BANDS, columns=["x", "gaz"])
        rules = alt.Chart(bands).mark_rule(strokeDash=[3, 3], opacity=0.4).encode(x="x:Q")

        def _labels(frame: pd.DataFrame, dy: int) -> alt.Chart:
            # alternate label heights so close bands (e.g. CO2 4.3 / CO 4.6) don't collide
            return alt.Chart(frame).mark_text(
                align="left", baseline="top", dx=3, dy=dy, fontSize=10, opacity=0.8
            ).encode(x="x:Q", y=alt.value(6), text="gaz:N")

        layers += [rules, _labels(bands.iloc[::2], 2), _labels(bands.iloc[1::2], 16)]
    return alt.layer(*layers).properties(height=height)


def comparison_long(rows: list[ComparisonRow]) -> pd.DataFrame:
    """Tidy long form for a grouped bar chart: one row per (gas, series)."""
    records: list[dict[str, object]] = []
    for row in rows:
        label = gas_label(row.gas)
        if row.true is not None:
            records.append({"gaz": label, "seria": GROUND_TRUTH, "log-VMR": row.true})
        for model, value in row.preds.items():
            records.append({"gaz": label, "seria": model, "log-VMR": value})
    return pd.DataFrame.from_records(records, columns=["gaz", "seria", "log-VMR"])


def comparison_chart(rows: list[ComparisonRow], height: int = 380) -> alt.Chart:
    """Grouped bars per gas - ground truth and every model are bars, rising from a
    shared floor so a taller bar means more gas. Narrow bars, one row, legend on top.
    """
    df = comparison_long(rows)
    floor = float(df["log-VMR"].min()) - 0.6
    df = df.assign(floor=floor)
    order = [GROUND_TRUTH] + [s for s in dict.fromkeys(df["seria"]) if s != GROUND_TRUTH]
    return (
        alt.Chart(df)
        .mark_bar(size=14, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("gaz:N", title=None, sort=GAS_ORDER, scale=alt.Scale(paddingInner=0.3)),
            xOffset=alt.XOffset("seria:N", sort=order),
            y=alt.Y("log-VMR:Q", title="log-VMR  (wyżej = więcej gazu)", scale=alt.Scale(zero=False)),
            y2="floor:Q",
            color=alt.Color("seria:N", title=None, sort=order, legend=alt.Legend(orient="top", labelLimit=240)),
            tooltip=["gaz:N", "seria:N", alt.Tooltip("log-VMR:Q", format=".2f")],
        )
        .properties(height=height)
    )


def comparison_dataframe(rows: list[ComparisonRow]) -> pd.DataFrame:
    """Wide table indexed by gas: ground truth, each model, and per-model error."""
    models = list(rows[0].preds.keys()) if rows else []
    data: dict[str, dict[str, float]] = {}
    for row in rows:
        entry: dict[str, float] = {}
        if row.true is not None:
            entry[GROUND_TRUTH] = row.true
        for model in models:
            entry[model] = row.preds.get(model)
        for model, err in row.errors.items():
            entry[f"|błąd| {model}"] = err
        data[gas_label(row.gas)] = entry
    return pd.DataFrame.from_dict(data, orient="index")


def training_curve_chart(history: pd.DataFrame) -> alt.Chart:
    """Train/validation mRMSE per epoch; the hybrid (quantum-on) phase is shaded."""
    melt = history.melt(
        id_vars=["epoch"],
        value_vars=["train_rmse_mean", "val_rmse_mean"],
        var_name="krzywa",
        value_name="mRMSE",
    )
    melt["krzywa"] = melt["krzywa"].map({"train_rmse_mean": "trening", "val_rmse_mean": "walidacja"})
    lines = alt.Chart(melt).mark_line(point=True).encode(
        x=alt.X("epoch:Q", title="epoka"),
        y=alt.Y("mRMSE:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("krzywa:N", title=None, legend=alt.Legend(orient="top")),
        tooltip=["epoch:Q", "krzywa:N", alt.Tooltip("mRMSE:Q", format=".3f")],
    )
    active = history[history["quantum_active"] > 0]
    if active.empty:
        return lines.properties(height=300)
    band = pd.DataFrame({"start": [float(active["epoch"].min())], "end": [float(history["epoch"].max())]})
    shade = alt.Chart(band).mark_rect(opacity=0.12, color="#0F52BA").encode(x="start:Q", x2="end:Q")
    return (shade + lines).properties(height=300)


def per_gas_rmse_chart(rmse: dict[str, float]) -> alt.Chart:
    """Per-gas validation RMSE of the quantum model."""
    df = pd.DataFrame([{"gaz": gas_label(g), "RMSE": float(v)} for g, v in rmse.items()])
    return (
        alt.Chart(df)
        .mark_bar(color="#0F52BA", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("gaz:N", title=None, sort=GAS_ORDER),
            y=alt.Y("RMSE:Q", title="RMSE [dex]"),
            tooltip=["gaz:N", alt.Tooltip("RMSE:Q", format=".3f")],
        )
        .properties(height=240)
    )


def parity_chart(df: pd.DataFrame) -> alt.FacetChart:
    """Predicted vs true log-VMR per gas (one point per planet) with the y=x line."""
    points = alt.Chart(df).mark_circle(size=12, opacity=0.3, color="#0F52BA").encode(
        x=alt.X("prawda:Q", title="prawda"), y=alt.Y("predykcja:Q", title="predykcja")
    )
    identity = alt.Chart(df).mark_line(color="#000926", strokeDash=[4, 4], opacity=0.6).encode(
        x="prawda:Q", y="prawda:Q"
    )
    return (points + identity).properties(width=128, height=128).facet(
        facet=alt.Facet("gaz:N", title=None, sort=GAS_ORDER), columns=5
    )
