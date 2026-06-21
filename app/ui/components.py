"""Pure Altair/pandas chart builders for the ExoBiome UI (Osoba 4).

No Streamlit calls here - every function returns an Altair chart or a DataFrame,
so the visual logic stays testable without a running app. ``app.py`` does the
``st.*`` rendering and lets the Streamlit theme colour the charts.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from app.data.types import GASES, ComparisonRow, PlanetRecord

GROUND_TRUTH = "prawda (ground truth)"


def gas_label(gas: str) -> str:
    """``"log_H2O"`` -> ``"H2O"`` for axis/legend labels."""
    return gas.replace("log_", "")


GAS_ORDER = [gas_label(g) for g in GASES]


def _x_axis(record: PlanetRecord) -> tuple[np.ndarray, str]:
    """Wavelength grid when it is real and varying, else the bin index."""
    wl = np.asarray(record.spectrum.wavelength, dtype=float)
    if np.isfinite(wl).all() and float(np.ptp(wl)) > 0.0:
        return wl, "długość fali [µm]"
    return np.arange(record.spectrum.flux.shape[0], dtype=float), "bin widma"


def spectrum_dataframe(record: PlanetRecord) -> pd.DataFrame:
    x, _ = _x_axis(record)
    flux = np.asarray(record.spectrum.flux, dtype=float)
    noise = np.abs(np.asarray(record.spectrum.noise, dtype=float))
    # Keep the x column name simple: Vega-Lite reads "[" / "]" in a field name as
    # a nested accessor, which silently breaks the x encoding. The human label
    # ("długość fali [µm]") is applied as the axis title in spectrum_chart.
    return pd.DataFrame({"x": x, "flux": flux, "lo": flux - noise, "hi": flux + noise})


def spectrum_chart(record: PlanetRecord) -> alt.LayerChart:
    """Transmission spectrum: flux line with a +/- noise band.

    The y-axis is zoomed to the data (``zero=False``) so the spectral features -
    the absorption signal the model actually reads - are visible; transit depths
    vary by < 1e-3 around ~0.018, so a zero-based axis flattens them to a line.
    """
    df = spectrum_dataframe(record)
    _, xtitle = _x_axis(record)
    x = alt.X("x:Q", title=xtitle, scale=alt.Scale(zero=False, nice=False))
    base = alt.Chart(df)
    band = base.mark_area(opacity=0.2, color="#22D3EE").encode(
        x=x, y=alt.Y("lo:Q", scale=alt.Scale(zero=False)), y2="hi:Q"
    )
    line = base.mark_line(strokeWidth=2, color="#67E8F9").encode(
        x=x,
        y=alt.Y("flux:Q", title="natężenie widma (transit depth)", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("x:Q", title="λ [µm]", format=".3f"),
            alt.Tooltip("flux:Q", format=".5f"),
        ],
    )
    return (band + line).properties(height=280)


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


def comparison_chart(rows: list[ComparisonRow]) -> alt.LayerChart:
    """Per gas: grouped bars for each model, with ground truth as a diamond marker.

    Splitting truth (a single target per gas) from the model bars makes the chart
    read as "how close does each model's bar get to the truth diamond", which is
    clearer than burying truth as just another bar.
    """
    df = comparison_long(rows)
    x = alt.X("gaz:N", title=None, sort=GAS_ORDER)
    base = alt.Chart(df)
    bars = (
        base.transform_filter(alt.datum.seria != GROUND_TRUTH)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=x,
            xOffset=alt.XOffset("seria:N"),
            y=alt.Y("log-VMR:Q", title="log-VMR  (wyżej = więcej gazu)"),
            color=alt.Color("seria:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=["gaz:N", "seria:N", alt.Tooltip("log-VMR:Q", format=".2f")],
        )
    )
    truth = (
        base.transform_filter(alt.datum.seria == GROUND_TRUTH)
        .mark_point(shape="diamond", filled=True, size=170, color="#E6EAF5", stroke="#0B1020", strokeWidth=1.5)
        .encode(x=x, y="log-VMR:Q", tooltip=[alt.Tooltip("log-VMR:Q", title="prawda", format=".2f")])
    )
    return (bars + truth).properties(height=340)


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
