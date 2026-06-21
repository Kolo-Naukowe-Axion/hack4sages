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
    x, xlabel = _x_axis(record)
    flux = np.asarray(record.spectrum.flux, dtype=float)
    noise = np.abs(np.asarray(record.spectrum.noise, dtype=float))
    return pd.DataFrame({xlabel: x, "flux": flux, "lo": flux - noise, "hi": flux + noise})


def spectrum_chart(record: PlanetRecord) -> alt.LayerChart:
    """Transmission spectrum: flux line with a +/- noise band."""
    df = spectrum_dataframe(record)
    xlabel = df.columns[0]
    base = alt.Chart(df)
    band = base.mark_area(opacity=0.18).encode(
        x=alt.X(f"{xlabel}:Q", title=xlabel, scale=alt.Scale(zero=False)),
        y=alt.Y("lo:Q", title="natężenie widma"),
        y2="hi:Q",
    )
    line = base.mark_line(strokeWidth=2).encode(
        x=alt.X(f"{xlabel}:Q", scale=alt.Scale(zero=False)),
        y="flux:Q",
        tooltip=[
            alt.Tooltip(f"{xlabel}:Q", format=".3f"),
            alt.Tooltip("flux:Q", format=".4f"),
        ],
    )
    return (band + line).properties(height=260)


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


def comparison_chart(rows: list[ComparisonRow]) -> alt.Chart:
    """Grouped bars: predicted log-VMR per gas, one bar per series (truth + models)."""
    df = comparison_long(rows)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("gaz:N", title=None, sort=GAS_ORDER),
            xOffset=alt.XOffset("seria:N"),
            y=alt.Y("log-VMR:Q", title="log-VMR"),
            color=alt.Color("seria:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=["gaz:N", "seria:N", alt.Tooltip("log-VMR:Q", format=".3f")],
        )
        .properties(height=320)
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
