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
