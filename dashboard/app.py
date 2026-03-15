"""
Chiller AI-Assisted Smart Control Dashboard
Streamlit + Plotly interactive dashboard for chiller fault detection & diagnostics.
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chiller AI Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS – dark industrial theme ──────────────────────────────────────
st.markdown(
    """
    <style>
        /* Background & base text */
        .stApp { background-color: #0d1117; color: #c9d1d9; }
        section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 16px;
        }
        [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
        [data-testid="stMetricValue"]  { color: #58a6ff !important; font-size: 1.6rem; font-weight: 700; }
        [data-testid="stMetricDelta"]  { color: #3fb950 !important; }

        /* Section headers */
        h1 { color: #e6edf3 !important; font-weight: 800; letter-spacing: -0.5px; }
        h2, h3 { color: #c9d1d9 !important; }

        /* Divider */
        hr { border-color: #21262d; }

        /* Sidebar status badge */
        .status-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.05em;
        }
        .status-normal  { background: #1a3d2b; color: #3fb950; border: 1px solid #238636; }
        .status-anomaly { background: #3d1a1a; color: #f85149; border: 1px solid #da3633; }

        /* Chart containers */
        [data-testid="stPlotlyChart"] > div {
            border: 1px solid #21262d;
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Plotly dark layout defaults ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(family="monospace", color="#8b949e"),
    title_font=dict(color="#c9d1d9", size=14),
    legend=dict(bgcolor="#1c2128", bordercolor="#30363d", borderwidth=1),
    margin=dict(l=60, r=20, t=50, b=50),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickcolor="#8b949e"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickcolor="#8b949e"),
)

COLORS = {
    "water_temp":        "#58a6ff",
    "control_signal":    "#e3b341",
    "power_consumption": "#f85149",
    "fault_leakage":     "#bc8cff",
    "anomaly_marker":    "rgba(248, 81, 73, 0.85)",
}


# ─── Data loading helpers ─────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _latest_glob(pattern: str) -> str | None:
    """Return the most-recently modified file matching a glob pattern, or None."""
    matches = glob.glob(os.path.join(DATA_DIR, pattern))
    return max(matches, key=os.path.getmtime) if matches else None


@st.cache_data(show_spinner=False)
def load_sensor_data() -> pd.DataFrame | None:
    path = _latest_glob("synthetic_sensor_data*.csv")
    if not path:
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_anomalies() -> dict | None:
    path = _latest_glob("anomalies_summary*.json")
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_maintenance_report() -> str | None:
    path = os.path.join(DATA_DIR, "maintenance_report.md")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


# ─── Load data ────────────────────────────────────────────────────────────────
df          = load_sensor_data()
anomaly_raw = load_anomalies()
report_md   = load_maintenance_report()

# Build tidy anomaly dataframe
if anomaly_raw and "anomalies" in anomaly_raw:
    anom_df = pd.DataFrame(anomaly_raw["anomalies"])
else:
    anom_df = pd.DataFrame()

summary = anomaly_raw.get("summary", {}) if anomaly_raw else {}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## System Status")

    # Determine status from last 50 sensor readings
    if df is not None and not anom_df.empty:
        last_time = df["time"].max()
        recent_cutoff = last_time - 50
        recent_anoms = anom_df[anom_df["time"] >= recent_cutoff]
        system_anomaly = len(recent_anoms) > 0
    else:
        system_anomaly = False

    if system_anomaly:
        st.markdown(
            '<span class="status-badge status-anomaly">⚠ ANOMALY DETECTED</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-normal">✓ NORMAL</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Top anomaly sources in sidebar
    if not anom_df.empty and "sensor_type" in anom_df.columns:
        st.markdown("**Top Anomaly Sources**")
        top_sensors = anom_df["sensor_type"].value_counts().head(4)
        for sensor, count in top_sensors.items():
            pct = 100 * count / len(anom_df)
            st.markdown(
                f"<small style='color:#8b949e'>{sensor}</small><br>"
                f"<span style='color:#f85149;font-weight:700'>{count}</span>"
                f"<span style='color:#8b949e'> ({pct:.0f}%)</span>",
                unsafe_allow_html=True,
            )
        st.markdown("---")

    # Maintenance report
    st.markdown("## Maintenance Report")
    if report_md:
        with st.expander("View full report", expanded=False):
            st.markdown(report_md)
        lines = [line for line in report_md.splitlines() if line.strip()]
        for line in lines[:12]:
            st.markdown(
                f"<small style='color:#8b949e'>{line}</small>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("No maintenance report found in `data/`.")


# ─── Main panel ──────────────────────────────────────────────────────────────
st.markdown(
    "<h1>⚙️ Chiller AI-Assisted Smart Control Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#8b949e;font-size:0.9rem;margin-top:-12px'>"
    "Real-time fault detection · Statistical anomaly engine · Claude AI diagnostics"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─── Guard: missing sensor data ──────────────────────────────────────────────
if df is None:
    st.warning(
        "No sensor data found in `data/`. "
        "Run `python scripts/generate_data.py` to produce a dataset."
    )
    st.stop()

# ─── KPI metrics row ─────────────────────────────────────────────────────────
total_anomalies = summary.get("total_anomalies", len(anom_df))
avg_temp        = df["water_temp"].mean()
peak_power      = df["power_consumption"].max()

if not anom_df.empty and "severity_score" in anom_df.columns:
    max_severity = anom_df["severity_score"].max()
    health_score = max(0.0, 100.0 - min(max_severity * 5, 100))
else:
    max_severity = 0.0
    health_score = 100.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg. Water Temp",   f"{avg_temp:.2f} °C",    delta=f"{avg_temp - 20:.2f} vs setpoint")
col2.metric("Peak Power",        f"{peak_power:.2f} kW",  delta=None)
col3.metric("Total Anomalies",   f"{total_anomalies:,}",  delta=None)
col4.metric(
    "Health Score",
    f"{health_score:.1f} / 100",
    delta=f"{health_score - 100:.1f}" if health_score < 100 else None,
    delta_color="inverse",
)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Time-window slider ───────────────────────────────────────────────────────
t_min = int(df["time"].min())
t_max = int(df["time"].max())

with st.expander("⏱  Filter time window", expanded=False):
    t_start, t_end = st.slider(
        "Select time range (s)",
        min_value=t_min,
        max_value=t_max,
        value=(t_min, t_max),
        step=1,
    )

df_view   = df[(df["time"] >= t_start) & (df["time"] <= t_end)].copy()
anom_view = (
    anom_df[(anom_df["time"] >= t_start) & (anom_df["time"] <= t_end)].copy()
    if not anom_df.empty
    else anom_df
)


# ─── Helper: overlay anomaly markers on a Plotly figure ──────────────────────

def add_anomaly_markers(
    fig: go.Figure,
    sensor_filter: list,
    y_ref: float,
    secondary_y: bool = False,
) -> None:
    """Add red triangle-down markers at anomaly timestamps."""
    if anom_view.empty:
        return

    subset = (
        anom_view[anom_view["sensor_type"].isin(sensor_filter)]
        if sensor_filter
        else anom_view
    )
    if subset.empty:
        return

    hover = (
        "<b>Anomaly @ t=%{x:.0f}s</b><br>"
        "Sensor: %{customdata[0]}<br>"
        "Severity: %{customdata[1]:.2f}σ<br>"
        "<extra></extra>"
        if {"sensor_type", "severity_score"}.issubset(subset.columns)
        else "<b>Anomaly @ t=%{x:.0f}s</b><extra></extra>"
    )

    trace = go.Scatter(
        x=subset["time"],
        y=[y_ref] * len(subset),
        mode="markers",
        marker=dict(
            symbol="triangle-down",
            color=COLORS["anomaly_marker"],
            size=9,
            line=dict(color="#f85149", width=1),
        ),
        name="Anomaly",
        customdata=subset[["sensor_type", "severity_score"]].values
        if {"sensor_type", "severity_score"}.issubset(subset.columns)
        else None,
        hovertemplate=hover,
        showlegend=True,
        legendgroup="anomaly",
    )
    fig.add_trace(trace, secondary_y=secondary_y)


# ─── Chart 1: Thermal & Control ──────────────────────────────────────────────
st.markdown("### Thermal & Control Overview")

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

fig1.add_trace(
    go.Scatter(
        x=df_view["time"],
        y=df_view["water_temp"],
        name="Water Temp (°C)",
        line=dict(color=COLORS["water_temp"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.06)",
        hovertemplate="t=%{x:.0f}s  |  Water Temp=%{y:.3f}°C<extra></extra>",
    ),
    secondary_y=False,
)

fig1.add_trace(
    go.Scatter(
        x=df_view["time"],
        y=df_view["control_signal"],
        name="Control Signal",
        line=dict(color=COLORS["control_signal"], width=1.5, dash="dot"),
        hovertemplate="t=%{x:.0f}s  |  Control Signal=%{y:.4f}<extra></extra>",
    ),
    secondary_y=True,
)

# Anomaly markers pinned just above the water_temp range
temp_y_ref = df_view["water_temp"].max() * 1.015 if not df_view.empty else 22.0
add_anomaly_markers(fig1, ["water_temp", "flow_rate"], y_ref=temp_y_ref, secondary_y=False)

fig1.update_layout(
    **PLOTLY_LAYOUT,
    title="Water Temperature vs. PID Control Signal",
    height=350,
    hovermode="x unified",
)
fig1.update_yaxes(
    title_text="Water Temp (°C)", secondary_y=False,
    title_font_color=COLORS["water_temp"], color=COLORS["water_temp"],
    gridcolor="#21262d",
)
fig1.update_yaxes(
    title_text="Control Signal", secondary_y=True,
    title_font_color=COLORS["control_signal"], color=COLORS["control_signal"],
    gridcolor="rgba(0,0,0,0)",
)
fig1.update_xaxes(title_text="Time (s)")

st.plotly_chart(fig1, width='stretch')


# ─── Chart 2: Efficiency & Power ─────────────────────────────────────────────
st.markdown("### Efficiency & Power Consumption")

fig2 = make_subplots(specs=[[{"secondary_y": True}]])

fig2.add_trace(
    go.Scatter(
        x=df_view["time"],
        y=df_view["power_consumption"],
        name="Power Consumption (kW)",
        line=dict(color=COLORS["power_consumption"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.06)",
        hovertemplate="t=%{x:.0f}s  |  Power=%{y:.4f} kW<extra></extra>",
    ),
    secondary_y=False,
)

fig2.add_trace(
    go.Scatter(
        x=df_view["time"],
        y=df_view["fault_leakage"],
        name="Fault: Leakage",
        line=dict(color=COLORS["fault_leakage"], width=1.5, dash="dash"),
        hovertemplate="t=%{x:.0f}s  |  Leakage=%{y:.3f}<extra></extra>",
    ),
    secondary_y=True,
)

# Anomaly markers pinned just above power range
power_y_ref = df_view["power_consumption"].max() * 1.015 if not df_view.empty else 5.0
add_anomaly_markers(
    fig2,
    ["power_consumption", "health_metric (power/control_signal)"],
    y_ref=power_y_ref,
    secondary_y=False,
)

fig2.update_layout(
    **PLOTLY_LAYOUT,
    title="Power Consumption vs. Fault Leakage Signal",
    height=350,
    hovermode="x unified",
)
fig2.update_yaxes(
    title_text="Power (kW)", secondary_y=False,
    title_font_color=COLORS["power_consumption"], color=COLORS["power_consumption"],
    gridcolor="#21262d",
)
fig2.update_yaxes(
    title_text="Leakage Intensity", secondary_y=True,
    title_font_color=COLORS["fault_leakage"], color=COLORS["fault_leakage"],
    range=[-0.05, 1.05],
    gridcolor="rgba(0,0,0,0)",
)
fig2.update_xaxes(title_text="Time (s)")

st.plotly_chart(fig2, width='stretch')


# ─── Anomaly breakdown section ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Anomaly Breakdown")

if anomaly_raw:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**By Detection Method**")
        by_method = summary.get("anomalies_by_method", {})
        if by_method:
            method_df = (
                pd.DataFrame(list(by_method.items()), columns=["Method", "Count"])
                .sort_values("Count", ascending=True)
            )
            fig_m = go.Figure(
                go.Bar(
                    x=method_df["Count"],
                    y=method_df["Method"],
                    orientation="h",
                    marker_color=["#58a6ff", "#e3b341"][: len(method_df)],
                    text=method_df["Count"],
                    textposition="inside",
                    textfont_color="white",
                )
            )
            fig_m.update_layout(
                **{**PLOTLY_LAYOUT, "margin": dict(l=120, r=20, t=10, b=30)},
                height=180,
                showlegend=False,
            )
            st.plotly_chart(fig_m, width='stretch')

    with col_b:
        st.markdown("**By Sensor**")
        by_sensor = summary.get("anomalies_by_sensor", {})
        if by_sensor:
            sensor_df = (
                pd.DataFrame(list(by_sensor.items()), columns=["Sensor", "Count"])
                .sort_values("Count", ascending=True)
            )
            palette = ["#3fb950", "#58a6ff", "#e3b341", "#bc8cff", "#f85149"]
            fig_s = go.Figure(
                go.Bar(
                    x=sensor_df["Count"],
                    y=sensor_df["Sensor"],
                    orientation="h",
                    marker_color=palette[: len(sensor_df)],
                    text=sensor_df["Count"],
                    textposition="inside",
                    textfont_color="white",
                )
            )
            fig_s.update_layout(
                **{**PLOTLY_LAYOUT, "margin": dict(l=270, r=20, t=10, b=30)},
                height=180,
                showlegend=False,
            )
            st.plotly_chart(fig_s, width='stretch')

    # Severity histogram
    st.markdown("**Anomaly Severity Distribution (σ)**")
    if not anom_df.empty and "severity_score" in anom_df.columns:
        fig_h = go.Figure(
            go.Histogram(
                x=anom_df["severity_score"],
                nbinsx=40,
                marker_color="#58a6ff",
                marker_line_color="#0d1117",
                marker_line_width=0.5,
                opacity=0.85,
                name="Frequency",
            )
        )
        fig_h.update_layout(
            **{**PLOTLY_LAYOUT, "margin": dict(l=60, r=20, t=10, b=50)},
            height=220,
            xaxis_title="Severity (σ)",
            yaxis_title="Count",
            showlegend=False,
        )
        st.plotly_chart(fig_h, width='stretch')

else:
    st.warning(
        "No anomaly data found in `data/`. "
        "Run `python src/main.py` to generate anomaly results."
    )

# ─── Recent anomalies table ───────────────────────────────────────────────────
if not anom_df.empty:
    with st.expander("📋  Recent anomaly events (last 20)", expanded=False):
        cols_show = [
            c for c in ["time", "sensor_type", "severity_score", "detection_method", "description"]
            if c in anom_df.columns
        ]
        recent = anom_df.sort_values("time", ascending=False).head(20)[cols_show].reset_index(drop=True)
        st.dataframe(
            recent.style.background_gradient(
                subset=["severity_score"] if "severity_score" in recent.columns else [],
                cmap="Reds",
            ),
        )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.75rem'>"
    "Chiller AI-Assisted Smart Control &nbsp;·&nbsp; Statistical Anomaly Engine"
    " &nbsp;·&nbsp; Claude AI Diagnostics"
    "</p>",
    unsafe_allow_html=True,
)

# ─── Entry-point launcher ─────────────────────────────────────────────────────
def launch_dashboard():
    app_path = Path(__file__)

    print("🚀 Starting Chiller AI Dashboard...")

    try:
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        # Running as a plain Python script – launch Streamlit
        launch_dashboard()
    # else: already inside the Streamlit runtime – do nothing
