"""
SPH-IncipientMotion Dashboard (Dash)

Professional dashboard for SPH + Chrono simulation analysis:
- General overview of historical runs
- AI surrogate predictor (Gaussian Process)
- Mesh convergence analysis

Run:
    python app.py
Then open http://127.0.0.1:8050
"""

import json
import math
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "results.sqlite"
CONV_CSV_PATH = PROJECT_ROOT / "data" / "reporte_convergencia.csv"
GCI_JSON_PATH = PROJECT_ROOT / "data" / "figuras_paper" / "convergence_gci_results.json"
SURROGATE_PATH = PROJECT_ROOT / "data" / "gp_surrogate.pkl"

APP_TITLE = "SPH-IncipientMotion Dashboard"
THEME = dbc.themes.DARKLY

G = 9.81
D_EQ = 0.100421
N_BASE = 209103
DISP_FAIL_THRESHOLD_M = 0.005


C_BG = "#0a0f1a"
C_BG_SOFT = "#10192a"
C_BORDER = "rgba(255, 255, 255, 0.10)"
C_TEXT = "#ecf2ff"
C_MUTED = "#9fb0ce"
C_GRID = "rgba(255, 255, 255, 0.08)"

C_BLUE = "#2d9cdb"
C_TEAL = "#2ad1c9"
C_GREEN = "#2ecc71"
C_ORANGE = "#ff9f43"
C_RED = "#ff5d73"
C_YELLOW = "#f4d35e"

C_MOVED = C_RED
C_STABLE = C_GREEN
C_ACCENT = C_BLUE


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body {
  margin: 0;
  padding: 0;
  background:
    radial-gradient(circle at 20% 0%, rgba(42, 209, 201, 0.12), transparent 30%),
    radial-gradient(circle at 85% 10%, rgba(45, 156, 219, 0.14), transparent 35%),
    #0a0f1a;
  color: #ecf2ff;
  font-family: 'Space Grotesk', 'IBM Plex Sans', 'Segoe UI', sans-serif;
}

.glass-card {
  background: linear-gradient(160deg, rgba(19, 33, 58, 0.92), rgba(16, 25, 42, 0.92));
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: 14px;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
}

.section-title {
  font-size: 0.92rem;
  letter-spacing: 0.4px;
  color: #9fb0ce;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}

.metric-title {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: #9fb0ce;
  margin-bottom: 6px;
}

.metric-value {
  font-size: 1.7rem;
  line-height: 1.1;
  font-weight: 700;
}

.nav-tabs {
  border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.nav-tabs .nav-link {
  border: none !important;
  color: #9fb0ce !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
}

.nav-tabs .nav-link.active {
  color: #ecf2ff !important;
  border-bottom: 2px solid #2d9cdb !important;
}

.Select-control, .Select-menu-outer {
  background-color: #10192a !important;
  border: 1px solid rgba(255, 255, 255, 0.10) !important;
  color: #ecf2ff !important;
}

.Select-value-label, .Select-placeholder, .Select-option {
  color: #ecf2ff !important;
}

.rc-slider-track { background-color: #2d9cdb !important; }
.rc-slider-handle {
  border-color: #2d9cdb !important;
  background-color: #d9ecff !important;
}
"""


PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": C_TEXT, "size": 11, "family": "IBM Plex Sans, Space Grotesk, Segoe UI, sans-serif"},
    "margin": {"l": 55, "r": 20, "t": 26, "b": 50},
    "xaxis": {"gridcolor": C_GRID, "showgrid": True, "zerolinecolor": C_GRID},
    "yaxis": {"gridcolor": C_GRID, "showgrid": True, "zerolinecolor": C_GRID},
    "hoverlabel": {"bgcolor": C_BG_SOFT, "font_color": C_TEXT, "bordercolor": C_GRID},
}


def _normalize_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value) != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "ok"}


def load_results() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query("SELECT * FROM results", conn)
    if df.empty:
        return df

    numeric_cols = [
        "max_displacement", "max_displacement_rel", "max_rotation", "max_velocity",
        "max_sph_force", "max_contact_force", "max_flow_velocity", "max_water_height",
        "sim_time_reached", "n_timesteps", "dam_height", "boulder_mass", "dp"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "failed" not in df.columns:
        df["failed"] = 0
    df["failed"] = df["failed"].apply(_normalize_bool)

    if "case_name" not in df.columns:
        df["case_name"] = [f"case_{i+1:03d}" for i in range(len(df))]

    df["status"] = np.where(df["failed"], "Se movio", "Estable")
    df["case_label"] = df["case_name"].astype(str).str.replace("_", " ", regex=False).str.title()

    h_safe = df["max_water_height"].replace(0, np.nan)
    df["froude"] = df["max_flow_velocity"] / np.sqrt(G * h_safe)
    fc_safe = df["max_contact_force"].replace(0, np.nan)
    df["force_ratio"] = df["max_sph_force"] / fc_safe

    if "dp" not in df.columns:
        df["dp"] = np.nan
    missing = df["dp"].isna()
    extracted = df.loc[missing, "case_name"].str.extract(r"dp(\d+)", expand=False)
    df.loc[missing, "dp"] = pd.to_numeric(extracted, errors="coerce") / 1000.0
    return df


def load_convergence() -> Tuple[pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame()
    if not CONV_CSV_PATH.exists():
        return empty, empty
    try:
        df = pd.read_csv(CONV_CSV_PATH, sep=";")
    except Exception:
        df = pd.read_csv(CONV_CSV_PATH)
    if df.empty:
        return empty, empty

    expected = [
        "dp", "case_name", "status", "max_displacement_m", "max_rotation_deg", "max_velocity_ms",
        "max_sph_force_N", "max_contact_force_N", "max_flow_velocity_ms", "max_water_height_m",
        "tiempo_computo_min", "error"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    for col in [
        "dp", "max_displacement_m", "max_rotation_deg", "max_velocity_ms", "max_sph_force_N",
        "max_contact_force_N", "max_flow_velocity_ms", "max_water_height_m", "tiempo_computo_min"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_ok = df[df["status"].astype(str).str.upper() == "OK"].copy()
    if df_ok.empty:
        return df, empty

    h_safe = df_ok["max_water_height_m"].replace(0, np.nan)
    df_ok["froude"] = df_ok["max_flow_velocity_ms"] / np.sqrt(G * h_safe)
    df_ok["particles_est"] = (N_BASE * (0.02 / df_ok["dp"]) ** 3).replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
    df_ok["dim_dp"] = D_EQ / df_ok["dp"]
    df_ok = df_ok.sort_values("dp", ascending=False)
    df_ok["delta_disp_pct"] = df_ok["max_displacement_m"].pct_change().abs() * 100
    df_ok["delta_fsph_pct"] = df_ok["max_sph_force_N"].pct_change().abs() * 100
    return df, df_ok


def load_convergence_enriched() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_all, df_ok = load_convergence()
    df_res = load_results()
    if df_res.empty:
        return df_all, df_ok

    known = set(df_all["case_name"].astype(str).tolist()) if not df_all.empty else set()
    rows = []
    for _, r in df_res[df_res["dp"].notna()].iterrows():
        name = str(r.get("case_name", ""))
        if name in known:
            continue
        rows.append({
            "dp": r.get("dp", np.nan),
            "case_name": name,
            "status": "OK",
            "max_displacement_m": r.get("max_displacement", np.nan),
            "max_displacement_pct": r.get("max_displacement_rel", np.nan),
            "max_rotation_deg": r.get("max_rotation", np.nan),
            "max_velocity_ms": r.get("max_velocity", np.nan),
            "max_sph_force_N": r.get("max_sph_force", np.nan),
            "max_contact_force_N": r.get("max_contact_force", np.nan),
            "max_flow_velocity_ms": r.get("max_flow_velocity", np.nan),
            "max_water_height_m": r.get("max_water_height", np.nan),
            "sim_time_reached_s": r.get("sim_time_reached", np.nan),
            "n_timesteps": r.get("n_timesteps", np.nan),
            "tiempo_computo_min": np.nan,
            "error": "",
        })
    if not rows:
        return df_all, df_ok

    df_new = pd.DataFrame(rows)
    df_all = pd.concat([df_all, df_new], ignore_index=True) if not df_all.empty else df_new
    h_safe = df_new["max_water_height_m"].replace(0, np.nan)
    df_new["froude"] = df_new["max_flow_velocity_ms"] / np.sqrt(G * h_safe)
    df_new["particles_est"] = (N_BASE * (0.02 / df_new["dp"]) ** 3).replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
    df_new["dim_dp"] = D_EQ / df_new["dp"]
    df_ok = pd.concat([df_ok, df_new], ignore_index=True) if not df_ok.empty else df_new
    df_ok = df_ok.sort_values("dp", ascending=False)
    df_ok["delta_disp_pct"] = df_ok["max_displacement_m"].pct_change().abs() * 100
    df_ok["delta_fsph_pct"] = df_ok["max_sph_force_N"].pct_change().abs() * 100
    return df_all, df_ok


def load_gci() -> Optional[Dict]:
    if not GCI_JSON_PATH.exists():
        return None
    try:
        with open(GCI_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_surrogate_bundle() -> Optional[Dict]:
    if not SURROGATE_PATH.exists():
        return None
    try:
        with open(SURROGATE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def make_metric_card(title: str, value: str, unit: str = "", color: str = C_ACCENT):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="metric-title"),
                html.Div(
                    [
                        html.Span(str(value), className="metric-value", style={"color": color}),
                        html.Span(f" {unit}" if unit else "", style={"color": C_MUTED, "fontSize": "0.9rem"}),
                    ]
                ),
            ],
            style={"textAlign": "center", "padding": "16px 12px"},
        ),
        className="glass-card",
        style={"borderTop": f"3px solid {color}"},
    )


def make_chart_card(title: str, graph_id: str, height: int = 340):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="section-title"),
                dcc.Graph(id=graph_id, config={"displayModeBar": False}, style={"height": f"{height}px"}),
            ]
        ),
        className="glass-card",
    )


def make_empty_state(title: str, detail: str = ""):
    content = [html.H4(title, style={"color": C_TEXT, "marginBottom": "10px"})]
    if detail:
        content.append(html.P(detail, style={"color": C_MUTED, "marginBottom": 0}))
    return dbc.Card(dbc.CardBody(content, style={"padding": "40px", "textAlign": "center"}), className="glass-card")


def blank_figure(message: str = "Sin datos") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOT_LAYOUT, height=320)
    fig.add_annotation(text=message, showarrow=False, font={"color": C_MUTED, "size": 13})
    return fig


app = dash.Dash(
    __name__,
    external_stylesheets=[THEME],
    title=APP_TITLE,
    update_title=None,
    suppress_callback_exceptions=True,
)
app.index_string = app.index_string.replace("</head>", f"<style>{CUSTOM_CSS}</style></head>")

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                [
                    html.Span("SPH", style={"fontWeight": "700", "color": C_TEAL, "marginRight": "6px"}),
                    html.Span("IncipientMotion", style={"fontWeight": "600"}),
                ],
                style={"fontSize": "1.15rem"},
            ),
            dbc.Badge("DualSPHysics v5.4 + Chrono", color="secondary", text_color="light", className="px-3 py-2"),
        ],
        fluid=True,
    ),
    color=C_BG,
    dark=True,
    style={"borderBottom": f"1px solid {C_BORDER}", "padding": "12px 0"},
)

app.layout = html.Div(
    [
        navbar,
        dbc.Container(
            [
                dbc.Tabs(
                    id="main-tabs",
                    active_tab="tab-overview",
                    children=[
                        dbc.Tab(label="Resumen General", tab_id="tab-overview"),
                        dbc.Tab(label="Predictor IA", tab_id="tab-predictor"),
                        dbc.Tab(label="Convergencia", tab_id="tab-convergence"),
                    ],
                    style={"marginTop": "18px"},
                ),
                html.Div(id="tab-content", style={"marginTop": "18px", "paddingBottom": "20px"}),
                html.Hr(style={"borderColor": C_GRID, "marginTop": "18px"}),
                html.P(
                    "SPH-IncipientMotion | Kevin Cortes | UCN 2026",
                    style={"color": C_MUTED, "textAlign": "center", "fontSize": "0.82rem", "marginBottom": "28px"},
                ),
                dcc.Download(id="overview-download"),
                dcc.Download(id="convergence-download"),
            ],
            fluid=True,
            style={"maxWidth": "1700px"},
        ),
    ],
    style={"backgroundColor": C_BG, "minHeight": "100vh"},
)


def build_overview_layout():
    df = load_results()
    if df.empty:
        return make_empty_state("No hay datos de simulacion", f"SQLite esperada: {DB_PATH}")

    min_disp = float(np.nanmin(df["max_displacement"])) if df["max_displacement"].notna().any() else 0.0
    max_disp = float(np.nanmax(df["max_displacement"])) if df["max_displacement"].notna().any() else 1.0
    if not np.isfinite(min_disp):
        min_disp = 0.0
    if not np.isfinite(max_disp) or max_disp <= 0:
        max_disp = 1.0

    return html.Div([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Casos", className="section-title"),
                    dcc.Dropdown(
                        id="overview-case-filter",
                        options=[{"label": c, "value": c} for c in sorted(df["case_name"].astype(str).unique())],
                        value=sorted(df["case_name"].astype(str).tolist()),
                        multi=True,
                        placeholder="Selecciona casos...",
                    ),
                ], lg=7, md=12),
                dbc.Col([
                    html.Label("Desplazamiento minimo [m]", className="section-title"),
                    dcc.Slider(
                        id="overview-min-disp",
                        min=0,
                        max=max_disp,
                        step=max(max_disp / 120.0, 0.01),
                        value=min_disp,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], lg=4, md=9),
                dbc.Col(
                    dbc.Button("Exportar CSV", id="overview-download-btn", color="info", className="w-100", style={"marginTop": "24px"}),
                    lg=1,
                    md=3,
                ),
            ], className="g-3")
        ]), className="glass-card mb-4"),

        dbc.Row(id="overview-metrics", className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(make_chart_card("Desplazamiento vs Rotacion", "overview-fig-scatter"), lg=6, md=12),
            dbc.Col(make_chart_card("Froude vs Desplazamiento", "overview-fig-froude"), lg=6, md=12),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(make_chart_card("Fuerza SPH vs Contacto", "overview-fig-forces"), lg=6, md=12),
            dbc.Col(make_chart_card("Velocidad de flujo vs Altura", "overview-fig-flow"), lg=6, md=12),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(make_chart_card("Desplazamiento por Caso", "overview-fig-bar", height=420), lg=12),
        ], className="g-3 mb-4"),
        dbc.Card(dbc.CardBody([
            html.Div("Tabla de resultados", className="section-title"),
            html.Div(id="overview-table"),
        ]), className="glass-card"),
    ])


def build_predictor_layout():
    df = load_results()
    surrogate = load_surrogate_bundle()
    if surrogate is None:
        return make_empty_state("Modelo surrogate no encontrado", f"Archivo esperado: {SURROGATE_PATH}")
    if df.empty:
        return make_empty_state("No hay datos historicos", f"SQLite esperada: {DB_PATH}")

    h_min = float(df["dam_height"].dropna().min()) if df["dam_height"].notna().any() else 0.10
    h_max = float(df["dam_height"].dropna().max()) if df["dam_height"].notna().any() else 0.60
    m_min = float(df["boulder_mass"].dropna().min()) if df["boulder_mass"].notna().any() else 0.50
    m_max = float(df["boulder_mass"].dropna().max()) if df["boulder_mass"].notna().any() else 3.50
    dps = sorted([float(v) for v in pd.concat([df["dp"], pd.Series([0.02, 0.015, 0.01, 0.008, 0.005, 0.004])]).dropna().unique()], reverse=True)
    stl_values = sorted([str(v) for v in df.get("stl_file", pd.Series(dtype=str)).dropna().unique()]) or ["BLIR3.stl"]

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Panel de parametros", className="section-title"),
                html.Label("Altura de ola H [m]", style={"color": C_TEXT, "fontSize": "0.85rem"}),
                dcc.Slider(id="pred-dam-height", min=max(0.01, h_min * 0.85), max=h_max * 1.15, step=0.005, value=float(np.clip(0.30, h_min, h_max)), tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                html.Label("Masa del bloque M [kg]", style={"color": C_TEXT, "fontSize": "0.85rem"}),
                dcc.Slider(id="pred-mass", min=max(0.10, m_min * 0.85), max=m_max * 1.15, step=0.01, value=float(np.clip(1.2, m_min, m_max)), tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                html.Label("Resolucion de referencia dp [m]", style={"color": C_TEXT, "fontSize": "0.85rem"}),
                dcc.Dropdown(id="pred-dp", options=[{"label": f"{d:.3f}", "value": d} for d in dps], value=dps[0] if dps else 0.02, clearable=False),
                html.Br(),
                html.Label("Geometria STL", style={"color": C_TEXT, "fontSize": "0.85rem"}),
                dcc.Dropdown(id="pred-stl", options=[{"label": s, "value": s} for s in stl_values], value=stl_values[0], clearable=False),
                html.Div([
                    html.Div("Modelo", className="section-title", style={"marginTop": "18px", "marginBottom": "6px"}),
                    html.P(
                        f"GP listo | n_real={int(surrogate.get('n_real', 0))} | n_synthetic={int(surrogate.get('n_synthetic', 0))} | LOO R2={float(surrogate.get('loo_r2', 0)):.3f}",
                        style={"color": C_MUTED, "fontSize": "0.82rem", "marginBottom": 0},
                    ),
                ]),
            ]), className="glass-card"), lg=4, md=12),
            dbc.Col([
                dbc.Row(id="pred-metrics", className="g-3 mb-3"),
                dbc.Row([
                    dbc.Col(make_chart_card("Probabilidad de movimiento", "pred-fig-prob"), lg=6, md=12),
                    dbc.Col(make_chart_card("Incertidumbre del modelo", "pred-fig-unc"), lg=6, md=12),
                ], className="g-3 mb-3"),
                dbc.Row([
                    dbc.Col(make_chart_card("Mapa de casos historicos (H vs M)", "pred-fig-phase", height=380), lg=12),
                ], className="g-3 mb-3"),
                dbc.Card(dbc.CardBody([
                    html.Div("Casos mas cercanos", className="section-title"),
                    html.Div(id="pred-neighbors-table"),
                ]), className="glass-card"),
            ], lg=8, md=12),
        ], className="g-3"),
    ])


def build_convergence_layout():
    _, df_ok = load_convergence_enriched()
    if df_ok.empty:
        return make_empty_state("Sin datos de convergencia", f"CSV esperado: {CONV_CSV_PATH}")

    return html.Div([
        dcc.Store(id="conv-trigger", data="ready"),
        dbc.Row([dbc.Col(dbc.Button("Exportar CSV", id="convergence-download-btn", color="info", className="w-100"), lg=2, md=4, sm=6)], className="mb-3"),
        dbc.Row(id="conv-metrics", className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(make_chart_card("Convergencia de desplazamiento", "conv-fig-disp"), lg=6, md=12),
            dbc.Col(make_chart_card("Convergencia de fuerza SPH", "conv-fig-force"), lg=6, md=12),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(make_chart_card("Costo computacional", "conv-fig-cost"), lg=6, md=12),
            dbc.Col(make_chart_card("Delta relativo entre resoluciones", "conv-fig-delta"), lg=6, md=12),
        ], className="g-3 mb-4"),
        dbc.Card(dbc.CardBody([
            html.Div("Tabla de convergencia", className="section-title"),
            html.Div(id="conv-table"),
        ]), className="glass-card"),
    ])


@app.callback(Output("tab-content", "children"), Input("main-tabs", "active_tab"))
def render_tab(active_tab: str):
    if active_tab == "tab-overview":
        return build_overview_layout()
    if active_tab == "tab-predictor":
        return build_predictor_layout()
    if active_tab == "tab-convergence":
        return build_convergence_layout()
    return html.Div()

@app.callback(
    [
        Output("overview-metrics", "children"),
        Output("overview-fig-scatter", "figure"),
        Output("overview-fig-froude", "figure"),
        Output("overview-fig-forces", "figure"),
        Output("overview-fig-flow", "figure"),
        Output("overview-fig-bar", "figure"),
        Output("overview-table", "children"),
    ],
    [Input("overview-case-filter", "value"), Input("overview-min-disp", "value")],
)
def update_overview(selected_cases, min_disp):
    df = load_results()
    empty = blank_figure("Sin datos")
    if df.empty:
        return [], empty, empty, empty, empty, empty, html.P("Sin datos", style={"color": C_MUTED})

    selected_cases = selected_cases or []
    if not selected_cases:
        return [], empty, empty, empty, empty, empty, html.P("Selecciona al menos un caso", style={"color": C_MUTED})

    min_disp = float(min_disp or 0)
    dff = df[df["case_name"].isin(selected_cases)].copy()
    dff = dff[dff["max_displacement"] >= min_disp]
    if dff.empty:
        return [], empty, empty, empty, empty, empty, html.P("No hay datos con este filtro", style={"color": C_MUTED})

    n_cases = len(dff)
    n_moved = int(dff["failed"].sum())
    moved_ratio = 100.0 * n_moved / max(n_cases, 1)
    avg_disp = dff["max_displacement"].mean()
    avg_rot = dff["max_rotation"].mean()
    avg_froude = dff["froude"].mean()
    avg_force_ratio = dff["force_ratio"].mean()

    metrics = [
        dbc.Col(make_metric_card("Casos", f"{n_cases}", color=C_ACCENT), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("Movidos", f"{n_moved}", color=C_MOVED), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("Estables", f"{n_cases - n_moved}", color=C_STABLE), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("Movidos [%]", f"{moved_ratio:.1f}", "%", C_ORANGE), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("Desplazamiento prom", f"{avg_disp:.3f}", "m", C_TEAL), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("Rotacion prom", f"{avg_rot:.1f}", "deg", C_YELLOW), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("Froude prom", f"{avg_froude:.3f}" if np.isfinite(avg_froude) else "N/A", "", C_BLUE), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("F_SPH / F_cont", f"{avg_force_ratio:.3f}" if np.isfinite(avg_force_ratio) else "N/A", "", C_ORANGE), lg=2, md=4, sm=6),
    ]

    fig_scatter = go.Figure()
    for status, color in (("Se movio", C_MOVED), ("Estable", C_STABLE)):
        sub = dff[dff["status"] == status]
        if sub.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=sub["max_displacement"],
            y=sub["max_rotation"],
            mode="markers+text",
            text=sub["case_label"],
            textposition="top center",
            marker={"size": np.clip(sub["max_velocity"].fillna(0) * 10 + 8, 8, 32), "color": color, "line": {"width": 1, "color": "white"}, "opacity": 0.85},
            name=status,
            hovertemplate="<b>%{text}</b><br>Disp: %{x:.3f} m<br>Rot: %{y:.1f} deg<extra></extra>",
        ))
    fig_scatter.update_layout(**PLOT_LAYOUT, height=340, xaxis_title="Desplazamiento [m]", yaxis_title="Rotacion [deg]")

    fig_froude = go.Figure()
    for status, color in (("Se movio", C_MOVED), ("Estable", C_STABLE)):
        sub = dff[dff["status"] == status]
        if sub.empty:
            continue
        fig_froude.add_trace(go.Scatter(
            x=sub["froude"],
            y=sub["max_displacement"],
            mode="markers",
            marker={"size": 11, "color": color, "line": {"width": 1, "color": "white"}},
            text=sub["case_label"],
            name=status,
            hovertemplate="<b>%{text}</b><br>Fr: %{x:.3f}<br>Disp: %{y:.3f} m<extra></extra>",
        ))
    fig_froude.add_vline(x=1.0, line_dash="dash", line_color=C_MUTED, annotation_text="Fr=1")
    fig_froude.update_layout(**PLOT_LAYOUT, height=340, xaxis_title="Numero de Froude [-]", yaxis_title="Desplazamiento [m]")

    dff_sorted = dff.sort_values("max_sph_force", ascending=True)
    fig_forces = go.Figure()
    fig_forces.add_trace(go.Bar(y=dff_sorted["case_label"], x=dff_sorted["max_sph_force"], orientation="h", name="F_SPH", marker={"color": C_BLUE}))
    fig_forces.add_trace(go.Bar(y=dff_sorted["case_label"], x=dff_sorted["max_contact_force"], orientation="h", name="F_contacto", marker={"color": C_ORANGE}))
    fig_forces.update_layout(**PLOT_LAYOUT, height=340, barmode="group", xaxis_title="Fuerza [N]")

    fig_flow = go.Figure()
    fig_flow.add_trace(go.Scatter(
        x=dff["max_flow_velocity"],
        y=dff["max_water_height"],
        mode="markers",
        marker={"size": np.clip(dff["max_displacement"].fillna(0) * 3 + 8, 8, 34), "color": [C_MOVED if v else C_STABLE for v in dff["failed"]], "line": {"width": 1, "color": "white"}, "opacity": 0.85},
        text=dff["case_label"],
        hovertemplate="<b>%{text}</b><br>v: %{x:.3f} m/s<br>h: %{y:.4f} m<extra></extra>",
        name="Casos",
    ))
    h_max = float(dff["max_water_height"].max()) if dff["max_water_height"].notna().any() else 0.0
    if h_max > 0:
        h_range = np.linspace(0.001, h_max * 1.3, 70)
        fig_flow.add_trace(go.Scatter(x=np.sqrt(G * h_range), y=h_range, mode="lines", line={"color": C_MUTED, "dash": "dash", "width": 1.4}, name="Fr=1", hoverinfo="skip"))
    fig_flow.update_layout(**PLOT_LAYOUT, height=340, xaxis_title="Velocidad de flujo [m/s]", yaxis_title="Altura de agua [m]")

    bar_df = dff.sort_values("max_displacement", ascending=True)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=bar_df["case_label"],
        x=bar_df["max_displacement"],
        orientation="h",
        marker={"color": [C_MOVED if v else C_STABLE for v in bar_df["failed"]], "line": {"width": 0.5, "color": "#1e1e1e"}},
        text=[f"{x:.3f} m" if np.isfinite(x) else "" for x in bar_df["max_displacement"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Disp: %{x:.3f} m<extra></extra>",
    ))
    fig_bar.update_layout(**PLOT_LAYOUT, height=420, xaxis_title="Desplazamiento [m]")

    table = dash_table.DataTable(
        data=dff.sort_values("max_displacement", ascending=False).to_dict("records"),
        columns=[
            {"name": "Caso", "id": "case_name"},
            {"name": "dp [m]", "id": "dp", "type": "numeric", "format": Format(precision=4, scheme=Scheme.fixed)},
            {"name": "Disp [m]", "id": "max_displacement", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
            {"name": "Rot [deg]", "id": "max_rotation", "type": "numeric", "format": Format(precision=1, scheme=Scheme.fixed)},
            {"name": "Vel [m/s]", "id": "max_velocity", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
            {"name": "F_SPH [N]", "id": "max_sph_force", "type": "numeric", "format": Format(precision=1, scheme=Scheme.fixed)},
            {"name": "F_cont [N]", "id": "max_contact_force", "type": "numeric", "format": Format(precision=1, scheme=Scheme.fixed)},
            {"name": "Froude", "id": "froude", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
            {"name": "F_SPH/F_cont", "id": "force_ratio", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
            {"name": "Estado", "id": "status"},
        ],
        sort_action="native",
        page_size=12,
        style_header={"backgroundColor": "rgba(12, 20, 36, 0.95)", "color": C_MUTED, "fontWeight": "600", "border": f"1px solid {C_GRID}", "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.5px"},
        style_cell={"backgroundColor": "rgba(19, 33, 58, 0.45)", "color": C_TEXT, "border": f"1px solid {C_GRID}", "fontSize": "0.84rem", "padding": "11px 8px", "textAlign": "center"},
        style_data_conditional=[
            {"if": {"filter_query": '{status} = "Se movio"'}, "backgroundColor": "rgba(255, 93, 115, 0.13)"},
            {"if": {"filter_query": '{status} = "Estable"'}, "backgroundColor": "rgba(46, 204, 113, 0.09)"},
            {"if": {"state": "active"}, "backgroundColor": "rgba(45, 156, 219, 0.18)", "border": f"1px solid {C_BLUE}"},
        ],
    )

    return metrics, fig_scatter, fig_froude, fig_forces, fig_flow, fig_bar, table


@app.callback(
    Output("overview-download", "data"),
    Input("overview-download-btn", "n_clicks"),
    State("overview-case-filter", "value"),
    State("overview-min-disp", "value"),
    prevent_initial_call=True,
)
def download_overview_csv(n_clicks, selected_cases, min_disp):
    df = load_results()
    if df.empty:
        return dash.no_update
    selected_cases = selected_cases or []
    if not selected_cases:
        return dash.no_update

    dff = df[df["case_name"].isin(selected_cases)].copy()
    dff = dff[dff["max_displacement"] >= float(min_disp or 0)]
    if dff.empty:
        return dash.no_update

    cols = [
        "case_name", "dp", "dam_height", "boulder_mass", "max_displacement", "max_rotation",
        "max_velocity", "max_sph_force", "max_contact_force", "max_flow_velocity", "max_water_height",
        "froude", "force_ratio", "status",
    ]
    available = [c for c in cols if c in dff.columns]
    export_df = dff[available].sort_values("max_displacement", ascending=False)
    return dcc.send_data_frame(export_df.to_csv, "overview_filtered.csv", index=False)


def _predict_motion(dam_height: float, boulder_mass: float) -> Optional[Dict]:
    bundle = load_surrogate_bundle()
    if bundle is None:
        return None

    gp = bundle.get("gp")
    scaler_x = bundle.get("scaler_X")
    scaler_y = bundle.get("scaler_y")
    if gp is None or scaler_x is None or scaler_y is None:
        return None

    x = np.array([[float(dam_height), float(boulder_mass)]])
    x_scaled = scaler_x.transform(x)
    pred_scaled, std_scaled = gp.predict(x_scaled, return_std=True)

    pred = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])
    std = float(std_scaled[0] * scaler_y.scale_[0])

    if std <= 1e-12:
        prob_move = 1.0 if pred > DISP_FAIL_THRESHOLD_M else 0.0
    else:
        z = (DISP_FAIL_THRESHOLD_M - pred) / std
        prob_move = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    prob_move = float(np.clip(prob_move, 0.0, 1.0))

    baseline_std = float(max(abs(float(scaler_y.scale_[0])), 1e-8))
    unc_pct = float(np.clip(100.0 * std / baseline_std, 0.0, 100.0))

    return {"pred_disp_m": pred, "std_m": std, "prob_move": prob_move, "unc_pct": unc_pct}

@app.callback(
    [
        Output("pred-metrics", "children"),
        Output("pred-fig-prob", "figure"),
        Output("pred-fig-unc", "figure"),
        Output("pred-fig-phase", "figure"),
        Output("pred-neighbors-table", "children"),
    ],
    [
        Input("pred-dam-height", "value"),
        Input("pred-mass", "value"),
        Input("pred-dp", "value"),
        Input("pred-stl", "value"),
    ],
)
def update_predictor(dam_height, boulder_mass, dp_value, stl_name):
    pred = _predict_motion(float(dam_height), float(boulder_mass))
    df = load_results()
    if pred is None:
        empty = blank_figure("Modelo no disponible")
        return [], empty, empty, empty, html.P("Modelo no disponible", style={"color": C_MUTED})

    pred_disp = pred["pred_disp_m"]
    std_m = pred["std_m"]
    prob_move = pred["prob_move"]
    unc_pct = pred["unc_pct"]

    status_label = "CRITICO" if prob_move >= 0.5 else "SEGURO"
    status_color = C_RED if prob_move >= 0.5 else C_GREEN

    metrics = [
        dbc.Col(make_metric_card("Estado", status_label, color=status_color), lg=3, md=6, sm=6),
        dbc.Col(make_metric_card("P(movimiento)", f"{prob_move * 100:.1f}", "%", C_ORANGE), lg=3, md=6, sm=6),
        dbc.Col(make_metric_card("Desplazamiento predicho", f"{pred_disp:.4f}", "m", C_TEAL), lg=3, md=6, sm=6),
        dbc.Col(make_metric_card("Incertidumbre sigma", f"{std_m:.4f}", "m", C_YELLOW), lg=3, md=6, sm=6),
    ]

    fig_prob = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_move * 100,
        number={"suffix": "%", "font": {"size": 34, "color": C_TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": C_MUTED},
            "bar": {"color": C_ORANGE},
            "steps": [
                {"range": [0, 35], "color": "rgba(46,204,113,0.35)"},
                {"range": [35, 70], "color": "rgba(244,211,94,0.35)"},
                {"range": [70, 100], "color": "rgba(255,93,115,0.35)"},
            ],
            "threshold": {"line": {"color": C_RED, "width": 3}, "value": 50},
        },
        title={"text": "Probabilidad de movimiento", "font": {"color": C_MUTED, "size": 13}},
    ))
    fig_prob.update_layout(**{**PLOT_LAYOUT, "margin": {"l": 15, "r": 15, "t": 45, "b": 10}}, height=320)

    fig_unc = go.Figure(go.Indicator(
        mode="gauge+number",
        value=unc_pct,
        number={"suffix": "%", "font": {"size": 34, "color": C_TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": C_MUTED},
            "bar": {"color": C_BLUE},
            "steps": [
                {"range": [0, 35], "color": "rgba(46,204,113,0.35)"},
                {"range": [35, 70], "color": "rgba(244,211,94,0.35)"},
                {"range": [70, 100], "color": "rgba(255,93,115,0.35)"},
            ],
        },
        title={"text": "Incertidumbre relativa del GP", "font": {"color": C_MUTED, "size": 13}},
    ))
    fig_unc.update_layout(**{**PLOT_LAYOUT, "margin": {"l": 15, "r": 15, "t": 45, "b": 10}}, height=320)

    fig_phase = go.Figure()
    if not df.empty and {"dam_height", "boulder_mass"}.issubset(df.columns):
        for label, color in (("Se movio", C_MOVED), ("Estable", C_STABLE)):
            sub = df[df["status"] == label]
            if sub.empty:
                continue
            fig_phase.add_trace(go.Scatter(
                x=sub["dam_height"],
                y=sub["boulder_mass"],
                mode="markers",
                name=label,
                text=sub["case_label"],
                marker={"size": np.clip(sub["max_displacement"].fillna(0) * 2.4 + 9, 9, 24), "color": color, "opacity": 0.78, "line": {"color": "white", "width": 1}},
                hovertemplate="<b>%{text}</b><br>H=%{x:.3f} m<br>M=%{y:.3f} kg<extra></extra>",
            ))

    fig_phase.add_trace(go.Scatter(
        x=[dam_height],
        y=[boulder_mass],
        mode="markers+text",
        name="Input actual",
        text=[f"Pred={pred_disp:.3f} m"],
        textposition="top center",
        marker={"symbol": "star", "size": 20, "color": C_YELLOW, "line": {"width": 1.5, "color": "black"}},
        hovertemplate=(
            "<b>Input actual</b><br>H=%{x:.3f} m<br>M=%{y:.3f} kg"
            f"<br>dp={float(dp_value):.3f} m<br>STL={stl_name}<extra></extra>"
        ),
    ))
    fig_phase.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="Altura de ola H [m]", yaxis_title="Masa del bloque M [kg]", legend={"orientation": "h", "y": 1.1})

    neighbors_df = pd.DataFrame()
    if not df.empty and {"dam_height", "boulder_mass"}.issubset(df.columns):
        dfn = df.copy()
        dfn["dist"] = np.sqrt((dfn["dam_height"] - float(dam_height)) ** 2 + (dfn["boulder_mass"] - float(boulder_mass)) ** 2)
        neighbors_df = dfn.sort_values("dist", ascending=True).head(6)

    if neighbors_df.empty:
        neighbors_component = html.P("No hay casos cercanos en SQLite", style={"color": C_MUTED})
    else:
        neighbors_component = dash_table.DataTable(
            data=neighbors_df.to_dict("records"),
            columns=[
                {"name": "Caso", "id": "case_name"},
                {"name": "H [m]", "id": "dam_height", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
                {"name": "M [kg]", "id": "boulder_mass", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
                {"name": "Disp [m]", "id": "max_displacement", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
                {"name": "Estado", "id": "status"},
                {"name": "Distancia", "id": "dist", "type": "numeric", "format": Format(precision=4, scheme=Scheme.fixed)},
            ],
            sort_action="native",
            page_size=6,
            style_header={"backgroundColor": "rgba(12, 20, 36, 0.95)", "color": C_MUTED, "fontWeight": "600", "border": f"1px solid {C_GRID}", "fontSize": "0.78rem", "textTransform": "uppercase"},
            style_cell={"backgroundColor": "rgba(19, 33, 58, 0.45)", "color": C_TEXT, "border": f"1px solid {C_GRID}", "fontSize": "0.84rem", "padding": "10px 8px", "textAlign": "center"},
            style_data_conditional=[
                {"if": {"filter_query": '{status} = "Se movio"'}, "backgroundColor": "rgba(255, 93, 115, 0.13)"},
                {"if": {"filter_query": '{status} = "Estable"'}, "backgroundColor": "rgba(46, 204, 113, 0.09)"},
            ],
        )

    return metrics, fig_prob, fig_unc, fig_phase, neighbors_component


@app.callback(
    [
        Output("conv-metrics", "children"),
        Output("conv-fig-disp", "figure"),
        Output("conv-fig-force", "figure"),
        Output("conv-fig-cost", "figure"),
        Output("conv-fig-delta", "figure"),
        Output("conv-table", "children"),
    ],
    Input("conv-trigger", "data"),
)
def update_convergence(_trigger):
    df_all, df_ok = load_convergence_enriched()
    gci = load_gci()
    if df_ok.empty:
        empty = blank_figure("Sin datos de convergencia")
        return [], empty, empty, empty, empty, html.P("Sin datos", style={"color": C_MUTED})

    df_ok = df_ok.sort_values("dp", ascending=False)
    gci_disp = "N/A"
    if gci and isinstance(gci, dict):
        metrics = gci.get("metrics", {})
        disp = metrics.get("Displacement", {}) if isinstance(metrics, dict) else {}
        val = disp.get("uncertainty_pct")
        if val is not None:
            gci_disp = f"{float(val):.1f}%"

    total_time = float(df_ok["tiempo_computo_min"].sum()) if "tiempo_computo_min" in df_ok.columns else np.nan
    metrics_cards = [
        dbc.Col(make_metric_card("Resoluciones OK", f"{len(df_ok)}", color=C_ACCENT), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("dp fino", f"{df_ok['dp'].min():.3f}", "m", C_BLUE), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("dp grueso", f"{df_ok['dp'].max():.3f}", "m", C_RED), lg=2, md=4, sm=6),
        dbc.Col(make_metric_card("Incertidumbre GCI", gci_disp, "", C_YELLOW), lg=3, md=6, sm=6),
        dbc.Col(make_metric_card("Tiempo total", f"{total_time:.1f}" if np.isfinite(total_time) else "N/A", "min", C_ORANGE), lg=3, md=6, sm=6),
    ]

    fig_disp = go.Figure()
    fig_disp.add_trace(go.Scatter(x=df_ok["dp"], y=df_ok["max_displacement_m"], mode="lines+markers", name="Desplazamiento", line={"color": C_BLUE, "width": 2.2}, marker={"size": 8, "line": {"color": "white", "width": 1}}))
    if not df_all.empty:
        failed = df_all[df_all["status"].astype(str).str.upper() != "OK"]
        failed = failed[failed["dp"].notna()]
        if not failed.empty:
            fig_disp.add_trace(go.Scatter(x=failed["dp"], y=[0] * len(failed), mode="markers+text", name="Fallo", text=["Fallo"] * len(failed), textposition="top center", marker={"symbol": "x", "size": 12, "color": C_RED, "line": {"width": 2}}))
    fig_disp.update_layout(**PLOT_LAYOUT, height=340, xaxis_title="dp [m]", yaxis_title="Desplazamiento [m]")
    fig_disp.update_xaxes(autorange="reversed")

    fig_force = go.Figure()
    fig_force.add_trace(go.Scatter(x=df_ok["dp"], y=df_ok["max_sph_force_N"], mode="lines+markers", name="F_SPH", line={"color": C_RED, "width": 2.2}, marker={"size": 8, "line": {"color": "white", "width": 1}}))
    fig_force.update_layout(**PLOT_LAYOUT, height=340, xaxis_title="dp [m]", yaxis_title="Fuerza SPH [N]")
    fig_force.update_xaxes(autorange="reversed")

    labels = [f"{d:.3f}" for d in df_ok["dp"]]
    fig_cost = make_subplots(specs=[[{"secondary_y": True}]])
    fig_cost.add_trace(go.Bar(x=labels, y=df_ok["tiempo_computo_min"], name="Tiempo [min]", marker_color=C_ACCENT, opacity=0.75), secondary_y=False)
    fig_cost.add_trace(go.Scatter(x=labels, y=df_ok["particles_est"], mode="lines+markers", name="Particulas estimadas", line={"color": C_ORANGE, "width": 2}, marker={"size": 7, "color": C_ORANGE}), secondary_y=True)
    fig_cost.update_layout(**{k: v for k, v in PLOT_LAYOUT.items() if k not in ("xaxis", "yaxis")}, height=340, xaxis={"gridcolor": C_GRID, "title": "dp [m]"})
    fig_cost.update_yaxes(title_text="Tiempo [min]", secondary_y=False, gridcolor=C_GRID, color=C_TEXT)
    fig_cost.update_yaxes(title_text="Particulas estimadas", secondary_y=True, gridcolor=C_GRID, color=C_TEXT)

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=df_ok["dp"], y=df_ok["delta_disp_pct"], mode="lines+markers", name="Delta desplazamiento [%]", line={"color": C_TEAL, "width": 2}, marker={"size": 8}))
    fig_delta.add_trace(go.Scatter(x=df_ok["dp"], y=df_ok["delta_fsph_pct"], mode="lines+markers", name="Delta F_SPH [%]", line={"color": C_YELLOW, "width": 2}, marker={"size": 8}))
    fig_delta.update_layout(**PLOT_LAYOUT, height=340, xaxis_title="dp [m]", yaxis_title="Variacion [%]")
    fig_delta.update_xaxes(autorange="reversed")

    table_df = df_all.sort_values("dp", ascending=False) if not df_all.empty else df_ok.copy()
    conv_table = dash_table.DataTable(
        data=table_df.fillna("").to_dict("records"),
        columns=[
            {"name": "dp [m]", "id": "dp", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
            {"name": "Caso", "id": "case_name"},
            {"name": "Estado", "id": "status"},
            {"name": "Disp [m]", "id": "max_displacement_m", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
            {"name": "Rot [deg]", "id": "max_rotation_deg", "type": "numeric", "format": Format(precision=1, scheme=Scheme.fixed)},
            {"name": "F_SPH [N]", "id": "max_sph_force_N", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
            {"name": "F_cont [N]", "id": "max_contact_force_N", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
            {"name": "Tiempo [min]", "id": "tiempo_computo_min", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
            {"name": "Error", "id": "error"},
        ],
        sort_action="native",
        page_size=12,
        style_header={"backgroundColor": "rgba(12, 20, 36, 0.95)", "color": C_MUTED, "fontWeight": "600", "border": f"1px solid {C_GRID}", "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.5px"},
        style_cell={"backgroundColor": "rgba(19, 33, 58, 0.45)", "color": C_TEXT, "border": f"1px solid {C_GRID}", "fontSize": "0.84rem", "padding": "11px 8px", "textAlign": "center"},
        style_data_conditional=[
            {"if": {"filter_query": '{status} != "OK"'}, "backgroundColor": "rgba(255, 93, 115, 0.16)", "color": "#ffd6dd"},
            {"if": {"filter_query": '{status} = "OK"'}, "backgroundColor": "rgba(46, 204, 113, 0.08)"},
        ],
    )

    return metrics_cards, fig_disp, fig_force, fig_cost, fig_delta, conv_table


@app.callback(
    Output("convergence-download", "data"),
    Input("convergence-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_convergence_csv(n_clicks):
    df_all, _ = load_convergence_enriched()
    if df_all.empty:
        return dash.no_update
    return dcc.send_data_frame(df_all.sort_values("dp", ascending=False).to_csv, "convergence_report.csv", index=False)


if __name__ == "__main__":
    print("=" * 64)
    print(APP_TITLE)
    print(f"DB:   {DB_PATH}")
    print(f"Conv: {CONV_CSV_PATH}")
    print(f"GP:   {SURROGATE_PATH}")
    print("URL:  http://127.0.0.1:8050")
    print("=" * 64)
    app.run(debug=True, port=8050)
