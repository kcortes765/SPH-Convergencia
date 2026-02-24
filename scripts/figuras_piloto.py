"""
figuras_piloto.py — Figuras completas del estudio piloto para la tesis

Genera TODAS las figuras necesarias a partir de:
  1. data/results.sqlite — resultados de 35 simulaciones
  2. data/gp_surrogate.pkl — GP entrenado
  3. data/figuras_uq/uq_summary.json — resultados UQ

Ejecutar DESPUES de:
  python src/ml_surrogate.py            # entrena GP
  python scripts/run_uq_analysis.py     # genera UQ

Ejecutar:
    python scripts/figuras_piloto.py
    python scripts/figuras_piloto.py --synthetic  # con datos fake (testing)

Output: data/figuras_piloto/ (PNG 300dpi + PDF vector, espanol)

Autor: Kevin Cortes (UCN 2026)
"""

import argparse
import json
import logging
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from itertools import combinations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "results.sqlite"
MODEL_PATH = PROJECT_ROOT / "data" / "gp_surrogate.pkl"
UQ_PATH = PROJECT_ROOT / "data" / "figuras_uq" / "uq_summary.json"
OUT = PROJECT_ROOT / "data" / "figuras_piloto"

# Estilo premium para tesis (hereda de figuras convergencia)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9.5,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.4,
    'axes.linewidth': 0.7,
    'lines.linewidth': 1.6,
    'lines.markersize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Paleta de colores
B = '#2166AC'    # Azul
R = '#B2182B'    # Rojo
G = '#2CA02C'    # Verde
O = '#FF7F00'    # Naranja
P = '#6A3D9A'    # Purpura
GR = '#555555'   # Gris
TEAL = '#00897B' # Teal

FEAT_LABELS = {
    'dam_height': 'Altura columna $h$ [m]',
    'boulder_mass': 'Masa boulder $M$ [kg]',
    'boulder_rot_z': r'Orientacion $\theta_z$ [grados]',
}

FEAT_COLORS = {
    'dam_height': B,
    'boulder_mass': R,
    'boulder_rot_z': P,
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_data(synthetic: bool = False) -> pd.DataFrame:
    """Carga resultados de SQLite o genera sinteticos."""
    if synthetic or not DB_PATH.exists():
        print("Usando datos sinteticos...")
        rng = np.random.default_rng(42)
        n = 100
        h = rng.uniform(0.10, 0.50, n)
        m = rng.uniform(0.80, 1.60, n)
        theta = rng.uniform(0, 90, n)
        energy = 1000 * 9.81 * h**2 * 0.5
        resistance = m * 9.81 * 0.6
        disp = 2.0 * (energy / resistance)**0.7
        disp *= (1.0 - 0.1 * np.sin(np.radians(theta)))
        disp += rng.normal(0, 0.15, n)
        disp = np.maximum(disp, 0)
        return pd.DataFrame({
            'case_name': [f"lhs_{i+1:03d}" for i in range(n)],
            'dam_height': h,
            'boulder_mass': m,
            'boulder_rot_z': theta,
            'max_displacement': disp,
            'max_rotation': rng.uniform(10, 95, n),
            'max_velocity': 0.5 + 1.5 * (h / 0.5)**0.8 + rng.normal(0, 0.1, n),
            'max_sph_force': 10 + 50 * h**2 / m + rng.normal(0, 3, n),
            'max_contact_force': rng.uniform(50, 5000, n),
            'max_flow_velocity': 0.3 + 1.2 * h + rng.normal(0, 0.05, n),
            'max_water_height': h * 0.5 + rng.normal(0, 0.01, n),
            'moved': (disp > 0.005).astype(int),
            'failed': (disp > 0.005).astype(int),
        })

    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("""
        SELECT * FROM results
        WHERE dam_height > 0 AND boulder_mass > 0
    """, conn)
    conn.close()
    if 'boulder_rot_z' not in df.columns:
        df['boulder_rot_z'] = 0.0
    else:
        df['boulder_rot_z'] = df['boulder_rot_z'].fillna(0.0)
    print(f"Datos reales: {len(df)} casos")
    return df


def load_gp():
    """Carga GP entrenado."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_gp(pkg, X):
    X_s = pkg['scaler_X'].transform(X)
    y_s, std_s = pkg['gp'].predict(X_s, return_std=True)
    y_mean = pkg['scaler_y'].inverse_transform(y_s.reshape(-1, 1)).ravel()
    y_std = std_s * pkg['scaler_y'].scale_[0]
    return y_mean, y_std


# ═══════════════════════════════════════════════════════════════════════
# FIGURAS — DATOS CRUDOS (antes de GP)
# ═══════════════════════════════════════════════════════════════════════

def fig01_lhs_cobertura(df):
    """Fig 01: Cobertura del espacio de parametros LHS (3 pares)."""
    feats = ['dam_height', 'boulder_mass', 'boulder_rot_z']
    pairs = list(combinations(range(3), 2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (i, j) in zip(axes, pairs):
        fi, fj = feats[i], feats[j]
        sc = ax.scatter(df[fi], df[fj], c=df['max_displacement'],
                        cmap='RdYlGn_r', s=50, edgecolors='white',
                        linewidths=0.5, zorder=5)
        ax.set_xlabel(FEAT_LABELS[fi])
        ax.set_ylabel(FEAT_LABELS[fj])

    plt.colorbar(sc, ax=axes[-1], label='Desplaz. max [m]', shrink=0.85)
    fig.suptitle(f'Cobertura LHS: {len(df)} casos en espacio 3D',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig01_lhs_cobertura')


def fig02_distribucion_resultados(df):
    """Fig 02: Histogramas de las 4 metricas principales."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    metrics = [
        ('max_displacement', 'Desplazamiento max [m]', B),
        ('max_rotation', 'Rotacion max [grados]', P),
        ('max_velocity', 'Velocidad max [m/s]', TEAL),
        ('max_sph_force', 'Fuerza SPH max [N]', R),
    ]

    for ax, (col, label, color) in zip(axes.flat, metrics):
        vals = df[col].dropna()
        ax.hist(vals, bins=12, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(vals.mean(), color='black', lw=1.5, ls='--',
                   label=f'Media = {vals.mean():.3f}')
        ax.axvline(vals.median(), color=O, lw=1.5, ls=':',
                   label=f'Mediana = {vals.median():.3f}')
        ax.set_xlabel(label)
        ax.set_ylabel('Frecuencia')
        ax.legend(fontsize=7)

    fig.suptitle('Distribucion de metricas del estudio piloto',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig02_distribucion_resultados')


def fig03_scatter_matrix(df):
    """Fig 03: Desplazamiento vs cada parametro de entrada."""
    feats = ['dam_height', 'boulder_mass', 'boulder_rot_z']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, feat in zip(axes, feats):
        color = FEAT_COLORS[feat]
        ax.scatter(df[feat], df['max_displacement'], c=color,
                   s=40, edgecolors='white', linewidths=0.5, alpha=0.8)

        # Tendencia lineal
        z = np.polyfit(df[feat], df['max_displacement'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(x_line, p(x_line), color=color, lw=1.5, ls='--', alpha=0.6)

        # Correlacion
        corr = df[feat].corr(df['max_displacement'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.8))

        ax.set_xlabel(FEAT_LABELS[feat])
        ax.set_ylabel('Desplazamiento max [m]')

    fig.suptitle('Correlacion parametros → desplazamiento',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig03_correlacion_parametros')


def fig04_clasificacion_movimiento(df):
    """Fig 04: Clasificacion movimiento/estable con colores."""
    feats = ['dam_height', 'boulder_mass', 'boulder_rot_z']
    pairs = list(combinations(range(3), 2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    moved = df['failed'].astype(bool) if 'failed' in df else df['moved'].astype(bool)

    for ax, (i, j) in zip(axes, pairs):
        fi, fj = feats[i], feats[j]
        ax.scatter(df.loc[~moved, fi], df.loc[~moved, fj],
                   c=G, s=50, edgecolors='white', linewidths=0.5,
                   label='Estable', zorder=5, marker='o')
        ax.scatter(df.loc[moved, fi], df.loc[moved, fj],
                   c=R, s=50, edgecolors='white', linewidths=0.5,
                   label='Movimiento', zorder=5, marker='^')
        ax.set_xlabel(FEAT_LABELS[fi])
        ax.set_ylabel(FEAT_LABELS[fj])
        ax.legend(fontsize=7)

    n_moved = moved.sum()
    n_stable = (~moved).sum()
    fig.suptitle(f'Clasificacion: {n_moved} movimiento, {n_stable} estables',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig04_clasificacion_movimiento')


def fig05_fuerzas_vs_parametros(df):
    """Fig 05: Fuerzas SPH y contacto vs parametros."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    feats = ['dam_height', 'boulder_mass', 'boulder_rot_z']

    for j, feat in enumerate(feats):
        # SPH force
        ax = axes[0, j]
        ax.scatter(df[feat], df['max_sph_force'], c=B, s=30,
                   edgecolors='white', linewidths=0.5, alpha=0.8)
        ax.set_xlabel(FEAT_LABELS[feat])
        ax.set_ylabel('Fuerza SPH max [N]')

        # Contact force
        ax = axes[1, j]
        ax.scatter(df[feat], df['max_contact_force'], c=R, s=30,
                   edgecolors='white', linewidths=0.5, alpha=0.8)
        ax.set_xlabel(FEAT_LABELS[feat])
        ax.set_ylabel('Fuerza contacto max [N]')
        ax.set_yscale('log')

    axes[0, 0].set_title('Fuerza SPH', fontweight='bold')
    axes[1, 0].set_title('Fuerza de contacto', fontweight='bold')
    fig.suptitle('Fuerzas sobre el boulder vs parametros de entrada',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig05_fuerzas_vs_parametros')


def fig06_flujo_vs_desplazamiento(df):
    """Fig 06: Velocidad del flujo y altura de agua vs desplazamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    if 'max_flow_velocity' in df.columns:
        ax1.scatter(df['max_flow_velocity'], df['max_displacement'],
                    c=TEAL, s=40, edgecolors='white', linewidths=0.5)
        ax1.set_xlabel('Velocidad max del flujo [m/s]')
        ax1.set_ylabel('Desplazamiento max [m]')
        ax1.set_title('(a) Velocidad del flujo')

    if 'max_water_height' in df.columns:
        ax2.scatter(df['max_water_height'], df['max_displacement'],
                    c=B, s=40, edgecolors='white', linewidths=0.5)
        ax2.set_xlabel('Altura max del agua [m]')
        ax2.set_ylabel('Desplazamiento max [m]')
        ax2.set_title('(b) Altura del agua')

    fig.suptitle('Condiciones del flujo vs respuesta del boulder',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig06_flujo_vs_desplazamiento')


def fig07_tabla_resumen(df):
    """Fig 07: Tabla resumen estadistico + casos extremos (adaptada a N>=100)."""
    n = len(df)
    cols = ['case_name', 'dam_height', 'boulder_mass', 'boulder_rot_z',
            'max_displacement', 'max_rotation', 'max_sph_force', 'failed']
    col_labels = ['Caso', 'h [m]', 'M [kg]', r'$\theta_z$ [°]',
                  'Desplaz [m]', 'Rot [°]', 'F_SPH [N]', 'Estado']

    # Ordenar por desplazamiento descendente
    df_sorted = df.sort_values('max_displacement', ascending=False).reset_index(drop=True)

    # Estadisticas globales
    n_moved = df['failed'].sum() if 'failed' in df else 0
    n_stable = n - n_moved
    stats_rows = [
        ['ESTADISTICAS', f'N={n}', f'MOV={n_moved}', f'EST={n_stable}',
         f'{df["max_displacement"].mean():.4f}',
         f'{df.get("max_rotation", pd.Series([0])).mean():.1f}',
         f'{df.get("max_sph_force", pd.Series([0])).mean():.1f}', '---'],
        ['', 'Media', 'Mediana', 'Std',
         f'{df["max_displacement"].median():.4f}',
         f'{df.get("max_rotation", pd.Series([0])).median():.1f}',
         f'{df.get("max_sph_force", pd.Series([0])).median():.1f}', '---'],
    ]

    # Top 10 mas desplazados + bottom 5 mas estables
    n_top = min(10, n)
    n_bot = min(5, n)
    top_df = df_sorted.head(n_top)
    bot_df = df_sorted.tail(n_bot)

    def _make_rows(sub_df):
        rows = []
        colors = []
        for _, row in sub_df.iterrows():
            failed = bool(row.get('failed', False))
            status = 'MOV' if failed else 'EST'
            bg = '#FFCDD2' if failed else '#C8E6C9'
            rows.append([
                row['case_name'],
                f"{row['dam_height']:.3f}",
                f"{row['boulder_mass']:.2f}",
                f"{row.get('boulder_rot_z', 0):.1f}",
                f"{row['max_displacement']:.4f}",
                f"{row.get('max_rotation', 0):.1f}",
                f"{row.get('max_sph_force', 0):.1f}",
                status,
            ])
            colors.append([bg] * len(cols))
        return rows, colors

    top_rows, top_colors = _make_rows(top_df)
    bot_rows, bot_colors = _make_rows(bot_df)
    sep_row = [['...'] * len(cols)]
    sep_color = [['#E0E0E0'] * len(cols)]

    # Combinar
    all_rows = stats_rows + top_rows + sep_row + bot_rows
    stats_colors = [['#E3F2FD'] * len(cols)] * 2
    all_colors = stats_colors + top_colors + sep_color + bot_colors

    n_rows = len(all_rows)
    fig, ax = plt.subplots(figsize=(14, max(5, 0.42 * n_rows + 2)))
    ax.axis('off')

    table = ax.table(cellText=all_rows, colLabels=col_labels,
                     cellColours=all_colors, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.4)

    for j in range(len(cols)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    fig.suptitle(f'Resumen: {n} casos (top {n_top} + bottom {n_bot}, dp=0.004)',
                 fontsize=13, fontweight='bold', y=0.98)
    save(fig, 'fig07_tabla_resumen')


# ═══════════════════════════════════════════════════════════════════════
# FIGURAS — GP SURROGATE (post-entrenamiento)
# ═══════════════════════════════════════════════════════════════════════

def fig08_gp_superficie_3pares(pkg, df):
    """Fig 08: Superficie GP con prediccion + incertidumbre (6 paneles)."""
    if pkg is None:
        print("  [SKIP] fig08 — no hay GP entrenado")
        return

    feats = pkg['features']
    pairs = list(combinations(range(len(feats)), 2))
    medians = np.array([df[f].median() for f in feats])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for col_idx, (i, j) in enumerate(pairs):
        fi, fj = feats[i], feats[j]
        xi = np.linspace(df[fi].min(), df[fi].max(), 60)
        xj = np.linspace(df[fj].min(), df[fj].max(), 60)
        Xi, Xj = np.meshgrid(xi, xj)

        X_grid = np.tile(medians, (Xi.size, 1))
        X_grid[:, i] = Xi.ravel()
        X_grid[:, j] = Xj.ravel()

        y_mean, y_std = predict_gp(pkg, X_grid)
        Z_mean = y_mean.reshape(Xi.shape)
        Z_std = y_std.reshape(Xi.shape)

        # Prediccion
        ax = axes[0, col_idx]
        cf = ax.contourf(Xi, Xj, Z_mean, levels=20, cmap='RdYlGn_r')
        ax.scatter(df[fi], df[fj], c='black', s=20, edgecolors='white',
                   linewidths=0.5, zorder=5)
        plt.colorbar(cf, ax=ax, label='Desplaz. [m]', shrink=0.85)
        ax.set_xlabel(FEAT_LABELS[fi])
        ax.set_ylabel(FEAT_LABELS[fj])
        others = [k for k in range(len(feats)) if k not in (i, j)]
        fixed = ", ".join(f"{feats[k]}={medians[k]:.1f}" for k in others)
        ax.set_title(f'Prediccion ({fixed})', fontsize=9)

        # Incertidumbre
        ax = axes[1, col_idx]
        cf2 = ax.contourf(Xi, Xj, Z_std * 2, levels=20, cmap='Oranges')
        ax.scatter(df[fi], df[fj], c='black', s=20, edgecolors='white',
                   linewidths=0.5, zorder=5)
        plt.colorbar(cf2, ax=ax, label='$2\\sigma$ [m]', shrink=0.85)
        ax.set_xlabel(FEAT_LABELS[fi])
        ax.set_ylabel(FEAT_LABELS[fj])
        ax.set_title(f'Incertidumbre ({fixed})', fontsize=9)

    fig.suptitle('GP Surrogate: prediccion e incertidumbre',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig08_gp_superficie_completa')


def fig09_gp_slices_detallados(pkg, df):
    """Fig 09: Cortes 1D detallados con datos y CI."""
    if pkg is None:
        print("  [SKIP] fig09 — no hay GP entrenado")
        return

    feats = pkg['features']
    medians = np.array([df[f].median() for f in feats])

    fig, axes = plt.subplots(1, len(feats), figsize=(5.5 * len(feats), 5))
    if len(feats) == 1:
        axes = [axes]

    for idx, (ax, feat) in enumerate(zip(axes, feats)):
        lo, hi = df[feat].min(), df[feat].max()
        margin = (hi - lo) * 0.1
        x_line = np.linspace(lo - margin, hi + margin, 150)

        X_cut = np.tile(medians, (150, 1))
        X_cut[:, idx] = x_line
        y_mean, y_std = predict_gp(pkg, X_cut)

        color = FEAT_COLORS.get(feat, GR)

        # GP mean + CI
        ax.plot(x_line, y_mean, color=color, lw=2.5, label='GP media')
        ax.fill_between(x_line, y_mean - 2*y_std, y_mean + 2*y_std,
                        alpha=0.15, color=color, label='IC 95%')
        ax.fill_between(x_line, y_mean - y_std, y_mean + y_std,
                        alpha=0.15, color=color, label='IC 68%')

        # Datos reales
        ax.scatter(df[feat], df['max_displacement'], c='black', s=30,
                   edgecolors='white', linewidths=0.5, zorder=5,
                   label=f'SPH ({len(df)} casos)')

        ax.set_xlabel(FEAT_LABELS[feat])
        ax.set_ylabel('Desplazamiento max [m]')
        others_str = ", ".join(
            f"{feats[k]}={medians[k]:.1f}" for k in range(len(feats)) if k != idx
        )
        ax.set_title(f'Fijando {others_str}', fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle('Cortes 1D del GP Surrogate',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig09_gp_slices_detallados')


def fig10_gp_validacion_loo(pkg, df):
    """Fig 10: LOO validation completa (3 paneles)."""
    if pkg is None:
        print("  [SKIP] fig10 — no hay GP entrenado")
        return

    from sklearn.gaussian_process import GaussianProcessRegressor

    feats = pkg['features']
    X = df[feats].values
    y = df['max_displacement'].values

    X_s = pkg['scaler_X'].transform(X)
    y_s = pkg['scaler_y'].transform(y.reshape(-1, 1)).ravel()

    n = len(y)
    y_pred = np.zeros(n)
    y_std = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        gp_loo = GaussianProcessRegressor(kernel=pkg['gp'].kernel_, optimizer=None)
        gp_loo.fit(X_s[mask], y_s[mask])
        pred, std = gp_loo.predict(X_s[i:i+1], return_std=True)
        y_pred[i] = pred[0]
        y_std[i] = std[0]

    y_pred_real = pkg['scaler_y'].inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_std_real = y_std * pkg['scaler_y'].scale_[0]
    residuals = y - y_pred_real
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Predicted vs Actual
    for i in range(n):
        ax1.errorbar(y[i], y_pred_real[i], yerr=2*y_std_real[i],
                     fmt='o', markersize=5, color=B,
                     ecolor='#BBDEFB', elinewidth=1, capsize=2, alpha=0.8)
    lims = [min(y.min(), y_pred_real.min()) - 0.2,
            max(y.max(), y_pred_real.max()) + 0.2]
    ax1.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='Perfecto')
    ax1.set_xlabel('Desplazamiento real [m]')
    ax1.set_ylabel('Desplazamiento predicho [m]')
    ax1.set_title(f'(a) LOO: R$^2$ = {r2:.4f}')
    ax1.legend(fontsize=8)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_aspect('equal')

    # Panel 2: Residuos
    bar_colors = [R if r > 0 else B for r in residuals]
    ax2.bar(range(n), residuals, color=bar_colors, alpha=0.7, edgecolor='none')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xlabel('Indice de caso')
    ax2.set_ylabel('Residuo [m]')
    ax2.set_title(f'(b) Residuos (RMSE = {rmse:.4f} m)')

    # Panel 3: Distribucion de residuos
    ax3.hist(residuals, bins=12, color=P, alpha=0.7, edgecolor='white',
             density=True)
    ax3.axvline(0, color='black', lw=1, ls='--')
    ax3.set_xlabel('Residuo [m]')
    ax3.set_ylabel('Densidad')
    ax3.set_title('(c) Distribucion de residuos')

    fig.suptitle('Validacion Leave-One-Out del GP Surrogate',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig10_gp_validacion_loo')


# ═══════════════════════════════════════════════════════════════════════
# FIGURAS — UQ (post Monte Carlo + Sobol)
# ═══════════════════════════════════════════════════════════════════════

def fig11_mc_completo(pkg, df):
    """Fig 11: MC completo — histograma + violin + CDF."""
    if pkg is None:
        print("  [SKIP] fig11 — no hay GP entrenado")
        return

    feats = pkg['features']
    rng = np.random.default_rng(42)
    n_mc = 10000

    X_mc = np.zeros((n_mc, len(feats)))
    for i, feat in enumerate(feats):
        lo, hi = df[feat].min(), df[feat].max()
        X_mc[:, i] = rng.uniform(lo, hi, n_mc)

    y_mean, y_std = predict_gp(pkg, X_mc)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Histograma
    ax1.hist(y_mean, bins=50, color=B, alpha=0.7, edgecolor='white',
             density=True)
    ci_lo, ci_hi = np.percentile(y_mean, [2.5, 97.5])
    ax1.axvline(np.mean(y_mean), color=R, lw=2, ls='-',
                label=f'Media = {np.mean(y_mean):.3f} m')
    ax1.axvline(ci_lo, color=O, lw=1.5, ls='--',
                label=f'CI 95% = [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax1.axvline(ci_hi, color=O, lw=1.5, ls='--')
    ax1.set_xlabel('Desplazamiento [m]')
    ax1.set_ylabel('Densidad')
    ax1.set_title('(a) Distribucion MC')
    ax1.legend(fontsize=7)

    # Violin por parametro (binned)
    feat_main = feats[0]  # dam_height
    bins = np.linspace(df[feat_main].min(), df[feat_main].max(), 6)
    bin_data = []
    bin_labels = []
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        mask = (X_mc[:, 0] >= lo_b) & (X_mc[:, 0] < hi_b)
        if mask.sum() > 10:
            bin_data.append(y_mean[mask])
            bin_labels.append(f'{(lo_b+hi_b)/2:.2f}')

    if bin_data:
        vp = ax2.violinplot(bin_data, showmeans=True, showmedians=True)
        for body in vp['bodies']:
            body.set_facecolor(B)
            body.set_alpha(0.5)
        ax2.set_xticks(range(1, len(bin_labels) + 1))
        ax2.set_xticklabels(bin_labels)
        ax2.set_xlabel(FEAT_LABELS[feat_main])
        ax2.set_ylabel('Desplazamiento [m]')
        ax2.set_title('(b) Distribucion por rango de $h$')

    # CDF
    sorted_y = np.sort(y_mean)
    cdf = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    ax3.plot(sorted_y, cdf, color=B, lw=2)
    ax3.axhline(0.5, color=GR, lw=0.8, ls=':', alpha=0.5)
    ax3.axhline(0.95, color=R, lw=0.8, ls=':', alpha=0.5)
    ax3.set_xlabel('Desplazamiento [m]')
    ax3.set_ylabel('Probabilidad acumulada')
    ax3.set_title('(c) CDF')

    fig.suptitle(f'Monte Carlo: {n_mc:,} muestras sobre GP Surrogate',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig11_mc_completo')


def fig12_sobol_completo(pkg, df):
    """Fig 12: Sobol indices — tornado + pie + convergencia."""
    if pkg is None:
        print("  [SKIP] fig12 — no hay GP entrenado")
        return

    # Calcular Sobol internamente
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from run_uq_analysis import sobol_indices, load_param_ranges

    feats = pkg['features']
    param_ranges = {f: (df[f].min(), df[f].max()) for f in feats}

    sobol = sobol_indices(pkg, param_ranges, n_base=4096, seed=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Tornado
    y_pos = np.arange(len(feats))
    labels = [FEAT_LABELS.get(f, f) for f in feats]
    s1_vals = [sobol['S1'][f] for f in feats]
    st_vals = [sobol['ST'][f] for f in feats]

    bar_h = 0.35
    ax1.barh(y_pos + bar_h/2, s1_vals, bar_h, color=B, alpha=0.8,
             label='$S_1$ (primer orden)')
    ax1.barh(y_pos - bar_h/2, st_vals, bar_h, color=R, alpha=0.8,
             label='$S_T$ (orden total)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Indice de Sobol')
    ax1.set_title('(a) Sensibilidad global')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.axvline(0, color='black', lw=0.5)

    # Pie chart de ST
    st_clean = [max(0, sobol['ST'][f]) for f in feats]
    total_st = sum(st_clean)
    if total_st > 0:
        pct = [s / total_st * 100 for s in st_clean]
        colors_pie = [FEAT_COLORS.get(f, GR) for f in feats]
        wedges, texts, autotexts = ax2.pie(
            pct, labels=labels, autopct='%1.1f%%',
            colors=colors_pie, startangle=90,
            textprops={'fontsize': 8})
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        ax2.set_title('(b) Contribucion relativa ($S_T$)')

    fig.suptitle('Analisis de Sensibilidad Global (Sobol)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig12_sobol_completo')


def fig13_frontera_detallada(pkg, df):
    """Fig 13: Frontera de estabilidad probabilistica (alta resolucion)."""
    if pkg is None:
        print("  [SKIP] fig13 — no hay GP entrenado")
        return

    from scipy.stats import norm

    feats = pkg['features']
    if len(feats) < 2:
        return

    f0, f1 = feats[0], feats[1]
    medians = np.array([df[f].median() for f in feats])

    x0 = np.linspace(df[f0].min(), df[f0].max(), 80)
    x1 = np.linspace(df[f1].min(), df[f1].max(), 80)
    X0, X1 = np.meshgrid(x0, x1)

    X_grid = np.tile(medians, (X0.size, 1))
    X_grid[:, 0] = X0.ravel()
    X_grid[:, 1] = X1.ravel()

    y_mean, y_std = predict_gp(pkg, X_grid)

    threshold = 0.005
    p_motion = np.array([
        1.0 - norm.cdf(threshold, loc=m, scale=max(s, 1e-10))
        for m, s in zip(y_mean, y_std)
    ])
    P = p_motion.reshape(X0.shape)

    fig, ax = plt.subplots(figsize=(8, 6))

    cf = ax.contourf(X0, X1, P, levels=np.linspace(0, 1, 21),
                     cmap='RdYlGn_r')
    plt.colorbar(cf, ax=ax, label='P(movimiento incipiente)')

    # Contornos clave
    for level, color, lw, ls in [(0.10, G, 1.5, ':'),
                                  (0.50, 'black', 2.5, '-'),
                                  (0.90, R, 1.5, ':'),
                                  (0.95, R, 2, '--')]:
        try:
            cs = ax.contour(X0, X1, P, levels=[level], colors=color,
                            linewidths=lw, linestyles=ls)
            ax.clabel(cs, fmt=f'P={level:.0%}', fontsize=8)
        except ValueError:
            pass

    # Datos reales
    moved = df['failed'].astype(bool) if 'failed' in df else df['moved'].astype(bool)
    ax.scatter(df.loc[~moved, f0], df.loc[~moved, f1],
               c='white', s=40, edgecolors=G, linewidths=1.5,
               zorder=5, label='Estable')
    ax.scatter(df.loc[moved, f0], df.loc[moved, f1],
               c='white', s=40, edgecolors=R, linewidths=1.5,
               marker='^', zorder=5, label='Movimiento')

    ax.set_xlabel(FEAT_LABELS[f0])
    ax.set_ylabel(FEAT_LABELS[f1])
    fixed = ", ".join(f"{feats[k]}={medians[k]:.1f}" for k in range(len(feats)) if k > 1)
    title = f'Frontera de estabilidad probabilistica'
    if fixed:
        title += f' ({fixed})'
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    save(fig, 'fig13_frontera_probabilidad')


def fig14_resumen_ejecutivo(df, pkg):
    """Fig 14: Tarjeta resumen visual (monospace verdict)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    n = len(df)
    n_moved = df['failed'].sum() if 'failed' in df else df['moved'].sum()
    n_stable = n - n_moved

    lines = []
    lines.append("=" * 58)
    lines.append("  ESTUDIO PILOTO — RESUMEN EJECUTIVO")
    lines.append("=" * 58)
    lines.append(f"  Casos totales:     {n}")
    lines.append(f"  Movimiento:        {n_moved} ({n_moved/n*100:.0f}%)")
    lines.append(f"  Estables:          {n_stable} ({n_stable/n*100:.0f}%)")
    lines.append(f"  dp produccion:     0.004 m")
    lines.append("-" * 58)
    lines.append(f"  Desplaz. medio:    {df['max_displacement'].mean():.4f} m")
    lines.append(f"  Desplaz. max:      {df['max_displacement'].max():.4f} m")
    lines.append(f"  Rotacion media:    {df['max_rotation'].mean():.1f} deg")
    lines.append(f"  F_SPH media:       {df['max_sph_force'].mean():.1f} N")
    lines.append("-" * 58)

    if pkg is not None:
        lines.append(f"  GP R² (LOO):       {pkg.get('loo_r2', 0):.4f}")
        lines.append(f"  GP RMSE:           {pkg.get('loo_rmse', 0):.4f} m")
        lines.append(f"  Features:          {pkg.get('features', [])}")

    if UQ_PATH.exists():
        with open(UQ_PATH) as f:
            uq = json.load(f)
        mc = uq.get('monte_carlo', {})
        sob = uq.get('sobol', {})
        lines.append("-" * 58)
        lines.append(f"  MC muestras:       {mc.get('n_samples', 0):,}")
        lines.append(f"  P(movimiento):     {mc.get('prob_motion', 0):.1%}")
        ci = mc.get('ci_95_m', [0, 0])
        lines.append(f"  CI 95%:            [{ci[0]:.3f}, {ci[1]:.3f}] m")
        st = sob.get('ST', {})
        if st:
            lines.append("-" * 58)
            lines.append("  SOBOL (orden total):")
            for feat, val in st.items():
                label = FEAT_LABELS.get(feat, feat)
                bar = '#' * int(float(val) * 40)
                lines.append(f"    {label:28s} {float(val):.3f} {bar}")

    lines.append("=" * 58)

    text = "\n".join(lines)
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9.5, va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.8', fc='#FAFAFA', ec=B, lw=2))

    save(fig, 'fig14_resumen_ejecutivo')


# ═══════════════════════════════════════════════════════════════════════
# FIGURAS — ANALISIS ESTADISTICO AVANZADO (8 figuras nuevas)
# ═══════════════════════════════════════════════════════════════════════


def fig15_heatmap_parametrico(df):
    """Fig 15: Mapa parametrico — tamano proporcional al desplazamiento."""
    feats = ['dam_height', 'boulder_mass', 'boulder_rot_z']
    pairs = list(combinations(range(3), 2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (i, j) in zip(axes, pairs):
        fi, fj = feats[i], feats[j]
        disp = df['max_displacement']
        dmax = disp.max() if disp.max() > 0 else 1
        sizes = 30 + 250 * (disp / dmax)
        sc = ax.scatter(df[fi], df[fj], c=disp, cmap='RdYlGn_r',
                        s=sizes, edgecolors='white', linewidths=0.5,
                        alpha=0.85, zorder=5)
        ax.set_xlabel(FEAT_LABELS[fi])
        ax.set_ylabel(FEAT_LABELS[fj])

    plt.colorbar(sc, ax=axes[-1], label='Desplazamiento max [m]', shrink=0.85)
    fig.suptitle(f'Mapa parametrico: {len(df)} simulaciones (tamano ~ desplazamiento)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig15_heatmap_parametrico')


def fig16_boxplots_por_parametro(df):
    """Fig 16: Boxplots de desplazamiento por quintiles de cada parametro."""
    feats = ['dam_height', 'boulder_mass', 'boulder_rot_z']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for ax, feat in zip(axes, feats):
        n_bins = 5
        bins = np.linspace(df[feat].min() - 1e-9, df[feat].max() + 1e-9, n_bins + 1)
        labels_bin = []
        data_groups = []
        for k in range(n_bins):
            mask = (df[feat] >= bins[k]) & (df[feat] < bins[k + 1])
            vals = df.loc[mask, 'max_displacement'].values
            if len(vals) > 0:
                data_groups.append(vals)
                mid = (bins[k] + bins[k + 1]) / 2
                labels_bin.append(f'{mid:.2f}')

        if not data_groups:
            continue

        color = FEAT_COLORS[feat]
        bp = ax.boxplot(data_groups, labels=labels_bin, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        # Overlay puntos individuales con jitter
        jitter_rng = np.random.default_rng(42)
        for k, vals in enumerate(data_groups):
            x_jitter = jitter_rng.normal(k + 1, 0.08, len(vals))
            ax.scatter(x_jitter, vals, c=color, s=15, alpha=0.4, zorder=5)

        ax.set_xlabel(FEAT_LABELS[feat])
        ax.set_ylabel('Desplazamiento max [m]')

    fig.suptitle(f'Distribucion del desplazamiento por rango de parametros ({len(df)} casos)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig16_boxplots_por_parametro')


def fig17_matriz_correlacion(df):
    """Fig 17: Heatmap de correlacion entre todas las variables."""
    cols_order = ['dam_height', 'boulder_mass', 'boulder_rot_z',
                  'max_displacement', 'max_rotation', 'max_velocity',
                  'max_sph_force', 'max_contact_force',
                  'max_flow_velocity', 'max_water_height']
    labels_map = {
        'dam_height': '$h$ [m]',
        'boulder_mass': '$M$ [kg]',
        'boulder_rot_z': r'$\theta_z$ [deg]',
        'max_displacement': 'Desplaz.',
        'max_rotation': 'Rotacion',
        'max_velocity': 'Velocidad',
        'max_sph_force': '$F_{SPH}$',
        'max_contact_force': '$F_{cont}$',
        'max_flow_velocity': '$V_{flujo}$',
        'max_water_height': '$H_{agua}$',
    }

    avail = [c for c in cols_order if c in df.columns and df[c].notna().sum() > 2]
    if len(avail) < 3:
        print("  [SKIP] fig17 — insuficientes columnas")
        return

    corr = df[avail].corr()
    nc = len(avail)

    fig, ax = plt.subplots(figsize=(max(6, nc * 0.9), max(5, nc * 0.8)))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    labels_show = [labels_map.get(c, c) for c in avail]
    ax.set_xticks(range(nc))
    ax.set_yticks(range(nc))
    ax.set_xticklabels(labels_show, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels_show, fontsize=8)

    for i_row in range(nc):
        for j_col in range(nc):
            val = corr.values[i_row, j_col]
            color = 'white' if abs(val) > 0.55 else 'black'
            fw = 'bold' if abs(val) > 0.7 else 'normal'
            ax.text(j_col, i_row, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=7, fontweight=fw)

    # Linea divisoria entradas / salidas
    n_inputs = sum(1 for c in avail if c in ('dam_height', 'boulder_mass', 'boulder_rot_z'))
    if 0 < n_inputs < nc:
        ax.axhline(n_inputs - 0.5, color='black', lw=1.5)
        ax.axvline(n_inputs - 0.5, color='black', lw=1.5)

    plt.colorbar(im, ax=ax, label='Coeficiente de Pearson', shrink=0.85)
    ax.set_title('Matriz de correlacion: parametros de entrada y metricas de salida',
                 fontsize=11, fontweight='bold', pad=15)
    fig.tight_layout()
    save(fig, 'fig17_matriz_correlacion')


def fig18_violin_por_regimen(df):
    """Fig 18: Violines comparando metricas entre estable vs movimiento."""
    moved = df['failed'].astype(bool) if 'failed' in df else df.get('moved', pd.Series(dtype=bool)).astype(bool)
    if moved.sum() == 0 or (~moved).sum() == 0:
        print("  [SKIP] fig18 — todos los casos son del mismo regimen")
        return

    metrics = [
        ('max_displacement', 'Desplazamiento [m]'),
        ('max_rotation', 'Rotacion max [deg]'),
        ('max_velocity', 'Velocidad max [m/s]'),
        ('max_sph_force', 'Fuerza SPH max [N]'),
    ]
    avail = [(m, l) for m, l in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(avail), figsize=(4.5 * len(avail), 5))
    if len(avail) == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, avail):
        d_stable = df.loc[~moved, metric].dropna().values
        d_moved = df.loc[moved, metric].dropna().values

        parts = ax.violinplot([d_stable, d_moved], showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor(G)
        parts['bodies'][0].set_alpha(0.5)
        if len(parts['bodies']) > 1:
            parts['bodies'][1].set_facecolor(R)
            parts['bodies'][1].set_alpha(0.5)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Estable\n(n={len(d_stable)})',
                            f'Movimiento\n(n={len(d_moved)})'])
        ax.set_ylabel(label)

    fig.suptitle(f'Comparacion por regimen: {(~moved).sum()} estables vs {moved.sum()} movimiento',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig18_violin_por_regimen')


def fig19_energia_adimensional(df):
    """Fig 19: Analisis adimensional — numero de estabilidad y densidad."""
    rho_w = 1000.0
    g = 9.81
    mu = 0.6
    V_boulder = 0.00053023  # m3 (BLIR3.stl a escala 0.04)

    E_wave = 0.5 * rho_w * g * df['dam_height'] ** 2
    R_grav = df['boulder_mass'] * g * mu
    stab_num = E_wave / R_grav
    density = df['boulder_mass'] / V_boulder

    moved = df['failed'].astype(bool) if 'failed' in df else df['moved'].astype(bool)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Numero de estabilidad vs desplazamiento
    ax = axes[0]
    ax.scatter(stab_num[~moved], df.loc[~moved, 'max_displacement'],
               c=G, s=50, edgecolors='white', linewidths=0.5, label='Estable')
    ax.scatter(stab_num[moved], df.loc[moved, 'max_displacement'],
               c=R, s=50, edgecolors='white', linewidths=0.5,
               label='Movimiento', marker='^')
    ax.set_xlabel(r'$\Pi = \frac{\rho_w g h^2}{2 M g \mu}$ [-]')
    ax.set_ylabel('Desplazamiento max [m]')
    ax.set_title('(a) Parametro de estabilidad')
    ax.legend(fontsize=8)

    # (b) Densidad boulder vs desplazamiento (color = h)
    ax = axes[1]
    sc = ax.scatter(density, df['max_displacement'], c=df['dam_height'],
                    cmap='coolwarm', s=50, edgecolors='white', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='$h$ [m]', shrink=0.85)
    ax.set_xlabel(r'Densidad boulder $\rho_b$ [kg/m$^3$]')
    ax.set_ylabel('Desplazamiento max [m]')
    ax.set_title('(b) Efecto de densidad')

    # (c) h vs desplazamiento (color = masa)
    ax = axes[2]
    sc = ax.scatter(df['dam_height'], df['max_displacement'], c=df['boulder_mass'],
                    cmap='RdYlBu_r', s=50, edgecolors='white', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='$M$ [kg]', shrink=0.85)
    ax.set_xlabel(FEAT_LABELS['dam_height'])
    ax.set_ylabel('Desplazamiento max [m]')
    ax.set_title('(c) Altura vs respuesta')

    fig.suptitle('Analisis fisico: parametros adimensionales y densidad',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig19_energia_adimensional')


def fig20_dependencia_parcial(pkg, df):
    """Fig 20: Dependencia parcial — efecto marginalizado de cada parametro via GP."""
    if pkg is None:
        print("  [SKIP] fig20 — no hay GP entrenado")
        return

    feats = pkg['features']
    n_mc = 500
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, len(feats), figsize=(5.5 * len(feats), 5))
    if len(feats) == 1:
        axes = [axes]

    for idx, (ax, feat) in enumerate(zip(axes, feats)):
        x_line = np.linspace(df[feat].min(), df[feat].max(), 50)
        pd_mean = np.zeros(50)
        pd_lo = np.zeros(50)
        pd_hi = np.zeros(50)

        for k, x_val in enumerate(x_line):
            X_mc = np.zeros((n_mc, len(feats)))
            for j, f in enumerate(feats):
                if j == idx:
                    X_mc[:, j] = x_val
                else:
                    X_mc[:, j] = rng.uniform(df[f].min(), df[f].max(), n_mc)

            y_pred, _ = predict_gp(pkg, X_mc)
            pd_mean[k] = np.mean(y_pred)
            pd_lo[k] = np.percentile(y_pred, 10)
            pd_hi[k] = np.percentile(y_pred, 90)

        color = FEAT_COLORS.get(feat, GR)
        ax.plot(x_line, pd_mean, color=color, lw=2.5, label='Media marginal')
        ax.fill_between(x_line, pd_lo, pd_hi, alpha=0.15, color=color, label='P10-P90')

        # Rug plot
        y_lo = ax.get_ylim()[0]
        ax.plot(df[feat].values, np.full(len(df), y_lo), '|',
                color='black', alpha=0.3, markersize=10)

        ax.set_xlabel(FEAT_LABELS[feat])
        ax.set_ylabel('Desplazamiento predicho [m]')
        ax.legend(fontsize=8)

    fig.suptitle('Dependencia Parcial: efecto marginalizado sobre GP',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig20_dependencia_parcial')


def fig21_dashboard_multioutput(df):
    """Fig 21: Dashboard 2x3 — multiples metricas y parametros."""
    moved = df['failed'].astype(bool) if 'failed' in df else df.get('moved', pd.Series(dtype=bool)).astype(bool)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Fila 1: Desplazamiento vs cada parametro
    for j, feat in enumerate(['dam_height', 'boulder_mass', 'boulder_rot_z']):
        ax = axes[0, j]
        if (~moved).any():
            ax.scatter(df.loc[~moved, feat], df.loc[~moved, 'max_displacement'],
                       c=G, s=35, edgecolors='white', linewidths=0.5,
                       label='Estable', alpha=0.8)
        if moved.any():
            ax.scatter(df.loc[moved, feat], df.loc[moved, 'max_displacement'],
                       c=R, s=35, edgecolors='white', linewidths=0.5,
                       label='Movimiento', marker='^', alpha=0.8)
        ax.set_xlabel(FEAT_LABELS[feat])
        ax.set_ylabel('Desplazamiento max [m]')
        if j == 0:
            ax.legend(fontsize=7)

    # Fila 2: Otras metricas vs dam_height
    other_metrics = [
        ('max_rotation', 'Rotacion max [deg]'),
        ('max_velocity', 'Velocidad max [m/s]'),
        ('max_sph_force', 'Fuerza SPH max [N]'),
    ]
    for j, (metric, ylabel) in enumerate(other_metrics):
        ax = axes[1, j]
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        sc = ax.scatter(df['dam_height'], df[metric],
                        c=df['boulder_mass'], cmap='RdYlBu_r',
                        s=35, edgecolors='white', linewidths=0.5, alpha=0.8)
        ax.set_xlabel(FEAT_LABELS['dam_height'])
        ax.set_ylabel(ylabel)
        if j == 2:
            plt.colorbar(sc, ax=ax, label='$M$ [kg]', shrink=0.85)

    axes[0, 1].set_title('Desplazamiento vs parametros', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Metricas vs altura de columna', fontsize=11, fontweight='bold')
    fig.suptitle(f'Dashboard: {len(df)} simulaciones SPH',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig21_dashboard_multioutput')


def fig22_altura_critica(df):
    """Fig 22: Estimacion de altura critica de movimiento por rango de masa."""
    moved = df['failed'].astype(bool) if 'failed' in df else df.get('moved', pd.Series(dtype=bool)).astype(bool)

    n_bins_mass = 4
    mass_bins = np.linspace(df['boulder_mass'].min() - 1e-9,
                            df['boulder_mass'].max() + 1e-9, n_bins_mass + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors_bin = [B, R, P, O]

    # (a) Respuesta por rango de masa
    ax = axes[0]
    for k in range(n_bins_mass):
        mask = (df['boulder_mass'] >= mass_bins[k]) & (df['boulder_mass'] < mass_bins[k + 1])
        sub = df[mask]
        sub_moved = moved[mask]
        mass_mid = (mass_bins[k] + mass_bins[k + 1]) / 2
        c = colors_bin[k % len(colors_bin)]

        if (~sub_moved).any():
            ax.scatter(sub.loc[~sub_moved, 'dam_height'],
                       sub.loc[~sub_moved, 'max_displacement'],
                       c=c, s=40, marker='o', alpha=0.5, edgecolors='white', linewidths=0.5)
        if sub_moved.any():
            ax.scatter(sub.loc[sub_moved, 'dam_height'],
                       sub.loc[sub_moved, 'max_displacement'],
                       c=c, s=40, marker='^', alpha=0.8, edgecolors='white',
                       linewidths=0.5, label=f'M~{mass_mid:.2f} kg (n={len(sub)})')

    ax.set_xlabel(FEAT_LABELS['dam_height'])
    ax.set_ylabel('Desplazamiento max [m]')
    ax.set_title('(a) Respuesta por rango de masa')
    ax.legend(fontsize=7, loc='upper left')

    # (b) Altura critica estimada vs masa
    ax = axes[1]
    mass_centers = []
    h_crit_vals = []
    for k in range(n_bins_mass):
        mask = (df['boulder_mass'] >= mass_bins[k]) & (df['boulder_mass'] < mass_bins[k + 1])
        sub = df[mask]
        sub_moved = moved[mask]
        if sub_moved.sum() > 0:
            h_min_moved = sub.loc[sub_moved, 'dam_height'].min()
            mass_centers.append((mass_bins[k] + mass_bins[k + 1]) / 2)
            h_crit_vals.append(h_min_moved)

    if len(mass_centers) >= 2:
        ax.plot(mass_centers, h_crit_vals, 'o-', color=R, lw=2, markersize=8,
                label='$h_{crit}$ (min $h$ con movimiento)')
        ax.fill_between(mass_centers, 0, h_crit_vals, alpha=0.1, color=G)
        ax.fill_between(mass_centers, h_crit_vals,
                        [max(h_crit_vals) * 1.3] * len(mass_centers),
                        alpha=0.1, color=R)
        ax.text(mass_centers[0], min(h_crit_vals) * 0.5, 'ESTABLE',
                fontsize=12, color=G, fontweight='bold', alpha=0.7)
        ax.text(mass_centers[-1], max(h_crit_vals) * 1.1, 'MOVIMIENTO',
                fontsize=12, color=R, fontweight='bold', alpha=0.7, ha='right')
        ax.legend(fontsize=8)
    elif mass_centers:
        ax.scatter(mass_centers, h_crit_vals, c=R, s=100, zorder=5)
        ax.set_title('(b) Pocos datos para frontera')

    ax.set_xlabel(FEAT_LABELS['boulder_mass'])
    ax.set_ylabel('Altura critica $h_{crit}$ [m]')
    if not ax.get_title():
        ax.set_title('(b) Frontera de movimiento incipiente')

    fig.suptitle('Analisis de altura critica de movimiento',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig22_altura_critica')


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

def main(synthetic: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GENERADOR DE FIGURAS — ESTUDIO PILOTO")
    print(f"  Output: {OUT}")
    print(f"{'='*60}\n")

    # Cargar datos
    df = load_data(synthetic)
    pkg = load_gp()

    # BLOQUE 1: Datos crudos (7 figuras)
    print("BLOQUE 1: Datos crudos (7 figuras)")
    fig01_lhs_cobertura(df)
    fig02_distribucion_resultados(df)
    fig03_scatter_matrix(df)
    fig04_clasificacion_movimiento(df)
    fig05_fuerzas_vs_parametros(df)
    fig06_flujo_vs_desplazamiento(df)
    fig07_tabla_resumen(df)

    # BLOQUE 2: GP surrogate (3 figuras)
    print("\nBLOQUE 2: GP surrogate (3 figuras)")
    fig08_gp_superficie_3pares(pkg, df)
    fig09_gp_slices_detallados(pkg, df)
    fig10_gp_validacion_loo(pkg, df)

    # BLOQUE 3: UQ / Monte Carlo / Sobol (4 figuras)
    print("\nBLOQUE 3: UQ / Monte Carlo / Sobol (4 figuras)")
    fig11_mc_completo(pkg, df)
    fig12_sobol_completo(pkg, df)
    fig13_frontera_detallada(pkg, df)
    fig14_resumen_ejecutivo(df, pkg)

    # BLOQUE 4: Analisis estadistico avanzado (8 figuras)
    print("\nBLOQUE 4: Analisis estadistico avanzado (8 figuras)")
    fig15_heatmap_parametrico(df)
    fig16_boxplots_por_parametro(df)
    fig17_matriz_correlacion(df)
    fig18_violin_por_regimen(df)
    fig19_energia_adimensional(df)
    fig20_dependencia_parcial(pkg, df)
    fig21_dashboard_multioutput(df)
    fig22_altura_critica(df)

    print(f"\n{'='*60}")
    print(f"  22 FIGURAS GENERADAS")
    print(f"  {OUT}")
    print(f"{'='*60}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Figuras completas del estudio piloto')
    parser.add_argument('--synthetic', action='store_true',
                        help='Usar datos sinteticos (testing)')
    args = parser.parse_args()

    main(synthetic=args.synthetic)
