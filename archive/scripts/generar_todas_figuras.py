"""
generar_todas_figuras.py — Generador Unificado de Figuras para Tesis

Genera TODAS las figuras necesarias para el documento de tesis en un solo run.
Lee datos de SQLite + CSV de convergencia + modelo GP.

Grupos de figuras:
  A. Convergencia (dp study) — Fig 1-4
  B. Resultados LHS (campana parametrica) — Fig 5-7
  C. GP Surrogate (ML) — Fig 8-10

Ejecutar:
    python generar_todas_figuras.py

Salida: data/figuras_tesis/ (PNG 300 DPI + PDF)

Autor: Kevin Cortes (UCN 2026)
"""

import sqlite3
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "results.sqlite"
CONV_CSV = PROJECT_ROOT / "data" / "reporte_convergencia.csv"
GCI_JSON = PROJECT_ROOT / "data" / "figuras_paper" / "convergence_gci_results.json"
OUT = PROJECT_ROOT / "data" / "figuras_tesis"

# Estilo consistente para todas las figuras
STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Colores consistentes
C_BLUE = '#2166AC'
C_RED = '#B2182B'
C_GREEN = '#10b981'
C_ORANGE = '#f59e0b'
C_PURPLE = '#8b5cf6'
C_GRAY = '#94a3b8'

# Constantes fisicas
D_EQ = 0.100421  # m
N_BASE = 209103  # particulas a dp=0.02
G = 9.81


def save_fig(fig, name: str):
    """Guarda figura en PNG + PDF."""
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    logger.info(f"  {name}")


# =========================================================================
# A. FIGURAS DE CONVERGENCIA
# =========================================================================

def load_convergence():
    """Carga datos de convergencia desde CSV (separador ;)."""
    if not CONV_CSV.exists():
        logger.warning(f"CSV convergencia no encontrado: {CONV_CSV}")
        return None
    df = pd.read_csv(CONV_CSV, sep=';')
    # Filtrar solo OK (excluir FALLO_SIM, PENDIENTE)
    df = df[df['status'].str.upper() == 'OK'].copy()
    df = df.sort_values('dp', ascending=False).reset_index(drop=True)
    # Alias para columna de tiempo
    if 'tiempo_computo_min' in df.columns and 'time_minutes' not in df.columns:
        df['time_minutes'] = df['tiempo_computo_min']
    return df


def fig_convergence_displacement(df):
    """Fig 1: Desplazamiento vs dp (convergencia principal)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(df['dp'], df['max_displacement_m'], 'o-', color=C_BLUE,
            markersize=8, linewidth=2, markeredgecolor='white', markeredgewidth=1)

    # Anotar delta%
    for i in range(1, len(df)):
        delta = abs(df['max_displacement_m'].iloc[i] - df['max_displacement_m'].iloc[i-1])
        delta_pct = delta / df['max_displacement_m'].iloc[i-1] * 100
        ax.annotate(f'{delta_pct:.1f}%',
                    xy=(df['dp'].iloc[i], df['max_displacement_m'].iloc[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, color=C_RED, fontweight='bold')

    ax.set_xlabel('Distancia entre particulas $dp$ [m]')
    ax.set_ylabel('Desplazamiento maximo $\\delta_{max}$ [m]')
    ax.set_title('Convergencia de malla: Desplazamiento del boulder')
    ax.invert_xaxis()
    ax.set_xlim(ax.get_xlim()[0] * 1.05, ax.get_xlim()[1] * 0.95)
    fig.tight_layout()
    save_fig(fig, 'fig01_convergencia_desplazamiento')


def fig_convergence_forces(df):
    """Fig 2: Fuerzas SPH y contacto vs dp."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(df['dp'], df['max_sph_force_N'], 'o-', color=C_RED,
             markersize=8, linewidth=2, label='$F_{SPH}$')
    ax1.set_xlabel('$dp$ [m]')
    ax1.set_ylabel('Fuerza SPH maxima [N]')
    ax1.set_title('(a) Fuerza hidrodinamica SPH')
    ax1.invert_xaxis()

    ax2.plot(df['dp'], df['max_contact_force_N'], 's-', color=C_ORANGE,
             markersize=8, linewidth=2, label='$F_{contacto}$')
    ax2.set_xlabel('$dp$ [m]')
    ax2.set_ylabel('Fuerza de contacto maxima [N]')
    ax2.set_title('(b) Fuerza de contacto Chrono')
    ax2.invert_xaxis()

    fig.suptitle('Convergencia de malla: Fuerzas sobre el boulder',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig02_convergencia_fuerzas')


def fig_convergence_cost(df):
    """Fig 3: Costo computacional (tiempo + particulas)."""
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    particles = N_BASE * (0.02 / df['dp']) ** 3

    color_time = C_BLUE
    color_part = C_RED

    ax1.bar(range(len(df)), df['time_minutes'], color=color_time, alpha=0.7,
            label='Tiempo [min]')
    ax1.set_ylabel('Tiempo de simulacion [min]', color=color_time)
    ax1.set_xlabel('Resolucion $dp$ [m]')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([f'{d:.3f}' for d in df['dp']])
    ax1.tick_params(axis='y', labelcolor=color_time)

    ax2 = ax1.twinx()
    ax2.plot(range(len(df)), particles / 1e6, 's-', color=color_part,
             markersize=8, linewidth=2, label='Particulas [M]')
    ax2.set_ylabel('Particulas [millones]', color=color_part)
    ax2.tick_params(axis='y', labelcolor=color_part)

    # Anotar valores sobre las barras
    for i, (t, p) in enumerate(zip(df['time_minutes'], particles)):
        ax1.text(i, t + 2, f'{t:.0f} min', ha='center', va='bottom',
                 fontsize=8, color=color_time)

    fig.suptitle('Costo computacional vs resolucion de malla',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'fig03_convergencia_costo')


def fig_convergence_summary(df):
    """Fig 4: Resumen multi-variable (4 metricas vs dp)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    metrics = [
        ('max_displacement_m', 'Desplazamiento [m]', C_BLUE),
        ('max_sph_force_N', 'Fuerza SPH [N]', C_RED),
    ]

    # Agregar metricas opcionales si existen
    if 'max_rotation_deg' in df.columns:
        metrics.append(('max_rotation_deg', 'Rotacion [deg]', C_PURPLE))
    if 'max_velocity_ms' in df.columns:
        metrics.append(('max_velocity_ms', 'Velocidad boulder [m/s]', C_GREEN))

    for ax, (col, label, color) in zip(axes.flat, metrics):
        if col in df.columns:
            ax.plot(df['dp'], df[col], 'o-', color=color, markersize=7, linewidth=2)
            ax.set_ylabel(label)
        ax.set_xlabel('$dp$ [m]')
        ax.invert_xaxis()

    # Limpiar ejes no usados
    for i in range(len(metrics), 4):
        axes.flat[i].set_visible(False)

    fig.suptitle('Convergencia multi-variable', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'fig04_convergencia_resumen')


# =========================================================================
# B. FIGURAS DE RESULTADOS LHS
# =========================================================================

def load_lhs_results():
    """Carga resultados de campana LHS desde SQLite."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("""
        SELECT * FROM results
        WHERE dam_height > 0 AND boulder_mass > 0
          AND case_name LIKE 'lhs_%'
    """, conn)
    conn.close()
    if df.empty:
        return None

    # Metricas derivadas
    df['froude'] = df['max_flow_velocity'] / np.sqrt(G * df['max_water_height'].clip(lower=0.001))
    df['force_ratio'] = df['max_sph_force'] / df['max_contact_force'].clip(lower=0.001)
    return df


def fig_lhs_scatter(df):
    """Fig 5: Desplazamiento vs Rotacion (scatter coloreado por status)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    moved = df[df['failed'] == 1]
    stable = df[df['failed'] == 0]

    if len(stable) > 0:
        ax.scatter(stable['max_displacement'], stable['max_rotation'],
                   c=C_GREEN, s=60, edgecolors='white', linewidths=0.5,
                   label=f'Estable ({len(stable)})', zorder=5)
    if len(moved) > 0:
        ax.scatter(moved['max_displacement'], moved['max_rotation'],
                   c=C_RED, s=60, edgecolors='white', linewidths=0.5,
                   label=f'Movimiento ({len(moved)})', zorder=5)

    ax.set_xlabel('Desplazamiento maximo [m]')
    ax.set_ylabel('Rotacion maxima [deg]')
    ax.set_title('Clasificacion de estabilidad del boulder')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, 'fig05_lhs_clasificacion')


def fig_lhs_froude(df):
    """Fig 6: Froude vs Desplazamiento."""
    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(df['froude'], df['max_displacement'],
                    c=df['boulder_mass'], cmap='viridis', s=60,
                    edgecolors='white', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='Masa boulder [kg]')
    ax.axvline(1.0, color=C_RED, ls='--', alpha=0.5, label='Fr = 1')
    ax.set_xlabel('Numero de Froude $Fr$')
    ax.set_ylabel('Desplazamiento maximo [m]')
    ax.set_title('Froude vs Desplazamiento (coloreado por masa)')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, 'fig06_lhs_froude')


def fig_lhs_forces(df):
    """Fig 7: Descomposicion de fuerzas por caso."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df))
    w = 0.35

    ax.bar(x - w/2, df['max_sph_force'], w, color=C_BLUE, alpha=0.8, label='$F_{SPH}$')
    ax.bar(x + w/2, df['max_contact_force'], w, color=C_ORANGE, alpha=0.8, label='$F_{contacto}$')

    ax.set_xticks(x)
    ax.set_xticklabels(df['case_name'], rotation=45, ha='right')
    ax.set_ylabel('Fuerza maxima [N]')
    ax.set_title('Descomposicion de fuerzas por caso')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, 'fig07_lhs_fuerzas')


# =========================================================================
# MAIN
# =========================================================================

def main():
    plt.rcParams.update(STYLE)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  GENERADOR DE FIGURAS PARA TESIS")
    print(f"  Salida: {OUT}")
    print(f"{'='*60}\n")

    fig_count = 0

    # --- A. Convergencia ---
    print("A. Figuras de convergencia...")
    df_conv = load_convergence()
    if df_conv is not None and len(df_conv) >= 3:
        fig_convergence_displacement(df_conv)
        fig_convergence_forces(df_conv)
        fig_convergence_cost(df_conv)
        fig_convergence_summary(df_conv)
        fig_count += 4
    else:
        print("  SALTADO: datos de convergencia insuficientes")

    # --- B. Resultados LHS ---
    print("\nB. Figuras de campana LHS...")
    df_lhs = load_lhs_results()
    if df_lhs is not None and len(df_lhs) >= 3:
        fig_lhs_scatter(df_lhs)
        fig_lhs_froude(df_lhs)
        fig_lhs_forces(df_lhs)
        fig_count += 3
    else:
        print("  SALTADO: datos LHS insuficientes")

    # --- C. GP Surrogate (delega a ml_surrogate) ---
    print("\nC. Figuras GP Surrogate...")
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / 'src'))
        from ml_surrogate import run_surrogate
        result = run_surrogate()
        fig_count += 3
    except Exception as e:
        print(f"  SALTADO: error en ml_surrogate ({e})")

    print(f"\n{'='*60}")
    print(f"  {fig_count} figuras generadas en {OUT}")
    print(f"{'='*60}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
