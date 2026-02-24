"""
generar_figuras_tesis.py — Graficos de calidad para la tesis

Usa TODA la data real disponible:
- Convergencia (4 dp completados)
- LHS campaign (5 casos)
- Series de tiempo (cinematica, fuerzas, flujo)
- Diego reference case

Genera PNGs en data/figuras_tesis/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.8,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'lines.linewidth': 2,
    'lines.antialiased': True,
    'savefig.facecolor': '#0d1117',
    'savefig.edgecolor': 'none',
})

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'figuras_tesis'
OUT.mkdir(parents=True, exist_ok=True)

# Color palette
C_BLUE = '#58a6ff'
C_GREEN = '#3fb950'
C_ORANGE = '#d29922'
C_RED = '#f85149'
C_PURPLE = '#bc8cff'
C_CYAN = '#39d2c0'
C_PINK = '#f778ba'
C_GRAY = '#8b949e'


def save(fig, name):
    path = OUT / f'{name}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f'  [OK] {name}.png')


# ============================================================================
# 1. CONVERGENCIA DE MALLA
# ============================================================================

def fig_convergencia():
    df = pd.read_csv(ROOT / 'data' / 'reporte_convergencia.csv', sep=';')
    df = df[df['status'] == 'OK'].sort_values('dp', ascending=False)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('ESTUDIO DE CONVERGENCIA DE MALLA\nDualSPHysics v5.4 + ProjectChrono — RTX 5090',
                 color=C_CYAN, y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    dp = df['dp'].values
    dp_labels = [f'{d:.3f}' for d in dp]

    # Desplazamiento
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(dp, df['max_displacement_m'], 'o-', color=C_BLUE, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
    ax.fill_between(dp, df['max_displacement_m'], alpha=0.15, color=C_BLUE)
    ax.set_xlabel('dp (m)')
    ax.set_ylabel('Desplazamiento max (m)')
    ax.set_title('Desplazamiento del Boulder', color=C_BLUE)
    ax.invert_xaxis()
    ax.grid(True)

    # Rotacion
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(dp, df['max_rotation_deg'], 's-', color=C_ORANGE, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
    ax.fill_between(dp, df['max_rotation_deg'], alpha=0.15, color=C_ORANGE)
    ax.set_xlabel('dp (m)')
    ax.set_ylabel('Rotacion max (deg)')
    ax.set_title('Rotacion del Boulder', color=C_ORANGE)
    ax.invert_xaxis()
    ax.grid(True)

    # Velocidad del boulder
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(dp, df['max_velocity_ms'], 'D-', color=C_GREEN, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
    ax.fill_between(dp, df['max_velocity_ms'], alpha=0.15, color=C_GREEN)
    ax.set_xlabel('dp (m)')
    ax.set_ylabel('Velocidad max (m/s)')
    ax.set_title('Velocidad del Boulder', color=C_GREEN)
    ax.invert_xaxis()
    ax.grid(True)

    # Fuerzas
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(dp, df['max_sph_force_N'], 'o-', color=C_CYAN, markersize=10, markeredgecolor='white', markeredgewidth=1.5, label='SPH')
    ax.semilogy(dp, df['max_contact_force_N'], '^-', color=C_RED, markersize=10, markeredgecolor='white', markeredgewidth=1.5, label='Contacto')
    ax.set_xlabel('dp (m)')
    ax.set_ylabel('Fuerza max (N)')
    ax.set_title('Fuerzas sobre Boulder', color=C_RED)
    ax.legend(facecolor='#161b22', edgecolor='#30363d')
    ax.invert_xaxis()
    ax.grid(True)

    # Delta %
    ax = fig.add_subplot(gs[1, 1])
    deltas = []
    for i in range(1, len(df)):
        prev = df.iloc[i-1]['max_displacement_m']
        curr = df.iloc[i]['max_displacement_m']
        deltas.append(abs(curr - prev) / prev * 100)
    trans = [f'{dp[i]}\n->\n{dp[i+1]}' for i in range(len(deltas))]
    bars = ax.bar(trans, deltas, color=[C_GREEN if d < 5 else C_ORANGE if d < 15 else C_RED for d in deltas],
                  alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=5, color=C_GREEN, linestyle='--', linewidth=1.5, alpha=0.7, label='Umbral 5%')
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{d:.1f}%', ha='center', color='white', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cambio relativo (%)')
    ax.set_title('Convergencia: Delta %', color=C_PURPLE)
    ax.legend(facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, axis='y')

    # Costo computacional
    ax = fig.add_subplot(gs[1, 2])
    colors_bar = [C_BLUE, C_CYAN, C_GREEN, C_ORANGE]
    ax.bar(dp_labels, df['tiempo_computo_min'], color=colors_bar, alpha=0.85,
           edgecolor='white', linewidth=0.5)
    for i, (t, d) in enumerate(zip(df['tiempo_computo_min'], dp_labels)):
        ax.text(i, t + 0.5, f'{t:.0f} min', ha='center', color='white', fontweight='bold')
    ax.set_xlabel('dp (m)')
    ax.set_ylabel('Tiempo (min)')
    ax.set_title('Costo Computacional', color=C_PURPLE)
    ax.grid(True, axis='y')

    save(fig, '01_convergencia_malla')


# ============================================================================
# 2. TRAYECTORIA DEL BOULDER (serie de tiempo - LHS cases)
# ============================================================================

def fig_trayectorias_lhs():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('TRAYECTORIA DEL BOULDER — 5 Casos LHS\ndam_height variable, dp=0.02m, material=PVC',
                 color=C_CYAN, y=1.02)

    colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE]
    lhs_info = [
        ('lhs_001', 'h=0.454m, m=1.22kg'),
        ('lhs_002', 'h=0.208m, m=2.32kg'),
        ('lhs_003', 'h=0.374m, m=2.61kg'),
        ('lhs_004', 'h=0.394m, m=1.89kg'),
        ('lhs_005', 'h=0.312m, m=1.62kg'),
    ]

    for i, (case, label) in enumerate(lhs_info):
        csv_path = ROOT / 'data' / 'processed' / case / 'ChronoExchange_mkbound_51.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, sep=';')
        t = df['time [s]']
        cx = df['fcenter.x [m]']
        cy = df['fcenter.y [m]']
        cz = df['fcenter.z [m]']

        disp = np.sqrt((cx - cx.iloc[0])**2 + (cy - cy.iloc[0])**2 + (cz - cz.iloc[0])**2)

        axes[0].plot(t, disp, color=colors[i], label=label, alpha=0.9)
        axes[1].plot(t, cx, color=colors[i], alpha=0.9)
        axes[2].plot(t, cz, color=colors[i], alpha=0.9)

    axes[0].set_xlabel('Tiempo (s)')
    axes[0].set_ylabel('Desplazamiento CM (m)')
    axes[0].set_title('Desplazamiento Total', color=C_BLUE)
    axes[0].legend(facecolor='#161b22', edgecolor='#30363d', fontsize=8)
    axes[0].grid(True)

    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('Posicion X (m)')
    axes[1].set_title('Posicion X (eje del canal)', color=C_GREEN)
    axes[1].grid(True)

    axes[2].set_xlabel('Tiempo (s)')
    axes[2].set_ylabel('Posicion Z (m)')
    axes[2].set_title('Posicion Z (vertical)', color=C_ORANGE)
    axes[2].grid(True)

    plt.tight_layout()
    save(fig, '02_trayectorias_lhs')


# ============================================================================
# 3. FUERZAS SOBRE EL BOULDER (serie de tiempo)
# ============================================================================

def fig_fuerzas_timeseries():
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('FUERZAS SOBRE EL BOULDER — Caso lhs_001\ndam_h=0.454m, mass=1.22kg, dp=0.02m',
                 color=C_CYAN, y=1.01)

    csv_path = ROOT / 'data' / 'processed' / 'lhs_001' / 'ChronoBody_forces.csv'
    if not csv_path.exists():
        print('  [--] ChronoBody_forces no encontrado para lhs_001')
        return

    df = pd.read_csv(csv_path, sep=';')
    t = df['Time']

    # SPH forces
    sph_mag = np.sqrt(df['Body_BLIR_fx']**2 + df['fy']**2 + df['fz']**2)
    axes[0].plot(t, sph_mag, color=C_CYAN, linewidth=0.8, alpha=0.9)
    axes[0].fill_between(t, sph_mag, alpha=0.1, color=C_CYAN)
    axes[0].set_xlabel('Tiempo (s)')
    axes[0].set_ylabel('|F_SPH| (N)')
    axes[0].set_title('Fuerza Hidrodinamica (SPH)', color=C_CYAN)
    axes[0].grid(True)

    # Contact forces
    contact_mag = np.sqrt(df['cfx']**2 + df['cfy']**2 + df['cfz']**2)
    axes[1].plot(t, contact_mag, color=C_RED, linewidth=0.8, alpha=0.9)
    axes[1].fill_between(t, contact_mag, alpha=0.1, color=C_RED)
    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('|F_contacto| (N)')
    axes[1].set_title('Fuerza de Contacto (Chrono)', color=C_RED)
    axes[1].grid(True)

    plt.tight_layout()
    save(fig, '03_fuerzas_timeseries')


# ============================================================================
# 4. VELOCIDAD DEL FLUJO (Gauges)
# ============================================================================

def fig_flujo_gauges():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CAMPO DE FLUJO — Gauges de Velocidad y Altura\nCaso lhs_001 (dam_h=0.454m)',
                 color=C_CYAN, y=1.02)

    colors_g = [C_BLUE, C_CYAN, C_GREEN, C_ORANGE, C_RED, C_PURPLE,
                C_PINK, C_GRAY, '#58a6ff', '#3fb950', '#d29922', '#f85149']

    # Velocity gauges
    for i in range(1, 13):
        csv_path = ROOT / 'data' / 'processed' / 'lhs_001' / f'GaugesVel_V{i:02d}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, sep=';')
        t = df['time [s]']
        vel_mag = np.sqrt(df['velx [m/s]']**2 + df['vely [m/s]']**2 + df['velz [m/s]']**2)
        axes[0].plot(t, vel_mag, color=colors_g[i-1], linewidth=0.7, alpha=0.8, label=f'V{i:02d}')

    axes[0].set_xlabel('Tiempo (s)')
    axes[0].set_ylabel('|Velocidad| (m/s)')
    axes[0].set_title('Velocidad del Flujo en Gauges', color=C_BLUE)
    axes[0].legend(facecolor='#161b22', edgecolor='#30363d', fontsize=7, ncol=3)
    axes[0].grid(True)

    # Water height gauges
    sentinel = -3.40282e+38
    for i in range(1, 9):
        csv_path = ROOT / 'data' / 'processed' / 'lhs_001' / f'GaugesMaxZ_hmax{i:02d}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, sep=';')
        t = df['time [s]']
        zmax = df['zmax [m]'].replace(sentinel, np.nan)
        axes[1].plot(t, zmax, color=colors_g[i-1], linewidth=0.7, alpha=0.8, label=f'hmax{i:02d}')

    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('Altura max agua (m)')
    axes[1].set_title('Altura del Agua en Gauges', color=C_GREEN)
    axes[1].legend(facecolor='#161b22', edgecolor='#30363d', fontsize=7, ncol=2)
    axes[1].grid(True)

    plt.tight_layout()
    save(fig, '04_flujo_gauges')


# ============================================================================
# 5. CAMPANA LHS — RESUMEN PARAMETRICO
# ============================================================================

def fig_lhs_parametrico():
    results = [
        {'case': 'lhs_001', 'dam_h': 0.454, 'mass': 1.224, 'disp': 6.10, 'rot': 91.1},
        {'case': 'lhs_002', 'dam_h': 0.208, 'mass': 2.321, 'disp': 1.42, 'rot': 64.7},
        {'case': 'lhs_003', 'dam_h': 0.374, 'mass': 2.610, 'disp': 5.16, 'rot': 100.0},
        {'case': 'lhs_004', 'dam_h': 0.394, 'mass': 1.886, 'disp': 5.69, 'rot': 137.6},
        {'case': 'lhs_005', 'dam_h': 0.312, 'mass': 1.620, 'disp': 4.47, 'rot': 74.6},
    ]
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('CAMPANA LHS — Relacion Parametrica\n5 Casos, dp=0.02m, Latin Hypercube Sampling (seed=42)',
                 color=C_CYAN, y=1.02)

    # Scatter: dam_height vs displacement, size=mass
    sc = axes[0].scatter(df['dam_h'], df['disp'], c=df['mass'], cmap='plasma',
                         s=df['mass']*150, alpha=0.85, edgecolors='white', linewidth=1.5,
                         zorder=5)
    for _, r in df.iterrows():
        axes[0].annotate(r['case'].replace('lhs_', '#'), (r['dam_h'], r['disp']),
                        textcoords="offset points", xytext=(10, 5), color='white', fontsize=9)
    axes[0].set_xlabel('Altura Dam Break (m)')
    axes[0].set_ylabel('Desplazamiento max (m)')
    axes[0].set_title('Desplaz. vs Altura de Ola', color=C_BLUE)
    axes[0].grid(True)
    cb = plt.colorbar(sc, ax=axes[0], label='Masa boulder (kg)')
    cb.ax.yaxis.label.set_color('#c9d1d9')
    cb.ax.tick_params(colors='#8b949e')

    # Scatter: mass vs rotation, color=dam_h
    sc2 = axes[1].scatter(df['mass'], df['rot'], c=df['dam_h'], cmap='cool',
                          s=200, alpha=0.85, edgecolors='white', linewidth=1.5,
                          marker='D', zorder=5)
    for _, r in df.iterrows():
        axes[1].annotate(r['case'].replace('lhs_', '#'), (r['mass'], r['rot']),
                        textcoords="offset points", xytext=(10, 5), color='white', fontsize=9)
    axes[1].set_xlabel('Masa boulder (kg)')
    axes[1].set_ylabel('Rotacion max (deg)')
    axes[1].set_title('Rotacion vs Masa', color=C_ORANGE)
    axes[1].grid(True)
    cb2 = plt.colorbar(sc2, ax=axes[1], label='Altura dam (m)')
    cb2.ax.yaxis.label.set_color('#c9d1d9')
    cb2.ax.tick_params(colors='#8b949e')

    plt.tight_layout()
    save(fig, '05_lhs_parametrico')


# ============================================================================
# 6. PIPELINE OVERVIEW (diagrama conceptual)
# ============================================================================

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(18, 5))
    fig.suptitle('PIPELINE AUTOMATIZADO — SPH Hydraulic Data Refinery',
                 color=C_CYAN, y=0.98)

    boxes = [
        (0.5, 'LHS\nSampling', C_PURPLE, 'scipy.stats.qmc\nN parametros'),
        (3.0, 'Geometry\nBuilder', C_BLUE, 'trimesh + lxml\nSTL → XML'),
        (5.5, 'Batch\nRunner', C_GREEN, 'GenCase + GPU\nDualSPHysics'),
        (8.0, 'Data\nCleaner', C_ORANGE, 'Chrono CSVs\n→ SQLite'),
        (10.5, 'ML\nSurrogate', C_RED, 'Gaussian Process\nRegression'),
    ]

    for x, label, color, desc in boxes:
        rect = plt.Rectangle((x - 0.8, 0.3), 1.6, 1.4, facecolor=color,
                             alpha=0.25, edgecolor=color, linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, 1.15, label, ha='center', va='center', fontsize=13,
                fontweight='bold', color=color, zorder=4)
        ax.text(x, 0.05, desc, ha='center', va='center', fontsize=8,
                color=C_GRAY, style='italic', zorder=4)

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.9
        x2 = boxes[i+1][0] - 0.9
        ax.annotate('', xy=(x2, 1.0), xytext=(x1, 1.0),
                   arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=2))

    # Status badges
    statuses = ['DONE', 'DONE', 'DONE', 'DONE', 'PENDING']
    status_colors = [C_GREEN, C_GREEN, C_GREEN, C_GREEN, C_GRAY]
    for (x, _, _, _), st, sc in zip(boxes, statuses, status_colors):
        ax.text(x, 1.9, st, ha='center', va='center', fontsize=9,
                fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=sc, alpha=0.8))

    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.3, 2.3)
    ax.set_aspect('equal')
    ax.axis('off')

    save(fig, '06_pipeline_overview')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print(f'Generando figuras de tesis en {OUT}/\n')

    print('1. Convergencia de malla...')
    try:
        fig_convergencia()
    except Exception as e:
        print(f'  [!!] {e}')

    print('2. Trayectorias LHS...')
    try:
        fig_trayectorias_lhs()
    except Exception as e:
        print(f'  [!!] {e}')

    print('3. Fuerzas time-series...')
    try:
        fig_fuerzas_timeseries()
    except Exception as e:
        print(f'  [!!] {e}')

    print('4. Flujo gauges...')
    try:
        fig_flujo_gauges()
    except Exception as e:
        print(f'  [!!] {e}')

    print('5. LHS parametrico...')
    try:
        fig_lhs_parametrico()
    except Exception as e:
        print(f'  [!!] {e}')

    print('6. Pipeline overview...')
    try:
        fig_pipeline()
    except Exception as e:
        print(f'  [!!] {e}')

    print(f'\nListo. Todos los graficos en: {OUT}/')
