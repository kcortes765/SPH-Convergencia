#!/usr/bin/env python3
"""
Análisis completo de un caso individual de simulación SPH-DualSPHysics.
Genera ~12 figuras thesis-quality a partir de los CSVs de Chrono + Gauges.

Uso:
    python scripts/analisis_caso_individual.py [--case CASE_ID] [--outdir DIR]

Por defecto analiza lhs_001 con datos en data/processed/lhs_001/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Configuración global de matplotlib ──────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.2,
})

SENTINEL = -3.40282e+38


def load_chrono_kinematics(data_dir: Path) -> pd.DataFrame:
    """Carga ChronoExchange_mkbound_51.csv (cinemática del bloque)."""
    path = data_dir / 'ChronoExchange_mkbound_51.csv'
    df = pd.read_csv(path, sep=';')
    # Filtrar solo filas no-predictor (datos finales, no intermedios)
    if 'predictor' in df.columns:
        df = df[df['predictor'].astype(str) == 'False'].copy()
    df = df.reset_index(drop=True)
    return df


def load_chrono_forces(data_dir: Path) -> pd.DataFrame:
    """Carga ChronoBody_forces.csv con columnas renombradas."""
    path = data_dir / 'ChronoBody_forces.csv'
    raw = pd.read_csv(path, sep=';')
    # Las columnas repiten nombres para cada cuerpo.
    # Formato: Time;Body_BLIR_fx;fy;fz;mx;my;mz;cfx;cfy;cfz;cmx;cmy;cmz;Body_beach_fx;...
    cols = raw.columns.tolist()
    # Renombrar manualmente
    new_cols = ['Time',
                'blir_fx', 'blir_fy', 'blir_fz',
                'blir_mx', 'blir_my', 'blir_mz',
                'blir_cfx', 'blir_cfy', 'blir_cfz',
                'blir_cmx', 'blir_cmy', 'blir_cmz',
                'beach_fx', 'beach_fy', 'beach_fz',
                'beach_mx', 'beach_my', 'beach_mz',
                'beach_cfx', 'beach_cfy', 'beach_cfz',
                'beach_cmx', 'beach_cmy', 'beach_cmz']
    # La última columna puede ser vacía por el ; final
    if len(cols) > len(new_cols):
        new_cols += [f'_extra_{i}' for i in range(len(cols) - len(new_cols))]
    raw.columns = new_cols[:len(cols)]
    # Eliminar columnas extra vacías
    raw = raw[[c for c in raw.columns if not c.startswith('_extra_')]]
    return raw


def load_gauges_vel(data_dir: Path) -> dict:
    """Carga todos los GaugesVel_V*.csv. Retorna dict {nombre: df}."""
    gauges = {}
    for f in sorted(data_dir.glob('GaugesVel_V*.csv')):
        df = pd.read_csv(f, sep=';')
        # Reemplazar sentinel
        df = df.mask(df < -1e+30, np.nan)
        name = f.stem  # e.g. GaugesVel_V01
        gauges[name] = df
    return gauges


def load_gauges_hmax(data_dir: Path) -> dict:
    """Carga todos los GaugesMaxZ_hmax*.csv."""
    gauges = {}
    for f in sorted(data_dir.glob('GaugesMaxZ_hmax*.csv')):
        df = pd.read_csv(f, sep=';')
        df = df.mask(df < -1e+30, np.nan)
        name = f.stem
        gauges[name] = df
    return gauges


def compute_derived(kin: pd.DataFrame, props: dict) -> pd.DataFrame:
    """Calcula métricas derivadas: desplazamiento, velocidad total, energía, rotación."""
    t_col = 'time [s]'

    # Posición inicial (primer registro)
    x0 = kin['fcenter.x [m]'].iloc[0]
    y0 = kin['fcenter.y [m]'].iloc[0]
    z0 = kin['fcenter.z [m]'].iloc[0]

    # Desplazamientos
    kin['dx'] = kin['fcenter.x [m]'] - x0
    kin['dy'] = kin['fcenter.y [m]'] - y0
    kin['dz'] = kin['fcenter.z [m]'] - z0
    kin['disp_total'] = np.sqrt(kin['dx']**2 + kin['dy']**2 + kin['dz']**2)

    # Velocidad total
    kin['vel_mag'] = np.sqrt(
        kin['fvel.x [m/s]']**2 +
        kin['fvel.y [m/s]']**2 +
        kin['fvel.z [m/s]']**2
    )

    # Velocidad angular total
    kin['omega_mag'] = np.sqrt(
        kin['fomega.x [rad/s]']**2 +
        kin['fomega.y [rad/s]']**2 +
        kin['fomega.z [rad/s]']**2
    )

    # Rotación acumulada (integral numérica de omega)
    dt = kin[t_col].diff().fillna(0)
    kin['rot_accum_deg'] = np.cumsum(kin['omega_mag'] * dt) * (180 / np.pi)

    # Energía cinética: 0.5 * m * v^2
    mass = props.get('mass_kg', 1.0)
    kin['E_kinetic'] = 0.5 * mass * kin['vel_mag']**2

    # Energía potencial: m * g * dz (relativo a posición inicial)
    g = 9.81
    kin['E_potential'] = mass * g * kin['dz']

    # Energía total
    kin['E_total'] = kin['E_kinetic'] + kin['E_potential']

    # Desplazamiento normalizado por d_eq
    d_eq = props.get('d_eq_m', 0.1)
    kin['disp_norm'] = kin['disp_total'] / d_eq

    return kin


def load_boulder_properties(case_dir: Path) -> dict:
    """Lee boulder_properties.txt y extrae parámetros clave."""
    props = {}
    prop_file = case_dir / 'boulder_properties.txt'
    if not prop_file.exists():
        return {'mass_kg': 1.2244, 'd_eq_m': 0.100421, 'density_kgm3': 2309.2, 'dp': 0.02}
    for line in prop_file.read_text().splitlines():
        line = line.strip()
        if line.startswith('#') or not line or ':' not in line:
            continue
        key, val = line.split(':', 1)
        key = key.strip()
        val = val.strip()
        if key == 'mass_kg':
            props['mass_kg'] = float(val)
        elif key == 'd_eq_m':
            props['d_eq_m'] = float(val)
        elif key == 'density_kgm3':
            props['density_kgm3'] = float(val)
        elif key == 'dp':
            props['dp'] = float(val)
        elif key == 'domain_position':
            # Parse (x, y, z)
            vals = val.strip('()').split(',')
            props['pos_x'] = float(vals[0])
            props['pos_y'] = float(vals[1])
            props['pos_z'] = float(vals[2])
    return props


def detect_phases(kin: pd.DataFrame) -> dict:
    """Detecta fases clave: settling, pre-impact, impact, transport, rest.

    Usa velocidad HORIZONTAL (vx) para detectar impacto de ola,
    ya que el settling inicial genera velocidad vertical que confundiría.
    """
    t = kin['time [s]'].values
    vel = kin['vel_mag'].values
    vx = np.abs(kin['fvel.x [m/s]'].values)
    disp = kin['disp_total'].values

    # FtPause termina en t=0.5 (primer dato)
    t_start = t[0]

    # Settling: desde inicio hasta que velocidad vertical se estabiliza
    vz = np.abs(kin['fvel.z [m/s]'].values)
    settled_mask = vz < 0.005
    after_settle = t > (t_start + 0.1)
    settled_idx = np.where(settled_mask & after_settle)[0]
    t_settled = t[settled_idx[0]] if len(settled_idx) > 0 else t_start + 0.5

    # Impact: usar aceleración SPH horizontal (face.x) como indicador.
    # Durante settling, face.x = 0 (solo gravedad). Cuando la ola llega,
    # face.x sube abruptamente (>1 m/s²).
    ax_sph = np.abs(kin['face.x [m/s^2]'].values)
    t_impact = None
    for i in range(len(ax_sph)):
        if ax_sph[i] > 1.0:  # SPH acceleration threshold
            t_impact = t[i]
            break
    if t_impact is None:
        # Fallback: buscar velocidad horizontal sostenida > 0.2 m/s
        high_vx = vx > 0.2
        for i in range(len(high_vx)):
            window = min(20, len(high_vx) - i)
            if high_vx[i] and window > 5 and np.mean(high_vx[i:i+window]) > 0.5:
                t_impact = t[i]
                break
    if t_impact is None:
        t_impact = t_settled + 1.0

    # Peak velocity
    idx_peak = np.argmax(vel)
    t_peak = t[idx_peak]
    v_peak = vel[idx_peak]

    # Rest: velocidad cae a <0.01 después del pico (con ventana de estabilidad)
    post_peak = np.where((t > t_peak) & (vel < 0.01))[0]
    t_rest = t[post_peak[0]] if len(post_peak) > 0 else t[-1]

    return {
        't_start': t_start,
        't_settled': t_settled,
        't_impact': t_impact,
        't_peak': t_peak,
        'v_peak': v_peak,
        't_rest': t_rest,
        'max_disp': disp.max(),
        'final_disp': disp[-1],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

def fig_displacement(kin, phases, props, out_dir):
    """Fig 1: Desplazamiento del bloque vs tiempo."""
    t = kin['time [s]']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel superior: componentes X, Y, Z
    ax1.plot(t, kin['dx'], label='$\\Delta x$', color='#e41a1c')
    ax1.plot(t, kin['dy'], label='$\\Delta y$', color='#377eb8')
    ax1.plot(t, kin['dz'], label='$\\Delta z$', color='#4daf4a')
    ax1.set_ylabel('Displacement [m]')
    ax1.legend(loc='upper left')
    ax1.set_title('Boulder Displacement Components')

    # Marcar fases
    for ax in [ax1, ax2]:
        ax.axvline(phases['t_impact'], color='red', ls='--', alpha=0.5, label='Wave impact' if ax == ax1 else '')
        ax.axvline(phases['t_rest'], color='gray', ls='--', alpha=0.5, label='Rest' if ax == ax1 else '')

    # Panel inferior: magnitud total + normalizado
    ax2.plot(t, kin['disp_total'], color='black', linewidth=1.5, label='Total $|\\Delta \\mathbf{r}|$')
    ax2_r = ax2.twinx()
    d_eq = props.get('d_eq_m', 0.1)
    ax2_r.plot(t, kin['disp_norm'], color='purple', linewidth=1.0, alpha=0.6, label=f'Normalized ($d_{{eq}}$={d_eq:.3f} m)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Displacement [m]')
    ax2_r.set_ylabel(f'Displacement / $d_{{eq}}$')
    ax2_r.spines['right'].set_color('purple')
    ax2_r.yaxis.label.set_color('purple')
    ax2_r.tick_params(axis='y', colors='purple')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    fig.savefig(out_dir / '01_displacement_vs_time.png')
    plt.close(fig)
    print(f"  [1/12] Displacement vs time")


def fig_velocity(kin, phases, out_dir):
    """Fig 2: Velocidad del bloque vs tiempo."""
    t = kin['time [s]']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Componentes
    ax1.plot(t, kin['fvel.x [m/s]'], label='$v_x$', color='#e41a1c')
    ax1.plot(t, kin['fvel.y [m/s]'], label='$v_y$', color='#377eb8')
    ax1.plot(t, kin['fvel.z [m/s]'], label='$v_z$', color='#4daf4a')
    ax1.set_ylabel('Velocity [m/s]')
    ax1.legend(loc='upper right')
    ax1.set_title('Boulder Velocity Components')

    # Magnitud
    ax2.plot(t, kin['vel_mag'], color='black', linewidth=1.5)
    ax2.fill_between(t, 0, kin['vel_mag'], alpha=0.15, color='steelblue')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('$|v|$ [m/s]')
    ax2.set_title('Boulder Velocity Magnitude')

    # Marcar pico
    idx_peak = kin['vel_mag'].idxmax()
    ax2.annotate(f"$v_{{max}}$ = {phases['v_peak']:.3f} m/s\n$t$ = {phases['t_peak']:.2f} s",
                 xy=(phases['t_peak'], phases['v_peak']),
                 xytext=(phases['t_peak'] + 0.5, phases['v_peak'] * 0.85),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    for ax in [ax1, ax2]:
        ax.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / '02_velocity_vs_time.png')
    plt.close(fig)
    print(f"  [2/12] Velocity vs time")


def fig_trajectory_xz(kin, phases, props, out_dir):
    """Fig 3: Trayectoria del bloque en plano XZ (vista lateral)."""
    x = kin['fcenter.x [m]']
    z = kin['fcenter.z [m]']
    t = kin['time [s]']
    vel = kin['vel_mag']

    fig, ax = plt.subplots(figsize=(12, 5))

    # Colorear por velocidad
    scatter = ax.scatter(x, z, c=vel, cmap='hot_r', s=1.5, alpha=0.7,
                         vmin=0, vmax=vel.max())
    cbar = fig.colorbar(scatter, ax=ax, label='$|v|$ [m/s]', shrink=0.8)

    # Marcar inicio y fin
    ax.plot(x.iloc[0], z.iloc[0], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(x.iloc[-1], z.iloc[-1], 'rs', markersize=10, label='End', zorder=5)

    # Anotar distancia
    total_disp = phases['final_disp']
    ax.annotate(f'$\\Delta r$ = {total_disp:.2f} m',
                xy=((x.iloc[0] + x.iloc[-1])/2, z.max() + 0.02),
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$z$ [m]')
    ax.set_title('Boulder Trajectory (XZ Plane — Side View)')
    ax.legend(loc='upper left')
    # No usar aspect='equal' porque dz << dx aplasta la visualización

    fig.tight_layout()
    fig.savefig(out_dir / '03_trajectory_xz.png')
    plt.close(fig)
    print(f"  [3/12] Trajectory XZ")


def fig_forces_sph(forces, phases, out_dir):
    """Fig 4: Fuerzas SPH sobre el bloque."""
    t = forces['Time']
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    comps = [('blir_fx', '$F_x^{SPH}$', '#e41a1c'),
             ('blir_fy', '$F_y^{SPH}$', '#377eb8'),
             ('blir_fz', '$F_z^{SPH}$', '#4daf4a')]

    for ax, (col, label, color) in zip(axes, comps):
        ax.plot(t, forces[col], color=color, linewidth=0.8)
        ax.set_ylabel(f'{label} [N]')
        ax.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)
        peak_val = forces[col].abs().max()
        ax.set_title(f'{label}  (peak: {peak_val:.2f} N)')

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('SPH Hydrodynamic Forces on Boulder', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / '04_forces_sph.png')
    plt.close(fig)
    print(f"  [4/12] SPH forces")


def fig_forces_contact(forces, phases, out_dir):
    """Fig 5: Fuerzas de contacto sobre el bloque."""
    t = forces['Time']
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    comps = [('blir_cfx', '$F_x^{contact}$', '#e41a1c'),
             ('blir_cfy', '$F_y^{contact}$', '#377eb8'),
             ('blir_cfz', '$F_z^{contact}$', '#4daf4a')]

    for ax, (col, label, color) in zip(axes, comps):
        ax.plot(t, forces[col], color=color, linewidth=0.8)
        ax.set_ylabel(f'{label} [N]')
        ax.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)
        peak_val = forces[col].abs().max()
        ax.set_title(f'{label}  (peak: {peak_val:.2f} N)')

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Contact Forces on Boulder (Chrono)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / '05_forces_contact.png')
    plt.close(fig)
    print(f"  [5/12] Contact forces")


def fig_angular_velocity(kin, phases, out_dir):
    """Fig 6: Velocidad angular y rotación acumulada."""
    t = kin['time [s]']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t, kin['fomega.x [rad/s]'], label='$\\omega_x$', color='#e41a1c')
    ax1.plot(t, kin['fomega.y [rad/s]'], label='$\\omega_y$', color='#377eb8')
    ax1.plot(t, kin['fomega.z [rad/s]'], label='$\\omega_z$', color='#4daf4a')
    ax1.set_ylabel('Angular velocity [rad/s]')
    ax1.legend()
    ax1.set_title('Boulder Angular Velocity Components')

    ax2.plot(t, kin['rot_accum_deg'], color='black', linewidth=1.5)
    ax2.fill_between(t, 0, kin['rot_accum_deg'], alpha=0.15, color='orange')
    max_rot = kin['rot_accum_deg'].max()
    ax2.set_ylabel('Accumulated rotation [deg]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title(f'Accumulated Rotation (max = {max_rot:.1f}°)')

    for ax in [ax1, ax2]:
        ax.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / '06_angular_velocity.png')
    plt.close(fig)
    print(f"  [6/12] Angular velocity + rotation")


def fig_wave_height(gauges_hmax, phases, out_dir):
    """Fig 7: Altura del agua en los gauges (propagación de ola)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.viridis
    n = len(gauges_hmax)
    for i, (name, df) in enumerate(sorted(gauges_hmax.items())):
        t = df['time [s]']
        z = df['zmax [m]']
        pos_x = df['posx [m]'].iloc[0]
        color = cmap(i / max(n - 1, 1))
        ax.plot(t, z, color=color, linewidth=0.8, label=f'x={pos_x:.1f} m')

    ax.axvline(phases['t_impact'], color='red', ls='--', alpha=0.5, label='Boulder impact')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Max water height $z_{max}$ [m]')
    ax.set_title('Wave Propagation — Water Height at Gauge Stations')
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / '07_wave_height_gauges.png')
    plt.close(fig)
    print(f"  [7/12] Wave height at gauges")


def fig_flow_velocity(gauges_vel, phases, out_dir):
    """Fig 8: Velocidad del flujo en gauges seleccionados."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Seleccionar ~6 gauges representativos
    keys = sorted(gauges_vel.keys())
    step = max(1, len(keys) // 6)
    selected = keys[::step]

    cmap = plt.cm.plasma
    for i, name in enumerate(selected):
        df = gauges_vel[name]
        t = df['time [s]']
        vel_mag = np.sqrt(df['velx [m/s]']**2 + df['vely [m/s]']**2 + df['velz [m/s]']**2)
        pos_x = df['posx [m]'].iloc[0]
        color = cmap(i / max(len(selected) - 1, 1))
        ax.plot(t, vel_mag, color=color, linewidth=0.8, label=f'x={pos_x:.1f} m')

    ax.axvline(phases['t_impact'], color='red', ls='--', alpha=0.5, label='Boulder impact')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Flow velocity $|\\mathbf{v}|$ [m/s]')
    ax.set_title('Flow Velocity at Gauge Stations')
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / '08_flow_velocity_gauges.png')
    plt.close(fig)
    print(f"  [8/12] Flow velocity at gauges")


def fig_phase_portrait(kin, phases, out_dir):
    """Fig 9: Phase portrait — velocidad vs desplazamiento."""
    fig, ax = plt.subplots(figsize=(8, 7))

    disp = kin['disp_total']
    vel = kin['vel_mag']
    t = kin['time [s]']

    scatter = ax.scatter(disp, vel, c=t, cmap='coolwarm', s=2, alpha=0.7)
    cbar = fig.colorbar(scatter, ax=ax, label='Time [s]', shrink=0.8)

    # Marcar puntos clave
    ax.plot(disp.iloc[0], vel.iloc[0], 'go', markersize=10, zorder=5, label='Start')
    idx_peak = vel.idxmax()
    ax.plot(disp.iloc[idx_peak], vel.iloc[idx_peak], 'r^', markersize=10, zorder=5, label='Peak $v$')
    ax.plot(disp.iloc[-1], vel.iloc[-1], 'ks', markersize=8, zorder=5, label='End')

    ax.set_xlabel('Total displacement [m]')
    ax.set_ylabel('Velocity magnitude [m/s]')
    ax.set_title('Phase Portrait: Velocity vs Displacement')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / '09_phase_portrait.png')
    plt.close(fig)
    print(f"  [9/12] Phase portrait")


def fig_energy(kin, phases, props, out_dir):
    """Fig 10: Energía cinética, potencial y total."""
    t = kin['time [s]']
    mass = props.get('mass_kg', 1.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, kin['E_kinetic'], label='$E_{kinetic}$', color='#e41a1c', linewidth=1.2)
    ax.plot(t, kin['E_potential'], label='$E_{potential}$', color='#377eb8', linewidth=1.2)
    ax.plot(t, kin['E_total'], label='$E_{total}$', color='black', linewidth=1.5, ls='--')

    ax.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4, label='Wave impact')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [J]')
    ax.set_title(f'Boulder Energy Evolution (m = {mass:.4f} kg)')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / '10_energy_evolution.png')
    plt.close(fig)
    print(f"  [10/12] Energy evolution")


def fig_force_vs_displacement(kin, forces, phases, out_dir):
    """Fig 11: Fuerza hidrodinámica vs desplazamiento del bloque."""
    # Alinear por tiempo — interpolar forces al mismo grid que kin
    t_kin = kin['time [s]'].values
    t_f = forces['Time'].values
    disp = kin['disp_total'].values

    # Magnitud de la fuerza SPH
    f_mag = np.sqrt(forces['blir_fx']**2 + forces['blir_fy']**2 + forces['blir_fz']**2).values
    f_interp = np.interp(t_kin, t_f, f_mag)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(disp, f_interp, c=t_kin, cmap='coolwarm', s=2, alpha=0.7)
    fig.colorbar(scatter, ax=ax, label='Time [s]', shrink=0.8)

    ax.set_xlabel('Total displacement [m]')
    ax.set_ylabel('SPH force magnitude [N]')
    ax.set_title('Hydrodynamic Force vs Boulder Displacement')

    fig.tight_layout()
    fig.savefig(out_dir / '11_force_vs_displacement.png')
    plt.close(fig)
    print(f"  [11/12] Force vs displacement")


def fig_dashboard(kin, forces, phases, props, out_dir):
    """Fig 12: Dashboard resumen multi-panel."""
    t_kin = kin['time [s]']
    t_f = forces['Time']

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # (a) Desplazamiento total
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_kin, kin['disp_total'], color='black', linewidth=1.2)
    ax1.fill_between(t_kin, 0, kin['disp_total'], alpha=0.1, color='steelblue')
    ax1.set_ylabel('Displacement [m]')
    ax1.set_title('(a) Boulder Displacement')
    ax1.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)

    # (b) Velocidad
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_kin, kin['vel_mag'], color='steelblue', linewidth=1.2)
    ax2.fill_between(t_kin, 0, kin['vel_mag'], alpha=0.1, color='steelblue')
    ax2.set_ylabel('$|v|$ [m/s]')
    ax2.set_title('(b) Boulder Velocity')
    ax2.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)

    # (c) Fuerzas SPH
    ax3 = fig.add_subplot(gs[1, 0])
    f_mag = np.sqrt(forces['blir_fx']**2 + forces['blir_fy']**2 + forces['blir_fz']**2)
    ax3.plot(t_f, f_mag, color='#e41a1c', linewidth=0.8)
    ax3.set_ylabel('$|F^{SPH}|$ [N]')
    ax3.set_title('(c) SPH Force Magnitude')
    ax3.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)

    # (d) Fuerzas de contacto
    ax4 = fig.add_subplot(gs[1, 1])
    cf_mag = np.sqrt(forces['blir_cfx']**2 + forces['blir_cfy']**2 + forces['blir_cfz']**2)
    ax4.plot(t_f, cf_mag, color='#ff7f00', linewidth=0.8)
    ax4.set_ylabel('$|F^{contact}|$ [N]')
    ax4.set_title('(d) Contact Force Magnitude')
    ax4.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)

    # (e) Rotación acumulada
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t_kin, kin['rot_accum_deg'], color='purple', linewidth=1.2)
    ax5.fill_between(t_kin, 0, kin['rot_accum_deg'], alpha=0.1, color='purple')
    ax5.set_ylabel('Rotation [deg]')
    ax5.set_xlabel('Time [s]')
    ax5.set_title('(e) Accumulated Rotation')
    ax5.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)

    # (f) Energía
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t_kin, kin['E_kinetic'], label='$E_k$', color='#e41a1c', linewidth=1.0)
    ax6.plot(t_kin, kin['E_potential'], label='$E_p$', color='#377eb8', linewidth=1.0)
    ax6.plot(t_kin, kin['E_total'], label='$E_{tot}$', color='black', linewidth=1.2, ls='--')
    ax6.set_ylabel('Energy [J]')
    ax6.set_xlabel('Time [s]')
    ax6.set_title('(f) Energy Evolution')
    ax6.legend(fontsize=8)
    ax6.axvline(phases['t_impact'], color='red', ls='--', alpha=0.4)

    # Título global con parámetros
    mass = props.get('mass_kg', 0)
    d_eq = props.get('d_eq_m', 0)
    dp = props.get('dp', 0)
    fig.suptitle(
        f"Simulation Dashboard — $m$ = {mass:.4f} kg, $d_{{eq}}$ = {d_eq*1000:.1f} mm, "
        f"$dp$ = {dp} m\n"
        f"$\\Delta r_{{max}}$ = {phases['max_disp']:.3f} m, "
        f"$v_{{max}}$ = {phases['v_peak']:.3f} m/s, "
        f"$\\theta_{{max}}$ = {kin['rot_accum_deg'].max():.1f}°",
        fontsize=13, y=1.02
    )

    fig.savefig(out_dir / '12_dashboard_summary.png')
    plt.close(fig)
    print(f"  [12/12] Dashboard summary")


def print_summary(kin, forces, phases, props):
    """Imprime resumen de métricas clave en consola."""
    d_eq = props.get('d_eq_m', 0.1)
    mass = props.get('mass_kg', 1.0)

    f_sph_mag = np.sqrt(forces['blir_fx']**2 + forces['blir_fy']**2 + forces['blir_fz']**2)
    cf_mag = np.sqrt(forces['blir_cfx']**2 + forces['blir_cfy']**2 + forces['blir_cfz']**2)

    print("\n" + "=" * 65)
    print("  RESUMEN DEL CASO")
    print("=" * 65)
    print(f"  Masa:                {mass:.4f} kg")
    print(f"  Densidad:            {props.get('density_kgm3', 0):.1f} kg/m³")
    print(f"  d_eq:                {d_eq*1000:.2f} mm")
    print(f"  dp:                  {props.get('dp', 0)} m")
    print(f"  Tiempo simulado:     {kin['time [s]'].iloc[0]:.2f} – {kin['time [s]'].iloc[-1]:.2f} s")
    print(f"  Timesteps:           {len(kin)}")
    print("-" * 65)
    print(f"  Desplaz. máximo:     {phases['max_disp']:.4f} m ({phases['max_disp']/d_eq:.1f} × d_eq)")
    print(f"  Desplaz. final:      {phases['final_disp']:.4f} m ({phases['final_disp']/d_eq:.1f} × d_eq)")
    print(f"  Velocidad pico:      {phases['v_peak']:.4f} m/s (t = {phases['t_peak']:.2f} s)")
    print(f"  Rotación acumulada:  {kin['rot_accum_deg'].max():.1f}°")
    print(f"  Fuerza SPH pico:     {f_sph_mag.max():.2f} N")
    print(f"  Fuerza contacto pico:{cf_mag.max():.2f} N")
    print("-" * 65)
    print(f"  Impacto de ola:      t = {phases['t_impact']:.2f} s")
    print(f"  Pico velocidad:      t = {phases['t_peak']:.2f} s")
    print(f"  Reposo final:        t = {phases['t_rest']:.2f} s")
    print(f"  Fase transporte:     {phases['t_rest'] - phases['t_impact']:.2f} s")
    print("=" * 65)

    # Guardar CSV resumen
    return {
        'mass_kg': mass,
        'density_kgm3': props.get('density_kgm3', 0),
        'd_eq_mm': d_eq * 1000,
        'dp_m': props.get('dp', 0),
        'max_displacement_m': phases['max_disp'],
        'final_displacement_m': phases['final_disp'],
        'disp_over_deq': phases['max_disp'] / d_eq,
        'peak_velocity_ms': phases['v_peak'],
        't_peak_vel_s': phases['t_peak'],
        'max_rotation_deg': kin['rot_accum_deg'].max(),
        'peak_sph_force_N': f_sph_mag.max(),
        'peak_contact_force_N': cf_mag.max(),
        't_impact_s': phases['t_impact'],
        't_rest_s': phases['t_rest'],
        'transport_duration_s': phases['t_rest'] - phases['t_impact'],
    }


def main():
    parser = argparse.ArgumentParser(description='Análisis de caso individual SPH')
    parser.add_argument('--case', default='lhs_001', help='ID del caso (default: lhs_001)')
    parser.add_argument('--outdir', default=None, help='Directorio de salida para figuras')
    args = parser.parse_args()

    # Rutas
    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data' / 'processed' / args.case
    case_dir = base / 'cases' / args.case

    if not data_dir.exists():
        print(f"ERROR: No se encontró {data_dir}")
        sys.exit(1)

    out_dir = Path(args.outdir) if args.outdir else base / 'data' / f'figuras_{args.case}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analizando caso: {args.case}")
    print(f"Datos en: {data_dir}")
    print(f"Figuras en: {out_dir}")
    print()

    # ── Cargar datos ────────────────────────────────────────────────────────
    print("Cargando datos...")
    props = load_boulder_properties(case_dir)
    kin = load_chrono_kinematics(data_dir)
    forces = load_chrono_forces(data_dir)
    gauges_vel = load_gauges_vel(data_dir)
    gauges_hmax = load_gauges_hmax(data_dir)

    print(f"  Kinematics: {len(kin)} timesteps, t = [{kin['time [s]'].iloc[0]:.3f}, {kin['time [s]'].iloc[-1]:.3f}] s")
    print(f"  Forces: {len(forces)} timesteps")
    print(f"  Vel gauges: {len(gauges_vel)}, Height gauges: {len(gauges_hmax)}")
    print()

    # ── Derivar métricas ────────────────────────────────────────────────────
    kin = compute_derived(kin, props)
    phases = detect_phases(kin)

    # ── Generar figuras ─────────────────────────────────────────────────────
    print("Generando figuras...")
    fig_displacement(kin, phases, props, out_dir)
    fig_velocity(kin, phases, out_dir)
    fig_trajectory_xz(kin, phases, props, out_dir)
    fig_forces_sph(forces, phases, out_dir)
    fig_forces_contact(forces, phases, out_dir)
    fig_angular_velocity(kin, phases, out_dir)
    fig_wave_height(gauges_hmax, phases, out_dir)
    fig_flow_velocity(gauges_vel, phases, out_dir)
    fig_phase_portrait(kin, phases, out_dir)
    fig_energy(kin, phases, props, out_dir)
    fig_force_vs_displacement(kin, forces, phases, out_dir)
    fig_dashboard(kin, forces, phases, props, out_dir)

    # ── Resumen ─────────────────────────────────────────────────────────────
    metrics = print_summary(kin, forces, phases, props)

    # Guardar CSV resumen
    csv_path = out_dir / 'resumen_metricas.csv'
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"\nMétricas guardadas en: {csv_path}")
    print(f"12 figuras guardadas en: {out_dir}")


if __name__ == '__main__':
    main()
