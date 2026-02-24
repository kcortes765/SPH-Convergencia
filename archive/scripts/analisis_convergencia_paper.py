"""
analisis_convergencia_paper.py — Figuras de convergencia de malla (formato paper)

Genera figuras publicables del estudio de convergencia dp para la tesis/paper.
Datos: 5 resoluciones (dp = 0.020, 0.015, 0.010, 0.008, 0.005 m)
Caso base: dam_h=0.3m, mass=1.06kg, material=lime-stone, boulder BLIR3

Autor: Kevin Cortes (UCN 2026)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

dp = np.array([0.020, 0.015, 0.010, 0.008, 0.005])
displacement_m = np.array([3.494775, 3.433462, 3.068732, 2.407458, 1.724826])
rotation_deg = np.array([95.82, 97.23, 60.34, 87.15, 86.79])
velocity_ms = np.array([1.1612, 1.2688, 1.1187, 1.1381, 1.1577])
f_sph_N = np.array([166.4205, 76.9959, 45.2706, 34.8810, 23.0070])
f_contact_N = np.array([2254.10, 4914.90, 130.67, 3229.07, 3082.96])
flow_vel_ms = np.array([0.4890, 0.4736, 0.5200, 0.5278, 0.4226])
water_h_m = np.array([0.1587, 0.1564, 0.1630, 0.1654, 0.3201])
time_min = np.array([13.2, 11.7, 23.7, 30.3, 117.8])
dim_min_dp = np.array([2.0, 2.7, 4.0, 5.0, 8.0])

# Derived
d_eq = 0.100421  # m
displacement_pct = displacement_m / d_eq * 100
delta_pct = np.abs(np.diff(displacement_m)) / displacement_m[:-1] * 100

# Estimated particle count (scaling from dp=0.02 base of 209,103)
n_particles_base = 209103
particles_est = n_particles_base * (0.02 / dp) ** 3

# ---------------------------------------------------------------------------
# Style config (publication quality)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
})

COLORS = {
    'primary': '#2166AC',
    'secondary': '#B2182B',
    'tertiary': '#4DAF4A',
    'quaternary': '#FF7F00',
    'gray': '#666666',
}

OUT_DIR = Path("C:/Seba/Tesis/data/figuras_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    """Save as PNG and PDF."""
    fig.savefig(OUT_DIR / f"{name}.png")
    fig.savefig(OUT_DIR / f"{name}.pdf")
    print(f"  Saved: {name}.png + .pdf")


# ===========================================================================
# FIGURE 1: Main convergence panel (2x2)
# ===========================================================================

fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))

# --- 1a: Displacement ---
ax = axes[0, 0]
ax.plot(dp, displacement_m, 'o-', color=COLORS['primary'], zorder=3)
ax.set_xlabel('Particle spacing $dp$ [m]')
ax.set_ylabel('Max displacement [m]')
ax.set_title('(a) Boulder displacement')
ax.invert_xaxis()
ax.set_xlim(0.022, 0.003)
# Annotate relative changes
for i in range(1, len(dp)):
    d_pct = abs(displacement_m[i] - displacement_m[i-1]) / displacement_m[i-1] * 100
    mid_dp = (dp[i] + dp[i-1]) / 2
    mid_val = (displacement_m[i] + displacement_m[i-1]) / 2
    ax.annotate(f'{d_pct:.1f}%', xy=(mid_dp, mid_val),
                fontsize=7.5, ha='center', va='bottom',
                color=COLORS['secondary'], fontweight='bold')

# --- 1b: SPH Force ---
ax = axes[0, 1]
ax.plot(dp, f_sph_N, 's-', color=COLORS['secondary'], zorder=3)
ax.set_xlabel('Particle spacing $dp$ [m]')
ax.set_ylabel('Max SPH force [N]')
ax.set_title('(b) Hydrodynamic force (SPH)')
ax.invert_xaxis()
ax.set_xlim(0.022, 0.003)
# Annotate relative changes
for i in range(1, len(dp)):
    d_pct = abs(f_sph_N[i] - f_sph_N[i-1]) / f_sph_N[i-1] * 100
    mid_dp = (dp[i] + dp[i-1]) / 2
    mid_val = (f_sph_N[i] + f_sph_N[i-1]) / 2
    ax.annotate(f'{d_pct:.0f}%', xy=(mid_dp, mid_val),
                fontsize=7.5, ha='center', va='bottom',
                color=COLORS['gray'], fontweight='bold')

# --- 1c: Rotation ---
ax = axes[1, 0]
ax.plot(dp, rotation_deg, 'D-', color=COLORS['tertiary'], zorder=3)
ax.set_xlabel('Particle spacing $dp$ [m]')
ax.set_ylabel('Max rotation [deg]')
ax.set_title('(c) Boulder rotation')
ax.invert_xaxis()
ax.set_xlim(0.022, 0.003)
ax.set_ylim(0, 120)

# --- 1d: Boulder velocity ---
ax = axes[1, 1]
ax.plot(dp, velocity_ms, '^-', color=COLORS['quaternary'], zorder=3)
ax.set_xlabel('Particle spacing $dp$ [m]')
ax.set_ylabel('Max velocity [m/s]')
ax.set_title('(d) Boulder velocity')
ax.invert_xaxis()
ax.set_xlim(0.022, 0.003)
ax.set_ylim(0.9, 1.4)

fig.suptitle('Mesh convergence study — Boulder kinematics vs. particle resolution',
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'fig01_convergence_kinematics')
plt.close()


# ===========================================================================
# FIGURE 2: Contact forces (the problematic metric)
# ===========================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

# --- 2a: Contact vs SPH forces ---
ax1.plot(dp, f_contact_N, 'o-', color=COLORS['secondary'],
         label='Contact force (Chrono)', zorder=3)
ax1.plot(dp, f_sph_N, 's-', color=COLORS['primary'],
         label='SPH force', zorder=3)
ax1.set_xlabel('Particle spacing $dp$ [m]')
ax1.set_ylabel('Max force [N]')
ax1.set_title('(a) Force comparison')
ax1.invert_xaxis()
ax1.set_xlim(0.022, 0.003)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())

# --- 2b: Contact force with resolution bands ---
ax2.bar(range(len(dp)), f_contact_N, color=[
    '#D32F2F' if d < 4 else '#FF9800' if d < 8 else '#4CAF50'
    for d in dim_min_dp
], edgecolor='black', linewidth=0.5, zorder=3)
ax2.set_xticks(range(len(dp)))
ax2.set_xticklabels([f'dp={d}\n({int(n)}p)' for d, n in zip(dp, dim_min_dp)])
ax2.set_ylabel('Max contact force [N]')
ax2.set_title('(b) Contact force by resolution')
ax2.set_xlabel('Resolution (particles in min. dimension)')

# Legend for color bands
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#D32F2F', edgecolor='black', label='< 4 part. (under-resolved)'),
    Patch(facecolor='#FF9800', edgecolor='black', label='4-7 part. (marginal)'),
    Patch(facecolor='#4CAF50', edgecolor='black', label='$\\geq$ 8 part. (adequate)'),
]
ax2.legend(handles=legend_elements, fontsize=7.5, loc='upper left')

fig.suptitle('Contact force behavior — Evidence of resolution sensitivity',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'fig02_contact_forces')
plt.close()


# ===========================================================================
# FIGURE 3: Computational cost scaling
# ===========================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

# --- 3a: Time vs dp ---
ax1.plot(dp, time_min, 'o-', color=COLORS['primary'], zorder=3)
ax1.set_xlabel('Particle spacing $dp$ [m]')
ax1.set_ylabel('Wall-clock time [min]')
ax1.set_title('(a) Computational cost vs. resolution')
ax1.invert_xaxis()
ax1.set_xlim(0.022, 0.003)
for i, (d, t) in enumerate(zip(dp, time_min)):
    ax1.annotate(f'{t:.0f} min', xy=(d, t), xytext=(0, 8),
                 textcoords='offset points', fontsize=7.5,
                 ha='center', color=COLORS['gray'])

# --- 3b: Time vs estimated particles (log-log) ---
ax2.loglog(particles_est, time_min, 'o-', color=COLORS['secondary'], zorder=3)
ax2.set_xlabel('Estimated particle count')
ax2.set_ylabel('Wall-clock time [min]')
ax2.set_title('(b) Cost scaling (RTX 5090)')

# Fit power law: time = a * N^b
log_n = np.log(particles_est)
log_t = np.log(time_min)
b, log_a = np.polyfit(log_n, log_t, 1)
a = np.exp(log_a)
n_fit = np.logspace(np.log10(particles_est.min()), np.log10(particles_est.max()), 50)
t_fit = a * n_fit ** b
ax2.plot(n_fit, t_fit, '--', color=COLORS['gray'], alpha=0.7,
         label=f'$t \\propto N^{{{b:.2f}}}$')
ax2.legend()

# Annotate particle counts
for n, t, d in zip(particles_est, time_min, dp):
    label = f'{n/1e6:.1f}M' if n > 1e6 else f'{n/1e3:.0f}K'
    ax2.annotate(label, xy=(n, t), xytext=(5, 5),
                 textcoords='offset points', fontsize=7, color=COLORS['gray'])

fig.suptitle('Computational cost analysis — GPU simulation on NVIDIA RTX 5090',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'fig03_computational_cost')
plt.close()


# ===========================================================================
# FIGURE 4: Convergence rate analysis (Richardson-like)
# ===========================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

# --- 4a: Relative change between successive dp ---
dp_mid = (dp[:-1] + dp[1:]) / 2
delta_disp = np.abs(np.diff(displacement_m)) / displacement_m[:-1] * 100
delta_rot = np.abs(np.diff(rotation_deg)) / rotation_deg[:-1] * 100
delta_vel = np.abs(np.diff(velocity_ms)) / velocity_ms[:-1] * 100
delta_fsph = np.abs(np.diff(f_sph_N)) / f_sph_N[:-1] * 100

ax1.plot(dp_mid, delta_disp, 'o-', color=COLORS['primary'], label='Displacement')
ax1.plot(dp_mid, delta_fsph, 's-', color=COLORS['secondary'], label='SPH force')
ax1.plot(dp_mid, delta_vel, '^-', color=COLORS['quaternary'], label='Velocity')
ax1.axhline(y=5, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='5% threshold')
ax1.set_xlabel('Particle spacing $dp$ [m]')
ax1.set_ylabel('Relative change [%]')
ax1.set_title('(a) Successive relative change')
ax1.invert_xaxis()
ax1.legend(fontsize=8)
ax1.set_ylim(0, None)

# --- 4b: Flow quantities (should be mesh-independent) ---
ax2.plot(dp, flow_vel_ms, 'o-', color=COLORS['primary'], label='Flow velocity [m/s]')
ax2b = ax2.twinx()
ax2b.plot(dp, water_h_m, 's-', color=COLORS['secondary'], label='Water height [m]')
ax2.set_xlabel('Particle spacing $dp$ [m]')
ax2.set_ylabel('Flow velocity [m/s]', color=COLORS['primary'])
ax2b.set_ylabel('Water height [m]', color=COLORS['secondary'])
ax2.set_title('(b) Flow field quantities')
ax2.invert_xaxis()
ax2.set_xlim(0.022, 0.003)
ax2.tick_params(axis='y', labelcolor=COLORS['primary'])
ax2b.tick_params(axis='y', labelcolor=COLORS['secondary'])

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

fig.suptitle('Convergence rate and flow field stability',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'fig04_convergence_rate')
plt.close()


# ===========================================================================
# FIGURE 5: Summary table as figure (for paper/poster)
# ===========================================================================

fig, ax = plt.subplots(figsize=(7.5, 2.8))
ax.axis('off')

columns = ['$dp$ [m]', '$N_{part}$ (est.)', 'Particles\nin $d_{min}$',
           'Displ. [m]', '$\\Delta$%', 'Rot. [deg]',
           '$F_{SPH}$ [N]', '$F_{cont}$ [N]', 'Time [min]']

cell_data = []
for i in range(len(dp)):
    n_str = f'{particles_est[i]/1e6:.1f}M' if particles_est[i] > 1e6 else f'{particles_est[i]/1e3:.0f}K'
    d_str = f'{delta_pct[i-1]:.1f}%' if i > 0 else '—'
    cell_data.append([
        f'{dp[i]:.3f}',
        n_str,
        f'{dim_min_dp[i]:.0f}',
        f'{displacement_m[i]:.3f}',
        d_str,
        f'{rotation_deg[i]:.1f}',
        f'{f_sph_N[i]:.1f}',
        f'{f_contact_N[i]:.0f}',
        f'{time_min[i]:.1f}',
    ])

table = ax.table(cellText=cell_data, colLabels=columns,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

# Style header
for j in range(len(columns)):
    table[0, j].set_facecolor('#2166AC')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Color-code resolution column
for i in range(len(dp)):
    d = dim_min_dp[i]
    if d < 4:
        table[i+1, 2].set_facecolor('#FFCDD2')
    elif d < 8:
        table[i+1, 2].set_facecolor('#FFE0B2')
    else:
        table[i+1, 2].set_facecolor('#C8E6C9')

fig.suptitle('Table 1: Mesh convergence study results — Boulder BLIR3, $d_{eq}$ = 10.0 cm',
             fontsize=11, fontweight='bold', y=0.98)
fig.tight_layout()
save_fig(fig, 'fig05_summary_table')
plt.close()


# ===========================================================================
# Console summary
# ===========================================================================

print("\n" + "=" * 70)
print("CONVERGENCE ANALYSIS SUMMARY")
print("=" * 70)
print(f"\nBoulder: BLIR3, d_eq = {d_eq*100:.1f} cm, mass = 1.061 kg")
print(f"Domain: dam_height = 0.3 m, material = lime-stone")
print(f"Hardware: NVIDIA RTX 5090 (32 GB VRAM)")
print(f"Total compute time: {time_min.sum():.0f} min ({time_min.sum()/60:.1f} h)")
print(f"\nParticle scaling exponent: t ~ N^{b:.2f}")

print(f"\n{'dp':>6}  {'Displ':>8}  {'delta%':>7}  {'Rot':>7}  {'F_sph':>8}  "
      f"{'F_cont':>8}  {'dim/dp':>6}  {'Time':>7}")
print("-" * 70)
for i in range(len(dp)):
    d_str = f'{delta_pct[i-1]:.1f}%' if i > 0 else '—'
    print(f"{dp[i]:>6.3f}  {displacement_m[i]:>8.3f}  {d_str:>7}  "
          f"{rotation_deg[i]:>7.1f}  {f_sph_N[i]:>8.1f}  "
          f"{f_contact_N[i]:>8.0f}  {dim_min_dp[i]:>6.0f}  "
          f"{time_min[i]:>7.1f}")

print(f"\nVERDICT:")
print(f"  Displacement: NOT CONVERGED (delta growing: 1.8% -> 10.6% -> 21.5% -> 28.3%)")
print(f"  SPH forces:   CONVERGING (monotonic decrease, 166 -> 23 N)")
print(f"  Contact:      NOT CONVERGED (erratic: 2254 -> 4915 -> 131 -> 3229 -> 3083)")
print(f"  Rotation:     NOT CONVERGED (erratic: 96 -> 97 -> 60 -> 87 -> 87)")
print(f"  Velocity:     CONVERGED (stable ~1.15 m/s, delta < 10%)")
print(f"  Flow field:   MOSTLY STABLE (vel ~0.48 m/s, water height anomaly at dp=0.005)")

print(f"\nRECOMMENDATION:")
print(f"  dp=0.005 (8 particles in d_min) is STILL INSUFFICIENT for convergence.")
print(f"  Need dp <= 0.004 (10+ particles in d_min) for reliable results.")
print(f"  Contact force erratic behavior suggests Chrono coupling sensitivity.")
print(f"  Water height anomaly at dp=0.005 (0.32m vs ~0.16m) needs investigation.")

print(f"\nFigures saved to: {OUT_DIR}")
print("=" * 70)
