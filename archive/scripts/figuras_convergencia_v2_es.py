"""
figuras_convergencia_v2_es.py — Figuras de convergencia en español (6 dp)
Mismas figuras que v2 pero traducidas. Términos técnicos en inglés se mantienen.

Autor: Kevin Cortes (UCN 2026)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

dp       = np.array([0.020,  0.015,  0.010,  0.008,  0.005,  0.004])
disp_m   = np.array([3.4948, 3.4335, 3.0687, 2.4075, 1.7248, 1.6147])
rot_deg  = np.array([95.82,  97.23,  60.34,  87.15,  86.79,  84.81])
vel_ms   = np.array([1.1612, 1.2688, 1.1187, 1.1381, 1.1577, 1.1675])
f_sph    = np.array([166.42, 77.00,  45.27,  34.88,  23.01,  22.80])
f_cont   = np.array([2254.1, 4914.9, 130.67, 3229.1, 3083.0, 358.8])
flow_ms  = np.array([0.4890, 0.4736, 0.5200, 0.5278, 0.4226, 0.4269])
water_m  = np.array([0.1587, 0.1564, 0.1630, 0.1654, 0.3201, 0.2450])
time_min = np.array([13.2,   11.7,   23.7,   30.3,   117.8,  260.1])
dim_dp   = np.array([2.0,    2.7,    4.0,    5.0,    8.0,    10.0])

d_eq = 0.100421
N_base = 209103
particles = N_base * (0.02 / dp) ** 3

delta_disp = np.abs(np.diff(disp_m)) / disp_m[:-1] * 100
delta_fsph = np.abs(np.diff(f_sph)) / f_sph[:-1] * 100
delta_vel  = np.abs(np.diff(vel_ms)) / vel_ms[:-1] * 100
delta_rot  = np.abs(np.diff(rot_deg)) / rot_deg[:-1] * 100

# ═══════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════

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

B = '#2166AC'; R = '#B2182B'; G = '#2CA02C'
O = '#FF7F00'; P = '#6A3D9A'; GR = '#555555'

OUT = Path("C:/Seba/Tesis/data/figuras_convergencia_v2_es")
OUT.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    print(f"  OK: {name}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIG 1: Desplazamiento
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(dp, disp_m, 'o-', color=B, zorder=5, clip_on=False)

for i in range(len(dp) - 1):
    mid_x = (dp[i] + dp[i+1]) / 2
    mid_y = (disp_m[i] + disp_m[i+1]) / 2
    ax.annotate(f'$\\Delta$ = {delta_disp[i]:.1f}%',
                xy=(mid_x, mid_y + 0.12),
                fontsize=8, ha='center', va='bottom', color=R, fontweight='bold')

ax.axvspan(0.002, 0.0045, color=G, alpha=0.06, zorder=0)
ax.text(0.0035, 3.35, 'Zona de\nconvergencia', fontsize=8, ha='center',
        color=G, fontstyle='italic', alpha=0.8)

ax.set_xlabel('Espaciamiento de partículas $dp$ [m]')
ax.set_ylabel('Desplazamiento máximo del boulder [m]')
ax.set_title('Desplazamiento del Boulder vs. Resolución de Partículas')
ax.invert_xaxis()
ax.set_xlim(0.022, 0.003)
ax.set_ylim(1.3, 3.8)
save(fig, 'fig01_desplazamiento_convergencia')


# ═══════════════════════════════════════════════════════════════════════
# FIG 2: Fuerza SPH
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(dp, f_sph, 's-', color=R, zorder=5, clip_on=False)

for i in range(len(dp) - 1):
    mid_x = (dp[i] + dp[i+1]) / 2
    mid_y = (f_sph[i] + f_sph[i+1]) / 2
    offset_y = max(3, (f_sph[i] - f_sph[i+1]) * 0.15)
    ax.annotate(f'$\\Delta$ = {delta_fsph[i]:.0f}%',
                xy=(mid_x, mid_y + offset_y),
                fontsize=8, ha='center', va='bottom', color=GR, fontweight='bold')

ax.axhspan(22.0, 23.5, color=G, alpha=0.1, zorder=0)
ax.text(0.012, 21.0, '$F_{SPH}$ convergida $\\approx$ 22.8 N',
        fontsize=8, color=G, fontstyle='italic')

ax.set_xlabel('Espaciamiento de partículas $dp$ [m]')
ax.set_ylabel('Fuerza SPH máxima [N]')
ax.set_title('Convergencia de Fuerza Hidrodinámica')
ax.invert_xaxis()
ax.set_xlim(0.022, 0.003)
save(fig, 'fig02_fuerza_sph_convergencia')


# ═══════════════════════════════════════════════════════════════════════
# FIG 3: Tasa de convergencia
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 4))
dp_mid = (dp[:-1] + dp[1:]) / 2

ax.plot(dp_mid, delta_disp, 'o-', color=B, label='Desplazamiento', zorder=5)
ax.plot(dp_mid, delta_fsph, 's-', color=R, label='Fuerza SPH', zorder=5)
ax.plot(dp_mid, delta_vel,  '^-', color=O, label='Velocidad', zorder=5)
ax.plot(dp_mid, delta_rot,  'D-', color=P, label='Rotación', zorder=4)

ax.axhline(5, color='black', ls='--', lw=0.8, alpha=0.4)
ax.text(0.0045, 5.8, 'Criterio de convergencia 5%', fontsize=7.5, color=GR)

for data, color, yoff in [(delta_disp, B, 1.5), (delta_fsph, R, -2.5),
                           (delta_vel, O, 1.0), (delta_rot, P, -3.0)]:
    ax.annotate(f'{data[-1]:.1f}%', xy=(dp_mid[-1], data[-1]),
                xytext=(8, yoff), textcoords='offset points',
                fontsize=7.5, color=color, fontweight='bold')

ax.set_xlabel('Espaciamiento de partículas $dp$ [m]')
ax.set_ylabel('Cambio relativo sucesivo [%]')
ax.set_title('Tasa de Convergencia — Todas las Métricas Cinemáticas')
ax.invert_xaxis()
ax.set_xlim(0.019, 0.0035)
ax.set_ylim(0, 60)
ax.legend(loc='upper left', framealpha=0.9)
save(fig, 'fig03_tasa_convergencia')


# ═══════════════════════════════════════════════════════════════════════
# FIG 4: Diagnóstico fuerza de contacto
# ═══════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

ax1.semilogy(dp, f_cont, 'o-', color=R, label='Contacto (Chrono NSC)', zorder=5)
ax1.semilogy(dp, f_sph, 's-', color=B, label='SPH (fluido)', zorder=5)

for i, (d, fc) in enumerate(zip(dp, f_cont)):
    yoff = 1.4 if i % 2 == 0 else 0.7
    ax1.annotate(f'{fc:.0f}', xy=(d, fc * yoff),
                 fontsize=7, ha='center', color=R)

ax1.set_xlabel('$dp$ [m]')
ax1.set_ylabel('Fuerza máxima [N]')
ax1.set_title('(a) Comparación de fuerzas (escala log)')
ax1.invert_xaxis()
ax1.set_xlim(0.022, 0.003)
ax1.legend(fontsize=8, loc='lower left')
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())

colors = []
for d in dim_dp:
    if d < 4: colors.append('#EF5350')
    elif d < 8: colors.append('#FFA726')
    elif d < 10: colors.append('#FFEE58')
    else: colors.append('#66BB6A')

ax2.bar(range(len(dp)), f_cont, color=colors, edgecolor='black', lw=0.5, zorder=3)
ax2.set_xticks(range(len(dp)))
labels = [f'dp={d}\n({int(n)}p)' for d, n in zip(dp, dim_dp)]
ax2.set_xticklabels(labels, fontsize=7)
ax2.set_ylabel('Fuerza de contacto máx. [N]')
ax2.set_title('(b) Fuerza de contacto por resolución')

for i in range(len(dp)):
    ax2.text(i, f_cont[i] + 100, f'{f_cont[i]:.0f}', ha='center', fontsize=7, fontweight='bold')

from matplotlib.patches import Patch
legend_elems = [
    Patch(fc='#EF5350', ec='black', lw=0.5, label='< 4p (sub-resuelto)'),
    Patch(fc='#FFA726', ec='black', lw=0.5, label='4-7p (marginal)'),
    Patch(fc='#FFEE58', ec='black', lw=0.5, label='8-9p (transicional)'),
    Patch(fc='#66BB6A', ec='black', lw=0.5, label='$\\geq$10p (adecuado)'),
]
ax2.legend(handles=legend_elems, fontsize=6.5, loc='upper right')

fig.suptitle('No-Convergencia de Fuerza de Contacto — Sensibilidad a la Resolución',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, 'fig04_diagnostico_fuerza_contacto')


# ═══════════════════════════════════════════════════════════════════════
# FIG 5: Velocidad + Rotación
# ═══════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

ax1.plot(dp, vel_ms, '^-', color=O, zorder=5)
ax1.axhspan(1.14, 1.18, color=O, alpha=0.08, zorder=0)
mean_vel = np.mean(vel_ms[2:])
ax1.axhline(mean_vel, color=GR, ls=':', lw=0.8)
ax1.text(0.012, mean_vel + 0.008, f'Media (dp$\\leq$0.01) = {mean_vel:.3f} m/s',
         fontsize=7.5, color=GR)

for i, (d, v) in enumerate(zip(dp, vel_ms)):
    yoff = 0.015 if v > mean_vel else -0.025
    ax1.annotate(f'{v:.3f}', xy=(d, v + yoff), fontsize=7, ha='center', color=O)

ax1.set_xlabel('$dp$ [m]')
ax1.set_ylabel('Velocidad máx. del boulder [m/s]')
ax1.set_title('(a) Velocidad del Boulder — Convergida')
ax1.invert_xaxis()
ax1.set_xlim(0.022, 0.003)
ax1.set_ylim(1.05, 1.32)

ax2.plot(dp, rot_deg, 'D-', color=P, zorder=5)

for i, (d, r) in enumerate(zip(dp, rot_deg)):
    yoff = 3 if r > 80 else -6
    if i == 2: yoff = -8
    ax2.annotate(f'{r:.1f}$\\degree$', xy=(d, r + yoff),
                 fontsize=7, ha='center', color=P)

ax2.annotate('ANOMALÍA\n(caída a 60.3$\\degree$)',
             xy=(0.010, 60.34), xytext=(0.013, 50),
             fontsize=7.5, color=R, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=R, lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec=R, lw=0.8))

ax2.axhspan(83, 88, color=P, alpha=0.06, zorder=0)
ax2.text(0.005, 89, 'Estabilizando ~85$\\degree$', fontsize=7.5,
         color=P, fontstyle='italic')

ax2.set_xlabel('$dp$ [m]')
ax2.set_ylabel('Rotación máxima [deg]')
ax2.set_title('(b) Rotación del Boulder — Estabilizando')
ax2.invert_xaxis()
ax2.set_xlim(0.022, 0.003)
ax2.set_ylim(40, 110)

fig.suptitle('Métricas Cinemáticas — Velocidad Convergida, Rotación Estabilizando',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, 'fig05_velocidad_rotacion')


# ═══════════════════════════════════════════════════════════════════════
# FIG 6: Anomalía altura de agua
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(dp, water_m, 'o-', color=B, zorder=5)

mean_wh_stable = np.mean(water_m[:4])
ax.axhspan(mean_wh_stable - 0.01, mean_wh_stable + 0.01, color=B, alpha=0.08, zorder=0)
ax.axhline(mean_wh_stable, color=GR, ls=':', lw=0.8)
ax.text(0.018, mean_wh_stable + 0.012,
        f'Media estable = {mean_wh_stable:.3f} m', fontsize=8, color=GR)

for i, (d, w) in enumerate(zip(dp, water_m)):
    yoff = 0.015 if w > 0.2 else -0.02
    ax.annotate(f'{w:.3f} m', xy=(d, w + yoff), fontsize=7.5, ha='center', color=B)

ax.annotate('ANOMALÍA\n$h$ = 0.320 m\n(2$\\times$ esperado)',
            xy=(0.005, 0.320), xytext=(0.008, 0.34),
            fontsize=8, color=R, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color=R, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec=R, lw=0.8))

ax.annotate('Anómalo\n$h$ = 0.245 m',
            xy=(0.004, 0.245), xytext=(0.007, 0.27),
            fontsize=8, color=O, ha='center',
            arrowprops=dict(arrowstyle='->', color=O, lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0', ec=O, lw=0.8))

ax.set_xlabel('Espaciamiento de partículas $dp$ [m]')
ax.set_ylabel('Altura máxima del agua [m]')
ax.set_title('Altura del Agua en Gauge — Anomalía en Resolución Fina')
ax.invert_xaxis()
ax.set_xlim(0.022, 0.003)
ax.set_ylim(0.10, 0.40)
save(fig, 'fig06_anomalia_altura_agua')


# ═══════════════════════════════════════════════════════════════════════
# FIG 7: Costo computacional
# ═══════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

ax1.plot(dp, time_min, 'o-', color=B, zorder=5)
for i, (d, t) in enumerate(zip(dp, time_min)):
    hrs = f' ({t/60:.1f}h)' if t > 60 else ''
    ax1.annotate(f'{t:.0f} min{hrs}', xy=(d, t),
                 xytext=(0, 10), textcoords='offset points',
                 fontsize=7.5, ha='center', color=GR)

ax1.set_xlabel('$dp$ [m]')
ax1.set_ylabel('Tiempo de cómputo [min]')
ax1.set_title('(a) Costo Computacional')
ax1.invert_xaxis()
ax1.set_xlim(0.022, 0.003)

ax2.loglog(particles, time_min, 'o-', color=R, zorder=5)
log_n, log_t = np.log(particles), np.log(time_min)
b, log_a = np.polyfit(log_n, log_t, 1)
a = np.exp(log_a)
n_fit = np.logspace(np.log10(particles.min()), np.log10(particles.max() * 3), 50)
t_fit = a * n_fit ** b
ax2.plot(n_fit, t_fit, '--', color=GR, alpha=0.6, lw=1.0,
         label=f'$t \\propto N^{{{b:.2f}}}$')

n_003 = N_base * (0.02 / 0.003) ** 3
t_003 = a * n_003 ** b
ax2.plot(n_003, t_003, 'v', color=G, markersize=9, zorder=6)
ax2.annotate(f'dp=0.003 (est.)\n{t_003:.0f} min ({t_003/60:.1f}h)',
             xy=(n_003, t_003), xytext=(-15, 15),
             textcoords='offset points', fontsize=7.5, color=G,
             fontweight='bold', ha='right',
             arrowprops=dict(arrowstyle='->', color=G, lw=1.0))

for i, (n, t, d) in enumerate(zip(particles, time_min, dp)):
    ax2.annotate(f'dp={d}', xy=(n, t),
                 xytext=(8, -5), textcoords='offset points',
                 fontsize=6.5, color=GR)

ax2.set_xlabel('Cantidad estimada de partículas')
ax2.set_ylabel('Tiempo de cómputo [min]')
ax2.set_title('(b) Escalamiento (RTX 5090)')
ax2.legend(fontsize=8)

fig.suptitle('Costo Computacional — Simulación GPU en NVIDIA RTX 5090',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, 'fig07_costo_computacional')


# ═══════════════════════════════════════════════════════════════════════
# FIG 8: Tabla resumen
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis('off')

cols = ['$dp$ [m]', '$N_{part}$', 'Part.\nen $d_{min}$',
        'Despl.\n[m]', '$\\Delta$%',
        'Rot.\n[deg]', '$F_{SPH}$\n[N]', '$F_{cont}$\n[N]',
        'Tiempo\n[min]', 'Estado']

rows = []
for i in range(len(dp)):
    n_str = f'{particles[i]/1e6:.1f}M' if particles[i] > 1e6 else f'{particles[i]/1e3:.0f}K'
    d_str = f'{delta_disp[i-1]:.1f}%' if i > 0 else '—'
    if dim_dp[i] >= 10: status = 'ADECUADO'
    elif dim_dp[i] >= 8: status = 'TRANSICIONAL'
    elif dim_dp[i] >= 4: status = 'MARGINAL'
    else: status = 'SUB-RESUELTO'
    rows.append([
        f'{dp[i]:.3f}', n_str, f'{dim_dp[i]:.0f}',
        f'{disp_m[i]:.3f}', d_str,
        f'{rot_deg[i]:.1f}', f'{f_sph[i]:.1f}', f'{f_cont[i]:.0f}',
        f'{time_min[i]:.1f}', status,
    ])

table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.6)

for j in range(len(cols)):
    table[0, j].set_facecolor('#1565C0')
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=7)

status_colors = {
    'SUB-RESUELTO': '#FFCDD2', 'MARGINAL': '#FFE0B2',
    'TRANSICIONAL': '#FFF9C4', 'ADECUADO': '#C8E6C9',
}
for i in range(len(dp)):
    status = rows[i][-1]
    for j in range(len(cols)):
        table[i+1, j].set_facecolor(status_colors[status])

fig.suptitle('Tabla 1: Resultados de Convergencia de Malla — Boulder BLIR3 ($d_{eq}$ = 10.0 cm, RTX 5090)',
             fontsize=11, fontweight='bold', y=0.95)
save(fig, 'fig08_tabla_resumen')


# ═══════════════════════════════════════════════════════════════════════
# FIG 9: Panel resumen completo
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(11, 7))

ax = axes[0, 0]
ax.plot(dp, disp_m, 'o-', color=B, zorder=5)
ax.axvspan(0.002, 0.0045, color=G, alpha=0.06, zorder=0)
for i in range(len(dp)):
    ax.annotate(f'{disp_m[i]:.2f}', xy=(dp[i], disp_m[i]),
                xytext=(0, 8), textcoords='offset points',
                fontsize=6.5, ha='center', color=B)
ax.set_title('(a) Desplazamiento [m]', fontsize=10)
ax.set_xlabel('$dp$ [m]')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)

ax = axes[0, 1]
ax.plot(dp, f_sph, 's-', color=R, zorder=5)
ax.axhspan(22, 23.5, color=G, alpha=0.1, zorder=0)
for i in range(len(dp)):
    ax.annotate(f'{f_sph[i]:.1f}', xy=(dp[i], f_sph[i]),
                xytext=(0, 8), textcoords='offset points',
                fontsize=6.5, ha='center', color=R)
ax.set_title('(b) Fuerza SPH [N]', fontsize=10)
ax.set_xlabel('$dp$ [m]')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)

ax = axes[0, 2]
ax.plot(dp, f_cont, 'D-', color=R, zorder=5)
for i in range(len(dp)):
    yoff = 200 if f_cont[i] < 3000 else -400
    ax.annotate(f'{f_cont[i]:.0f}', xy=(dp[i], f_cont[i] + yoff),
                fontsize=6.5, ha='center', color=R)
ax.set_title('(c) Fuerza Contacto [N] — ERRÁTICA', fontsize=10, color=R)
ax.set_xlabel('$dp$ [m]')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)

ax = axes[1, 0]
ax.plot(dp, vel_ms, '^-', color=O, zorder=5)
ax.axhspan(1.14, 1.18, color=O, alpha=0.08, zorder=0)
for i in range(len(dp)):
    ax.annotate(f'{vel_ms[i]:.3f}', xy=(dp[i], vel_ms[i]),
                xytext=(0, 8), textcoords='offset points',
                fontsize=6.5, ha='center', color=O)
ax.set_title('(d) Velocidad [m/s] — CONVERGIDA', fontsize=10, color=G)
ax.set_xlabel('$dp$ [m]')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)
ax.set_ylim(1.05, 1.32)

ax = axes[1, 1]
ax.plot(dp, rot_deg, 'D-', color=P, zorder=5)
ax.axhspan(83, 88, color=P, alpha=0.06, zorder=0)
for i in range(len(dp)):
    yoff = 4 if i != 2 else -8
    ax.annotate(f'{rot_deg[i]:.1f}', xy=(dp[i], rot_deg[i] + yoff),
                fontsize=6.5, ha='center', color=P)
ax.set_title('(e) Rotación [deg] — ESTABILIZANDO', fontsize=10, color=O)
ax.set_xlabel('$dp$ [m]')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)
ax.set_ylim(40, 110)

ax = axes[1, 2]
dp_mid = (dp[:-1] + dp[1:]) / 2
ax.plot(dp_mid, delta_disp, 'o-', color=B, label='Desplazamiento', zorder=5)
ax.plot(dp_mid, delta_fsph, 's-', color=R, label='Fuerza SPH', zorder=4)
ax.plot(dp_mid, delta_vel, '^-', color=O, label='Velocidad', zorder=4)
ax.axhline(5, color='black', ls='--', lw=0.7, alpha=0.4)
ax.text(0.005, 6.5, '5%', fontsize=7, color=GR)

ax.annotate(f'{delta_disp[-1]:.1f}%', xy=(dp_mid[-1], delta_disp[-1]),
            xytext=(8, 2), textcoords='offset points',
            fontsize=7, color=B, fontweight='bold')
ax.annotate(f'{delta_fsph[-1]:.1f}%', xy=(dp_mid[-1], delta_fsph[-1]),
            xytext=(8, 2), textcoords='offset points',
            fontsize=7, color=R, fontweight='bold')
ax.annotate(f'{delta_vel[-1]:.1f}%', xy=(dp_mid[-1], delta_vel[-1]),
            xytext=(8, -6), textcoords='offset points',
            fontsize=7, color=O, fontweight='bold')

ax.set_title('(f) Tasa de Convergencia [%]', fontsize=10)
ax.set_xlabel('$dp$ [m]')
ax.invert_xaxis(); ax.set_xlim(0.019, 0.0035)
ax.set_ylim(0, 60)
ax.legend(fontsize=6.5, loc='upper left')

fig.suptitle('Estudio de Convergencia de Malla — 6 Resoluciones, RTX 5090, Boulder BLIR3',
             fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
save(fig, 'fig09_panel_resumen_completo')

print("\n9 figuras en español guardadas en:", OUT)
