"""
convergencia_formal.py — Verificación Formal de Convergencia de Malla (Celik 2008)

Implementa:
  1. Richardson Extrapolation con refinement ratios no uniformes
  2. Grid Convergence Index (GCI) con safety factor Fs=1.25
  3. Orden aparente de convergencia p (método iterativo Celik 2008)
  4. Clasificación automática: monotónica / oscilatoria / divergente
  5. Banda de incertidumbre para métricas oscilatorias (fuerzas de contacto)
  6. Análisis de anomalías (water height, contact forces)
  7. 8 figuras formato paper (serif, 300 DPI, PDF+PNG)
  8. Tabla resumen con GCI y veredicto para cada métrica

Referencias:
  - Celik et al. (2008), J. Fluids Eng. 130(7):078001 — procedimiento GCI
  - Roache (1997), "Quantification of Uncertainty in CFD" — GCI original
  - Lind et al. (2020), Proc. Royal Society A — convergencia SPH, p esperado 1.0-1.8
  - Ramachandran et al. (2023), Comp. Particle Mechanics — convergencia DualSPHysics

Autor: Kevin Cortes (UCN 2026)
Fecha: 2026-02-21
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

# Ordered coarse → fine (as run)
dp_all      = np.array([0.020, 0.015, 0.010, 0.008, 0.005])
disp_m      = np.array([3.494775, 3.433462, 3.068732, 2.407458, 1.724826])
rot_deg     = np.array([95.82, 97.23, 60.34, 87.15, 86.79])
vel_ms      = np.array([1.1612, 1.2688, 1.1187, 1.1381, 1.1577])
f_sph_N     = np.array([166.4205, 76.9959, 45.2706, 34.8810, 23.0070])
f_cont_N    = np.array([2254.10, 4914.90, 130.67, 3229.07, 3082.96])
flow_ms     = np.array([0.4890, 0.4736, 0.5200, 0.5278, 0.4226])
water_h_m   = np.array([0.1587, 0.1564, 0.1630, 0.1654, 0.3201])
time_min    = np.array([13.2, 11.7, 23.7, 30.3, 117.8])
dim_min_dp  = np.array([2.0, 2.7, 4.0, 5.0, 8.0])

d_eq = 0.100421  # m
N_base = 209103  # particles at dp=0.02
particles_est = N_base * (0.02 / dp_all) ** 3

# Dict for iteration
METRICS = {
    'Displacement':     {'data': disp_m,    'unit': 'm',   'symbol': r'$\delta_{max}$'},
    'SPH Force':        {'data': f_sph_N,   'unit': 'N',   'symbol': r'$F_{SPH}$'},
    'Velocity':         {'data': vel_ms,    'unit': 'm/s', 'symbol': r'$v_{max}$'},
    'Rotation':         {'data': rot_deg,   'unit': 'deg', 'symbol': r'$\theta_{max}$'},
    'Contact Force':    {'data': f_cont_N,  'unit': 'N',   'symbol': r'$F_{cont}$'},
    'Flow Velocity':    {'data': flow_ms,   'unit': 'm/s', 'symbol': r'$v_{flow}$'},
}


# ═══════════════════════════════════════════════════════════════════════════
# GCI ENGINE (Celik 2008)
# ═══════════════════════════════════════════════════════════════════════════

def celik_apparent_order(phi1, phi2, phi3, r21, r32, p_formal=2.0,
                         max_iter=200, tol=1e-8):
    """
    Celik et al. (2008) iterative method for apparent order p.
    phi1=finest, phi2=medium, phi3=coarsest.
    r21=h2/h1, r32=h3/h2 (both > 1).
    """
    eps21 = phi2 - phi1
    eps32 = phi3 - phi2

    if abs(eps21) < 1e-15 or abs(eps32) < 1e-15:
        return np.nan, 'degenerate'

    s = np.sign(eps32 / eps21)

    if s > 0 and abs(eps32 / eps21) < 1:
        conv_type = 'monotonic_convergence'
    elif s > 0 and abs(eps32 / eps21) >= 1:
        conv_type = 'monotonic_divergence'
    else:
        conv_type = 'oscillatory'

    p = p_formal
    for _ in range(max_iter):
        try:
            num = r21**p - s
            den = r32**p - s
            if num <= 0 or den <= 0:
                break
            q = np.log(num / den)
            p_new = (1.0 / np.log(r21)) * abs(np.log(abs(eps32 / eps21)) + q)
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            break
        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new

    p = np.clip(p, 0.5, max(p_formal, 4.0))
    return p, conv_type


def richardson_extrapolate(phi1, phi2, r21, p):
    """Extrapolate to zero spacing."""
    return phi1 + (phi1 - phi2) / (r21**p - 1)


def compute_gci(phi1, phi2, phi3, r21, r32, p, Fs=1.25):
    """Full GCI analysis per Celik 2008."""
    eps21 = phi2 - phi1
    eps32 = phi3 - phi2

    e_a21 = abs(eps21 / phi1) if abs(phi1) > 1e-15 else np.nan
    e_a32 = abs(eps32 / phi2) if abs(phi2) > 1e-15 else np.nan

    phi_ext = richardson_extrapolate(phi1, phi2, r21, p)
    e_ext = abs(phi_ext - phi1) / abs(phi_ext) if abs(phi_ext) > 1e-15 else np.nan

    gci_fine = Fs * e_a21 / (r21**p - 1)
    gci_med = Fs * e_a32 / (r32**p - 1)

    ar = gci_med / (r21**p * gci_fine) if abs(gci_fine) > 1e-15 else np.nan

    return {
        'phi_ext': phi_ext,
        'e_a21': e_a21,
        'e_a32': e_a32,
        'e_ext': e_ext,
        'GCI_fine': gci_fine,
        'GCI_med': gci_med,
        'AR': ar,
        'in_asymptotic': (0.9 < ar < 1.1) if not np.isnan(ar) else False,
    }


def oscillatory_uncertainty(values):
    """For non-convergent metrics: report uncertainty band."""
    arr = np.array(values)
    U = 0.5 * (arr.max() - arr.min())
    return {
        'mean': arr.mean(),
        'U': U,
        'U_pct': U / abs(arr.mean()) * 100 if abs(arr.mean()) > 1e-15 else np.nan,
        'min': arr.min(),
        'max': arr.max(),
    }


def full_gci_analysis(dp_seq, phi_seq, metric_name, triplet_indices=(4, 3, 1)):
    """
    Run complete GCI on a chosen triplet.
    Default triplet: dp=0.005(idx4), dp=0.008(idx3), dp=0.015(idx1)
    r21=1.6, r32=1.875 — both > 1.3 (Celik requirement).
    """
    i1, i2, i3 = triplet_indices  # fine, medium, coarse
    h1, h2, h3 = dp_seq[i1], dp_seq[i2], dp_seq[i3]
    p1, p2, p3 = phi_seq[i1], phi_seq[i2], phi_seq[i3]

    r21 = h2 / h1
    r32 = h3 / h2

    p_app, conv_type = celik_apparent_order(p1, p2, p3, r21, r32)

    result = {
        'metric': metric_name,
        'triplet_dp': (h1, h2, h3),
        'triplet_phi': (p1, p2, p3),
        'r21': r21,
        'r32': r32,
        'p_apparent': p_app,
        'conv_type': conv_type,
    }

    if conv_type == 'monotonic_convergence':
        gci = compute_gci(p1, p2, p3, r21, r32, p_app)
        result.update(gci)
    else:
        osc = oscillatory_uncertainty(phi_seq)
        result.update(osc)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

# Best triplet: dp=0.005, 0.008, 0.015 (r21=1.6, r32=1.875)
TRIPLET = (4, 3, 1)  # indices into dp_all (fine, med, coarse)

results = {}
for name, info in METRICS.items():
    results[name] = full_gci_analysis(dp_all, info['data'], name, TRIPLET)

# Secondary triplet for validation: dp=0.010, 0.015, 0.020
TRIPLET_COARSE = (2, 1, 0)
results_coarse = {}
for name, info in METRICS.items():
    results_coarse[name] = full_gci_analysis(dp_all, info['data'], name, TRIPLET_COARSE)


# ═══════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
    'axes.linewidth': 0.7,
    'lines.linewidth': 1.4,
    'lines.markersize': 6,
})

C = {
    'blue': '#2166AC', 'red': '#B2182B', 'green': '#4DAF4A',
    'orange': '#FF7F00', 'purple': '#6A3D9A', 'gray': '#666666',
    'light_blue': '#92C5DE', 'light_red': '#F4A582',
}

OUT = Path("C:/Seba/Tesis/data/figuras_paper")
OUT.mkdir(parents=True, exist_ok=True)

def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    print(f"  [{name}] saved")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Convergence with Richardson Extrapolation bands
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.flatten()

for idx, (name, info) in enumerate(METRICS.items()):
    ax = axes[idx]
    data = info['data']
    r = results[name]

    ax.plot(dp_all, data, 'o-', color=C['blue'], zorder=5, label='Simulated')

    if r['conv_type'] == 'monotonic_convergence':
        # Richardson extrapolated value
        phi_ext = r['phi_ext']
        gci = r['GCI_fine']
        ax.axhline(phi_ext, color=C['green'], ls='--', lw=1.0, alpha=0.8,
                    label=f'Richardson: {phi_ext:.3f} {info["unit"]}')
        # GCI band on finest
        band = abs(data[-1]) * gci
        ax.fill_between([dp_all[-1] - 0.001, dp_all[-1] + 0.001],
                        data[-1] - band, data[-1] + band,
                        color=C['green'], alpha=0.15, zorder=2)
        ax.annotate(f'p={r["p_apparent"]:.2f}\nGCI={gci*100:.1f}%',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=7, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C['green'], alpha=0.9))
    else:
        # Oscillatory band
        osc_mean = r.get('mean', np.nanmean(data))
        osc_U = r.get('U', 0)
        ax.axhspan(osc_mean - osc_U, osc_mean + osc_U,
                    color=C['red'], alpha=0.08, zorder=1)
        ax.axhline(osc_mean, color=C['red'], ls=':', lw=0.8, alpha=0.6)
        ax.annotate(f'{r["conv_type"].replace("_", " ")}\nU={osc_U:.1f} {info["unit"]}',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=7, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C['red'], alpha=0.9))

    ax.set_title(f'{info["symbol"]} — {name}', fontsize=9.5)
    ax.set_xlabel('$dp$ [m]')
    ax.set_ylabel(f'{info["symbol"]} [{info["unit"]}]')
    ax.invert_xaxis()
    ax.set_xlim(0.022, 0.003)

fig.suptitle('Mesh Convergence with Richardson Extrapolation and GCI (Celik 2008)',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'fig01_richardson_convergence')
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Convergence classification map
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis('off')

col_labels = ['Metric', 'Triplet $dp$', '$r_{21}$', '$r_{32}$',
              'Type', '$p_{app}$', '$\\phi_{ext}$', 'GCI$_{fine}$', 'AR', 'Verdict']

rows = []
colors_row = []
for name in METRICS:
    r = results[name]
    info = METRICS[name]
    tp = r['conv_type'].replace('_', ' ').title()

    if r['conv_type'] == 'monotonic_convergence':
        p_str = f"{r['p_apparent']:.2f}"
        ext_str = f"{r['phi_ext']:.3f}"
        gci_str = f"{r['GCI_fine']*100:.1f}%"
        ar_str = f"{r['AR']:.3f}"
        in_ar = r.get('in_asymptotic', False)
        verdict = 'CONVERGED' if r['GCI_fine'] < 0.05 else 'CONVERGING'
        row_color = '#C8E6C9' if verdict == 'CONVERGED' else '#FFF9C4'
    else:
        p_str = '—'
        ext_str = '—'
        gci_str = f"U={r.get('U', 0):.1f}"
        ar_str = '—'
        verdict = 'NOT CONVERGED'
        row_color = '#FFCDD2'

    rows.append([
        name,
        f'{r["triplet_dp"][0]}, {r["triplet_dp"][1]}, {r["triplet_dp"][2]}',
        f'{r["r21"]:.2f}',
        f'{r["r32"]:.2f}',
        tp[:12],
        p_str,
        ext_str,
        gci_str,
        ar_str,
        verdict,
    ])
    colors_row.append(row_color)

table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.6)

# Header style
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#1565C0')
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=7)

# Row colors
for i in range(len(rows)):
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(colors_row[i])

fig.suptitle('Table 2: Formal Convergence Verification — GCI Method (Celik et al. 2008)',
             fontsize=11, fontweight='bold', y=0.98)
savefig(fig, 'fig02_gci_table')
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Apparent order p comparison (two triplets)
# ═══════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# Bar chart of p for each metric
metrics_names = list(METRICS.keys())
p_fine = [results[n]['p_apparent'] if results[n]['conv_type'] == 'monotonic_convergence' else 0
          for n in metrics_names]
p_coarse = [results_coarse[n]['p_apparent'] if results_coarse[n]['conv_type'] == 'monotonic_convergence' else 0
            for n in metrics_names]
conv_fine = [results[n]['conv_type'] for n in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax1.bar(x - width/2, p_fine, width, label='Fine triplet\n(0.005, 0.008, 0.015)',
                color=C['blue'], edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, p_coarse, width, label='Coarse triplet\n(0.010, 0.015, 0.020)',
                color=C['light_blue'], edgecolor='black', linewidth=0.5)

# Mark non-convergent with X
for i, ct in enumerate(conv_fine):
    if ct != 'monotonic_convergence':
        ax1.text(x[i] - width/2, 0.1, 'X', ha='center', va='bottom',
                fontsize=12, color=C['red'], fontweight='bold')

ax1.axhline(2.0, color=C['gray'], ls='--', lw=0.8, label='Formal order (p=2)')
ax1.axhspan(1.0, 1.8, color=C['green'], alpha=0.08, label='Expected SPH range')
ax1.set_xticks(x)
ax1.set_xticklabels([n.replace(' ', '\n') for n in metrics_names], fontsize=7)
ax1.set_ylabel('Apparent order $p$')
ax1.set_title('(a) Observed convergence order')
ax1.legend(fontsize=6.5, loc='upper right')
ax1.set_ylim(0, 3.5)

# GCI comparison
gci_fine_vals = []
gci_labels = []
gci_colors = []
for n in metrics_names:
    r = results[n]
    if r['conv_type'] == 'monotonic_convergence':
        gci_fine_vals.append(r['GCI_fine'] * 100)
        gci_labels.append(n)
        gci_colors.append(C['green'] if r['GCI_fine'] < 0.05 else C['orange'])

ax2.barh(range(len(gci_labels)), gci_fine_vals, color=gci_colors,
         edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(gci_labels)))
ax2.set_yticklabels(gci_labels, fontsize=8)
ax2.set_xlabel('GCI$_{fine}$ [%]')
ax2.set_title('(b) Grid Convergence Index')
ax2.axvline(5, color=C['red'], ls='--', lw=0.8, label='5% threshold')
ax2.legend(fontsize=7)

for i, v in enumerate(gci_fine_vals):
    ax2.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=7.5)

fig.suptitle('Convergence Order and Grid Convergence Index',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'fig03_order_and_gci')
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Contact force deep-dive (the finding)
# ═══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(10, 4))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

# 4a: F_contact vs F_sph (log scale)
ax1 = fig.add_subplot(gs[0])
ax1.semilogy(dp_all, f_cont_N, 'o-', color=C['red'], label='$F_{contact}$ (Chrono)')
ax1.semilogy(dp_all, f_sph_N, 's-', color=C['blue'], label='$F_{SPH}$ (fluid)')
ax1.set_xlabel('$dp$ [m]')
ax1.set_ylabel('Force [N]')
ax1.set_title('(a) Force magnitude')
ax1.invert_xaxis()
ax1.legend(fontsize=7)
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())

# 4b: Contact force coefficient of variation across dp
ax2 = fig.add_subplot(gs[1])
# Sliding pairs analysis
cv_contact = []
cv_sph = []
dp_pairs = []
for i in range(len(dp_all) - 1):
    pair = [f_cont_N[i], f_cont_N[i+1]]
    cv_contact.append(np.std(pair) / np.mean(pair) * 100)
    pair_s = [f_sph_N[i], f_sph_N[i+1]]
    cv_sph.append(np.std(pair_s) / np.mean(pair_s) * 100)
    dp_pairs.append(f'{dp_all[i]:.3f}-\n{dp_all[i+1]:.3f}')

x2 = np.arange(len(dp_pairs))
ax2.bar(x2 - 0.17, cv_contact, 0.34, color=C['red'], label='Contact', edgecolor='black', lw=0.5)
ax2.bar(x2 + 0.17, cv_sph, 0.34, color=C['blue'], label='SPH', edgecolor='black', lw=0.5)
ax2.set_xticks(x2)
ax2.set_xticklabels(dp_pairs, fontsize=6.5)
ax2.set_ylabel('Pairwise CV [%]')
ax2.set_title('(b) Inter-resolution variability')
ax2.legend(fontsize=7)

# 4c: Diagnosis text box
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')
diagnosis = (
    "DIAGNOSIS: Contact Force Non-Convergence\n"
    "═══════════════════════════════════════════\n\n"
    "Root causes (5 interacting mechanisms):\n\n"
    "1. WCSPH pressure oscillations propagate\n"
    "   into force integrals on body surface\n\n"
    "2. Kernel truncation at fluid-boulder\n"
    "   interface changes discontinuously\n\n"
    "3. Chrono NSC contact detection depends\n"
    "   on surface particle count (~dp⁻²)\n\n"
    "4. Timestep aliasing: dt~CFL·dp/cs\n"
    "   samples impulsive peaks differently\n\n"
    "5. Particle disorder random force noise\n"
    "   (uncorrelated between dp levels)\n\n"
    "─────────────────────────────────────\n"
    "RECOMMENDATION: Use displacement and\n"
    "impulse (∫F·dt) instead of peak force\n"
    "as stability criteria.\n\n"
    "Refs: Lind et al. (2020), Martínez-\n"
    "Estévez et al. (2023)"
)
ax3.text(0.05, 0.95, diagnosis, transform=ax3.transAxes,
         fontsize=7, va='top', ha='left', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', fc='#FFF3E0', ec=C['orange'], lw=1.2))

fig.suptitle('Contact Force Resolution Sensitivity — A Multi-Mechanism Problem',
             fontsize=12, fontweight='bold', y=1.02)
savefig(fig, 'fig04_contact_force_diagnosis')
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Convergence rate (successive relative change)
# ═══════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

dp_mid = (dp_all[:-1] + dp_all[1:]) / 2
metrics_for_rate = {
    'Displacement': (disp_m, C['blue'], 'o-'),
    'SPH Force': (f_sph_N, C['red'], 's-'),
    'Velocity': (vel_ms, C['orange'], '^-'),
    'Flow Velocity': (flow_ms, C['purple'], 'D-'),
}

for name, (data, color, marker) in metrics_for_rate.items():
    delta = np.abs(np.diff(data)) / data[:-1] * 100
    ax1.plot(dp_mid, delta, marker, color=color, label=name, markersize=5)

ax1.axhline(5, color='black', ls='--', lw=0.7, alpha=0.5, label='5% convergence criterion')
ax1.set_xlabel('$dp$ [m]')
ax1.set_ylabel('Successive relative change [%]')
ax1.set_title('(a) Convergence rate')
ax1.invert_xaxis()
ax1.legend(fontsize=7)
ax1.set_ylim(0, None)

# 5b: Water height anomaly
ax2.plot(dp_all, water_h_m, 'o-', color=C['red'], zorder=5)
ax2.fill_between(dp_all[:4], water_h_m[:4] - 0.01, water_h_m[:4] + 0.01,
                 color=C['blue'], alpha=0.15, label='Stable band (dp 0.020-0.008)')
ax2.annotate('ANOMALY\n$h = 0.32$ m\n(2× expected)',
             xy=(0.005, 0.3201), xytext=(0.009, 0.28),
             fontsize=8, ha='center',
             arrowprops=dict(arrowstyle='->', color=C['red']),
             color=C['red'], fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec=C['red']))
ax2.set_xlabel('$dp$ [m]')
ax2.set_ylabel('Max water height [m]')
ax2.set_title('(b) Water height anomaly at $dp=0.005$')
ax2.invert_xaxis()
ax2.set_xlim(0.022, 0.003)
ax2.legend(fontsize=7)

fig.suptitle('Convergence Rate Analysis and Flow Field Anomaly',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'fig05_rate_and_anomaly')
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Computational cost (power-law + extrapolation)
# ═══════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# 6a: log-log scaling
ax1.loglog(particles_est, time_min, 'o-', color=C['blue'], zorder=5)
log_n, log_t = np.log(particles_est), np.log(time_min)
b, log_a = np.polyfit(log_n, log_t, 1)
a = np.exp(log_a)
n_fit = np.logspace(np.log10(particles_est.min()), np.log10(5e7), 50)
t_fit = a * n_fit ** b
ax1.plot(n_fit, t_fit, '--', color=C['gray'], alpha=0.7,
         label=f'$t \\propto N^{{{b:.2f}}}$')

# Extrapolate dp=0.004 and dp=0.003
for dp_ext, marker in [(0.004, 'v'), (0.003, 'D')]:
    n_ext = N_base * (0.02 / dp_ext) ** 3
    t_ext = a * n_ext ** b
    ax1.plot(n_ext, t_ext, marker, color=C['red'], markersize=8, zorder=6)
    label = f'dp={dp_ext}: {t_ext:.0f} min ({t_ext/60:.1f}h)'
    ax1.annotate(label, xy=(n_ext, t_ext), xytext=(10, 5),
                 textcoords='offset points', fontsize=7, color=C['red'])

for n, t, d in zip(particles_est, time_min, dp_all):
    lab = f'{n/1e6:.1f}M' if n > 1e6 else f'{n/1e3:.0f}K'
    ax1.annotate(f'dp={d}\n{lab}', xy=(n, t), xytext=(-5, -15),
                 textcoords='offset points', fontsize=6, color=C['gray'], ha='center')

ax1.set_xlabel('Estimated particle count $N$')
ax1.set_ylabel('Wall-clock time [min]')
ax1.set_title('(a) Scaling law (RTX 5090)')
ax1.legend(fontsize=7)

# 6b: Cost per simulated second
cost_per_s = time_min * 60 / 10.0  # seconds of compute per second of simulation
ax2.bar(range(len(dp_all)), cost_per_s, color=C['blue'], edgecolor='black', lw=0.5)
ax2.set_xticks(range(len(dp_all)))
ax2.set_xticklabels([f'{d}' for d in dp_all])
ax2.set_xlabel('$dp$ [m]')
ax2.set_ylabel('Compute time / sim. time [s/s]')
ax2.set_title('(b) Cost ratio')
for i, v in enumerate(cost_per_s):
    ax2.text(i, v + 5, f'{v:.0f}x', ha='center', fontsize=7.5, fontweight='bold')

fig.suptitle('Computational Cost Analysis and Scaling Extrapolation',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'fig06_cost_scaling')
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Resolution quality map
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 4))

# Bubble chart: x=dp, y=displacement, size=particles, color=dim_min/dp
scatter = ax.scatter(dp_all, disp_m, s=particles_est / 5000,
                     c=dim_min_dp, cmap='RdYlGn', vmin=1, vmax=10,
                     edgecolors='black', linewidth=0.8, zorder=5)

# Convergence target zone
r_disp = results['Displacement']
if r_disp['conv_type'] == 'monotonic_convergence':
    phi_ext = r_disp['phi_ext']
    gci = r_disp['GCI_fine']
    band = abs(disp_m[-1]) * gci
    ax.axhspan(phi_ext - band, phi_ext + band, color=C['green'], alpha=0.1,
               label=f'Richardson target: {phi_ext:.2f} ± {band:.2f} m')
    ax.axhline(phi_ext, color=C['green'], ls='--', lw=1)

for i, (d, disp, n) in enumerate(zip(dp_all, disp_m, particles_est)):
    lab = f'{n/1e6:.1f}M' if n > 1e6 else f'{n/1e3:.0f}K'
    ax.annotate(f'dp={d}\n{lab}\n{dim_min_dp[i]:.0f}p',
                xy=(d, disp), xytext=(12, 0), textcoords='offset points',
                fontsize=7, va='center')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Particles in $d_{min}$', fontsize=9)

ax.set_xlabel('Particle spacing $dp$ [m]')
ax.set_ylabel('Max displacement [m]')
ax.set_title('Resolution Quality Map — Bubble size $\\propto$ particle count')
ax.invert_xaxis()
ax.set_xlim(0.025, 0.002)
ax.legend(fontsize=8, loc='upper left')

fig.tight_layout()
savefig(fig, 'fig07_resolution_quality_map')
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Complete summary panel (the "money figure")
# ═══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(11, 7))
gs = GridSpec(2, 3, hspace=0.35, wspace=0.35)

# 8a: Displacement + Richardson
ax = fig.add_subplot(gs[0, 0])
ax.plot(dp_all, disp_m, 'o-', color=C['blue'], zorder=5, label='Simulated')
r = results['Displacement']
if r['conv_type'] == 'monotonic_convergence':
    ax.axhline(r['phi_ext'], color=C['green'], ls='--', lw=1,
               label=f'$\\phi_{{ext}}$ = {r["phi_ext"]:.2f} m')
    band = abs(disp_m[-1]) * r['GCI_fine']
    ax.fill_between([0.003, 0.006], disp_m[-1]-band, disp_m[-1]+band,
                    color=C['green'], alpha=0.15)
ax.set_xlabel('$dp$ [m]'); ax.set_ylabel('Displacement [m]')
ax.set_title(f'(a) Displacement — p={r["p_apparent"]:.2f}')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)
ax.legend(fontsize=7)

# 8b: SPH Force + Richardson
ax = fig.add_subplot(gs[0, 1])
ax.plot(dp_all, f_sph_N, 's-', color=C['red'], zorder=5, label='Simulated')
r = results['SPH Force']
if r['conv_type'] == 'monotonic_convergence':
    ax.axhline(r['phi_ext'], color=C['green'], ls='--', lw=1,
               label=f'$\\phi_{{ext}}$ = {r["phi_ext"]:.1f} N')
ax.set_xlabel('$dp$ [m]'); ax.set_ylabel('$F_{SPH}$ [N]')
ax.set_title(f'(b) SPH Force — p={r["p_apparent"]:.2f}')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)
ax.legend(fontsize=7)

# 8c: Contact force with uncertainty band
ax = fig.add_subplot(gs[0, 2])
ax.plot(dp_all, f_cont_N, 'D-', color=C['red'], zorder=5, label='Simulated')
r = results['Contact Force']
if 'mean' in r:
    ax.axhspan(r['min'], r['max'], color=C['red'], alpha=0.08)
    ax.axhline(r['mean'], color=C['red'], ls=':', lw=0.8,
               label=f'Mean ± U = {r["mean"]:.0f} ± {r["U"]:.0f} N')
ax.set_xlabel('$dp$ [m]'); ax.set_ylabel('$F_{contact}$ [N]')
ax.set_title(f'(c) Contact Force — {r["conv_type"].replace("_"," ")}')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)
ax.legend(fontsize=7)

# 8d: Velocity (converged)
ax = fig.add_subplot(gs[1, 0])
ax.plot(dp_all, vel_ms, '^-', color=C['orange'], zorder=5, label='Simulated')
r = results['Velocity']
p_str = f'p={r["p_apparent"]:.2f}' if r['conv_type'] == 'monotonic_convergence' else r['conv_type']
ax.set_xlabel('$dp$ [m]'); ax.set_ylabel('Velocity [m/s]')
ax.set_title(f'(d) Boulder Velocity — {p_str}')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)
ax.set_ylim(1.0, 1.35)

# 8e: Rotation
ax = fig.add_subplot(gs[1, 1])
ax.plot(dp_all, rot_deg, 'D-', color=C['purple'], zorder=5, label='Simulated')
r = results['Rotation']
if 'mean' in r:
    ax.axhspan(r['min'], r['max'], color=C['purple'], alpha=0.08)
ax.set_xlabel('$dp$ [m]'); ax.set_ylabel('Rotation [deg]')
ax.set_title(f'(e) Rotation — {r["conv_type"].replace("_"," ")}')
ax.invert_xaxis(); ax.set_xlim(0.022, 0.003)
ax.set_ylim(40, 120)

# 8f: Verdict summary
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')

verdict_text = "CONVERGENCE VERDICT\n"
verdict_text += "═" * 35 + "\n\n"

for name in METRICS:
    r = results[name]
    if r['conv_type'] == 'monotonic_convergence':
        icon = '●'  # converging
        color_str = 'green'
        detail = f'p={r["p_apparent"]:.2f}, GCI={r["GCI_fine"]*100:.1f}%'
    elif r['conv_type'] == 'oscillatory':
        icon = '○'  # oscillatory
        color_str = 'red'
        detail = f'U={r.get("U", 0):.1f} ({r.get("U_pct", 0):.0f}%)'
    else:
        icon = '△'
        color_str = 'orange'
        detail = r['conv_type']
    verdict_text += f"{icon}  {name:18s} {detail}\n"

verdict_text += "\n" + "─" * 35 + "\n"
verdict_text += "● Monotonic convergence (GCI valid)\n"
verdict_text += "○ Oscillatory (uncertainty band)\n"
verdict_text += "△ Divergence\n"
verdict_text += "\n" + "─" * 35 + "\n"
verdict_text += f"Best triplet: dp = {TRIPLET}\n"
verdict_text += f"Total compute: {time_min.sum():.0f} min\n"
verdict_text += f"Hardware: NVIDIA RTX 5090"

ax.text(0.05, 0.95, verdict_text, transform=ax.transAxes,
        fontsize=7.5, va='top', ha='left', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', fc='#F5F5F5', ec='black', lw=1))

fig.suptitle('Mesh Convergence Study — Complete Analysis (5 resolutions, RTX 5090)',
             fontsize=13, fontweight='bold', y=1.01)
savefig(fig, 'fig08_complete_summary')
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT + JSON EXPORT
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("  FORMAL CONVERGENCE VERIFICATION — Celik et al. (2008)")
print("=" * 75)
print(f"  Boulder: BLIR3 (d_eq = {d_eq*100:.1f} cm, mass = 1.061 kg)")
print(f"  Domain: dam_h = 0.3 m, material = lime-stone")
print(f"  Best triplet: dp = (0.005, 0.008, 0.015), r21=1.60, r32=1.88")
print(f"  Hardware: RTX 5090, total compute = {time_min.sum():.0f} min")
print("=" * 75)

for name in METRICS:
    r = results[name]
    info = METRICS[name]
    print(f"\n  {name} ({info['symbol']}):")
    print(f"    Type:           {r['conv_type']}")

    if r['conv_type'] == 'monotonic_convergence':
        print(f"    Apparent order: p = {r['p_apparent']:.3f}")
        print(f"    Richardson ext: phi_ext = {r['phi_ext']:.4f} {info['unit']}")
        print(f"    GCI_fine:       {r['GCI_fine']*100:.2f}%")
        print(f"    GCI_medium:     {r['GCI_med']*100:.2f}%")
        print(f"    Asymptotic R:   {r['AR']:.4f} {'(OK)' if r.get('in_asymptotic') else '(NOT in range)'}")
    else:
        print(f"    Mean:           {r.get('mean', 0):.3f} {info['unit']}")
        print(f"    Uncertainty:    ±{r.get('U', 0):.3f} {info['unit']} ({r.get('U_pct', 0):.1f}%)")
        print(f"    Range:          [{r.get('min', 0):.3f}, {r.get('max', 0):.3f}]")

print("\n" + "=" * 75)
print("  VERDICT")
print("=" * 75)
converged = [n for n in METRICS if results[n]['conv_type'] == 'monotonic_convergence']
not_conv = [n for n in METRICS if results[n]['conv_type'] != 'monotonic_convergence']
print(f"  Converging ({len(converged)}): {', '.join(converged)}")
print(f"  Not converged ({len(not_conv)}): {', '.join(not_conv)}")
print(f"\n  Recommendation: dp <= 0.004 m (>= 10 particles in d_min)")
print(f"  Use displacement as primary criterion (not contact force)")
print(f"  Water height anomaly at dp=0.005 requires investigation")
print("=" * 75)

# Export JSON for RETOMAR.md
export = {
    'timestamp': datetime.now().isoformat(),
    'study': 'convergence_formal_gci',
    'n_resolutions': len(dp_all),
    'dp_values': dp_all.tolist(),
    'best_triplet': [0.005, 0.008, 0.015],
    'metrics': {},
}
for name in METRICS:
    r = results[name]
    export['metrics'][name] = {
        'convergence_type': r['conv_type'],
        'p_apparent': float(r.get('p_apparent', 0)),
        'phi_extrapolated': float(r.get('phi_ext', 0)) if 'phi_ext' in r else None,
        'GCI_fine_pct': float(r.get('GCI_fine', 0) * 100) if 'GCI_fine' in r else None,
        'uncertainty_pct': float(r.get('U_pct', 0)) if 'U_pct' in r else None,
    }

json_path = OUT / 'convergence_gci_results.json'
with open(json_path, 'w') as f:
    json.dump(export, f, indent=2)
print(f"\n  JSON exported: {json_path}")
print(f"  Figures: {OUT}")
