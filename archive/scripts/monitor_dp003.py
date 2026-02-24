"""
monitor_dp003.py — Bot que monitorea GitHub cada 10 min esperando dp=0.003

Cuando detecta que dp=0.003 terminó:
1. Descarga los datos (status JSON + log)
2. Actualiza el reporte de convergencia con los 7 dp
3. Genera TODAS las figuras de convergencia (7 dp)
4. Emite un veredicto de convergencia
5. Guarda todo en data/figuras_7dp/

Uso:
    python monitor_dp003.py              # Loop infinito cada 10 min
    python monitor_dp003.py --once       # Una sola verificacion
    python monitor_dp003.py --force      # Saltar monitoreo, correr analisis con data manual

Autor: Kevin Cortes (UCN 2026)
"""

import json
import time
import sys
import urllib.request
import base64
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
REPO = "kcortes765/SPH-Convergencia"
GITHUB_API = f"https://api.github.com/repos/{REPO}"
CHECK_INTERVAL_S = 600  # 10 minutos
LAST_KNOWN_SHA = "0997e85"  # ultimo commit conocido (09:01 UTC Feb 21)

# Status file local
STATUS_FILE = PROJECT_ROOT / "data" / "monitor_dp003_log.txt"


def log_status(msg: str):
    """Log a consola y archivo."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    logger.info(line)
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def github_get(path: str) -> dict:
    """GET request a GitHub API."""
    url = f"{GITHUB_API}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "SPH-Monitor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.warning(f"GitHub API error: {e}")
        return {}


def github_get_file(path: str) -> str:
    """Descarga contenido de un archivo desde GitHub."""
    data = github_get(f"contents/{path}")
    if "content" in data:
        return base64.b64decode(data["content"]).decode()
    return ""


def check_new_commits() -> bool:
    """Verifica si hay commits nuevos despues del ultimo conocido."""
    data = github_get("commits?per_page=3")
    if not data:
        return False
    latest_sha = data[0]["sha"][:7]
    if latest_sha != LAST_KNOWN_SHA:
        msg = data[0]["commit"]["message"].strip().split("\n")[0]
        log_status(f"NUEVO COMMIT: {latest_sha} — {msg}")
        return True
    return False


def get_dp003_status() -> dict:
    """Descarga y parsea el status JSON de convergencia v2."""
    content = github_get_file("data/convergencia_v2_status.json")
    if content:
        return json.loads(content)
    return {}


def is_dp003_complete(status: dict) -> bool:
    """Verifica si dp=0.003 ya termino."""
    estado = status.get("estado", "")
    progreso = status.get("progreso", "")
    # Posibles indicadores de completitud
    if "COMPLETADO" in estado.upper():
        return True
    if "TERMINADO dp=0.003" in estado:
        return True
    if progreso == "2/2":
        return True
    # Verificar si hay resultado de dp=0.003 en el ultimo resultado
    ultimo = status.get("ultimo_resultado", {})
    if ultimo and ultimo.get("dp") == 0.003 and ultimo.get("status") == "OK":
        return True
    return False


def extract_dp003_data(status: dict) -> dict:
    """Extrae los datos de dp=0.003 del status JSON."""
    ultimo = status.get("ultimo_resultado", {})
    if ultimo and ultimo.get("dp") == 0.003:
        return ultimo
    return {}


def check_reporte_csv() -> dict:
    """Intenta obtener el reporte CSV v2 de GitHub."""
    content = github_get_file("data/reporte_convergencia_v2.csv")
    if content:
        return {"csv": content}
    # Tambien verificar si el reporte original fue actualizado
    content = github_get_file("data/reporte_convergencia.csv")
    if content:
        return {"csv": content}
    return {}


def run_analysis_7dp(dp003_data: dict = None):
    """
    Ejecuta el analisis completo de convergencia con 7 dp.
    Si dp003_data es None, intenta leerlo del status.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpec

    log_status("=" * 60)
    log_status("ANALISIS DE CONVERGENCIA COMPLETO — 7 dp")
    log_status("=" * 60)

    # ===================================================================
    # DATA: 7 dp (6 confirmados + dp=0.003 nuevo)
    # ===================================================================

    # Datos confirmados (6 dp)
    dp_confirmed = [0.020, 0.015, 0.010, 0.008, 0.005, 0.004]
    disp_confirmed = [3.4948, 3.4335, 3.0687, 2.4075, 1.7248, 1.6147]
    rot_confirmed = [95.82, 97.23, 60.34, 87.15, 86.79, 84.81]
    vel_confirmed = [1.1612, 1.2688, 1.1187, 1.1381, 1.1577, 1.1675]
    fsph_confirmed = [166.42, 77.00, 45.27, 34.88, 23.01, 22.80]
    fcont_confirmed = [2254.1, 4914.9, 130.67, 3229.1, 3083.0, 358.8]
    flow_confirmed = [0.4890, 0.4736, 0.5200, 0.5278, 0.4226, 0.4269]
    water_confirmed = [0.1587, 0.1564, 0.1630, 0.1654, 0.3201, 0.2450]
    time_confirmed = [13.2, 11.7, 23.7, 30.3, 117.8, 260.1]
    dim_dp_confirmed = [2.0, 2.7, 4.0, 5.0, 8.0, 10.0]

    # Agregar dp=0.003 si tenemos datos
    if dp003_data and dp003_data.get("status") == "OK":
        dp_confirmed.append(0.003)
        disp_confirmed.append(dp003_data["max_displacement_m"])
        rot_confirmed.append(dp003_data["max_rotation_deg"])
        vel_confirmed.append(dp003_data["max_velocity_ms"])
        fsph_confirmed.append(dp003_data["max_sph_force_N"])
        fcont_confirmed.append(dp003_data["max_contact_force_N"])
        flow_confirmed.append(dp003_data["max_flow_velocity_ms"])
        water_confirmed.append(dp003_data["max_water_height_m"])
        time_confirmed.append(dp003_data["tiempo_computo_min"])
        dim_dp_confirmed.append(0.0399 / 0.003)  # dim_min / dp
        n_dp = 7
        log_status(f"dp=0.003 DATA: disp={dp003_data['max_displacement_m']:.4f}m, "
                   f"rot={dp003_data['max_rotation_deg']:.2f}deg, "
                   f"vel={dp003_data['max_velocity_ms']:.4f}m/s, "
                   f"fsph={dp003_data['max_sph_force_N']:.2f}N, "
                   f"fcont={dp003_data['max_contact_force_N']:.2f}N, "
                   f"time={dp003_data['tiempo_computo_min']:.1f}min")
    else:
        n_dp = 6
        log_status("ADVERTENCIA: dp=0.003 no disponible, usando 6 dp")

    # Convertir a arrays
    dp = np.array(dp_confirmed)
    disp_m = np.array(disp_confirmed)
    rot_deg = np.array(rot_confirmed)
    vel_ms = np.array(vel_confirmed)
    f_sph = np.array(fsph_confirmed)
    f_cont = np.array(fcont_confirmed)
    flow_ms = np.array(flow_confirmed)
    water_m = np.array(water_confirmed)
    time_min = np.array(time_confirmed)
    dim_dp = np.array(dim_dp_confirmed)

    d_eq = 0.100421
    N_base = 209103
    particles = N_base * (0.02 / dp) ** 3

    # Deltas sucesivos
    delta_disp = np.abs(np.diff(disp_m)) / disp_m[:-1] * 100
    delta_fsph = np.abs(np.diff(f_sph)) / f_sph[:-1] * 100
    delta_vel = np.abs(np.diff(vel_ms)) / vel_ms[:-1] * 100
    delta_rot = np.abs(np.diff(rot_deg)) / rot_deg[:-1] * 100

    # ===================================================================
    # STYLE
    # ===================================================================

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

    B = '#2166AC'
    R = '#B2182B'
    GN = '#2CA02C'
    O = '#FF7F00'
    P = '#6A3D9A'
    GR = '#555555'

    OUT = PROJECT_ROOT / "data" / f"figuras_{n_dp}dp"
    OUT.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(OUT / f"{name}.png")
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
        log_status(f"  Figura: {name}")

    # ===================================================================
    # VEREDICTO DE CONVERGENCIA
    # ===================================================================

    log_status("\n--- VEREDICTO DE CONVERGENCIA ---")

    # Displacement
    last_delta_disp = delta_disp[-1]
    disp_converged = last_delta_disp < 5.0
    log_status(f"Displacement: delta={last_delta_disp:.1f}% {'CONVERGED (<5%)' if disp_converged else 'NOT YET (<5% needed)'}")

    # SPH Force
    last_delta_fsph = delta_fsph[-1]
    fsph_converged = last_delta_fsph < 5.0
    log_status(f"SPH Force: delta={last_delta_fsph:.1f}% {'CONVERGED' if fsph_converged else 'NOT YET'}")

    # Velocity
    last_delta_vel = delta_vel[-1]
    vel_converged = last_delta_vel < 5.0
    log_status(f"Velocity: delta={last_delta_vel:.1f}% {'CONVERGED' if vel_converged else 'NOT YET'}")

    # Rotation
    last_delta_rot = delta_rot[-1]
    rot_converged = last_delta_rot < 10.0
    log_status(f"Rotation: delta={last_delta_rot:.1f}% {'STABILIZED (<10%)' if rot_converged else 'UNSTABLE'}")

    # Contact force (always erratic)
    fcont_cv = np.std(f_cont) / np.mean(f_cont) * 100
    log_status(f"Contact Force: CV={fcont_cv:.0f}% — ERRATIC (known issue)")

    # Overall verdict
    primary_converged = disp_converged and fsph_converged and vel_converged
    if primary_converged:
        verdict = "CONVERGENCIA ALCANZADA"
        verdict_detail = (f"Las 3 metricas primarias (desplazamiento, fuerza SPH, velocidad) "
                         f"muestran delta < 5% entre los dos dp mas finos. "
                         f"dp={dp[-1]} es ADECUADO para produccion.")
    elif disp_converged or fsph_converged:
        verdict = "CONVERGENCIA PARCIAL"
        converged_list = []
        if disp_converged: converged_list.append("desplazamiento")
        if fsph_converged: converged_list.append("F_SPH")
        if vel_converged: converged_list.append("velocidad")
        verdict_detail = (f"Convergidas: {', '.join(converged_list)}. "
                         f"dp={dp[-1]} aceptable con caveats.")
    else:
        verdict = "CONVERGENCIA NO ALCANZADA"
        verdict_detail = f"Metricas siguen cambiando >5% entre dp={dp[-2]} y dp={dp[-1]}."

    log_status(f"\n{'='*50}")
    log_status(f"VEREDICTO: {verdict}")
    log_status(verdict_detail)
    log_status(f"{'='*50}")

    # ===================================================================
    # FIGURAS
    # ===================================================================

    log_status(f"\nGenerando {10 if n_dp == 7 else 9} figuras en {OUT}...")

    # --- FIG 1: Displacement convergence ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dp, disp_m, 'o-', color=B, zorder=5, clip_on=False,
            markersize=7, markeredgecolor='white', markeredgewidth=1)

    for i in range(len(dp) - 1):
        mid_x = (dp[i] + dp[i+1]) / 2
        mid_y = (disp_m[i] + disp_m[i+1]) / 2
        ax.annotate(f'$\\Delta$ = {delta_disp[i]:.1f}%',
                    xy=(mid_x, mid_y + 0.12),
                    fontsize=8, ha='center', va='bottom', color=R,
                    fontweight='bold')

    # Convergence zone
    if n_dp == 7:
        ax.axvspan(0.002, 0.0045, color=GN, alpha=0.06, zorder=0)
        ax.text(0.0035, max(disp_m) * 0.95, 'Convergence\nzone', fontsize=8,
                ha='center', color=GN, fontstyle='italic', alpha=0.8)

    ax.set_xlabel('Particle spacing $dp$ [m]')
    ax.set_ylabel('Maximum boulder displacement [m]')
    ax.set_title(f'Boulder Displacement vs. Particle Resolution ({n_dp} levels)')
    ax.invert_xaxis()
    ax.set_xlim(0.022, max(0.002, dp[-1] - 0.001))
    fig.tight_layout()
    save(fig, 'fig01_displacement_convergence')

    # --- FIG 2: SPH Force convergence ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dp, f_sph, 's-', color=R, zorder=5, clip_on=False,
            markersize=7, markeredgecolor='white', markeredgewidth=1)

    for i in range(len(dp) - 1):
        mid_x = (dp[i] + dp[i+1]) / 2
        mid_y = (f_sph[i] + f_sph[i+1]) / 2
        offset_y = max(3, (f_sph[i] - f_sph[i+1]) * 0.15)
        ax.annotate(f'$\\Delta$ = {delta_fsph[i]:.0f}%',
                    xy=(mid_x, mid_y + offset_y),
                    fontsize=8, ha='center', va='bottom', color=GR,
                    fontweight='bold')

    # Converged band
    fsph_fine = f_sph[-2:]
    fsph_band_center = np.mean(fsph_fine)
    fsph_band_half = max(1.0, np.ptp(fsph_fine) * 1.5)
    ax.axhspan(fsph_band_center - fsph_band_half, fsph_band_center + fsph_band_half,
               color=GN, alpha=0.1, zorder=0)
    ax.text(0.012, fsph_band_center - fsph_band_half - 2,
            f'$F_{{SPH}}$ converged $\\approx$ {fsph_band_center:.1f} N',
            fontsize=8, color=GN, fontstyle='italic')

    ax.set_xlabel('Particle spacing $dp$ [m]')
    ax.set_ylabel('Maximum SPH force [N]')
    ax.set_title('Hydrodynamic Force Convergence')
    ax.invert_xaxis()
    ax.set_xlim(0.022, max(0.002, dp[-1] - 0.001))
    fig.tight_layout()
    save(fig, 'fig02_sph_force_convergence')

    # --- FIG 3: Convergence rate (all metrics) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dp_mid = (dp[:-1] + dp[1:]) / 2

    ax.plot(dp_mid, delta_disp, 'o-', color=B, label='Displacement', zorder=5)
    ax.plot(dp_mid, delta_fsph, 's-', color=R, label='SPH Force', zorder=5)
    ax.plot(dp_mid, delta_vel, '^-', color=O, label='Velocity', zorder=5)
    ax.plot(dp_mid, delta_rot, 'D-', color=P, label='Rotation', zorder=4)

    ax.axhline(5, color='black', ls='--', lw=0.8, alpha=0.4)
    ax.text(dp_mid[-1] + 0.001, 6.0, '5% criterion', fontsize=7.5, color=GR)

    # Last point labels
    for data, color, name in [(delta_disp, B, 'Disp'),
                               (delta_fsph, R, 'F_SPH'),
                               (delta_vel, O, 'Vel'),
                               (delta_rot, P, 'Rot')]:
        ax.annotate(f'{data[-1]:.1f}%', xy=(dp_mid[-1], data[-1]),
                    xytext=(8, 2), textcoords='offset points',
                    fontsize=7.5, color=color, fontweight='bold')

    ax.set_xlabel('Particle spacing $dp$ [m]')
    ax.set_ylabel('Successive relative change [%]')
    ax.set_title(f'Convergence Rate — All Metrics ({n_dp} levels)')
    ax.invert_xaxis()
    ax.set_xlim(0.019, max(0.003, dp_mid[-1] - 0.001))
    ax.set_ylim(0, max(60, max(delta_disp.max(), delta_rot.max()) * 1.2))
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()
    save(fig, 'fig03_convergence_rate')

    # --- FIG 4: Contact force diagnosis ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.semilogy(dp, f_cont, 'o-', color=R, label='Contact (Chrono)', zorder=5)
    ax1.semilogy(dp, f_sph, 's-', color=B, label='SPH (fluid)', zorder=5)
    for i, (d, fc) in enumerate(zip(dp, f_cont)):
        yoff = 1.4 if i % 2 == 0 else 0.7
        ax1.annotate(f'{fc:.0f}', xy=(d, fc * yoff), fontsize=7, ha='center', color=R)

    ax1.set_xlabel('$dp$ [m]')
    ax1.set_ylabel('Maximum force [N]')
    ax1.set_title('(a) Force Comparison (log scale)')
    ax1.invert_xaxis()
    ax1.set_xlim(0.022, max(0.002, dp[-1] - 0.001))
    ax1.legend(fontsize=8, loc='lower left')
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())

    # Resolution color bars
    colors = []
    for d in dim_dp:
        if d < 4: colors.append('#EF5350')
        elif d < 8: colors.append('#FFA726')
        elif d < 10: colors.append('#FFEE58')
        else: colors.append('#66BB6A')

    ax2.bar(range(len(dp)), f_cont, color=colors, edgecolor='black', lw=0.5, zorder=3)
    ax2.set_xticks(range(len(dp)))
    ax2.set_xticklabels([f'dp={d}\n({int(n)}p)' for d, n in zip(dp, dim_dp)], fontsize=7)
    ax2.set_ylabel('Max contact force [N]')
    ax2.set_title('(b) Contact Force by Resolution')

    for i, val in enumerate(f_cont):
        ax2.text(i, val + 100, f'{val:.0f}', ha='center', fontsize=7, fontweight='bold')

    legend_elems = [
        Patch(fc='#EF5350', ec='black', lw=0.5, label='< 4p (under-resolved)'),
        Patch(fc='#FFA726', ec='black', lw=0.5, label='4-7p (marginal)'),
        Patch(fc='#FFEE58', ec='black', lw=0.5, label='8-9p (transitional)'),
        Patch(fc='#66BB6A', ec='black', lw=0.5, label='$\\geq$10p (adequate)'),
    ]
    ax2.legend(handles=legend_elems, fontsize=6.5, loc='upper right')

    fig.suptitle('Contact Force — Resolution Sensitivity',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig04_contact_force_diagnosis')

    # --- FIG 5: Velocity + Rotation ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(dp, vel_ms, '^-', color=O, zorder=5,
             markersize=7, markeredgecolor='white', markeredgewidth=1)
    mean_vel_fine = np.mean(vel_ms[-3:])
    ax1.axhspan(mean_vel_fine - 0.02, mean_vel_fine + 0.02, color=O, alpha=0.08, zorder=0)
    ax1.axhline(mean_vel_fine, color=GR, ls=':', lw=0.8)

    for i, (d, v) in enumerate(zip(dp, vel_ms)):
        yoff = 0.015 if v > mean_vel_fine else -0.025
        ax1.annotate(f'{v:.3f}', xy=(d, v + yoff), fontsize=7, ha='center', color=O)

    ax1.set_xlabel('$dp$ [m]')
    ax1.set_ylabel('Max boulder velocity [m/s]')
    ax1.set_title('(a) Boulder Velocity')
    ax1.invert_xaxis()
    ax1.set_xlim(0.022, max(0.002, dp[-1] - 0.001))

    ax2.plot(dp, rot_deg, 'D-', color=P, zorder=5,
             markersize=7, markeredgecolor='white', markeredgewidth=1)

    for i, (d, r) in enumerate(zip(dp, rot_deg)):
        yoff = 3 if r > 80 else -8
        if abs(r - 60.34) < 1: yoff = -8
        ax2.annotate(f'{r:.1f}$\\degree$', xy=(d, r + yoff),
                     fontsize=7, ha='center', color=P)

    # Stable band
    rot_fine = rot_deg[-3:]
    rot_mean = np.mean(rot_fine)
    ax2.axhspan(rot_mean - 3, rot_mean + 3, color=P, alpha=0.06, zorder=0)

    ax2.set_xlabel('$dp$ [m]')
    ax2.set_ylabel('Max rotation [deg]')
    ax2.set_title('(b) Boulder Rotation')
    ax2.invert_xaxis()
    ax2.set_xlim(0.022, max(0.002, dp[-1] - 0.001))

    fig.suptitle('Kinematic Metrics — Velocity and Rotation',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig05_velocity_rotation')

    # --- FIG 6: Computational cost ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(dp, time_min, 'o-', color=B, zorder=5)
    for i, (d, t) in enumerate(zip(dp, time_min)):
        hrs = f' ({t/60:.1f}h)' if t > 60 else ''
        ax1.annotate(f'{t:.0f} min{hrs}', xy=(d, t),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=7.5, ha='center', color=GR)

    ax1.set_xlabel('$dp$ [m]')
    ax1.set_ylabel('Wall-clock time [min]')
    ax1.set_title('(a) Computational Cost')
    ax1.invert_xaxis()
    ax1.set_xlim(0.022, max(0.002, dp[-1] - 0.001))

    ax2.loglog(particles, time_min, 'o-', color=R, zorder=5)
    log_n, log_t = np.log(particles), np.log(time_min)
    b_exp, log_a = np.polyfit(log_n, log_t, 1)
    a_coef = np.exp(log_a)
    n_fit = np.logspace(np.log10(particles.min()), np.log10(particles.max() * 1.5), 50)
    t_fit = a_coef * n_fit ** b_exp
    ax2.plot(n_fit, t_fit, '--', color=GR, alpha=0.6, lw=1.0,
             label=f'$t \\propto N^{{{b_exp:.2f}}}$')

    for i, (n, t, d) in enumerate(zip(particles, time_min, dp)):
        lab = f'{n/1e6:.1f}M' if n > 1e6 else f'{n/1e3:.0f}K'
        ax2.annotate(f'dp={d}', xy=(n, t),
                     xytext=(8, -5), textcoords='offset points',
                     fontsize=6.5, color=GR)

    ax2.set_xlabel('Estimated particle count')
    ax2.set_ylabel('Wall-clock time [min]')
    ax2.set_title('(b) Scaling (RTX 5090)')
    ax2.legend(fontsize=8)

    fig.suptitle('Computational Cost — GPU Simulation on NVIDIA RTX 5090',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig06_computational_cost')

    # --- FIG 7: Water height anomaly ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dp, water_m, 'o-', color=B, zorder=5)

    mean_wh_stable = np.mean(water_m[:4])
    ax.axhspan(mean_wh_stable - 0.01, mean_wh_stable + 0.01, color=B, alpha=0.08, zorder=0)
    ax.axhline(mean_wh_stable, color=GR, ls=':', lw=0.8)

    for i, (d, w) in enumerate(zip(dp, water_m)):
        yoff = 0.015 if w > 0.2 else -0.02
        ax.annotate(f'{w:.3f}', xy=(d, w + yoff), fontsize=7.5, ha='center', color=B)

    ax.set_xlabel('Particle spacing $dp$ [m]')
    ax.set_ylabel('Maximum water height [m]')
    ax.set_title('Water Height at Gauge — Anomaly Investigation')
    ax.invert_xaxis()
    ax.set_xlim(0.022, max(0.002, dp[-1] - 0.001))
    fig.tight_layout()
    save(fig, 'fig07_water_height')

    # --- FIG 8: Summary table ---
    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.5 * n_dp)))
    ax.axis('off')

    cols = ['$dp$ [m]', '$N_{part}$', 'Part.\nin $d_{min}$',
            'Displ.\n[m]', '$\\Delta$%',
            'Rot.\n[deg]', '$F_{SPH}$\n[N]', '$F_{cont}$\n[N]',
            'Time\n[min]', 'Status']

    rows = []
    for i in range(len(dp)):
        n_str = f'{particles[i]/1e6:.1f}M' if particles[i] > 1e6 else f'{particles[i]/1e3:.0f}K'
        d_str = f'{delta_disp[i-1]:.1f}%' if i > 0 else '\u2014'
        if dim_dp[i] >= 10: status = 'ADEQUATE'
        elif dim_dp[i] >= 8: status = 'TRANSITIONAL'
        elif dim_dp[i] >= 4: status = 'MARGINAL'
        else: status = 'UNDER-RES.'
        rows.append([
            f'{dp[i]:.3f}', n_str, f'{dim_dp[i]:.1f}',
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
        'UNDER-RES.': '#FFCDD2', 'MARGINAL': '#FFE0B2',
        'TRANSITIONAL': '#FFF9C4', 'ADEQUATE': '#C8E6C9',
    }
    for i in range(len(dp)):
        for j in range(len(cols)):
            table[i+1, j].set_facecolor(status_colors[rows[i][-1]])

    fig.suptitle(f'Mesh Convergence Results — {n_dp} Resolutions, Boulder BLIR3, RTX 5090',
                 fontsize=11, fontweight='bold', y=0.95)
    save(fig, 'fig08_summary_table')

    # --- FIG 9: The "money figure" ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))

    # 9a: Displacement
    ax = axes[0, 0]
    ax.plot(dp, disp_m, 'o-', color=B, zorder=5)
    if n_dp == 7:
        ax.axvspan(0.002, 0.0045, color=GN, alpha=0.06, zorder=0)
    for i in range(len(dp)):
        ax.annotate(f'{disp_m[i]:.2f}', xy=(dp[i], disp_m[i]),
                    xytext=(0, 8), textcoords='offset points',
                    fontsize=6.5, ha='center', color=B)
    ax.set_title('(a) Displacement [m]', fontsize=10)
    ax.set_xlabel('$dp$ [m]')
    ax.invert_xaxis(); ax.set_xlim(0.022, max(0.002, dp[-1] - 0.001))

    # 9b: SPH Force
    ax = axes[0, 1]
    ax.plot(dp, f_sph, 's-', color=R, zorder=5)
    ax.axhspan(fsph_band_center - fsph_band_half, fsph_band_center + fsph_band_half,
               color=GN, alpha=0.1, zorder=0)
    for i in range(len(dp)):
        ax.annotate(f'{f_sph[i]:.1f}', xy=(dp[i], f_sph[i]),
                    xytext=(0, 8), textcoords='offset points',
                    fontsize=6.5, ha='center', color=R)
    ax.set_title('(b) SPH Force [N]', fontsize=10)
    ax.set_xlabel('$dp$ [m]')
    ax.invert_xaxis(); ax.set_xlim(0.022, max(0.002, dp[-1] - 0.001))

    # 9c: Contact Force
    ax = axes[0, 2]
    ax.plot(dp, f_cont, 'D-', color=R, zorder=5)
    for i in range(len(dp)):
        yoff = 200 if f_cont[i] < 3000 else -400
        ax.annotate(f'{f_cont[i]:.0f}', xy=(dp[i], f_cont[i] + yoff),
                    fontsize=6.5, ha='center', color=R)
    ax.set_title('(c) Contact Force [N] \u2014 ERRATIC', fontsize=10, color=R)
    ax.set_xlabel('$dp$ [m]')
    ax.invert_xaxis(); ax.set_xlim(0.022, max(0.002, dp[-1] - 0.001))

    # 9d: Velocity
    ax = axes[1, 0]
    ax.plot(dp, vel_ms, '^-', color=O, zorder=5)
    ax.axhspan(mean_vel_fine - 0.02, mean_vel_fine + 0.02, color=O, alpha=0.08, zorder=0)
    for i in range(len(dp)):
        ax.annotate(f'{vel_ms[i]:.3f}', xy=(dp[i], vel_ms[i]),
                    xytext=(0, 8), textcoords='offset points',
                    fontsize=6.5, ha='center', color=O)
    ax.set_title('(d) Velocity [m/s]', fontsize=10, color=GN if vel_converged else O)
    ax.set_xlabel('$dp$ [m]')
    ax.invert_xaxis(); ax.set_xlim(0.022, max(0.002, dp[-1] - 0.001))

    # 9e: Rotation
    ax = axes[1, 1]
    ax.plot(dp, rot_deg, 'D-', color=P, zorder=5)
    ax.axhspan(rot_mean - 3, rot_mean + 3, color=P, alpha=0.06, zorder=0)
    for i in range(len(dp)):
        yoff = 4 if rot_deg[i] > 80 else -8
        if abs(rot_deg[i] - 60.34) < 1: yoff = -8
        ax.annotate(f'{rot_deg[i]:.1f}', xy=(dp[i], rot_deg[i] + yoff),
                    fontsize=6.5, ha='center', color=P)
    ax.set_title('(e) Rotation [deg]', fontsize=10)
    ax.set_xlabel('$dp$ [m]')
    ax.invert_xaxis(); ax.set_xlim(0.022, max(0.002, dp[-1] - 0.001))

    # 9f: Convergence rate
    ax = axes[1, 2]
    ax.plot(dp_mid, delta_disp, 'o-', color=B, label='Displacement', zorder=5)
    ax.plot(dp_mid, delta_fsph, 's-', color=R, label='SPH Force', zorder=4)
    ax.plot(dp_mid, delta_vel, '^-', color=O, label='Velocity', zorder=4)
    ax.axhline(5, color='black', ls='--', lw=0.7, alpha=0.4)
    ax.text(dp_mid[-1] + 0.0005, 6.5, '5%', fontsize=7, color=GR)

    ax.annotate(f'{delta_disp[-1]:.1f}%', xy=(dp_mid[-1], delta_disp[-1]),
                xytext=(8, 2), textcoords='offset points',
                fontsize=7, color=B, fontweight='bold')
    ax.annotate(f'{delta_fsph[-1]:.1f}%', xy=(dp_mid[-1], delta_fsph[-1]),
                xytext=(8, 2), textcoords='offset points',
                fontsize=7, color=R, fontweight='bold')
    ax.annotate(f'{delta_vel[-1]:.1f}%', xy=(dp_mid[-1], delta_vel[-1]),
                xytext=(8, -6), textcoords='offset points',
                fontsize=7, color=O, fontweight='bold')

    ax.set_title('(f) Convergence Rate [%]', fontsize=10)
    ax.set_xlabel('$dp$ [m]')
    ax.invert_xaxis()
    ax.set_ylim(0, max(60, delta_disp.max() * 1.2))
    ax.legend(fontsize=6.5, loc='upper left')

    fig.suptitle(f'Mesh Convergence Study \u2014 {n_dp} Resolutions, RTX 5090, Boulder BLIR3',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig09_complete_story')

    # --- FIG 10: Verdict card ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    verdict_color = GN if primary_converged else (O if disp_converged else R)
    verdict_icon = '\u2705' if primary_converged else ('\u26A0' if disp_converged else '\u274C')

    text = f"MESH CONVERGENCE VERDICT\n"
    text += "=" * 45 + "\n\n"
    text += f"Study: {n_dp} resolutions (dp = {dp[0]:.3f} to {dp[-1]:.3f} m)\n"
    text += f"Boulder: BLIR3 (d_eq = {d_eq*100:.1f} cm)\n"
    text += f"Hardware: NVIDIA RTX 5090\n"
    text += f"Total compute: {time_min.sum():.0f} min ({time_min.sum()/60:.1f} h)\n\n"
    text += "-" * 45 + "\n"
    text += f"{'Metric':<20} {'Last delta':>12} {'Status':>12}\n"
    text += "-" * 45 + "\n"

    metrics_verdict = [
        ('Displacement', delta_disp[-1], disp_converged),
        ('SPH Force', delta_fsph[-1], fsph_converged),
        ('Velocity', delta_vel[-1], vel_converged),
        ('Rotation', delta_rot[-1], rot_converged),
        ('Contact Force', fcont_cv, False),
    ]

    for name, val, conv in metrics_verdict:
        status_str = 'CONVERGED' if conv else 'NOT CONV.'
        icon = '\u25CF' if conv else '\u25CB'
        text += f"{icon} {name:<18} {val:>10.1f}%  {status_str:>10}\n"

    text += "\n" + "=" * 45 + "\n"
    text += f"VERDICT: {verdict}\n"
    text += f"\n{verdict_detail}\n"
    text += "=" * 45

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='left', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='#FAFAFA', ec=verdict_color, lw=2))

    fig.suptitle('Convergence Verdict', fontsize=13, fontweight='bold')
    save(fig, 'fig10_verdict')

    # ===================================================================
    # CSV REPORTE ACTUALIZADO
    # ===================================================================
    import csv
    csv_path = PROJECT_ROOT / "data" / f"reporte_convergencia_{n_dp}dp.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["dp", "particles", "dim_min_dp", "displacement_m",
                         "delta_pct", "rotation_deg", "velocity_ms",
                         "sph_force_N", "contact_force_N", "flow_velocity_ms",
                         "water_height_m", "time_min"])
        for i in range(len(dp)):
            d_pct = f"{delta_disp[i-1]:.1f}" if i > 0 else ""
            writer.writerow([
                f"{dp[i]:.3f}", f"{particles[i]:.0f}", f"{dim_dp[i]:.1f}",
                f"{disp_m[i]:.4f}", d_pct, f"{rot_deg[i]:.2f}",
                f"{vel_ms[i]:.4f}", f"{f_sph[i]:.2f}", f"{f_cont[i]:.2f}",
                f"{flow_ms[i]:.4f}", f"{water_m[i]:.4f}", f"{time_min[i]:.1f}",
            ])
    log_status(f"CSV: {csv_path}")

    # ===================================================================
    # JSON EXPORT (para RETOMAR.md y memoria)
    # ===================================================================
    export = {
        "timestamp": datetime.now().isoformat(),
        "n_dp": n_dp,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "primary_converged": primary_converged,
        "metrics": {
            "displacement": {"last_delta_pct": float(last_delta_disp), "converged": bool(disp_converged)},
            "sph_force": {"last_delta_pct": float(last_delta_fsph), "converged": bool(fsph_converged)},
            "velocity": {"last_delta_pct": float(last_delta_vel), "converged": bool(vel_converged)},
            "rotation": {"last_delta_pct": float(last_delta_rot), "converged": bool(rot_converged)},
            "contact_force": {"cv_pct": float(fcont_cv), "converged": False},
        },
        "recommended_dp": float(dp[-1]),
        "total_compute_min": float(time_min.sum()),
    }

    json_path = OUT / "convergence_verdict.json"
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, default=lambda o: bool(o) if hasattr(o, '__bool__') and type(o).__name__ == 'bool_' else float(o))
    log_status(f"JSON: {json_path}")

    log_status(f"\n{'='*60}")
    log_status(f"COMPLETADO: {10 if n_dp == 7 else 9} figuras en {OUT}")
    log_status(f"VEREDICTO: {verdict}")
    log_status(f"{'='*60}")

    return export


def monitor_loop():
    """Loop principal: verifica GitHub cada 10 min."""
    log_status("=" * 60)
    log_status("MONITOR dp=0.003 INICIADO")
    log_status(f"Verificando {REPO} cada {CHECK_INTERVAL_S//60} min")
    log_status("=" * 60)

    check_count = 0

    while True:
        check_count += 1
        log_status(f"\n--- Check #{check_count} ---")

        # 1. Verificar commits nuevos
        has_new = check_new_commits()

        # 2. Obtener status actual
        status = get_dp003_status()
        estado = status.get("estado", "DESCONOCIDO")
        progreso = status.get("progreso", "?/?")
        elapsed = status.get("tiempo_transcurrido_min", 0)

        log_status(f"Estado: {estado} | Progreso: {progreso} | Tiempo: {elapsed:.0f} min")

        # 3. Verificar si termino
        if is_dp003_complete(status):
            log_status("\n!!! dp=0.003 COMPLETADO !!!")
            dp003_data = extract_dp003_data(status)

            if dp003_data:
                log_status("Datos encontrados en status JSON, ejecutando analisis...")
                result = run_analysis_7dp(dp003_data)
                log_status("Monitor terminado exitosamente.")
                return result
            else:
                log_status("Status dice completado pero no hay datos. Verificando CSV...")
                csv_data = check_reporte_csv()
                if csv_data:
                    log_status("CSV encontrado en repo. Analisis manual necesario.")
                else:
                    log_status("No se encontraron datos de dp=0.003. Continuando monitoreo...")
                    # Seguir monitoreando
        else:
            if has_new:
                log_status("Hay commits nuevos pero dp=0.003 no ha terminado aun.")
            else:
                log_status("Sin cambios. Esperando...")

        log_status(f"Proximo check en {CHECK_INTERVAL_S//60} min...")
        time.sleep(CHECK_INTERVAL_S)


def main():
    args = sys.argv[1:]

    if "--force" in args:
        # Correr analisis sin esperar dp=0.003
        log_status("Modo --force: corriendo analisis con datos disponibles")
        run_analysis_7dp(dp003_data=None)
        return

    if "--once" in args:
        # Una sola verificacion
        log_status("Modo --once: verificacion unica")
        status = get_dp003_status()
        estado = status.get("estado", "DESCONOCIDO")
        progreso = status.get("progreso", "?/?")
        elapsed = status.get("tiempo_transcurrido_min", 0)
        log_status(f"Estado: {estado} | Progreso: {progreso} | Tiempo: {elapsed:.0f} min")

        if is_dp003_complete(status):
            log_status("dp=0.003 COMPLETADO! Ejecutando analisis...")
            dp003_data = extract_dp003_data(status)
            run_analysis_7dp(dp003_data)
        else:
            log_status("dp=0.003 aun corriendo.")
        return

    # Loop infinito
    try:
        monitor_loop()
    except KeyboardInterrupt:
        log_status("\nMonitor detenido por usuario (Ctrl+C)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
